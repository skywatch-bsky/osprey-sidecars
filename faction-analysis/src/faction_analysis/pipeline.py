# pattern: Imperative Shell
"""One run of the sidecar, and what happens before and after it.

A poll first recovers: it completes a run whose promotion was intended but not confirmed, and it cleans up
runs that nobody will complete. It then asks the schedule, and runs. A run writes all of its output before
it records the intent to promote, so a run with recorded intent is fully written and is only ever completed
forward, never removed. Reads and writes go through two small protocols, so tests drive this with fakes.
"""

import logging
import secrets
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import pandas as pd

from faction_analysis import lock, model_store, promotion, schema, train
from faction_analysis.config import AppConfig, RunKind
from faction_analysis.conflict import (
    BLOCK,
    Events,
    HostileCounts,
    cell_counts,
    cells_frame,
    hostile_counts,
    swarm_flags,
)
from faction_analysis.features import participation_coefficient, within_faction_z
from faction_analysis.graph import (
    AffinityGraph,
    DeletedRecords,
    aggregate_window_edges,
    build_graph,
    drop_deleted_creates,
    index_vertices,
    superseded_mask,
)
from faction_analysis.labels import LabelInput
from faction_analysis.model import EXCLAMATIONS, LENGTH, MENTIONS, QUESTIONS, UPPERCASE, HostilityModel, text_frame
from faction_analysis.partition import (
    UNPLACED,
    NoFactionError,
    absorb_small_communities,
    build_igraph,
    run_partition,
    warm_start_membership,
    weighted_modularity,
)
from faction_analysis.queries import BLOCK_COLLECTION, FOLLOW_COLLECTION, Window
from faction_analysis.runstate import RunRow, RunState, new_run_id
from faction_analysis.schedule import ScheduleState, decide
from faction_analysis.telemetry import TelemetryHandles, noop_telemetry, record_failure, record_run_metrics
from faction_analysis.tracking import MOVE_KINDS, LineageEvent, MoveDiff, diff_moves, seed_confidence, track

logger = logging.getLogger(__name__)
RunStatus = Literal['promoted', 'failed', 'unresolved', 'stopped', 'none']
KIND_NAMES = ('reply', 'quote')
WEEKLY_RETENTION = timedelta(weeks=8)
FULL_RETENTION = timedelta(days=395)
EVENT_PARTS = ('event_ms', 'kind', 'author', 'target', 'from_faction', 'to_faction', 'thread_key', 'h')


class RunRepository(Protocol):
    """Everything a run reads from or writes to ClickHouse."""

    def insert_run_row(self, row: RunRow) -> None: ...

    def fetch_run_states(self) -> dict[str, RunState]: ...

    def fetch_run_rows(self, run_id: str) -> list[RunRow]: ...

    def latest_promoted_run(self, kind: str | None = None) -> RunRow | None: ...

    def fetch_image(self, run_id: str) -> pd.DataFrame: ...

    def fetch_faction_registry(self) -> pd.DataFrame: ...

    def stream_follow_creates(self, window: Window) -> Iterator[dict[str, np.ndarray]]: ...

    def stream_repost_creates(self, window: Window) -> Iterator[dict[str, np.ndarray]]: ...

    def fetch_deletes(self, window: Window, collection: str) -> DeletedRecords: ...

    def fetch_account_hashes(self, window: Window) -> pd.DataFrame: ...

    def stream_interactions(self, window: Window) -> Iterator[pd.DataFrame]: ...

    def fetch_daily_row_counts(self, window: Window) -> dict[date, int]: ...

    def fetch_tags(self, run_id: str, window: Window) -> pd.DataFrame: ...

    def fetch_languages(self, run_id: str, window: Window) -> pd.DataFrame: ...

    def fetch_handles(self, run_id: str, window: Window) -> pd.DataFrame: ...

    def insert_frame(self, noun: str, frame: pd.DataFrame) -> None: ...

    def fetch_text_scores(self, model_version: str) -> tuple[np.ndarray, np.ndarray]: ...

    def fetch_label_inputs(self, window: Window) -> LabelInput: ...

    def fetch_interaction_texts(self, window: Window, uri_hashes: Sequence[int]) -> pd.DataFrame: ...

    def fetch_parent_texts(self, window: Window, parent_uris: Sequence[str]) -> dict[str, str]: ...

    def fetch_jev_labels(self, prompt_version: str) -> pd.DataFrame: ...

    def drop_partition(self, noun: str, run_id: str) -> None: ...

    def delete_rows(self, noun: str, run_id: str) -> None: ...


@dataclass(frozen=True)
class Scorer:
    """The model on disk under its version."""

    model: HostilityModel
    version: str

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.probabilities(frame)


ModelLoader = Callable[[Path], Scorer | None]
Trainer = Callable[[RunRepository, AppConfig, Path, datetime], train.RefreshOutcome]


def load_scorer(model_dir: Path) -> Scorer | None:
    loaded = model_store.load_current(model_dir)
    return None if loaded is None else Scorer(loaded[0], loaded[1].version)


def default_trainer(api_key: str) -> Trainer:
    def run_refresh(repo: RunRepository, config: AppConfig, model_dir: Path, now: datetime) -> train.RefreshOutcome:
        factory = train.default_factory(api_key, config) if api_key else None
        return train.refresh(repo, config, model_dir, now=now, factory=factory)

    return run_refresh


@dataclass(frozen=True)
class RunOutcome:
    """What a poll did."""

    status: RunStatus
    reason: str = ''
    run_id: str | None = None
    kind: RunKind | None = None
    stage: str | None = None
    exit_code: int = 0


@dataclass
class RunContext:
    """What a run needs besides the repository."""

    config: AppConfig
    repo: RunRepository
    model_dir: Path
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    fault: Callable[[str], None] | None = None
    trainer: Trainer | None = None
    loader: ModelLoader = load_scorer
    telemetry: TelemetryHandles = field(default_factory=noop_telemetry)
    dry_run: bool = False

    def trip(self, point: str) -> None:
        """Call the fault hook at a named point. Tests crash or fail the run here."""
        if self.fault is not None:
            self.fault(point)


class RecordingRepository:
    """Keeps every write instead of sending it: a dry run computes everything and writes nothing."""

    def __init__(self, inner: RunRepository) -> None:
        self._inner = inner
        self.writes: list[tuple[str, object]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def insert_run_row(self, row: RunRow) -> None:
        self.writes.append(('runs', row))

    def insert_frame(self, noun: str, frame: pd.DataFrame) -> None:
        self.writes.append((noun, frame))

    def drop_partition(self, noun: str, run_id: str) -> None:
        self.writes.append((f'drop_partition:{noun}', run_id))

    def delete_rows(self, noun: str, run_id: str) -> None:
        self.writes.append((f'delete_rows:{noun}', run_id))


def recording(inner: RunRepository) -> RunRepository:
    """A repository for a dry run: reads pass through and writes are kept and never sent."""
    return cast(RunRepository, RecordingRepository(inner))


# ---- a poll


def poll(ctx: RunContext, *, kind: RunKind | None = None) -> RunOutcome:
    """Recover, decide, run. One call per poll of the loop."""
    if ctx.dry_run:
        return run_once(ctx, kind or 'weekly')
    started = time.monotonic()
    if _recover_at_startup(ctx):
        _record_stage(ctx, 'startup_recover', started)
        return RunOutcome('unresolved', 'a run waits for its promotion to be confirmed', exit_code=1)
    _record_stage(ctx, 'startup_recover', started)
    if kind is None:
        kind = _decision(ctx)
        if kind is None:
            return RunOutcome('none', 'no run is due')
    return run_once(ctx, kind)


def _decision(ctx: RunContext) -> RunKind | None:
    latest = ctx.repo.latest_promoted_run()
    latest_full = ctx.repo.latest_promoted_run('full')
    decision = decide(
        ScheduleState(
            now=ctx.now(),
            latest_promoted_at=latest.started_at if latest else None,
            latest_full_at=latest_full.started_at if latest_full else None,
        ),
        weekly_after_days=ctx.config.schedule.weekly_after_days,
        monthly_after_days=ctx.config.schedule.monthly_after_days,
    )
    return None if decision == 'none' else 'full' if decision == 'full' else 'weekly'


def _recover_at_startup(ctx: RunContext) -> bool:
    """Complete what an earlier process left unfinished. True when a new run must wait.

    The intent file comes first: it is under the sidecar's control and a ClickHouse insert is not. A run it
    names is completed whatever a re-read shows. An unreadable file raises, and the sidecar stops: the run
    it describes must not be cleaned up.
    """
    now = ctx.now()
    intent = lock.load(ctx.model_dir)
    if intent is not None:
        try:
            _resend(ctx, intent.run, ('promoting', 'promoted'), now)
        except Exception:  # noqa: BLE001 - nothing is removed; the next start completes the run
            logger.exception(
                'completing run %s failed. Nothing is removed and the next start retries it', intent.run.run_id
            )
            raise
        if _confirmed(ctx, intent.run.run_id):
            lock.clear(ctx.model_dir)
            return False
        return True
    try:
        states = ctx.repo.fetch_run_states()
        plan = promotion.startup_plan(None, states, _ms(now))
        started = time.monotonic()
        for run_id in plan.stale_run_ids:
            _abandon(ctx, run_id, now)
        _record_stage(ctx, 'cleanup', started)
        if plan.complete_run_id is None:
            return False
        rows = ctx.repo.fetch_run_rows(plan.complete_run_id)
        row = next((item for item in rows if item.status == 'promoting'), None)
        if row is None:
            # The rows no longer show what the states showed. Nobody may start a run before this is settled.
            logger.error(
                'run %s has no promoting row to complete; no run starts until it is settled', plan.complete_run_id
            )
            return True
        ctx.repo.insert_run_row(row.with_status('promoted', now))
        return not _confirmed(ctx, plan.complete_run_id)
    except Exception:  # noqa: BLE001 - completion is retried on the next poll, never abandoned
        # While this run is unfinished no faction ID can be issued twice, so nothing new starts.
        logger.exception('startup recovery failed; no run starts until it succeeds')
        return True


def _abandon(ctx: RunContext, run_id: str, now: datetime) -> None:
    """Mark a stale run failed and remove its rows. Nobody will complete it."""
    rows = ctx.repo.fetch_run_rows(run_id)
    if rows and not any(item.status == 'failed' for item in rows):
        ctx.repo.insert_run_row(rows[-1].with_failure('unknown', 'RunAbandoned', now))
    _cleanup(ctx, run_id)


def _cleanup(ctx: RunContext, run_id: str) -> None:
    """Remove the rows of a run from the run-owned tables. The caches are never touched."""
    for noun in schema.IMAGE_TABLES:
        ctx.repo.drop_partition(noun, run_id)
    for noun in schema.RUN_OWNED:
        if noun not in schema.IMAGE_TABLES and noun != schema.RUNS:
            ctx.repo.delete_rows(noun, run_id)


def _resend(ctx: RunContext, row: RunRow, statuses: tuple[str, ...], now: datetime) -> None:
    for status in statuses:
        ctx.repo.insert_run_row(row.with_status(status, now))  # ty: ignore[invalid-argument-type]
        ctx.trip(f'after_{status}_row')


def _confirmed(ctx: RunContext, run_id: str) -> bool:
    return any(item.status == 'promoted' for item in ctx.repo.fetch_run_rows(run_id))


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


# ---- one run


def run_once(ctx: RunContext, kind: RunKind) -> RunOutcome:
    """Compute one image and promote it, or bring the run to a settled state."""
    now = ctx.now()
    run = _new_run(ctx, kind, now)
    clock = _Clock()
    try:
        ctx.repo.insert_run_row(run)
        with clock('load_previous'):
            previous = _load_previous(ctx)
        with clock('extract_graph'):
            graph, dids, collisions = _extract_graph(ctx, run)
        with clock('partition'):
            partitioned = _partition(ctx, graph, dids, previous, kind)
        with clock('track'):
            tracked = _track(ctx, run, graph, dids, partitioned, previous)
        with clock('write_members'):
            _write(ctx, schema.MEMBERS, tracked.members)
            ctx.trip('after_write_members')
        with clock('train_model'):
            refresh = _refresh_model(ctx, kind, now)
        scorer = ctx.loader(ctx.model_dir)
        with clock('extract_interactions'):
            interactions = _extract_interactions(ctx, run, dids, tracked, scorer, clock)
        flags = _flags(ctx, interactions.events)
        with clock('write_features'):
            features = _write_features(ctx, run, graph, dids, tracked, interactions.events, flags)
            ctx.trip('after_write_features')
        with clock('write_conflict'):
            cells = _write_conflict(ctx, run, interactions, flags)
            ctx.trip('after_write_conflict')
        with clock('write_registry'):
            _write_registry(ctx, run, tracked, features, cells, previous, now)
            ctx.trip('after_write_registry')
        run = _complete(ctx, run, tracked, partitioned, interactions, collisions, refresh, scorer, clock)
        with clock('promote'):
            status = _promote(ctx, run, now)
        if status == 'unresolved':
            return RunOutcome('unresolved', 'the promotion was not confirmed by a read', run.run_id, kind, exit_code=1)
        _record_run(ctx, kind, clock, run)
        return RunOutcome('promoted', '', run.run_id, kind)
    except Exception as exc:
        stage = clock.stage or 'load_previous'
        # The traceback goes to the log. Telemetry receives the stage and the class of the error only.
        logger.exception('run %s failed in %s', run.run_id, stage)
        record_failure(ctx.telemetry, stage, exc)
        return _fail(ctx, run, stage, exc)


class _Clock:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.stage: str | None = None

    @contextmanager
    def __call__(self, stage: str) -> Generator[None]:
        self.stage = stage
        started = time.monotonic()
        try:
            yield
        finally:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + time.monotonic() - started


def _new_run(ctx: RunContext, kind: RunKind, now: datetime) -> RunRow:
    config = ctx.config
    window_end = now.replace(minute=0, second=0, microsecond=0)
    window = Window(window_end - timedelta(days=config.graph.window_days), window_end)
    return RunRow(
        run_id=new_run_id(_ms(now), secrets.token_hex(4)),
        status='running',
        kind=kind,
        recorded_at=now,
        started_at=now,
        window_start=window.start,
        window_end=window.end,
        window_days=window.days,
        objective=config.partition.objective,
        resolution=config.partition.resolution,
        seed=config.partition.seed,
        next_faction_id=0,
        accounts_placed=0,
        accounts_unplaced=0,
        hash_collisions=0,
        factions=0,
        moves=0,
        births=0,
        merges=0,
        splits=0,
        deaths=0,
        agreement_prev=None,
        modularity=None,
        source_gap_days=0,
        text_model_version='',
        text_coverage=0.0,
        stage_seconds={},
        failed_stage='',
        error_class='',
    )


def _run_window(run: RunRow) -> Window:
    return Window(run.window_start, run.window_end)


# ---- load


@dataclass(frozen=True)
class PreviousImage:
    """The previous promoted image, the confidence of the latest monthly image and the registry."""

    run: RunRow | None
    did: np.ndarray
    faction: np.ndarray
    since: np.ndarray
    confidence: np.ndarray
    faction_ids: np.ndarray
    next_faction_id: int
    registry: dict[int, tuple[int | None, str]]


def _load_previous(ctx: RunContext) -> PreviousImage:
    empty = np.empty(0, dtype=object)
    none = np.empty(0, dtype=np.int64)
    latest = ctx.repo.latest_promoted_run()
    if latest is None:
        return PreviousImage(None, empty, none, empty, empty, none, 1, {})
    image = ctx.repo.fetch_image(latest.run_id)
    did = image['did'].to_numpy(dtype=object)
    faction = image['faction_id'].to_numpy(dtype=np.int64)
    confidence = _optional_list(image['confidence'].to_numpy())
    if latest.kind != 'full':
        # A weekly run carries the confidence of the latest monthly image forward.
        latest_full = ctx.repo.latest_promoted_run('full')
        if latest_full is not None:
            full_image = ctx.repo.fetch_image(latest_full.run_id)
            merged = image.loc[:, ['did']].merge(full_image.loc[:, ['did', 'confidence']], on='did', how='left')
            confidence = _optional_list(merged['confidence'].to_numpy())
    registry_rows = ctx.repo.fetch_faction_registry()
    registry = {
        faction: (_as_id(parent), str(born))
        for faction, parent, born in zip(
            registry_rows['faction_id'].tolist(),
            registry_rows['parent_faction_id'].tolist(),
            registry_rows['born_run_id'].tolist(),
            strict=True,
        )
    }
    # A faction still active in the registry is retired when no account of the new image continues it, also
    # when the previous image itself has been removed by its retention.
    active = (
        registry_rows.loc[registry_rows['status'] == 'active', 'faction_id'].to_numpy(dtype=np.int64)
        if len(registry_rows)
        else none
    )
    return PreviousImage(
        run=latest,
        did=did,
        faction=faction,
        since=image['since_run_id'].to_numpy(dtype=object),
        confidence=confidence,
        faction_ids=np.unique(np.concatenate([np.unique(faction), active])),
        next_faction_id=int(latest.next_faction_id) or 1,
        registry=registry,
    )


def _as_id(value: object) -> int | None:
    """A faction ID from a row that may hold NULL.

    A nullable integer column comes back as floating point when any of its values is NULL, so a whole
    float is an ID as well.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.integer | np.floating):
        return None
    return int(value) if np.isfinite(value) and int(value) == value else None


def _optional_list(values: np.ndarray) -> np.ndarray:
    return np.array(
        [None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value) for value in values],
        dtype=object,
    )


def _optional_ints(values: np.ndarray) -> np.ndarray:
    return np.array([None if int(value) < 0 else int(value) for value in values], dtype=object)


def _optional_floats(values: np.ndarray) -> np.ndarray:
    return np.array([None if not np.isfinite(value) else float(value) for value in values], dtype=object)


# ---- extract


@dataclass(frozen=True)
class UnnamedAccounts:
    """Accounts the graph cannot name. They stay unplaced and their edges are dropped."""

    hash_collisions: int
    accounts: int


def _extract_graph(ctx: RunContext, run: RunRow) -> tuple[AffinityGraph, np.ndarray, UnnamedAccounts]:
    """Pull the window, fold it into the affinity graph and name the DID of every vertex.

    An account hash that two DIDs share is a collision: one DID cannot be told from the other, so every
    account behind it stays unplaced. A hash on an edge that the account query does not name at all is
    treated the same way. The query reads every DID of the window, so that should not happen.
    """
    window = _run_window(run)
    follows = [block for block in ctx.repo.stream_follow_creates(window)]
    reposts = [block for block in ctx.repo.stream_repost_creates(window)]
    f_src, f_dst, f_uri, f_ms = (_concat(follows, name) for name in ('src', 'dst', 'uri_hash', 'event_ms'))
    r_src, r_dst = (_concat(reposts, name) for name in ('src', 'dst'))
    deleted = ctx.repo.fetch_deletes(window, FOLLOW_COLLECTION)
    f_src, f_dst, _removed = drop_deleted_creates(f_src, f_dst, f_uri, f_ms, deleted)
    accounts = ctx.repo.fetch_account_hashes(window)
    shared = accounts.loc[accounts['dids'] > 1]
    single = accounts.loc[accounts['dids'] == 1].sort_values('h')
    named = single['h'].to_numpy(dtype=np.uint64)
    on_edges = index_vertices(f_src, f_dst, r_src, r_dst)[0]
    dropped = on_edges[~np.isin(on_edges, named)]
    if len(dropped):
        keep = ~np.isin(f_src, dropped) & ~np.isin(f_dst, dropped)
        f_src, f_dst = f_src[keep], f_dst[keep]
        keep = ~np.isin(r_src, dropped) & ~np.isin(r_dst, dropped)
        r_src, r_dst = r_src[keep], r_dst[keep]
    behind = np.isin(shared['h'].to_numpy(dtype=np.uint64), dropped)
    edges = aggregate_window_edges(f_src, f_dst, r_src, r_dst)
    graph = build_graph(edges, ctx.config.graph)
    # Every vertex left is named, because each hash on an edge that has no single DID was dropped.
    names = single['did'].to_numpy(dtype=object)
    clipped = np.minimum(np.searchsorted(named, graph.account_hash), max(len(named) - 1, 0))
    unnamed = UnnamedAccounts(
        hash_collisions=int(behind.sum()),
        accounts=int(shared['dids'].to_numpy()[behind].sum()),
    )
    return graph, names[clipped], unnamed


def _concat(blocks: list[dict[str, np.ndarray]], name: str, dtype: np.dtype | None = None) -> np.ndarray:
    if not blocks:
        return np.empty(0, dtype=dtype or np.uint64)
    parts = [np.asarray(block[name]) for block in blocks]
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


# ---- partition and track


@dataclass(frozen=True)
class Partitioned:
    """The partition after absorption, with dense faction indices per vertex."""

    membership: np.ndarray
    dense: np.ndarray
    faction_ids: np.ndarray
    unplaced: int
    modularity: float
    confidence: np.ndarray


def _partition(
    ctx: RunContext, graph: AffinityGraph, dids: np.ndarray, previous: PreviousImage, kind: RunKind
) -> Partitioned:
    config = ctx.config.partition
    u, v, weight = graph.u, graph.v, graph.weight
    initial = None
    if kind == 'weekly' and previous.run is not None:
        position = pd.Index(previous.did).get_indexer(dids)
        found = position >= 0
        labels = np.unique(previous.faction)
        before = np.full(graph.vertices, UNPLACED, dtype=np.int64)
        before[found] = np.searchsorted(labels, previous.faction[position[found]])
        initial = warm_start_membership(before)
    result = run_partition(
        build_igraph(graph.vertices, u, v, weight),
        objective=config.objective,
        resolution=config.resolution,
        seed=config.seed,
        n_iterations=config.n_iterations,
        initial_membership=initial,
    )
    try:
        absorbed = absorb_small_communities(result.membership, u, v, weight, config.min_faction_size)
    except NoFactionError as exc:
        raise NoFactionError('no_faction_reached_min_size') from exc
    confidence = np.full(graph.vertices, np.nan, dtype=np.float64)
    if kind == 'full' and config.confidence_seeds > 0:
        extra = []
        for offset in range(1, config.confidence_seeds + 1):
            other = run_partition(
                build_igraph(graph.vertices, u, v, weight),
                objective=config.objective,
                resolution=config.resolution,
                seed=config.seed + offset,
                n_iterations=config.n_iterations,
            )
            extra.append(absorb_small_communities(other.membership, u, v, weight, config.min_faction_size).membership)
        confidence = seed_confidence(absorbed.membership, extra, graph.degree(), split_share=config.split_share)
    placed = absorbed.membership >= 0
    faction_ids = np.unique(absorbed.membership[placed]).astype(np.int64)
    dense = np.full(graph.vertices, -1, dtype=np.int64)
    dense[placed] = np.searchsorted(faction_ids, absorbed.membership[placed])
    return Partitioned(
        membership=absorbed.membership,
        dense=dense,
        faction_ids=faction_ids,
        unplaced=absorbed.unplaced_accounts,
        modularity=weighted_modularity(result.membership, u, v, weight),
        confidence=confidence,
    )


@dataclass(frozen=True)
class TrackedRun:
    faction: np.ndarray
    faction_ids: np.ndarray
    dense: np.ndarray
    members: pd.DataFrame
    diff: MoveDiff
    lineage: tuple[LineageEvent, ...]
    next_faction_id: int
    agreement: float | None
    degree: np.ndarray
    placed_dids: np.ndarray


def _track(
    ctx: RunContext,
    run: RunRow,
    graph: AffinityGraph,
    dids: np.ndarray,
    partitioned: Partitioned,
    previous: PreviousImage,
) -> TrackedRun:
    """Give the communities faction IDs, then compare the image with the previous promoted image."""
    vertices = graph.vertices
    previous_faction = np.full(vertices, UNPLACED, dtype=np.int64)
    confidence = partitioned.confidence
    # The image of the latest promoted run can be empty when its retention removed it.
    if previous.run is not None and len(previous.did):
        position = pd.Index(previous.did).get_indexer(dids)
        found = position >= 0
        previous_faction[found] = previous.faction[position[found]]
        carried = np.array(
            [np.nan if value is None else float(value) for value in previous.confidence[position[found]]],
            dtype=np.float64,
        )
        # A weekly run carries the monthly confidence forward for accounts the monthly image placed.
        confidence = np.where(np.isnan(partitioned.confidence), carried, partitioned.confidence)
    degree = graph.degree()
    tracked = track(
        previous_faction,
        partitioned.membership,
        degree,
        next_faction_id=previous.next_faction_id,
        split_share=ctx.config.partition.split_share,
        previous_faction_ids=previous.faction_ids if previous.run is not None else None,
    )
    placed = partitioned.membership >= 0
    diff = diff_moves(previous.did, previous.faction, previous.since, dids[placed], tracked.faction[placed], run.run_id)
    before = np.full(vertices, UNPLACED, dtype=np.int64)
    before[placed] = diff.prev_faction_id
    members = pd.DataFrame(
        {
            'run_id': run.run_id,
            'did': pd.Series(dids[placed], dtype=object),
            'faction_id': tracked.faction[placed].astype('uint32'),
            'since_run_id': pd.Series(diff.since_run_id, dtype=object),
            'prev_faction_id': _optional_ints(before[placed]),
            'confidence': _optional_floats(confidence[placed]),
            'retire_at': ctx.now() + (FULL_RETENTION if run.kind == 'full' else WEEKLY_RETENTION),
        }
    )
    # The faction ID of each community, in the order the conflict cells index them. Every community has at
    # least one account, so each entry is written.
    community = partitioned.dense
    in_a_community = community >= 0
    faction_of_community = np.empty(len(partitioned.faction_ids), dtype=np.int64)
    faction_of_community[community[in_a_community]] = tracked.faction[in_a_community]
    return TrackedRun(
        faction=tracked.faction,
        faction_ids=faction_of_community,
        dense=community,
        members=members,
        diff=diff,
        lineage=tracked.lineage,
        next_faction_id=tracked.next_faction_id,
        agreement=tracked.adjusted_rand,
        degree=degree,
        placed_dids=dids[placed],
    )


# ---- write


def _write(ctx: RunContext, noun: str, frame: pd.DataFrame) -> None:
    """Write a frame. In a dry run the repository keeps it instead of sending it."""
    ctx.repo.insert_frame(noun, frame)


def _refresh_model(ctx: RunContext, kind: RunKind, now: datetime) -> train.RefreshOutcome | None:
    """On a monthly run, refresh the model first when it is missing or old."""
    if kind != 'full' or ctx.trainer is None or ctx.dry_run:
        return None
    manifest = model_store.current_manifest(ctx.model_dir)
    if not model_store.is_stale(manifest, now=now, max_age_days=ctx.config.label.model_max_age_days):
        return None
    return ctx.trainer(ctx.repo, ctx.config, ctx.model_dir, now)


@dataclass(frozen=True)
class Interactions:
    events: Events
    coverage: float
    model_version: str


def _extract_interactions(
    ctx: RunContext,
    run: RunRow,
    dids: np.ndarray,
    tracked: TrackedRun,
    scorer: Scorer | None,
    clock: _Clock,
) -> Interactions:
    """Stream the interactions whose two ends are placed and score the texts that have no score.

    Faction attribution always comes from the image of this run, so a cached score follows a move or a
    split without being computed again. A block exists from its create until its delete, as a follow does,
    so a block that a later delete removed is no signal.
    """
    index = pd.Index(dids)
    dense = tracked.dense
    window = _run_window(run)
    deleted_blocks = ctx.repo.fetch_deletes(window, BLOCK_COLLECTION)
    cache_hash: np.ndarray = np.empty(0, dtype=np.uint64)
    cache_h: np.ndarray = np.empty(0, dtype=np.float64)
    if scorer is not None:
        cache_hash, cache_h = ctx.repo.fetch_text_scores(scorer.version)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in EVENT_PARTS}
    texts_total = 0
    scored_total = 0
    for frame in ctx.repo.stream_interactions(window):
        author = index.get_indexer(frame['author_did'].astype(object))
        target = index.get_indexer(frame['target_did'].astype(object))
        kind = frame['kind'].to_numpy(dtype=np.int8)
        superseded = superseded_mask(
            frame['uri_hash'].to_numpy(dtype=np.uint64), frame['event_ms'].to_numpy(dtype=np.int64), deleted_blocks
        )
        keep = (kind != BLOCK) | ~superseded
        keep &= (
            (author >= 0) & (target >= 0) & (dense[np.maximum(author, 0)] >= 0) & (dense[np.maximum(target, 0)] >= 0)
        )
        if not keep.any():
            continue
        author, target, kind = author[keep], target[keep], kind[keep]
        kept = frame.loc[keep]
        uri = kept['uri_hash'].to_numpy(dtype=np.uint64)
        h = np.full(len(kept), np.nan, dtype=np.float32)
        is_text = kind != BLOCK
        texts_total += int(is_text.sum())
        if scorer is not None:
            cached = np.zeros(len(uri), dtype=bool)
            if len(cache_hash):
                clipped = np.minimum(np.searchsorted(cache_hash, uri), len(cache_hash) - 1)
                cached = cache_hash[clipped] == uri
                h[cached] = cache_h[clipped][cached]
            text = kept['text']
            has_text = is_text & text.notna().to_numpy() & (text.astype(str).str.len().to_numpy() > 0)
            todo = has_text & ~cached
            if todo.any():
                with clock('score_text'):
                    fresh = _score(ctx, kept.loc[todo], scorer)
                h[todo] = fresh['h'].to_numpy(dtype=np.float32)
                _write(ctx, schema.TEXT_SCORES, fresh)
                scored_total += int(todo.sum())
            scored_total += int((has_text & cached).sum())
        parts['event_ms'].append(kept['event_ms'].to_numpy(dtype=np.int64))
        parts['kind'].append(kind)
        parts['author'].append(author.astype(np.int64))
        parts['target'].append(target.astype(np.int64))
        parts['from_faction'].append(dense[author])
        parts['to_faction'].append(dense[target])
        parts['thread_key'].append(kept['thread_key'].to_numpy(dtype=np.uint64))
        parts['h'].append(h)
    dtypes = {
        'event_ms': np.int64,
        'kind': np.int8,
        'author': np.int64,
        'target': np.int64,
        'from_faction': np.int64,
        'to_faction': np.int64,
        'thread_key': np.uint64,
        'h': np.float32,
    }
    events = Events(
        **{name: _concat_list(parts[name], dtypes[name]) for name in EVENT_PARTS},
        accounts=len(dids),
        factions=len(tracked.faction_ids),
        faction_ids=tracked.faction_ids,
    )
    return Interactions(events, scored_total / texts_total if texts_total else 0.0, scorer.version if scorer else '')


def _concat_list(parts: list[np.ndarray], dtype: type | np.dtype) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate([np.asarray(part, dtype=dtype) for part in parts])


def _score(ctx: RunContext, rows: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Score texts with the model on disk. Text is stored for examples only."""
    label = ctx.config.label
    frame = text_frame(
        [value if isinstance(value, str) else '' for value in rows['text'].tolist()],
        np.asarray(rows[LENGTH], dtype=np.int64),
        np.asarray(rows[UPPERCASE], dtype=np.int64),
        np.asarray(rows[EXCLAMATIONS], dtype=np.int64),
        np.asarray(rows[QUESTIONS], dtype=np.int64),
        np.asarray(rows[MENTIONS], dtype=np.int64),
        np.asarray(rows['kind'], dtype=np.int8),
    )
    h = scorer.probabilities(frame).astype(np.float32)
    uri = rows['uri_hash'].to_numpy(dtype=np.uint64)
    example = (h >= label.example_min_h) | (uri % np.uint64(label.example_sample_modulus) == 0)
    return pd.DataFrame(
        {
            'uri_hash': uri,
            'h': h,
            'model_version': scorer.version,
            'kind': [KIND_NAMES[int(value)] for value in rows['kind'].tolist()],
            'scored_at': ctx.now(),
            'is_example': example.astype(np.uint8),
            'author_did': _only_examples(example, rows['author_did'].to_numpy(dtype=object)),
            'target_did': _only_examples(example, rows['target_did'].to_numpy(dtype=object)),
            'post_text': _only_examples(example, rows['text'].to_numpy(dtype=object)),
            'event_at': _only_examples(
                example,
                np.array(
                    [datetime.fromtimestamp(int(ms) / 1000, tz=UTC) for ms in rows['event_ms'].tolist()], dtype=object
                ),
            ),
        }
    )


def _only_examples(example: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.array(
        [value if keep else None for value, keep in zip(values, example.tolist(), strict=True)], dtype=object
    )


def _flags(ctx: RunContext, events: Events) -> tuple[np.ndarray, np.ndarray]:
    """Mark the pile-on replies and the swarm quotes once, for every window."""
    config = ctx.config.conflict
    return swarm_flags(
        events,
        min_repliers=config.pileon_min_repliers,
        reply_window_minutes=config.pileon_window_minutes,
        min_quoters=config.swarm_min_quoters,
        quote_window_minutes=config.swarm_window_minutes,
    )


@dataclass(frozen=True)
class Features:
    hostile: HostileCounts
    descriptions: dict[int, tuple[list[str], list[str], list[str]]]


def _write_features(
    ctx: RunContext,
    run: RunRow,
    graph: AffinityGraph,
    dids: np.ndarray,
    tracked: TrackedRun,
    events: Events,
    flags: tuple[np.ndarray, np.ndarray],
) -> Features:
    """Per-account features, then the three description reads against the written image."""
    placed = tracked.faction >= 0
    degree = tracked.degree
    counts = hostile_counts(events, flags[0], flags[1])
    frame = pd.DataFrame(
        {
            'run_id': run.run_id,
            'faction_id': tracked.faction[placed].astype('uint32'),
            'did': pd.Series(dids[placed], dtype=object),
            'degree': degree[placed],
            'within_z': _optional_floats(within_faction_z(degree, tracked.faction)[placed]),
            'participation': _optional_floats(
                participation_coefficient(tracked.faction, graph.u, graph.v, graph.weight, degree)[placed]
            ),
            'pileon_replies_sent': counts.pileon_sent[placed].astype('uint32'),
            'pileon_replies_received': counts.pileon_received[placed].astype('uint32'),
            'swarm_quotes_sent': counts.swarm_sent[placed].astype('uint32'),
            'swarm_quotes_received': counts.swarm_received[placed].astype('uint32'),
            'blocks_sent': counts.blocks_sent[placed].astype('uint32'),
            'blocks_received': counts.blocks_received[placed].astype('uint32'),
            'text_hostility_sent': counts.text_sent[placed].astype('float32'),
            'text_hostility_received': counts.text_received[placed].astype('float32'),
            'retire_at': ctx.now() + (FULL_RETENTION if run.kind == 'full' else WEEKLY_RETENTION),
        }
    )
    _write(ctx, schema.MEMBER_FEATURES, frame)
    descriptions: dict[int, tuple[list[str], list[str], list[str]]] = {
        int(faction): ([], [], []) for faction in tracked.faction_ids.tolist()
    }
    if ctx.dry_run:
        return Features(counts, descriptions)
    window = _run_window(run)
    tags = _top_lists(ctx.repo.fetch_tags(run.run_id, window))
    languages = _top_lists(ctx.repo.fetch_languages(run.run_id, window))
    handles = _top_lists(ctx.repo.fetch_handles(run.run_id, window), value='handle')
    for faction in descriptions:
        descriptions[faction] = (tags.get(faction, []), languages.get(faction, []), handles.get(faction, []))
    return Features(counts, descriptions)


def _top_lists(frame: pd.DataFrame, value: str | None = None) -> dict[int, list[str]]:
    """The values of each faction in the order the query returned them, which is by descending count."""
    lists: dict[int, list[str]] = {}
    if len(frame) == 0:
        return lists
    column = value if value is not None else frame.columns[1]
    for faction, item in zip(frame['faction_id'].tolist(), frame[column].tolist(), strict=True):
        # A missing value is None or NA, depending on the column. Neither is a description.
        if item is not None and not pd.isna(item):
            lists.setdefault(int(faction), []).append(str(item))
    return lists


@dataclass(frozen=True)
class Cells:
    """The cells of the longest window and, per faction, how many it sent and its best score."""

    long: pd.DataFrame
    per_faction: dict[int, tuple[int, float | None]]


def _write_conflict(
    ctx: RunContext, run: RunRow, interactions: Interactions, flags: tuple[np.ndarray, np.ndarray]
) -> Cells:
    config = ctx.config.conflict
    events = interactions.events
    end = _ms(run.window_end)
    longest = pd.DataFrame(columns=list(schema.CONFLICT_COLUMNS))
    for days in sorted(config.conflict_window_days):
        mask = events.event_ms >= end - days * 86_400_000
        counts = cell_counts(events.take(mask), flags[0][mask], flags[1][mask])
        frame = cells_frame(
            counts,
            prior_strength=config.cell_prior_strength,
            min_text_scored=config.min_text_scored,
            has_model=bool(interactions.model_version),
        )
        frame.insert(0, 'window_days', np.full(len(frame), days, dtype='uint16'))
        frame.insert(0, 'run_id', run.run_id)
        frame['written_at'] = ctx.now()
        longest = frame
        _write(ctx, schema.CONFLICT, frame.loc[:, list(schema.CONFLICT_COLUMNS)])
    per_faction: dict[int, tuple[int, float | None]] = {}
    if len(longest):
        for faction, group in longest.groupby('from_faction_id'):
            scores = [value for value in group['conflict_score'].tolist() if value is not None]
            per_faction[_as_id(faction) or 0] = (int(len(group)), max(scores) if scores else None)
    return Cells(longest, per_faction)


def _write_registry(
    ctx: RunContext,
    run: RunRow,
    tracked: TrackedRun,
    features: Features,
    cells: Cells,
    previous: PreviousImage,
    now: datetime,
) -> None:
    """The registry, the per-run faction statistics, the lineage and the moves."""
    registry = previous.registry
    parents = {
        event.faction_id: event.related_faction_id
        for event in tracked.lineage
        if event.event == 'split' and event.related_faction_id is not None
    }
    counts = features.hostile
    faction_rows = []
    run_rows = []
    for faction in sorted(tracked.faction_ids.tolist()):
        here = tracked.faction == faction
        known = registry.get(int(faction))
        parent = parents.get(int(faction), known[0] if known else None)
        tags, languages, handles = features.descriptions.get(int(faction), ([], [], []))
        sent = counts.pileon_sent[here].sum() + counts.swarm_sent[here].sum() + counts.blocks_sent[here].sum()
        received = (
            counts.pileon_received[here].sum() + counts.swarm_received[here].sum() + counts.blocks_received[here].sum()
        )
        cell_count, best = cells.per_faction.get(int(faction), (0, None))
        faction_rows.append(
            {
                'faction_id': int(faction),
                'run_id': run.run_id,
                'status': 'active',
                'parent_faction_id': _as_id(parent),
                'born_run_id': run.run_id if known is None else known[1],
                'member_count': int(here.sum()),
                'top_tags': tags,
                'top_languages': languages,
                'sample_handles': handles,
                'written_at': now,
            }
        )
        run_rows.append(
            {
                'run_id': run.run_id,
                'faction_id': int(faction),
                'member_count': int(here.sum()),
                'mean_degree': float(tracked.degree[here].mean()) if here.any() else 0.0,
                'hostile_sent': int(sent),
                'hostile_received': int(received),
                'conflict_cells': cell_count,
                'max_conflict_score': best,
                'written_at': now,
            }
        )
    for event in tracked.lineage:
        if event.event in {'merge', 'death'}:
            known = registry.get(int(event.faction_id))
            faction_rows.append(
                {
                    'faction_id': int(event.faction_id),
                    'run_id': run.run_id,
                    'status': 'merged' if event.event == 'merge' else 'retired',
                    'parent_faction_id': None if known is None else _as_id(known[0]),
                    'born_run_id': known[1] if known else run.run_id,
                    'member_count': 0,
                    'top_tags': [],
                    'top_languages': [],
                    'sample_handles': [],
                    'written_at': now,
                }
            )
    _write(ctx, schema.FACTIONS, pd.DataFrame(faction_rows, columns=list(schema.FACTIONS_COLUMNS)))
    _write(ctx, schema.FACTION_RUNS, pd.DataFrame(run_rows, columns=list(schema.FACTION_RUNS_COLUMNS)))
    _write(
        ctx,
        schema.LINEAGE,
        pd.DataFrame(
            [
                (run.run_id, int(event.faction_id), event.event, event.related_faction_id, now)
                for event in tracked.lineage
            ],
            columns=list(schema.LINEAGE_COLUMNS),
        ),
    )
    moves = tracked.diff.moves
    _write(
        ctx,
        schema.MOVES,
        pd.DataFrame(
            {
                'run_id': run.run_id,
                'did': pd.Series(moves.did, dtype=object),
                'move_kind': [MOVE_KINDS[int(code)] for code in moves.kind.tolist()],
                'from_faction_id': _optional_ints(moves.from_faction_id),
                'to_faction_id': _optional_ints(moves.to_faction_id),
                'written_at': now,
            },
            columns=list(schema.MOVES_COLUMNS),
        ),
    )


def _complete(
    ctx: RunContext,
    run: RunRow,
    tracked: TrackedRun,
    partitioned: Partitioned,
    interactions: Interactions,
    collisions: UnnamedAccounts,
    refresh: train.RefreshOutcome | None,
    scorer: Scorer | None,
    clock: _Clock,
) -> RunRow:
    """The complete run row, with every count, that the promotion records."""
    events = {'birth': 0, 'merge': 0, 'split': 0, 'death': 0}
    for event in tracked.lineage:
        events[event.event] += 1
    return replace(
        run,
        next_faction_id=tracked.next_faction_id,
        accounts_placed=int((tracked.faction >= 0).sum()),
        accounts_unplaced=partitioned.unplaced + collisions.accounts,
        hash_collisions=collisions.hash_collisions,
        factions=len(tracked.faction_ids),
        moves=len(tracked.diff.moves),
        births=events['birth'],
        merges=events['merge'],
        splits=events['split'],
        deaths=events['death'],
        agreement_prev=tracked.agreement,
        modularity=partitioned.modularity,
        source_gap_days=_source_gap_days(ctx, run),
        text_model_version=scorer.version if scorer else '',
        text_coverage=interactions.coverage,
        stage_seconds={name: float(seconds) for name, seconds in clock.seconds.items()},
    )


def _source_gap_days(ctx: RunContext, run: RunRow) -> int:
    """Days in the window with fewer than a quarter of the median number of source rows."""
    window = _run_window(run)
    counts = ctx.repo.fetch_daily_row_counts(window)
    rows = []
    day = window.start.astimezone(UTC).date()
    while day < window.end.astimezone(UTC).date():
        rows.append(counts.get(day, 0))
        day += timedelta(days=1)
    if not rows:
        return 0
    median = float(np.median(rows))
    return int(sum(1 for value in rows if value < median / 4))


# ---- promote, fail


def _promote(ctx: RunContext, run: RunRow, now: datetime) -> Literal['promoted', 'unresolved']:
    """Record the intent, write both rows, and clear the intent only once a read confirms them.

    A dry run keeps its repository, which records the rows instead of sending them. It writes no intent file
    and has nothing to confirm.
    """
    run = run.with_status('promoting', now)
    if not ctx.dry_run:
        lock.install(ctx.model_dir, promotion.IntentRecord(run))
    ctx.trip('after_intent')
    ctx.repo.insert_run_row(run)
    ctx.trip('after_promoting_row')
    ctx.repo.insert_run_row(run.with_status('promoted', now))
    ctx.trip('after_promoted_row')
    if ctx.dry_run:
        return 'promoted'
    if not _confirmed(ctx, run.run_id):
        return 'unresolved'
    lock.clear(ctx.model_dir)
    return 'promoted'


def _fail(ctx: RunContext, run: RunRow, stage: str, exc: Exception) -> RunOutcome:
    """Bring a failed run to a settled state. Recovery is forward only."""
    error = type(exc).__name__
    failed = RunOutcome('failed', error, run.run_id, run.kind, stage, exit_code=1)
    if ctx.dry_run:
        return failed
    now = ctx.now()
    # An unreadable intent file raises here on purpose: its run must not be cleaned up.
    intent = lock.load(ctx.model_dir)
    if intent is not None and intent.run.run_id != run.run_id:
        intent = None
    state: RunState | None = None
    try:
        state = ctx.repo.fetch_run_states().get(run.run_id)
    except Exception:  # noqa: BLE001 - a failed re-read is the case the decision must account for
        state = None
    recovery = promotion.recover(run, intent, state)
    try:
        if recovery.mark_failed:
            ctx.repo.insert_run_row(run.with_failure(stage, error, now))
        if recovery.cleanup:
            _cleanup(ctx, run.run_id)
    except Exception:  # noqa: BLE001 - the rows of a run that is not promoted stay invisible
        logger.exception('recording the failure of run %s did not complete. The next start cleans it up', run.run_id)
        return failed
    if recovery.resend:
        try:
            _resend(ctx, intent.run if intent else run, recovery.resend, now)
        except Exception:  # noqa: BLE001 - the intent stays and the next poll completes the run
            logger.exception('completing run %s failed. It is retried on the next poll', run.run_id)
            return RunOutcome(
                'unresolved',
                f'completing the run failed with {type(exc).__name__}; it is retried on the next poll',
                run.run_id,
                run.kind,
                stage,
                exit_code=1,
            )
        if _confirmed(ctx, run.run_id):
            lock.clear(ctx.model_dir)
            # The intent was recorded, so the run was fully written. It is now promoted, whatever broke.
            return RunOutcome('promoted', f'completed after {error} in {stage}', run.run_id, run.kind, stage)
        return RunOutcome(
            'unresolved', 'the promotion was not confirmed by a read', run.run_id, run.kind, stage, exit_code=1
        )
    if recovery.stop:
        return RunOutcome('stopped', 'the runs table could not be re-read', run.run_id, run.kind, stage, exit_code=1)
    return failed


# ---- telemetry


def _record_stage(ctx: RunContext, stage: str, started: float) -> None:
    ctx.telemetry.stage_duration_seconds.record(time.monotonic() - started, {'stage': stage})


def _record_run(ctx: RunContext, kind: RunKind, clock: _Clock, run: RunRow) -> None:
    record_run_metrics(
        ctx.telemetry,
        sum(clock.seconds.values()),
        run_kind=kind,
        accounts_placed=run.accounts_placed,
        accounts_unplaced=run.accounts_unplaced,
        factions=run.factions,
        moves=run.moves,
        had_rows=run.accounts_placed > 0,
    )

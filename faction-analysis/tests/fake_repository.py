"""An in-memory ``RunRepository`` over a tiny world, for the pipeline tests.

It keeps what a run writes per run, holds inserts that should not be visible yet, and can make an insert
fail. It never touches ClickHouse.
"""

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from faction_analysis import schema
from faction_analysis.graph import DeletedRecords, fold_deletes
from faction_analysis.labels import LabelInput
from faction_analysis.queries import BLOCK_COLLECTION, FOLLOW_COLLECTION, Window
from faction_analysis.runstate import RunRow, RunState

MEMBER_COLUMNS = ('did', 'faction_id', 'since_run_id', 'prev_faction_id', 'confidence')


@dataclass
class PipelineWorld:
    """What the source holds: edges, accounts, interactions and the material of descriptions."""

    follows: list[tuple[int, int, int, int]] = field(default_factory=list)
    reposts: list[tuple[int, int]] = field(default_factory=list)
    deletes: list[tuple[int, int]] = field(default_factory=list)
    block_deletes: list[tuple[int, int]] = field(default_factory=list)
    accounts: list[tuple[int, str, int]] = field(default_factory=list)
    interactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    tags: dict[str, list[str]] = field(default_factory=dict)
    languages: dict[str, list[str]] = field(default_factory=dict)
    handles: dict[str, str] = field(default_factory=dict)
    daily_rows: dict[date, int] = field(default_factory=dict)


def two_faction_world(
    now: datetime, *, split: bool = False, collision: bool = False
) -> tuple[PipelineWorld, list[str]]:
    """Two cliques of six accounts, one cross follow, a pile-on, a swarm and two blocks.

    With ``split`` the first clique is cut into two halves that are no longer connected, so the next run
    sees a split. With ``collision`` the first two accounts hash to the same value.
    """
    dids = [f'did:plc:{index:024x}' for index in range(12)]
    hash_of = {did: index + 1 for index, did in enumerate(dids)}
    if collision:
        hash_of[dids[1]] = hash_of[dids[0]]
    base = int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
    world = PipelineWorld()
    # Two cliques of six accounts and one follow across. With ``split`` every edge between the halves
    # {0, 1, 2} and {3, 4, 5} of the first clique is gone, so it falls apart in the next run.
    pairs: list[tuple[int, int]] = [
        (a, b) for start in (0, 6) for a in range(start, start + 6) for b in range(a + 1, start + 6)
    ]
    if split:
        cut = {(a, b) for a in (0, 1, 2) for b in (3, 4, 5)}
        pairs = [pair for pair in pairs if pair not in cut]
    else:
        pairs.append((0, 6))
    uri = 1_000
    for a, b in pairs:
        world.follows.append((hash_of[dids[a]], hash_of[dids[b]], uri, base - 30 * 86_400_000))
        world.follows.append((hash_of[dids[b]], hash_of[dids[a]], uri + 1, base - 30 * 86_400_000))
        uri += 2
    for a, b in ((0, 1), (0, 2), (6, 7)):
        world.reposts.append((hash_of[dids[a]], hash_of[dids[b]]))
    world.accounts = [(hash_of[did], did, 1) for did in dids]
    if collision:
        # The account query returns one row per hash, with the number of DIDs behind it.
        world.accounts = [(hash_of[dids[0]], dids[0], 2)] + world.accounts[2:]
    start = base - 86_400_000
    rows = [
        (start, 0, 1, 'a', 0, 6, 101, 'you are an idiot', 0.9),
        (start + 60_000, 0, 2, 'a', 1, 6, 101, 'nobody asked you!!!', 0.8),
        (start + 120_000, 0, 3, 'b', 6, 0, 102, 'thanks, that helped', 0.1),
        (start + 180_000, 1, 4, 'b', 7, 1, 201, 'SHUT UP loser', 0.7),
        (start + 240_000, 1, 5, 'b', 8, 1, 201, 'what an idiot', 0.85),
        (start + 300_000, 2, 6, 'a', 0, 6, 0, None, None),
        (start + 360_000, 2, 7, 'a', 2, 7, 0, None, None),
        (start + 420_000, 0, 8, 'a', 3, 6, 103, '', None),
    ]
    world.interactions = pd.DataFrame(
        {
            'event_ms': [row[0] for row in rows],
            'kind': np.asarray([row[1] for row in rows], dtype=np.int8),
            'uri_hash': np.asarray([row[2] for row in rows], dtype=np.uint64),
            'author_did': pd.Series([dids[row[4]] for row in rows], dtype=object),
            'target_did': pd.Series([dids[row[5]] for row in rows], dtype=object),
            'thread_key': np.asarray([row[6] for row in rows], dtype=np.uint64),
            'parent_uri': ['at://parent/' + str(row[6]) for row in rows],
            'text': pd.Series([row[7] for row in rows], dtype=object),
            'languages': [('en',)] * len(rows),
            'text_length': np.asarray([len(row[7]) if row[7] else 0 for row in rows], dtype=np.int64),
            'uppercase': np.asarray(
                [sum(1 for char in row[7] if char.isupper()) if row[7] else 0 for row in rows], dtype=np.int64
            ),
            'exclamations': np.asarray([row[7].count('!') if row[7] else 0 for row in rows], dtype=np.int64),
            'questions': np.asarray([row[7].count('?') if row[7] else 0 for row in rows], dtype=np.int64),
            'mentions': np.asarray([row[7].count('@') if row[7] else 0 for row in rows], dtype=np.int64),
        }
    )
    for index, did in enumerate(dids):
        world.tags[did] = ['art', 'ink'] if index < 6 else ['music']
        world.languages[did] = ['en'] if index % 2 else ['de', 'en']
        world.handles[did] = f'user{index}.bsky.social'
    world.daily_rows = {date(2026, 11, 26): 1_000, date(2026, 11, 25): 1_000}
    return world, dids


class FakeRepository:
    """The ``RunRepository`` protocol over a :class:`PipelineWorld`."""

    def __init__(self, world: PipelineWorld) -> None:
        self.world = world
        self.run_rows: dict[str, list[RunRow]] = defaultdict(list)
        self.run_owned: dict[str, dict[str, pd.DataFrame]] = {
            noun: {} for noun in schema.RUN_OWNED if noun != schema.RUNS
        }
        self.cache_frames: dict[str, list[pd.DataFrame]] = {noun: [] for noun in schema.CACHES}
        self.dropped: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self.held: dict[str, list[RunRow]] = defaultdict(list)
        self.fail_on_status: set[str] = set()
        self.hold_next_promoting = False

    # runs

    def insert_run_row(self, row: RunRow) -> None:
        if row.status in self.fail_on_status:
            raise RuntimeError(f'insert of a {row.status} row failed')
        if self.hold_next_promoting and row.status == 'promoting':
            # The first promoting row commits late: its run stays invisible until ``release``.
            self.hold_next_promoting = False
            self.hold(row.run_id)
        if row.run_id in self.held:
            self.held[row.run_id].append(row)
            return
        self.run_rows[row.run_id].append(row)

    def fetch_run_rows(self, run_id: str) -> list[RunRow]:
        return sorted(self.run_rows.get(run_id, []), key=lambda item: item.recorded_at)

    def fetch_run_states(self) -> dict[str, RunState]:
        states: dict[str, RunState] = {}
        for run_id, rows in self.run_rows.items():
            statuses = {item.status for item in rows}
            states[run_id] = RunState(
                run_id=run_id,
                promoted='promoted' in statuses,
                failed='failed' in statuses,
                promoting='promoting' in statuses,
                running='running' in statuses,
            )
        return states

    def latest_promoted_run(self, kind: str | None = None) -> RunRow | None:
        best: RunRow | None = None
        for rows in self.run_rows.values():
            for row in rows:
                if row.status != 'promoted' or (kind is not None and row.kind != kind):
                    continue
                if best is None or row.run_id > best.run_id:
                    best = row
        return best

    def fetch_image(self, run_id: str) -> pd.DataFrame:
        frame = self.run_owned[schema.MEMBERS].get(run_id)
        if frame is None:
            return pd.DataFrame(columns=list(MEMBER_COLUMNS))
        return frame.loc[:, list(MEMBER_COLUMNS)]

    def fetch_faction_registry(self) -> pd.DataFrame:
        latest: dict[int, pd.Series] = {}
        for run_id in sorted(self.run_owned[schema.FACTIONS]):
            if not any(row.status == 'promoted' for row in self.run_rows.get(run_id, [])):
                continue
            for _, row in self.run_owned[schema.FACTIONS][run_id].iterrows():
                latest[int(row['faction_id'])] = row
        if not latest:
            return pd.DataFrame(columns=['faction_id', 'status', 'parent_faction_id', 'born_run_id'])
        return pd.DataFrame(list(latest.values())).loc[:, ['faction_id', 'status', 'parent_faction_id', 'born_run_id']]

    # source

    def stream_follow_creates(self, window: Window) -> Iterator[dict[str, np.ndarray]]:
        start, end = _bounds(window)
        rows = [row for row in self.world.follows if start <= row[3] < end]
        yield {
            'src': np.asarray([row[0] for row in rows], dtype=np.uint64),
            'dst': np.asarray([row[1] for row in rows], dtype=np.uint64),
            'uri_hash': np.asarray([row[2] for row in rows], dtype=np.uint64),
            'event_ms': np.asarray([row[3] for row in rows], dtype=np.int64),
        }

    def stream_repost_creates(self, window: Window) -> Iterator[dict[str, np.ndarray]]:
        yield {
            'src': np.asarray([row[0] for row in self.world.reposts], dtype=np.uint64),
            'dst': np.asarray([row[1] for row in self.world.reposts], dtype=np.uint64),
        }

    def fetch_deletes(self, window: Window, collection: str) -> DeletedRecords:
        if collection not in (FOLLOW_COLLECTION, BLOCK_COLLECTION):
            return DeletedRecords(np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64))
        rows = self.world.deletes if collection == FOLLOW_COLLECTION else self.world.block_deletes
        return fold_deletes(
            np.asarray([row[0] for row in rows], dtype=np.uint64),
            np.asarray([row[1] for row in rows], dtype=np.int64),
        )

    def fetch_account_hashes(self, window: Window) -> pd.DataFrame:
        return pd.DataFrame(self.world.accounts, columns=['h', 'did', 'dids'])

    def stream_interactions(self, window: Window) -> Iterator[pd.DataFrame]:
        start, end = _bounds(window)
        frame = self.world.interactions
        if len(frame):
            frame = frame.loc[(frame['event_ms'] >= start) & (frame['event_ms'] < end)]
        yield frame.reset_index(drop=True)

    def fetch_daily_row_counts(self, window: Window) -> dict[date, int]:
        return dict(self.world.daily_rows)

    # The classifier inputs are not part of the pipeline tests. A refresh would need them.

    def fetch_label_inputs(self, window: Window) -> LabelInput:
        raise NotImplementedError('the pipeline tests never refresh the model')

    def fetch_interaction_texts(self, window: Window, uri_hashes: Sequence[int]) -> pd.DataFrame:
        raise NotImplementedError('the pipeline tests never refresh the model')

    def fetch_parent_texts(self, window: Window, parent_uris: Sequence[str]) -> dict[str, str]:
        raise NotImplementedError('the pipeline tests never refresh the model')

    def fetch_jev_labels(self, prompt_version: str) -> pd.DataFrame:
        raise NotImplementedError('the pipeline tests never refresh the model')

    def _members_of(self, run_id: str) -> pd.DataFrame:
        frame = self.run_owned[schema.MEMBERS].get(run_id)
        if frame is None:
            return pd.DataFrame(columns=list(MEMBER_COLUMNS))
        return frame

    def fetch_tags(self, run_id: str, window: Window) -> pd.DataFrame:
        return self._lists(run_id, self.world.tags)

    def fetch_languages(self, run_id: str, window: Window) -> pd.DataFrame:
        return self._lists(run_id, self.world.languages)

    def _lists(self, run_id: str, material: dict[str, list[str]]) -> pd.DataFrame:
        counts: dict[tuple[int, str], int] = defaultdict(int)
        for did, faction in zip(self._members_of(run_id)['did'], self._members_of(run_id)['faction_id'], strict=True):
            for value in material.get(did, []):
                counts[(int(faction), value)] += 1
        rows = sorted(counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1]))
        return pd.DataFrame(
            [(faction, value, count) for (faction, value), count in rows], columns=['faction_id', 'tag', 'n']
        )

    def fetch_handles(self, run_id: str, window: Window) -> pd.DataFrame:
        members = self._members_of(run_id)
        return pd.DataFrame(
            [
                (int(faction), did, self.world.handles.get(did, ''))
                for did, faction in zip(members['did'], members['faction_id'], strict=True)
            ],
            columns=['faction_id', 'did', 'handle'],
        )

    # writes

    def insert_frame(self, noun: str, frame: pd.DataFrame) -> None:
        if noun in self.run_owned:
            # A table receives several writes from one run. Like ClickHouse, this appends them.
            for run_id, part in frame.groupby('run_id'):
                held = self.run_owned[noun]
                before = held.get(str(run_id))
                held[str(run_id)] = (
                    part.reset_index(drop=True) if before is None else pd.concat([before, part], ignore_index=True)
                )
            return
        self.cache_frames[noun].append(frame)

    def fetch_text_scores(self, model_version: str) -> tuple[np.ndarray, np.ndarray]:
        frames = [frame.loc[frame['model_version'] == model_version] for frame in self.cache_frames[schema.TEXT_SCORES]]
        if not frames:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64)
        all_rows = pd.concat(frames).drop_duplicates('uri_hash', keep='last').sort_values('uri_hash')
        return (
            all_rows['uri_hash'].to_numpy(dtype=np.uint64),
            all_rows['h'].to_numpy(dtype=np.float64),
        )

    def drop_partition(self, noun: str, run_id: str) -> None:
        self.dropped.append((noun, run_id))
        self.run_owned[noun].pop(run_id, None)

    def delete_rows(self, noun: str, run_id: str) -> None:
        self.deleted.append((noun, run_id))
        frame = self.run_owned[noun].get(run_id)
        if frame is not None:
            self.run_owned[noun].pop(run_id, None)

    # test controls

    def hold(self, run_id: str) -> None:
        """Keep the rows inserted from now on invisible until ``release``."""
        self.held[run_id] = []

    def release(self, run_id: str) -> None:
        self.run_rows[run_id].extend(self.held.pop(run_id, []))

    def statuses(self, run_id: str) -> list[str]:
        return [row.status for row in self.fetch_run_rows(run_id)]

    def promoted_run_ids(self) -> list[str]:
        return sorted(run_id for run_id, state in self.fetch_run_states().items() if state.promoted)

    def run_ids(self) -> list[str]:
        return sorted(self.run_rows)


def _bounds(window: Window) -> tuple[int, int]:
    return int(window.start.timestamp() * 1000), int(window.end.timestamp() * 1000)

# pattern: Functional Core
"""Run identity and run-state resolution.

A run is one attempt to compute and publish a faction image. Its state is resolved from the rows it has in
the ``runs`` table, so a reader never depends on a single row being the last write.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from faction_analysis.config import RunKind
from faction_analysis.schema import RUNS_COLUMNS

RunStatus = Literal['running', 'promoting', 'promoted', 'failed']
RunStateName = Literal['running', 'unresolved', 'promoted', 'failed']

_RUN_ID_PATTERN = re.compile(r'^\d{13}\.[0-9a-f]{8}$')
_EPOCH_MS_DIGITS = 13


def new_run_id(now_ms: int, random_suffix: str) -> str:
    """``<13-digit epoch milliseconds>.<8 hex characters>``. Lexical order is start order.

    ``random_suffix`` must come from a random source with at least 32 bits. Two attempts in the same
    millisecond get different IDs with overwhelming probability.
    """
    if isinstance(now_ms, bool) or not isinstance(now_ms, int) or not 0 <= now_ms < 10**_EPOCH_MS_DIGITS:
        raise ValueError(f'now_ms must be epoch milliseconds in [0, 10^{_EPOCH_MS_DIGITS}): {now_ms!r}')
    if not re.fullmatch(r'[0-9a-f]{8}', random_suffix):
        raise ValueError(f'random_suffix must be 8 lowercase hex characters: {random_suffix!r}')
    return f'{now_ms:0{_EPOCH_MS_DIGITS}d}.{random_suffix}'


def is_valid_run_id(value: str) -> bool:
    return bool(_RUN_ID_PATTERN.fullmatch(value))


def resolve_run_state(promoted: bool, failed: bool, promoting: bool, running: bool) -> RunStateName:
    """Promoted, else failed, else unresolved, else running.

    A ``promoted`` row always wins, so a late duplicate of an earlier status can never undo a promotion.
    """
    if promoted:
        return 'promoted'
    if failed:
        return 'failed'
    if promoting:
        return 'unresolved'
    return 'running'


@dataclass(frozen=True)
class RunRow:
    """One row of the ``runs`` table. A run writes the same row under each status it reaches."""

    run_id: str
    status: RunStatus
    kind: RunKind
    recorded_at: datetime
    started_at: datetime
    window_start: datetime
    window_end: datetime
    window_days: int
    objective: str
    resolution: float
    seed: int
    next_faction_id: int
    accounts_placed: int
    accounts_unplaced: int
    hash_collisions: int
    factions: int
    moves: int
    births: int
    merges: int
    splits: int
    deaths: int
    agreement_prev: float | None
    modularity: float | None
    source_gap_days: int
    text_model_version: str
    text_coverage: float
    stage_seconds: Mapping[str, float]
    failed_stage: str
    error_class: str

    def __post_init__(self) -> None:
        if not is_valid_run_id(self.run_id):
            raise ValueError(f'invalid run_id: {self.run_id!r}')

    def with_status(self, status: RunStatus, recorded_at: datetime) -> RunRow:
        return replace(self, status=status, recorded_at=recorded_at)

    def with_failure(self, stage: str, error_class: str, recorded_at: datetime) -> RunRow:
        return replace(self, status='failed', failed_stage=stage, error_class=error_class, recorded_at=recorded_at)


@dataclass(frozen=True)
class RunState:
    run_id: str
    promoted: bool
    promoting: bool
    failed: bool
    running: bool
    last_recorded_at: datetime | None = None

    @property
    def name(self) -> RunStateName:
        return resolve_run_state(self.promoted, self.failed, self.promoting, self.running)


def _status(value: object) -> RunStatus:
    if value == 'running':
        return 'running'
    if value == 'promoting':
        return 'promoting'
    if value == 'promoted':
        return 'promoted'
    if value == 'failed':
        return 'failed'
    raise ValueError(f'unknown run status: {value!r}')


def _kind(value: object) -> RunKind:
    if value == 'weekly':
        return 'weekly'
    if value == 'full':
        return 'full'
    raise ValueError(f'unknown run kind: {value!r}')


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f'expected an integer: {value!r}')
    return int(value)


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f'expected a number: {value!r}')
    return float(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else _float(value)


def _text(value: object) -> str:
    return '' if value is None else str(value)


def run_row_from_columns(row: Sequence[object]) -> RunRow:
    """Build a ``RunRow`` from a result row in ``RUNS_COLUMNS`` order."""
    values = dict(zip(RUNS_COLUMNS, row, strict=True))
    stage_seconds = values['stage_seconds']
    return RunRow(
        run_id=str(values['run_id']),
        status=_status(values['status']),
        kind=_kind(values['kind']),
        recorded_at=_utc(values['recorded_at']),
        started_at=_utc(values['started_at']),
        window_start=_utc(values['window_start']),
        window_end=_utc(values['window_end']),
        window_days=_int(values['window_days']),
        objective=str(values['objective']),
        resolution=_float(values['resolution']),
        seed=_int(values['seed']),
        next_faction_id=_int(values['next_faction_id']),
        accounts_placed=_int(values['accounts_placed']),
        accounts_unplaced=_int(values['accounts_unplaced']),
        hash_collisions=_int(values['hash_collisions']),
        factions=_int(values['factions']),
        moves=_int(values['moves']),
        births=_int(values['births']),
        merges=_int(values['merges']),
        splits=_int(values['splits']),
        deaths=_int(values['deaths']),
        agreement_prev=_optional_float(values['agreement_prev']),
        modularity=_optional_float(values['modularity']),
        source_gap_days=_int(values['source_gap_days']),
        text_model_version=_text(values['text_model_version']),
        text_coverage=_float(values['text_coverage']),
        stage_seconds=(
            {str(key): _float(item) for key, item in stage_seconds.items()}
            if isinstance(stage_seconds, Mapping)
            else {}
        ),
        failed_stage=_text(values['failed_stage']),
        error_class=_text(values['error_class']),
    )


def run_row_to_columns(row: RunRow) -> list[object]:
    """Serialise a ``RunRow`` in ``RUNS_COLUMNS`` order, for an insert."""
    return [
        row.run_id,
        row.status,
        row.kind,
        row.recorded_at,
        row.started_at,
        row.window_start,
        row.window_end,
        int(row.window_days),
        row.objective,
        float(row.resolution),
        int(row.seed),
        int(row.next_faction_id),
        int(row.accounts_placed),
        int(row.accounts_unplaced),
        int(row.hash_collisions),
        int(row.factions),
        int(row.moves),
        int(row.births),
        int(row.merges),
        int(row.splits),
        int(row.deaths),
        row.agreement_prev,
        row.modularity,
        int(row.source_gap_days),
        row.text_model_version,
        float(row.text_coverage),
        {str(key): float(value) for key, value in row.stage_seconds.items()},
        row.failed_stage,
        row.error_class,
    ]


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f'expected a datetime: {value!r}')
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

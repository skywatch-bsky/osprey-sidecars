# pattern: Functional Core
"""Publication safety: what to do with a run, given the intent file and the rows the server holds.

A run is fully written before its promotion is intended. From then on nothing of it is ever removed: a run
with recorded intent is completed forward by re-sending its rows, whatever a re-read shows. A late commit of
an original insert only adds a duplicate row, which readers ignore because they take distinct run IDs.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from faction_analysis.runstate import (
    RunRow,
    RunState,
    RunStatus,
    is_valid_run_id,
    run_row_from_columns,
    run_row_to_columns,
)
from faction_analysis.schema import RUNS_COLUMNS

INTENT_VERSION = 1
STALE_AFTER_MS = 86_400_000
_TIME_COLUMNS = ('recorded_at', 'started_at', 'window_start', 'window_end')


class IntentUnreadable(ValueError):
    """The intent file exists but does not describe a run. It must never be treated as absent."""


@dataclass(frozen=True)
class IntentRecord:
    """The run a promotion-intent file names, with its complete row."""

    run: RunRow


def intent_document(run: RunRow) -> str:
    return json.dumps(
        {'version': INTENT_VERSION, 'run_id': run.run_id, 'row': run_row_to_columns(run)}, default=_to_json
    )


def _to_json(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f'cannot store {type(value).__name__} in the intent file')


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise IntentUnreadable(f'a time column of the intent row is not a string: {value!r}')
    return datetime.fromisoformat(value)


def _columns_from_json(row: list[object]) -> list[object]:
    return [_time(value) if name in _TIME_COLUMNS else value for name, value in zip(RUNS_COLUMNS, row, strict=True)]


def parse_intent(text: str) -> IntentRecord:
    """Parse an intent file. Any defect raises ``IntentUnreadable``."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IntentUnreadable(f'intent file is not JSON: {exc.msg}') from exc
    if not isinstance(document, dict) or document.get('version') != INTENT_VERSION:
        raise IntentUnreadable('intent file has an unknown version')
    row = document.get('row')
    run_id = document.get('run_id')
    if not isinstance(row, list) or not isinstance(run_id, str) or not is_valid_run_id(run_id):
        raise IntentUnreadable('intent file has no run')
    try:
        run = run_row_from_columns(_columns_from_json(row))
    except (ValueError, TypeError) as exc:
        raise IntentUnreadable(f'intent file row is not a run row: {exc}') from exc
    if run.run_id != run_id:
        raise IntentUnreadable('intent file names one run and holds the row of another')
    if run.status != 'promoting':
        raise IntentUnreadable('intent file must hold the promoting row')
    return IntentRecord(run)


@dataclass(frozen=True)
class Recovery:
    """What the shell does with a run that failed before its promotion was confirmed.

    ``resend`` lists the rows to send again, in order. ``mark_failed`` inserts a failed row and ``cleanup``
    removes the rows of the run from the run-owned tables. ``stop`` exits with an error after acting, so the
    next start tries again.
    """

    resend: tuple[RunStatus, ...] = ()
    mark_failed: bool = False
    cleanup: bool = False
    stop: bool = False


def recover(run: RunRow, intent: IntentRecord | None, state: RunState | None) -> Recovery:
    """Decide what to do with ``run`` after a failure.

    ``state`` is what a re-read of the ``runs`` table returned for this run, or ``None`` when that read itself
    failed. The intent file comes first, because it is under the sidecar's control and an insert is not.
    """
    if intent is not None and intent.run.run_id == run.run_id:
        return Recovery(resend=('promoting', 'promoted'))
    if state is None:
        return Recovery(stop=True)
    if state.name in {'promoted', 'failed'}:
        return Recovery()
    if state.name == 'unresolved':
        return Recovery(resend=('promoted',))
    return Recovery(mark_failed=True, cleanup=True)


def run_id_ms(run_id: str) -> int:
    """The epoch milliseconds a run ID starts with."""
    if not is_valid_run_id(run_id):
        raise ValueError(f'invalid run_id: {run_id!r}')
    return int(run_id.split('.', 1)[0])


@dataclass(frozen=True)
class StartupPlan:
    """What the shell does before it may start a new run.

    ``complete_run_id`` names a run to complete by re-sending its promoting and promoted rows. New runs wait
    until its promoted row is confirmed. ``stale_run_ids`` name runs older than a day that nobody will
    complete: they are marked failed and their rows are removed.
    """

    complete_run_id: str | None = None
    stale_run_ids: tuple[str, ...] = ()


def startup_plan(
    intent_run_id: str | None,
    states: Mapping[str, RunState],
    now_ms: int,
    *,
    stale_after_ms: int = STALE_AFTER_MS,
) -> StartupPlan:
    """Plan the start of a poll from the intent file and the run states."""
    unresolved = sorted(
        run_id for run_id, state in states.items() if state.name == 'unresolved' and run_id != intent_run_id
    )
    complete = intent_run_id or (unresolved[0] if unresolved else None)
    stale = tuple(
        sorted(
            run_id
            for run_id, state in states.items()
            if run_id != complete
            and state.name in {'running', 'failed'}
            and run_id_ms(run_id) < now_ms - stale_after_ms
        )
    )
    return StartupPlan(complete_run_id=complete, stale_run_ids=stale)

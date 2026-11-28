# pattern: Functional Core
"""Schedule of the two kinds of run.

The sidecar polls. It starts a monthly cold start when there is none yet or the last one is old enough.
Otherwise it starts a weekly warm start when the latest promoted run is old enough. A failed run leaves no
promoted row, so the next poll starts a run again.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

Decision = Literal['full', 'weekly', 'none']


@dataclass(frozen=True)
class ScheduleState:
    """What the ``runs`` table tells the schedule at the time of a poll.

    ``blocked`` is true while a run waits for its promotion to be confirmed. No new run may start then,
    because its faction IDs would be allocated next to an unfinished one.
    """

    now: datetime
    latest_promoted_at: datetime | None
    latest_full_at: datetime | None
    blocked: bool = False


def decide(state: ScheduleState, *, weekly_after_days: int, monthly_after_days: int) -> Decision:
    """The kind of run to start, or ``none``."""
    if weekly_after_days < 1 or monthly_after_days < weekly_after_days:
        raise ValueError('schedule limits must be positive, and the monthly one at least the weekly one')
    if state.blocked:
        return 'none'
    if state.latest_promoted_at is None:
        return 'full'
    if state.latest_full_at is None or state.now - state.latest_full_at >= timedelta(days=monthly_after_days):
        return 'full'
    if state.now - state.latest_promoted_at >= timedelta(days=weekly_after_days):
        return 'weekly'
    return 'none'

from datetime import UTC, datetime, timedelta

import pytest

from faction_analysis.schedule import ScheduleState, decide

NOW = datetime(2026, 11, 28, 12, tzinfo=UTC)
LIMITS = {'weekly_after_days': 7, 'monthly_after_days': 28}


def state(promoted_age: timedelta | None, full_age: timedelta | None, blocked: bool = False) -> ScheduleState:
    return ScheduleState(
        now=NOW,
        latest_promoted_at=None if promoted_age is None else NOW - promoted_age,
        latest_full_at=None if full_age is None else NOW - full_age,
        blocked=blocked,
    )


def test_no_run_yet_starts_a_monthly_cold_start() -> None:
    assert decide(state(None, None), **LIMITS) == 'full'


def test_one_day_short_of_a_month_starts_nothing() -> None:
    assert decide(state(timedelta(days=6, hours=23), timedelta(days=27, hours=23)), **LIMITS) == 'none'


def test_exactly_a_month_starts_a_monthly_cold_start() -> None:
    assert decide(state(timedelta(days=6, hours=23), timedelta(days=28)), **LIMITS) == 'full'


def test_a_monthly_run_is_due_even_when_a_weekly_one_would_be_too() -> None:
    assert decide(state(timedelta(days=28), timedelta(days=28)), **LIMITS) == 'full'


def test_one_hour_short_of_a_week_starts_nothing() -> None:
    assert decide(state(timedelta(days=6, hours=23), timedelta(days=20)), **LIMITS) == 'none'


def test_exactly_a_week_starts_a_weekly_warm_start() -> None:
    assert decide(state(timedelta(days=7), timedelta(days=21)), **LIMITS) == 'weekly'


def test_a_failed_run_is_retried_on_the_next_poll() -> None:
    # A failed run leaves no promoted row, so the latest promoted run is still the one from eight days ago.
    assert decide(state(timedelta(days=8), timedelta(days=15)), **LIMITS) == 'weekly'


def test_no_run_starts_while_one_waits_for_its_promotion() -> None:
    assert decide(state(timedelta(days=8), timedelta(days=35), blocked=True), **LIMITS) == 'none'


def test_limits_must_make_sense() -> None:
    with pytest.raises(ValueError):
        decide(state(None, None), weekly_after_days=0, monthly_after_days=28)
    with pytest.raises(ValueError):
        decide(state(None, None), weekly_after_days=30, monthly_after_days=28)

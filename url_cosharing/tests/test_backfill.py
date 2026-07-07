# pattern: Functional Core
from datetime import date

import pytest

from url_cosharing.backfill import parse_date_range, run_backfill

TODAY = date(2026, 7, 7)


class TestParseDateRange:
    def test_single_day_defaults_end_to_start(self) -> None:
        assert parse_date_range(['2026-06-25'], TODAY) == (date(2026, 6, 25), date(2026, 6, 25))

    def test_explicit_range(self) -> None:
        assert parse_date_range(['2026-06-25', '2026-06-27'], TODAY) == (
            date(2026, 6, 25),
            date(2026, 6, 27),
        )

    def test_today_allowed(self) -> None:
        assert parse_date_range(['2026-07-07'], TODAY) == (TODAY, TODAY)

    def test_no_args_rejected(self) -> None:
        with pytest.raises(ValueError, match='usage'):
            parse_date_range([], TODAY)

    def test_too_many_args_rejected(self) -> None:
        with pytest.raises(ValueError, match='usage'):
            parse_date_range(['2026-06-25', '2026-06-26', '2026-06-27'], TODAY)

    def test_invalid_date_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_date_range(['2026-13-40'], TODAY)

    def test_reversed_range_rejected(self) -> None:
        with pytest.raises(ValueError, match='after'):
            parse_date_range(['2026-06-27', '2026-06-25'], TODAY)

    def test_future_end_rejected(self) -> None:
        with pytest.raises(ValueError, match='future'):
            parse_date_range(['2026-07-08'], TODAY)


class TestRunBackfill:
    def test_runs_each_day_oldest_to_newest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Evolution tracking depends on chronological order across the range."""
        calls: list[date] = []
        monkeypatch.setattr(
            'url_cosharing.backfill.run_cycle',
            lambda db, config, run_date, telemetry=None: calls.append(run_date),
        )

        run_backfill(db=object(), config=object(), start=date(2026, 6, 25), end=date(2026, 6, 28))

        assert calls == [
            date(2026, 6, 25),
            date(2026, 6, 26),
            date(2026, 6, 27),
            date(2026, 6, 28),
        ]

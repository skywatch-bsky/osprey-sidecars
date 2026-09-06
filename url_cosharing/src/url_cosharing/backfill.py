# pattern: Imperative Shell
"""Recompute url_cosharing results for a historical date range.

Usage: uv run python -m url_cosharing.backfill START_DATE [END_DATE]

Dates are ISO (YYYY-MM-DD) run_dates; END_DATE is inclusive and defaults to
START_DATE. Days are recomputed oldest to newest so evolution tracking chains
across the backfilled range. Existing rows for each run_date are overwritten
(the daemon's idempotent delete-then-insert), including results produced by
earlier methodology versions.

A run_date's detection window is the window_days days ending the day before
it, so the earliest backfillable day is bounded by source-table retention.
All three output tables persist indefinitely (the membership table's 7-day
TTL was removed 2026-07-07 to keep snapshots for post-hoc analysis).

Errors abort the run rather than skipping a day — a gap would corrupt
evolution tracking for the days after it. Re-running the full range is safe.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta

from url_cosharing.config import AppConfig, Exclusions, load_exclusions
from url_cosharing.db import CosharingDb
from url_cosharing.main import run_cycle
from url_cosharing.telemetry import TelemetryHandles, setup_telemetry

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('url_cosharing.backfill')


def parse_date_range(argv: list[str], today: date) -> tuple[date, date]:
    """Parse and validate [start, end] from CLI args (Functional Core logic
    kept separate so the shell below stays untestable-thin).
    """
    if not 1 <= len(argv) <= 2:
        raise ValueError('usage: python -m url_cosharing.backfill START_DATE [END_DATE]')
    start = date.fromisoformat(argv[0])
    end = date.fromisoformat(argv[1]) if len(argv) == 2 else start
    if start > end:
        raise ValueError(f'start {start} is after end {end}')
    if end > today:
        raise ValueError(f'end {end} is in the future; source data for its window does not exist yet')
    return start, end


def run_backfill(
    db: CosharingDb,
    config: AppConfig,
    start: date,
    end: date,
    telemetry: TelemetryHandles | None = None,
    exclusions: Exclusions = Exclusions.empty(),
) -> None:
    """Recompute the range under ONE exclusions revision, passed through to
    every run_cycle so evolution tracking compares like-for-like history."""
    total = (end - start).days + 1
    for offset in range(total):
        run_date = start + timedelta(days=offset)
        logger.info(f'backfilling {run_date} ({offset + 1}/{total})')
        run_cycle(db, config, run_date=run_date, telemetry=telemetry, exclusions=exclusions)


def main() -> None:
    try:
        start, end = parse_date_range(sys.argv[1:], date.today())
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(2)

    config = AppConfig.from_env()
    # Fail fast on a missing/invalid exclusions file before touching history.
    try:
        exclusions = load_exclusions(config.analysis.exclusions_file) if config.analysis.exclusions_file else (
            Exclusions.empty()
        )
    except (OSError, ValueError) as exc:
        logger.error(f'failed to load exclusions: {exc}')
        sys.exit(2)

    telemetry = setup_telemetry(config.telemetry)
    db = CosharingDb(config.clickhouse)
    try:
        run_backfill(db, config, start, end, telemetry=telemetry, exclusions=exclusions)
    finally:
        db.close()
        telemetry.shutdown()


if __name__ == '__main__':
    main()

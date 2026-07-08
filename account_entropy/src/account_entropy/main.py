# pattern: Imperative Shell
from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timedelta, timezone

from account_entropy.analyzer import score_accounts
from account_entropy.config import AppConfig
from account_entropy.db import AccountEntropyDb
from account_entropy.queries import account_activity_query
from account_entropy.telemetry import (
    TelemetryHandles,
    noop_telemetry,
    record_failure,
    record_run_metrics,
    set_run_attributes,
    setup_telemetry,
    stage_span,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('account_entropy')

_shutdown = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    logger.info(f'received signal {signum}, shutting down after current cycle')
    _shutdown = True


def run_cycle(db: AccountEntropyDb, config: AppConfig, telemetry: TelemetryHandles | None = None) -> None:
    run_timestamp = datetime.now(timezone.utc)
    window_end = run_timestamp
    window_start = run_timestamp - timedelta(days=config.analysis.window_days)
    telemetry = telemetry or noop_telemetry()
    run_start = time.monotonic()

    with telemetry.tracer.start_as_current_span(
        'account_entropy.run_cycle',
        attributes={'window_days': config.analysis.window_days},
        record_exception=False,
        set_status_on_exception=False,
    ) as root_span:
        try:
            logger.info('running account activity query')
            with stage_span(
                telemetry,
                'account_entropy.fetch_account_rows',
                window_days=config.analysis.window_days,
            ):
                query = account_activity_query(config.analysis)
                rows = db.fetch_account_rows(query)
            logger.info(f'fetched {len(rows)} accounts with activity')

            if not rows:
                logger.info('no accounts to score, skipping')
                set_run_attributes(
                    root_span,
                    {
                        'window_days': config.analysis.window_days,
                        'accounts_fetched': 0,
                        'accounts_scored': 0,
                        'bot_like_count': 0,
                        'had_rows': False,
                    },
                )
                record_run_metrics(
                    telemetry,
                    time.monotonic() - run_start,
                    window_days=config.analysis.window_days,
                    accounts_fetched=0,
                    accounts_scored=0,
                    bot_like_count=0,
                    had_rows=False,
                )
                return

            with stage_span(
                telemetry,
                'account_entropy.score_accounts',
                window_days=config.analysis.window_days,
            ):
                results = score_accounts(rows, config.analysis, run_timestamp, window_start, window_end)
            bot_like_count = sum(1 for r in results if r.is_bot_like)
            logger.info(f'scored {len(results)} accounts, {bot_like_count} bot-like')

            with stage_span(
                telemetry,
                'account_entropy.insert_results',
                window_days=config.analysis.window_days,
            ):
                db.insert_results(config.analysis.output_table, results)
            logger.info(f'wrote {len(results)} results to {config.analysis.output_table}')

            set_run_attributes(
                root_span,
                {
                    'window_days': config.analysis.window_days,
                    'accounts_fetched': len(rows),
                    'accounts_scored': len(results),
                    'bot_like_count': bot_like_count,
                    'had_rows': True,
                },
            )
            record_run_metrics(
                telemetry,
                time.monotonic() - run_start,
                window_days=config.analysis.window_days,
                accounts_fetched=len(rows),
                accounts_scored=len(results),
                bot_like_count=bot_like_count,
                had_rows=True,
            )
        except Exception as exc:
            record_failure(telemetry, 'run_cycle', exc)
            root_span.set_attribute('error.type', type(exc).__name__)
            raise


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = AppConfig.from_env()
    telemetry = setup_telemetry(config.telemetry)
    logger.info(f'starting account-entropy detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'window_days={config.analysis.window_days}, '
        f'hourly_entropy_norm_threshold={config.analysis.hourly_entropy_norm_threshold}, '
        f'interval_entropy_norm_threshold={config.analysis.interval_entropy_norm_threshold}, '
        f'cv_threshold={config.analysis.cv_threshold}'
    )

    db = AccountEntropyDb(config.clickhouse)

    try:
        while not _shutdown:
            try:
                run_cycle(db, config, telemetry)
            except Exception:
                logger.exception('error during analysis cycle')

            if not _shutdown:
                logger.info(f'sleeping {config.analysis.interval_seconds}s until next cycle')
                time.sleep(config.analysis.interval_seconds)
    finally:
        db.close()
        telemetry.shutdown()
        logger.info('shutdown complete')


if __name__ == '__main__':
    main()

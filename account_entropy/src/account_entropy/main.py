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


def run_cycle(db: AccountEntropyDb, config: AppConfig) -> None:
    run_timestamp = datetime.now(timezone.utc)
    window_end = run_timestamp
    window_start = run_timestamp - timedelta(days=config.analysis.window_days)

    logger.info('running account activity query')
    query = account_activity_query(config.analysis)
    rows = db.fetch_account_rows(query)
    logger.info(f'fetched {len(rows)} accounts with activity')

    if not rows:
        logger.info('no accounts to score, skipping')
        return

    results = score_accounts(rows, config.analysis, run_timestamp, window_start, window_end)
    bot_like_count = sum(1 for r in results if r.is_bot_like)
    logger.info(f'scored {len(results)} accounts, {bot_like_count} bot-like')

    db.insert_results(config.analysis.output_table, results)
    logger.info(f'wrote {len(results)} results to {config.analysis.output_table}')


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = AppConfig.from_env()
    logger.info(f'starting account-entropy detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'window_days={config.analysis.window_days}, hourly_entropy_threshold={config.analysis.hourly_entropy_threshold}, interval_entropy_threshold={config.analysis.interval_entropy_threshold}'
    )

    db = AccountEntropyDb(config.clickhouse)

    try:
        while not _shutdown:
            try:
                run_cycle(db, config)
            except Exception:
                logger.exception('error during analysis cycle')

            if not _shutdown:
                logger.info(f'sleeping {config.analysis.interval_seconds}s until next cycle')
                time.sleep(config.analysis.interval_seconds)
    finally:
        db.close()
        logger.info('shutdown complete')


if __name__ == '__main__':
    main()

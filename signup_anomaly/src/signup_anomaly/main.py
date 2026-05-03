# pattern: Imperative Shell
from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone

from signup_anomaly.analyzer import score_rows
from signup_anomaly.config import AppConfig
from signup_anomaly.db import SignupAnomalyDb
from signup_anomaly.queries import (
    daily_aggregation_query,
    hourly_aggregation_query,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('signup_anomaly')

_shutdown = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    logger.info(f'received signal {signum}, shutting down after current cycle')
    _shutdown = True


def run_cycle(db: SignupAnomalyDb, config: AppConfig) -> None:
    run_timestamp = datetime.now(timezone.utc)

    for granularity, query_fn in [
        ('daily', daily_aggregation_query),
        ('hourly', hourly_aggregation_query),
    ]:
        logger.info(f'running {granularity} aggregation query')
        query = query_fn(config.analysis)
        rows = db.fetch_aggregated_rows(query)
        logger.info(f'{granularity}: fetched {len(rows)} aggregated rows')

        if not rows:
            logger.info(f'{granularity}: no rows to score, skipping')
            continue

        results = score_rows(rows, config.analysis, granularity, run_timestamp)
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(f'{granularity}: scored {len(results)} rows, {anomaly_count} anomalies')

        db.insert_results(config.analysis.output_table, results)
        logger.info(f'{granularity}: wrote {len(results)} results to {config.analysis.output_table}')


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = AppConfig.from_env()
    logger.info(f'starting signup-anomaly detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'daily threshold={config.analysis.daily_p_value_threshold}, hourly threshold={config.analysis.hourly_p_value_threshold}'
    )

    db = SignupAnomalyDb(config.clickhouse)

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

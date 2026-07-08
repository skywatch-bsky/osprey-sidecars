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
from signup_anomaly.telemetry import (
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
logger = logging.getLogger('signup_anomaly')

_shutdown = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    logger.info(f'received signal {signum}, shutting down after current cycle')
    _shutdown = True


def run_cycle(db: SignupAnomalyDb, config: AppConfig, telemetry: TelemetryHandles | None = None) -> None:
    run_timestamp = datetime.now(timezone.utc)
    telemetry = telemetry or noop_telemetry()

    with telemetry.tracer.start_as_current_span(
        'signup_anomaly.run_cycle',
        record_exception=False,
        set_status_on_exception=False,
    ) as root_span:
        try:
            for granularity, query_fn in [
                ('daily', daily_aggregation_query),
                ('hourly', hourly_aggregation_query),
            ]:
                granularity_start = time.monotonic()
                with stage_span(telemetry, f'signup_anomaly.{granularity}.cycle', granularity=granularity):
                    logger.info(f'running {granularity} aggregation query')
                    with stage_span(
                        telemetry,
                        'signup_anomaly.fetch_aggregated_rows',
                        granularity=granularity,
                    ):
                        query = query_fn(config.analysis)
                        rows = db.fetch_aggregated_rows(query)
                    logger.info(f'{granularity}: fetched {len(rows)} aggregated rows')

                    if not rows:
                        logger.info(f'{granularity}: no rows to score, skipping')
                        set_run_attributes(
                            root_span,
                            {
                                'granularity': granularity,
                                'rows_fetched': 0,
                                'rows_scored': 0,
                                'results_inserted': 0,
                                'anomaly_count': 0,
                                'had_rows': False,
                            },
                        )
                        record_run_metrics(
                            telemetry,
                            time.monotonic() - granularity_start,
                            granularity=granularity,
                            rows_fetched=0,
                            rows_scored=0,
                            results_inserted=0,
                            anomaly_count=0,
                            had_rows=False,
                        )
                        continue

                    with stage_span(telemetry, 'signup_anomaly.score_rows', granularity=granularity):
                        results = score_rows(rows, config.analysis, granularity, run_timestamp)
                    anomaly_count = sum(1 for r in results if r.is_anomaly)
                    logger.info(f'{granularity}: scored {len(results)} rows, {anomaly_count} anomalies')

                    with stage_span(telemetry, 'signup_anomaly.insert_results', granularity=granularity):
                        db.insert_results(config.analysis.output_table, results)
                    logger.info(f'{granularity}: wrote {len(results)} results to {config.analysis.output_table}')

                    set_run_attributes(
                        root_span,
                        {
                            'granularity': granularity,
                            'rows_fetched': len(rows),
                            'rows_scored': len(results),
                            'results_inserted': len(results),
                            'anomaly_count': anomaly_count,
                            'had_rows': True,
                        },
                    )
                    record_run_metrics(
                        telemetry,
                        time.monotonic() - granularity_start,
                        granularity=granularity,
                        rows_fetched=len(rows),
                        rows_scored=len(results),
                        results_inserted=len(results),
                        anomaly_count=anomaly_count,
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
    logger.info(f'starting signup-anomaly detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'daily threshold={config.analysis.daily_p_value_threshold}, hourly threshold={config.analysis.hourly_p_value_threshold}'
    )

    db = SignupAnomalyDb(config.clickhouse)

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

# pattern: Imperative Shell
from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone

from url_overdispersion.analyzer import score_rows
from url_overdispersion.config import AppConfig
from url_overdispersion.db import UrlOverdispersionDb
from url_overdispersion.queries import (
    daily_aggregation_query,
    hourly_aggregation_query,
)
from url_overdispersion.telemetry import (
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
logger = logging.getLogger('url_overdispersion')

_shutdown = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    logger.info(f'received signal {signum}, shutting down after current cycle')
    _shutdown = True


def run_cycle(db: UrlOverdispersionDb, config: AppConfig, telemetry: TelemetryHandles | None = None) -> None:
    run_timestamp = datetime.now(timezone.utc)
    telemetry = telemetry or noop_telemetry()

    with telemetry.tracer.start_as_current_span(
        'url_overdispersion.run_cycle',
        record_exception=False,
        set_status_on_exception=False,
    ) as root_span:
        try:
            totals = {
                'rows_fetched': 0,
                'rows_scored': 0,
                'results_inserted': 0,
                'anomaly_count': 0,
                'had_rows': False,
            }
            for granularity, query_fn in [
                ('daily', daily_aggregation_query),
                ('hourly', hourly_aggregation_query),
            ]:
                granularity_start = time.monotonic()
                with stage_span(telemetry, f'url_overdispersion.{granularity}.cycle', granularity=granularity):
                    logger.info(f'running {granularity} aggregation query')
                    with stage_span(
                        telemetry,
                        'url_overdispersion.fetch_aggregated_rows',
                        granularity=granularity,
                    ):
                        query = query_fn(config.analysis)
                        rows = db.fetch_aggregated_rows(query)
                    logger.info(f'{granularity}: fetched {len(rows)} aggregated rows')

                    if not rows:
                        logger.info(f'{granularity}: no rows to score, skipping')
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

                    with stage_span(telemetry, 'url_overdispersion.score_rows', granularity=granularity):
                        results = score_rows(
                            rows, config.analysis, granularity, run_timestamp, config.analysis.watchlist_domains
                        )
                    anomaly_count = sum(1 for r in results if r.is_anomaly)
                    logger.info(f'{granularity}: scored {len(results)} rows, {anomaly_count} anomalies')

                    with stage_span(telemetry, 'url_overdispersion.insert_results', granularity=granularity):
                        db.insert_results(config.analysis.output_table, results)
                    logger.info(f'{granularity}: wrote {len(results)} results to {config.analysis.output_table}')

                    totals['rows_fetched'] += len(rows)
                    totals['rows_scored'] += len(results)
                    totals['results_inserted'] += len(results)
                    totals['anomaly_count'] += anomaly_count
                    totals['had_rows'] = True
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
            set_run_attributes(root_span, totals)
        except Exception as exc:
            record_failure(telemetry, 'run_cycle', exc)
            root_span.set_attribute('error.type', type(exc).__name__)
            raise


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = AppConfig.from_env()
    telemetry = setup_telemetry(config.telemetry)
    logger.info(f'starting url-overdispersion detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'volume threshold={config.analysis.volume_p_threshold}, density threshold={config.analysis.density_p_threshold}'
    )

    db = UrlOverdispersionDb(config.clickhouse)

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

# pattern: Imperative Shell
from __future__ import annotations

import logging
import re
import signal
import time
from datetime import date, datetime

from quote_cosharing.analyzer import (
    build_graph,
    cluster_graph,
    compute_evolution,
    compute_temporal_metrics,
)
from quote_cosharing.config import AppConfig
from quote_cosharing.db import QuoteCosharingDb
from quote_cosharing.queries import (
    fetch_historical_membership_query,
    fetch_member_timestamps_query,
    fetch_pairs_query,
)
from quote_cosharing.telemetry import (
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
logger = logging.getLogger('quote_cosharing')

_shutdown = False


def _sanitize_did(did: str) -> str:
    """Remove non-allowed characters from DID for defence-in-depth SQL safety.

    DIDs should match pattern did:plc:[a-z0-9]+. Keep only these characters.
    """
    return re.sub(r'[^a-z0-9:.]', '', did)


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    logger.info(f'received signal {signum}, shutting down after current cycle')
    _shutdown = True


def run_cycle(db: QuoteCosharingDb, config: AppConfig, telemetry: TelemetryHandles | None = None) -> None:
    run_date = date.today()
    telemetry = telemetry or noop_telemetry()
    run_start = time.monotonic()

    with telemetry.tracer.start_as_current_span(
        'quote_cosharing.run_cycle',
        attributes={'run_date': run_date.isoformat()},
        record_exception=False,
        set_status_on_exception=False,
    ) as root_span:
        try:
            logger.info('gathering pairs')
            with stage_span(telemetry, 'quote_cosharing.fetch_pairs', run_date=run_date.isoformat()):
                query = fetch_pairs_query(config.analysis)
                pairs = db.fetch_pairs(query)
            logger.info(f'fetched {len(pairs)} pairs')

            if not pairs:
                logger.warning('no pairs found for yesterday — check scheduled query')
                set_run_attributes(
                    root_span,
                    {
                        'run_date': run_date.isoformat(),
                        'pairs_fetched': 0,
                        'graph_nodes': 0,
                        'graph_edges': 0,
                        'cluster_count': 0,
                        'membership_rows': 0,
                        'had_pairs': False,
                    },
                )
                record_run_metrics(
                    telemetry,
                    time.monotonic() - run_start,
                    pairs_fetched=0,
                    graph_nodes=0,
                    graph_edges=0,
                    cluster_count=0,
                    membership_rows=0,
                    had_pairs=False,
                )
                return

            logger.info('building graph')
            with stage_span(telemetry, 'quote_cosharing.build_graph', run_date=run_date.isoformat()):
                graph = build_graph(pairs, config.analysis.min_edge_weight)

            logger.info('clustering graph')
            with stage_span(telemetry, 'quote_cosharing.cluster_graph', run_date=run_date.isoformat()):
                clusters = cluster_graph(graph, config.analysis.resolution, config.analysis.min_cluster_size)
            logger.info(f'found {len(clusters)} clusters')

            logger.info('gathering member timestamps')
            all_dids = set()
            for cluster in clusters:
                all_dids.update(cluster.members)

            with stage_span(telemetry, 'quote_cosharing.fetch_member_timestamps', run_date=run_date.isoformat()):
                if all_dids:
                    sanitized_dids = [_sanitize_did(did) for did in sorted(all_dids)]
                    dids_placeholder = ','.join(f"'{did}'" for did in sanitized_dids)
                    timestamps_query = fetch_member_timestamps_query(config.analysis, dids_placeholder)
                    timestamp_rows = db.fetch_member_timestamps(timestamps_query)

                    member_timestamps: dict[str, list[datetime]] = {}
                    for row in timestamp_rows:
                        if row.did not in member_timestamps:
                            member_timestamps[row.did] = []
                        member_timestamps[row.did].append(row.ts)
                else:
                    member_timestamps = {}

            logger.info('computing temporal metrics')
            with stage_span(telemetry, 'quote_cosharing.compute_temporal_metrics', run_date=run_date.isoformat()):
                timestamped_clusters = [compute_temporal_metrics(cluster, member_timestamps) for cluster in clusters]

            logger.info('gathering historical membership')
            with stage_span(telemetry, 'quote_cosharing.fetch_historical_membership', run_date=run_date.isoformat()):
                history_query = fetch_historical_membership_query(config.analysis)
                history_rows = db.fetch_historical_membership(history_query)

            previous_membership: dict[str, frozenset[str]] = {}
            for row in history_rows:
                if row.cluster_id not in previous_membership:
                    previous_membership[row.cluster_id] = frozenset()
                previous_membership[row.cluster_id] = previous_membership[row.cluster_id] | frozenset([row.did])

            logger.info('computing evolution')
            with stage_span(telemetry, 'quote_cosharing.compute_evolution', run_date=run_date.isoformat()):
                events = compute_evolution(
                    clusters,
                    previous_membership,
                    run_date,
                    config.analysis.jaccard_threshold,
                )

            logger.info('clearing stale data for today (idempotent re-run guard)')
            with stage_span(telemetry, 'quote_cosharing.delete_stale_run_date', run_date=run_date.isoformat()):
                db.delete_run_date(config.analysis.clusters_table, run_date)
                db.delete_run_date(config.analysis.membership_table, run_date)

            logger.info('persisting results')
            cluster_rows = [
                (run_date, ts_cluster, event)
                for ts_cluster, event in zip(timestamped_clusters, events)
                if event.evolution_type != 'death'
            ]
            with stage_span(telemetry, 'quote_cosharing.persist_clusters', run_date=run_date.isoformat()):
                db.insert_clusters(config.analysis.clusters_table, cluster_rows)
            logger.info(f'wrote {len(cluster_rows)} cluster results to {config.analysis.clusters_table}')

            membership_rows = []
            for ts_cluster, event in zip(timestamped_clusters, events):
                if event.evolution_type != 'death':
                    for did in ts_cluster.members:
                        membership_rows.append((run_date, event.cluster_id, did))

            with stage_span(telemetry, 'quote_cosharing.persist_membership', run_date=run_date.isoformat()):
                db.insert_membership(config.analysis.membership_table, membership_rows)
            logger.info(f'wrote {len(membership_rows)} membership rows to {config.analysis.membership_table}')

            graph_edges = int(graph.ecount()) if hasattr(graph, 'ecount') else 0
            graph_nodes = int(graph.vcount()) if hasattr(graph, 'vcount') else 0
            set_run_attributes(
                root_span,
                {
                    'run_date': run_date.isoformat(),
                    'pairs_fetched': len(pairs),
                    'graph_nodes': graph_nodes,
                    'graph_edges': graph_edges,
                    'cluster_count': len(cluster_rows),
                    'membership_rows': len(membership_rows),
                    'had_pairs': True,
                },
            )
            record_run_metrics(
                telemetry,
                time.monotonic() - run_start,
                pairs_fetched=len(pairs),
                graph_nodes=graph_nodes,
                graph_edges=graph_edges,
                cluster_count=len(cluster_rows),
                membership_rows=len(membership_rows),
                had_pairs=True,
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
    logger.info(f'starting quote-cosharing detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'resolution={config.analysis.resolution}, min_edge_weight={config.analysis.min_edge_weight}, min_cluster_size={config.analysis.min_cluster_size}, jaccard_threshold={config.analysis.jaccard_threshold}'
    )

    db = QuoteCosharingDb(config.clickhouse)

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

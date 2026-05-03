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


def run_cycle(db: QuoteCosharingDb, config: AppConfig) -> None:
    run_date = date.today()

    logger.info('gathering pairs')
    query = fetch_pairs_query(config.analysis)
    pairs = db.fetch_pairs(query)
    logger.info(f'fetched {len(pairs)} pairs')

    if not pairs:
        logger.warning('no pairs found for yesterday — check scheduled query')
        return

    logger.info('building graph')
    graph = build_graph(pairs, config.analysis.min_edge_weight)

    logger.info('clustering graph')
    clusters = cluster_graph(graph, config.analysis.resolution, config.analysis.min_cluster_size)
    logger.info(f'found {len(clusters)} clusters')

    logger.info('gathering member timestamps')
    all_dids = set()
    for cluster in clusters:
        all_dids.update(cluster.members)

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
    timestamped_clusters = [compute_temporal_metrics(cluster, member_timestamps) for cluster in clusters]

    logger.info('gathering historical membership')
    history_query = fetch_historical_membership_query(config.analysis)
    history_rows = db.fetch_historical_membership(history_query)

    previous_membership: dict[str, frozenset[str]] = {}
    for row in history_rows:
        if row.cluster_id not in previous_membership:
            previous_membership[row.cluster_id] = frozenset()
        previous_membership[row.cluster_id] = previous_membership[row.cluster_id] | frozenset([row.did])

    logger.info('computing evolution')
    events = compute_evolution(
        clusters,
        previous_membership,
        run_date,
        config.analysis.jaccard_threshold,
    )

    logger.info('clearing stale data for today (idempotent re-run guard)')
    db.delete_run_date(config.analysis.clusters_table, run_date)
    db.delete_run_date(config.analysis.membership_table, run_date)

    logger.info('persisting results')
    cluster_rows = [
        (run_date, ts_cluster, event)
        for ts_cluster, event in zip(timestamped_clusters, events)
        if event.evolution_type != 'death'
    ]
    db.insert_clusters(config.analysis.clusters_table, cluster_rows)
    logger.info(f'wrote {len(cluster_rows)} cluster results to {config.analysis.clusters_table}')

    membership_rows = []
    for ts_cluster, event in zip(timestamped_clusters, events):
        if event.evolution_type != 'death':
            for did in ts_cluster.members:
                membership_rows.append((run_date, event.cluster_id, did))

    db.insert_membership(config.analysis.membership_table, membership_rows)
    logger.info(f'wrote {len(membership_rows)} membership rows to {config.analysis.membership_table}')


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = AppConfig.from_env()
    logger.info(f'starting quote-cosharing detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'resolution={config.analysis.resolution}, min_edge_weight={config.analysis.min_edge_weight}, min_cluster_size={config.analysis.min_cluster_size}, jaccard_threshold={config.analysis.jaccard_threshold}'
    )

    db = QuoteCosharingDb(config.clickhouse)

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

# pattern: Imperative Shell
from __future__ import annotations

import logging
import re
import signal
import time
from datetime import date, datetime

from url_cosharing.analyzer import (
    cluster_core,
    compute_evolution,
    compute_temporal_metrics,
)
from url_cosharing.config import AppConfig
from url_cosharing.db import CosharingDb, RunMetadata
from url_cosharing.dismantling import dismantle
from url_cosharing.queries import (
    fetch_historical_membership_query,
    fetch_member_timestamps_query,
    fetch_url_shares_query,
)
from url_cosharing.similarity import similarity_network

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('url_cosharing')

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


def run_cycle(db: CosharingDb, config: AppConfig) -> None:
    run_date = date.today()
    analysis = config.analysis

    logger.info('fetching url shares')
    rows = db.fetch_url_shares(fetch_url_shares_query(analysis))
    logger.info(f'fetched {len(rows)} share rows')

    network = similarity_network(
        rows,
        analysis.min_unique_urls,
        analysis.min_url_sharers,
        analysis.max_url_df_pctl,
        analysis.edge_epsilon,
        logger,
    )
    logger.info(
        f'similarity network: {network.accounts_eligible}/{network.accounts_raw} accounts, '
        f'{network.urls_eligible} urls, {network.graph_edges} edges'
    )

    result = dismantle(
        network.graph,
        analysis.edge_quantile_grid,
        analysis.centrality_quantile_grid,
        analysis.density_floor,
        analysis.max_flagged_fraction,
        analysis.min_cluster_size,
        logger,
    )

    clusters = cluster_core(
        result.core, network.matrix, network.tfidf, analysis.resolution, analysis.min_cluster_size
    )
    logger.info(f'found {len(clusters)} clusters (knee_found={result.knee_found})')

    logger.info('gathering member timestamps')
    all_dids = set()
    for cluster in clusters:
        all_dids.update(cluster.members)

    if all_dids:
        sanitized_dids = [_sanitize_did(did) for did in sorted(all_dids)]
        dids_placeholder = ','.join(f"'{did}'" for did in sanitized_dids)
        timestamps_query = fetch_member_timestamps_query(analysis, dids_placeholder)
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
    history_query = fetch_historical_membership_query(analysis)
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
        analysis.jaccard_threshold,
    )

    logger.info('clearing stale data for today (idempotent re-run guard)')
    db.delete_run_date(analysis.clusters_table, run_date)
    db.delete_run_date(analysis.membership_table, run_date)
    db.delete_run_date(analysis.runs_table, run_date)

    logger.info('persisting results')
    cluster_rows = [
        (run_date, ts_cluster, event)
        for ts_cluster, event in zip(timestamped_clusters, events)
        if event.evolution_type != 'death'
    ]
    db.insert_clusters(analysis.clusters_table, cluster_rows)
    logger.info(f'wrote {len(cluster_rows)} cluster results to {analysis.clusters_table}')

    membership_rows = []
    for ts_cluster, event in zip(timestamped_clusters, events):
        if event.evolution_type != 'death':
            for did in ts_cluster.members:
                membership_rows.append((run_date, event.cluster_id, did))

    db.insert_membership(analysis.membership_table, membership_rows)
    logger.info(f'wrote {len(membership_rows)} membership rows to {analysis.membership_table}')

    db.insert_run(
        analysis.runs_table,
        RunMetadata(
            run_date=run_date,
            window_days=analysis.window_days,
            accounts_raw=network.accounts_raw,
            accounts_eligible=network.accounts_eligible,
            urls_eligible=network.urls_eligible,
            graph_edges=network.graph_edges,
            edge_quantile=result.edge_quantile,
            centrality_quantile=result.centrality_quantile,
            min_component_density=result.min_component_density,
            knee_found=result.knee_found,
            guardrail_triggered=result.guardrail_triggered,
            flagged_accounts=result.core.vcount(),
            cluster_count=len(cluster_rows),
        ),
    )
    logger.info(f'wrote run metadata to {analysis.runs_table}')


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = AppConfig.from_env()
    logger.info(f'starting url-cosharing detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'window_days={config.analysis.window_days}, min_unique_urls={config.analysis.min_unique_urls}, '
        f'min_url_sharers={config.analysis.min_url_sharers}, density_floor={config.analysis.density_floor}, '
        f'max_flagged_fraction={config.analysis.max_flagged_fraction}, resolution={config.analysis.resolution}, '
        f'min_cluster_size={config.analysis.min_cluster_size}, jaccard_threshold={config.analysis.jaccard_threshold}'
    )

    db = CosharingDb(config.clickhouse)

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

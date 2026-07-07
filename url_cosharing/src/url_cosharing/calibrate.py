# pattern: Imperative Shell
"""Dump the density-dismantling grid surface for offline calibration.

Usage: uv run python -m url_cosharing.calibrate
Reads the same URL_COSHARING_* / OSPREY_CLICKHOUSE_* env vars as the sidecar,
fetches the current window from ClickHouse, runs the full similarity +
dismantling pipeline, and prints the per-cell surface as TSV to stdout.
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from url_cosharing.config import AppConfig
from url_cosharing.db import CosharingDb
from url_cosharing.dismantling import DismantlingResult, dismantle
from url_cosharing.queries import fetch_raw_account_count_query, fetch_url_shares_query
from url_cosharing.similarity import SimilarityNetwork, similarity_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('url_cosharing.calibrate')


def format_surface(network: SimilarityNetwork, result: DismantlingResult, accounts_raw: int) -> str:
    """Pure formatter: TSV surface plus a summary footer (Functional Core logic
    kept separate so the shell below stays untestable-thin).
    """
    lines = ['edge_quantile\tcentrality_quantile\tmin_component_density\tsurviving_nodes\tsurviving_edges']
    for cell in result.surface:
        lines.append(
            f'{cell.edge_quantile}\t{cell.centrality_quantile}\t'
            f'{cell.min_component_density:.4f}\t{cell.surviving_nodes}\t{cell.surviving_edges}'
        )
    lines.append('')
    lines.append(
        f'# accounts_raw={accounts_raw} accounts_eligible={network.accounts_eligible} '
        f'urls_eligible={network.urls_eligible} graph_edges={network.graph_edges}'
    )
    lines.append(
        f'# knee_found={result.knee_found} edge_quantile={result.edge_quantile} '
        f'centrality_quantile={result.centrality_quantile} '
        f'min_component_density={result.min_component_density:.4f} '
        f'guardrail_triggered={result.guardrail_triggered} flagged_accounts={result.core.vcount()}'
    )
    return '\n'.join(lines)


def main() -> None:
    config = AppConfig.from_env()
    analysis = config.analysis
    db = CosharingDb(config.clickhouse)
    try:
        as_of = date.today()
        rows = db.fetch_url_shares(fetch_url_shares_query(analysis, as_of))
        logger.info(f'fetched {len(rows)} share rows')
        accounts_raw = db.fetch_raw_account_count(fetch_raw_account_count_query(analysis, as_of))
        network = similarity_network(rows, analysis.edge_epsilon)
        result = dismantle(
            network.graph,
            analysis.edge_quantile_grid,
            analysis.centrality_quantile_grid,
            analysis.density_floor,
            analysis.max_flagged_fraction,
            analysis.min_cluster_size,
            logger,
        )
        sys.stdout.write(format_surface(network, result, accounts_raw) + '\n')
    finally:
        db.close()


if __name__ == '__main__':
    main()

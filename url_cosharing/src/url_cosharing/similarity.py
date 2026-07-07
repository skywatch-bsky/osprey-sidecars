# pattern: Functional Core
from __future__ import annotations

import logging
from dataclasses import dataclass

import igraph as ig
import numpy as np
from scipy.sparse import csr_array, diags_array, triu


@dataclass(frozen=True)
class UrlShareRow:
    did: str
    url: str
    share_count: int


@dataclass(frozen=True)
class ShareMatrix:
    counts: csr_array  # accounts × urls share counts
    accounts: tuple[str, ...]  # row index -> did (sorted)
    urls: tuple[str, ...]  # col index -> url (sorted)


@dataclass(frozen=True)
class SimilarityNetwork:
    graph: ig.Graph
    matrix: ShareMatrix
    tfidf: csr_array
    accounts_raw: int  # distinct accounts in the unfiltered input
    accounts_eligible: int  # distinct accounts after filters
    urls_eligible: int  # distinct urls after filters
    graph_edges: int


def filter_shares(
    rows: list[UrlShareRow],
    min_unique_urls: int,
    min_url_sharers: int,
    max_url_df_fraction: float,
    logger: logging.Logger | None = None,
) -> list[UrlShareRow]:
    """Re-apply the SQL prefilters in Python (defense in depth).

    Semantics match fetch_url_shares_query: activity and df are both computed
    over the raw input rows in a single pass, then applied together.

    The df ceiling is max_url_df_fraction of distinct accounts (sklearn max_df
    semantics, matching Cinus et al. WWW '25), not a percentile of the df
    distribution: URL df distributions are so head-heavy that a low percentile
    can fall below min_url_sharers, silently emptying the eligible set.
    """
    if not rows:
        return []

    urls_by_did: dict[str, set[str]] = {}
    dids_by_url: dict[str, set[str]] = {}
    for row in rows:
        urls_by_did.setdefault(row.did, set()).add(row.url)
        dids_by_url.setdefault(row.url, set()).add(row.did)

    df_ceiling = max_url_df_fraction * len(urls_by_did)

    active_dids = {did for did, urls in urls_by_did.items() if len(urls) >= min_unique_urls}
    eligible_urls = {
        url
        for url, dids in dids_by_url.items()
        if min_url_sharers <= len(dids) <= df_ceiling
    }

    if not eligible_urls and logger:
        logger.warning(
            f'filter_shares: no eligible URLs from {len(dids_by_url)} candidates '
            f'(min_url_sharers={min_url_sharers}, df_ceiling={df_ceiling}); '
            'similarity network will be empty'
        )

    kept = [row for row in rows if row.did in active_dids and row.url in eligible_urls]
    if logger:
        logger.info(
            f'filter_shares: {len(rows)} rows -> {len(kept)} '
            f'(accounts {len(urls_by_did)} -> {len({r.did for r in kept})}, '
            f'urls {len(dids_by_url)} -> {len({r.url for r in kept})}, df_ceiling={df_ceiling})'
        )
    return kept


def build_share_matrix(rows: list[UrlShareRow]) -> ShareMatrix:
    """Build sparse account×url share count matrix from rows."""
    if not rows:
        return ShareMatrix(
            counts=csr_array((0, 0), dtype=np.float64),
            accounts=(),
            urls=(),
        )

    accounts = tuple(sorted({row.did for row in rows}))
    urls = tuple(sorted({row.url for row in rows}))
    did_to_idx = {did: idx for idx, did in enumerate(accounts)}
    url_to_idx = {url: idx for idx, url in enumerate(urls)}

    data = np.array([row.share_count for row in rows], dtype=np.float64)
    row_idx = np.array([did_to_idx[row.did] for row in rows])
    col_idx = np.array([url_to_idx[row.url] for row in rows])

    counts = csr_array((data, (row_idx, col_idx)), shape=(len(accounts), len(urls)))
    return ShareMatrix(counts=counts, accounts=accounts, urls=urls)


def tfidf_transform(counts: csr_array) -> csr_array:
    """tf * ln(N / df), rows L2-normalized. Zero rows stay zero."""
    n_accounts, n_urls = counts.shape
    if n_accounts == 0 or n_urls == 0:
        return counts.copy()

    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(n_accounts / df)
    weighted = counts @ diags_array(idf, format='csr')

    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
    return csr_array(diags_array(inv_norms, format='csr') @ weighted)


def build_similarity_graph(tfidf: csr_array, accounts: tuple[str, ...], edge_epsilon: float) -> ig.Graph:
    """Undirected graph over all accounts; edges are cosine similarities >= edge_epsilon."""
    graph = ig.Graph(n=len(accounts))
    if len(accounts) > 0:
        graph.vs['name'] = list(accounts)
    if tfidf.shape[0] == 0:
        graph.es['similarity'] = []
        return graph

    sims = triu(tfidf @ tfidf.T, k=1).tocoo()
    keep = sims.data >= edge_epsilon
    edges = list(zip(sims.row[keep].tolist(), sims.col[keep].tolist()))
    weights = np.minimum(sims.data[keep], 1.0).tolist()

    graph.add_edges(edges)
    graph.es['similarity'] = weights
    return graph


def similarity_network(
    rows: list[UrlShareRow],
    min_unique_urls: int,
    min_url_sharers: int,
    max_url_df_fraction: float,
    edge_epsilon: float,
    logger: logging.Logger | None = None,
) -> SimilarityNetwork:
    """End-to-end pipeline: filter, build matrix, TF-IDF, compute similarities."""
    accounts_raw = len({row.did for row in rows})
    kept = filter_shares(rows, min_unique_urls, min_url_sharers, max_url_df_fraction, logger)
    matrix = build_share_matrix(kept)
    tfidf = tfidf_transform(matrix.counts)
    graph = build_similarity_graph(tfidf, matrix.accounts, edge_epsilon)
    return SimilarityNetwork(
        graph=graph,
        matrix=matrix,
        tfidf=tfidf,
        accounts_raw=accounts_raw,
        accounts_eligible=len(matrix.accounts),
        urls_eligible=len(matrix.urls),
        graph_edges=graph.ecount(),
    )

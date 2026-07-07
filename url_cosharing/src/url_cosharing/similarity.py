# pattern: Functional Core
from __future__ import annotations

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
    accounts_raw: int  # distinct accounts in the input rows (already SQL-filtered)
    accounts_eligible: int  # distinct accounts in the share matrix
    urls_eligible: int  # distinct urls in the share matrix
    graph_edges: int


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
    edge_epsilon: float,
) -> SimilarityNetwork:
    """End-to-end pipeline: build matrix, TF-IDF, compute similarities.

    Eligibility filtering (activity floor, df floor, df ceiling) is owned
    entirely by fetch_url_shares_query, which computes all filters in a
    single pass over the raw rolling-window population and applies them
    together. Its output can legitimately contain accounts with fewer
    surviving URLs than min_unique_urls and URLs with fewer surviving
    sharers than min_url_sharers, so no filter may be recomputed here:
    re-deriving eligibility from the already-filtered result set uses
    reduced per-account and per-URL counts and erases valid detections.
    """
    accounts_raw = len({row.did for row in rows})
    matrix = build_share_matrix(rows)
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

# URL Co-Sharing Graph Sidecar

Last verified: 2026-07-07

## Purpose

Detects coordinated inauthentic behaviour by identifying clusters of accounts that repeatedly share the same URLs. Builds a similarity network via TF-IDF cosine distance, applies density-based dismantling to isolate dense coordinated cores, runs Leiden CPM on the core with similarity weights, and tracks cluster evolution across days. Reads from `osprey_execution_results` (7-day rolling window), writes cluster results to `url_cosharing_clusters`, membership snapshots to `url_cosharing_membership`, and run metadata to `url_cosharing_runs`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `similarity.py` — build share matrix from URL share rows, TF-IDF transform, construct similarity graph (Core, no I/O)
- `dismantling.py` — density-based dismantling to isolate cores (Core, no I/O)
- `analyzer.py` — Leiden CPM clustering on core with bipartite metrics, Jaccard evolution tracking (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling, orchestration (Shell)

## Detection Pipeline

1. **Fetch:** 7-day rolling window of URL shares per account from `osprey_execution_results`
2. **Similarity:** Build share matrix (accounts × URLs), compute TF-IDF cosine similarity (edge_epsilon for sparsity), filter by min_unique_urls per account and min_url_sharers per URL
3. **Dismantling:** Density-based edge filtering over quantile grid to isolate coordinated cores (density_floor threshold, max_flagged_fraction guardrail)
4. **Core Clustering:** Leiden CPM on surviving core with cosine-similarity edge weights, min_cluster_size filtering
5. **Evolution:** Jaccard matching against prior membership snapshots (jaccard_threshold for continuation/merge/split classification)
6. **Write:** Cluster rows with similarity metrics (mean_edge_similarity, subgraph_density) and evolution, membership snapshots for all members, run metadata (edge/centrality quantiles, density, counts)

## Key Configuration Knobs

- `window_days` — rolling window for historical URL shares (default 7)
- `min_unique_urls`, `min_url_sharers`, `max_url_df_pctl` — similarity network construction filters
- `edge_epsilon` — sparsity threshold for TF-IDF cosine edges
- `edge_quantile_grid`, `centrality_quantile_grid` — dismantling surface points
- `density_floor`, `max_flagged_fraction` — knee-finding thresholds
- `resolution` — Leiden CPM resolution parameter
- `min_cluster_size` — minimum cluster membership
- `jaccard_threshold` — evolution classification threshold

## Metrics

**Cluster `total_weight`:** Σ over cluster URLs of C(k, 2) where k = number of cluster members sharing that URL. Semantics: binomial co-share count (accounts only paired if they share the same URL).

**Cluster `mean_edge_similarity`:** Mean of similarity weights on edges in the cluster subgraph (0 if no edges).

**Cluster `subgraph_density`:** (edges in subgraph) / (possible edges in subgraph), range [0, 1].

**Run metadata:** Captures stage counts (accounts_raw/eligible, urls_eligible, graph_edges), chosen quantiles, whether knee was found, flagged account count, and cluster count written.

## Contract

- **Input:** `osprey_execution_results` (account DIDs, URL shares, dates)
- **Output:** `url_cosharing_clusters` (cluster results, metrics, evolution), `url_cosharing_membership` (daily snapshots, TTL 7 days), `url_cosharing_runs` (run metadata)
- **Dependencies:** ClickHouse only. `similarity.py` and `dismantling.py` have no I/O or ClickHouse imports (pure functional core).

## Commands

- `cd url_cosharing && uv run pytest` — Run tests
- `docker compose up url-cosharing` — Start sidecar

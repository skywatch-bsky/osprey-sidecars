# URL co-sharing graph sidecar

Identifies clusters of accounts that repeatedly share the same URLs using TF-IDF cosine-similarity networks combined with density-based dismantling to isolate high-precision coordinated cores, then applies Leiden community detection.

## How it works

1. Fetch per-account URL share counts from `osprey_execution_results` (7-day rolling window) with activity and document-frequency filters
2. Compute TF-IDF vectors (tf = co-share count, idf = ln(N / df), L2-normalized) and build cosine-similarity network (edges = similarity in [0, 1])
3. Density-based dismantling: grid search over edge/centrality quantile pairs to isolate high-density coordinated cores via knee-detection heuristic and guardrails
4. Leiden CPM decomposes the dismantled core with cosine-similarity edge weights to identify clusters
5. Evolution tracking: Jaccard similarity matching against prior day membership snapshots to classify cluster transitions (birth/continuation/merge/split/death)
6. Write cluster results with similarity metrics to `url_cosharing_clusters`, daily membership snapshots to `url_cosharing_membership`, and run metadata to `url_cosharing_runs`

## Usage

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run locally
uv run python -m url_cosharing.main

# Run via Docker
docker build -t url-cosharing .
docker run --env-file .env url-cosharing
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OSPREY_CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `OSPREY_CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `OSPREY_CLICKHOUSE_USER` | `default` | ClickHouse user |
| `OSPREY_CLICKHOUSE_PASSWORD` | `clickhouse` | ClickHouse password |
| `OSPREY_CLICKHOUSE_DB` | `default` | Database name |
| `URL_COSHARING_WINDOW_DAYS` | `7` | Rolling window for URL share history (days) |
| `URL_COSHARING_MIN_UNIQUE_URLS` | `10` | Minimum unique URLs per account to be included |
| `URL_COSHARING_MIN_URL_SHARERS` | `5` | Minimum accounts sharing a URL to be included |
| `URL_COSHARING_MAX_URL_DF_FRACTION` | `0.90` | Exclude URLs shared by more than this fraction of accounts (sklearn max_df semantics) |
| `URL_COSHARING_EDGE_EPSILON` | `0.05` | Similarity threshold for including edges in graph |
| `URL_COSHARING_EDGE_QUANTILE_GRID` | `0.50,0.60,0.70,0.80,0.90,0.95,0.99` | Edge-weight quantiles for dismantling grid search |
| `URL_COSHARING_CENTRALITY_QUANTILE_GRID` | `0.50,0.60,0.70,0.80,0.90,0.95,0.99` | Centrality quantiles for dismantling grid search |
| `URL_COSHARING_DENSITY_FLOOR` | `0.5` | Minimum component density threshold for dismantling |
| `URL_COSHARING_MAX_FLAGGED_FRACTION` | `0.05` | Maximum fraction of eligible accounts to flag (guardrail) |
| `URL_COSHARING_MAX_FLAGGED_ACCOUNTS` | `750` | Maximum absolute number of accounts to flag; effective cap is `min(max_flagged_fraction × eligible_accounts, max_flagged_accounts)` |
| `URL_COSHARING_RESOLUTION` | `0.05` | Leiden CPM resolution parameter |
| `URL_COSHARING_MIN_CLUSTER_SIZE` | `3` | Minimum cluster membership size |
| `URL_COSHARING_JACCARD_THRESHOLD` | `0.5` | Jaccard similarity threshold for evolution matching |
| `URL_COSHARING_EVOLUTION_WINDOW_DAYS` | `7` | Historical window for cluster matching |
| `URL_COSHARING_INTERVAL_SECONDS` | `3600` | Seconds between analysis cycles |
| `URL_COSHARING_RUNS_TABLE` | `url_cosharing_runs` | ClickHouse table for run metadata |
| `URL_COSHARING_CLUSTERS_TABLE` | `url_cosharing_clusters` | ClickHouse table for cluster results |
| `URL_COSHARING_MEMBERSHIP_TABLE` | `url_cosharing_membership` | ClickHouse table for membership snapshots |
| `URL_COSHARING_SOURCE_TABLE` | `osprey_execution_results` | Source table for URL shares |

## Output schema

- `url_cosharing_clusters` — cluster results with member count, metrics (`mean_edge_similarity`, `subgraph_density`), and evolution tracking
- `url_cosharing_membership` — daily membership snapshots per cluster (no TTL; retained for post-hoc analysis)
- `url_cosharing_runs` — run metadata including stage counts, quantile choices, knee-finding result, flagged account count, and cluster count

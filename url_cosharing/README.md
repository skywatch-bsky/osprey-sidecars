# URL co-sharing graph sidecar

Identifies clusters of accounts that repeatedly share the same URLs within the same day using Leiden community detection with Newman-weighted edges to prevent viral URLs from dominating cluster formation.

## How it works

1. Reads from `url_cosharing_pairs` (populated by a ClickHouse materialized view from `osprey_execution_results`)
2. Builds a weighted co-sharing graph where:
   - **Raw weight** (co-share count): Used for `min_edge_weight` filtering and investigations
   - **Newman weight** (Σ 1/(k_url − 1)): Used by Leiden clustering to down-weight viral URLs
3. Runs Leiden community detection optimized on Newman weights to identify clusters
4. Computes per-cluster metrics and tracks cluster stability via Jaccard similarity between daily membership snapshots
5. Writes results to `url_cosharing_clusters` (cluster `total_weight` is the raw co-share sum) and daily membership snapshots to `url_cosharing_membership`

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
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DATABASE` | `default` | Database name |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between analysis cycles |

## Output schema

- `url_cosharing_clusters` — cluster results with member count, metrics, and evolution tracking
- `url_cosharing_membership` — daily membership snapshots per cluster (TTL 7 days)

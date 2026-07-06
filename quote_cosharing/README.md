# Quote post co-sharing graph sidecar

Identifies clusters of accounts that quote-post the same target URIs within the same day using Leiden community detection with Newman-weighted edges to prevent viral quoted URIs from dominating cluster formation.

## How it works

1. Reads from `quote_cosharing_pairs` (populated by a ClickHouse materialized view from `osprey_execution_results`)
2. Builds a weighted co-sharing graph where:
   - **Raw weight** (co-quote count): Used for `min_edge_weight` filtering and investigations
   - **Newman weight** (Σ 1/(k_uri − 1)): Used by Leiden clustering to down-weight viral quoted URIs
3. Runs Leiden community detection optimized on Newman weights to identify clusters
4. Computes per-cluster metrics and tracks cluster stability via Jaccard similarity between daily membership snapshots
5. Writes results to `quote_cosharing_clusters` (cluster `total_weight` is the raw co-quote sum) and daily membership snapshots to `quote_cosharing_membership`

## Usage

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run locally
uv run python -m quote_cosharing.main

# Run via Docker
docker build -t quote-cosharing .
docker run --env-file .env quote-cosharing
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DATABASE` | `default` | Database name |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between analysis cycles |

## Output schema

- `quote_cosharing_clusters` — cluster results with member count, metrics, and evolution tracking
- `quote_cosharing_membership` — daily membership snapshots per cluster (TTL 7 days)

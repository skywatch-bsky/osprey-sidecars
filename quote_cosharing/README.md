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


## OpenTelemetry

OpenTelemetry is disabled by default and is operational observability only, not durable domain audit data. Enabling it emits OTLP traces and metrics only; this sidecar does not start a Prometheus endpoint or require a collector unless telemetry is enabled.

Telemetry environment variables:

- `QUOTE_COSHARING_OTEL_ENABLED` (default `false`)
- `QUOTE_COSHARING_OTEL_SERVICE_NAME` (default `quote-cosharing`)
- `QUOTE_COSHARING_OTEL_SERVICE_VERSION` (default `0.1.0`)
- `QUOTE_COSHARING_OTEL_ENVIRONMENT` (default `local`)
- `QUOTE_COSHARING_OTEL_TRACES_ENABLED` (default follows `QUOTE_COSHARING_OTEL_ENABLED`)
- `QUOTE_COSHARING_OTEL_METRICS_ENABLED` (default follows `QUOTE_COSHARING_OTEL_ENABLED`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (optional collector endpoint used by the OTel SDK)

Keep telemetry low-cardinality. Never add DIDs, user IDs, account IDs, URLs/domains, quoted URIs, PDS hosts, rkeys, cluster IDs, sample values, table names, SQL/query text, ClickHouse credentials, or exception messages as attributes or metric labels. Package-specific forbidden values include: did, shared_uris, cluster_id.

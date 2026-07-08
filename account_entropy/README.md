# Account entropy sidecar

Detects automated posting patterns using bias-corrected entropy normalized for account post volume, plus an interval regularity signal. Reads from `osprey_execution_results`, writes scored results to `account_entropy_results`.

## How it works

1. Queries `osprey_execution_results` for accounts with sufficient post volume
2. Computes Shannon entropy over hourly posting distribution and inter-post timing intervals
3. Applies Miller–Madow bias correction and normalizes by `log2(min(N, bins))` to a 0–1 scale
4. Computes coefficient of variation of inter-post intervals as a regularity signal
5. Flags accounts when hourly entropy is high AND (interval entropy is low OR regularity is extreme)
6. Writes results to `account_entropy_results`

## Usage

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run locally
uv run python -m account_entropy.main

# Run via Docker
docker build -t account-entropy .
docker run --env-file .env account-entropy
```

## Configuration

### ClickHouse

| Variable | Default | Description |
|----------|---------|-------------|
| `OSPREY_CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `OSPREY_CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `OSPREY_CLICKHOUSE_USER` | `default` | ClickHouse username |
| `OSPREY_CLICKHOUSE_PASSWORD` | `clickhouse` | ClickHouse password |
| `OSPREY_CLICKHOUSE_DB` | `default` | Database name |

### Analysis

| Variable | Default | Description |
|----------|---------|-------------|
| `ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD` | `0.85` | Normalized hourly entropy threshold (0–1 scale); flags when ≥ threshold |
| `ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD` | `0.53` | Normalized interval entropy threshold (0–1 scale); flags when ≤ threshold |
| `ACCOUNT_ENTROPY_CV_THRESHOLD` | `0.5` | Coefficient-of-variation threshold; flags when ≤ threshold |
| `ACCOUNT_ENTROPY_INTERVAL_SECONDS` | `3600` | Analysis window interval in seconds |
| `ACCOUNT_ENTROPY_WINDOW_DAYS` | `7` | Lookback window in days |
| `ACCOUNT_ENTROPY_MIN_POSTS` | `10` | Minimum posts to score an account |
| `ACCOUNT_ENTROPY_INTERVAL_BIN_EDGES` | `60,300,900,3600,14400,86400` | Bin boundaries in seconds for inter-post interval histogram |
| `ACCOUNT_ENTROPY_SOURCE_TABLE` | `osprey_execution_results` | Source table name |
| `ACCOUNT_ENTROPY_OUTPUT_TABLE` | `account_entropy_results` | Output table name |


## OpenTelemetry

OpenTelemetry is disabled by default and is operational observability only, not durable domain audit data. Enabling it emits OTLP traces and metrics only; this sidecar does not start a Prometheus endpoint or require a collector unless telemetry is enabled.

Telemetry environment variables:

- `ACCOUNT_ENTROPY_OTEL_ENABLED` (default `false`)
- `ACCOUNT_ENTROPY_OTEL_SERVICE_NAME` (default `account-entropy`)
- `ACCOUNT_ENTROPY_OTEL_SERVICE_VERSION` (default `0.1.0`)
- `ACCOUNT_ENTROPY_OTEL_ENVIRONMENT` (default `local`)
- `ACCOUNT_ENTROPY_OTEL_TRACES_ENABLED` (default follows `ACCOUNT_ENTROPY_OTEL_ENABLED`)
- `ACCOUNT_ENTROPY_OTEL_METRICS_ENABLED` (default follows `ACCOUNT_ENTROPY_OTEL_ENABLED`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (optional collector endpoint used by the OTel SDK)

Keep telemetry low-cardinality. Never add DIDs, user IDs, account IDs, URLs/domains, quoted URIs, PDS hosts, rkeys, cluster IDs, sample values, table names, SQL/query text, ClickHouse credentials, or exception messages as attributes or metric labels. Package-specific forbidden values include: user_id, sample_rkeys, rkey.

# Signup anomaly sidecar

Detects anomalous PDS signup spikes using Poisson statistics at daily and hourly granularity.

## How it works

1. Queries `osprey_execution_results` for identity creation events
2. Computes per-PDS signup counts against rolling baselines
3. Falls back to population median for new or low-volume PDS hosts
4. Computes dispersion diagnostics (variance/mean ratio) to flag overdispersion
5. Writes scored results to `pds_signup_anomalies`

Excluded hosts (`bsky.network`, `bridgy-fed`, `mostr.pub`) are never scored.

## Usage

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run locally
uv run python -m signup_anomaly.main

# Run via Docker
docker build -t signup-anomaly .
docker run --env-file .env signup-anomaly
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DATABASE` | `default` | Database name |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between analysis cycles |

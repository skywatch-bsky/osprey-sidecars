# Quote post overdispersion sidecar

Detects anomalous quote-post concentration targeting specific posts using Poisson and normal-approximation statistics.

## How it works

1. Queries `osprey_execution_results` for posts with embedded quote-post URIs
2. Scores quote volume and density per target URI against baselines
3. Flags target posts exceeding Poisson-derived thresholds
4. Writes scored results to `quote_overdispersion_results`

## Usage

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run locally
uv run python -m quote_overdispersion.main

# Run via Docker
docker build -t quote-overdispersion .
docker run --env-file .env quote-overdispersion
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DATABASE` | `default` | Database name |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between analysis cycles |

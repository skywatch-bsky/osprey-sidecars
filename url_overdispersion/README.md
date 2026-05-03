# URL overdispersion sidecar

Detects anomalous domain sharing patterns using Poisson and normal-approximation statistics.

## How it works

1. Queries `osprey_execution_results` for posts containing URLs
2. Scores domain sharing volume and density against baselines
3. Flags domains exceeding Poisson-derived thresholds
4. Writes scored results to `url_overdispersion_results`

## Usage

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run locally
uv run python -m url_overdispersion.main

# Run via Docker
docker build -t url-overdispersion .
docker run --env-file .env url-overdispersion
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DATABASE` | `default` | Database name |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between analysis cycles |

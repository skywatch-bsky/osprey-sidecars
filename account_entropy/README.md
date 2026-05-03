# Account entropy sidecar

Detects automated posting patterns by computing Shannon entropy over account posting timestamps. Low entropy (regular timing) indicates likely automation.

## How it works

1. Queries `osprey_execution_results` for accounts with sufficient post volume
2. Computes Shannon entropy over inter-post timing distributions
3. Scores accounts against configurable entropy thresholds (see config.py)
4. Writes results to `account_entropy_results`

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

| Variable | Default | Description |
|----------|---------|-------------|
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DATABASE` | `default` | Database name |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between analysis cycles |

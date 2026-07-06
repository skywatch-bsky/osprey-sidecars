# URL overdispersion sidecar

Detects anomalous domain sharing patterns using negative-binomial volume tests and beta-binomial density tests with false-discovery-rate control.

## How it works

1. Queries `osprey_execution_results` for posts containing URLs
2. Builds dense, zero-filled volume and density baselines with rolling medians and variances
3. Scores domain sharing volume via negative binomial (with Poisson fallback) and sharer density via beta-binomial (with binomial fallback)
4. Applies Benjamini–Hochberg false-discovery-rate adjustment per signal per analysis cycle
5. Flags domains where volume or density q-value falls below the FDR target
6. Writes scored results to `url_overdispersion_results`

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
| `URL_OVERDISPERSION_VOLUME_P_THRESHOLD` | `0.01` | FDR target for volume anomalies (q-value threshold) |
| `URL_OVERDISPERSION_DENSITY_P_THRESHOLD` | `0.01` | FDR target for density anomalies (q-value threshold) |
| `URL_OVERDISPERSION_BASELINE_DAYS` | `14` | Rolling baseline window in days |
| `URL_OVERDISPERSION_COLD_START_MIN_DAYS` | `3` | Minimum baseline days to use entity baseline |
| `URL_OVERDISPERSION_MIN_SHARERS` | `3` | Minimum unique sharers required to score a domain on a given day |

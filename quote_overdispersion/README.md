# Quote post overdispersion sidecar

Detects anomalous quote-post concentration targeting specific posts using negative-binomial volume tests and beta-binomial density tests with false-discovery-rate control.

## How it works

1. Queries `osprey_execution_results` for posts with embedded quote-post URIs
2. Builds dense, zero-filled volume and density baselines with rolling medians and variances
3. Scores quoted-post volume via negative binomial (with Poisson fallback) and sharer density via beta-binomial (with binomial fallback)
4. Applies Benjamini–Hochberg false-discovery-rate adjustment per signal per analysis cycle
5. Flags quoted posts where volume or density q-value falls below the FDR target
6. Writes scored results to `quote_overdispersion_results`

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
| `QUOTE_OVERDISPERSION_VOLUME_P_THRESHOLD` | `0.01` | FDR target for volume anomalies (q-value threshold) |
| `QUOTE_OVERDISPERSION_DENSITY_P_THRESHOLD` | `0.01` | FDR target for density anomalies (q-value threshold) |
| `QUOTE_OVERDISPERSION_BASELINE_DAYS` | `14` | Rolling baseline window in days |
| `QUOTE_OVERDISPERSION_COLD_START_MIN_DAYS` | `3` | Minimum baseline days to use entity baseline |
| `QUOTE_OVERDISPERSION_MIN_SHARERS` | `3` | Minimum unique sharers required to score a quoted post on a given day |

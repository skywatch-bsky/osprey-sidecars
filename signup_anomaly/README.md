# Signup anomaly sidecar

Detects anomalous PDS signup spikes using negative binomial (NB) statistics with Poisson fallback, at daily and hourly granularity, with Benjamini–Hochberg false-discovery-rate control.

## How it works

1. Queries `osprey_execution_results` for identity creation events
2. Builds dense, zero-filled baselines: rolling **median** (expected count) and dispersion factor (variance/mean ratio) over a sliding window of historical data
3. Tests each signup count against an NB or Poisson model:
   - **Negative binomial (NB):** method-of-moments fit when baseline is overdispersed (variance > mean); more conservative p-values than Poisson
   - **Poisson fallback:** when baseline is not overdispersed or dispersion is unknown
4. Falls back to population median baseline for new or low-activity PDS hosts (cold-start)
5. Adjusts raw p-values using Benjamini–Hochberg (BH) FDR per cycle and granularity; `q_value < threshold` flags anomaly
6. Writes scored results including p-value, q-value, and dispersion diagnostics to `pds_signup_anomalies`

Excluded hosts (`bsky.network`, `bridgy-fed`, `mostr.pub`) are never scored.

**Hourly baselines** use hour-of-day matching: each 1 a.m. today is compared to 1 a.m. yesterday, 1 a.m. two days ago, etc., preventing circadian patterns from inflating baselines.

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
| `SIGNUP_ANOMALY_DAILY_P_THRESHOLD` | `0.01` | BH-FDR target (q-value threshold) for daily anomalies |
| `SIGNUP_ANOMALY_HOURLY_P_THRESHOLD` | `0.05` | BH-FDR target (q-value threshold) for hourly anomalies |
| `SIGNUP_ANOMALY_BASELINE_DAYS` | `7` | Rolling window size (days) for baseline computation |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between analysis cycles |

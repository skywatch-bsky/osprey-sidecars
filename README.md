# Osprey Sidecars

Statistical detection sidecars for the [Osprey](https://github.com/skywatch-bsky/skywatch-osprey) AT Protocol moderation engine. Each sidecar polls ClickHouse (osprey_execution_results), runs a detection algorithm, and writes results to its own output table.

## Architecture

All sidecars follow Functional Core / Imperative Shell:

| Layer | Files | Role |
|-------|-------|------|
| Core | `config.py`, `queries.py`, `analyzer.py` | Pure functions — config parsing, SQL generation, statistical computation |
| Shell | `db.py`, `main.py` | I/O — ClickHouse client, polling loop, signal handling |

No sidecar imports from `osprey_worker` or from any other sidecar.

## Sidecars

| Sidecar | Detects | Method | Output Table |
|---------|---------|--------|--------------|
| [account_entropy](account_entropy/) | Automated posting patterns | Bias-corrected normalized entropy + interval CV | `account_entropy_results` |
| [signup_anomaly](signup_anomaly/) | Anomalous PDS signup spikes | Negative binomial / Poisson on dense median baselines, BH-FDR | `pds_signup_anomalies` |
| [url_overdispersion](url_overdispersion/) | Anomalous domain sharing | NB volume + beta-binomial density, BH-FDR per signal | `url_overdispersion_results` |
| [quote_overdispersion](quote_overdispersion/) | Anomalous quote-post concentration | NB volume + beta-binomial density, BH-FDR per signal | `quote_overdispersion_results` |
| [url_cosharing](url_cosharing/) | Coordinated URL sharing | Leiden (CPM) on Newman-weighted co-sharing graph | `url_cosharing_clusters` |
| [quote_cosharing](quote_cosharing/) | Coordinated quote-posting | Leiden (CPM) on Newman-weighted co-sharing graph | `quote_cosharing_clusters` |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- ClickHouse with `osprey_execution_results` table populated

## Quick start

```bash
# Run a sidecar locally
cd <sidecar_dir>
uv sync
uv run python -m signup_anomaly.main

# Run tests
uv run pytest

# Run via Docker Compose (from the main Osprey repo)
docker compose up <service-name>
```

## Environment variables

Each sidecar reads ClickHouse connection details from environment variables. Check each sidecar's `config.py` for the full set of environment variables.

Common variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DATABASE` | `default` | ClickHouse database name |
| `POLL_INTERVAL_SECONDS` | varies | Seconds between analysis cycles |

### account_entropy thresholds

Account entropy sidecar uses three normalized threshold environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD` | `0.85` | Hourly entropy ≥ this triggers hourly_flag (normalized to [0, 1]) |
| `ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD` | `0.53` | Interval entropy ≤ this triggers interval_flag (normalized to [0, 1]) |
| `ACCOUNT_ENTROPY_CV_THRESHOLD` | `0.5` | Coefficient of variation ≤ this triggers cv_flag |

### Statistical methodology

See [docs/calibration.md](docs/calibration.md) for post-deploy validation queries, healthy metric ranges, and tuning guidance for all sidecars. Note that threshold variables with `*_P_THRESHOLD` or `*_NORM_THRESHOLD` suffixes are FDR (False Discovery Rate) targets used in the BH (Benjamini–Hochberg) multiple-comparison correction.

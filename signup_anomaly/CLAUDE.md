# Signup Anomaly Detector

Last verified: 2026-03-25

## Purpose
Standalone sidecar that detects anomalous PDS signup patterns using Poisson statistics. Runs independently of the Osprey rules engine -- not a uv workspace member.

## Contracts
- **Exposes**: No API. Runs as a polling loop writing results to ClickHouse.
- **Guarantees**: Each cycle queries `osprey_execution_results` for identity events, scores per-PDS signup counts against rolling baselines (entity or population fallback), computes dispersion diagnostics (variance and dispersion index with the same cold-start fallback), and inserts scored rows into `pds_signup_anomalies`.
- **Expects**: ClickHouse reachable with `osprey_execution_results` table populated and `pds_signup_anomalies` table created (via `clickhouse-init/02-signup-anomalies.sql`).

## Architecture
Functional Core / Imperative Shell:
- `config.py`, `queries.py`, `analyzer.py` -- pure functions, no I/O
- `db.py`, `main.py` -- I/O shell (ClickHouse client, signal handling, sleep loop)

Cold-start handling: when a PDS has fewer than `cold_start_min_days` of history, the analyzer falls back to population median lambda as baseline and population dispersion index for dispersion diagnostics.

## Dependencies
- **Uses**: ClickHouse (`osprey_execution_results` read, `pds_signup_anomalies` write)
- **Used by**: Nothing yet (results table available for UI/alerting)
- **Boundary**: No imports from osprey_worker or example_plugins -- fully standalone

## Key Decisions
- Standalone project (not uv workspace): avoids coupling to Osprey dependency tree
- Poisson model: signup counts are naturally Poisson-distributed; `scipy.stats.poisson.sf` for p-values
- Dual granularity (daily + hourly): daily catches sustained anomalies, hourly catches bursts
- Population median fallback: prevents false negatives for new/low-volume PDS hosts
- Dispersion diagnostic: `rolling_variance` and `dispersion_index` (variance/mean) expose overdispersion relative to Poisson assumption; dispersion_index > 1 signals the Poisson model may underfit

## Commands
- `cd signup_anomaly && uv run pytest` -- run tests
- `docker compose up signup-anomaly` -- run via compose

## Invariants
- Excluded hosts (bsky.network, bridgy-fed, mostr.pub) are never scored
- `bsky.network` exclusion uses LIKE '%bsky.network' (catches subdomains)
- p_value is always 1.0 when expected_lambda <= 0
- is_anomaly is 0 when observed_count is 0, regardless of p_value
- dispersion_index is NULL when rolling_mean < 1.0 (avoids noisy ratios at low counts)
- dispersion fallback mirrors baseline fallback: entity dispersion when history is sufficient, population dispersion otherwise

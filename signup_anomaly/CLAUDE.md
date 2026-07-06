# Signup Anomaly Detector

Last verified: 2026-07-06

## Purpose
Standalone sidecar that detects anomalous PDS signup patterns using negative binomial (NB) statistics with Poisson fallback, employing Benjamini–Hochberg false-discovery-rate control per cycle and granularity. Runs independently of the Osprey rules engine -- not a uv workspace member.

## Contracts
- **Exposes**: No API. Runs as a polling loop writing results to ClickHouse.
- **Guarantees**: Each cycle queries `osprey_execution_results` for identity events, builds dense zero-filled baselines with rolling median expected counts and dispersion factors (variance-to-mean ratios), scores per-PDS signup counts using NB with Poisson fallback, adjusts raw p-values via Benjamini–Hochberg FDR per cycle per granularity, and inserts scored rows into `pds_signup_anomalies` (including new `q_value` column).
- **Expects**: ClickHouse reachable with `osprey_execution_results` table populated and `pds_signup_anomalies` table created with `q_value` column (via `clickhouse-init/02-signup-anomalies.sql`).

## Architecture
Functional Core / Imperative Shell:
- `config.py`, `queries.py`, `counts.py`, `fdr.py`, `analyzer.py` -- pure functions, no I/O
- `db.py`, `main.py` -- I/O shell (ClickHouse client, signal handling, sleep loop)

**Baseline computation** (`queries.py`): densified daily and hourly window queries generate rolling median (expected count), rolling mean and variance for dispersion-factor computation, and population-level medians (fallback).

**Dispersion-aware testing** (`counts.py`): method-of-moments negative binomial when variance > mean (i.e., dispersion factor φ > 1); Poisson fallback otherwise. Non-integer NB shape parameter r supported (scipy nbinom with regularized incomplete beta).

**FDR control** (`fdr.py`, `analyzer.py`): raw p-values from `score_row` are collected per cycle per granularity, adjusted via Benjamini–Hochberg, and `is_anomaly` decisions use q-value thresholds.

Cold-start handling: when a PDS has fewer than `cold_start_min_days` of history, the analyzer falls back to population median lambda as baseline and population dispersion factor for NB fitting.

## Dependencies
- **Uses**: ClickHouse (`osprey_execution_results` read, `pds_signup_anomalies` write)
- **Used by**: Nothing yet (results table available for UI/alerting)
- **Boundary**: No imports from osprey_worker or example_plugins -- fully standalone

## Key Decisions
- Standalone project (not uv workspace): avoids coupling to Osprey dependency tree
- Negative binomial model with Poisson fallback: method-of-moments NB when baseline is overdispersed (φ > 1), providing more calibrated false-positive rates under overdispersion
- Dense baselines: zero-filled windows prevent selective-visibility bias in baseline medians; all-zero windows still trigger cold-start fallback correctly
- Median expected count: robust to outliers; no parametric distribution assumption for the baseline itself
- Dual granularity (daily + hourly): daily catches sustained anomalies, hourly catches bursts; hourly uses hour-of-day matching (e.g., 1 a.m. vs. 1 a.m.) to avoid circadian pattern bias
- Population median fallback: prevents false negatives for new/low-activity PDS hosts
- Benjamini–Hochberg FDR: per-cycle per-granularity families; threshold env vars reinterpreted as q-value targets (FDR levels), not p-value cutoffs
- Dispersion diagnostic: `dispersion_index` column is the variance-to-mean **dispersion factor** φ of the baseline window; φ > 1 indicates overdispersion relative to Poisson and triggers NB model

## Commands
- `cd signup_anomaly && uv run pytest` -- run tests
- `docker compose up signup-anomaly` -- run via compose

## Invariants
- Excluded hosts (bsky.network, bridgy-fed, mostr.pub) are never scored
- `bsky.network` exclusion uses LIKE '%bsky.network' (catches subdomains)
- p_value is always 1.0 when expected_lambda <= 0 (no anomaly possible with zero baseline)
- is_anomaly is 0 when observed_count is 0, regardless of q_value (zero counts never flag)
- NB model used only when φ > 1.0 and expected_lambda > 0; otherwise Poisson path (or 1.0 if expected_lambda ≤ 0)
- dispersion_index is NULL when rolling_mean ≤ 0 (avoids invalid ratios at zero/negative means)
- dispersion fallback mirrors baseline fallback: entity dispersion when history is sufficient, population dispersion otherwise
- q_value is the Benjamini–Hochberg adjusted p-value, monotone non-decreasing when p-values are sorted
- New column `q_value` (Float64) inserted after `p_value` in `pds_signup_anomalies`

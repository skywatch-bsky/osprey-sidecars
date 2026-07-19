# Quote Post Overdispersion Sidecar

Last verified: 2026-07-06

## Purpose

Detects anomalous quote-post concentration targeting specific posts using dispersion-aware statistical tests and false-discovery-rate control. Reads from `osprey_execution_results`, writes scored results to `quote_overdispersion_results` with volume and density anomaly scores.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `counts.py` — negative-binomial volume test with Poisson fallback (Core)
- `density.py` — beta-binomial sharer-density test with binomial fallback (Core)
- `fdr.py` — Benjamini–Hochberg false-discovery-rate adjustment (Core)
- `queries.py` — SQL query generation for dense, zero-filled baselines with rolling medians and variances (Core)
- `analyzer.py` — two-pass scoring with per-signal BH-FDR control and baseline selection (Core)
- `db.py` — ClickHouse client wrapper with row fetch and result insert (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Contract

- **Input:** `osprey_execution_results` (posts with embedded quote-post URIs, filtered to `Collection = 'app.bsky.feed.post'` and `OperationKind = 'create'`)
- **Output:** `quote_overdispersion_results` (scored quoted URIs per time bucket with volume and density anomaly scores, q-values, and diagnostic columns)
- **Dependencies:** ClickHouse (`clickhouse-connect`) for data; `scipy` for the NB/Poisson/beta-binomial/binomial distributions in `counts.py` and `density.py`. No imports from osprey_worker or other sidecars.

## Methodology

### Volume Testing
- **Baseline:** Median daily quote count (rolling 7-day window, densified with zeros)
- **Test:** Negative binomial with method-of-moments dispersion factor; Poisson when variance ≤ mean
- **One-sided:** Upper tail (high volume anomaly)

### Density Testing
- **Baseline:** Mean sharer density within entity baseline (0–1 range), or population median
- **Test:** Beta-binomial with method-of-moments α/β; binomial when variance exceeds ceiling
- **One-sided:** Upper tail (high concentration anomaly)
- **Known approximation:** First share is always unique (dependent event); model treats all sharers as independent

### Baseline Selection
- **Entity baseline:** Used when ≥ 1 day of rolling data exist AND rolling median volume > 0 AND rolling mean density > 0
- **Population baseline:** Used when entity baseline unavailable; computed over today's scored quoted posts in the same batch
- **Fallback:** Returns zero baseline (p-value = 1.0 for all tests)

### FDR Control
- **Two-pass per cycle:** Score all rows with raw p-values, then adjust volume and density separately via Benjamini–Hochberg
- **Per-signal decision:** Anomaly when volume q-value < target OR density q-value < target
- **Granularity separation:** Daily and hourly cycles run independently; each produces a separate BH family
- **Population scope:** Population medians computed only over today's scored quoted posts (≥ 3 unique sharers)

### Baseline Densification
- **Dense grid:** All (quoted_uri, day) pairs from first-seen through today, with zero volume/NULL density for inactive days
- **min_sharers isolation:** Applied only to today's scored rows (the final WHERE clause), not to historical baseline construction
- **Rolling window:** 7-day entity lookback, 14-hour hourly lookback partitioned by hour-of-day
- **Window sampling:** arraySlice to first 5 sample user DIDs per (quoted_uri, bucket)

## Output Columns

Scored results include volume and density p-values, q-values, and four diagnostic columns for post-deploy calibration:
- `rolling_volume_median`, `rolling_volume_variance`: Median volume and variance from entity baseline
- `rolling_density_mean`, `rolling_density_variance`: Mean and variance of sharer density from entity baseline

These columns enable recomputing dispersion factors and FDR adjustments without re-fetching baselines.

## Commands

- `cd quote_overdispersion && uv run pytest` — Run tests
- `docker compose up quote-overdispersion` — Start sidecar


## OpenTelemetry

`telemetry.py` is imperative shell. Functional core modules must not import OpenTelemetry; keep OTel setup and span/metric helpers in `telemetry.py` and orchestration calls in `main.py`.

Allowed dimensions are fixed stage names, coarse counts, booleans, granularity where applicable, `window_days` where applicable, and `error.type`. Do not emit high-cardinality or sensitive values, including quoted_uri, sample_dids, DIDs, URLs/domains, quoted URIs, PDS hosts, rkeys, sample arrays, table names, SQL/query text, credentials, or exception messages.

Run tests with `cd quote_overdispersion && uv run pytest`.

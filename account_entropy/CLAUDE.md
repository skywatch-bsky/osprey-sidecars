# Account Entropy Sidecar

Last verified: 2026-07-06

## Purpose

Detects accounts with automated posting patterns using bias-corrected, normalized Shannon entropy plus a coefficient-of-variation regularity signal. Reads from `osprey_execution_results`, writes scored results to `account_entropy_results`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — normalized entropy computation (Miller–Madow bias-corrected, scaled to [0,1]), coefficient-of-variation signal, bot-likelihood scoring (Core)
- `db.py` — ClickHouse client wrapper with row fetch and result insert (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Contract

- **Input:** `osprey_execution_results` (post timestamps for accounts with N+ posts, filtered to `Collection = 'app.bsky.feed.post'` and `OperationKind = 'create'`)
- **Output:** `account_entropy_results` (entropy scores per account per analysis window with normalized entropy, CV signal, and bot-like flag)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Methodology

### Entropy Normalization

**Raw entropy:** Shannon entropy in bits (retained as diagnostic context).

**Normalized entropy:** Applies Miller–Madow bias correction H_mm = H + (K_occupied − 1) / (2 * N * ln 2) bits, then divides by the achievable maximum log2(min(N, bins)) to produce a 0–1 scale. This is the key fix: accounts with fewer posts than available bins are no longer penalized for their lower post volume.

- **Hourly:** 24 bins; normalized by log2(min(post_count, 24))
- **Interval:** 7 bins (from default 6 bin edges); normalized by log2(min(interval_count, 7))

### Regularity Signal

**Coefficient of variation:** stddev(inter-post intervals) / mean(inter-post intervals) in seconds. Captures metronomic posting cadence: CV near 0 = maximally regular (botlike); high CV = variable posting (human-like).

### Scoring

Three independent signals; bot-like requires the conjunction:

| Signal | Condition | Flag |
|--------|-----------|------|
| Hourly entropy | normalized >= 0.85 (default) | `hourly_flag` |
| Interval entropy | normalized <= 0.53 (default) | `interval_flag` |
| Regularity | CV <= 0.5 (default) | `cv_flag` |
| **Bot-like** | hourly_flag AND (interval_flag OR cv_flag) | `is_bot_like` |

## Output Columns

In addition to scoring timestamps, identifiers, and raw entropy in bits:
- `hourly_entropy_norm` (float) — Normalized hourly entropy [0, 1]
- `interval_entropy_norm` (float) — Normalized interval entropy [0, 1]
- `interval_cv` (float) — Coefficient of variation of inter-post intervals
- `cv_flag` (int) — 1 if interval_cv <= threshold, else 0

## Commands

- `cd account_entropy && uv run pytest` — Run tests
- `docker compose up account-entropy` — Start sidecar

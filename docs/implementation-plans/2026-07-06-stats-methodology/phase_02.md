# Statistical Methodology Fixes — Phase 2: url_overdispersion Implementation Plan

**Goal:** NB volume test, beta-binomial density test, dense median baselines free of selection bias, hour-of-day matching, and per-signal BH-FDR for the URL overdispersion sidecar.

**Architecture:** Three new Functional Core modules (`counts.py`, `fdr.py`, `density.py` — duplicated per sidecar by convention; `counts.py`/`fdr.py` are byte-identical to Phase 1's under this sidecar's package), rewritten baseline SQL in `queries.py` (densification, `medianExact`, `min_sharers` moved to scored rows only), a two-pass scorer in `analyzer.py`, and column additions in `db.py` + schema.

**Tech Stack:** Python 3.11+, scipy ≥ 1.15 (`nbinom`, `poisson`, `betabinom`, `binom` — `betabinom` has been in scipy since 1.4), clickhouse-connect, ClickHouse window functions, pytest via `uv run pytest`.

**Scope:** Phase 2 of 7 from `docs/design-plans/2026-07-06-stats-methodology.md` (independent of phases 1, 3–6).

**Codebase verified:** 2026-07-06 via codebase-investigator agents.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### stats-methodology.AC1: Count tests are dispersion-aware
- **stats-methodology.AC1.1 Success:** `count_p_value` returns the NB survival probability (MoM: r = μ²/(σ²−μ), p = μ/σ²) when variance > mean, verified against a hand-computed example
- **stats-methodology.AC1.2 Success:** Falls back to Poisson `sf(observed−1, μ)` when variance ≤ mean or variance is unavailable
- **stats-methodology.AC1.3 Success:** Non-integer r produces a valid p-value in (0, 1] (scipy regression guard)
- **stats-methodology.AC1.4 Edge:** mean ≤ 0 or missing baseline → p = 1.0; observed = 0 → never flagged

### stats-methodology.AC2: FDR control
- **stats-methodology.AC2.1 Success:** `bh_adjust` matches a hand-computed BH example, q-values monotone, input order preserved
- **stats-methodology.AC2.2 Success:** `is_anomaly = 1` iff `q_value < threshold` (plus surviving guards like `observed_count > 0`)
- **stats-methodology.AC2.3 Success:** Daily and hourly rows in one cycle are adjusted as separate families; volume and density p-values as separate families
- **stats-methodology.AC2.4 Edge:** Empty input → empty output; single p-value → q = p

### stats-methodology.AC3: Density test
- **stats-methodology.AC3.1 Success:** Beta-binomial `sf` used when the MoM fit (from rolling density mean + variance) is valid; verified against a known example
- **stats-methodology.AC3.2 Success:** Falls back to plain binomial when variance is degenerate/absent
- **stats-methodology.AC3.3 Success:** One-sided high: observed density at or below baseline never flags
- **stats-methodology.AC3.4 Edge:** expected_density ≤ 0 or total_shares = 0 → p = 1.0

### stats-methodology.AC4: Baselines
- **stats-methodology.AC4.1 Success:** Generated SQL densifies entity×bucket grids — a bucket with zero events contributes 0 to the rolling window
- **stats-methodology.AC4.2 Success:** Expected count = rolling median; rolling mean/variance still emitted for the dispersion factor
- **stats-methodology.AC4.3 Success:** Hourly baseline windows partition by (entity, hour-of-day)
- **stats-methodology.AC4.4 Success:** `min_sharers`-type filters apply only to scored rows; population medians computed without the `dispersion_index IS NOT NULL` / `rolling_mean >= 1` qualifiers

### stats-methodology.AC7: Schemas
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar) — *url scope: `03-url-overdispersion.sql`*

### stats-methodology.AC9: Suites
- **stats-methodology.AC9.1 Success:** `uv run pytest` passes in all six sidecars — *url_overdispersion scope*

---

## Context from codebase verification

Current state (verified 2026-07-06):

- `url_overdispersion/src/url_overdispersion/analyzer.py` (148 lines): `compute_p_value` (12–20, Poisson sf), `compute_density_p_value` (23–42, normal-approximation z-test — already one-sided via `if z <= 0: return 1.0`, but the normal approximation on a proportion is the thing being replaced), `determine_baseline` (45–74, returns `(volume_lambda, density_lambda, source)`; requires both rolling means present for entity path, both population medians > 0 for fallback), `score_row` (77–125, OR-logic `is_anomaly` at 103–104), `score_rows` (128–147, adds `on_watchlist` via `watchlist_domains` membership).
- `queries.py` (135 lines): `daily_aggregation_query` (7–69) — CTEs `domain_shares` (with **`HAVING unique_sharers >= {min_sharers}` applied to all history**, line 24) → `baseline` (rolling `avg` volume + density over `PARTITION BY domain ORDER BY bucket ROWS BETWEEN {baseline_days} PRECEDING AND 1 PRECEDING`; **no densification**) → `population_stats` (**`rolling_volume_mean IS NOT NULL` qualifier**, line 50). `hourly_aggregation_query` (72–134) — continuous `{baseline_days * 24}`-row window, **no hour-of-day matching**, `intDiv(..., 24)` conversion. Keep the existing entity-extraction and sample expressions in `domain_shares` verbatim (domain from post-domains array, `sample_dids`, `sample_urls` via `arraySlice`), plus all `Collection`/`OperationKind` filters.
- `db.py` (137 lines): frozen `AggregatedRow` (13–26: domain, bucket_start, total_shares, unique_sharers, sharer_density, rolling_volume_mean, rolling_density_mean, baseline_days_available, sample_dids, sample_urls, population_volume_median, population_density_median) and `ScoredResult` (29–47, 17 fields); insert column list at 92–110 (17 names ending `on_watchlist`); output table `url_overdispersion_results`.
- `config.py`: `volume_p_threshold` (env `URL_OVERDISPERSION_VOLUME_P_THRESHOLD`, 0.01), `density_p_threshold` (env `URL_OVERDISPERSION_DENSITY_P_THRESHOLD`, 0.01), `baseline_days` (14), `cold_start_min_days` (3), `min_sharers` (3), `watchlist_domains`, tables. Env var names kept, thresholds reinterpreted as FDR targets.
- Schema: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/03-url-overdispersion.sql`, table `default.url_overdispersion_results`, 17 columns, MergeTree `ORDER BY (run_timestamp, granularity, domain)`, no TTL.
- `pyproject.toml`: `scipy>=1.15.0` present (covers `betabinom`). Tests: `uv run pytest`; SQL string assertions; `FakeDb` in `test_main.py`. Ruff single quotes/120.

External-dependency findings (internet-researcher, 2026-07-06):
- `scipy.stats.betabinom(n, a, b)` (since scipy 1.4): one-sided upper tail is `betabinom.sf(x - 1, n, a, b)` = P(X ≥ x).
- Beta MoM from a mean proportion μ and proportion variance σ²: M = μ(1−μ)/σ² − 1, α = μM, β = (1−μ)M; valid iff 0 < μ < 1, σ² > 0, and M > 0 (equivalently σ² < μ(1−μ)). Invalid fit ⇒ fall back to plain binomial.
- `scipy.stats.nbinom.sf` with non-integer r: guard with a regression test against the exact incomplete-beta identity `P(X ≥ observed) = scipy.special.betainc(observed, r, 1 − p)` (scipy#16120); `scipy.special.nbdtrc` is NOT a valid cross-check or fallback — it truncates non-integer r to an integer. See Task 1.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: `counts.py` — dispersion-aware count test (duplicated module)

**Verifies:** stats-methodology.AC1.1, stats-methodology.AC1.2, stats-methodology.AC1.3, stats-methodology.AC1.4

**Files:**
- Create: `url_overdispersion/src/url_overdispersion/counts.py`
- Test: `url_overdispersion/tests/test_counts.py` (unit)

**Implementation:**

Create `counts.py` with exactly this content (the cross-sidecar contract module; byte-identical to the copy in `signup_anomaly` and `quote_overdispersion` — sidecars deliberately share no code, so each carries its own copy):

```python
# pattern: Functional Core
"""Dispersion-aware count tests: negative binomial with Poisson fallback."""

from scipy.stats import nbinom, poisson


def count_p_value(observed: int, mean: float, variance: float | None) -> float:
    """P(X >= observed) under NB when variance > mean, else Poisson(mean).

    NB via method of moments: r = mean**2 / (variance - mean), p = mean / variance.
    mean <= 0 -> 1.0. Non-integer r supported (scipy nbinom).
    """
    if mean <= 0:
        return 1.0
    if variance is not None and variance > mean:
        r = mean * mean / (variance - mean)
        p = mean / variance
        return float(nbinom.sf(observed - 1, r, p))
    return float(poisson.sf(observed - 1, mean))
```

**Testing:**

Create `tests/test_counts.py` with the same test matrix as the signup_anomaly copy (each sidecar's duplicated module carries its own tests):
- AC1.1 hand-computed NB: `count_p_value(3, 2.0, 4.0) == pytest.approx(0.3125, abs=1e-12)` (r = 2, p = 0.5; P(X≥3) = 1 − (0.25 + 0.25 + 0.1875)).
- AC1.2: `count_p_value(5, 3.0, 2.0)`, `count_p_value(5, 3.0, 3.0)`, and `count_p_value(5, 3.0, None)` all equal `float(poisson.sf(4, 3.0))`.
- AC1.3 (scipy#16120 regression): `count_p_value(7, 3.0, 5.0)` (r = 4.5, p = 0.6) is in (0, 1] and equals `float(scipy.special.betainc(7, 4.5, 0.4))` within `abs=1e-10` — the exact identity `P(X ≥ observed) = betainc(observed, r, 1 − p)` (≈ 0.07519). Do NOT use `scipy.special.nbdtrc` here: it truncates r = 4.5 to 4 (0.05476) and fails against a correct `nbinom.sf`. Documented fallback if a future scipy pin breaks this: implement the NB branch as `float(betainc(observed, r, 1.0 - p))` with `observed` kept as an int.
- AC1.4: mean ≤ 0 → 1.0 (both `0.0` and `-1.0`); `count_p_value(0, 10.0, 20.0) == 1.0` and `count_p_value(0, 10.0, None) == 1.0`.
- Monotonicity and NB-vs-Poisson ordering: `count_p_value(15, 5.0, 15.0) > float(poisson.sf(14, 5.0))`.

**Verification:**
Run: `cd url_overdispersion && uv run pytest tests/test_counts.py`
Expected: all pass.

**Commit:** `feat: add NB/Poisson count test module to url_overdispersion`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `fdr.py` — Benjamini–Hochberg adjustment (duplicated module)

**Verifies:** stats-methodology.AC2.1, stats-methodology.AC2.4

**Files:**
- Create: `url_overdispersion/src/url_overdispersion/fdr.py`
- Test: `url_overdispersion/tests/test_fdr.py` (unit)

**Implementation:**

Create `fdr.py` with exactly this content (byte-identical to the other sidecars' copies):

```python
# pattern: Functional Core
"""Benjamini-Hochberg false-discovery-rate adjustment. Pure Python, no dependencies."""


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values: step-up with cumulative-min monotonicity.

    Input order preserved. Empty list -> empty list. q-values capped at 1.0.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    q_values = [0.0] * n
    running_min = 1.0
    for position in range(n - 1, -1, -1):
        idx = order[position]
        rank = position + 1
        running_min = min(running_min, min(1.0, p_values[idx] * n / rank))
        q_values[idx] = running_min
    return q_values
```

**Testing:**

Create `tests/test_fdr.py` with the standard matrix:
- AC2.1 hand example: `bh_adjust([0.01, 0.04, 0.03, 0.005]) == [0.02, 0.04, 0.04, 0.02]` (element-wise `pytest.approx`).
- Monotonicity in p-order and `q >= p` on a messier vector (e.g. `[0.9, 0.001, 0.5, 0.02, 0.02]`); order preservation; ties share a q.
- AC2.4: `[] -> []`, `[0.03] -> [0.03]`; cap at 1.0.

**Verification:**
Run: `cd url_overdispersion && uv run pytest tests/test_fdr.py`
Expected: all pass.

**Commit:** `feat: add Benjamini-Hochberg FDR module to url_overdispersion`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `density.py` — beta-binomial one-sided density test

**Verifies:** stats-methodology.AC3.1, stats-methodology.AC3.2, stats-methodology.AC3.3, stats-methodology.AC3.4

**Files:**
- Create: `url_overdispersion/src/url_overdispersion/density.py`
- Test: `url_overdispersion/tests/test_density.py` (unit)

**Implementation:**

Create `density.py` with exactly this content (contract module; duplicated in quote_overdispersion in Phase 3):

```python
# pattern: Functional Core
"""One-sided sharer-density test: beta-binomial with binomial fallback."""

from scipy.stats import betabinom, binom


def density_p_value(
    unique_sharers: int,
    total_shares: int,
    expected_density: float,
    density_variance: float | None,
) -> float:
    """P(U >= unique_sharers) — one-sided, high-density direction.

    Beta-binomial with alpha/beta by method of moments from
    (expected_density, density_variance) when the fit is valid;
    plain binomial(total_shares, expected_density) otherwise.
    expected_density <= 0 or total_shares == 0 -> 1.0.
    """
    if expected_density <= 0 or total_shares == 0:
        return 1.0
    if unique_sharers / total_shares <= expected_density:
        return 1.0
    mu = min(expected_density, 1.0)
    if density_variance is not None and density_variance > 0 and mu < 1.0:
        m = mu * (1.0 - mu) / density_variance - 1.0
        if m > 0:
            return float(betabinom.sf(unique_sharers - 1, total_shares, mu * m, (1.0 - mu) * m))
    return float(binom.sf(unique_sharers - 1, total_shares, mu))
```

Notes:
- The explicit `<= expected_density -> 1.0` guard makes AC3.3 structural rather than probabilistic (mirrors the `z <= 0` guard in the code being replaced).
- The MoM fit `M = μ(1−μ)/σ² − 1, α = μM, β = (1−μ)M` is valid only when `M > 0`, i.e. the observed between-day density variance exceeds nothing and sits below the Bernoulli ceiling `μ(1−μ)`; anything else (σ² absent, zero, or ≥ μ(1−μ)) falls back to plain binomial (AC3.2).
- Known, accepted approximation (document in the docstring of the calling analyzer, not here): the first share of a URL is always by a unique sharer, so U ≥ 1 deterministically; the binomial/beta-binomial model ignores that dependence.

**Testing:**

Create `tests/test_density.py`:
- AC3.1 (hand-verified beta-binomial): `density_p_value(9, 10, 0.5, 0.05)` — MoM gives M = 0.25/0.05 − 1 = 4, α = β = 2; for BetaBinom(n=10, 2, 2), P(X=9) = 10·B(11,3)/B(2,2) = 120/1716 and P(X=10) = B(12,2)/B(2,2) = 66/1716, so P(X ≥ 9) = **186/1716 ≈ 0.108392**. Assert `pytest.approx(186/1716, rel=1e-9)`. Also assert it equals `float(betabinom.sf(8, 10, 2.0, 2.0))` to pin the α/β wiring.
- AC3.2 (binomial fallbacks): `density_p_value(9, 10, 0.5, None) == pytest.approx(float(binom.sf(8, 10, 0.5)))` = 11/1024; variance `0.0` and variance `0.3` (≥ μ(1−μ) = 0.25, M ≤ 0) take the same binomial path.
- AC3.3 (one-sided): `density_p_value(5, 10, 0.5, 0.05) == 1.0` (at baseline) and `density_p_value(3, 10, 0.5, 0.05) == 1.0` (below); also with `expected_density=1.0`.
- AC3.4: `density_p_value(9, 10, 0.0, 0.05) == 1.0`, `density_p_value(9, 10, -0.1, None) == 1.0`, `density_p_value(0, 0, 0.5, 0.05) == 1.0`.
- Fat-tail sanity: for the same inputs, the beta-binomial p-value exceeds the plain binomial one (`density_p_value(9, 10, 0.5, 0.05) > float(binom.sf(8, 10, 0.5))`) — overdispersion makes extremes less surprising, which is the point of the change.

**Verification:**
Run: `cd url_overdispersion && uv run pytest tests/test_density.py`
Expected: all pass.

**Commit:** `feat: add beta-binomial density test module to url_overdispersion`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->
<!-- START_TASK_4 -->
### Task 4: Daily baseline SQL — densified, median-centred, min_sharers on scored rows only

**Verifies:** stats-methodology.AC4.1, stats-methodology.AC4.2, stats-methodology.AC4.4 (daily half)

**Files:**
- Modify: `url_overdispersion/src/url_overdispersion/queries.py:7-69` (`daily_aggregation_query`)
- Test: `url_overdispersion/tests/test_queries.py` (unit)

**Implementation:**

Rewrite `daily_aggregation_query(config)` to this pipeline. Keep the existing `domain_shares` SELECT expressions (domain extraction, `total_shares`, `unique_sharers`, `sample_dids`, `sample_urls`, `Collection`/`OperationKind`/timestamp filters) verbatim inside `raw_shares` — only the CTE name changes and the `HAVING` clause is **removed** there:

```sql
WITH raw_shares AS (
    -- former domain_shares, WITHOUT the HAVING clause
    SELECT ... existing expressions ...
    FROM {config.source_table}
    WHERE ... existing filters ...
        AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
    GROUP BY domain, bucket
),
scored_entities AS (
    SELECT domain
    FROM raw_shares
    WHERE bucket = toDate(now()) AND unique_sharers >= {config.min_sharers}
),
entities AS (
    SELECT r.domain AS domain, min(r.bucket) AS first_seen
    FROM raw_shares r
    INNER JOIN scored_entities s ON r.domain = s.domain
    GROUP BY r.domain
),
calendar AS (
    SELECT toDate(now()) - number AS bucket FROM numbers({config.baseline_days + 1})
),
dense AS (
    SELECT
        e.domain AS domain,
        c.bucket AS bucket,
        coalesce(r.total_shares, 0) AS total_shares,
        coalesce(r.unique_sharers, 0) AS unique_sharers,
        if(coalesce(r.total_shares, 0) > 0, toFloat64(r.unique_sharers) / r.total_shares, NULL) AS sharer_density,
        r.sample_dids AS sample_dids,
        r.sample_urls AS sample_urls
    FROM entities e
    CROSS JOIN calendar c
    LEFT JOIN raw_shares r ON r.domain = e.domain AND r.bucket = c.bucket
    WHERE c.bucket >= e.first_seen
),
baseline AS (
    SELECT
        domain, bucket, total_shares, unique_sharers, sharer_density, sample_dids, sample_urls,
        medianExact(total_shares) OVER w AS rolling_volume_median,
        avg(total_shares) OVER w AS rolling_volume_mean,
        ifNotFinite(varPop(total_shares) OVER w, NULL) AS rolling_volume_variance,
        avg(sharer_density) OVER w AS rolling_density_mean,
        ifNotFinite(varPop(sharer_density) OVER w, NULL) AS rolling_density_variance,
        count() OVER w AS baseline_buckets_available
    FROM dense
    WINDOW w AS (
        PARTITION BY domain
        ORDER BY bucket
        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
    )
),
population_stats AS (
    SELECT
        median(rolling_volume_median) AS population_volume_median,
        median(if(rolling_volume_mean > 0, rolling_volume_variance / rolling_volume_mean, NULL)) AS population_volume_dispersion,
        median(rolling_density_mean) AS population_density_median,
        median(rolling_density_variance) AS population_density_variance
    FROM baseline
    WHERE bucket = toDate(now())
        AND baseline_buckets_available >= {config.cold_start_min_days}
)
SELECT
    b.domain,
    b.bucket AS bucket_start,
    b.total_shares,
    b.unique_sharers,
    coalesce(b.sharer_density, 0) AS sharer_density,
    b.rolling_volume_median,
    b.rolling_volume_mean,
    b.rolling_volume_variance,
    b.rolling_density_mean,
    b.rolling_density_variance,
    toUInt16(b.baseline_buckets_available) AS baseline_days_available,
    b.sample_dids,
    b.sample_urls,
    p.population_volume_median,
    p.population_volume_dispersion,
    p.population_density_median,
    p.population_density_variance
FROM baseline b
CROSS JOIN population_stats p
WHERE b.bucket = toDate(now()) AND b.unique_sharers >= {config.min_sharers}
```

Design decisions to encode in the function docstring:
- **min_sharers moves to scored rows only (AC4.4):** the old `HAVING unique_sharers >= N` filtered *history*, so quiet days vanished from baselines (inflating them) and population medians were computed over an activity-selected subset. Now history is unfiltered; `min_sharers` gates only which of today's rows get scored (the final `WHERE` and the `scored_entities` gate, which also bounds the dense grid — one grid row per scored domain per day, instead of a grid over every domain ever seen).
- **Densification (AC4.1):** zero-share days contribute explicit 0 volume; `sharer_density` is NULL on zero-share days (0/0 is undefined, and treating it as density 0 would poison the density baseline), and ClickHouse `avg`/`varPop` skip NULLs, so density baselines average only over active days while volume baselines include the zeros. `count() OVER w` counts grid days since `first_seen` — the cold-start guard keeps working.
- **Median centre (AC4.2):** `medianExact(total_shares)` is the volume expectation; mean/variance stay for the dispersion factor φ.
- **Population stats (AC4.4):** no `rolling_volume_mean IS NOT NULL` qualifier (dense windows make it non-NULL anyway; the `median(...)` aggregates skip NULL dispersion/density-variance entries individually). Scope note: population medians are computed over today's *scored* entities — the fallback exists to score cold-start rows against their peers in the same scoring batch.

**Testing:**

Update `TestDailyAggregationQuery` (string-containment style):
- AC4.4: `'HAVING' not in query` (the old history filter is gone); `unique_sharers >= 3` appears in `scored_entities` and in the final `WHERE`; `'rolling_volume_mean IS NOT NULL' not in query`.
- AC4.1: `CROSS JOIN`, `numbers(15)` (baseline_days=14 default + 1), `LEFT JOIN raw_shares`, `c.bucket >= e.first_seen`, `coalesce(r.total_shares, 0)`.
- AC4.2: `medianExact(total_shares) OVER w`, `avg(total_shares) OVER w`, `varPop(total_shares) OVER w`, plus density stats `avg(sharer_density) OVER w` and `varPop(sharer_density) OVER w`.
- NULL-density rule: query contains the `if(coalesce(r.total_shares, 0) > 0, ..., NULL)` density expression.
- Keep/adapt existing filter assertions (`Collection = 'app.bsky.feed.post'`, source table interpolation, window frame `ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING`).

**Verification:**
Run: `cd url_overdispersion && uv run pytest tests/test_queries.py -k Daily`
Expected: all pass.

**Commit:** combined with Task 5.
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Hourly baseline SQL — hour-of-day matching

**Verifies:** stats-methodology.AC4.1, stats-methodology.AC4.3

**Files:**
- Modify: `url_overdispersion/src/url_overdispersion/queries.py:72-134` (`hourly_aggregation_query`)
- Test: `url_overdispersion/tests/test_queries.py` (unit)

**Implementation:**

Same pipeline as Task 4 with the hourly differences:
- `raw_shares` buckets by `toStartOfHour(__timestamp)`; `scored_entities` uses `bucket = toStartOfHour(now())`.
- `calendar`: `SELECT toStartOfHour(now()) - toIntervalHour(number) AS bucket FROM numbers({(config.baseline_days + 1) * 24})`.
- Window (AC4.3):

```sql
    WINDOW w AS (
        PARTITION BY domain, toHour(bucket)
        ORDER BY bucket
        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
    )
```

  One bucket per day per partition ⇒ `baseline_buckets_available` is already in days. Delete the `* 24` window scaling, the `intDiv(..., 24)` conversion, and the `* 24` cold-start scaling; guards use `>= {config.cold_start_min_days}` directly. Final filter `WHERE b.bucket = toStartOfHour(now()) AND b.unique_sharers >= {config.min_sharers}`.

Docstring notes the thin-window trade-off: 14 same-hour observations per baseline (default `baseline_days=14`); the lever is `URL_OVERDISPERSION_BASELINE_DAYS` (calibration doc in Phase 7).

**Testing:**

Update `TestHourlyAggregationQuery`:
- AC4.3: `PARTITION BY domain, toHour(bucket)` present; `'336 PRECEDING' not in query` (14×24 artefact gone); `'intDiv' not in query`; `ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING`.
- AC4.1: `numbers(360)` (15 days × 24), `toIntervalHour(number)`, `LEFT JOIN`, `c.bucket >= e.first_seen`.
- `'HAVING' not in query`; cold-start guard un-scaled (`>= 3`).
- Keep/adapt existing assertions (`toStartOfHour(now())`, filters).

**Verification:**
Run: `cd url_overdispersion && uv run pytest tests/test_queries.py`
Expected: all pass.

**Commit:** `feat: densified median baselines, scored-row min_sharers, hour matching in url_overdispersion queries`
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 6-7) -->
<!-- START_TASK_6 -->
### Task 6: Row types and fetch/insert plumbing

**Verifies:** stats-methodology.AC7.1 (column-list side)

**Files:**
- Modify: `url_overdispersion/src/url_overdispersion/db.py:13-26` (`AggregatedRow`), `db.py:29-47` (`ScoredResult`), fetch mapping, insert columns at `db.py:92-110`
- Test: `url_overdispersion/tests/test_db.py` (unit)

**Implementation:**

1. `AggregatedRow` — add five fields and keep the rest: `rolling_volume_median: float | None` (before `rolling_volume_mean`), `rolling_volume_variance: float | None` and `rolling_density_variance: float | None` (after their means), `population_volume_dispersion: float | None` and `population_density_variance: float | None` (after the existing population medians). Update `fetch_aggregated_rows` to the Task 4/5 SELECT order: `domain, bucket_start, total_shares, unique_sharers, sharer_density, rolling_volume_median, rolling_volume_mean, rolling_volume_variance, rolling_density_mean, rolling_density_variance, baseline_days_available, sample_dids, sample_urls, population_volume_median, population_volume_dispersion, population_density_median, population_density_variance`.
2. `ScoredResult` — add `volume_q_value: float` after `volume_p_value`, `density_q_value: float` after `density_p_value`, and four diagnostic fields after `expected_density_lambda`: `rolling_volume_median: float | None`, `rolling_volume_variance: float | None`, `rolling_density_mean: float | None`, `rolling_density_variance: float | None` (the "rolling stats columns" from the design — they make post-deploy calibration queries possible without re-deriving baselines).
3. Insert column list becomes (23 names):

```python
column_names = [
    'run_timestamp', 'granularity', 'domain', 'bucket_start', 'total_shares',
    'unique_sharers', 'sharer_density', 'expected_volume_lambda',
    'expected_density_lambda', 'rolling_volume_median', 'rolling_volume_variance',
    'rolling_density_mean', 'rolling_density_variance',
    'volume_p_value', 'volume_q_value', 'density_p_value', 'density_q_value',
    'is_anomaly', 'baseline_source', 'baseline_days_available',
    'sample_dids', 'sample_urls', 'on_watchlist',
]
```

   with the row-building code updated to match.

**Testing:**

Update `tests/test_db.py`: construction of both dataclasses with new fields (incl. `None` cases), frozen-ness, fetch mapping against a mock row in the new column order (assert median vs mean not transposed), insert asserting the 23-name list with `volume_q_value` directly after `volume_p_value` and `density_q_value` after `density_p_value`.

**Verification:**
Run: `cd url_overdispersion && uv run pytest tests/test_db.py`
Expected: pass (analyzer/main tests red until Task 7 — do not commit yet).

**Commit:** combined with Task 7.
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Analyzer — NB volume + beta-binomial density with per-signal BH

**Verifies:** stats-methodology.AC1.2 (integration), stats-methodology.AC2.2, stats-methodology.AC2.3, stats-methodology.AC3.1 (integration), stats-methodology.AC4.2 (analyzer side)

**Files:**
- Modify: `url_overdispersion/src/url_overdispersion/analyzer.py`
- Test: `url_overdispersion/tests/test_analyzer.py`, `url_overdispersion/tests/test_main.py` (unit)

**Implementation:**

1. Delete `compute_p_value` (12–20) and `compute_density_p_value` (23–42) plus the `scipy.stats` imports. Import `count_p_value` from `.counts`, `density_p_value` from `.density`, `bh_adjust` from `.fdr` (match the sidecar's existing intra-package import style), and `from dataclasses import replace`.

2. `determine_baseline(row, cold_start_min_days) -> tuple[float, float, str]` — entity path when `baseline_days_available >= cold_start_min_days` **and** `rolling_volume_median is not None and rolling_volume_median > 0` **and** `rolling_density_mean is not None and rolling_density_mean > 0` → `(rolling_volume_median, rolling_density_mean, 'entity')`. Population path when both `population_volume_median` and `population_density_median` are present and > 0. Else `(0.0, 0.0, 'population')`. (Volume centre switches from mean to median; the `> 0` median condition routes sparsely-active domains — dense-window median 0 — to the population fallback, same rationale as Phase 1.)

3. New pure helper `determine_variances(row, cold_start_min_days, baseline_source) -> tuple[float | None, float | None]` returning `(volume_variance, density_variance)` to feed the two tests:
   - entity source: `phi = rolling_volume_variance / rolling_volume_mean` when both present and mean > 0; `volume_variance = phi * rolling_volume_median` when `phi > 1`, else `None` (Poisson fallback, AC1.2). `density_variance = rolling_density_variance` (may be `None` → binomial fallback, AC3.2).
   - population source: same construction from `population_volume_dispersion` × the population volume median, and `population_density_variance`.

4. `score_row(...)` computes both raw p-values and returns a provisional `ScoredResult` with `volume_q_value=1.0`, `density_q_value=1.0`, `is_anomaly=0`:

```python
    volume_p_value = count_p_value(row.total_shares, volume_lambda, volume_variance)
    density_p_value_ = density_p_value(row.unique_sharers, row.total_shares, density_lambda, density_variance)
```

   Populate the four diagnostic fields (`rolling_volume_median`, `rolling_volume_variance`, `rolling_density_mean`, `rolling_density_variance`) straight from the row. Keep `on_watchlist` handling in `score_rows` unchanged.

5. `score_rows(...)` — two-pass, **two families per call** (AC2.3: volume and density adjusted separately; daily-vs-hourly separation comes free because `run_cycle` calls once per granularity):

```python
    provisional = [...]  # score_row over rows, with on_watchlist as today
    volume_q = bh_adjust([r.volume_p_value for r in provisional])
    density_q = bh_adjust([r.density_p_value for r in provisional])
    results = []
    for result, vq, dq in zip(provisional, volume_q, density_q):
        is_anomaly = 1 if (vq < config.volume_p_threshold or dq < config.density_p_threshold) else 0
        results.append(replace(result, volume_q_value=vq, density_q_value=dq, is_anomaly=is_anomaly))
    return results
```

   The OR-logic survives; it now operates on q-values (AC2.2). The `observed > 0` guard is inherent here (`min_sharers >= 3` implies `total_shares > 0` on every scored row; `density_p_value` and `count_p_value` also return 1.0 on zero counts by contract).

**Testing:**

Update `tests/test_analyzer.py`:
- `determine_baseline`: median-based entity path; median 0 → population; both paths' `> 0` conditions.
- `determine_variances`: φ > 1 yields `phi * median`; φ ≤ 1 yields `None`; missing variance yields `None`; population source uses population dispersion/variance.
- `score_row`: assert `volume_p_value` equals a direct `count_p_value(...)` call and `density_p_value` a direct `density_p_value(...)` call with the same derived inputs (AC1.2/AC3.1 integration).
- AC2.2/AC2.3 (`score_rows`): a 4-row family where hand-picked p-values make exactly one row's volume q and a different row's density q cross their thresholds — assert per-row `is_anomaly`; assert volume and density q-values were adjusted independently (e.g. identical p-vectors in the two signals yield identical q-vectors, and padding the family with high-p rows changes both consistently).
- Watchlist behaviour regression: unchanged flag semantics.
- Update `tests/test_main.py` `FakeDb` shapes; assert both q columns flow into inserts.

**Verification:**
Run: `cd url_overdispersion && uv run pytest`
Expected: full suite passes.

**Commit:** `feat: NB volume and beta-binomial density scoring with per-signal BH-FDR in url_overdispersion`
<!-- END_TASK_7 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_TASK_8 -->
### Task 8: ClickHouse schema — q-value and rolling-stats columns

**Verifies:** stats-methodology.AC7.1 (url scope)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/03-url-overdispersion.sql`

**Implementation:**

1. In `CREATE TABLE IF NOT EXISTS default.url_overdispersion_results`, add six columns: `rolling_volume_median Nullable(Float64)`, `rolling_volume_variance Nullable(Float64)`, `rolling_density_mean Nullable(Float64)`, `rolling_density_variance Nullable(Float64)` after `expected_density_lambda`; `volume_q_value Float64` after `volume_p_value`; `density_q_value Float64` after `density_p_value`.
2. Append idempotent migrations:

```sql
ALTER TABLE default.url_overdispersion_results ADD COLUMN IF NOT EXISTS rolling_volume_median Nullable(Float64) AFTER expected_density_lambda;
ALTER TABLE default.url_overdispersion_results ADD COLUMN IF NOT EXISTS rolling_volume_variance Nullable(Float64) AFTER rolling_volume_median;
ALTER TABLE default.url_overdispersion_results ADD COLUMN IF NOT EXISTS rolling_density_mean Nullable(Float64) AFTER rolling_volume_variance;
ALTER TABLE default.url_overdispersion_results ADD COLUMN IF NOT EXISTS rolling_density_variance Nullable(Float64) AFTER rolling_density_mean;
ALTER TABLE default.url_overdispersion_results ADD COLUMN IF NOT EXISTS volume_q_value Float64 AFTER volume_p_value;
ALTER TABLE default.url_overdispersion_results ADD COLUMN IF NOT EXISTS density_q_value Float64 AFTER density_p_value;
```

Engine/ORDER BY untouched. Cross-check every Task 6 insert column name appears in the CREATE TABLE.

**Verification:**
Run: `grep -c 'ADD COLUMN IF NOT EXISTS' /Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/03-url-overdispersion.sql`
Expected: `6`.

**Commit** (in skywatch-osprey): `Add q-value and rolling-stat columns to url_overdispersion_results`
<!-- END_TASK_8 -->

<!-- START_TASK_9 -->
### Task 9: Documentation updates

**Verifies:** None (documentation; AC8.2 finalized in Phase 7)

**Files:**
- Modify: `url_overdispersion/README.md`
- Modify: `url_overdispersion/CLAUDE.md`

**Implementation:**

Update both to the shipped behaviour: NB volume test (MoM from rolling median + dispersion factor, Poisson fallback), beta-binomial density test (MoM α/β from rolling density mean/variance, binomial fallback, one-sided high), dense zero-filled baselines with `min_sharers` applied only to scored rows, hour-of-day-matched hourly windows, per-cycle per-signal BH with `is_anomaly = volume_q < target OR density_q < target`, threshold env vars unchanged in name but now FDR targets, new output columns. Note the accepted approximation (first-share-is-unique dependence ignored) per the design.

**Verification:**
Run: `grep -n 'normal-approximation\|z-test' url_overdispersion/README.md url_overdispersion/CLAUDE.md`
Expected: no stale references to the replaced z-test method.

**Commit:** `docs: update url_overdispersion README and CLAUDE.md for NB + beta-binomial + FDR`
<!-- END_TASK_9 -->

<!-- START_TASK_10 -->
### Task 10: Full suite gate

**Verifies:** stats-methodology.AC9.1 (url_overdispersion scope)

**Files:** none (verification only)

**Step 1:** Run: `cd url_overdispersion && uv run pytest` — Expected: all pass.

**Step 2:** Run: `cd url_overdispersion && uv run ruff check src tests && uv run ruff format --check src tests` — Expected: clean.

**Step 3:** Run: `git status --short` in both repos — Expected: clean trees; Task 8's commit exists in skywatch-osprey.
<!-- END_TASK_10 -->

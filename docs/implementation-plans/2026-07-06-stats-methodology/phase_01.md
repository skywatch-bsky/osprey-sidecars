# Statistical Methodology Fixes — Phase 1: signup_anomaly Implementation Plan

**Goal:** Dispersion-aware NB/Poisson count test, dense hour-of-day-matched median baselines, and Benjamini–Hochberg FDR control for the PDS signup-anomaly sidecar.

**Architecture:** Two new Functional Core modules (`counts.py`, `fdr.py` — duplicated per sidecar by convention, no shared imports), a rewritten baseline SQL generator in `queries.py`, a two-pass scorer in `analyzer.py` (raw p-values, then per-cycle BH adjustment), and mechanical column additions in `db.py` and the ClickHouse schema.

**Tech Stack:** Python 3.11+, scipy ≥ 1.15 (`nbinom`, `poisson`), clickhouse-connect, ClickHouse window functions (`medianExact`, `avg`, `varPop` over `ROWS BETWEEN`), pytest via `uv run pytest`.

**Scope:** Phase 1 of 7 from `docs/design-plans/2026-07-06-stats-methodology.md` (independent of phases 2–6).

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
- **stats-methodology.AC2.3 Success:** Daily and hourly rows in one cycle are adjusted as separate families; volume and density p-values as separate families — *signup scope: there is a single (volume) signal, so the family split here is daily vs. hourly*
- **stats-methodology.AC2.4 Edge:** Empty input → empty output; single p-value → q = p

### stats-methodology.AC4: Baselines
- **stats-methodology.AC4.1 Success:** Generated SQL densifies entity×bucket grids — a bucket with zero events contributes 0 to the rolling window
- **stats-methodology.AC4.2 Success:** Expected count = rolling median; rolling mean/variance still emitted for the dispersion factor
- **stats-methodology.AC4.3 Success:** Hourly baseline windows partition by (entity, hour-of-day)
- **stats-methodology.AC4.4 Success:** `min_sharers`-type filters apply only to scored rows; population medians computed without the `dispersion_index IS NOT NULL` / `rolling_mean >= 1` qualifiers — *signup scope: no `min_sharers` filter exists here; the qualifier removal applies*

### stats-methodology.AC7: Schemas
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar) — *signup scope: `02-signup-anomalies.sql`*

### stats-methodology.AC9: Suites
- **stats-methodology.AC9.1 Success:** `uv run pytest` passes in all six sidecars — *signup_anomaly scope*

---

## Context from codebase verification

Current state (all paths verified 2026-07-06):

- `signup_anomaly/src/signup_anomaly/analyzer.py` (90 lines): `compute_p_value(observed, expected_lambda)` at lines 12–15 uses `poisson.sf(observed - 1, expected_lambda)`; `determine_baseline` (18–28) prefers entity `rolling_mean` when `baseline_days_available >= cold_start_min_days`, else `population_median_lambda`; `determine_dispersion` (31–41) analogous; `score_row` (44–81) picks threshold by granularity, **overrides to the daily threshold when `baseline_source == 'population'`** (line 58), sets `is_anomaly = p_value < threshold and observed_count > 0`; `score_rows` (84–90) maps.
- `queries.py` (185 lines): `daily_aggregation_query` (7–90) builds CTEs `daily_counts` → `baseline` → `population_stats`. Baseline uses `avg(signup_count) OVER (PARTITION BY pds_host ORDER BY day ROWS BETWEEN {baseline_days} PRECEDING AND 1 PRECEDING)`; **no densification** (days with zero signups are absent from the window); `dispersion_index` is NULLed when `rolling_mean < 1.0`; `population_stats` filters `dispersion_index IS NOT NULL AND baseline_days_available >= {cold_start_min_days}` — both biases this phase removes. `hourly_aggregation_query` (93–176) uses a continuous `{baseline_days * 24}`-hour window with **no hour-of-day matching**, converts hours back to days via `intDiv(..., 24)`. `_build_exclusion_clause` (179–185) renders host exclusions (`PdsHost NOT LIKE '%bsky.network'`, `!= 'bridgy-fed.appspot.com'`, `!= 'mostr.pub'`); source filter is `ActionName = 'identity'` with `PdsHost IS NOT NULL`.
- `db.py` (117 lines): frozen `AggregatedRow` (13–24: pds_host, observed_count, distinct_accounts, rolling_mean, baseline_days_available, sample_dids, population_median_lambda, rolling_variance, dispersion_index, population_dispersion_index) and `ScoredResult` (27–42); insert column list at 79–94 (14 columns ending `dispersion_index`); output table `pds_signup_anomalies`.
- `config.py`: `AnalysisConfig` fields `daily_p_value_threshold` (env `SIGNUP_ANOMALY_DAILY_P_THRESHOLD`, 0.01), `hourly_p_value_threshold` (env `SIGNUP_ANOMALY_HOURLY_P_THRESHOLD`, 0.05), `baseline_days` (7), `cold_start_min_days` (3), `excluded_hosts`, `source_table`, `output_table`. **Env var names are kept, reinterpreted as FDR targets** per the design.
- Schema: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/02-signup-anomalies.sql`, table `default.pds_signup_anomalies`, MergeTree `ORDER BY (run_timestamp, granularity, pds_host)`.
- `pyproject.toml`: `scipy>=1.15.0` already present. Tests run with `cd signup_anomaly && uv run pytest`; SQL tests assert substrings of generated queries; `FakeDb` stub in `test_main.py`. Ruff: single quotes, 120 cols.

External-dependency findings (internet-researcher, 2026-07-06):
- `scipy.stats.nbinom.sf(k, n, p)` supports real-valued `n` (r); the survival function is computed via the regularized incomplete beta: `P(X ≥ observed) = I_{1−p}(observed, r) = scipy.special.betainc(observed, r, 1 − p)`. scipy issue #16120 concerns non-integer inputs being mishandled in some versions; the guard is a regression test asserting `nbinom.sf` agrees with the `betainc` identity. Do **not** use `scipy.special.nbdtrc` as the cross-check or fallback — it truncates a non-integer shape parameter to an integer (4.5 → 4, with only a RuntimeWarning), which is precisely the failure mode the guard exists to catch. If the pinned scipy ever fails the guard, reimplement the NB branch as `float(betainc(observed, r, 1.0 - p))` (keep `observed` an int).

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: `counts.py` — dispersion-aware count test

**Verifies:** stats-methodology.AC1.1, stats-methodology.AC1.2, stats-methodology.AC1.3, stats-methodology.AC1.4

**Files:**
- Create: `signup_anomaly/src/signup_anomaly/counts.py`
- Test: `signup_anomaly/tests/test_counts.py` (unit)

**Implementation:**

Create `counts.py` exactly as follows (this module is the cross-sidecar contract from the design; phases 2 and 3 carry byte-identical copies under their own packages):

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

Behavioural notes the tests pin down:
- `observed = 0` gives `sf(-1, ...) = 1.0` in both branches — a zero count can never look anomalous.
- Callers pass `mean` = rolling **median** and `variance = phi * median` where `phi = rolling_variance / rolling_mean` (the dispersion factor), so the model is NB(mean=median, variance=phi·median). That wiring happens in Task 6; this module stays a pure two-moment interface.

**Testing:**

Create `tests/test_counts.py` following the existing class-per-function style:

- AC1.1 (hand-computed NB example): `count_p_value(3, 2.0, 4.0)` → MoM gives r = 4/2 = 2, p = 2/4 = 0.5. For NB(r=2, p=0.5), P(X=k) = (k+1)·0.25·0.5^k, so P(X≥3) = 1 − (0.25 + 0.25 + 0.1875) = **0.3125** exactly. Assert `pytest.approx(0.3125, abs=1e-12)`.
- AC1.2 (Poisson fallback, variance ≤ mean): `count_p_value(5, 3.0, 2.0) == pytest.approx(float(poisson.sf(4, 3.0)))`; equality case `variance == mean` also falls back; and `count_p_value(5, 3.0, None)` (variance unavailable) equals the same Poisson value.
- AC1.3 (non-integer r regression guard for scipy#16120): with `mean=3.0, variance=5.0` (r = 9/2 = 4.5, p = 0.6) and `observed=7`, assert the result is in `(0.0, 1.0]` **and** equals `float(scipy.special.betainc(7, 4.5, 0.4))` to within `abs=1e-10` — the exact incomplete-beta identity `P(X ≥ observed) = betainc(observed, r, 1 − p)`, which honours real-valued r (≈ 0.07519 here). Do NOT cross-check with `scipy.special.nbdtrc`: it truncates non-integer r to an integer (4.5 → 4, giving 0.05476) and would fail this test against a correct `nbinom.sf`. If a future scipy pin breaks the guard, reimplement `count_p_value`'s NB branch as `float(betainc(observed, r, 1.0 - p))` with `observed` kept as an int — documented fallback per the design.
- AC1.4: `count_p_value(5, 0.0, 4.0) == 1.0`, `count_p_value(5, -1.0, None) == 1.0`, and `count_p_value(0, 10.0, 20.0) == 1.0` (observed 0 never flags in either branch: also assert `count_p_value(0, 10.0, None) == 1.0`).
- Sanity: p-value decreases as observed increases (`count_p_value(20, 5.0, 10.0) < count_p_value(10, 5.0, 10.0)`), and the NB branch yields a **larger** p than Poisson at the same mean for overdispersed baselines (`count_p_value(15, 5.0, 15.0) > float(poisson.sf(14, 5.0))`) — the entire point of the change.

**Verification:**
Run: `cd signup_anomaly && uv run pytest tests/test_counts.py`
Expected: all pass.

**Commit:** `feat: add NB/Poisson dispersion-aware count test module to signup_anomaly`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `fdr.py` — Benjamini–Hochberg adjustment

**Verifies:** stats-methodology.AC2.1, stats-methodology.AC2.4

**Files:**
- Create: `signup_anomaly/src/signup_anomaly/fdr.py`
- Test: `signup_anomaly/tests/test_fdr.py` (unit)

**Implementation:**

Create `fdr.py` exactly as follows (cross-sidecar contract; duplicated in phases 2–3):

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

Create `tests/test_fdr.py`:

- AC2.1 (hand-computed example): `bh_adjust([0.01, 0.04, 0.03, 0.005]) == [0.02, 0.04, 0.04, 0.02]` (exact: sorted p = 0.005, 0.01, 0.03, 0.04 with n=4 give raw 0.02, 0.02, 0.04, 0.04; cumulative min from the top changes nothing; mapped back to input positions). Assert element-wise with `pytest.approx`.
- AC2.1 (monotone): for that example, q-values ordered by ascending p are non-decreasing. Add a second, messier case (e.g. `[0.9, 0.001, 0.5, 0.02, 0.02]`) asserting the same monotonicity property and that every q ≥ its p.
- AC2.1 (order preserved): `bh_adjust(ps)[i]` corresponds to `ps[i]` — verify by comparing against adjusting `sorted(ps)` and un-sorting.
- AC2.4: `bh_adjust([]) == []`; `bh_adjust([0.03]) == [0.03]` (single p → q = p).
- Cap: `bh_adjust([0.8, 0.9])` produces values ≤ 1.0.
- Ties: `bh_adjust([0.05, 0.05])` → both `0.05` (tied p-values share the higher rank's adjustment after the cumulative min).

**Verification:**
Run: `cd signup_anomaly && uv run pytest tests/test_fdr.py`
Expected: all pass.

**Commit:** `feat: add Benjamini-Hochberg FDR module to signup_anomaly`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Daily baseline SQL — densified, median-centred, unbiased population stats

**Verifies:** stats-methodology.AC4.1, stats-methodology.AC4.2, stats-methodology.AC4.4 (daily half)

**Files:**
- Modify: `signup_anomaly/src/signup_anomaly/queries.py:7-90` (`daily_aggregation_query`)
- Test: `signup_anomaly/tests/test_queries.py` (unit, SQL-structure assertions)

**Implementation:**

Rewrite `daily_aggregation_query(config)` to produce this CTE pipeline (keep the function pure, f-string interpolation of config values only, and keep `_build_exclusion_clause` usage and all existing source filters — `ActionName = 'identity'`, `PdsHost IS NOT NULL`, host exclusions):

```sql
WITH raw_counts AS (
    SELECT
        PdsHost AS pds_host,
        toDate(__timestamp) AS day,
        count() AS signup_count,
        countDistinct(UserId) AS distinct_accounts,
        arraySlice(groupArray(UserId), 1, 5) AS sample_dids
    FROM {config.source_table}
    WHERE ActionName = 'identity'
        AND PdsHost IS NOT NULL
        {exclusions}
        AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
    GROUP BY pds_host, day
),
hosts AS (
    SELECT pds_host, min(day) AS first_seen
    FROM raw_counts
    GROUP BY pds_host
),
calendar AS (
    SELECT toDate(now()) - number AS day
    FROM numbers({config.baseline_days + 1})
),
dense AS (
    SELECT
        h.pds_host AS pds_host,
        c.day AS day,
        coalesce(r.signup_count, 0) AS signup_count,
        coalesce(r.distinct_accounts, 0) AS distinct_accounts,
        r.sample_dids AS sample_dids
    FROM hosts h
    CROSS JOIN calendar c
    LEFT JOIN raw_counts r ON r.pds_host = h.pds_host AND r.day = c.day
    WHERE c.day >= h.first_seen
),
baseline AS (
    SELECT
        pds_host,
        day,
        signup_count,
        distinct_accounts,
        sample_dids,
        medianExact(signup_count) OVER w AS rolling_median,
        avg(signup_count) OVER w AS rolling_mean,
        ifNotFinite(varPop(signup_count) OVER w, NULL) AS rolling_variance,
        count() OVER w AS baseline_days_available
    FROM dense
    WINDOW w AS (
        PARTITION BY pds_host
        ORDER BY day
        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
    )
),
population_stats AS (
    SELECT
        median(rolling_median) AS population_median_lambda,
        median(if(rolling_mean > 0, rolling_variance / rolling_mean, NULL)) AS population_dispersion_index
    FROM baseline
    WHERE day = toDate(now())
        AND baseline_days_available >= {config.cold_start_min_days}
)
SELECT
    b.pds_host,
    b.signup_count AS observed_count,
    b.distinct_accounts,
    b.rolling_median,
    b.rolling_mean,
    b.rolling_variance,
    if(b.rolling_mean > 0, b.rolling_variance / b.rolling_mean, NULL) AS dispersion_index,
    b.baseline_days_available,
    b.sample_dids,
    p.population_median_lambda,
    p.population_dispersion_index
FROM baseline b
CROSS JOIN population_stats p
WHERE b.day = toDate(now())
```

Design decisions encoded here (carry these into the docstring):
- **Densification (AC4.1):** every host seen in the lookback gets one row per calendar day from its `first_seen` forward; a quiet day contributes an explicit 0 to the rolling window instead of silently vanishing. The `c.day >= h.first_seen` bound stops densification from back-filling zeros before a host ever existed, which would both bias its baseline low and defeat the cold-start guard (with an unconditional grid, `count() OVER w` would always equal `baseline_days`). A host with no activity in the whole lookback simply isn't scored — same as today.
- **Median centre (AC4.2):** `medianExact` (ClickHouse's exact quantile, valid as a window aggregate) is the new expected count; `avg`/`varPop` remain solely to form the dispersion factor φ = variance/mean and diagnostics.
- **Qualifier removal (AC4.4):** `population_stats` no longer filters `dispersion_index IS NOT NULL`, and the `rolling_mean >= 1.0` NULL-guard on `dispersion_index` is reduced to the mathematically required `rolling_mean > 0` (aggregate `median(...)` skips NULLs, so hosts with a zero-mean window drop out of the dispersion median only, not the λ median). The `baseline_days_available >= cold_start_min_days` guard stays — a population median over cold-start hosts would be circular.
- The column named `dispersion_index` keeps its name for schema compatibility; documentation-wise it is the **dispersion factor** (variance-to-mean ratio), clarified in Task 8 and Phase 7 docs.
- ClickHouse note: with default `join_use_nulls = 0`, the LEFT JOIN fills unmatched numeric columns with 0 and arrays with `[]`; the `coalesce` calls are belt-and-braces so the query stays correct under `join_use_nulls = 1` sessions. `r.sample_dids` needs no wrapper (`[]` default is the desired value for zero-signup days).

**Testing:**

Update `TestDailyAggregationQuery` in `tests/test_queries.py` (string-containment style, matching existing tests):
- AC4.1: query contains `CROSS JOIN calendar` (or `CROSS JOIN` plus `numbers({baseline_days + 1})` rendered with the config value, e.g. `numbers(8)` for `baseline_days=7`), `LEFT JOIN raw_counts`, `coalesce(r.signup_count, 0)`, and `c.day >= h.first_seen`.
- AC4.2: query contains `medianExact(signup_count) OVER w`, and still contains `avg(signup_count) OVER w` and `varPop(signup_count) OVER w`.
- AC4.4: assert `'dispersion_index IS NOT NULL' not in query` and `'>= 1.0' not in query`; assert `median(rolling_median)` present; assert the cold-start guard `baseline_days_available >= 3` (default config) is present in `population_stats`.
- Regressions: existing assertions for `ActionName = 'identity'`, host exclusions, source-table interpolation, and `ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING` must be kept/adapted.
- Remove assertions that pinned the old CTE names/filters that no longer exist (`daily_counts`, mean-guard CASE).

**Verification:**
Run: `cd signup_anomaly && uv run pytest tests/test_queries.py -k Daily`
Expected: all pass. (`test_db.py`/`test_main.py` remain green — the fetch layer changes in Task 5.)

**Commit:** combined with Task 4 (`feat: densified median baselines for signup_anomaly queries`) — the two query builders share test fixtures and the row shape changes land together in Task 5.
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Hourly baseline SQL — hour-of-day matching

**Verifies:** stats-methodology.AC4.1, stats-methodology.AC4.3

**Files:**
- Modify: `signup_anomaly/src/signup_anomaly/queries.py:93-176` (`hourly_aggregation_query`)
- Test: `signup_anomaly/tests/test_queries.py` (unit)

**Implementation:**

Rewrite `hourly_aggregation_query(config)` with the same pipeline as Task 3, with these differences:

- `raw_counts` buckets by `toStartOfHour(__timestamp) AS bucket` over `INTERVAL {config.baseline_days + 1} DAY`.
- `calendar` generates hours: `SELECT toStartOfHour(now()) - toIntervalHour(number) AS bucket FROM numbers({(config.baseline_days + 1) * 24})`.
- `dense` joins hosts × hourly calendar with `c.bucket >= h.first_seen` (where `first_seen` is `min(bucket)`).
- **Hour-of-day matching (AC4.3):** the window becomes

```sql
    WINDOW w AS (
        PARTITION BY pds_host, toHour(bucket)
        ORDER BY bucket
        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
    )
```

  Each partition holds exactly one bucket per day (the same wall-clock hour on successive days), so `ROWS BETWEEN {baseline_days} PRECEDING AND 1 PRECEDING` means "the last N same-hour observations". Consequently `count() OVER w AS baseline_days_available` is **already in days** — delete the old `* 24` scaling on the window, the `intDiv(..., 24)` conversion, and the `* 24` on the cold-start guard. `population_stats` and the final SELECT use `baseline_days_available >= {config.cold_start_min_days}` and `WHERE b.bucket = toStartOfHour(now())` respectively.
- Final SELECT emits `baseline_days_available` directly (type is UInt64 from `count()`; wrap as `toUInt16(b.baseline_days_available)` to keep the existing UInt16 contract).

Document the thin-baseline trade-off in the docstring: with `baseline_days = 7` there are only 7 same-hour observations per window; raising `SIGNUP_ANOMALY_BASELINE_DAYS` is the tuning lever (calibration guidance lands in Phase 7).

**Testing:**

Update `TestHourlyAggregationQuery`:
- AC4.3: query contains `PARTITION BY pds_host, toHour(bucket)`; assert `'168 PRECEDING' not in query` and `'intDiv' not in query` (old continuous-window artefacts gone); assert `ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING`.
- AC4.1: hourly calendar present — `numbers(192)` for default config (8 days × 24), `toIntervalHour(number)`, `LEFT JOIN`, `coalesce`.
- Cold-start guard uses the un-scaled day count: `baseline_days_available >= 3`.
- Keep/adapt existing assertions (filters, exclusions, `toStartOfHour(now())`).

**Verification:**
Run: `cd signup_anomaly && uv run pytest tests/test_queries.py`
Expected: all pass.

**Commit:** `feat: densified median baselines with hour-of-day matching for signup_anomaly queries`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 5-6) -->
<!-- START_TASK_5 -->
### Task 5: Row types and fetch/insert plumbing

**Verifies:** stats-methodology.AC7.1 (column-list side; schema file itself is Task 7)

**Files:**
- Modify: `signup_anomaly/src/signup_anomaly/db.py:13-24` (`AggregatedRow`), `db.py:27-42` (`ScoredResult`), fetch mapping, insert columns at `db.py:79-94`
- Test: `signup_anomaly/tests/test_db.py` (unit)

**Implementation:**

1. `AggregatedRow` gains `rolling_median: float | None` (place after `rolling_mean` — wait, current field order is pds_host, observed_count, distinct_accounts, rolling_mean, baseline_days_available, sample_dids, population_median_lambda, rolling_variance, dispersion_index, population_dispersion_index; insert `rolling_median` immediately before `rolling_mean` to mirror the new SELECT order). Update `fetch_aggregated_rows` to map the Task 3/4 SELECT column order: `pds_host, observed_count, distinct_accounts, rolling_median, rolling_mean, rolling_variance, dispersion_index, baseline_days_available, sample_dids, population_median_lambda, population_dispersion_index`.
2. `ScoredResult` gains `q_value: float` immediately after `p_value`.
3. Insert column list adds `'q_value'` after `'p_value'` (15 columns total), and the row-building code adds the value in matching position.

**Testing:**

Update `tests/test_db.py`:
- `AggregatedRow` construction with `rolling_median` (and `None` case).
- `ScoredResult` construction with `q_value`; frozen-ness still asserted.
- Fetch test: mock client returns a row tuple in the new column order; assert field mapping (especially `rolling_median` vs `rolling_mean` not swapped).
- Insert test: asserts the 15-name column list, `'q_value'` present directly after `'p_value'`.

**Verification:**
Run: `cd signup_anomaly && uv run pytest tests/test_db.py`
Expected: pass (analyzer/main tests break until Task 6 — expected mid-subcomponent; do not commit yet).

**Commit:** combined with Task 6.
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Analyzer — median-centred NB scoring and per-cycle BH

**Verifies:** stats-methodology.AC1.2 (integration), stats-methodology.AC2.2, stats-methodology.AC2.3, stats-methodology.AC4.2 (analyzer side)

**Files:**
- Modify: `signup_anomaly/src/signup_anomaly/analyzer.py` (whole scoring path)
- Test: `signup_anomaly/tests/test_analyzer.py`, `signup_anomaly/tests/test_main.py` (unit)

**Implementation:**

1. **Delete** the local `compute_p_value` (lines 12–15) and the `from scipy.stats import poisson` import. Import instead:

```python
from signup_anomaly.counts import count_p_value
from signup_anomaly.fdr import bh_adjust
```

   (Match the import style already used between sidecar modules — relative or absolute exactly as the existing `from signup_anomaly.db import ...` lines do.)

2. **`determine_baseline(row, cold_start_min_days) -> tuple[float, str]`** — same shape, new centre: use the entity's `rolling_median` when `row.baseline_days_available >= cold_start_min_days and row.rolling_median is not None and row.rolling_median > 0`; else `population_median_lambda` when present and > 0 (source `'population'`); else `(0.0, 'population')`. The `> 0` condition matters: with dense zero-filled windows a host active on fewer than half its window days has median 0, and a zero centre must route to the population fallback rather than making every count trivially "infinite-fold above baseline" or (worse) unfloggable — `count_p_value` with `mean <= 0` returns 1.0 by contract.

3. **`determine_dispersion(row, cold_start_min_days) -> float | None`** — unchanged precedence logic (entity `dispersion_index` given history, else `population_dispersion_index`, else `None`); it now receives the SQL-computed φ from the dense window.

4. **`score_row(...)`** — compute:

```python
    expected_lambda, baseline_source = determine_baseline(row, config.cold_start_min_days)
    phi = determine_dispersion(row, config.cold_start_min_days)
    variance = phi * expected_lambda if (phi is not None and phi > 1.0 and expected_lambda > 0) else None
    p_value = count_p_value(row.observed_count, expected_lambda, variance)
```

   `score_row` no longer decides `is_anomaly` (that needs the whole family); it returns a `ScoredResult` with `q_value=1.0` and `is_anomaly=0` placeholders. Keep the threshold-selection logic out of `score_row` entirely.

5. **`score_rows(rows, config, granularity, run_timestamp)`** — the BH family boundary (AC2.3). Each call is one family: `run_cycle` in `main.py` already invokes it once for daily and once for hourly per cycle, so daily and hourly are adjusted separately with no `main.py` change:

```python
def score_rows(rows, config, granularity, run_timestamp):
    provisional = [score_row(row, config, granularity, run_timestamp) for row in rows]
    q_values = bh_adjust([r.p_value for r in provisional])
    results = []
    for result, q_value in zip(provisional, q_values):
        threshold = (
            config.daily_p_value_threshold
            if granularity == 'daily' or result.baseline_source == 'population'
            else config.hourly_p_value_threshold
        )
        is_anomaly = 1 if (q_value < threshold and result.observed_count > 0) else 0
        results.append(replace(result, q_value=q_value, is_anomaly=is_anomaly))
    return results
```

   (`from dataclasses import replace`.) The population-source stricter-threshold behaviour (currently `score_row` line 58) is preserved, now expressed as a per-row FDR target. Threshold env vars keep their names and defaults; they are FDR targets from this commit on.

**Testing:**

Update `tests/test_analyzer.py`:
- `determine_baseline`: entity median preferred when history sufficient and median > 0; median == 0 routes to population; median None routes to population; all-missing → (0.0, 'population').
- `score_row`: overdispersed row (φ > 1) produces `p_value == count_p_value(observed, median, phi * median)` — assert by calling `count_p_value` directly with the same numbers; φ ≤ 1 or None produces the Poisson path (AC1.2 integration: equals `count_p_value(observed, median, None)`).
- AC2.2 (`score_rows`): build 4 rows with known p-values by choosing observed counts against a fixed baseline; assert `is_anomaly == 1` exactly for rows whose hand-computed q < threshold and `observed_count > 0`, and that a row with `observed_count == 0` is never flagged even with tiny q.
- AC2.3 (family separation): the same row list passed through two `score_rows` calls (daily vs hourly) gets q-values computed only from its own call's p-values — e.g. assert q-values from a 2-row call differ from the q-values the same p-values receive inside a 10-row call padded with high-p rows (n changes the correction).
- Population-source threshold override still applies (hourly granularity + population source compares q to the daily target).
- Update `tests/test_main.py` `FakeDb` flows for the new row shapes; assert `q_value` flows into inserts and that daily and hourly cycles insert independently-adjusted values.

**Verification:**
Run: `cd signup_anomaly && uv run pytest`
Expected: full suite passes.

**Commit:** `feat: NB scoring on median baselines with per-cycle BH-FDR in signup_anomaly`
<!-- END_TASK_6 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_TASK_7 -->
### Task 7: ClickHouse schema — `q_value` column

**Verifies:** stats-methodology.AC7.1 (signup scope)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/02-signup-anomalies.sql`

**Implementation:**

In the sibling repo checkout at `/Users/scarndp/dev/skywatch/skywatch-osprey`:

1. Add `q_value Float64` to the `CREATE TABLE IF NOT EXISTS default.pds_signup_anomalies` column list, directly after `p_value Float64`.
2. Append an idempotent migration for live deployments:

```sql
ALTER TABLE default.pds_signup_anomalies ADD COLUMN IF NOT EXISTS q_value Float64 AFTER p_value;
```

Engine, ORDER BY, and all other columns unchanged. The sidecar (Task 5) names all 15 insert columns explicitly, so it fails fast against an unmigrated table; sidecar and schema deploy together per the design's migration note.

**Verification:**
Run: `grep -n 'q_value' /Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/02-signup-anomalies.sql`
Expected: two matches (CREATE TABLE line and ALTER line). Cross-check every name in the Task 5 insert column list appears in the CREATE TABLE.

**Commit** (in skywatch-osprey): `Add q_value column to pds_signup_anomalies`
<!-- END_TASK_7 -->

<!-- START_TASK_8 -->
### Task 8: Documentation updates

**Verifies:** None (documentation; AC8.2 finalized in Phase 7)

**Files:**
- Modify: `signup_anomaly/README.md`
- Modify: `signup_anomaly/CLAUDE.md`

**Implementation:**

Update both to the shipped behaviour:
- Method: negative binomial (method-of-moments from rolling median + dispersion factor) with Poisson fallback when the window is not overdispersed; expected count is the rolling **median** over a dense, zero-filled window; hourly baselines compare like-hour-to-like-hour.
- FDR: raw p-values are BH-adjusted per cycle per granularity; `is_anomaly` means `q_value < threshold`; `SIGNUP_ANOMALY_DAILY_P_THRESHOLD` / `SIGNUP_ANOMALY_HOURLY_P_THRESHOLD` are FDR targets (names unchanged).
- Terminology: the `dispersion_index` column is the variance-to-mean **dispersion factor** of the baseline window (a Poisson-fit diagnostic), not "overdispersion" of the test statistic — the naming clarification called out in the design's minor items.
- New `q_value` output column documented.

**Verification:**
Run: `grep -n 'Poisson statistics' signup_anomaly/README.md signup_anomaly/CLAUDE.md`
Expected: no stale "pure Poisson" description remains.

**Commit:** `docs: update signup_anomaly README and CLAUDE.md for NB + FDR methodology`
<!-- END_TASK_8 -->

<!-- START_TASK_9 -->
### Task 9: Full suite gate

**Verifies:** stats-methodology.AC9.1 (signup_anomaly scope)

**Files:** none (verification only)

**Step 1: Full test suite**

Run: `cd signup_anomaly && uv run pytest`
Expected: all tests pass, zero failures/errors.

**Step 2: Lint**

Run: `cd signup_anomaly && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: clean.

**Step 3: Working trees committed**

Run: `git status --short` in `osprey-sidecars` and `skywatch-osprey`.
Expected: clean; Task 7's schema commit exists in skywatch-osprey.
<!-- END_TASK_9 -->

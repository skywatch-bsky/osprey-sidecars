# Statistical Methodology Fixes — Phase 3: quote_overdispersion Implementation Plan

**Goal:** Mirror of Phase 2 for the quote sidecar: NB volume test, beta-binomial density test, dense median baselines, hour-of-day matching, per-signal BH-FDR. Entity is the quoted-post AT-URI; AT-URI extraction is unchanged.

**Architecture:** Three new Functional Core modules (`counts.py`, `fdr.py`, `density.py`) duplicated byte-identically from the cross-sidecar contract under the `quote_overdispersion` package, rewritten baseline SQL, two-pass scorer, column additions in `db.py` + `07-quote-overdispersion.sql`.

**Tech Stack:** Python 3.11+, scipy ≥ 1.15 (`nbinom`, `poisson`, `betabinom`, `binom`), clickhouse-connect, ClickHouse window functions, pytest via `uv run pytest`.

**Scope:** Phase 3 of 7 from `docs/design-plans/2026-07-06-stats-methodology.md` (independent of phases 1–2, 4–6).

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
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar) — *quote scope: `07-quote-overdispersion.sql`*

### stats-methodology.AC9: Suites
- **stats-methodology.AC9.1 Success:** `uv run pytest` passes in all six sidecars — *quote_overdispersion scope*

---

## Context from codebase verification

quote_overdispersion is a structural mirror of url_overdispersion with these verified divergences (2026-07-06):

- Entity extraction (keep verbatim): `if(PostEmbedRecordUri != '', PostEmbedRecordUri, PostEmbedRecordWithMediaUri) AS quoted_uri` at `queries.py:12` (daily) and `:74` (hourly), with source filter `(PostEmbedRecordUri != '' OR PostEmbedRecordWithMediaUri != '')`, `Collection = 'app.bsky.feed.post'`, `OperationKind = 'create'`.
- `analyzer.py` (168 lines): `extract_quoted_author_did(quoted_uri)` at line 12 (keep unchanged); `compute_p_value` at 41 (Poisson sf — replace); `compute_density_p_value` at 51 (normal z-test — replace); `determine_baseline` at 72; `score_row` at 103 with OR-logic `is_anomaly` at 128–130; `score_rows` at 152 (no watchlist parameter, unlike url).
- `queries.py` (129 lines): `daily_aggregation_query` (7–66) with `HAVING unique_sharers >= {min_sharers}` on history (line 23) and `population_stats` qualified by `rolling_volume_mean IS NOT NULL` (line 47); `hourly_aggregation_query` (69–128) with continuous `{baseline_days * 24}` window, `intDiv(..., 24)`, no hour-of-day matching. CTEs are named `domain_shares`/`baseline`/`population_stats` even though the entity is `quoted_uri`.
- `db.py` (132 lines): `AggregatedRow` (13–25: quoted_uri, bucket_start, total_shares, unique_sharers, sharer_density, rolling_volume_mean, rolling_density_mean, baseline_days_available, sample_dids, population_volume_median, population_density_median — **no sample_urls**); `ScoredResult` (28–45, includes `quoted_author_did`, **no on_watchlist**); insert column list at 89–106 (16 names).
- `config.py`: env prefix `QUOTE_OVERDISPERSION_*`; `volume_p_threshold` 0.01, `density_p_threshold` 0.01, `baseline_days` **14**, `cold_start_min_days` 3, `min_sharers` 3; **no watchlist field**; output table `quote_overdispersion_results`.
- Schema: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/07-quote-overdispersion.sql`, table `default.quote_overdispersion_results` (16 columns incl. `quoted_uri`, `quoted_author_did`; `ORDER BY (run_timestamp, granularity, quoted_uri)`).
- `pyproject.toml`: `scipy>=1.15.0` present. Tests: 93 tests currently green; same styles as url (`uv run pytest`).

External-dependency findings: identical to Phase 2 (scipy `betabinom` since 1.4; beta MoM `M = μ(1−μ)/σ² − 1`, `α = μM`, `β = (1−μ)M`, valid iff `M > 0`; nbinom regression guard for scipy#16120 via the exact identity `P(X ≥ observed) = scipy.special.betainc(observed, r, 1 − p)` — `nbdtrc` is NOT valid, it truncates non-integer r).

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: `counts.py` (duplicated contract module)

**Verifies:** stats-methodology.AC1.1, stats-methodology.AC1.2, stats-methodology.AC1.3, stats-methodology.AC1.4

**Files:**
- Create: `quote_overdispersion/src/quote_overdispersion/counts.py`
- Test: `quote_overdispersion/tests/test_counts.py` (unit)

**Implementation:**

Create `counts.py` with exactly this content (byte-identical to the signup_anomaly/url_overdispersion copies):

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

`tests/test_counts.py`, standard matrix (identical numbers to the other sidecars' copies):
- AC1.1: `count_p_value(3, 2.0, 4.0) == pytest.approx(0.3125, abs=1e-12)` (r = 2, p = 0.5, exact hand computation).
- AC1.2: `count_p_value(5, 3.0, 2.0)` / `(5, 3.0, 3.0)` / `(5, 3.0, None)` all `== float(poisson.sf(4, 3.0))`.
- AC1.3: `count_p_value(7, 3.0, 5.0)` (r = 4.5, p = 0.6) in (0, 1] and `== float(scipy.special.betainc(7, 4.5, 0.4))` within `abs=1e-10` (scipy#16120 guard via the exact identity `P(X ≥ observed) = betainc(observed, r, 1 − p)`, ≈ 0.07519; `nbdtrc` is NOT valid — it truncates r = 4.5 to 4; fallback implementation is `float(betainc(observed, r, 1.0 - p))` with int `observed` if a future pin breaks it).
- AC1.4: mean ≤ 0 → 1.0; `count_p_value(0, 10.0, 20.0) == 1.0` and with `None` variance.
- NB > Poisson tail under overdispersion: `count_p_value(15, 5.0, 15.0) > float(poisson.sf(14, 5.0))`.

**Verification:**
Run: `cd quote_overdispersion && uv run pytest tests/test_counts.py`
Expected: all pass.

**Commit:** `feat: add NB/Poisson count test module to quote_overdispersion`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `fdr.py` (duplicated contract module)

**Verifies:** stats-methodology.AC2.1, stats-methodology.AC2.4

**Files:**
- Create: `quote_overdispersion/src/quote_overdispersion/fdr.py`
- Test: `quote_overdispersion/tests/test_fdr.py` (unit)

**Implementation:**

Create `fdr.py` with exactly this content (byte-identical to the other copies):

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

`tests/test_fdr.py`, standard matrix: hand example `[0.01, 0.04, 0.03, 0.005] -> [0.02, 0.04, 0.04, 0.02]`; monotone in p-order and `q >= p` on `[0.9, 0.001, 0.5, 0.02, 0.02]`; order preservation; ties; `[] -> []`; `[0.03] -> [0.03]`; cap at 1.0.

**Verification:**
Run: `cd quote_overdispersion && uv run pytest tests/test_fdr.py`
Expected: all pass.

**Commit:** `feat: add Benjamini-Hochberg FDR module to quote_overdispersion`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `density.py` (duplicated contract module)

**Verifies:** stats-methodology.AC3.1, stats-methodology.AC3.2, stats-methodology.AC3.3, stats-methodology.AC3.4

**Files:**
- Create: `quote_overdispersion/src/quote_overdispersion/density.py`
- Test: `quote_overdispersion/tests/test_density.py` (unit)

**Implementation:**

Create `density.py` with exactly this content (byte-identical to the url_overdispersion copy):

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

**Testing:**

`tests/test_density.py`, standard matrix (identical numbers to the url copy):
- AC3.1: `density_p_value(9, 10, 0.5, 0.05) == pytest.approx(186/1716, rel=1e-9)` (M = 4, α = β = 2; P(X≥9) = (120 + 66)/1716) and `== float(betabinom.sf(8, 10, 2.0, 2.0))`.
- AC3.2: `density_p_value(9, 10, 0.5, None) == pytest.approx(float(binom.sf(8, 10, 0.5)))` = 11/1024; variance `0.0` and `0.3` (M ≤ 0) also take the binomial path.
- AC3.3: at/below baseline → 1.0 (`(5, 10, 0.5, 0.05)` and `(3, 10, 0.5, 0.05)`).
- AC3.4: `expected_density <= 0` → 1.0; `total_shares == 0` → 1.0.
- Fat tail: beta-binomial p > plain binomial p for the AC3.1 inputs.

**Verification:**
Run: `cd quote_overdispersion && uv run pytest tests/test_density.py`
Expected: all pass.

**Commit:** `feat: add beta-binomial density test module to quote_overdispersion`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->
<!-- START_TASK_4 -->
### Task 4: Daily baseline SQL — densified, median-centred, min_sharers on scored rows only

**Verifies:** stats-methodology.AC4.1, stats-methodology.AC4.2, stats-methodology.AC4.4 (daily half)

**Files:**
- Modify: `quote_overdispersion/src/quote_overdispersion/queries.py:7-66` (`daily_aggregation_query`)
- Test: `quote_overdispersion/tests/test_queries.py` (unit)

**Implementation:**

Apply exactly the Phase 2 Task 4 pipeline shape with quote-specific substitutions. Full target structure:

```sql
WITH raw_shares AS (
    SELECT
        if(PostEmbedRecordUri != '', PostEmbedRecordUri, PostEmbedRecordWithMediaUri) AS quoted_uri,
        toDate(__timestamp) AS bucket,
        count() AS total_shares,
        uniq(UserId) AS unique_sharers,
        arraySlice(groupArray(DISTINCT UserId), 1, 5) AS sample_dids
    FROM {config.source_table}
    WHERE Collection = 'app.bsky.feed.post'
        AND OperationKind = 'create'
        AND (PostEmbedRecordUri != '' OR PostEmbedRecordWithMediaUri != '')
        AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
    GROUP BY quoted_uri, bucket
),
scored_entities AS (
    SELECT quoted_uri
    FROM raw_shares
    WHERE bucket = toDate(now()) AND unique_sharers >= {config.min_sharers}
),
entities AS (
    SELECT r.quoted_uri AS quoted_uri, min(r.bucket) AS first_seen
    FROM raw_shares r
    INNER JOIN scored_entities s ON r.quoted_uri = s.quoted_uri
    GROUP BY r.quoted_uri
),
calendar AS (
    SELECT toDate(now()) - number AS bucket FROM numbers({config.baseline_days + 1})
),
dense AS (
    SELECT
        e.quoted_uri AS quoted_uri,
        c.bucket AS bucket,
        coalesce(r.total_shares, 0) AS total_shares,
        coalesce(r.unique_sharers, 0) AS unique_sharers,
        if(coalesce(r.total_shares, 0) > 0, toFloat64(r.unique_sharers) / r.total_shares, NULL) AS sharer_density,
        r.sample_dids AS sample_dids
    FROM entities e
    CROSS JOIN calendar c
    LEFT JOIN raw_shares r ON r.quoted_uri = e.quoted_uri AND r.bucket = c.bucket
    WHERE c.bucket >= e.first_seen
),
baseline AS (
    SELECT
        quoted_uri, bucket, total_shares, unique_sharers, sharer_density, sample_dids,
        medianExact(total_shares) OVER w AS rolling_volume_median,
        avg(total_shares) OVER w AS rolling_volume_mean,
        ifNotFinite(varPop(total_shares) OVER w, NULL) AS rolling_volume_variance,
        avg(sharer_density) OVER w AS rolling_density_mean,
        ifNotFinite(varPop(sharer_density) OVER w, NULL) AS rolling_density_variance,
        count() OVER w AS baseline_buckets_available
    FROM dense
    WINDOW w AS (
        PARTITION BY quoted_uri
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
    b.quoted_uri,
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
    p.population_volume_median,
    p.population_volume_dispersion,
    p.population_density_median,
    p.population_density_variance
FROM baseline b
CROSS JOIN population_stats p
WHERE b.bucket = toDate(now()) AND b.unique_sharers >= {config.min_sharers}
```

Same docstring rationale as Phase 2 Task 4 (min_sharers to scored rows only; zero-days contribute 0 to volume and NULL to density so `avg`/`varPop` skip them; `first_seen` bound preserves the cold-start guard; population stats scoped to today's scored entities, no `rolling_volume_mean IS NOT NULL` qualifier). One quote-specific note for the docstring: quoted-post lifecycles are short (most quoting happens within a day or two of the original post), so entity baselines will hit the population fallback more often than domains do — expected, not a bug.

**Testing:**

Update `TestDailyAggregationQuery` in `tests/test_queries.py`:
- Entity extraction preserved: the `if(PostEmbedRecordUri != '', ...)` coalescing expression and the `(PostEmbedRecordUri != '' OR ...)` filter still asserted (keep existing assertions).
- AC4.4: `'HAVING' not in query`; `unique_sharers >= 3` in `scored_entities` and final `WHERE`; `'rolling_volume_mean IS NOT NULL' not in query`.
- AC4.1: `CROSS JOIN`, `numbers(15)`, `LEFT JOIN raw_shares`, `c.bucket >= e.first_seen`, `coalesce(r.total_shares, 0)`.
- AC4.2: `medianExact(total_shares) OVER w` plus retained `avg`/`varPop` for volume and density.
- Remove stale assertions pinned to the old `domain_shares` CTE name / mean-only baseline.

**Verification:**
Run: `cd quote_overdispersion && uv run pytest tests/test_queries.py -k Daily`
Expected: all pass.

**Commit:** combined with Task 5.
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Hourly baseline SQL — hour-of-day matching

**Verifies:** stats-methodology.AC4.1, stats-methodology.AC4.3

**Files:**
- Modify: `quote_overdispersion/src/quote_overdispersion/queries.py:69-128` (`hourly_aggregation_query`)
- Test: `quote_overdispersion/tests/test_queries.py` (unit)

**Implementation:**

Same as Task 4 with the hourly substitutions (mirror of Phase 2 Task 5): `toStartOfHour(__timestamp)` buckets; calendar `SELECT toStartOfHour(now()) - toIntervalHour(number) AS bucket FROM numbers({(config.baseline_days + 1) * 24})`; window

```sql
    WINDOW w AS (
        PARTITION BY quoted_uri, toHour(bucket)
        ORDER BY bucket
        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
    )
```

`baseline_buckets_available` is already in days (one same-hour bucket per day per partition) — remove the `* 24` window scaling, `intDiv(..., 24)`, and `* 24` cold-start scaling; final filter `WHERE b.bucket = toStartOfHour(now()) AND b.unique_sharers >= {config.min_sharers}`.

**Testing:**

Update `TestHourlyAggregationQuery`: `PARTITION BY quoted_uri, toHour(bucket)` present; `'336 PRECEDING' not in query`; `'intDiv' not in query`; `ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING`; `numbers(360)`; `'HAVING' not in query`; un-scaled cold-start guard; existing entity-extraction assertions kept.

**Verification:**
Run: `cd quote_overdispersion && uv run pytest tests/test_queries.py`
Expected: all pass.

**Commit:** `feat: densified median baselines, scored-row min_sharers, hour matching in quote_overdispersion queries`
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 6-7) -->
<!-- START_TASK_6 -->
### Task 6: Row types and fetch/insert plumbing

**Verifies:** stats-methodology.AC7.1 (column-list side)

**Files:**
- Modify: `quote_overdispersion/src/quote_overdispersion/db.py:13-25` (`AggregatedRow`), `db.py:28-45` (`ScoredResult`), fetch mapping, insert columns at `db.py:89-106`
- Test: `quote_overdispersion/tests/test_db.py` (unit)

**Implementation:**

Mirror Phase 2 Task 6 without `sample_urls`/`on_watchlist` and with `quoted_uri`/`quoted_author_did`:

1. `AggregatedRow` adds `rolling_volume_median: float | None` (before `rolling_volume_mean`), `rolling_volume_variance: float | None`, `rolling_density_variance: float | None` (after their means), `population_volume_dispersion: float | None`, `population_density_variance: float | None`. Fetch mapping follows the Task 4/5 SELECT order: `quoted_uri, bucket_start, total_shares, unique_sharers, sharer_density, rolling_volume_median, rolling_volume_mean, rolling_volume_variance, rolling_density_mean, rolling_density_variance, baseline_days_available, sample_dids, population_volume_median, population_volume_dispersion, population_density_median, population_density_variance`.
2. `ScoredResult` adds `volume_q_value: float` after `volume_p_value`, `density_q_value: float` after `density_p_value`, and diagnostics `rolling_volume_median`, `rolling_volume_variance`, `rolling_density_mean`, `rolling_density_variance` (all `float | None`) after `expected_density_lambda`.
3. Insert column list becomes (22 names):

```python
column_names = [
    'run_timestamp', 'granularity', 'quoted_uri', 'quoted_author_did',
    'bucket_start', 'total_shares', 'unique_sharers', 'sharer_density',
    'expected_volume_lambda', 'expected_density_lambda',
    'rolling_volume_median', 'rolling_volume_variance',
    'rolling_density_mean', 'rolling_density_variance',
    'volume_p_value', 'volume_q_value', 'density_p_value', 'density_q_value',
    'is_anomaly', 'baseline_source', 'baseline_days_available', 'sample_dids',
]
```

**Testing:**

Update `tests/test_db.py`: dataclass construction with new fields and `None`s; frozen-ness; fetch mapping order (median vs mean transposition guard); insert asserting the 22-name list and q-value adjacency.

**Verification:**
Run: `cd quote_overdispersion && uv run pytest tests/test_db.py`
Expected: pass (analyzer/main red until Task 7 — do not commit yet).

**Commit:** combined with Task 7.
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Analyzer — NB volume + beta-binomial density with per-signal BH

**Verifies:** stats-methodology.AC1.2 (integration), stats-methodology.AC2.2, stats-methodology.AC2.3, stats-methodology.AC3.1 (integration), stats-methodology.AC4.2 (analyzer side)

**Files:**
- Modify: `quote_overdispersion/src/quote_overdispersion/analyzer.py`
- Test: `quote_overdispersion/tests/test_analyzer.py`, `quote_overdispersion/tests/test_main.py` (unit)

**Implementation:**

Mirror Phase 2 Task 7 exactly, with these quote-specific deltas:

1. `extract_quoted_author_did` (line 12) is untouched; `score_row` keeps calling it and populating `quoted_author_did`.
2. Delete `compute_p_value` (41) and `compute_density_p_value` (51) and the scipy imports; import `count_p_value` from `.counts`, `density_p_value` from `.density`, `bh_adjust` from `.fdr`, `replace` from `dataclasses`.
3. `determine_baseline` — entity path requires `baseline_days_available >= cold_start_min_days`, `rolling_volume_median` present and > 0, `rolling_density_mean` present and > 0 → `(rolling_volume_median, rolling_density_mean, 'entity')`; else population medians (> 0) → `'population'`; else `(0.0, 0.0, 'population')`.
4. New helper `determine_variances(row, cold_start_min_days, baseline_source)` identical in shape to Phase 2's: volume variance = `phi * volume_centre` when `phi = rolling_volume_variance / rolling_volume_mean > 1` (or population dispersion × population median for the population path), else `None`; density variance = `rolling_density_variance` / `population_density_variance`, else `None`.
5. `score_row` computes both raw p-values, fills diagnostics and `quoted_author_did`, returns provisional result with `volume_q_value=1.0`, `density_q_value=1.0`, `is_anomaly=0`.
6. `score_rows` (no watchlist parameter here): BH per signal within the call —

```python
    provisional = [score_row(row, config, granularity, run_timestamp) for row in rows]
    volume_q = bh_adjust([r.volume_p_value for r in provisional])
    density_q = bh_adjust([r.density_p_value for r in provisional])
    results = []
    for result, vq, dq in zip(provisional, volume_q, density_q):
        is_anomaly = 1 if (vq < config.volume_p_threshold or dq < config.density_p_threshold) else 0
        results.append(replace(result, volume_q_value=vq, density_q_value=dq, is_anomaly=is_anomaly))
    return results
```

**Testing:**

Update `tests/test_analyzer.py` and `tests/test_main.py` with the Phase 2 Task 7 matrix adapted to quote types: baseline median semantics (median 0 → population), variance derivation, `score_row` p-values equal direct module calls, AC2.2/AC2.3 family behaviour with hand-picked 4-row families, `extract_quoted_author_did` regression tests untouched and still green, `FakeDb` shapes updated, q-values flow to inserts.

**Verification:**
Run: `cd quote_overdispersion && uv run pytest`
Expected: full suite passes.

**Commit:** `feat: NB volume and beta-binomial density scoring with per-signal BH-FDR in quote_overdispersion`
<!-- END_TASK_7 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_TASK_8 -->
### Task 8: ClickHouse schema — q-value and rolling-stats columns

**Verifies:** stats-methodology.AC7.1 (quote scope)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/07-quote-overdispersion.sql`

**Implementation:**

1. In `CREATE TABLE IF NOT EXISTS default.quote_overdispersion_results`, add: `rolling_volume_median Nullable(Float64)`, `rolling_volume_variance Nullable(Float64)`, `rolling_density_mean Nullable(Float64)`, `rolling_density_variance Nullable(Float64)` after `expected_density_lambda`; `volume_q_value Float64` after `volume_p_value`; `density_q_value Float64` after `density_p_value`.
2. Append idempotent migrations (six `ALTER TABLE default.quote_overdispersion_results ADD COLUMN IF NOT EXISTS ... AFTER ...` statements mirroring Phase 2 Task 8's, against this table).

**Verification:**
Run: `grep -c 'ADD COLUMN IF NOT EXISTS' /Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/07-quote-overdispersion.sql`
Expected: `6`. Cross-check every Task 6 insert name appears in the CREATE TABLE.

**Commit** (in skywatch-osprey): `Add q-value and rolling-stat columns to quote_overdispersion_results`
<!-- END_TASK_8 -->

<!-- START_TASK_9 -->
### Task 9: Documentation updates

**Verifies:** None (documentation; AC8.2 finalized in Phase 7)

**Files:**
- Modify: `quote_overdispersion/README.md`
- Modify: `quote_overdispersion/CLAUDE.md`

**Implementation:**

Same content plan as Phase 2 Task 9, adapted to quoted-post subjects: NB volume + beta-binomial density (one-sided), dense median baselines with `min_sharers` on scored rows only, hour-of-day matching, per-cycle per-signal BH, env var names unchanged as FDR targets, new columns, accepted approximations.

**Verification:**
Run: `grep -n 'normal-approximation\|z-test' quote_overdispersion/README.md quote_overdispersion/CLAUDE.md`
Expected: no stale references.

**Commit:** `docs: update quote_overdispersion README and CLAUDE.md for NB + beta-binomial + FDR`
<!-- END_TASK_9 -->

<!-- START_TASK_10 -->
### Task 10: Full suite gate

**Verifies:** stats-methodology.AC9.1 (quote_overdispersion scope)

**Files:** none (verification only)

**Step 1:** Run: `cd quote_overdispersion && uv run pytest` — Expected: all pass.

**Step 2:** Run: `cd quote_overdispersion && uv run ruff check src tests && uv run ruff format --check src tests` — Expected: clean.

**Step 3:** Run: `git status --short` in both repos — Expected: clean trees; Task 8's commit exists in skywatch-osprey.
<!-- END_TASK_10 -->

# Statistical Methodology Fixes Design

## Summary

This plan fixes eight statistical-correctness issues found in Skywatch's six "sidecar" services — small standalone jobs that read AT Protocol activity from ClickHouse, score it for anomalies (unusual posting volume, bursty URL/quote sharing, bot-like timing, coordinated co-sharing clusters), and write flags back for the moderation pipeline. The fixes replace ad hoc statistical shortcuts with textbook-correct methods: dispersion-aware count tests (negative binomial instead of plain Poisson), a properly one-sided density test (beta-binomial), Benjamini–Hochberg FDR correction so `is_anomaly` reflects a controlled false-discovery rate rather than a raw p-value cutoff, bias-corrected and normalized entropy for bot detection, and popularity-down-weighted co-sharing edges so a single viral URL can't manufacture a fake cluster. Baseline calculations are also hardened (dense zero-filled windows, robust medians, hour-of-day matching) to remove several sources of selection bias.

The approach treats each sidecar as an independent workstream: since the sidecars deliberately share no code (a pre-existing convention), the three new statistical modules (`counts.py`, `fdr.py`, `density.py`) are specified once as contracts here and then duplicated verbatim into each sidecar that needs them, mirroring how `compute_p_value` is already duplicated today. This lets six of the seven phases run in parallel across separate worktrees/agents with no cross-dependencies, followed by a final documentation-and-calibration phase that updates cross-repo docs and adds a playbook for validating flag rates after deployment (validation itself is out of scope for this plan).

## Definition of Done

1. All eight review findings fixed across the six sidecars:
   - Density test rebuilt on a correct sampling model (one-sided, high-density direction preserved).
   - Poisson volume/signup tests replaced with negative binomial fed by existing rolling mean/variance estimates.
   - Baselines fixed: zero-bucket inclusion, rolling median instead of mean, removal of truncation/selection bias in population fallbacks, hour-of-day-matched hourly baselines.
   - Benjamini–Hochberg FDR control per analysis cycle; `is_anomaly` becomes q-value based with a new `q_value` column.
   - Entropy scores normalized (H / log2(min(N, bins))) with Miller–Madow bias correction; thresholds re-expressed on the 0–1 normalized scale; raw entropy retained as context.
   - Co-sharing edge weights down-weighted by URL popularity.
   - `build_graph` hardened against duplicate account pairs (parallel-edge risk).
   - Minor items: unused interval mean/stddev either used (e.g., coefficient of variation) or removed; "overdispersion" naming clarified in docs.
2. Matching schema and scheduled-query changes in skywatch-osprey's `clickhouse-init/` (breaking changes allowed).
3. Unit tests for every changed statistical function; all sidecar test suites pass.
4. A calibration document describing how to validate flag rates/behaviour once deployed to prod (validation itself is not performed now).
5. Documentation updated: per-sidecar READMEs and CLAUDE.md files in this repo, and `skywatch-osprey/docs/statistical-sidecars.md` rewritten to match the new methods, including adding the currently-missing quote_* sidecar sections.
6. Implementation plan structured so the six sidecars can be executed by parallel agents/worktrees.

**Out of scope:** backtesting against real production data, downstream consumers/UI, Ozone integration.

## Acceptance Criteria

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

### stats-methodology.AC5: Entropy
- **stats-methodology.AC5.1 Success:** `normalized_entropy` = 1.0 for uniform over achievable bins, 0.0 for single-bin, always in [0, 1]
- **stats-methodology.AC5.2 Success:** Miller–Madow correction applied and numerically verified
- **stats-methodology.AC5.3 Success:** An account with 10 uniformly-spread posts can cross the hourly threshold (impossible under the old 3.9-bit rule)
- **stats-methodology.AC5.4 Success:** `interval_cv` computed; `cv_flag` when CV ≤ threshold; `is_bot_like = hourly_flag AND (interval_flag OR cv_flag)`

### stats-methodology.AC6: Co-sharing
- **stats-methodology.AC6.1 Success:** Pairs MV emits `newman_weight = Σ 1/(k_url − 1)`; verified by query-structure test
- **stats-methodology.AC6.2 Success:** `build_graph` aggregates duplicate (a, b) pairs — weights summed, URLs unioned, no parallel edges, no None weights
- **stats-methodology.AC6.3 Success:** Leiden receives `newman_weight`; `min_edge_weight` still filters raw weight
- **stats-methodology.AC6.4 Success:** Batch edge construction yields a graph identical to the per-edge loop on the same input

### stats-methodology.AC7: Schemas
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar)

### stats-methodology.AC8: Documentation
- **stats-methodology.AC8.1 Success:** statistical-sidecars.md describes NB, beta-binomial, FDR, normalized entropy, and Newman weighting; includes quote_* sections
- **stats-methodology.AC8.2 Success:** Per-sidecar README/CLAUDE.md match shipped behaviour
- **stats-methodology.AC8.3 Success:** `docs/calibration.md` exists with per-sidecar validation queries, healthy ranges, and tuning levers

### stats-methodology.AC9: Suites
- **stats-methodology.AC9.1 Success:** `uv run pytest` passes in all six sidecars

## Glossary

- **Sidecar**: A small, independently deployable service in this repo that runs its own ClickHouse queries, scores data statistically, and writes results back — deliberately isolated from the others (no shared code).
- **Negative binomial (NB) distribution**: A count distribution like Poisson but with an extra parameter for variance, used when observed counts are more variable ("overdispersed") than Poisson assumes.
- **Method of moments (MoM)**: A way to fit a distribution's parameters by matching its theoretical mean/variance to the sample's observed mean/variance, rather than using maximum likelihood.
- **Overdispersion**: When a count variable's variance exceeds its mean, violating the Poisson assumption that they're equal; the trigger for switching to NB.
- **Survival function (`sf`)**: `P(X ≥ observed)`, i.e. 1 minus the CDF; used here as the one-sided p-value for "is this count unusually high."
- **Beta-binomial distribution**: A binomial distribution whose success probability is itself uncertain (drawn from a Beta distribution), giving a fatter-tailed model of proportions than plain binomial — used for the density test.
- **One-sided test**: A statistical test that only flags deviation in one direction (here, high density/counts), never low.
- **Benjamini–Hochberg (BH) / False Discovery Rate (FDR)**: A method for adjusting many p-values at once so that, on average, only a target fraction of flagged results are false positives, rather than controlling each test's error rate individually.
- **q-value**: The BH-adjusted p-value; comparing q-value to a threshold controls FDR across a batch of tests, unlike comparing raw p-values.
- **Family (in FDR context)**: A batch of p-values adjusted together as one BH run; this plan defines families per granularity (daily vs. hourly) and per signal (volume vs. density) so unrelated tests don't dilute each other's correction.
- **Rolling mean / variance / median**: Summary statistics computed over a moving time window (e.g., trailing N days) used as the expected baseline against which new observations are compared.
- **Densification**: Filling in a time-bucket grid so that entities with zero activity in a bucket get an explicit zero row, instead of being silently absent — needed so rolling averages aren't biased upward by missing zeros.
- **Selection bias (here)**: Baseline statistics being computed only over already-filtered/active rows, which skews the "normal" baseline away from true population behaviour.
- **Hour-of-day matching**: Comparing an hourly observation only against historical observations from the same hour, to avoid conflating daily rhythm (e.g., quiet at 3am) with anomalous behaviour.
- **Entropy (Shannon entropy)**: A measure of how spread out/unpredictable a distribution is (e.g., an account's posting times across hours); low entropy suggests scripted, repetitive behaviour.
- **Miller–Madow bias correction**: A small additive correction to entropy estimates that compensates for the fact that entropy is systematically underestimated from finite samples.
- **Normalized entropy**: Raw entropy divided by the maximum entropy achievable given the sample size and bin count, rescaling it to a fixed [0, 1] range so thresholds are comparable across accounts with different activity levels.
- **Coefficient of variation (CV)**: Standard deviation divided by mean; a scale-free measure of variability, used here as an additional bot-like-timing signal.
- **Co-sharing / cosharing cluster**: A group of accounts identified as suspicious because they repeatedly share the same URLs (or quote the same posts) together.
- **Newman weighting**: A network-science edge-weighting scheme (Newman, 2001) that divides each shared URL's contribution by (number of people sharing it − 1), so widely shared/viral URLs contribute less to a pair's connection strength than niche ones.
- **Leiden algorithm**: A graph community-detection algorithm used to partition the co-sharing network into clusters of densely connected accounts.
- **Parallel edges**: Multiple graph edges between the same pair of nodes; the plan hardens `build_graph` to merge these into one weighted edge instead of leaving duplicates.
- **AT-URI**: The AT Protocol's URI scheme (e.g., `at://did:plc:.../app.bsky.feed.post/...`) used to reference records like posts; relevant here for identifying quoted posts in `quote_overdispersion`/`quote_cosharing`.
- **Materialized view (MV)**: A ClickHouse table that's automatically kept up to date from an underlying query/stream, used here to pre-aggregate co-sharing pairs.
- **TTL (Time To Live)**: A ClickHouse table setting that automatically expires old rows after a set period; used here to bound how much history is lost when the pairs table is migrated.
- **Functional Core / Imperative Shell (FCIS)**: An architecture pattern separating pure logic (Core: `config.py`/`queries.py`/`analyzer.py`) from side-effecting I/O (Shell: `db.py`/`main.py`).
- **scipy `nbinom`**: SciPy's negative binomial distribution implementation, called out here because some versions have a known bug (scipy#16120) with non-integer shape parameters.

## Architecture

Six independent workstreams (one per sidecar) plus a final documentation workstream. Each sidecar keeps its standalone-project constraint: no shared imports, so common statistical functions are duplicated per sidecar as small identical modules — the same pattern as today's duplicated `compute_p_value`. The plan specifies these modules once as contracts; each workstream carries its own copy and tests.

Two repos are touched:

- `osprey-sidecars` (this repo): analyzer/queries/config/tests/docs per sidecar.
- `skywatch-osprey`: `clickhouse-init/*.sql` schema and materialized-view changes, plus `docs/statistical-sidecars.md`.

### Cross-cutting module contracts

**`counts.py`** (signup_anomaly, url_overdispersion, quote_overdispersion):

```python
def count_p_value(observed: int, mean: float, variance: float | None) -> float:
    """P(X >= observed) under NB when variance > mean, else Poisson(mean).

    NB via method of moments: r = mean**2 / (variance - mean), p = mean / variance.
    mean <= 0 or observed is None -> 1.0. Non-integer r supported (scipy nbinom).
    """
```

The expected count passed as `mean` is the rolling *median* (robust centre). Dispersion comes from rolling mean/variance as a factor `phi = rolling_variance / rolling_mean`; callers pass `variance = phi * median` so the model is NB(mean=median, variance=phi*median), i.e. `r = median/(phi-1)`, `p = 1/phi` when `phi > 1`.

**`fdr.py`** (same three sidecars):

```python
def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values: step-up with cumulative-min monotonicity.

    Input order preserved. Empty list -> empty list. Pure Python, no statsmodels.
    """
```

Families are per analysis cycle per granularity (daily and hourly adjusted separately), and per signal (volume and density p-values adjusted separately). Discrete p-values make BH conservative; accepted and documented.

**`density.py`** (url_overdispersion, quote_overdispersion):

```python
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
```

**Entropy functions** (account_entropy): `compute_entropy` stays; new `normalized_entropy(counts, max_bins) -> float` applies Miller–Madow bias correction (`H + (K_occupied - 1) / (2 * N * ln 2)` bits) then divides by `log2(min(N, max_bins))` — the achievable maximum given sample size. Result clamped to [0, 1]. New `coefficient_of_variation(mean, stddev) -> float`.

### Output schema conventions

- Every raw p-value column gains a sibling `q_value` column (`volume_q_value`, `density_q_value`, `q_value` for signup).
- `is_anomaly` semantics change to `q_value < threshold` (existing guards like `observed_count > 0` survive). Threshold env vars keep their names, reinterpreted as FDR targets.
- account_entropy adds `hourly_entropy_norm`, `interval_entropy_norm`, `interval_cv`, `cv_flag`; raw bit-valued entropies retained. `is_bot_like = hourly_flag AND (interval_flag OR cv_flag)`.
- cosharing pairs tables add `newman_weight Float64`; clusters tables unchanged except semantics of `total_weight` documented.

### Baseline SQL redesign (shared pattern)

Applied identically in signup_anomaly, url_overdispersion, quote_overdispersion `queries.py`:

1. **Densification:** distinct entities in the lookback CROSS JOIN a generated bucket calendar, LEFT JOIN observed counts, zeros filled. Rolling windows run over true elapsed time.
2. **Robust rolling stats:** `medianExact` (baseline centre), `avg` and `varPop` (dispersion factor) over the dense window.
3. **Hour-of-day matching:** hourly baselines partition by `(entity, toHour(bucket))` so each hour is compared with the same hour on prior days.
4. **Selection-bias removal:** activity filters (`min_sharers`) apply only to the final scored rows, not baseline history. Population medians drop the `dispersion_index IS NOT NULL` / `rolling_mean >= 1` qualifiers.

### Co-sharing pipeline changes

The `*_cosharing_pairs_mv` materialized views compute `newman_weight = sum(1 / (k_url - 1))` per pair (Newman 2001 collaboration weighting), where `k_url` is the number of accounts sharing that URL that day — available in the existing `qualifying_urls` CTE. Raw `weight` stays for filtering and investigations. Migration: drop/recreate MV and pairs table (7-day TTL bounds the loss).

`build_graph` aggregates rows by (account_a, account_b) before edge creation — duplicates sum weights and union shared URLs — then builds edges in a single `add_edges` batch with attribute lists. Leiden clusters on `newman_weight`; `min_edge_weight` continues to filter on raw `weight`.

## Existing Patterns

- **Functional Core / Imperative Shell** per sidecar (`config.py`/`queries.py`/`analyzer.py` pure; `db.py`/`main.py` I/O). All new statistical code is Core; no Shell changes beyond column lists.
- **Duplicated-module convention:** sidecars deliberately share no code. New modules (`counts.py`, `fdr.py`, `density.py`) are duplicated per sidecar, mirroring the existing duplicated `compute_p_value`.
- **Env-var config into frozen dataclasses** — new thresholds follow `AnalysisConfig` conventions.
- **Query generation as pure string functions** with unit tests asserting SQL structure.
- **clickhouse-init numbered SQL files** in skywatch-osprey own all table/MV DDL.

One divergence: analyzers gain scipy as a dependency where absent (account_entropy has no new dependency; scipy already present in the p-value sidecars).

## Implementation Phases

Phases 1–6 are mutually independent — execute in parallel worktrees/agents. Phase 7 depends on all of them.

<!-- START_PHASE_1 -->
### Phase 1: signup_anomaly
**Goal:** NB/Poisson dispersion-aware test, dense hour-matched median baselines, BH-FDR q-values.

**Components:**
- `signup_anomaly/src/signup_anomaly/counts.py`, `fdr.py` — new Core modules per contracts
- `signup_anomaly/src/signup_anomaly/queries.py` — baseline SQL redesign (densification, medianExact, hour-of-day partitioning, population-median qualifier removal)
- `signup_anomaly/src/signup_anomaly/analyzer.py` — score via `count_p_value`, cycle-level `bh_adjust` per granularity, q-value `is_anomaly`
- `signup_anomaly/src/signup_anomaly/db.py` — row types gain `q_value`; insert column list updated
- `skywatch-osprey/clickhouse-init/02-signup-anomalies.sql` — add `q_value Float64`
- `signup_anomaly/tests/` — updated + new unit tests
- `signup_anomaly/README.md`, `CLAUDE.md` — method and invariant updates

**Dependencies:** None.

**Done when:** `cd signup_anomaly && uv run pytest` passes; covers stats-methodology.AC1, AC2, AC4 (signup scope).
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: url_overdispersion
**Goal:** NB volume test, beta-binomial density test, fixed baselines, BH-FDR.

**Components:**
- `url_overdispersion/src/url_overdispersion/counts.py`, `fdr.py`, `density.py` — new Core modules per contracts
- `url_overdispersion/src/url_overdispersion/queries.py` — baseline redesign + rolling density variance + HAVING moved to scored rows
- `url_overdispersion/src/url_overdispersion/analyzer.py` — two-pass scoring (raw p-values, then per-granularity per-signal BH), q-value OR-logic `is_anomaly`
- `url_overdispersion/src/url_overdispersion/db.py` — `volume_q_value`, `density_q_value`, rolling stats columns
- `skywatch-osprey/clickhouse-init/03-url-overdispersion.sql` — column additions
- `url_overdispersion/tests/`, `README.md`, `CLAUDE.md`

**Dependencies:** None.

**Done when:** `cd url_overdispersion && uv run pytest` passes; covers stats-methodology.AC1, AC2, AC3, AC4 (url scope).
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: quote_overdispersion
**Goal:** Same as Phase 2 for the quote sidecar (entity = quoted subject; AT-URI extraction unchanged).

**Components:** mirror of Phase 2 under `quote_overdispersion/`, plus `skywatch-osprey/clickhouse-init/07-quote-overdispersion.sql`.

**Dependencies:** None.

**Done when:** `cd quote_overdispersion && uv run pytest` passes; covers stats-methodology.AC1, AC2, AC3, AC4 (quote scope).
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: account_entropy
**Goal:** Bias-corrected normalized entropy, CV signal, revised conjunction.

**Components:**
- `account_entropy/src/account_entropy/analyzer.py` — `normalized_entropy`, `coefficient_of_variation`, revised `score_account`
- `account_entropy/src/account_entropy/config.py` — normalized-scale thresholds (`hourly >= 0.85`, `interval <= 0.53`, `cv <= 0.5` defaults), env var names updated
- `account_entropy/src/account_entropy/db.py` — new columns (`hourly_entropy_norm`, `interval_entropy_norm`, `interval_cv`, `cv_flag`)
- `skywatch-osprey/clickhouse-init/04-account-entropy.sql` — column additions
- `account_entropy/tests/`, `README.md`, `CLAUDE.md`

**Dependencies:** None.

**Done when:** `cd account_entropy && uv run pytest` passes; covers stats-methodology.AC5.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: url_cosharing
**Goal:** Newman-weighted clustering, hardened graph construction.

**Components:**
- `skywatch-osprey/clickhouse-init/05-url-cosharing.sql` — `newman_weight` in pairs table + MV computation; drop/recreate migration notes
- `url_cosharing/src/url_cosharing/queries.py` — fetch `newman_weight`
- `url_cosharing/src/url_cosharing/analyzer.py` — `PairRow.newman_weight`, duplicate-pair aggregation, batch edge construction, Leiden on `newman_weight`
- `url_cosharing/tests/`, `README.md`, `CLAUDE.md`

**Dependencies:** None.

**Done when:** `cd url_cosharing && uv run pytest` passes; covers stats-methodology.AC6 (url scope).
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: quote_cosharing
**Goal:** Same as Phase 5 for quote subjects.

**Components:** mirror of Phase 5 under `quote_cosharing/`, plus `skywatch-osprey/clickhouse-init/06-quote-cosharing.sql`.

**Dependencies:** None.

**Done when:** `cd quote_cosharing && uv run pytest` passes; covers stats-methodology.AC6 (quote scope).
<!-- END_PHASE_6 -->

<!-- START_PHASE_7 -->
### Phase 7: Documentation and calibration
**Goal:** Cross-repo docs reflect the new methods; calibration playbook exists.

**Components:**
- `skywatch-osprey/docs/statistical-sidecars.md` — rewrite affected sections (NB model, beta-binomial density, FDR, normalized entropy, Newman weighting); add quote_overdispersion and quote_cosharing sections; document Jaccard-threshold tuning guidance (0.5 default, literature ~0.3)
- `osprey-sidecars/docs/calibration.md` — per-sidecar post-deploy validation queries, expected healthy ranges, tuning levers (including CPM resolution re-tuning after weight change)
- `osprey-sidecars/README.md` — methods table updated

**Dependencies:** Phases 1–6 (documents what they shipped).

**Done when:** Docs match shipped behaviour; covers stats-methodology.AC7, AC8.
<!-- END_PHASE_7 -->

## Additional Considerations

**Migration ordering:** clickhouse-init changes are additive columns except the cosharing pairs MV drop/recreate. The 7-day TTL on pairs bounds data loss; clusters and membership tables are untouched by the MV migration. Sidecars must deploy together with their schema change (breaking changes accepted per DoD).

**scipy nbinom edge case:** some scipy versions mishandle non-integer parameters in `nbinom` (scipy#16120). Each `counts.py` test suite pins a regression test with non-integer r; if the pinned version misbehaves, fall back to `scipy.special.nbdtrc` or a gamma-mixture formulation.

**Thin hourly baselines:** hour-of-day matching gives signup_anomaly 7 same-hour observations (baseline_days=7) and the overdispersion sidecars 14. Documented in calibration.md; raising `baseline_days` is the tuning lever.

**Known approximations (documented, accepted):** binomial/beta-binomial density model ignores the first-share-is-always-unique dependence; BH on discrete p-values is conservative; entropy normalization treats bins as exchangeable.

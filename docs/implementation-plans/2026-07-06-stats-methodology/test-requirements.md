# Test Requirements — Statistical Methodology Fixes

Maps every acceptance criterion (`stats-methodology.AC1.1` … `stats-methodology.AC9.1`) from
`docs/design-plans/2026-07-06-stats-methodology.md` to either an automated test or documented
human verification. No criterion is orphaned.

**Scoping conventions used below**
- AC1 and AC2 are implemented per-sidecar in Phases 1–3 (signup_anomaly, url_overdispersion, quote_overdispersion). Each sidecar carries a byte-identical `counts.py` / `fdr.py` copy with its own test file — the duplicated-module convention (sidecars share no code).
- AC3 is implemented per-sidecar in Phases 2–3 (url_overdispersion, quote_overdispersion). density.py is duplicated there.
- AC6 is implemented per-sidecar in Phases 5–6 (url_cosharing, quote_cosharing).
- AC4 is signup/url/quote (Phases 1–3); AC5 is account_entropy (Phase 4).
- Test types: **unit** = pure-function / SQL-string-structure test; **integration** = analyzer/main wiring across modules within one sidecar (e.g. `score_row` calling the stat modules, `FakeDb` cycle). There are no cross-service e2e tests — the sidecars are independent and the harness explicitly excludes backtesting/prod data.

---

## AC1 — Count tests are dispersion-aware (per-sidecar: signup / url / quote)

| Criterion | Type | Test file(s) | Verifies |
|---|---|---|---|
| stats-methodology.AC1.1 | unit | `signup_anomaly/tests/test_counts.py`, `url_overdispersion/tests/test_counts.py`, `quote_overdispersion/tests/test_counts.py` | `count_p_value(3, 2.0, 4.0)` hits the NB branch (MoM r=2, p=0.5) and equals the pinned hand value `pytest.approx(0.3125, abs=1e-12)`. |
| stats-methodology.AC1.2 | unit | `signup_anomaly/tests/test_counts.py`, `url_overdispersion/tests/test_counts.py`, `quote_overdispersion/tests/test_counts.py` | Poisson fallback: `count_p_value(5, 3.0, 2.0)`, `(5, 3.0, 3.0)` (equality case), and `(5, 3.0, None)` all equal `poisson.sf(4, 3.0)`. |
| stats-methodology.AC1.2 (analyzer integration) | integration | `signup_anomaly/tests/test_analyzer.py`, `url_overdispersion/tests/test_analyzer.py`, `quote_overdispersion/tests/test_analyzer.py` | `score_row` routes φ≤1 / missing-variance rows through the Poisson path, matching a direct `count_p_value(observed, centre, None)` call. |
| stats-methodology.AC1.3 | unit | `signup_anomaly/tests/test_counts.py`, `url_overdispersion/tests/test_counts.py`, `quote_overdispersion/tests/test_counts.py` | Non-integer r regression guard (scipy#16120): `count_p_value(7, 3.0, 5.0)` (r=4.5, p=0.6) is in (0,1] and equals `scipy.special.betainc(7, 4.5, 0.4)` within `abs=1e-10` (≈0.07519). Cross-check is `betainc`, NOT `nbdtrc` (which truncates 4.5→4 and would falsely fail against a correct `nbinom.sf`). |
| stats-methodology.AC1.4 | unit | `signup_anomaly/tests/test_counts.py`, `url_overdispersion/tests/test_counts.py`, `quote_overdispersion/tests/test_counts.py` | mean≤0 → 1.0 (`0.0` and `-1.0`); observed=0 never flags (`(0, 10.0, 20.0)` and `(0, 10.0, None)` both → 1.0). |

Supporting sanity assertions (same test files, not AC-numbered): p-value monotone decreasing in observed; NB tail exceeds Poisson under overdispersion (`count_p_value(15, 5.0, 15.0) > poisson.sf(14, 5.0)`).

---

## AC2 — FDR control (per-sidecar: signup / url / quote)

| Criterion | Type | Test file(s) | Verifies |
|---|---|---|---|
| stats-methodology.AC2.1 | unit | `signup_anomaly/tests/test_fdr.py`, `url_overdispersion/tests/test_fdr.py`, `quote_overdispersion/tests/test_fdr.py` | `bh_adjust([0.01, 0.04, 0.03, 0.005]) == [0.02, 0.04, 0.04, 0.02]` (pinned hand-computed BH); q-values monotone in p-order; input order preserved (cross-checked against sort-then-unsort); ties share a q. |
| stats-methodology.AC2.2 | integration | `signup_anomaly/tests/test_analyzer.py`, `url_overdispersion/tests/test_analyzer.py`, `quote_overdispersion/tests/test_analyzer.py` | `score_rows`: `is_anomaly=1` iff q < threshold with surviving guards. signup: q < threshold AND `observed_count > 0` (a zero-count row never flags even at tiny q). url/quote: OR-logic `volume_q < target OR density_q < target` on q-values. |
| stats-methodology.AC2.3 | integration | signup: `signup_anomaly/tests/test_analyzer.py`, `signup_anomaly/tests/test_main.py`; url: `url_overdispersion/tests/test_analyzer.py`; quote: `quote_overdispersion/tests/test_analyzer.py` | Family boundaries are per `score_rows` call. **Granularity split (all three):** the same p-vector gets different q-values in a 2-row call vs a 10-row padded call (n changes the correction); daily and hourly separate because `run_cycle` calls `score_rows` once per granularity — asserted in the main/`FakeDb` cycle. **Per-signal split (url + quote only):** volume and density are adjusted as independent BH families in one call. *signup_anomaly has only a volume signal, so its family split is daily-vs-hourly only — the per-signal split is verified in url/quote.* |
| stats-methodology.AC2.4 | unit | `signup_anomaly/tests/test_fdr.py`, `url_overdispersion/tests/test_fdr.py`, `quote_overdispersion/tests/test_fdr.py` | `bh_adjust([]) == []`; `bh_adjust([0.03]) == [0.03]` (single p → q=p); cap at 1.0 for high-p vectors. |

---

## AC3 — Density test (per-sidecar: url / quote)

| Criterion | Type | Test file(s) | Verifies |
|---|---|---|---|
| stats-methodology.AC3.1 | unit | `url_overdispersion/tests/test_density.py`, `quote_overdispersion/tests/test_density.py` | Beta-binomial branch: `density_p_value(9, 10, 0.5, 0.05)` (MoM M=4, α=β=2) equals the pinned hand value `pytest.approx(186/1716, rel=1e-9)` and equals `betabinom.sf(8, 10, 2.0, 2.0)` (pins α/β wiring). |
| stats-methodology.AC3.1 (analyzer integration) | integration | `url_overdispersion/tests/test_analyzer.py`, `quote_overdispersion/tests/test_analyzer.py` | `score_row`'s `density_p_value` equals a direct `density_p_value(...)` call on the same derived (density mean, density variance) inputs. |
| stats-methodology.AC3.2 | unit | `url_overdispersion/tests/test_density.py`, `quote_overdispersion/tests/test_density.py` | Binomial fallback: `density_p_value(9, 10, 0.5, None) == binom.sf(8, 10, 0.5)` = pinned `11/1024`; variance `0.0` and `0.3` (≥ μ(1−μ)=0.25 ⇒ M≤0) take the same binomial path. |
| stats-methodology.AC3.3 | unit | `url_overdispersion/tests/test_density.py`, `quote_overdispersion/tests/test_density.py` | One-sidedness is a **structural guard**: observed density at baseline (`(5, 10, 0.5, 0.05)`) or below (`(3, 10, 0.5, 0.05)`) returns exactly 1.0, never a probabilistic near-1 — the `unique/total <= expected_density -> 1.0` short-circuit. |
| stats-methodology.AC3.4 | unit | `url_overdispersion/tests/test_density.py`, `quote_overdispersion/tests/test_density.py` | `expected_density ≤ 0` → 1.0 (`0.0` and `-0.1`); `total_shares == 0` → 1.0. |

Supporting (same files): beta-binomial p exceeds plain binomial p on the AC3.1 inputs (fat-tail sanity).

---

## AC4 — Baselines (per-sidecar: signup / url / quote)

| Criterion | Type | Test file(s) | Verifies |
|---|---|---|---|
| stats-methodology.AC4.1 | unit | `signup_anomaly/tests/test_queries.py`, `url_overdispersion/tests/test_queries.py`, `quote_overdispersion/tests/test_queries.py` | Generated SQL densifies the entity×bucket grid: contains `CROSS JOIN calendar`/`numbers(...)` (`numbers(8)` signup daily, `numbers(192)` signup hourly; `numbers(15)`/`numbers(360)` url/quote), `LEFT JOIN`, `coalesce(...,0)`, and the `>= first_seen` bound so zero buckets contribute 0 to the window. |
| stats-methodology.AC4.2 | unit | `signup_anomaly/tests/test_queries.py`, `url_overdispersion/tests/test_queries.py`, `quote_overdispersion/tests/test_queries.py` | Expected count is the rolling median: `medianExact(...) OVER w` present; `avg(...) OVER w` and `varPop(...) OVER w` retained for the dispersion factor. |
| stats-methodology.AC4.2 (analyzer integration) | integration | `signup_anomaly/tests/test_analyzer.py`, `url_overdispersion/tests/test_analyzer.py`, `quote_overdispersion/tests/test_analyzer.py` | `determine_baseline` uses the entity rolling **median** (not mean) as the NB centre when history is sufficient and median > 0; median 0 / None routes to the population fallback. |
| stats-methodology.AC4.3 | unit | `signup_anomaly/tests/test_queries.py`, `url_overdispersion/tests/test_queries.py`, `quote_overdispersion/tests/test_queries.py` | Hourly window partitions by hour-of-day: `PARTITION BY <entity>, toHour(bucket)`; old continuous-window artefacts gone (`'168 PRECEDING'`/`'336 PRECEDING'` and `'intDiv'` absent); `ROWS BETWEEN N PRECEDING AND 1 PRECEDING`. |
| stats-methodology.AC4.4 | unit | `signup_anomaly/tests/test_queries.py`, `url_overdispersion/tests/test_queries.py`, `quote_overdispersion/tests/test_queries.py` | Selection-bias removal: `'dispersion_index IS NOT NULL'` / `'rolling_volume_mean IS NOT NULL'` and `'>= 1.0'` absent; `median(rolling_median)` present. url/quote: `'HAVING' not in query`, `min_sharers` (`unique_sharers >= 3`) appears only in `scored_entities` and the final `WHERE`. *signup has no `min_sharers` filter — only the population-qualifier removal applies there.* |

---

## AC5 — Entropy (account_entropy)

| Criterion | Type | Test file(s) | Verifies |
|---|---|---|---|
| stats-methodology.AC5.1 | unit | `account_entropy/tests/test_analyzer.py` (`TestNormalizedEntropy`) | Uniform-over-achievable → 1.0 (`[10]*24` and `[1]*10+[0]*14` both clamp to 1.0); single-bin → 0.0 (`[10,0,0]`); arbitrary histograms stay within [0,1]. |
| stats-methodology.AC5.2 | unit | `account_entropy/tests/test_analyzer.py` (`TestNormalizedEntropy`) | Miller–Madow numerically verified: `normalized_entropy([5,5], 24)` equals the closed form `(1.0 + 1/(20·ln2)) / log2(10)` ≈ pinned **0.322745** via `pytest.approx(rel=1e-9)` against the expression (not a rounded literal). |
| stats-methodology.AC5.3 | integration | `account_entropy/tests/test_analyzer.py` | An `AccountActivityRow` with 10 posts spread over 10 distinct hours yields `hourly_flag == 1` at the 0.85 default (normalized entropy 1.0), with the contrast that raw `hourly_entropy = log2(10) ≈ 3.32 < 3.9` — impossible under the old bit rule. |
| stats-methodology.AC5.4 | unit + integration | `account_entropy/tests/test_analyzer.py`, `account_entropy/tests/test_config.py`, `account_entropy/tests/test_db.py`, `account_entropy/tests/test_main.py` | `coefficient_of_variation` computed (`TestCoefficientOfVariation`: 0.05, 1.5, and `mean≤0 → 0.0` edges); `cv_flag` set when `interval_cv ≤ cv_threshold`; conjunction truth table `is_bot_like = hourly_flag AND (interval_flag OR cv_flag)` pinned across three constructed rows; thresholds parsed in config; new columns flow through `db.py` insert and the `FakeDb` cycle. |

---

## AC6 — Co-sharing (per-sidecar: url / quote)

| Criterion | Type | Test file(s) | Verifies |
|---|---|---|---|
| stats-methodology.AC6.1 | unit | `url_cosharing/tests/test_queries.py`, `quote_cosharing/tests/test_queries.py` | `fetch_pairs_query` selects `newman_weight`; the `weight >= {min_edge_weight}` filter references **raw** weight (`'AND weight >='` present, `'newman_weight >=' not in query`). *MV-side `newman_weight = Σ 1/(k−1)` DDL is verified by human review + Task 1 grep — see human-verification section.* |
| stats-methodology.AC6.2 | unit | `url_cosharing/tests/test_analyzer.py` (`TestBuildGraph`), `quote_cosharing/tests/test_analyzer.py` (`TestBuildGraph`) | Duplicate + reversed-duplicate (a,b)/(b,a) pairs aggregate into one edge: raw weights summed (e.g. → `weight == 7`), Newman weights summed (`pytest.approx(1.7)`), URL/URI lists unioned; `graph.ecount() == 1`; `count_multiple() == [1]*ecount()` (no parallel edges); no `None` attributes. |
| stats-methodology.AC6.3 | unit | `url_cosharing/tests/test_analyzer.py`, `quote_cosharing/tests/test_analyzer.py` | Filter half: a `weight=1, newman_weight=99.0` pair is dropped at `min_edge_weight=2` (Newman weight can't rescue a thin raw edge). Leiden half: discriminating `TestClusterGraph` case — A–B/C–D (`newman_weight=5.0`) with bridge B–C (`weight=10, newman_weight=0.001`) yields 2 clusters at `resolution=0.05` (raw weights would merge to one), proving Leiden consumes `newman_weight`; `total_weight` still sums raw weight. |
| stats-methodology.AC6.4 | unit | `url_cosharing/tests/test_analyzer.py`, `quote_cosharing/tests/test_analyzer.py` | Batch construction equals the old per-edge loop: a **test-local reference implementation** replicating add-edge/`get_eid`/per-edge attribute assignment runs on duplicate-free input; assert identical vertex names, edge set (frozensets of name pairs), and per-edge `weight`/`newman_weight`/`shared_urls|shared_uris`. |

---

## AC7 — Schemas

| Criterion | Type | Coverage | Verifies |
|---|---|---|---|
| stats-methodology.AC7.1 (per-sidecar column-list) | unit | `signup_anomaly/tests/test_db.py`, `url_overdispersion/tests/test_db.py`, `quote_overdispersion/tests/test_db.py`, `account_entropy/tests/test_db.py`, `url_cosharing/tests/test_db.py`, `quote_cosharing/tests/test_db.py` | Each sidecar's insert `column_names` list has the expected shape (15 signup / 23 url / 22 quote-overdispersion / 17 account_entropy / cosharing lists) with new columns in the right adjacency (`q_value` after `p_value`, `volume_q_value` after `volume_p_value`, etc.); fetch mapping does not transpose median vs mean. |
| stats-methodology.AC7.1 (cross-repo consistency) | **human** | Phase 7 Task 7 sweep + human review | See human-verification section: the six `db.py` column lists are diffed against the ClickHouse `CREATE TABLE` DDL, but the DDL itself never executes in CI. |

---

## AC8 — Documentation (all human-verification)

| Criterion | Type | Verifies |
|---|---|---|
| stats-methodology.AC8.1 | **human** | `statistical-sidecars.md` documents NB, beta-binomial, FDR, normalized entropy, Newman weighting and includes quote_overdispersion + quote_cosharing sections — see human-verification section. |
| stats-methodology.AC8.2 | **human** | Per-sidecar README/CLAUDE.md match shipped behaviour (no stale z-test / 3.9-bit / Poisson-only claims) — see human-verification section. |
| stats-methodology.AC8.3 | **human** | `docs/calibration.md` exists with per-sidecar validation queries, healthy ranges, tuning levers — see human-verification section. |

---

## AC9 — Suites

| Criterion | Type | Verifies |
|---|---|---|
| stats-methodology.AC9.1 | **human** (running the suites is the verification) | `uv run pytest` passes in all six sidecars — see human-verification section for exact commands. |

---

## Human verification

Every criterion below cannot be fully closed by an automated test in CI, with the justification and a concrete approach.

### stats-methodology.AC7.1 — cross-repo schema consistency (partially automated)
**Why not fully automated:** The per-sidecar `test_db.py` unit tests pin each `db.py` insert column list, and Phase 7 Task 7 has a comm-based sweep diffing those lists against the `clickhouse-init` DDL. But the ClickHouse DDL lives in the sibling `skywatch-osprey` repo and is **never executed in CI** — there is no live ClickHouse in the test harness, so a name that exists in `db.py` but is misspelled/missing in the `CREATE TABLE` (or an `ALTER … ADD COLUMN` that never ran) would pass every sidecar's pytest yet break at insert time in prod.
**Approach:** Run the Phase 7 Task 7 Step 1 sweep for all six sidecars (02–07): extract insert names from each `db.py`, extract column names from the matching `clickhouse-init/*.sql`, `comm -23` to confirm no insert column is missing from its schema. Confirm each Phase's `ALTER TABLE … ADD COLUMN IF NOT EXISTS` count matches the added-column count (signup 1, url 6, quote-overdispersion 6, account_entropy 4, url/quote cosharing `newman_weight`). Confirm `01-init.sql` was not touched. A human confirms names line up name-for-name against the merged DDL, since ClickHouse inserts by name.

### stats-methodology.AC8.1 — statistical-sidecars.md content accuracy
**Why not automated:** Documentation-accuracy is a semantic match against merged code; grep can confirm keywords are present but not that the described maths matches what shipped.
**Approach:** Human review of `skywatch-osprey/docs/statistical-sidecars.md` against merged Phase 1–6 source. Confirm the NB/beta-binomial/FDR/normalized-entropy/Newman sections describe the actual formulas and defaults, and that quote_overdispersion and quote_cosharing sections now exist. Grep gate (Phase 7 Tasks 1–4) is a floor, not the verification: `negative binomial`, `q_value`, `medianExact`, `hour-of-day`, `beta-binomial`, `quote_overdispersion`, `Miller`, `interval_cv`, `newman`, `Jaccard`, `quote_cosharing`.

### stats-methodology.AC8.2 — per-sidecar README/CLAUDE.md accuracy
**Why not automated:** Same as AC8.1 — prose-vs-code fidelity needs judgement.
**Approach:** Human review that each sidecar's README/CLAUDE.md states its headline change (NB+FDR for signup/url/quote, normalized entropy+CV for account_entropy, Newman weighting for url/quote cosharing) and the renamed env vars. Stale-method grep (Phase 7 Task 7 Step 2) as a floor: no current-tense `normal-approximation` / `z-test` / `3.9 bits` / `3.9-bit` / bare "Poisson statistics" survives.

### stats-methodology.AC8.3 — calibration.md exists and is usable
**Why not automated:** The queries are meant to run post-deploy against live ClickHouse; CI has no such target, and query *usefulness* (healthy ranges, tuning levers) is editorial.
**Approach:** Human review that `osprey-sidecars/docs/calibration.md` exists with one section per sidecar, at least one runnable ClickHouse validation query each (`grep -c 'SELECT'` ≥ 6 as a floor), healthy-range guidance, tuning levers, and the CPM-resolution re-tuning procedure for the cosharing sidecars. Column/table names must match merged schemas (spot-check against `clickhouse-init`).

### stats-methodology.AC6.1 — MV Newman-weight DDL (partially automated)
**Why not fully automated:** The `fetch_pairs_query` structure test is automated, but `newman_weight = sum(1.0 / (k−1))` lives in the ClickHouse materialized-view DDL, which never executes in CI.
**Approach:** Phase 5/6 Task 1 grep confirms `newman_weight` + `k_url`/`k_uri` in the pairs `CREATE TABLE` and MV select, and `grep -v '^--' | grep -c DROP → 0` (no live DROPs in the init file). A human confirms the `sum(1.0/(k−1))` expression and the `HAVING length(...) >= 3` guard (ensures k−1 ≥ 2, no division by zero) read correctly against the merged DDL.

### stats-methodology.AC9.1 — six suites pass (the run is the verification)
**Why not automated in one shot:** No aggregate runner; each sidecar is an independent `uv` project. Running the suites *is* the acceptance test.
**Approach:** Run each, expecting zero failures:
```
cd signup_anomaly       && uv run pytest
cd url_overdispersion   && uv run pytest
cd quote_overdispersion && uv run pytest
cd account_entropy      && uv run pytest
cd url_cosharing        && uv run pytest
cd quote_cosharing      && uv run pytest
```
Lint gate per sidecar (each phase's final task): `uv run ruff check src tests && uv run ruff format --check src tests`.

---

## Pinned reference values (asserted exactly in the tests above)

| Value | Where | Meaning |
|---|---|---|
| `0.3125` | AC1.1 `test_counts.py` (×3) | NB(r=2, p=0.5), P(X≥3), `abs=1e-12` |
| `betainc(7, 4.5, 0.4)` ≈ 0.07519 | AC1.3 `test_counts.py` (×3) | non-integer-r cross-check (NOT `nbdtrc`) |
| `[0.02, 0.04, 0.04, 0.02]` | AC2.1 `test_fdr.py` (×3) | BH of `[0.01, 0.04, 0.03, 0.005]` |
| `186/1716` ≈ 0.108392 | AC3.1 `test_density.py` (×2) | BetaBinom(n=10, α=β=2), P(X≥9), `rel=1e-9` |
| `11/1024` | AC3.2 `test_density.py` (×2) | Binomial(10, 0.5), P(X≥9) fallback |
| `≈ 0.322745` | AC5.2 `test_analyzer.py` | normalized_entropy([5,5], 24), closed form `(1+1/(20·ln2))/log2(10)`, `rel=1e-9` |
| `1.7` | AC6.2 `test_analyzer.py` (×2) | summed Newman weight after duplicate aggregation |
```

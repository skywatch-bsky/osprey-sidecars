# Human Test Plan — Statistical Methodology Fixes (2026-07-06)

Generated from the test-analyst coverage validation at HEAD `d3d5b13` (branch `stats-methodology`).
Automated coverage: 24/24 acceptance criteria covered by the six sidecar suites (745 tests).
This plan covers what automation cannot: cross-repo schema execution, MV DDL, documentation
fidelity, and a live end-to-end cycle.

## Prerequisites

- Both repos checked out: `/Users/scarndp/dev/skywatch/osprey-sidecars` (branch `stats-methodology`) and sibling `skywatch-osprey`.
- `uv` installed; each sidecar is an independent `uv` project (no aggregate runner).
- A ClickHouse instance reachable for the deploy-time checks (there is **no** live ClickHouse in CI — the schema-consistency and calibration items below are exactly the gaps automated tests cannot close).
- All six suites green before starting (the AC9.1 gate):

  ```bash
  cd signup_anomaly       && uv run pytest    # expect 127 passed
  cd url_overdispersion   && uv run pytest    # expect 145 passed
  cd quote_overdispersion && uv run pytest    # expect 138 passed
  cd account_entropy      && uv run pytest    # expect  90 passed
  cd url_cosharing        && uv run pytest    # expect 123 passed
  cd quote_cosharing      && uv run pytest    # expect 122 passed
  ```

  Plus per sidecar: `uv run ruff check src tests && uv run ruff format --check src tests`.

## Phase 1: Cross-repo schema consistency (AC7.1 — human)

The load-bearing manual check: `db.py` insert column lists are unit-pinned per sidecar, but the
ClickHouse `CREATE TABLE` DDL lives in `skywatch-osprey/clickhouse-init/` and never executes in CI.
A name that exists in `db.py` but is misspelled/missing in the DDL passes every pytest yet breaks at
insert time. ClickHouse inserts **by name**, so this must line up name-for-name.

| Step | Action | Expected |
|------|--------|----------|
| 1.1 | For each sidecar `NN`, extract insert names from `db.py` and column names from `clickhouse-init/NN-*.sql`, then `comm -23 <(sorted db.py names) <(sorted DDL names)` | Empty output — no insert column missing from its schema. Pairs: 02↔signup_anomaly, 03↔url_overdispersion, 07↔quote_overdispersion, 04↔account_entropy, 05↔url_cosharing, 06↔quote_cosharing |
| 1.2 | Count `ALTER TABLE … ADD COLUMN IF NOT EXISTS` per migrated table | signup 1, url 6, quote-overdispersion 6, account_entropy 4; cosharing 05/06 have 0 (fresh tables, columns in `CREATE TABLE`) |
| 1.3 | Confirm `01-init.sql` untouched and no live `DROP` in any init file (`grep -v '^--' NN.sql \| grep -c DROP` → 0) | All zero |
| 1.4 | Eyeball the merged DDL for the migrated tables: `q_value` after `p_value` (signup); `volume_q_value` after `volume_p_value` and `density_q_value` after `density_p_value` (url/quote overdispersion); `newman_weight Float64` (cosharing pairs) | Column names and adjacency match the unit-pinned `db.py` lists name-for-name |

## Phase 2: Newman-weight materialized-view DDL (AC6.1 — human)

`fetch_pairs_query` structure is unit-tested, but `newman_weight = Σ 1/(k−1)` lives in the MV DDL
that never runs in CI.

| Step | Action | Expected |
|------|--------|----------|
| 2.1 | Read `clickhouse-init/05-url-cosharing.sql` and `06-quote-cosharing.sql` | `sum(1.0 / (s1.k_url - 1)) AS newman_weight` (url) and `sum(1.0 / (s1.k_uri - 1)) AS newman_weight` (quote) present in the MV select |
| 2.2 | Confirm the division guard | `HAVING length(sharers) >= 3` (url) / `HAVING length(quoters) >= 3` (quote) ensures k ≥ 3 ⇒ k−1 ≥ 2, no divide-by-zero |
| 2.3 | Confirm `newman_weight Float64` column exists in the pairs `CREATE TABLE` | Present in both |

## Phase 3: Documentation accuracy (AC8.1 / 8.2 / 8.3 — human)

Grep confirms keywords; a human must confirm the described maths matches shipped code.

| Step | Action | Expected |
|------|--------|----------|
| 3.1 | Read `skywatch-osprey/docs/statistical-sidecars.md` against merged Phase 1–6 source | NB, beta-binomial, FDR, normalized-entropy (Miller–Madow), Newman sections describe the actual formulas/defaults; `quote_overdispersion` and `quote_cosharing` sections exist |
| 3.2 | Read each sidecar's `README`/`CLAUDE.md` | Each states its headline change (NB+FDR for signup/url/quote; normalized entropy+CV for account_entropy; Newman weighting for url/quote cosharing) and renamed env vars; no current-tense `z-test` / `normal-approximation` / `3.9 bits` / bare "Poisson statistics" survives |
| 3.3 | Read `osprey-sidecars/docs/calibration.md` | Exists; one section per sidecar; ≥1 runnable ClickHouse validation query each (25 SELECTs total); healthy-range guidance, tuning levers, and the CPM-resolution re-tuning procedure for cosharing. Spot-check column/table names against merged `clickhouse-init` schemas |

## End-to-End: Deploy against live ClickHouse and observe one full cycle

**Purpose:** Validate the entire rewrite end-to-end — the one thing no unit test can do, since CI
has no ClickHouse and the FakeDb cycle stubs the real insert. This exercises the actual
`column_names` insert path against the actual DDL for all six sidecars.

1. Apply the merged `clickhouse-init` DDL (all seven files) to a ClickHouse with `osprey_execution_results` populated.
2. Run each sidecar for at least one full cycle (`docker compose up <sidecar>` or the sidecar's run loop) against that instance.
3. **Expected:** each sidecar completes a cycle with **zero insert errors** — the real `column_names` list matches the live table schema name-for-name.
4. Query each results table:
   - `pds_signup_anomalies`: `q_value` populated (not all 1.0/NULL); `is_anomaly=1` rows all have `observed_count > 0` and `q_value <` the daily/hourly threshold.
   - `url_overdispersion_results` / `quote_overdispersion_results`: `volume_q_value`, `density_q_value` populated; anomalies satisfy `volume_q < target OR density_q < target`; diagnostic columns (`rolling_volume_median`, `rolling_density_mean`, variances) non-NULL for entity-baseline rows.
   - `account_entropy_results`: normalized entropies ∈ [0,1]; `is_bot_like` rows satisfy `hourly_flag AND (interval_flag OR cv_flag)`; the new CV columns present.
   - url/quote cosharing cluster tables: clusters present; `total_weight` (raw) and Newman-weighted edges wired; run the calibration.md CPM-resolution query and confirm cluster counts land in the documented healthy range.
5. **Expected:** each table's row shapes and value ranges match the calibration.md "healthy ranges." Re-run the Phase 1.1 `comm` check against the **live** `system.columns` for final name-for-name confirmation.

## Human Verification Required (summary)

| Criterion | Why Manual | Steps |
|---|---|---|
| AC7.1 cross-repo | DDL never executes in CI; name/adjacency mismatch passes pytest, breaks at insert | Phase 1 + E2E steps 1–3 |
| AC6.1 MV DDL | `Σ 1/(k−1)` lives in MV DDL, uncompiled in CI | Phase 2 |
| AC8.1/8.2/8.3 | Prose-vs-code semantic fidelity needs judgement; grep is a floor | Phase 3 |
| AC9.1 | No aggregate runner; running the six suites *is* the acceptance test | Prerequisites block |

## Traceability

| Acceptance Criterion | Automated Test | Manual Step |
|---|---|---|
| AC1.1–1.4 | `*/tests/test_counts.py`, `*/tests/test_analyzer.py` | E2E step 4 (value ranges) |
| AC2.1–2.4 | `*/tests/test_fdr.py`, `test_analyzer.py`, `test_main.py` | E2E step 4 |
| AC3.1–3.4 | `{url,quote}/tests/test_density.py`, `test_analyzer.py` | E2E step 4 |
| AC4.1–4.4 | `*/tests/test_queries.py`, `test_analyzer.py` | E2E step 4 (diagnostic cols) |
| AC5.1–5.4 | `account_entropy/tests/test_analyzer.py`, `test_config.py`, `test_db.py`, `test_main.py` | E2E step 4 |
| AC6.1 | `*_cosharing/tests/test_queries.py` | Phase 2 (MV DDL) |
| AC6.2–6.4 | `*_cosharing/tests/test_analyzer.py` | E2E step 4 |
| AC7.1 (column-list) | all six `test_db.py` (`TestInsertColumnList` + fetch-order) | — |
| AC7.1 (cross-repo) | — | Phase 1 + E2E steps 1–3 |
| AC8.1 | — (grep floor only) | Phase 3.1 |
| AC8.2 | — (grep floor only) | Phase 3.2 |
| AC8.3 | — (grep floor only) | Phase 3.3 |
| AC9.1 | — (the run is the test) | Prerequisites |

Every acceptance criterion maps to either an automated test or a manual step; none is orphaned.

# Human Test Plan — Density-Based Dismantling for URL Co-Sharing

Generated 2026-07-07 from `docs/implementation-plans/2026-07-06-density-dismantling/test-requirements.md`.
Automated coverage: PASS (16/16 automatable acceptance criteria pinned; suite 186 passed).
This plan covers the deploy-time verification CI structurally cannot close (no ClickHouse in CI).

## Prerequisites

- A ClickHouse instance reachable from the sidecar (CI has none — this is the whole reason this plan exists).
- `osprey_execution_results` populated with a realistic window of `app.bsky.feed.post` create events carrying `FacetLinkList`.
- The merged DDL from `skywatch-osprey/clickhouse-init/05-url-cosharing.sql` **not yet applied** (H1 applies it).
- `cd url_cosharing && uv run pytest` green locally — confirmed at plan generation: **186 passed**. Re-run if the tree has changed.
- The sidecar's ClickHouse env vars (`CLICKHOUSE_*`, `URL_COSHARING_*`) set for the target instance.

## Phase H1: DDL apply + name-for-name schema reconcile

Closes the live half of AC1.5, AC2.5, AC3.2. Table inserts in ClickHouse bind **by column name** — a name that pytest never sees (it lives in the SQL file) passes every unit test and blows up at first insert.

| Step | Action | Expected |
|------|--------|----------|
| 1 | Static pre-check: `grep -c 'ADD COLUMN IF NOT EXISTS' clickhouse-init/05-url-cosharing.sql` | `2` (the two `ALTER … ADD COLUMN` migrations) |
| 2 | `grep -c 'CREATE TABLE IF NOT EXISTS default.url_cosharing_runs' clickhouse-init/05-url-cosharing.sql` | `1` |
| 3 | `grep -v '^--' clickhouse-init/05-url-cosharing.sql \| grep -c 'DROP'` | `0` (no destructive statements) |
| 4 | Apply the merged DDL to the live instance | Runs clean, no errors; idempotent `IF NOT EXISTS` re-run is a no-op |
| 5 | Reconcile `url_cosharing_runs` live columns vs `db.py` `insert_run` (13 names): `comm -23` of sorted `system.columns` names vs sorted `insert_run` `column_names` | **empty** both directions |
| 6 | Reconcile `url_cosharing_clusters` live columns vs `db.py` `insert_clusters` (16 names) the same way | **empty** both directions |
| 7 | Confirm adjacency: in `url_cosharing_clusters`, `mean_edge_similarity` and `subgraph_density` follow `jaccard_score` (or wherever `db.py` orders them) | Column order matches `insert_clusters` `column_names` exactly |

Note for step 7: the requirements prose says the new columns "end" the list; the code puts them at indices 11-12 with three columns after. Reconcile against **`db.py`'s actual order**, not the prose.

## Phase H2: Live calibration run

Closes AC4.3's live half (CI only unit-tests the pure `format_surface`).

| Step | Action | Expected |
|------|--------|----------|
| 1 | On the populated instance: `cd url_cosharing && uv run python -m url_cosharing.calibrate` | Clean exit, no traceback |
| 2 | Inspect stdout | TSV header `edge_quantile\tcentrality_quantile\tmin_component_density\tsurviving_nodes\tsurviving_edges`, one line per grid cell (`len(edge_grid)×len(centrality_grid)` rows), then the two-line summary footer |
| 3 | Confirm the surface is non-degenerate | `min_component_density` values vary across cells and sit in `[0,1]`; survivor counts decrease as quantile thresholds rise |
| 4 | Confirm teardown | Process exits without hanging (proves `CosharingDb.close()` wires against the real client) |

Purpose: proves the real read path, the pure Core pipeline, and the DB client teardown all wire against production shapes before an unattended run touches them.

## Phase H3: Full write cycle observed live

Closes the production-read half of AC1.1 and the live-insert halves of AC1.5 / AC2.5 / AC3.2 / AC3.4. This is where the coverage nuances (exact stage counts, `cluster_count` equality, full membership coverage) get observed against real data.

| Step | Action | Expected |
|------|--------|----------|
| 1 | With H1's DDL applied, run one full cycle: `docker compose up url_cosharing` (or the sidecar run loop) against the populated instance | **Zero insert errors** |
| 2 | `SELECT * FROM url_cosharing_runs WHERE run_date = today()` | Exactly one row with **populated** stage counts — this is the proof the daily run read `osprey_execution_results` and not the pairs MV (AC1.1 production half) |
| 3 | Sanity-check that run row | `accounts_raw >= accounts_eligible`, `urls_eligible > 0`, `graph_edges >= 0`; `knee_found`/`guardrail_triggered`/`edge_quantile`/`centrality_quantile`/`min_component_density` mutually consistent (if `knee_found=false` then quantiles and density are `0.0`) |
| 4 | On a flagging day: `SELECT cluster_id, mean_edge_similarity, subgraph_density FROM url_cosharing_clusters WHERE run_date = today()` | Every row has **non-null** `mean_edge_similarity` and `subgraph_density` in `[0,1]` (AC3.2 live insert) |
| 5 | Verify `cluster_count` equality (the nuance the unit test only bounds): `SELECT cluster_count FROM url_cosharing_runs WHERE run_date=today()` vs `SELECT count() FROM url_cosharing_clusters WHERE run_date=today()` | **Equal** — confirms `cluster_count` = written (non-death) rows, not the raw Leiden partition count |
| 6 | Verify full membership coverage (the nuance the unit test only lower-bounds): every member of every written cluster appears in `url_cosharing_membership` for `today()`; no membership row references a `cluster_id` absent from `url_cosharing_clusters` | Membership DID set == union of all written-cluster members; death-event clusters absent from both tables (AC3.4) |
| 7 | Health-band check: `flagged_pct = flagged_accounts / accounts_eligible` (run the `docs/calibration.md` runs-table health query) | Lands in the paper's **0.4–1.5%** band on a normal day |
| 8 | Final: re-run H1's `comm` name-for-name check against the **live** `system.columns` after inserts have happened | Still empty — confirms no silent schema drift under real writes |

## End-to-End: Empty / fully-filtered day

Purpose: validates AC1.5's live guarantee that a barren day still completes and writes a zero-count run row (unit + FakeDb prove the wiring; only a live run proves the real insert of a zero row).

1. Point the sidecar at a window with no qualifying shares (or set `URL_COSHARING_MIN_UNIQUE_URLS` absurdly high so every account filters out).
2. Run one cycle.
3. `SELECT * FROM url_cosharing_runs WHERE run_date = today()` → exactly one row, all stage counts `0`, `knee_found = false`, `guardrail_triggered = false`, quantiles and `min_component_density` all `0.0`.
4. `SELECT count() FROM url_cosharing_clusters WHERE run_date = today()` and same for `url_cosharing_membership` → both `0`.
5. Confirm the run exits normally (no exception, no partial write).

## End-to-End: No-knee day (floor never met)

Purpose: validates AC2.3's live behaviour — a day with data but no density transition flags nobody and records `knee_found=false`.

1. On a populated instance, force `URL_COSHARING_DENSITY_FLOOR=1.0` (a perfect-clique-only floor rarely met by organic data).
2. Run one cycle.
3. `url_cosharing_runs` today's row: `knee_found = false`, `edge_quantile = 0.0`, `centrality_quantile = 0.0`, `flagged_accounts = 0`, `cluster_count = 0`.
4. `url_cosharing_clusters` and `url_cosharing_membership` have zero rows for `today()`.
5. Confirm the run still wrote its metadata row (a no-knee day is not a failed day).

## Human Verification Required

| Criterion | Why Manual | Steps |
|-----------|------------|-------|
| H1 — DDL applies + name-for-name reconcile | DDL lives in the sister repo's `clickhouse-init` and never executes in CI; a misnamed column passes every pytest but breaks at insert time (ClickHouse binds by name) | Phase H1 steps 1–7 |
| H2 — Calibration live run | `uv run python -m url_cosharing.calibrate` fetches a real window; CI only unit-tests the pure `format_surface` | Phase H2 steps 1–4 |
| H3 — Production read-path + full write cycle | FakeDb stubs both the real insert and the real `osprey_execution_results` read; only a live cycle proves `column_names` matches the live schema and that the run reads the source table not the pairs MV | Phase H3 steps 1–8 + both barren/no-knee E2E scenarios |

## Traceability

| Acceptance Criterion | Automated Test | Manual Step |
|----------------------|----------------|-------------|
| AC1.1 | `test_queries.py::TestFetchUrlSharesQuery`, `test_db.py::TestFetchUrlShares`, `test_main.py::TestRunCycle` | H3 step 2 (production read) |
| AC1.2 | `test_queries.py::TestFetchUrlSharesQuery`, `test_similarity.py::TestSqlFinalRowsNotRefiltered` | — |
| AC1.3 | `test_queries.py::TestFetchUrlSharesQuery`, `test_similarity.py::TestSqlFinalRowsNotRefiltered` | — |
| AC1.4 | `test_similarity.py::TestTfidfTransform`, `TestBuildSimilarityGraph` | — |
| AC1.5 | `test_similarity.py::TestSimilarityNetwork`/`TestBuildShareMatrix`, `test_dismantling.py::TestDismantleNoTransition`, `test_main.py` (empty) | H1 + H3; E2E "Empty/fully-filtered day" |
| AC2.1 | `test_dismantling.py::TestDismantleSurface` | — |
| AC2.2 | `test_dismantling.py::TestDismantleKnee` | — |
| AC2.3 | `test_dismantling.py::TestDismantleNoTransition`, `test_main.py` (no-knee) | E2E "No-knee day" |
| AC2.4 | `test_dismantling.py::TestDismantleGuardrails` | — |
| AC2.5 | `test_db.py::TestInsertRun`, `test_main.py` (full-cycle) | H1 + H3 steps 2–5 (stage counts + cluster_count equality) |
| AC3.1 | `test_analyzer.py::TestClusterCore` | — |
| AC3.2 | `test_analyzer.py::TestClusterCore`/`TestComputeTemporalMetrics`, `test_db.py` (16-col), `test_queries.py` | H1 step 7 + H3 step 4 |
| AC3.3 | `test_analyzer.py::TestComputeEvolution`/`TestComputeJaccard` | — |
| AC3.4 | `test_main.py` (full-cycle, membership coverage) | H3 step 6 (full coverage + death skip) |
| AC4.1 | `test_config.py::TestAnalysisConfig` | — |
| AC4.2 | purity grep gate; `test_similarity.py` + `test_dismantling.py` (DB-free) | — |
| AC4.3 | `cd url_cosharing && uv run pytest` (186 passed); `test_calibrate.py::TestFormatSurface`; removal sweeps | H2 (live calibrate) |

Every acceptance criterion maps to a passing automated test or a documented human step; none is orphaned. The three deploy-time items (H1/H2/H3) are exactly the ones CI structurally cannot close, as the requirements doc anticipated.

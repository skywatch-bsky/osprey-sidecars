# Test Requirements — Density-Based Dismantling for URL Co-Sharing (2026-07-06)

Traces every acceptance criterion in `docs/design-plans/2026-07-06-density-dismantling.md`
(`density-dismantling.AC1.1` … `density-dismantling.AC4.3`) to the automated test that pins it
and/or the human verification that closes the gap CI cannot.

All automated tests live in the `url_cosharing` sidecar and run via `cd url_cosharing && uv run pytest`.
There is **no ClickHouse in CI**: DDL execution, the calibration module's live run, and the
production read-path are deploy-time human checks, exactly as in the sister plan
`docs/test-plans/2026-07-06-stats-methodology.md`.

Phase file "Testing" sections are the ground truth for what each test must assert. Where planning
diverged from the raw design (single-pass filter semantics, `>= edge_epsilon` boundary, 2-D
forward-difference knee rule, `cluster_count` = written rows), the criterion is rationalized
against the shipped decision below and that decision is the thing the test pins.

---

## density-dismantling.AC1: Similarity network construction

### density-dismantling.AC1.1
> **density-dismantling.AC1.1 Success:** A daily run fetches per-account URL share counts over the configured rolling window (default 7 days) from `osprey_execution_results`, not from `url_cosharing_pairs`.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (SQL builder) | `url_cosharing/tests/test_queries.py` (`TestFetchUrlSharesQuery`) | `fetch_url_shares_query(config)` reads `FROM {config.source_table}` — assert `'osprey_execution_results'` present and `'url_cosharing_pairs'` absent; window bounds `yesterday() - 6` / `<= yesterday()` for default `window_days=7`; `Collection = 'app.bsky.feed.post'`, `OperationKind = 'create'`, `arrayJoin(FacetLinkList)`, `length(FacetLinkList) > 0`; selects `did`, `url`, `share_count`. Custom `window_days=1` → `yesterday() - 0`. |
| unit (typed fetch) | `url_cosharing/tests/test_db.py` (`TestFetchUrlShares`) | `CosharingDb.fetch_url_shares` maps `result_rows` positionally into `UrlShareRow(did, url, share_count)`, coerces `share_count` to `int`, passes `settings={'max_execution_time': 300}`, returns `[]` on empty result. |
| integration (pipeline) | `url_cosharing/tests/test_main.py` (`FakeDb` full-cycle) | `run_cycle` calls `db.fetch_url_shares(fetch_url_shares_query(...))`; the pairs path (`fetch_pairs`/`build_graph`/`PairRow`) is gone. Phase 5 Task 5 sweep `grep -rn 'PairRow\|build_graph\|fetch_pairs\|pairs_table' url_cosharing/src url_cosharing/tests` must return nothing (enforced as a checklist gate, not a pytest). |

**Human verification:** the *production-behaviour half* — that the daily unattended run actually queries
`osprey_execution_results` and not the pairs MV against a live instance — is observed at deploy time via
the `url_cosharing_runs` row appearing with populated stage counts (see Human Verification item **H3**).
The FakeDb integration test proves the wiring; only live execution proves the real read.

---

### density-dismantling.AC1.2
> **density-dismantling.AC1.2 Success:** Accounts sharing fewer than `min_unique_urls` (default 10) unique URLs in the window are excluded before matrix construction.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (SQL builder) | `url_cosharing/tests/test_queries.py` (`TestFetchUrlSharesQuery`) | `HAVING uniqExact(url) >= 10` present with default config; changing `min_unique_urls` changes the literal. |
| unit (Python filter) | `url_cosharing/tests/test_similarity.py` (`TestFilterShares`) | `filter_shares` drops an account with 2 unique URLs when `min_unique_urls=3`; keeps an account with exactly 3 (**boundary**). Activity counted over **raw** window rows (single-pass semantics). |

**Human verification:** none (fully automated at both SQL and Python layers).

---

### density-dismantling.AC1.3
> **density-dismantling.AC1.3 Success:** URLs shared by fewer than `min_url_sharers` (default 5) accounts, or with document frequency above the `max_url_df_pctl` (default 0.90) percentile, are excluded before TF-IDF.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (SQL builder) | `url_cosharing/tests/test_queries.py` (`TestFetchUrlSharesQuery`) | `df >= 5` present with default config; df ceiling via the repo idiom — assert `f'quantile({config.max_url_df_pctl})(' in query` (f-string renders `0.90` as `0.9`, so match the formatted value, not a hard literal); overriding `min_url_sharers`/`max_url_df_pctl` moves both. |
| unit (Python filter) | `url_cosharing/tests/test_similarity.py` (`TestFilterShares`) | **Floor:** URL shared by 1 account dropped at `min_url_sharers=2`, exactly 2 kept (boundary). **Ceiling (hand-computed):** dfs `[1,1,1,1,10]` with `max_url_df_pctl=0.5` → `np.quantile` ceiling `1.0` → df-10 URL dropped, df-1 URLs kept subject to floor. |

**Rationale note (single-pass):** df is computed over the raw window before any account is dropped, and
activity before any URL is dropped; both filters then applied together. A `TestFilterShares` regression pin
asserts an account whose *raw* unique-URL count passes but whose *surviving* URLs number fewer than
`min_unique_urls` is **still kept** — mirroring the SQL. Python's `np.quantile` (linear interp) and
ClickHouse's approximate `quantile` may disagree near the ceiling; the Python check is defense-in-depth,
not an exactness contract, so no cross-engine equality test is required.

**Human verification:** none.

---

### density-dismantling.AC1.4
> **density-dismantling.AC1.4 Success:** Edge weights are cosine similarities between account TF-IDF vectors, in [0, 1]; similarities below `edge_epsilon` are not materialized as edges.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (TF-IDF) | `url_cosharing/tests/test_similarity.py` (`TestTfidfTransform`) | Hand-computed `tf × ln(N/df)`, L2-normalized rows; a URL with `df == N` gets idf 0 and yields a zero row (no NaN); every nonzero row has L2 norm ≈ 1.0; all values finite and ≥ 0. |
| unit (graph) | `url_cosharing/tests/test_similarity.py` (`TestBuildSimilarityGraph`) | Identical share vectors → one edge, `similarity == pytest.approx(1.0)` (pins the `np.minimum(data, 1.0)` clamp for the upper bound). Disjoint URL sets → cosine 0 → **no edge**. All weights in `[0, 1]` on a 4-account mixed case; vertex `name` order matches `matrix.accounts`. |

**Rationale note (`>= edge_epsilon` boundary):** the design says similarities *below* `edge_epsilon` are not
materialized; the implementation keeps the boundary with `keep = sims.data >= edge_epsilon`.
`TestBuildSimilarityGraph` pins this explicitly — a pair just **below** `edge_epsilon` gets **no** edge; a
pair **exactly at** `edge_epsilon` gets **one** edge.

**Human verification:** none.

---

### density-dismantling.AC1.5
> **density-dismantling.AC1.5 Edge:** An empty or fully-filtered input produces an empty graph; the run completes normally and writes a run-metadata row with zero counts.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (empty-graph half) | `url_cosharing/tests/test_similarity.py` (`TestSimilarityNetwork`, `TestBuildShareMatrix`) | Empty input → `graph.vcount() == 0`, `graph.ecount() == 0`, all stage counts zero, nothing raises; `build_share_matrix([])` → shape `(0,0)`, empty tuples. Non-empty-but-fully-filtered input (every account below `min_unique_urls`) → `accounts_raw > 0`, `accounts_eligible == 0`, empty graph, completes normally. |
| unit (dismantler on empty) | `url_cosharing/tests/test_dismantling.py` (`TestDismantleNoTransition`) | Empty graph and vertices-but-zero-edges graph → empty `DismantlingResult`, empty surface, no exception (eigenvector centrality never called). |
| integration (run-metadata half) | `url_cosharing/tests/test_main.py` (empty-input case) | `FakeDb.url_shares = []` → cycle completes, **no** cluster/membership inserts, exactly one `insert_run` with all-zero stage counts and `knee_found=False`; `delete_run_date` still called for clusters, membership, AND runs tables. |

**Human verification:** the run-metadata row is verifiably written with zero counts against a live table only
at deploy time (Human Verification item **H1** — DDL apply — plus the E2E cycle in **H3**).

---

## density-dismantling.AC2: Density-based dismantling

### density-dismantling.AC2.1
> **density-dismantling.AC2.1 Success:** Grid search over the configured edge-similarity × eigenvector-centrality quantile grids produces a minimum-component-density surface (isolates dropped before density is computed).

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit | `url_cosharing/tests/test_dismantling.py` (`TestDismantleSurface`) | `result.surface` has `len(edge_grid) × len(centrality_grid)` cells with the correct quantile labels (row-major over `(edge_grid, centrality_grid)`). **Isolates dropped before density:** a hand-built graph where a mid-grid cell strands a degree-0 vertex — that cell's `surviving_nodes` excludes it and `min_component_density` reflects only ≥2-vertex components (pin an exact expected density, e.g. `1.0` for a surviving 3-clique). All density values in `[0, 1]`. |

**Rationale note:** per-cell order is fixed by planning — keep vertices with centrality ≥ threshold (induced
subgraph) → delete edges with similarity < threshold → delete degree-0 vertices → only then compute
per-component density. Every surviving component has ≥ 2 vertices, so `density()` is well-defined; a cell
with nothing surviving records `0.0`.

**Human verification:** none.

---

### density-dismantling.AC2.2
> **density-dismantling.AC2.2 Success:** Knee detection selects the threshold pair at the largest discrete jump in minimum component density whose resulting density meets `density_floor`; a synthetic graph with a planted dense core plus organic background recovers the planted core.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit | `url_cosharing/tests/test_dismantling.py` (`TestDismantleKnee`) | On `planted_core_graph()` (8-node 0.9-weight clique `core0..core7` + ~30 sparse organic `bg*` vertices + weak bridges) with grids spanning 0.5–0.99, `density_floor=0.5`, generous `max_flagged_fraction=0.5`, `min_cluster_size=3`: `knee_found is True` and `set(result.core.vs['name']) == {'core0'..'core7'}` — planted core recovered exactly. Chosen `min_component_density >= density_floor`; chosen quantiles are members of the input grids. **Determinism:** two calls on the same graph give identical chosen quantiles and core membership. |

**Rationale note (2-D forward-difference knee rule):** the design's "largest discrete jump in minimum density"
is generalized to 2-D as `jump(i,j) = max(d[i][j] − d[i−1][j], d[i][j] − d[i][j−1])` (missing-predecessor
terms omitted; cell `(0,0)` is never a candidate). Candidates rank by `(jump, density, eq, cq)` descending;
first cell passing the density floor **and** guardrails wins. The test asserts the recovered core and the
floor constraint, which is the observable contract of that rule.

**Human verification:** none (synthetic planted-core recovery is deterministic and fully automated).

---

### density-dismantling.AC2.3
> **density-dismantling.AC2.3 Failure:** When no cell satisfies the density floor (no transition), zero accounts are flagged and the run row records that no threshold was selected.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (core half) | `url_cosharing/tests/test_dismantling.py` (`TestDismantleNoTransition`) | Uniform sparse graph (e.g. 20-node ring at 0.2) with `density_floor=0.9` → `knee_found is False`, `core.vcount() == 0`, `edge_quantile == 0.0`, `centrality_quantile == 0.0`, `min_component_density == 0.0`, `guardrail_triggered is False`. Disconnected input (4-clique @ 0.9 + separate 5-ring @ 0.2) completes without exception and returns a fully-populated surface — graceful handling, not cross-component score comparability. |
| integration (pipeline half) | `url_cosharing/tests/test_main.py` (no-knee case) | Populated fixture with `density_floor=1.0` forced → zero clusters written, run row records `knee_found=False`, `edge_quantile == 0.0`. |

**Human verification:** none. (Operational note, not a test requirement: monitoring alerts on **consecutive**
`knee_found=false` days only when paired with rising `accounts_eligible` — an operator judgement documented in
`docs/calibration.md`, not an acceptance test.)

---

### density-dismantling.AC2.4
> **density-dismantling.AC2.4 Failure:** When the selected thresholds would flag more than `max_flagged_fraction` of eligible accounts, the candidate is rejected and the next-best candidate (or none) is used; the guardrail activation is recorded.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (reject → none) | `url_cosharing/tests/test_dismantling.py` (`TestDismantleGuardrails`) | Graph constructed so **no** candidate fits under a small `max_flagged_fraction` (e.g. planted core over 38 vertices, `max_flagged_fraction=0.05` → cap ≈ 1.9 nodes) → `knee_found is False` **and** `guardrail_triggered is True`. |
| unit (reject → next-best) | `url_cosharing/tests/test_dismantling.py` (`TestDismantleGuardrails`) | Two density plateaus (large medium-dense group + small very-dense group) where the largest-jump candidate over-flags but a later candidate (small group) passes → `knee_found is True`, `core` is the small group, `guardrail_triggered is True`. |
| unit (min-survivor guardrail) | `url_cosharing/tests/test_dismantling.py` (`TestDismantleGuardrails`) | Winning cell would leave 2 survivors with `min_cluster_size=3` → rejected. |

**Rationale note:** "flagged accounts" = vertices of the surviving core at the chosen cell; the fraction is
measured against `graph.vcount()` (post-filter eligible population). Rejection sets `guardrail_triggered=True`
and the walk continues; if a floor-passing candidate was rejected but none ultimately passes,
`knee_found=False` with `guardrail_triggered=True`.

**Human verification:** none.

---

### density-dismantling.AC2.5
> **density-dismantling.AC2.5 Success:** The chosen quantile pair, resulting minimum density, and account/URL counts at each filter stage are written to the run-metadata table.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (write path) | `url_cosharing/tests/test_db.py` (`TestInsertRun`) | `insert_run` passes the exact 13-name `column_names` list matching the `url_cosharing_runs` DDL order and one data row mapped from a `RunMetadata` instance (this is the cross-repo name-for-name guard against `clickhouse-init/05-url-cosharing.sql`). |
| integration (populated) | `url_cosharing/tests/test_main.py` (full-cycle case) | On the coordinated fixture: run row has `knee_found=True`, correct `flagged_accounts` and `cluster_count`, and stage counts (`accounts_raw`, `accounts_eligible`, `urls_eligible`, `graph_edges`, chosen `edge_quantile`/`centrality_quantile`, `min_component_density`) matching the fixture. |

**Rationale note (`cluster_count` = written rows):** `cluster_count` is `len(cluster_rows)` — the rows actually
written (non-`death` events), **not** the raw Leiden partition count. `insert_run` is the last write of the
cycle so it can reference the constructed cluster-row list. The integration test asserts `cluster_count`
against the written rows, not the partition count.

**Human verification:** the row landing in the live `url_cosharing_runs` table with the DDL applied is item **H3**.

---

## density-dismantling.AC3: Hybrid clustering and outputs

### density-dismantling.AC3.1
> **density-dismantling.AC3.1 Success:** Leiden CPM (cosine-similarity edge weights) decomposes the surviving core; clusters below `min_cluster_size` are dropped.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit | `url_cosharing/tests/test_analyzer.py` (`TestClusterCore`) | Two well-separated similarity cliques decompose into two clusters; a cluster with fewer than `min_cluster_size` members is dropped; empty core → `[]`. Leiden honours `weights='similarity'` — pin as `TestClusterGraph` did for `newman_weight` (weights chosen so CPM at the test resolution separates the cliques **only if** similarity weights are used). |

**Human verification:** none.

---

### density-dismantling.AC3.2
> **density-dismantling.AC3.2 Success:** Cluster rows carry the new similarity metrics (`mean_edge_similarity`, `subgraph_density`) alongside existing columns; `total_weight` keeps its co-share-count semantics (Σ over cluster URLs of C(sharing members, 2)).

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (`total_weight`) | `url_cosharing/tests/test_analyzer.py` (`TestClusterCore`) | Hand-computed: 3 members, `u1` shared by all 3 and `u2` by 2 → `total_weight == C(3,2) + C(2,2) == 3 + 1 == 4`; `unique_urls == 2` (URLs with ≥ 2 member sharers only); a URL shared by 1 member contributes 0 to both. |
| unit (similarity metrics) | `url_cosharing/tests/test_analyzer.py` (`TestClusterCore`) | 3-clique with similarities `[0.9, 0.8, 0.7]` → `mean_edge_similarity == pytest.approx(0.8)`, `subgraph_density == pytest.approx(1.0)`; 3-path (2 edges) → `subgraph_density == pytest.approx(2/3)`. `sample_urls` ranked by cluster TF-IDF mass (high within-cluster mass beats widely-shared; zero-mass URLs excluded; ties break lexicographically via stable argsort). |
| unit (temporal propagation) | `url_cosharing/tests/test_analyzer.py` (`TestComputeTemporalMetrics`) | `compute_temporal_metrics` copies `mean_edge_similarity`/`subgraph_density` through into `TimestampedCluster` (extend the existing field-preservation test). |
| unit (column list) | `url_cosharing/tests/test_db.py` (`TestInsertColumnList`), `url_cosharing/tests/test_queries.py` | `insert_clusters` `column_names` is the exact 16-name list ending `mean_edge_similarity`, `subgraph_density`; `insert_clusters_query` lists the two new columns with 16 `?` placeholders. |

**Human verification:** column adjacency/naming in the live `url_cosharing_clusters` table matches `db.py`
name-for-name — item **H1** (DDL apply) + item **H3** (E2E insert).

---

### density-dismantling.AC3.3
> **density-dismantling.AC3.3 Success:** Evolution classification against prior membership snapshots (birth, death, continuation, merge, split) behaves as before, verified by existing tests continuing to pass.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (regression) | `url_cosharing/tests/test_analyzer.py` (`TestComputeEvolution`, `TestComputeJaccard`) | The existing evolution/Jaccard tests pass **unchanged** apart from mechanical constructor-argument additions (the two new `ClusterResult` fields). `compute_jaccard`/`compute_evolution` logic untouched — birth/death/continuation/merge/split behaviour preserved. |

**Human verification:** none (regression is the whole contract here — the criterion is satisfied by the
existing suite staying green).

---

### density-dismantling.AC3.4
> **density-dismantling.AC3.4 Success:** Daily membership snapshots are written for all cluster members.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| integration | `url_cosharing/tests/test_main.py` (full-cycle case) | Membership rows captured by `FakeDb` cover **every** member of every written cluster; death-event clusters are skipped in both cluster and membership writes. |

**Human verification:** membership rows landing in the live `url_cosharing_membership` table — item **H3**.

---

## density-dismantling.AC4: Cross-cutting

### density-dismantling.AC4.1
> **density-dismantling.AC4.1:** Every new parameter is an env var with a documented default (`URL_COSHARING_*`), parsed into the frozen `AnalysisConfig`.

**Automated tests:**

| Test type | File | Behaviour pinned |
|---|---|---|
| unit (defaults) | `url_cosharing/tests/test_config.py` (`TestAnalysisConfig.test_from_env_defaults`) | Each new field takes its documented default: `window_days==7`, `min_unique_urls==10`, `min_url_sharers==5`, `max_url_df_pctl==0.90`, `edge_epsilon==0.05`, both grids `==(0.50,0.60,0.70,0.80,0.90,0.95,0.99)`, `density_floor==0.5`, `max_flagged_fraction==0.02`, `runs_table=='url_cosharing_runs'`. |
| unit (overrides) | `url_cosharing/tests/test_config.py` (`TestAnalysisConfig.test_from_env_overrides`) | Each `URL_COSHARING_*` env var parses a non-default value (e.g. `EDGE_QUANTILE_GRID='0.6,0.8'` → `(0.6,0.8)`). |
| unit (validation) | `url_cosharing/tests/test_config.py` (`TestAnalysisConfig`) | Grid parsing: whitespace tolerated, non-numeric / empty / value ≤ 0 or ≥ 1 / non-increasing / duplicate all raise `ValueError`. Unit-interval validation: `DENSITY_FLOOR='1.5'`, `MAX_FLAGGED_FRACTION='-0.1'` raise `ValueError`. `RUNS_TABLE='bad;name'` fails `_validate_table_name`. Frozen dataclass: grids stored as `tuple[float, ...]` (hashable, immutable). |

**Human verification:** none.

---

### density-dismantling.AC4.2
> **density-dismantling.AC4.2:** `similarity.py` and `dismantling.py` are pure functional-core modules (no I/O, no ClickHouse imports).

**Automated tests / static gates:**

| Test type | File / command | Behaviour pinned |
|---|---|---|
| static gate | `grep -n 'clickhouse\|from url_cosharing.db' url_cosharing/src/url_cosharing/similarity.py url_cosharing/src/url_cosharing/dismantling.py` → empty | Import-purity check, run as a Phase 3/4/5 completion-checklist gate. Both modules carry `# pattern: Functional Core`. |
| indirect (behavioural) | `url_cosharing/tests/test_similarity.py`, `url_cosharing/tests/test_dismantling.py` | Every test builds inputs in-memory (hand-built `UrlShareRow` lists, explicit igraph edge lists) with no DB fixture — the modules being testable without any client mock is itself evidence of purity. |

**Human verification:** none (the grep gate is deterministic and CI-runnable; no live system needed).

---

### density-dismantling.AC4.3
> **density-dismantling.AC4.3:** `cd url_cosharing && uv run pytest` passes with all new and existing tests.

**Automated tests:**

| Test type | File / command | Behaviour pinned |
|---|---|---|
| suite gate | `cd url_cosharing && uv run pytest` | The full suite (config, queries, db, similarity, dismantling, analyzer, main, calibrate) passes — this is itself the acceptance test. Post-removal sweeps (`grep -rn 'PairRow\|build_graph\|fetch_pairs\|min_edge_weight\|min_cosharers\|pairs_table' url_cosharing/src url_cosharing/tests` → nothing) are Phase 5/6 completion gates. |
| unit (calibrate formatter) | `url_cosharing/tests/test_calibrate.py` (`TestFormatSurface`) | Pure `format_surface` produces the TSV header, one line per surface cell, and the summary footer with exact counts/flags; empty (no-knee) surface still produces header + footer without error. |

**Human verification:** running the suite *is* the test; no manual step for AC4.3 itself. (The calibrate
module's **live** run is a separate human item — **H2**.)

---

## Human Verification Required (summary)

CI has no ClickHouse, so three things automated tests structurally cannot close. These follow the repo's
existing `docs/test-plans/` convention (Phase 1 cross-repo schema check + E2E cycle in the
stats-methodology plan).

| ID | Item | Why manual | Verification approach |
|---|---|---|---|
| **H1** | DDL applies cleanly to a live ClickHouse (Phase 1 Task 2) | The `url_cosharing_runs` `CREATE TABLE`, the two `ALTER TABLE … ADD COLUMN` migrations, and the cluster-column additions live in `skywatch-osprey/clickhouse-init/05-url-cosharing.sql` and never execute in CI. A name misspelled/missing there passes every pytest but breaks at insert time (ClickHouse inserts **by name**). | Apply the merged DDL to a ClickHouse instance at deploy time. Static pre-checks (from Phase 1 Task 2): `grep -c 'ADD COLUMN IF NOT EXISTS' …05-url-cosharing.sql` → 2; `grep -c 'CREATE TABLE IF NOT EXISTS default.url_cosharing_runs' …` → 1; `grep -v '^--' …05-url-cosharing.sql \| grep -c 'DROP'` → 0. Then name-for-name: reconcile the live `url_cosharing_runs` and `url_cosharing_clusters` columns against `db.py`'s `insert_run` (13 names) and `insert_clusters` (16 names) `column_names` lists — e.g. `comm -23` of sorted `db.py` names vs sorted `system.columns` names → empty. Confirm `mean_edge_similarity`/`subgraph_density` follow `jaccard_score` in `url_cosharing_clusters`. |
| **H2** | Calibration module runs against a populated ClickHouse (Phase 6 Task 3) | `uv run python -m url_cosharing.calibrate` fetches a real window and runs the full pipeline; CI only unit-tests the pure `format_surface`. | On a ClickHouse with `osprey_execution_results` populated: `cd url_cosharing && uv run python -m url_cosharing.calibrate`. Expect a TSV surface (one line per grid cell: `edge_quantile`, `centrality_quantile`, `min_component_density`, survivors) plus the two summary footer lines, and a clean exit. Confirms the read path, the Core pipeline, and `CosharingDb.close()` all wire against production shapes. |
| **H3** | Production read-path + full write cycle observed live (AC1.1 production half; AC1.5/AC2.5/AC3.2/AC3.4 live inserts) | The FakeDb integration test stubs the real insert and the real `osprey_execution_results` read. Only a live cycle proves `column_names` matches the live schema name-for-name and that the daily run actually reads `osprey_execution_results` (not the pairs MV). | With H1's DDL applied, run one full cycle (`docker compose up url_cosharing` or the sidecar run loop) against a populated instance. Expect **zero insert errors**. Then query: `url_cosharing_runs` has today's row with populated stage counts (proves the real read) and `knee_found`/`guardrail_triggered`/quantiles/`min_component_density` consistent; on a flagging day `url_cosharing_clusters` rows carry non-null `mean_edge_similarity`/`subgraph_density` and `url_cosharing_membership` covers every cluster member; `flagged_pct = flagged_accounts / accounts_eligible` lands in the paper's 0.4–1.5% band (run the `docs/calibration.md` runs-table health query). Re-run the H1 `comm` name-for-name check against the **live** `system.columns` for final confirmation. |

---

## Traceability

| Acceptance Criterion | Automated test(s) | Manual step |
|---|---|---|
| density-dismantling.AC1.1 | `test_queries.py::TestFetchUrlSharesQuery`, `test_db.py::TestFetchUrlShares`, `test_main.py` (FakeDb cycle) | H3 (production read half) |
| density-dismantling.AC1.2 | `test_queries.py::TestFetchUrlSharesQuery`, `test_similarity.py::TestFilterShares` | — |
| density-dismantling.AC1.3 | `test_queries.py::TestFetchUrlSharesQuery`, `test_similarity.py::TestFilterShares` | — |
| density-dismantling.AC1.4 | `test_similarity.py::TestTfidfTransform`, `test_similarity.py::TestBuildSimilarityGraph` | — |
| density-dismantling.AC1.5 | `test_similarity.py::TestSimilarityNetwork`/`TestBuildShareMatrix`, `test_dismantling.py::TestDismantleNoTransition`, `test_main.py` (empty-input) | H1 + H3 (zero-count row live) |
| density-dismantling.AC2.1 | `test_dismantling.py::TestDismantleSurface` | — |
| density-dismantling.AC2.2 | `test_dismantling.py::TestDismantleKnee` | — |
| density-dismantling.AC2.3 | `test_dismantling.py::TestDismantleNoTransition`, `test_main.py` (no-knee) | — |
| density-dismantling.AC2.4 | `test_dismantling.py::TestDismantleGuardrails` | — |
| density-dismantling.AC2.5 | `test_db.py::TestInsertRun`, `test_main.py` (full-cycle) | H1 + H3 (live run row) |
| density-dismantling.AC3.1 | `test_analyzer.py::TestClusterCore` | — |
| density-dismantling.AC3.2 | `test_analyzer.py::TestClusterCore`/`TestComputeTemporalMetrics`, `test_db.py::TestInsertColumnList`, `test_queries.py` | H1 + H3 (column adjacency + live insert) |
| density-dismantling.AC3.3 | `test_analyzer.py::TestComputeEvolution`/`TestComputeJaccard` (regression) | — |
| density-dismantling.AC3.4 | `test_main.py` (full-cycle, membership coverage) | H3 (live membership rows) |
| density-dismantling.AC4.1 | `test_config.py::TestAnalysisConfig` (defaults/overrides/validation) | — |
| density-dismantling.AC4.2 | import-purity grep gate; `test_similarity.py` + `test_dismantling.py` (DB-free by construction) | — |
| density-dismantling.AC4.3 | `cd url_cosharing && uv run pytest` (full suite); `test_calibrate.py::TestFormatSurface`; removal sweeps | H2 (live calibrate run) |

Every acceptance criterion maps to at least one automated test or a documented human verification item; none is orphaned.

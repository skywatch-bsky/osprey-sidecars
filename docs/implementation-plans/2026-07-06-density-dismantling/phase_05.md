# Density-Dismantling Implementation Plan — Phase 5: Pipeline integration

**Goal:** Wire the new detection path end-to-end (fetch → similarity → dismantling → Leiden-on-core → temporal/evolution → writes incl. run metadata), then remove the pairs-based path and its config.

**Architecture:** `analyzer.py` gains `cluster_core` (Leiden CPM on `similarity` weights, cluster metrics from the bipartite matrix) while evolution tracking stays untouched. `db.py` gains `RunMetadata`/`insert_run` and extends `insert_clusters`. `main.py`'s `run_cycle` is rewritten around the new path and always writes a run row — including on empty/no-knee days. Removals (`PairRow`, `build_graph`, `fetch_pairs*`, `min_edge_weight`, `min_cosharers`, `pairs_table`) land last so every commit stays green.

**Tech Stack:** python-igraph, leidenalg (CPM), scipy sparse, numpy, clickhouse-connect, pytest.

**Scope:** Phase 5 of 6 from `docs/design-plans/2026-07-06-density-dismantling.md`.

**Codebase verified:** 2026-07-07 (codebase-investigator; verbatim analyzer.py/main.py/db.py and complete removal-site inventory).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### density-dismantling.AC1: Similarity network construction
- **density-dismantling.AC1.1 Success:** A daily run fetches per-account URL share counts over the configured rolling window (default 7 days) from `osprey_execution_results`, not from `url_cosharing_pairs`. *(End-to-end: the pairs path is deleted here.)*
- **density-dismantling.AC1.5 Edge:** An empty or fully-filtered input produces an empty graph; the run completes normally and writes a run-metadata row with zero counts.

### density-dismantling.AC2: Density-based dismantling
- **density-dismantling.AC2.5 Success:** The chosen quantile pair, resulting minimum density, and account/URL counts at each filter stage are written to the run-metadata table.

### density-dismantling.AC3: Hybrid clustering and outputs
- **density-dismantling.AC3.1 Success:** Leiden CPM (cosine-similarity edge weights) decomposes the surviving core; clusters below `min_cluster_size` are dropped.
- **density-dismantling.AC3.2 Success:** Cluster rows carry the new similarity metrics (`mean_edge_similarity`, `subgraph_density`) alongside existing columns; `total_weight` keeps its co-share-count semantics (Σ over cluster URLs of C(sharing members, 2)).
- **density-dismantling.AC3.3 Success:** Evolution classification against prior membership snapshots (birth, death, continuation, merge, split) behaves as before, verified by existing tests continuing to pass.
- **density-dismantling.AC3.4 Success:** Daily membership snapshots are written for all cluster members.

### density-dismantling.AC4: Cross-cutting
- **density-dismantling.AC4.2:** `similarity.py` and `dismantling.py` are pure functional-core modules (no I/O, no ClickHouse imports). *(Final check here.)*
- **density-dismantling.AC4.3:** `cd url_cosharing && uv run pytest` passes with all new and existing tests.

---

## Codebase Context (verified 2026-07-07)

- `analyzer.py` (Core): `PairRow`, `ClusterResult` (9 fields ending `resolution_parameter`), `TimestampedCluster(ClusterResult)` (+2 temporal fields), `EvolutionEvent`, `build_graph(pairs, min_edge_weight)`, `cluster_graph(graph, resolution, min_cluster_size)` (Leiden `weights='newman_weight'`), `compute_temporal_metrics`, `compute_jaccard`, `compute_evolution` (stable IDs `{run_date}-{counter:04d}`).
- `main.py` (Shell): `run_cycle` = fetch pairs → early-return if none → build_graph → cluster_graph → member timestamps (with `_sanitize_did`) → temporal metrics → historical membership → evolution → `delete_run_date` × 2 → insert clusters (skipping `death` events) → insert membership. Startup log line prints `resolution/min_edge_weight/min_cluster_size/jaccard_threshold`.
- `db.py` (Shell): `insert_clusters` uses `column_names` (14 names, ends `jaccard_score`); `insert_membership`; `delete_run_date`; fetches for pairs/membership/timestamps.
- `queries.py`: `insert_clusters_query`/`insert_membership_query` builders exist alongside the `column_names`-based inserts — both must stay column-consistent.
- Removal-site inventory (from investigation): `min_edge_weight` → config.py:40,55 / main.py:60,136 / analyzer.py:73,97,99; `min_cosharers` → config.py:42,57; `pairs_table` → config.py:45,60 / queries.py:16; `fetch_pairs_query` → queries.py:7 / main.py:21,51; `PairRow` → analyzer.py:15 / db.py:10,37,45 / test_analyzer.py (70+ uses) / test_main.py (30+ uses). *(Line numbers shift after Phase 1's config edits — re-grep at execution time; the inventory defines the complete term list.)*
- Test conventions: `test_analyzer.py` classes `TestBuildGraph`(9)/`TestClusterGraph`(6)/`TestComputeTemporalMetrics`(4)/`TestComputeJaccard`(6)/`TestComputeEvolution`(8); `test_main.py` uses a `FakeDb` stub with captured inserts.

**Green-at-every-commit sequencing:** `cluster_core` is a NEW function (old `cluster_graph`, `build_graph`, `PairRow` survive until Task 5). `ClusterResult` gains the two new fields **without defaults** in Task 1, so every construction site (old `cluster_graph` passes `0.0`s, test fixtures updated mechanically) is explicit — no silent zero can reach the insert path.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: `cluster_core` — Leiden on the surviving core with bipartite metrics

**Verifies:** density-dismantling.AC3.1, AC3.2 (implementation; tests in Task 2)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/analyzer.py`

**Implementation:**

1. Extend `ClusterResult` with two fields appended after `resolution_parameter` (no defaults):

```python
    mean_edge_similarity: float
    subgraph_density: float
```

2. Update the old `cluster_graph` to pass `mean_edge_similarity=0.0, subgraph_density=0.0` (it is deleted in Task 5; this keeps it constructible meanwhile). `compute_temporal_metrics` builds `TimestampedCluster` from an existing cluster's fields — ensure it copies the two new fields through (add them to its constructor call).

3. Add imports `import numpy as np` and `from url_cosharing.similarity import ShareMatrix` (Core→Core), `from scipy.sparse import csr_array`, and the new function:

```python
def cluster_core(
    core: ig.Graph,
    matrix: ShareMatrix,
    tfidf: csr_array,
    resolution: float,
    min_cluster_size: int,
) -> list[ClusterResult]:
    """Leiden CPM over cosine-similarity weights on the dismantled core.

    Cluster metrics come from two sources: edge metrics (mean_edge_similarity,
    subgraph_density, total_edges) from the core subgraph; URL metrics
    (unique_urls, total_weight, sample_urls) from the bipartite share matrix.
    total_weight keeps its co-share-count semantics: sum over cluster URLs of
    C(k, 2) where k is the number of cluster members sharing that URL.
    """
    if core.vcount() == 0:
        return []

    partition = leidenalg.find_partition(
        core,
        leidenalg.CPMVertexPartition,
        weights='similarity',
        resolution_parameter=resolution,
    )

    account_to_row = {did: idx for idx, did in enumerate(matrix.accounts)}

    clusters_by_id: dict[int, list[int]] = {}
    for vertex_idx, cluster_id in enumerate(partition.membership):
        clusters_by_id.setdefault(cluster_id, []).append(vertex_idx)

    results = []
    for vertex_indices in clusters_by_id.values():
        if len(vertex_indices) < min_cluster_size:
            continue

        members = frozenset(core.vs[idx]['name'] for idx in vertex_indices)
        subgraph = core.induced_subgraph(vertex_indices)
        total_edges = subgraph.ecount()
        mean_edge_similarity = (
            float(np.mean(subgraph.es['similarity'])) if total_edges > 0 else 0.0
        )
        subgraph_density = float(subgraph.density(loops=False))

        member_rows = [account_to_row[did] for did in sorted(members)]
        sub_counts = matrix.counts[member_rows, :]
        sharers = np.asarray((sub_counts > 0).sum(axis=0)).ravel()
        unique_urls = int((sharers >= 2).sum())
        total_weight = int((sharers * (sharers - 1) // 2).sum())

        mass = np.asarray(tfidf[member_rows, :].sum(axis=0)).ravel()
        order = np.argsort(-mass, kind='stable')
        sample_urls = [matrix.urls[k] for k in order[:10] if mass[k] > 0]

        results.append(
            ClusterResult(
                cluster_id='',
                members=members,
                member_count=len(members),
                total_edges=total_edges,
                total_weight=total_weight,
                unique_urls=unique_urls,
                sample_dids=sorted(members)[:10],
                sample_urls=sample_urls,
                resolution_parameter=resolution,
                mean_edge_similarity=mean_edge_similarity,
                subgraph_density=subgraph_density,
            )
        )
    return results
```

Notes:
- `sharers * (sharers - 1) // 2` is C(k, 2) and is 0 for k ∈ {0, 1} — no masking needed.
- `np.argsort(-mass, kind='stable')` makes TF-IDF ties break by column index, i.e. lexicographic URL order — deterministic `sample_urls`.
- Every account in `core` is guaranteed to be in `matrix.accounts` (the core is a subgraph of the Phase 3 graph, whose vertices are exactly the matrix rows), so `account_to_row[did]` cannot KeyError.
- `compute_jaccard`, `compute_evolution`, `compute_temporal_metrics` logic is otherwise untouched (AC3.3).

4. Mechanically update every direct `ClusterResult(`/`TimestampedCluster(` construction in `tests/test_analyzer.py` and `tests/test_main.py` to pass the two new arguments (use `0.0` where the test doesn't care) — grep for both constructors.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes (old paths still alive, new function exercised in Task 2).

**Commit:** combined with Task 2.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `cluster_core` tests

**Verifies:** density-dismantling.AC3.1, AC3.2, AC3.3

**Files:**
- Test: `url_cosharing/tests/test_analyzer.py` (unit)

**Testing:**

New class `TestClusterCore`. Build inputs by hand: a small `ShareMatrix` (via `similarity.build_share_matrix` on `UrlShareRow`s), its `tfidf_transform`, and a core graph (either from `similarity.build_similarity_graph` or an explicit igraph with `name` + `similarity` attributes aligned to `matrix.accounts`). Cases:

- AC3.1: two well-separated similarity cliques decompose into two clusters; a cluster with fewer than `min_cluster_size` members is dropped; empty core returns `[]`; Leiden runs on `similarity` weights (pin as the existing `TestClusterGraph` does for `newman_weight` — e.g. weights chosen so CPM at the test resolution only separates the cliques if similarity weights are honoured).
- AC3.2 `total_weight` (hand-computed): 3 members where `u1` is shared by all 3 and `u2` by 2 → `total_weight == C(3,2) + C(2,2? )` — concretely `3 + 1 = 4`; `unique_urls == 2` (URLs with ≥ 2 member sharers only); a URL shared by 1 member contributes 0 to both.
- AC3.2 similarity metrics (hand-computed): a 3-clique with similarities `[0.9, 0.8, 0.7]` → `mean_edge_similarity == pytest.approx(0.8)`, `subgraph_density == pytest.approx(1.0)`; a 3-path (2 edges) → density `pytest.approx(2/3)`.
- `sample_urls` ranked by cluster TF-IDF mass: construct so a URL with high within-cluster mass beats a more widely-shared one; zero-mass URLs never appear; ties break lexicographically.
- `TimestampedCluster` propagation: `compute_temporal_metrics` preserves `mean_edge_similarity`/`subgraph_density` (extend the existing field-preservation test).
- AC3.3: the existing `TestComputeEvolution` tests pass unchanged apart from the mechanical constructor-argument additions — evolution logic untouched.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes.

**Commit:**
```bash
git add url_cosharing/src/url_cosharing/analyzer.py url_cosharing/tests/test_analyzer.py url_cosharing/tests/test_main.py
git commit -m "feat(url_cosharing): Leiden-on-core clustering with bipartite similarity metrics"
```
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (task 3) -->

<!-- START_TASK_3 -->
### Task 3: Persistence — extended cluster insert and `url_cosharing_runs` writes

**Verifies:** density-dismantling.AC2.5 (write path; end-to-end in Task 4), AC3.2 (columns)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/db.py`
- Modify: `url_cosharing/src/url_cosharing/queries.py`
- Test: `url_cosharing/tests/test_db.py`, `url_cosharing/tests/test_queries.py` (unit)

**Implementation:**

1. `db.py` — extend `insert_clusters`: append `'mean_edge_similarity', 'subgraph_density'` to `column_names` and `cluster.mean_edge_similarity, cluster.subgraph_density` to the row list (same positions).

2. `db.py` — add `RunMetadata` (next to `MembershipRow`) and `insert_run`:

```python
@dataclass(frozen=True)
class RunMetadata:
    run_date: date
    window_days: int
    accounts_raw: int
    accounts_eligible: int
    urls_eligible: int
    graph_edges: int
    edge_quantile: float
    centrality_quantile: float
    min_component_density: float
    knee_found: bool
    guardrail_triggered: bool
    flagged_accounts: int
    cluster_count: int
```

```python
    def insert_run(self, table: str, run: RunMetadata) -> None:
        column_names = [
            'run_date',
            'window_days',
            'accounts_raw',
            'accounts_eligible',
            'urls_eligible',
            'graph_edges',
            'edge_quantile',
            'centrality_quantile',
            'min_component_density',
            'knee_found',
            'guardrail_triggered',
            'flagged_accounts',
            'cluster_count',
        ]
        data = [[
            run.run_date,
            run.window_days,
            run.accounts_raw,
            run.accounts_eligible,
            run.urls_eligible,
            run.graph_edges,
            run.edge_quantile,
            run.centrality_quantile,
            run.min_component_density,
            run.knee_found,
            run.guardrail_triggered,
            run.flagged_accounts,
            run.cluster_count,
        ]]
        self._client.insert(table=table, data=data, column_names=column_names)
```

3. `queries.py` — update `insert_clusters_query` to list the two new columns (and two more `?` placeholders), keeping it consistent with `column_names`.

**Testing:**

- `test_db.py`: `insert_clusters` passes the 16-name `column_names` (assert the exact list — this is the cross-repo name-for-name guard against `clickhouse-init/05-url-cosharing.sql`); `insert_run` passes the exact 13-name list matching the `url_cosharing_runs` DDL order and one data row with values mapped from a `RunMetadata` instance.
- `test_queries.py`: `insert_clusters_query` contains the new column names and 16 placeholders.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes.

**Commit:** `feat(url_cosharing): persist run metadata and similarity cluster columns`
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (task 4) -->

<!-- START_TASK_4 -->
### Task 4: `run_cycle` rewrite — new detection path, run row always written

**Verifies:** density-dismantling.AC1.5, AC2.5, AC3.4 (pipeline level)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/main.py`
- Test: `url_cosharing/tests/test_main.py` (integration through the full Functional Core with a `FakeDb`)

**Implementation:**

Rewrite `run_cycle` (keep `_sanitize_did`, signal handling, the polling loop, the delete-before-insert idempotency pattern, and the death-event skip):

```python
def run_cycle(db: CosharingDb, config: AppConfig) -> None:
    run_date = date.today()
    analysis = config.analysis

    logger.info('fetching url shares')
    rows = db.fetch_url_shares(fetch_url_shares_query(analysis))
    logger.info(f'fetched {len(rows)} share rows')

    network = similarity_network(
        rows,
        analysis.min_unique_urls,
        analysis.min_url_sharers,
        analysis.max_url_df_pctl,
        analysis.edge_epsilon,
        logger,
    )
    logger.info(
        f'similarity network: {network.accounts_eligible}/{network.accounts_raw} accounts, '
        f'{network.urls_eligible} urls, {network.graph_edges} edges'
    )

    result = dismantle(
        network.graph,
        analysis.edge_quantile_grid,
        analysis.centrality_quantile_grid,
        analysis.density_floor,
        analysis.max_flagged_fraction,
        analysis.min_cluster_size,
        logger,
    )

    clusters = cluster_core(
        result.core, network.matrix, network.tfidf, analysis.resolution, analysis.min_cluster_size
    )
    logger.info(f'found {len(clusters)} clusters (knee_found={result.knee_found})')

    # ... member timestamps + compute_temporal_metrics — UNCHANGED from current code ...
    # ... historical membership + compute_evolution — UNCHANGED from current code ...

    logger.info('clearing stale data for today (idempotent re-run guard)')
    db.delete_run_date(analysis.clusters_table, run_date)
    db.delete_run_date(analysis.membership_table, run_date)
    db.delete_run_date(analysis.runs_table, run_date)

    # ... insert non-death cluster rows and membership rows — UNCHANGED from current code
    #     (this block constructs `cluster_rows`; the insert_run call below depends on it) ...

    db.insert_run(
        analysis.runs_table,
        RunMetadata(
            run_date=run_date,
            window_days=analysis.window_days,
            accounts_raw=network.accounts_raw,
            accounts_eligible=network.accounts_eligible,
            urls_eligible=network.urls_eligible,
            graph_edges=network.graph_edges,
            edge_quantile=result.edge_quantile,
            centrality_quantile=result.centrality_quantile,
            min_component_density=result.min_component_density,
            knee_found=result.knee_found,
            guardrail_triggered=result.guardrail_triggered,
            flagged_accounts=result.core.vcount(),
            cluster_count=len(cluster_rows),
        ),
    )
    logger.info(f'wrote run metadata to {analysis.runs_table}')
```

Critical behaviour changes vs. the current code:
- **No early return on empty input** — the old `if not pairs: return` disappears; an empty/fully-filtered day flows through (empty graph → no knee → zero clusters) and still writes the run row with zero counts (AC1.5, AC2.3 pipeline half). The timestamps step must tolerate zero members (it already does via the `if all_dids:` branch).
- `cluster_count` counts the rows actually written (non-death events), not raw partition count. Ordering constraint: `db.insert_run(...)` uses `len(cluster_rows)`, so it must come after the (elided) cluster-row construction/insert block — keep it the last write of the cycle as shown.
- Imports: drop `build_graph`, `cluster_graph`, `fetch_pairs_query`; add `cluster_core`, `similarity_network`, `dismantle`, `fetch_url_shares_query`, `RunMetadata`.
- Startup log in `main()`: replace the `min_edge_weight` line with the new headline knobs, e.g. `window_days`, `min_unique_urls`, `density_floor`, `max_flagged_fraction`, `resolution`, `min_cluster_size`, `jaccard_threshold` (repo precedent: the account_entropy startup-log fix).

**Testing:**

Rework `test_main.py`'s `FakeDb`: replace `pairs`/`fetch_pairs` with `url_shares: list[UrlShareRow]`/`fetch_url_shares`, add `captured_runs: list[tuple[str, RunMetadata]]`/`insert_run`, keep captured clusters/membership/deletes. Update `base_config`/`app_config` fixtures for the Phase 1 config fields, choosing **test-friendly knobs**: `min_unique_urls=2`, `min_url_sharers=2`, `max_url_df_pctl=0.99`, `edge_epsilon=0.01`, grids `(0.5, 0.9)`, `density_floor=0.5`, `max_flagged_fraction=0.9`, `min_cluster_size=3`.

Integration fixture (deterministic, no RNG): a coordinated group of 4 accounts each sharing the same 4 URLs (`c1..c4`), plus ≥ 6 background accounts each sharing 2 distinct URLs of their own **and one distinct `c*` URL each** — the background overlap keeps `df(c*) < N` so TF-IDF doesn't zero the coordinated URLs, while background-only URLs fall to the df floor. Hand-tune counts until the pipeline flags exactly the coordinated 4 (the Phase 4 planted-core tests prove the dismantler; this fixture proves the wiring). Cases:

- Full cycle: coordinated accounts land in one written cluster; membership rows cover every cluster member (AC3.4); cluster rows carry non-zero `mean_edge_similarity`/`subgraph_density`; run row has `knee_found=True`, correct `flagged_accounts` and `cluster_count`, and stage counts matching the fixture (AC2.5).
- Empty input: `FakeDb.url_shares = []` → run completes, no cluster/membership inserts, run row written with all-zero counts and `knee_found=False` (AC1.5).
- No knee: same fixture but `density_floor=1.0` forced in config → zero clusters written, run row records `knee_found=False`, `edge_quantile == 0.0` (AC2.3 pipeline half).
- Idempotency: `delete_run_date` called for clusters, membership, AND runs tables before inserts.
- Death events still skipped in cluster/membership writes (reuse the existing test's approach).

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes.

**Commit:** `feat(url_cosharing): wire density-dismantling detection path into run_cycle`
<!-- END_TASK_4 -->

<!-- END_SUBCOMPONENT_C -->

<!-- START_SUBCOMPONENT_D (task 5) -->

<!-- START_TASK_5 -->
### Task 5: Remove the pairs-based path and obsolete config

**Verifies:** density-dismantling.AC1.1 (pairs table no longer read), AC4.2, AC4.3

**Files:**
- Modify: `url_cosharing/src/url_cosharing/analyzer.py`, `config.py`, `db.py`, `main.py` (verify only), `queries.py`
- Modify: `url_cosharing/tests/test_analyzer.py`, `test_config.py`, `test_db.py`, `test_main.py`, `test_queries.py`

**Implementation (complete removal list — re-grep each term to catch drift):**

1. `analyzer.py`: delete `PairRow`, `build_graph`, and the old `cluster_graph`.
2. `queries.py`: delete `fetch_pairs_query`.
3. `db.py`: delete `fetch_pairs` and the `PairRow` import.
4. `config.py`: delete fields and env parsing for `min_edge_weight`, `min_cosharers`, `pairs_table`.
5. Tests: delete `TestBuildGraph`, old `TestClusterGraph` (superseded by `TestClusterCore`), `fetch_pairs_query` tests, `fetch_pairs` db tests; remove the three fields from every `AnalysisConfig(` construction (test_config assertions included); remove now-unused `PairRow` imports.
6. Sweep: `grep -rn 'PairRow\|build_graph\|fetch_pairs\|min_edge_weight\|min_cosharers\|pairs_table' url_cosharing/src url_cosharing/tests` → must return nothing.
7. Final AC4.2 check: `grep -n 'clickhouse\|from url_cosharing.db' url_cosharing/src/url_cosharing/similarity.py url_cosharing/src/url_cosharing/dismantling.py` → empty.

Do NOT touch the `url_cosharing_pairs` DDL or MV in skywatch-osprey — the pairs MV stays live for the `cosharing_pairs` MCP investigation tooling (design contract).

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes (AC4.3). Both greps above return nothing.

**Commit:** `refactor(url_cosharing): remove pairs-based detection path and obsolete config`
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_D -->

---

## Phase completion checklist

- [ ] `cd url_cosharing && uv run pytest` passes (density-dismantling.AC4.3).
- [ ] Run row written on every cycle — populated, empty-input, and no-knee days (AC1.5, AC2.5).
- [ ] Cluster rows carry `mean_edge_similarity`/`subgraph_density`; `total_weight` = Σ C(k, 2) pinned by hand-computed test (AC3.2).
- [ ] Evolution tests pass with logic untouched (AC3.3); membership snapshots written for all members (AC3.4).
- [ ] No reference to the pairs path or removed config anywhere in `url_cosharing/` (AC1.1).
- [ ] `similarity.py`/`dismantling.py` import-purity greps clean (AC4.2).

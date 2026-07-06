# Statistical Methodology Fixes — Phase 6: quote_cosharing Implementation Plan

**Goal:** Mirror of Phase 5 for quote subjects: Newman-weighted co-quoting edges and hardened duplicate-safe batch graph construction.

**Architecture:** The pairs MV in `clickhouse-init/06-quote-cosharing.sql` gains `newman_weight` from the per-subject quoter counts already in its `qualifying_uris` CTE; the sidecar fetches it; `build_graph` pre-aggregates duplicates and batch-builds edges; Leiden clusters on `newman_weight`; `min_edge_weight` keeps filtering raw `weight`. Clusters/membership tables unchanged.

**Tech Stack:** Python 3.11+, python-igraph ≥ 1.0, leidenalg ≥ 0.11 (CPMVertexPartition), clickhouse-connect, pytest via `uv run pytest`.

**Scope:** Phase 6 of 7 from `docs/design-plans/2026-07-06-stats-methodology.md` (independent of phases 1–5).

**Codebase verified:** 2026-07-06 via codebase-investigator agents.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### stats-methodology.AC6: Co-sharing
- **stats-methodology.AC6.1 Success:** Pairs MV emits `newman_weight = Σ 1/(k_url − 1)`; verified by query-structure test — *quote scope: k is the per-quoted-URI quoter count*
- **stats-methodology.AC6.2 Success:** `build_graph` aggregates duplicate (a, b) pairs — weights summed, URLs unioned, no parallel edges, no None weights
- **stats-methodology.AC6.3 Success:** Leiden receives `newman_weight`; `min_edge_weight` still filters raw weight
- **stats-methodology.AC6.4 Success:** Batch edge construction yields a graph identical to the per-edge loop on the same input

### stats-methodology.AC7: Schemas
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar) — *quote_cosharing scope: `06-quote-cosharing.sql`*

### stats-methodology.AC9: Suites
- **stats-methodology.AC9.1 Success:** `uv run pytest` passes in all six sidecars — *quote_cosharing scope*

---

## Context from codebase verification

quote_cosharing is a structural mirror of url_cosharing; verified divergences are naming only (2026-07-06):

- `analyzer.py` (362 lines): `PairRow` (14–20) has **`shared_uris`** (not `shared_urls`); `ClusterResult` has **`unique_uris`** / **`sample_uris`**; `build_graph` (72–108) has the identical per-edge loop and parallel-edge risk; `cluster_graph` (111–178) uses `leidenalg.find_partition(..., leidenalg.CPMVertexPartition, weights='weight', resolution_parameter=resolution)` and unions `edge['shared_uris']`; `compute_temporal_metrics` / `compute_evolution` untouched this phase.
- `queries.py` (80 lines): `fetch_pairs_query` (7–18) selects `date, account_a, account_b, weight, shared_uris` with `WHERE date = yesterday() AND weight >= {min_edge_weight}`.
- `db.py` (145 lines): `fetch_pairs` (37–53) positional mapping; `insert_clusters` (97–137) column list uses `unique_uris` / `sample_uris` — unchanged this phase.
- `config.py`: env prefix `QUOTE_COSHARING_*`; `resolution` 0.05, `min_edge_weight` 2, `min_cluster_size` 3, `pairs_table` `'quote_cosharing_pairs'`.
- Schema `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/06-quote-cosharing.sql`: pairs table (7–15: `shared_uris Array(String)`, TTL 7 days, ORDER BY (date, account_a, account_b)); MV `quote_cosharing_pairs_mv` (56–99) with CTEs `uri_quoters` → `qualifying_uris` (**`groupArray(did) AS quoters ... HAVING length(quoters) >= 3`**) → `quoters_expanded` → self-join emitting `count() AS weight, groupArray(DISTINCT s1.quoted_uri) AS shared_uris`.
- Tests: 5 files, `uv run pytest`; `test_analyzer.py` is 763 lines. Ruff single quotes/120.

External-dependency findings: identical to Phase 5 (pre-aggregate in Python + single `add_edges` batch; `weights='newman_weight'` attribute-name form; CPM resolution scales with edge-weight rescaling — calibration lever in Phase 7).

---

<!-- START_TASK_1 -->
### Task 1: Pairs MV and table — `newman_weight`

**Verifies:** stats-methodology.AC6.1 (MV side), stats-methodology.AC7.1 (quote_cosharing scope)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/06-quote-cosharing.sql`

**Implementation:**

Mirror Phase 5 Task 1 against the quote objects:

1. Pairs table: add `newman_weight Float64` after `weight UInt32`; engine/ORDER BY/TTL unchanged.
2. MV: carry `length(q.quoters) AS k_uri` through `quoters_expanded` and emit the Newman sum:

```sql
quoters_expanded AS (
    SELECT
        q.date,
        q.quoted_uri,
        arrayJoin(q.quoters) AS did,
        length(q.quoters) AS k_uri
    FROM qualifying_uris q
)
SELECT
    s1.date,
    s1.did AS account_a,
    s2.did AS account_b,
    count() AS weight,
    sum(1.0 / (s1.k_uri - 1)) AS newman_weight,
    groupArray(DISTINCT s1.quoted_uri) AS shared_uris
FROM quoters_expanded s1
INNER JOIN quoters_expanded s2
  ON s1.date = s2.date
  AND s1.quoted_uri = s2.quoted_uri
WHERE s1.did < s2.did
GROUP BY s1.date, account_a, account_b;
```

   `HAVING length(quoters) >= 3` guarantees `k_uri − 1 ≥ 2` (no division by zero).
3. Migration comment block (comments only — no live DROPs in the init file):

```sql
-- MIGRATION (existing deployments only — run manually, once, before re-applying this file):
--   DROP VIEW IF EXISTS default.quote_cosharing_pairs_mv;
--   DROP TABLE IF EXISTS default.quote_cosharing_pairs;
-- The pairs table has a 7-day TTL, so at most 7 days of pair history is lost;
-- clusters and membership tables are not touched by this migration.
```

**Verification:**
Run: `grep -n 'newman_weight\|k_uri' /Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/06-quote-cosharing.sql`
Expected: `newman_weight` in the pairs CREATE TABLE and MV select; `k_uri` defined and used. `grep -v '^--' 06-quote-cosharing.sql | grep -c DROP` → `0`.

**Commit** (in skywatch-osprey): `Add Newman collaboration weight to quote_cosharing pairs MV`
<!-- END_TASK_1 -->

<!-- START_SUBCOMPONENT_A (tasks 2-4) -->
<!-- START_TASK_2 -->
### Task 2: `newman_weight` plumbing — `PairRow`, fetch query, fetch mapping

**Verifies:** stats-methodology.AC6.1 (query-structure test side)

**Files:**
- Modify: `quote_cosharing/src/quote_cosharing/analyzer.py:14-20` (`PairRow`)
- Modify: `quote_cosharing/src/quote_cosharing/queries.py:7-18` (`fetch_pairs_query`)
- Modify: `quote_cosharing/src/quote_cosharing/db.py:37-53` (`fetch_pairs`)
- Test: `quote_cosharing/tests/test_analyzer.py`, `quote_cosharing/tests/test_queries.py`, `quote_cosharing/tests/test_db.py` (unit)

**Implementation:**

Mirror Phase 5 Task 2 with `shared_uris` naming:
1. `PairRow` gains `newman_weight: float` after `weight` (order: date, account_a, account_b, weight, newman_weight, shared_uris).
2. `fetch_pairs_query` selects `newman_weight` between `weight` and `shared_uris`; the `weight >= {min_edge_weight}` filter stays on raw weight.
3. `fetch_pairs` maps `row[4]` → `newman_weight`, `row[5]` → `shared_uris`.

**Testing:**

- `test_queries.py`: `'newman_weight'` in `fetch_pairs_query`; filter references raw `weight` (`'AND weight >='` present, `'newman_weight >=' not in query`).
- `test_analyzer.py` / `test_db.py`: `PairRow` construction, frozen-ness, fetch mapping order. Update all existing `PairRow(...)` constructions across the suite with a `newman_weight` value.

**Verification:**
Run: `cd quote_cosharing && uv run pytest tests/test_queries.py tests/test_db.py`
Expected: pass.

**Commit:** combined with Task 3.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `build_graph` — duplicate aggregation + batch construction

**Verifies:** stats-methodology.AC6.2, stats-methodology.AC6.3 (filter half), stats-methodology.AC6.4

**Files:**
- Modify: `quote_cosharing/src/quote_cosharing/analyzer.py:72-108` (`build_graph`)
- Test: `quote_cosharing/tests/test_analyzer.py` (unit)

**Implementation:**

Apply the Phase 5 Task 3 rewrite verbatim, with `shared_uris` in place of `shared_urls` (dict values aggregate `(weight, newman_weight, set_of_uris)`; edge attributes `'weight'`, `'newman_weight'`, `'shared_uris'`; key normalization `a < b`; single `add_edges` batch; deterministic sorted keys and sorted URI lists). Full target code:

```python
def build_graph(pairs: list[PairRow], min_edge_weight: int) -> ig.Graph:
    """
    Build an undirected weighted graph from pairs, filtering by minimum raw edge weight.

    Duplicate (account_a, account_b) pairs are aggregated before edge creation:
    raw weights and Newman weights are summed, shared URI lists are unioned.
    Edges are added in a single batch with attribute lists, so the graph can
    never contain parallel edges or None-valued attributes.

    Returns an empty graph (0 vertices, 0 edges) if no qualifying pairs.
    """
    filtered_pairs = [p for p in pairs if p.weight >= min_edge_weight]

    if not filtered_pairs:
        return ig.Graph()

    aggregated: dict[tuple[str, str], tuple[int, float, set[str]]] = {}
    for pair in filtered_pairs:
        key = (
            (pair.account_a, pair.account_b)
            if pair.account_a < pair.account_b
            else (pair.account_b, pair.account_a)
        )
        if key in aggregated:
            weight, newman_weight, uris = aggregated[key]
            aggregated[key] = (
                weight + pair.weight,
                newman_weight + pair.newman_weight,
                uris | set(pair.shared_uris),
            )
        else:
            aggregated[key] = (pair.weight, pair.newman_weight, set(pair.shared_uris))

    sorted_dids = sorted({did for key in aggregated for did in key})
    did_to_idx = {did: idx for idx, did in enumerate(sorted_dids)}

    graph = ig.Graph(len(sorted_dids))
    graph.vs['name'] = sorted_dids

    sorted_keys = sorted(aggregated)
    graph.add_edges([(did_to_idx[a], did_to_idx[b]) for a, b in sorted_keys])
    graph.es['weight'] = [aggregated[key][0] for key in sorted_keys]
    graph.es['newman_weight'] = [aggregated[key][1] for key in sorted_keys]
    graph.es['shared_uris'] = [sorted(aggregated[key][2]) for key in sorted_keys]

    return graph
```

**Testing:**

Same matrix as Phase 5 Task 3, with URIs: duplicate + reversed-duplicate aggregation into one edge (summed weights/newman, unioned URIs, `count_multiple() == [1] * ecount()`, no `None` attributes); raw-weight filter ignores `newman_weight`; batch-vs-per-edge-loop equivalence against a test-local reference implementation on duplicate-free input; existing `TestBuildGraph` cases updated and green.

**Verification:**
Run: `cd quote_cosharing && uv run pytest tests/test_analyzer.py -k BuildGraph`
Expected: all pass.

**Commit:** `feat: aggregate duplicate pairs and batch-build edges with Newman weights in quote_cosharing`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Leiden on `newman_weight`

**Verifies:** stats-methodology.AC6.3 (Leiden half)

**Files:**
- Modify: `quote_cosharing/src/quote_cosharing/analyzer.py:111-178` (`cluster_graph`)
- Test: `quote_cosharing/tests/test_analyzer.py` (unit)

**Implementation:**

Change `weights='weight'` → `weights='newman_weight'` in the `leidenalg.find_partition` call. `total_weight` keeps summing raw `weight`; `unique_uris`/samples unchanged. Docstring updated as in Phase 5 Task 4 (CPM on Newman weights; `QUOTE_COSHARING_RESOLUTION` re-tuning caveat → Phase 7 calibration doc).

**Testing:**

Same discriminating test as Phase 5 Task 4: A–B and C–D with `newman_weight=5.0`, bridge B–C with `weight=10, newman_weight=0.001`, `resolution=0.05` → exactly two clusters {A, B} / {C, D} (raw weights would have merged them); `total_weight` still raw. Existing cluster/evolution tests updated (set `newman_weight` equal to `weight` where old community structure must be preserved) and green.

**Verification:**
Run: `cd quote_cosharing && uv run pytest tests/test_analyzer.py`
Expected: all pass.

**Commit:** `feat: cluster quote_cosharing graph on Newman weights`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_5 -->
### Task 5: Documentation updates

**Verifies:** None (documentation; AC8.2 finalized in Phase 7)

**Files:**
- Modify: `quote_cosharing/README.md`
- Modify: `quote_cosharing/CLAUDE.md`

**Implementation:**

Mirror Phase 5 Task 5 for quote subjects: dual weights on pairs (raw co-quote count for filtering/investigations, `newman_weight = Σ 1/(k_uri − 1)` for clustering), duplicate-safe batch `build_graph`, `total_weight` semantics note, resolution re-tuning caveat pointing at `docs/calibration.md` (Phase 7).

**Verification:**
Run: `grep -n 'newman' quote_cosharing/README.md quote_cosharing/CLAUDE.md`
Expected: both files describe the Newman weighting.

**Commit:** `docs: update quote_cosharing README and CLAUDE.md for Newman weighting`
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Full suite gate

**Verifies:** stats-methodology.AC9.1 (quote_cosharing scope)

**Files:** none (verification only)

**Step 1:** Run: `cd quote_cosharing && uv run pytest` — Expected: all pass.

**Step 2:** Run: `cd quote_cosharing && uv run ruff check src tests && uv run ruff format --check src tests` — Expected: clean.

**Step 3:** Run: `git status --short` in both repos — Expected: clean trees; Task 1's commit exists in skywatch-osprey.
<!-- END_TASK_6 -->

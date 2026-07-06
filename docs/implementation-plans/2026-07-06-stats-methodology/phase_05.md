# Statistical Methodology Fixes — Phase 5: url_cosharing Implementation Plan

**Goal:** Newman-weighted co-sharing edges so one viral URL can't manufacture a cluster, plus hardened duplicate-safe batch graph construction.

**Architecture:** The pairs materialized view (skywatch-osprey `clickhouse-init/05-url-cosharing.sql`) gains a `newman_weight` column computed from per-URL sharer counts already present in its `qualifying_urls` CTE. The sidecar fetches it, `build_graph` pre-aggregates duplicate pairs in Python and builds edges in one batch, and Leiden clusters on `newman_weight` while `min_edge_weight` continues to filter raw `weight`. Clusters/membership tables are unchanged.

**Tech Stack:** Python 3.11+, python-igraph ≥ 1.0, leidenalg ≥ 0.11 (CPMVertexPartition), clickhouse-connect, ClickHouse refreshable materialized view, pytest via `uv run pytest`.

**Scope:** Phase 5 of 7 from `docs/design-plans/2026-07-06-stats-methodology.md` (independent of phases 1–4, 6).

**Codebase verified:** 2026-07-06 via codebase-investigator agents.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### stats-methodology.AC6: Co-sharing
- **stats-methodology.AC6.1 Success:** Pairs MV emits `newman_weight = Σ 1/(k_url − 1)`; verified by query-structure test
- **stats-methodology.AC6.2 Success:** `build_graph` aggregates duplicate (a, b) pairs — weights summed, URLs unioned, no parallel edges, no None weights
- **stats-methodology.AC6.3 Success:** Leiden receives `newman_weight`; `min_edge_weight` still filters raw weight
- **stats-methodology.AC6.4 Success:** Batch edge construction yields a graph identical to the per-edge loop on the same input

### stats-methodology.AC7: Schemas
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar) — *url_cosharing scope: `05-url-cosharing.sql`*

### stats-methodology.AC9: Suites
- **stats-methodology.AC9.1 Success:** `uv run pytest` passes in all six sidecars — *url_cosharing scope*

---

## Context from codebase verification

Current state (verified 2026-07-06):

- `url_cosharing/src/url_cosharing/analyzer.py` (363 lines): `PairRow` (14–20: date, account_a, account_b, weight, shared_urls); `build_graph` (72–108) — **per-edge loop** calling `graph.add_edges([(a, b)])` then `graph.get_eid(...)` per pair; on duplicate (a, b) inputs this creates parallel edges where only the last gets attributes and earlier ones keep `None` weights (the hardening target); `cluster_graph` (111–178) — `leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights='weight', resolution_parameter=resolution)`, per-cluster `total_weight = sum(subgraph.es['weight'])`, `unique_urls` from unioned `shared_urls` edge attributes; `compute_temporal_metrics` (181–233) and `compute_evolution` (236–362) — untouched by this phase.
- `queries.py` (81 lines): `fetch_pairs_query` (7–18) — `SELECT date, account_a, account_b, weight, shared_urls FROM {pairs_table} WHERE date = yesterday() AND weight >= {min_edge_weight}`. Other query builders untouched.
- `db.py` (146 lines): `fetch_pairs` (37–53) maps positional columns into `PairRow`. `insert_clusters` (97–137, 14 columns) and `insert_membership` — unchanged this phase.
- `config.py`: `resolution` (env `URL_COSHARING_RESOLUTION`, default 0.05), `min_edge_weight` (env `URL_COSHARING_MIN_EDGE_WEIGHT`, default 2), `min_cluster_size` (3), `jaccard_threshold` (0.5), `pairs_table` `'url_cosharing_pairs'`. No config changes this phase.
- Schema `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/05-url-cosharing.sql`: pairs table (lines 7–15: date, account_a, account_b, weight UInt32, shared_urls Array(String); MergeTree ORDER BY (date, account_a, account_b), **TTL date + INTERVAL 7 DAY DELETE**); clusters (19–35) and membership (39–45) tables; MV `url_cosharing_pairs_mv` (56–99): `REFRESH EVERY 1 DAY TO default.url_cosharing_pairs`, CTEs `url_sharers` → `qualifying_urls` (**`groupArray(did) AS sharers ... HAVING length(sharers) >= 3`** — the per-URL sharer count `k_url = length(sharers)` is right there) → `sharers_expanded` → self-join emitting `count() AS weight, groupArray(DISTINCT s1.url) AS shared_urls`.
- Tests: `uv run pytest`; `test_analyzer.py` (603 lines) covers build_graph/cluster_graph/jaccard/evolution; FakeDb in `test_main.py`. Ruff single quotes/120.

External-dependency findings (internet-researcher, 2026-07-06):
- Idiomatic parallel-edge-safe construction: pre-aggregate in Python, then one `graph.add_edges(edge_list)` call followed by list-assignment of edge attributes (`graph.es['weight'] = [...]`). igraph's `simplify(combine_edges=...)` can sum scalar weights but cannot union list attributes — pre-aggregation is the clean route.
- `leidenalg.find_partition(..., weights=<edge attribute name or list>, resolution_parameter=...)` — passing `weights='newman_weight'` selects that edge attribute.
- CPM quality compares community edge-weight density against `resolution_parameter`; rescaling edge weights rescales the effective resolution proportionally. Newman weights are systematically smaller than raw co-share counts (each shared URL contributes `1/(k_url − 1) ≤ 0.5` instead of 1), so the default resolution may need re-tuning after deployment — a documented calibration lever (Phase 7), not a code change here.

---

<!-- START_TASK_1 -->
### Task 1: Pairs MV and table — `newman_weight`

**Verifies:** stats-methodology.AC6.1 (MV side), stats-methodology.AC7.1 (url_cosharing scope)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/05-url-cosharing.sql`

**Implementation:**

1. **Pairs table** (lines 7–15): add `newman_weight Float64` after `weight UInt32`. Table engine, ORDER BY, and TTL unchanged.

2. **MV** (lines 56–99): carry the per-URL sharer count through `sharers_expanded` and sum its Newman contribution in the final select. The `qualifying_urls` CTE already has the sharer array; only two edits:

```sql
sharers_expanded AS (
    SELECT
        q.date,
        q.url,
        arrayJoin(q.sharers) AS did,
        length(q.sharers) AS k_url
    FROM qualifying_urls q
)
SELECT
    s1.date,
    s1.did AS account_a,
    s2.did AS account_b,
    count() AS weight,
    sum(1.0 / (s1.k_url - 1)) AS newman_weight,
    groupArray(DISTINCT s1.url) AS shared_urls
FROM sharers_expanded s1
INNER JOIN sharers_expanded s2
  ON s1.date = s2.date
  AND s1.url = s2.url
WHERE s1.did < s2.did
GROUP BY s1.date, account_a, account_b;
```

   `k_url ≥ 3` is guaranteed by the existing `HAVING length(sharers) >= 3`, so `k_url − 1 ≥ 2` and the division is always defined. Each URL a pair co-shares contributes `1/(k_url − 1)` (Newman 2001 collaboration weighting): a niche URL shared by 3 accounts contributes 0.5 to each of its pairs; a viral URL shared by 500 contributes ~0.002 — a single viral URL can no longer manufacture a heavy edge.

3. **Migration notes** — add a comment block above the MV (do NOT add live `DROP` statements to this init file; it is re-run with `IF NOT EXISTS` semantics and unconditional drops would wipe state on every init):

```sql
-- MIGRATION (existing deployments only — run manually, once, before re-applying this file):
--   DROP VIEW IF EXISTS default.url_cosharing_pairs_mv;
--   DROP TABLE IF EXISTS default.url_cosharing_pairs;
-- The pairs table has a 7-day TTL, so at most 7 days of pair history is lost;
-- clusters and membership tables are not touched by this migration.
```

**Verification:**
Run: `grep -n 'newman_weight\|k_url' /Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/05-url-cosharing.sql`
Expected: `newman_weight` in the pairs CREATE TABLE and in the MV select; `k_url` defined in `sharers_expanded` and used in the sum. Confirm no live `DROP` statements outside comments: `grep -v '^--' 05-url-cosharing.sql | grep -c DROP` → `0`.

**Commit** (in skywatch-osprey): `Add Newman collaboration weight to url_cosharing pairs MV`
<!-- END_TASK_1 -->

<!-- START_SUBCOMPONENT_A (tasks 2-4) -->
<!-- START_TASK_2 -->
### Task 2: `newman_weight` plumbing — `PairRow`, fetch query, fetch mapping

**Verifies:** stats-methodology.AC6.1 (query-structure test side)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/analyzer.py:14-20` (`PairRow`)
- Modify: `url_cosharing/src/url_cosharing/queries.py:7-18` (`fetch_pairs_query`)
- Modify: `url_cosharing/src/url_cosharing/db.py:37-53` (`fetch_pairs`)
- Test: `url_cosharing/tests/test_analyzer.py`, `url_cosharing/tests/test_queries.py`, `url_cosharing/tests/test_db.py` (unit)

**Implementation:**

1. `PairRow` gains `newman_weight: float` after `weight` (frozen dataclass, field order: date, account_a, account_b, weight, newman_weight, shared_urls).
2. `fetch_pairs_query` selects `newman_weight` after `weight`:

```python
    return f"""
        SELECT
            date,
            account_a,
            account_b,
            weight,
            newman_weight,
            shared_urls
        FROM {config.pairs_table}
        WHERE date = yesterday()
            AND weight >= {config.min_edge_weight}
    """
```

   The `weight >= {min_edge_weight}` filter stays on **raw** weight (AC6.3's second clause).
3. `fetch_pairs` maps the new positional column: `row[4]` is `newman_weight` (float), `row[5]` becomes `shared_urls`.

**Testing:**

- `test_queries.py`: `fetch_pairs_query` contains `newman_weight`; still filters `weight >= 2` (default config) — assert the filter references `weight`, not `newman_weight` (e.g. `'AND weight >='` in query and `'newman_weight >=' not in query`).
- `test_analyzer.py`: `PairRow` constructs with `newman_weight` and stays frozen.
- `test_db.py`: fetch mapping test with a mock row in the new 6-column order; assert `newman_weight` lands in the right field and `shared_urls` still parses.
- Update every existing `PairRow(...)` construction in the test suite (add a `newman_weight` value; where tests don't care, a value like `float(weight) / 2` keeps intent clear).

**Verification:**
Run: `cd url_cosharing && uv run pytest tests/test_queries.py tests/test_db.py`
Expected: pass (analyzer tests updated fully in Tasks 3–4).

**Commit:** combined with Task 3.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `build_graph` — duplicate aggregation + batch construction

**Verifies:** stats-methodology.AC6.2, stats-methodology.AC6.3 (filter half), stats-methodology.AC6.4

**Files:**
- Modify: `url_cosharing/src/url_cosharing/analyzer.py:72-108` (`build_graph`)
- Test: `url_cosharing/tests/test_analyzer.py` (unit)

**Implementation:**

Replace `build_graph` with (keep the docstring style; Functional Core):

```python
def build_graph(pairs: list[PairRow], min_edge_weight: int) -> ig.Graph:
    """
    Build an undirected weighted graph from pairs, filtering by minimum raw edge weight.

    Duplicate (account_a, account_b) pairs are aggregated before edge creation:
    raw weights and Newman weights are summed, shared URL lists are unioned.
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
            weight, newman_weight, urls = aggregated[key]
            aggregated[key] = (
                weight + pair.weight,
                newman_weight + pair.newman_weight,
                urls | set(pair.shared_urls),
            )
        else:
            aggregated[key] = (pair.weight, pair.newman_weight, set(pair.shared_urls))

    sorted_dids = sorted({did for key in aggregated for did in key})
    did_to_idx = {did: idx for idx, did in enumerate(sorted_dids)}

    graph = ig.Graph(len(sorted_dids))
    graph.vs['name'] = sorted_dids

    sorted_keys = sorted(aggregated)
    graph.add_edges([(did_to_idx[a], did_to_idx[b]) for a, b in sorted_keys])
    graph.es['weight'] = [aggregated[key][0] for key in sorted_keys]
    graph.es['newman_weight'] = [aggregated[key][1] for key in sorted_keys]
    graph.es['shared_urls'] = [sorted(aggregated[key][2]) for key in sorted_keys]

    return graph
```

Notes:
- The MV's `GROUP BY` plus `s1.did < s2.did` should already prevent duplicates upstream — this is the defence-in-depth layer the design calls for, because the old code silently produced parallel edges with `None` weights if that upstream invariant ever broke (re-run overlap, manual backfill, table merge).
- The key normalization (`a < b` ordering) also collapses a (b, a) duplicate of (a, b).
- `sorted_keys` / sorted URL lists make construction deterministic, which the AC6.4 equivalence test relies on.

**Testing:**

In `tests/test_analyzer.py` `TestBuildGraph`:
- AC6.2 (duplicate aggregation): input with `PairRow(a, b, weight=2, newman_weight=0.5, urls=['u1'])` twice plus a `(b, a)` reversed duplicate `weight=3, newman_weight=0.7, urls=['u1', 'u2']` → one single edge with `weight == 7`, `newman_weight == pytest.approx(1.7)`, `shared_urls == ['u1', 'u2']`; `graph.ecount() == 1`; no edge attribute is `None` (`assert all(w is not None for w in graph.es['weight'])` and same for `newman_weight`).
- AC6.2 (no parallel edges on any input): after building from a duplicate-heavy list, `graph.count_multiple() == [1] * graph.ecount()`.
- AC6.3 (raw-weight filter): a pair with `weight=1, newman_weight=99.0` is dropped at `min_edge_weight=2` — Newman weight does not rescue a thin raw edge.
- AC6.4 (batch ≡ per-edge loop): write a test-local reference builder that replicates the old per-edge loop over **already-unique** pairs (add edge, `get_eid`, set attributes one at a time, now including `newman_weight`), run both on the same duplicate-free input, and assert identical vertex name lists, identical edge sets (as frozensets of name pairs), and identical `weight`/`newman_weight`/`shared_urls` per edge (compare via dict keyed on the name pair).
- Existing `TestBuildGraph` cases (empty input, weight filtering, vertex naming) updated for the `newman_weight` field and kept green.

**Verification:**
Run: `cd url_cosharing && uv run pytest tests/test_analyzer.py -k BuildGraph`
Expected: all pass.

**Commit:** `feat: aggregate duplicate pairs and batch-build edges with Newman weights in url_cosharing`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Leiden on `newman_weight`

**Verifies:** stats-methodology.AC6.3 (Leiden half)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/analyzer.py:111-178` (`cluster_graph`)
- Test: `url_cosharing/tests/test_analyzer.py` (unit)

**Implementation:**

One-line change in the Leiden invocation (line ~126–131): `weights='newman_weight'` instead of `weights='weight'`:

```python
    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        weights='newman_weight',
        resolution_parameter=resolution,
    )
```

Everything else in `cluster_graph` is unchanged: `total_weight` remains the sum of **raw** `weight` over the cluster subgraph (its semantics — "total co-share count inside the cluster" — get documented in Task 5 and Phase 7), `unique_urls`/samples unchanged, `min_cluster_size` unchanged.

Update the `cluster_graph` docstring: clustering optimizes CPM over Newman-weighted edges; note that CPM's resolution compares against edge-weight density, so the `URL_COSHARING_RESOLUTION` default (0.05) may warrant re-tuning after the weight change (calibration playbook, Phase 7).

**Testing:**

In `TestClusterGraph`:
- Discriminating test (proves Leiden consumes `newman_weight`, not `weight`): build a 4-vertex graph via `build_graph` from pairs A–B (`weight=10, newman_weight=5.0`), C–D (`weight=10, newman_weight=5.0`), B–C (`weight=10, newman_weight=0.001`). With `resolution=0.05` and CPM on Newman weights, the B–C bridge (density 0.001 < 0.05) cannot justify merging, so `cluster_graph(graph, 0.05, 2)` yields two clusters {A, B} and {C, D}; on raw weights (all 10 > 0.05) it would yield one 4-node cluster — assert exactly 2 clusters with the expected memberships.
- `total_weight` still sums raw weights: for cluster {A, B} above, `total_weight == 10`.
- Existing cluster tests updated: every graph they build now flows through the new `build_graph` (pairs need `newman_weight`); where the old tests relied on clustering behaviour, set `newman_weight` equal to `weight` so their community structure — and thus their assertions — are preserved.

**Verification:**
Run: `cd url_cosharing && uv run pytest tests/test_analyzer.py`
Expected: all pass (including evolution/temporal tests untouched by this phase).

**Commit:** `feat: cluster url_cosharing graph on Newman weights`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_5 -->
### Task 5: Documentation updates

**Verifies:** None (documentation; AC8.2 finalized in Phase 7)

**Files:**
- Modify: `url_cosharing/README.md`
- Modify: `url_cosharing/CLAUDE.md`

**Implementation:**

Update both: pairs carry two weights — raw `weight` (co-share count; used for `min_edge_weight` filtering and investigations) and `newman_weight` (Σ 1/(k_url − 1); used by Leiden so viral URLs are down-weighted); `build_graph` aggregates duplicate pairs and batch-builds edges; cluster `total_weight` remains the raw co-share sum (semantics note); resolution re-tuning caveat with pointer to `docs/calibration.md` (created in Phase 7).

**Verification:**
Run: `grep -n 'newman' url_cosharing/README.md url_cosharing/CLAUDE.md`
Expected: both files describe the Newman weighting.

**Commit:** `docs: update url_cosharing README and CLAUDE.md for Newman weighting`
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Full suite gate

**Verifies:** stats-methodology.AC9.1 (url_cosharing scope)

**Files:** none (verification only)

**Step 1:** Run: `cd url_cosharing && uv run pytest` — Expected: all pass.

**Step 2:** Run: `cd url_cosharing && uv run ruff check src tests && uv run ruff format --check src tests` — Expected: clean.

**Step 3:** Run: `git status --short` in both repos — Expected: clean trees; Task 1's commit exists in skywatch-osprey.
<!-- END_TASK_6 -->

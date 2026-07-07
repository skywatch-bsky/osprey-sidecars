# Density-Dismantling Implementation Plan — Phase 4: Density-based dismantling core

**Goal:** Pure Functional Core that grid-searches edge-similarity × eigenvector-centrality quantile thresholds, computes the minimum-component-density surface, and selects thresholds by knee detection under guardrails — isolating the high-precision coordinated core.

**Architecture:** One new module `dismantling.py` (`# pattern: Functional Core` — no I/O, no ClickHouse imports, logger permitted). Input: the similarity igraph graph from Phase 3. Output: a structured `DismantlingResult` carrying the surviving core subgraph, chosen thresholds, the full grid surface (for the runs table and offline calibration), and guardrail flags.

**Tech Stack:** python-igraph >= 1.0 (eigenvector centrality, connected components, density, select/delete idioms), numpy (`np.quantile`), pytest.

**Scope:** Phase 4 of 6 from `docs/design-plans/2026-07-06-density-dismantling.md`.

**Codebase verified:** 2026-07-07 (codebase-investigator + external API research).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### density-dismantling.AC2: Density-based dismantling
- **density-dismantling.AC2.1 Success:** Grid search over the configured edge-similarity × eigenvector-centrality quantile grids produces a minimum-component-density surface (isolates dropped before density is computed).
- **density-dismantling.AC2.2 Success:** Knee detection selects the threshold pair at the largest discrete jump in minimum component density whose resulting density meets `density_floor`; a synthetic graph with a planted dense core plus organic background recovers the planted core.
- **density-dismantling.AC2.3 Failure:** When no cell satisfies the density floor (no transition), zero accounts are flagged and the run row records that no threshold was selected.
- **density-dismantling.AC2.4 Failure:** When the selected thresholds would flag more than `max_flagged_fraction` of eligible accounts, the candidate is rejected and the next-best candidate (or none) is used; the guardrail activation is recorded.

### density-dismantling.AC4: Cross-cutting
- **density-dismantling.AC4.2:** `similarity.py` and `dismantling.py` are pure functional-core modules (no I/O, no ClickHouse imports). *(`dismantling.py` half.)*

*(AC2.5 — writing the chosen values to the run-metadata table — is Phase 5; this phase's `DismantlingResult` carries every value that row needs.)*

---

## Algorithm decisions fixed by this phase

The design specifies the shape ("largest forward-difference jump subject to a density floor, with guardrails") but leaves the 2-D details open. This phase fixes them:

- **Centrality is computed once, on the full similarity graph**, via `graph.eigenvector_centrality(weights='similarity', scale=True)`. On disconnected graphs igraph computes per-component values normalized to a global max of 1; strictly, scores are only comparable within a component. The joint quantile filter follows the source paper's methodology regardless — this caveat is documented in the module docstring and revisited during Phase 6 calibration.
- **Thresholds are quantiles of the observed distributions:** `edge_threshold = np.quantile(edge_similarities, eq)` and `centrality_threshold = np.quantile(centralities, cq)` (linear interpolation, numpy default).
- **Per-cell filtering order:** keep vertices with centrality ≥ threshold (induced subgraph) → delete edges with similarity < threshold → delete isolates (degree 0). Only then compute per-component density; every surviving component has ≥ 2 vertices, so `density()` is well-defined. A cell with nothing surviving records density 0.0.
- **Knee = largest forward-difference jump *arriving at* a cell** along either grid axis: `jump(i, j) = max(d[i][j] − d[i−1][j], d[i][j] − d[i][j−1])` (terms with `i−1 < 0` / `j−1 < 0` omitted; cell (0, 0) has no predecessor and is never a knee candidate). This is the 2-D generalization of "largest discrete jump in minimum density".
- **Candidate selection:** all cells with a defined jump are ranked by `(jump, density, eq, cq)` descending — largest jump first; ties broken toward higher density, then toward stricter (higher) quantiles, which flags fewer accounts. Walking down that ranking:
  1. Skip cells whose density < `density_floor` (they are not knees at all — if *no* cell in the grid has density ≥ `density_floor`, the result is `knee_found=False`, AC2.3).
  2. A floor-passing cell is **rejected by guardrails** if `surviving_nodes > max_flagged_fraction × graph.vcount()` or `surviving_nodes < min_cluster_size`; rejection sets `guardrail_triggered=True` and the walk continues to the next-best candidate (AC2.4).
  3. First candidate passing floor + guardrails wins. None → `knee_found=False` (with `guardrail_triggered=True` if at least one floor-passing candidate was rejected).
- **"Flagged accounts" = vertices of the surviving core** at the chosen cell; the fraction is measured against the eligible accounts (`graph.vcount()`, the post-filter account population from Phase 3).
- **Determinism:** no randomness anywhere; identical input graphs produce identical results (eigenvector centrality is deterministic for a fixed graph; ARPACK non-determinism is not in play for igraph's default implementation — pinned by a determinism test regardless).

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: `dismantling.py` — grid search, surface, knee selection

**Verifies:** density-dismantling.AC2.1, AC2.2, AC2.3, AC2.4 (implementation; tests in Task 2)

**Files:**
- Create: `url_cosharing/src/url_cosharing/dismantling.py`

**Implementation:**

```python
# pattern: Functional Core
from __future__ import annotations

import logging
from dataclasses import dataclass

import igraph as ig
import numpy as np


@dataclass(frozen=True)
class GridCell:
    edge_quantile: float
    centrality_quantile: float
    min_component_density: float
    surviving_nodes: int
    surviving_edges: int


@dataclass(frozen=True)
class DismantlingResult:
    core: ig.Graph                  # surviving subgraph; empty graph when knee_found is False
    knee_found: bool
    edge_quantile: float            # 0.0 when no knee selected
    centrality_quantile: float      # 0.0 when no knee selected
    min_component_density: float    # 0.0 when no knee selected
    guardrail_triggered: bool
    surface: tuple[GridCell, ...]   # full grid, row-major over (edge_grid, centrality_grid)


def _empty_result(surface: tuple[GridCell, ...], guardrail_triggered: bool) -> DismantlingResult:
    return DismantlingResult(
        core=ig.Graph(),
        knee_found=False,
        edge_quantile=0.0,
        centrality_quantile=0.0,
        min_component_density=0.0,
        guardrail_triggered=guardrail_triggered,
        surface=surface,
    )


def _apply_thresholds(
    graph: ig.Graph,
    centralities: list[float],
    edge_threshold: float,
    centrality_threshold: float,
) -> ig.Graph:
    keep = [idx for idx, value in enumerate(centralities) if value >= centrality_threshold]
    sub = graph.induced_subgraph(keep)
    sub.delete_edges(sub.es.select(similarity_lt=edge_threshold))
    sub.delete_vertices(sub.vs.select(_degree=0))
    return sub


def _min_component_density(sub: ig.Graph) -> float:
    if sub.vcount() == 0:
        return 0.0
    components = sub.connected_components()
    return min(components.subgraph(idx).density(loops=False) for idx in range(len(components)))


def dismantle(
    graph: ig.Graph,
    edge_quantile_grid: tuple[float, ...],
    centrality_quantile_grid: tuple[float, ...],
    density_floor: float,
    max_flagged_fraction: float,
    min_cluster_size: int,
    logger: logging.Logger | None = None,
) -> DismantlingResult:
    """Density-based unsupervised network dismantling (Cinus et al., WWW '25),
    with automated threshold selection by knee detection on the
    minimum-component-density surface.

    Eigenvector centrality is computed on the full graph; on disconnected
    graphs igraph normalizes per-component to a shared max of 1, so scores are
    only strictly comparable within a component — the joint quantile filter
    follows the source paper and accepts this approximation.
    """
    if graph.vcount() == 0 or graph.ecount() == 0:
        return _empty_result(surface=(), guardrail_triggered=False)

    centralities = graph.eigenvector_centrality(weights='similarity', scale=True)
    edge_similarities = np.array(graph.es['similarity'])

    n_edge = len(edge_quantile_grid)
    n_cent = len(centrality_quantile_grid)
    density = np.zeros((n_edge, n_cent))
    cells: list[GridCell] = []
    subgraphs: dict[tuple[int, int], ig.Graph] = {}

    for i, eq in enumerate(edge_quantile_grid):
        edge_threshold = float(np.quantile(edge_similarities, eq))
        for j, cq in enumerate(centrality_quantile_grid):
            centrality_threshold = float(np.quantile(centralities, cq))
            sub = _apply_thresholds(graph, centralities, edge_threshold, centrality_threshold)
            density[i, j] = _min_component_density(sub)
            subgraphs[(i, j)] = sub
            cells.append(
                GridCell(
                    edge_quantile=eq,
                    centrality_quantile=cq,
                    min_component_density=density[i, j],
                    surviving_nodes=sub.vcount(),
                    surviving_edges=sub.ecount(),
                )
            )

    candidates = []
    for i in range(n_edge):
        for j in range(n_cent):
            jumps = []
            if i > 0:
                jumps.append(density[i, j] - density[i - 1, j])
            if j > 0:
                jumps.append(density[i, j] - density[i, j - 1])
            if jumps:
                candidates.append((max(jumps), density[i, j], i, j))

    candidates.sort(
        key=lambda c: (c[0], c[1], edge_quantile_grid[c[2]], centrality_quantile_grid[c[3]]),
        reverse=True,
    )

    guardrail_triggered = False
    max_flagged = max_flagged_fraction * graph.vcount()
    for jump, cell_density, i, j in candidates:
        if cell_density < density_floor:
            continue
        sub = subgraphs[(i, j)]
        if sub.vcount() > max_flagged or sub.vcount() < min_cluster_size:
            guardrail_triggered = True
            if logger:
                logger.info(
                    f'guardrail rejected candidate eq={edge_quantile_grid[i]} '
                    f'cq={centrality_quantile_grid[j]}: {sub.vcount()} survivors'
                )
            continue
        if logger:
            logger.info(
                f'knee selected: eq={edge_quantile_grid[i]} cq={centrality_quantile_grid[j]} '
                f'density={cell_density} survivors={sub.vcount()}'
            )
        return DismantlingResult(
            core=sub,
            knee_found=True,
            edge_quantile=edge_quantile_grid[i],
            centrality_quantile=centrality_quantile_grid[j],
            min_component_density=float(cell_density),
            guardrail_triggered=guardrail_triggered,
            surface=tuple(cells),
        )

    if logger:
        logger.info('no qualifying knee found; flagging nothing')
    return _empty_result(surface=tuple(cells), guardrail_triggered=guardrail_triggered)
```

Memory note: `subgraphs` holds one filtered copy per grid cell (49 by default). Cells are strictly smaller than the input graph and the account population is bounded by the Phase 2/3 filters; if this becomes a problem at scale, recomputing the winning subgraph after selection is the drop-in fix — noted here, not required.

**Verification:**
Run: `cd url_cosharing && uv run pytest` (module imports cleanly; tests come in Task 2)
Expected: suite passes.

**Commit:** combined with Task 2.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `test_dismantling.py` — planted-core, no-transition, and guardrail cases

**Verifies:** density-dismantling.AC2.1, AC2.2, AC2.3, AC2.4

**Files:**
- Test: `url_cosharing/tests/test_dismantling.py` (unit; `# pattern: Functional Core` header comment)

**Testing:**

Build synthetic graphs deterministically (explicit edge lists — **no randomness**; repo tests never use RNG). Helper fixture pattern:

- `planted_core_graph()`: 8-node clique with `similarity` weights 0.9 (vertex names `core0..core7`) plus ~30 "organic" background vertices (`bg0..bg29`) connected in a sparse ring/chain with weights 0.1–0.3, plus a handful of weak bridges (weight 0.1) between background and core so the graph is connected and the background is not trivially separable.
- Small explicit graphs for boundary cases.

Test classes and cases:

- `TestDismantleSurface` (AC2.1):
  - Surface has `len(edge_grid) × len(centrality_grid)` cells with the right quantile labels.
  - Isolates dropped before density: a hand-built graph where a mid-grid cell strands one vertex with no edges — that cell's `surviving_nodes` excludes it and its `min_component_density` reflects only ≥2-vertex components (pin with an exact expected density like `1.0` for a surviving 3-clique).
  - Density values within [0, 1].
- `TestDismantleKnee` (AC2.2):
  - `planted_core_graph()` with grids spanning 0.5–0.99, `density_floor=0.5`, generous `max_flagged_fraction` (e.g. 0.5), `min_cluster_size=3`: `knee_found is True` and `set(result.core.vs['name']) == {'core0', ..., 'core7'}` — the planted core is recovered exactly.
  - Chosen `min_component_density >= density_floor`; chosen quantiles are members of the input grids.
  - Determinism: two calls on the same graph give identical chosen quantiles and core membership.
- `TestDismantleNoTransition` (AC2.3):
  - Uniform sparse graph (all weights equal, e.g. a 20-node ring at 0.2) with `density_floor=0.9`: no cell passes the floor → `knee_found is False`, `core.vcount() == 0`, `edge_quantile == 0.0`, `centrality_quantile == 0.0`, `min_component_density == 0.0`, `guardrail_triggered is False`.
  - Empty graph → same empty result, empty surface, no exception.
  - Graph with vertices but zero edges → empty result, no exception (eigenvector centrality is never called).
  - Disconnected input (regression pin for the per-component centrality caveat): two components with no edges between them (e.g. a 4-clique at 0.9 and a separate 5-ring at 0.2) → `dismantle` completes without exception and returns a fully-populated surface (one `GridCell` per grid cell, all densities in [0, 1]) — the contract is graceful handling, not cross-component score comparability.
- `TestDismantleGuardrails` (AC2.4):
  - `max_flagged_fraction` small enough that the best candidate's core exceeds it (e.g. planted-core graph with `max_flagged_fraction=0.05` over 38 vertices → cap ≈ 1.9 nodes): candidate rejected → `guardrail_triggered is True`, and the result is either a next-best candidate that fits or `knee_found is False`; assert whichever the constructed graph implies — construct so that NO candidate fits, then assert `knee_found is False and guardrail_triggered is True`.
  - Next-best selection: construct a graph with two density plateaus (a large medium-dense group and a small very-dense group) where the largest-jump candidate flags too many accounts but a later candidate (the small group) passes → `knee_found is True`, core is the small group, `guardrail_triggered is True`.
  - `min_cluster_size` guardrail: winning cell would leave 2 survivors with `min_cluster_size=3` → rejected.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes.

**Commit:**
```bash
git add url_cosharing/src/url_cosharing/dismantling.py url_cosharing/tests/test_dismantling.py
git commit -m "feat(url_cosharing): density-based dismantling with knee detection and guardrails"
```
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

---

## Phase completion checklist

- [ ] `cd url_cosharing && uv run pytest` passes.
- [ ] `dismantling.py` has no I/O or ClickHouse imports (`grep -n 'clickhouse\|from url_cosharing.db' src/url_cosharing/dismantling.py` → empty) — completes density-dismantling.AC4.2.
- [ ] Planted-core recovery, no-transition, and both guardrail behaviours pinned by tests (AC2.1–AC2.4).
- [ ] `DismantlingResult` carries every field the Phase 5 runs-table row needs (`knee_found`, quantiles, `min_component_density`, `guardrail_triggered`, core for `flagged_accounts`).

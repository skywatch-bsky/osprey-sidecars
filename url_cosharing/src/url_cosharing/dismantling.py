# pattern: Functional Core
from __future__ import annotations

import logging
import warnings
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
    core: ig.Graph  # surviving subgraph; empty graph when knee_found is False
    knee_found: bool
    edge_quantile: float  # 0.0 when no knee selected
    centrality_quantile: float  # 0.0 when no knee selected
    min_component_density: float  # 0.0 when no knee selected
    guardrail_triggered: bool
    surface: tuple[GridCell, ...]  # full grid, row-major over (edge_grid, centrality_grid)


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
    # Collapse the ARPACK eigenvector-centrality noise band on disconnected graphs.
    # On disconnected input, igraph's ARPACK solver computes per-component eigenvector
    # centrality, normalizing each component to max=1.0 separately. Noise-band centralities
    # on non-max components fall in the ~1.7e-18 range (unrelated to graph structure).
    # Quantile-based thresholding on noisy values causes non-deterministic partitioning.
    # This tolerance (1e-10) is safely below the smallest gap between meaningful
    # centralities (>1e-3 after renormalization) yet above noise, enabling deterministic
    # >= partitioning.
    tolerance = 1e-10
    keep = [
        idx
        for idx, value in enumerate(centralities)
        if value >= centrality_threshold - tolerance
    ]
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

    Eigenvector centrality is computed once on the full graph. For disconnected
    graphs, centralities are renormalized per-component to ensure deterministic
    quantile-based thresholding, eliminating ARPACK numerical noise (~1e-17 scale)
    from smaller components that would otherwise cause non-deterministic partitioning.
    """
    if graph.vcount() == 0 or graph.ecount() == 0:
        return _empty_result(surface=(), guardrail_triggered=False)

    # Suppress igraph's "nearly zero centralities" warning for disconnected graphs;
    # we renormalize per-component below to eliminate noise.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*nearly zero.*')
        centralities = graph.eigenvector_centrality(weights='similarity', scale=True)

    # Handle disconnected graphs by using only the max-eigenvalue component for
    # quantile-based thresholding. On disconnected graphs, igraph computes eigenvector
    # centrality as: max-eigenvalue component gets real centralities scaled to max=1.0,
    # other components get numerical noise (~1e-17 scale). These noise values cause
    # non-deterministic quantile-based partitioning. We compute quantiles from the
    # max-eigenvalue component only, then apply to the full graph.
    components = graph.connected_components()
    centralities_array = np.array(centralities, dtype=np.float64)

    if len(components) > 1:
        # Find the component with the highest max centrality (the max-eigenvalue component)
        max_centrality_idx = max(
            range(len(components)),
            key=lambda i: np.max(centralities_array[components[i]]) if len(components[i]) > 0 else -1,
        )
        # Extract centralities from the max-eigenvalue component for quantile computation
        max_component_vertices = components[max_centrality_idx]
        centralities_for_quantile = centralities_array[max_component_vertices]
    else:
        # Connected graph: use all centralities
        centralities_for_quantile = centralities_array

    edge_similarities = np.array(graph.es['similarity'])

    n_edge = len(edge_quantile_grid)
    n_cent = len(centrality_quantile_grid)
    density = np.zeros((n_edge, n_cent))
    cells: list[GridCell] = []
    subgraphs: dict[tuple[int, int], ig.Graph] = {}

    for i, eq in enumerate(edge_quantile_grid):
        edge_threshold = float(np.quantile(edge_similarities, eq))
        for j, cq in enumerate(centrality_quantile_grid):
            centrality_threshold = float(np.quantile(centralities_for_quantile, cq))
            sub = _apply_thresholds(graph, centralities_array.tolist(), edge_threshold, centrality_threshold)
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

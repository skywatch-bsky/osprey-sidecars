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

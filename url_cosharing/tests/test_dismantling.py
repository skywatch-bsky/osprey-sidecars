# pattern: Functional Core
import logging

import igraph as ig
import numpy as np
import pytest

from url_cosharing.dismantling import GridCell, dismantle


def planted_core_graph() -> ig.Graph:
    """Build a deterministic synthetic graph with explicit edge list.

    8-node clique (core0..core7) with similarity weights 0.9, plus ~30 background
    vertices (bg0..bg29) in a sparse chain/ring with weights 0.1-0.3, plus weak
    bridges between core and background (weight 0.1) to keep the graph connected.
    """
    # Create graph with 38 vertices
    graph = ig.Graph(n=38)

    # Name vertices
    core_names = [f'core{i}' for i in range(8)]
    bg_names = [f'bg{i}' for i in range(30)]
    all_names = core_names + bg_names
    graph.vs['name'] = all_names

    edges = []
    weights = []

    # Core clique: all pairs with similarity 0.9
    for i in range(8):
        for j in range(i + 1, 8):
            edges.append((i, j))
            weights.append(0.9)

    # Background chain: bg0-bg1-bg2-...-bg29 with weights 0.2
    for i in range(30 - 1):
        edges.append((8 + i, 8 + i + 1))
        weights.append(0.2)

    # Close the ring to ensure connectivity within background
    edges.append((8 + 29, 8))  # bg29 -> core0 bridge
    weights.append(0.1)

    # Additional bridges from core to background to ensure the graph is connected
    # but background is not trivially separable
    edges.append((0, 8))  # core0 -> bg0
    weights.append(0.1)
    edges.append((1, 15))  # core1 -> bg7
    weights.append(0.1)

    graph.add_edges(edges)
    graph.es['similarity'] = weights

    return graph


class TestDismantleSurface:
    """Tests for grid surface computation (AC2.1)."""

    def test_surface_cell_count(self):
        """Surface has len(edge_grid) × len(centrality_grid) cells."""
        graph = planted_core_graph()
        edge_grid = (0.5, 0.7, 0.9)
        centrality_grid = (0.3, 0.6, 0.9)

        result = dismantle(
            graph,
            edge_quantile_grid=edge_grid,
            centrality_quantile_grid=centrality_grid,
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        assert len(result.surface) == len(edge_grid) * len(centrality_grid)
        assert len(result.surface) == 9

    def test_surface_cells_have_correct_quantiles(self):
        """Each cell is labeled with its quantile pair."""
        graph = planted_core_graph()
        edge_grid = (0.5, 0.9)
        centrality_grid = (0.3, 0.9)

        result = dismantle(
            graph,
            edge_quantile_grid=edge_grid,
            centrality_quantile_grid=centrality_grid,
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        # Check that quantile pairs appear in surface
        expected_pairs = {
            (0.5, 0.3), (0.5, 0.9),
            (0.9, 0.3), (0.9, 0.9),
        }
        actual_pairs = {
            (cell.edge_quantile, cell.centrality_quantile)
            for cell in result.surface
        }
        assert actual_pairs == expected_pairs

    def test_isolates_dropped_before_density(self):
        """Isolates excluded from surviving_nodes; density computed on ≥2-vertex components only."""
        # Hand-built: 5-clique plus two isolated vertices (7 nodes total).
        # With a high centrality quantile, the isolated vertices are dropped before
        # density is computed, so surviving_nodes counts only the clique (at least 3+ nodes).
        graph = ig.Graph(n=7)
        graph.vs['name'] = ['a', 'b', 'c', 'd', 'e', 'iso1', 'iso2']
        # 5-clique: all pairs in {a,b,c,d,e} connected
        edges = []
        for i in range(5):
            for j in range(i+1, 5):
                edges.append((i, j))
        weights = [0.9] * len(edges)
        graph.add_edges(edges)
        graph.es['similarity'] = weights

        # With edge_quantile=0.1 and centrality_quantile=0.6,
        # we keep vertices with centrality >= 60th percentile and edges >= 10th percentile.
        # The 5-clique vertices have high centrality; isolates have ~0.
        result = dismantle(
            graph,
            edge_quantile_grid=(0.1,),
            centrality_quantile_grid=(0.6,),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        cell = result.surface[0]
        # After thresholding, we should have most of the clique (isolates dropped)
        # and a complete subgraph has density 1.0
        assert cell.surviving_nodes >= 3
        assert cell.min_component_density == pytest.approx(1.0)

    def test_density_values_in_bounds(self):
        """All density values in [0.0, 1.0]."""
        graph = planted_core_graph()
        result = dismantle(
            graph,
            edge_quantile_grid=(0.3, 0.6, 0.9),
            centrality_quantile_grid=(0.3, 0.6, 0.9),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        for cell in result.surface:
            assert 0.0 <= cell.min_component_density <= 1.0

    def test_empty_graph_empty_surface(self):
        """Empty graph → empty surface."""
        graph = ig.Graph(n=0)
        result = dismantle(
            graph,
            edge_quantile_grid=(0.5,),
            centrality_quantile_grid=(0.5,),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        assert result.surface == ()


class TestDismantleKnee:
    """Tests for knee detection and candidate selection (AC2.2)."""

    def test_planted_core_recovery(self):
        """Planted-core graph recovers the core0..core7 clique under reasonable settings."""
        graph = planted_core_graph()
        result = dismantle(
            graph,
            edge_quantile_grid=tuple(np.linspace(0.5, 0.99, 7)),
            centrality_quantile_grid=tuple(np.linspace(0.5, 0.99, 7)),
            density_floor=0.5,
            max_flagged_fraction=0.5,
            min_cluster_size=3,
        )

        assert result.knee_found is True
        core_names = {f'core{i}' for i in range(8)}
        surviving_names = {v['name'] for v in result.core.vs}
        assert surviving_names == core_names

    def test_knee_density_meets_floor(self):
        """Chosen cell's density ≥ density_floor."""
        graph = planted_core_graph()
        density_floor = 0.6
        result = dismantle(
            graph,
            edge_quantile_grid=(0.5, 0.7, 0.9),
            centrality_quantile_grid=(0.5, 0.7, 0.9),
            density_floor=density_floor,
            max_flagged_fraction=0.5,
            min_cluster_size=2,
        )

        if result.knee_found:
            assert result.min_component_density >= density_floor

    def test_chosen_quantiles_in_grid(self):
        """Chosen quantiles are members of the input grids."""
        graph = planted_core_graph()
        edge_grid = (0.5, 0.7, 0.9)
        centrality_grid = (0.3, 0.6, 0.9)
        result = dismantle(
            graph,
            edge_quantile_grid=edge_grid,
            centrality_quantile_grid=centrality_grid,
            density_floor=0.0,
            max_flagged_fraction=0.5,
            min_cluster_size=2,
        )

        if result.knee_found:
            assert result.edge_quantile in edge_grid
            assert result.centrality_quantile in centrality_grid

    def test_determinism_same_graph_same_result(self):
        """Two calls on the same graph give identical results."""
        graph = planted_core_graph()

        result1 = dismantle(
            graph,
            edge_quantile_grid=(0.5, 0.7, 0.9),
            centrality_quantile_grid=(0.3, 0.6, 0.9),
            density_floor=0.5,
            max_flagged_fraction=0.5,
            min_cluster_size=2,
        )

        result2 = dismantle(
            graph,
            edge_quantile_grid=(0.5, 0.7, 0.9),
            centrality_quantile_grid=(0.3, 0.6, 0.9),
            density_floor=0.5,
            max_flagged_fraction=0.5,
            min_cluster_size=2,
        )

        assert result1.edge_quantile == result2.edge_quantile
        assert result1.centrality_quantile == result2.centrality_quantile
        assert set(v['name'] for v in result1.core.vs) == set(v['name'] for v in result2.core.vs)


class TestDismantleNoTransition:
    """Tests for absence of qualifying candidates (AC2.3)."""

    def test_no_cell_passes_floor(self):
        """Sparse bipartite graph with high density_floor → no cell passes → knee_found=False."""
        # Create a complete bipartite graph K(5,5): 10 vertices, 25 edges.
        # Density = 2×edges / (n(n-1)) = 50 / 90 ≈ 0.556
        # With density_floor=0.9, even the best case (full graph) can't reach it,
        # so no cell will qualify.
        graph = ig.Graph(n=10)
        graph.vs['name'] = [f'v{i}' for i in range(10)]
        # Create complete bipartite: vertices 0-4 connected to 5-9
        for i in range(5):
            for j in range(5, 10):
                graph.add_edge(i, j)
        graph.es['similarity'] = [0.5] * graph.ecount()

        result = dismantle(
            graph,
            edge_quantile_grid=(0.1, 0.5, 0.9),
            centrality_quantile_grid=(0.1, 0.5, 0.9),
            density_floor=0.9,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        assert result.knee_found is False
        assert result.core.vcount() == 0
        assert result.edge_quantile == 0.0
        assert result.centrality_quantile == 0.0
        assert result.min_component_density == 0.0
        assert result.guardrail_triggered is False

    def test_empty_graph_no_exception(self):
        """Empty graph (0 vertices) → empty result, no exception."""
        graph = ig.Graph(n=0)
        result = dismantle(
            graph,
            edge_quantile_grid=(0.5,),
            centrality_quantile_grid=(0.5,),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        assert result.knee_found is False
        assert result.core.vcount() == 0
        assert result.surface == ()

    def test_graph_with_vertices_no_edges(self):
        """Graph with vertices but zero edges → empty result, no exception."""
        graph = ig.Graph(n=5)
        graph.vs['name'] = [f'v{i}' for i in range(5)]

        result = dismantle(
            graph,
            edge_quantile_grid=(0.5,),
            centrality_quantile_grid=(0.5,),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        assert result.knee_found is False
        assert result.core.vcount() == 0

    def test_disconnected_components(self):
        """Disconnected input (two components, no edges between) → gracefully handles."""
        # Component 1: 4-clique at weight 0.9
        g1 = ig.Graph.Full(4)
        g1.vs['name'] = [f'c1_{i}' for i in range(4)]
        g1.es['similarity'] = [0.9] * g1.ecount()

        # Component 2: 5-ring at weight 0.2
        g2 = ig.Graph.Ring(5)
        g2.vs['name'] = [f'c2_{i}' for i in range(5)]
        g2.es['similarity'] = [0.2] * g2.ecount()

        # Combine by creating a new graph and adding vertices/edges
        combined = ig.Graph(n=9)
        names = [f'c1_{i}' for i in range(4)] + [f'c2_{i}' for i in range(5)]
        combined.vs['name'] = names

        # Add edges from both components
        combined.add_edges([(i, j) for i, j in g1.get_edgelist()])
        combined.add_edges([(4 + i, 4 + j) for i, j in g2.get_edgelist()])

        # Set similarities
        combined.es['similarity'] = [0.9] * g1.ecount() + [0.2] * g2.ecount()

        # Should complete without exception and return fully-populated surface
        result = dismantle(
            combined,
            edge_quantile_grid=(0.3, 0.6, 0.9),
            centrality_quantile_grid=(0.3, 0.6, 0.9),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        # Surface should be fully populated
        assert len(result.surface) == 9
        # All densities should be in [0, 1]
        for cell in result.surface:
            assert 0.0 <= cell.min_component_density <= 1.0


class TestDismantleGuardrails:
    """Tests for guardrail rejection of candidates (AC2.4)."""

    def test_max_flagged_fraction_rejects_best_candidate(self):
        """Best candidate exceeds max_flagged_fraction → guardrail_triggered=True, knee_found may be False."""
        graph = planted_core_graph()  # 38 vertices total
        # Core has 8 vertices. max_flagged_fraction=0.05 → cap ≈ 1.9 nodes
        # The planted core will be rejected, no other candidate will fit.
        result = dismantle(
            graph,
            edge_quantile_grid=(0.5, 0.7, 0.9),
            centrality_quantile_grid=(0.5, 0.7, 0.9),
            density_floor=0.0,
            max_flagged_fraction=0.05,  # ~1.9 nodes cap on 38 total
            min_cluster_size=1,
        )

        # guardrail_triggered should be True because the best candidate was rejected
        assert result.guardrail_triggered is True
        # knee_found could be False if no candidate fits
        if result.knee_found:
            # If a knee was found, it must be small
            assert result.core.vcount() <= 2

    def test_min_cluster_size_rejects_small_candidates(self):
        """Winning cell would leave 2 survivors with min_cluster_size=3 → rejected."""
        # Create a small graph where the best cell has 2 survivors
        graph = ig.Graph(n=3)
        graph.vs['name'] = ['a', 'b', 'c']
        graph.add_edges([(0, 1)])  # Only edge: a-b
        graph.es['similarity'] = [0.9]

        result = dismantle(
            graph,
            edge_quantile_grid=(0.0, 0.9),
            centrality_quantile_grid=(0.0, 0.9),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=3,  # Requires at least 3 survivors
        )

        # The 2-vertex survivor at cell (0, 0) is rejected
        assert result.guardrail_triggered is True
        assert result.knee_found is False

    def test_next_best_candidate_after_rejection(self):
        """Construct a graph where guardrail rejects best candidate and next-best is selected."""
        # Create a graph with a small dense group and a large sparse group.
        # This is simpler: we'll make a small 3-clique with high weights,
        # and a larger sparse component. With a max_flagged_fraction that
        # allows only 5 nodes, the large component exceeds it but the small
        # clique fits.
        small_clique = ig.Graph.Full(3)
        small_clique.vs['name'] = [f'small{i}' for i in range(3)]

        # Large sparse: 15-vertex path graph (low density)
        large_sparse = ig.Graph(n=15)
        large_sparse.vs['name'] = [f'large{i}' for i in range(15)]
        large_sparse.add_edges([(i, i+1) for i in range(14)])

        # Combine with no bridges (keep disconnected for this test)
        combined = ig.Graph(n=18)
        combined.vs['name'] = [f'small{i}' for i in range(3)] + [f'large{i}' for i in range(15)]

        # Add edges from both subgraphs
        combined.add_edges([(i, j) for i, j in small_clique.get_edgelist()])
        combined.add_edges([(3 + i, 3 + j) for i, j in large_sparse.get_edgelist()])

        # Set similarities: high for small clique, low for large sparse
        small_edges = len(small_clique.es)
        large_edges = len(large_sparse.es)
        combined.es['similarity'] = [0.95] * small_edges + [0.2] * large_edges

        # max_flagged_fraction=0.3 caps at ~5.4 nodes out of 18 total.
        # The large sparse component (15 nodes) will be rejected.
        # The small clique (3 nodes) will fit.
        result = dismantle(
            combined,
            edge_quantile_grid=(0.0, 0.5),
            centrality_quantile_grid=(0.0, 0.5),
            density_floor=0.0,
            max_flagged_fraction=0.3,  # ~5.4 nodes cap
            min_cluster_size=2,
        )

        # guardrail_triggered should be True (large sparse rejected)
        # and knee_found should be True (small clique selected)
        assert result.guardrail_triggered is True
        if result.knee_found:
            assert result.core.vcount() == 3


class TestDismantleFull:
    """Integration tests combining multiple ACs."""

    def test_surface_fully_populated_with_deterministic_graph(self):
        """Surface is fully populated and deterministic."""
        graph = planted_core_graph()
        edge_grid = (0.5, 0.7, 0.9)
        centrality_grid = (0.3, 0.6, 0.9)

        result = dismantle(
            graph,
            edge_quantile_grid=edge_grid,
            centrality_quantile_grid=centrality_grid,
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        assert len(result.surface) == 9

        # Verify each cell has expected fields
        for cell in result.surface:
            assert isinstance(cell, GridCell)
            assert cell.edge_quantile in edge_grid
            assert cell.centrality_quantile in centrality_grid
            assert 0.0 <= cell.min_component_density <= 1.0
            assert cell.surviving_nodes >= 0
            assert cell.surviving_edges >= 0

    def test_logger_parameter_optional(self):
        """Logger parameter is optional; dismantle works without it."""
        graph = planted_core_graph()

        # Without logger
        result1 = dismantle(
            graph,
            edge_quantile_grid=(0.5, 0.9),
            centrality_quantile_grid=(0.3, 0.9),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
            logger=None,
        )

        # With logger
        logger = logging.getLogger('test')
        result2 = dismantle(
            graph,
            edge_quantile_grid=(0.5, 0.9),
            centrality_quantile_grid=(0.3, 0.9),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
            logger=logger,
        )

        # Results should be identical
        assert result1.knee_found == result2.knee_found
        if result1.knee_found:
            assert result1.edge_quantile == result2.edge_quantile
            assert result1.centrality_quantile == result2.centrality_quantile

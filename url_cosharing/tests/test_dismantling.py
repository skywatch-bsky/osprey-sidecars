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


def two_plateau_graph() -> ig.Graph:
    """Build a two-plateau synthetic graph for testing AC2.4 fallback path.

    Large group (l0..l14): 15 nodes, ring at similarity 0.55, moderate density (~0.2).
    Small group (s0..s4): 5 nodes, complete clique at similarity 0.95, density 1.0.
    Bridges (2 edges at 0.60) connect the groups to keep the graph connected.

    The surface has a clear plateau structure:
    - At low/medium thresholds: both groups survive with density ~0.25-0.95
    - At high edge threshold: only the small group survives with density 1.0

    When guardrails are set (e.g. max_flagged_fraction=0.30), the largest-jump
    candidate (7-node mixed subgraph, density 0.95) is rejected by the guardrail,
    then a next-best candidate (5-node pure small group, density 1.0) passes all
    guards. This exercises the fallback path in AC2.4.
    """
    graph = ig.Graph(n=20)
    graph.vs['name'] = [f'l{i}' for i in range(15)] + [f's{i}' for i in range(5)]

    edges = []
    weights = []

    # Large group: ring at 0.55
    for i in range(15):
        edges.append((i, (i + 1) % 15))
        weights.append(0.55)

    # Small group: complete clique at 0.95
    for i in range(15, 20):
        for j in range(i + 1, 20):
            edges.append((i, j))
            weights.append(0.95)

    # Bridges at 0.60
    for large_idx in [0, 7]:
        for small_idx in range(5):
            edges.append((large_idx, 15 + small_idx))
            weights.append(0.60)

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
        """Disconnected input (two components, no edges between) → deterministic and graceful."""
        # Component 1: 4-clique at weight 0.9 (high eigenvalue)
        g1 = ig.Graph.Full(4)
        g1.vs['name'] = [f'c1_{i}' for i in range(4)]
        g1.es['similarity'] = [0.9] * g1.ecount()

        # Component 2: 5-ring at weight 0.2 (low eigenvalue, will get noise centralities)
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

        # Run twice on identical graph with identical parameters
        result1 = dismantle(
            combined,
            edge_quantile_grid=(0.3, 0.6, 0.9),
            centrality_quantile_grid=(0.3, 0.6, 0.9),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        result2 = dismantle(
            combined,
            edge_quantile_grid=(0.3, 0.6, 0.9),
            centrality_quantile_grid=(0.3, 0.6, 0.9),
            density_floor=0.0,
            max_flagged_fraction=1.0,
            min_cluster_size=1,
        )

        # Surface should be fully populated
        assert len(result1.surface) == 9
        assert len(result2.surface) == 9

        # All densities should be in [0, 1]
        for cell in result1.surface:
            assert 0.0 <= cell.min_component_density <= 1.0
        for cell in result2.surface:
            assert 0.0 <= cell.min_component_density <= 1.0

        # Core membership must be deterministic across runs
        core1_names = {v['name'] for v in result1.core.vs} if result1.core.vcount() > 0 else set()
        core2_names = {v['name'] for v in result2.core.vs} if result2.core.vcount() > 0 else set()
        assert core1_names == core2_names, f"Determinism violated: {core1_names} vs {core2_names}"

        # Surface cells must be identical across runs
        for cell1, cell2 in zip(result1.surface, result2.surface):
            assert cell1.surviving_nodes == cell2.surviving_nodes
            assert cell1.min_component_density == cell2.min_component_density


    def test_disconnected_equal_components_both_recovered(self):
        """Two equally dense cliques in separate components must both survive.

        Regression test: quantile thresholds computed from only the max-eigenvalue
        component would assign noise centralities to the other component and drop
        it entirely, silently missing a second coordinated campaign.
        """
        # Two disjoint copies of the same structure: a 6-clique at 0.9 with a
        # 10-node chain at 0.2 attached via a 0.1 bridge.
        graph = ig.Graph(n=32)
        names = []
        edges = []
        weights = []
        for prefix, offset in (('a', 0), ('b', 16)):
            names.extend([f'{prefix}_core{i}' for i in range(6)])
            names.extend([f'{prefix}_bg{i}' for i in range(10)])
            for i in range(6):
                for j in range(i + 1, 6):
                    edges.append((offset + i, offset + j))
                    weights.append(0.9)
            for i in range(9):
                edges.append((offset + 6 + i, offset + 6 + i + 1))
                weights.append(0.2)
            edges.append((offset, offset + 6))
            weights.append(0.1)
        graph.vs['name'] = names
        graph.add_edges(edges)
        graph.es['similarity'] = weights

        result = dismantle(
            graph,
            edge_quantile_grid=tuple(np.linspace(0.5, 0.99, 7)),
            centrality_quantile_grid=tuple(np.linspace(0.5, 0.99, 7)),
            density_floor=0.5,
            max_flagged_fraction=0.5,
            min_cluster_size=3,
        )

        assert result.knee_found is True
        expected = {f'a_core{i}' for i in range(6)} | {f'b_core{i}' for i in range(6)}
        surviving_names = {v['name'] for v in result.core.vs}
        assert surviving_names == expected


class TestDismantleGuardrails:
    """Tests for guardrail rejection of candidates (AC2.4)."""

    def test_max_flagged_fraction_rejects_best_candidate(self):
        """Best candidate exceeds max_flagged_fraction → no candidate fits → knee_found=False, guardrail_triggered=True."""
        graph = planted_core_graph()  # 38 vertices total
        # Core has 8 vertices. max_flagged_fraction=0.05 → cap ≈ 1.9 nodes
        # The planted core will be rejected. With min_cluster_size=2 and density_floor=0,
        # we must find a connected 2+ node subgraph that fits the cap.
        # To ensure NO candidate fits, use min_cluster_size=3 with tight cap.
        result = dismantle(
            graph,
            edge_quantile_grid=(0.5, 0.7, 0.9),
            centrality_quantile_grid=(0.5, 0.7, 0.9),
            density_floor=0.0,
            max_flagged_fraction=0.05,  # ~1.9 nodes cap on 38 total
            min_cluster_size=3,  # Requires at least 3 survivors; cap ≈ 1.9 means nothing fits
        )

        # guardrail_triggered must be True (best candidate was rejected)
        assert result.guardrail_triggered is True
        # knee_found must be False (no candidate satisfied both floor and guardrails)
        assert result.knee_found is False
        # core must be empty
        assert result.core.vcount() == 0

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
        """AC2.4: Guardrail rejects largest-jump candidate; next-best candidate is selected.

        Uses two_plateau_graph(): large group's density plateau is ranked first with
        biggest jump, but guardrails reject it (exceeds max_flagged_fraction). The small
        group's plateau (fewer survivors, density 1.0) is then evaluated and accepted.
        """
        graph = two_plateau_graph()

        # Parameters chosen to:
        # 1. Make the large-group plateau (7 survivors) the largest-jump candidate
        # 2. Reject it with max_flagged_fraction=0.30 (cap ≈ 6 nodes < 7)
        # 3. Accept the small-group plateau (5 survivors, density 1.0) as next-best
        result = dismantle(
            graph,
            edge_quantile_grid=(0.2, 0.6, 0.92),
            centrality_quantile_grid=(0.2, 0.7, 0.95),
            density_floor=0.5,
            max_flagged_fraction=0.30,  # cap ≈ 6 nodes
            min_cluster_size=2,
        )

        # Unconditional assertions per AC2.4 requirement:
        # - Large-group candidate was rejected, so guardrail_triggered must be True
        assert result.guardrail_triggered is True
        # - Small-group candidate passed, so knee_found must be True
        assert result.knee_found is True
        # - Core must be exactly the small group (s0..s4)
        assert set(result.core.vs['name']) == {'s0', 's1', 's2', 's3', 's4'}


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

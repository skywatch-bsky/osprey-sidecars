# pattern: Functional Core (pure formatter tests)
import igraph as ig
import numpy as np
from scipy.sparse import csr_array

from url_cosharing.calibrate import format_surface
from url_cosharing.dismantling import DismantlingResult, GridCell
from url_cosharing.similarity import ShareMatrix, SimilarityNetwork


class TestFormatSurface:
    """Unit tests for the pure format_surface() formatter function.

    Tests: header row, per-cell TSV lines, summary footer values,
    empty-surface case.
    """

    def test_format_surface_header_and_cells(self) -> None:
        """Format surface includes header row, one line per cell, correct TSV format."""
        # Minimal graph and network
        graph = ig.Graph(n=3)
        graph.add_edges([(0, 1), (1, 2)])
        graph.es['similarity'] = [0.7, 0.8]
        graph.vs['name'] = ['a', 'b', 'c']

        matrix = ShareMatrix(
            counts=csr_array((3, 2), dtype=np.float64),
            accounts=('a', 'b', 'c'),
            urls=('url1', 'url2'),
        )
        tfidf = csr_array((3, 2), dtype=np.float64)

        network = SimilarityNetwork(
            graph=graph,
            matrix=matrix,
            tfidf=tfidf,
            accounts_eligible=3,
            urls_eligible=2,
            graph_edges=2,
        )

        # Build a result with 2 grid cells
        surface = (
            GridCell(
                edge_quantile=0.5,
                centrality_quantile=0.6,
                min_component_density=0.75,
                surviving_nodes=2,
                surviving_edges=1,
            ),
            GridCell(
                edge_quantile=0.7,
                centrality_quantile=0.8,
                min_component_density=0.9,
                surviving_nodes=3,
                surviving_edges=2,
            ),
        )
        result = DismantlingResult(
            core=graph,
            knee_found=True,
            edge_quantile=0.5,
            centrality_quantile=0.6,
            min_component_density=0.75,
            guardrail_triggered=False,
            surface=surface,
        )

        output = format_surface(network, result, accounts_raw=5)
        lines = output.split('\n')

        # Verify header present
        assert lines[0] == 'edge_quantile\tcentrality_quantile\tmin_component_density\tsurviving_nodes\tsurviving_edges'

        # Verify one line per cell (cells on lines 1-2)
        assert lines[1] == '0.5\t0.6\t0.7500\t2\t1'
        assert lines[2] == '0.7\t0.8\t0.9000\t3\t2'

        # Verify empty line before footer
        assert lines[3] == ''

        # Verify footer carries exact counts
        assert 'accounts_raw=5' in lines[4]
        assert 'accounts_eligible=3' in lines[4]
        assert 'urls_eligible=2' in lines[4]
        assert 'graph_edges=2' in lines[4]

    def test_format_surface_summary_footer_values(self) -> None:
        """Summary footer lines include knee_found, quantiles, density, guardrail, flagged_accounts."""
        graph = ig.Graph(n=2)
        graph.add_edges([(0, 1)])
        graph.es['similarity'] = [0.85]
        graph.vs['name'] = ['x', 'y']

        matrix = ShareMatrix(
            counts=csr_array((2, 1), dtype=np.float64),
            accounts=('x', 'y'),
            urls=('url1',),
        )
        tfidf = csr_array((2, 1), dtype=np.float64)

        network = SimilarityNetwork(
            graph=graph,
            matrix=matrix,
            tfidf=tfidf,
            accounts_eligible=7,
            urls_eligible=3,
            graph_edges=1,
        )

        # Core has 2 nodes (flagged_accounts = core.vcount())
        surface = (
            GridCell(
                edge_quantile=0.6,
                centrality_quantile=0.7,
                min_component_density=0.5,
                surviving_nodes=2,
                surviving_edges=1,
            ),
        )
        result = DismantlingResult(
            core=graph,
            knee_found=True,
            edge_quantile=0.6,
            centrality_quantile=0.7,
            min_component_density=0.5,
            guardrail_triggered=True,
            surface=surface,
        )

        output = format_surface(network, result, accounts_raw=10)
        lines = output.split('\n')

        # Second footer line (index 4) should have summary values
        footer_summary = lines[4]
        assert 'knee_found=True' in footer_summary
        assert 'edge_quantile=0.6' in footer_summary
        assert 'centrality_quantile=0.7' in footer_summary
        assert 'min_component_density=0.5000' in footer_summary
        assert 'guardrail_triggered=True' in footer_summary
        assert 'flagged_accounts=2' in footer_summary  # core.vcount()

    def test_format_surface_empty_surface(self) -> None:
        """Empty surface (no grid cells) produces header and footer without error."""
        graph = ig.Graph()  # empty graph
        matrix = ShareMatrix(
            counts=csr_array((0, 0), dtype=np.float64),
            accounts=(),
            urls=(),
        )
        tfidf = csr_array((0, 0), dtype=np.float64)

        network = SimilarityNetwork(
            graph=graph,
            matrix=matrix,
            tfidf=tfidf,
            accounts_eligible=0,
            urls_eligible=0,
            graph_edges=0,
        )

        # Empty surface, knee_found = False (expected when no phase transition found)
        surface = ()
        result = DismantlingResult(
            core=graph,
            knee_found=False,
            edge_quantile=0.0,
            centrality_quantile=0.0,
            min_component_density=0.0,
            guardrail_triggered=False,
            surface=surface,
        )

        output = format_surface(network, result, accounts_raw=0)
        lines = output.split('\n')

        # Should still have header and footer
        assert lines[0] == 'edge_quantile\tcentrality_quantile\tmin_component_density\tsurviving_nodes\tsurviving_edges'
        # No data lines
        assert lines[1] == ''
        # Footer lines present
        assert 'accounts_raw=0' in lines[2]
        assert 'knee_found=False' in lines[3]

    def test_format_surface_density_precision(self) -> None:
        """Min component density formatted to 4 decimal places."""
        graph = ig.Graph(n=1)
        graph.vs['name'] = ['a']

        matrix = ShareMatrix(
            counts=csr_array((1, 1), dtype=np.float64),
            accounts=('a',),
            urls=('url1',),
        )
        tfidf = csr_array((1, 1), dtype=np.float64)

        network = SimilarityNetwork(
            graph=graph,
            matrix=matrix,
            tfidf=tfidf,
            accounts_eligible=1,
            urls_eligible=1,
            graph_edges=0,
        )

        surface = (
            GridCell(
                edge_quantile=0.5,
                centrality_quantile=0.5,
                min_component_density=0.123456,  # should round to 0.1235
                surviving_nodes=1,
                surviving_edges=0,
            ),
        )
        result = DismantlingResult(
            core=graph,
            knee_found=False,
            edge_quantile=0.0,
            centrality_quantile=0.0,
            min_component_density=0.0,
            guardrail_triggered=False,
            surface=surface,
        )

        output = format_surface(network, result, accounts_raw=1)
        lines = output.split('\n')

        # Density cell should be formatted to 4 decimals
        assert '0.1235' in lines[1]

    def test_format_surface_returns_string_with_trailing_newline_ready(self) -> None:
        """Output is suitable for sys.stdout.write() (no extra newlines in string itself)."""
        graph = ig.Graph(n=1)
        graph.vs['name'] = ['a']

        matrix = ShareMatrix(
            counts=csr_array((1, 1), dtype=np.float64),
            accounts=('a',),
            urls=('url1',),
        )
        tfidf = csr_array((1, 1), dtype=np.float64)

        network = SimilarityNetwork(
            graph=graph,
            matrix=matrix,
            tfidf=tfidf,
            accounts_eligible=1,
            urls_eligible=1,
            graph_edges=0,
        )

        surface = ()
        result = DismantlingResult(
            core=graph,
            knee_found=False,
            edge_quantile=0.0,
            centrality_quantile=0.0,
            min_component_density=0.0,
            guardrail_triggered=False,
            surface=surface,
        )

        output = format_surface(network, result, accounts_raw=1)

        # String should not have trailing \n (caller adds it via sys.stdout.write(...) + '\n')
        assert not output.endswith('\n')
        # But should have internal newlines
        assert '\n' in output

# pattern: Functional Core
import numpy as np
import pytest
from scipy.sparse import csr_array, diags_array

from url_cosharing.similarity import (
    UrlShareRow,
    build_share_matrix,
    build_similarity_graph,
    similarity_network,
    tfidf_transform,
)


class TestSqlFinalRowsNotRefiltered:
    """similarity_network must not re-apply SQL eligibility filters (AC1.2, AC1.3).

    fetch_url_shares_query computes the activity floor, df floor, and df
    ceiling in a single pass over the raw rolling-window population, then
    applies them together. Its output can legitimately contain accounts with
    fewer surviving URLs than min_unique_urls and URLs with fewer surviving
    sharers than min_url_sharers; recomputing any filter over that output
    uses reduced counts and erases valid detections.
    """

    def test_account_with_single_surviving_url_is_kept(self):
        # Raw window: a1 shares u1, u2, u3 (passes min_unique_urls=3); u1 has
        # df=2 via a1 and a2 (passes min_url_sharers=2); u2, u3 have df=1
        # (dropped); a2 has 1 unique URL (dropped). SQL's final output is a
        # single row, a1/u1 — an activity re-check (1 URL < 3) or a df
        # re-check (1 sharer < 2) would wrongly erase it.
        sql_final = [UrlShareRow(did='a1', url='u1', share_count=1)]

        result = similarity_network(sql_final, edge_epsilon=0.0)

        assert result.accounts_eligible == 1
        assert result.urls_eligible == 1
        assert result.graph.vcount() == 1

    def test_url_shared_by_every_input_account_is_kept(self):
        # A df ceiling recomputed over prefiltered rows uses the active-account
        # count as denominator instead of the raw sharing population and would
        # drop a legitimately hot coordination URL shared by everyone.
        rows = [UrlShareRow(did=f'a{i}', url='hot', share_count=1) for i in range(10)]
        rows += [UrlShareRow(did=f'a{i}', url=f'u{i}', share_count=1) for i in range(10)]

        result = similarity_network(rows, edge_epsilon=0.0)

        assert 'hot' in result.matrix.urls
        assert result.accounts_eligible == 10
        assert result.urls_eligible == 11

    def test_all_input_rows_reach_the_matrix(self):
        # No row of SQL's output may be dropped, whatever its shape.
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
        ]

        result = similarity_network(rows, edge_epsilon=0.0)

        assert result.accounts_eligible == 2
        assert result.urls_eligible == 2
        assert result.matrix.counts.sum() == 2

class TestBuildShareMatrix:
    """Tests for sparse account-by-url share matrix."""

    def test_known_2x3_case(self):
        """2 accounts × 3 URLs, exact values."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=2),
            UrlShareRow(did='a1', url='u2', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=3),
        ]
        matrix = build_share_matrix(rows)

        assert matrix.accounts == ('a1', 'a2')
        assert matrix.urls == ('u1', 'u2')

        dense = matrix.counts.toarray()
        expected = np.array([
            [2.0, 1.0],  # a1: u1=2, u2=1
            [0.0, 3.0],  # a2: u1=0, u2=3
        ])
        np.testing.assert_array_equal(dense, expected)

    def test_duplicate_entries_sum(self):
        """Duplicate (did, url) entries are summed."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=2),
            UrlShareRow(did='a1', url='u1', share_count=3),
        ]
        matrix = build_share_matrix(rows)

        assert matrix.counts.toarray()[0, 0] == 5.0

    def test_empty_input(self):
        """Empty input → shape (0, 0) and empty tuples."""
        matrix = build_share_matrix([])

        assert matrix.counts.shape == (0, 0)
        assert matrix.accounts == ()
        assert matrix.urls == ()

    def test_accounts_urls_sorted(self):
        """Accounts and URLs are sorted."""
        rows = [
            UrlShareRow(did='a3', url='u2', share_count=1),
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a2', url='u3', share_count=1),
        ]
        matrix = build_share_matrix(rows)

        assert matrix.accounts == ('a1', 'a2', 'a3')
        assert matrix.urls == ('u1', 'u2', 'u3')

class TestTfidfTransform:
    """Tests for hand-rolled sparse TF-IDF transform."""

    def test_known_vector_case(self):
        """2 accounts, 2 URLs: u1 is niche (df=1), u2 is ubiquitous (df=2).

        a1 shares u1×2 and u2×1
        a2 shares u2×3

        idf(u1) = ln(2/1) = ln 2 ≈ 0.693
        idf(u2) = ln(2/2) = ln 1 = 0

        Pre-norm a1: [2×ln2, 0] = [1.386, 0]
        L2 norm: sqrt(1.386²) = 1.386
        Normalized: [1.0, 0.0]

        Pre-norm a2: [0, 0×ln1] = [0, 0]
        Already zero, stays zero (no NaN)
        """
        # Create counts matrix manually
        data = np.array([2.0, 1.0, 3.0])
        row = np.array([0, 0, 1])
        col = np.array([0, 1, 1])
        counts = csr_array((data, (row, col)), shape=(2, 2))

        result = tfidf_transform(counts)
        dense = result.toarray()

        # a1 (row 0): [1.0, 0.0]
        assert dense[0, 0] == pytest.approx(1.0)
        assert dense[0, 1] == pytest.approx(0.0)

        # a2 (row 1): [0.0, 0.0] (ubiquitous URL, no contribution)
        assert dense[1, 0] == pytest.approx(0.0)
        assert dense[1, 1] == pytest.approx(0.0)

    def test_3account_case_with_hand_computed(self):
        """3 accounts, 3 URLs: all with df < N."""
        # a1: u1×1, u2×1
        # a2: u1×1, u3×1
        # a3: u2×1, u3×1
        # df = [2, 2, 2], idf = [ln(3/2), ln(3/2), ln(3/2)] ≈ [0.405, 0.405, 0.405]
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 2, 1, 2])
        counts = csr_array((data, (row, col)), shape=(3, 3))

        result = tfidf_transform(counts)
        dense = result.toarray()

        # Each nonzero row should have L2 norm ≈ 1
        for i in range(3):
            row_vec = dense[i]
            norm = np.linalg.norm(row_vec)
            if norm > 0:
                assert norm == pytest.approx(1.0)

    def test_empty_matrix(self):
        """Empty matrix returns empty, no error."""
        empty = csr_array((0, 0))
        result = tfidf_transform(empty)

        assert result.shape == (0, 0)

    def test_all_values_finite_and_nonnegative(self):
        """Property: all values finite, ≥ 0."""
        data = np.array([1.0, 2.0, 3.0, 1.0, 1.0])
        row = np.array([0, 0, 1, 1, 2])
        col = np.array([0, 1, 1, 2, 2])
        counts = csr_array((data, (row, col)), shape=(3, 3))

        result = tfidf_transform(counts)

        assert np.all(np.isfinite(result.data))
        assert np.all(result.data >= 0)

class TestBuildSimilarityGraph:
    """Tests for TF-IDF cosine similarity graph."""

    def test_identical_vectors_cosine_one(self):
        """Two accounts with identical share vectors → similarity ≈ 1.0 (AC1.4)."""
        # Two identical rows: [1, 0, 1]
        data = np.array([1.0, 1.0, 1.0, 1.0])
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 2, 0, 2])
        tfidf = csr_array((data, (row, col)), shape=(2, 3))

        # Normalize (they already are, but for clarity)
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
        tfidf_norm = diags_array(inv_norms) @ tfidf

        graph = build_similarity_graph(tfidf_norm, ('a1', 'a2'), edge_epsilon=0.0)

        assert graph.ecount() == 1
        assert graph.es[0]['similarity'] == pytest.approx(1.0)

    def test_disjoint_vectors_cosine_zero(self):
        """Two accounts with disjoint URL sets → cosine 0 → no edge."""
        # a1: [1, 0], a2: [0, 1]
        data = np.array([1.0, 1.0])
        row = np.array([0, 1])
        col = np.array([0, 1])
        tfidf = csr_array((data, (row, col)), shape=(2, 2))

        graph = build_similarity_graph(tfidf, ('a1', 'a2'), edge_epsilon=0.0)

        assert graph.ecount() == 0

    def test_epsilon_boundary(self):
        """Similarity just below epsilon gets no edge; at epsilon gets one."""
        # Create vectors that produce similarity ≈ 0.5
        # a1: [1, 1], a2: [1, 0] → dot product 1, norms sqrt(2) and 1 → cos ≈ 0.707
        data = np.array([1.0, 1.0, 1.0])
        row = np.array([0, 0, 1])
        col = np.array([0, 1, 0])
        tfidf = csr_array((data, (row, col)), shape=(2, 2))

        # Normalize
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
        tfidf_norm = diags_array(inv_norms) @ tfidf

        # Below epsilon: no edge
        graph_below = build_similarity_graph(tfidf_norm, ('a1', 'a2'), edge_epsilon=0.8)
        assert graph_below.ecount() == 0

        # At epsilon: edge
        sim_val = (tfidf_norm @ tfidf_norm.T).tocoo().data[0]  # Get actual similarity
        graph_at = build_similarity_graph(tfidf_norm, ('a1', 'a2'), edge_epsilon=sim_val)
        assert graph_at.ecount() == 1

    def test_weights_in_bounds(self):
        """All edge weights in [0, 1]."""
        data = np.array([1.0, 2.0, 1.0, 1.0])
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 1, 1, 1])
        tfidf = csr_array((data, (row, col)), shape=(3, 2))

        # Normalize
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
        tfidf_norm = diags_array(inv_norms) @ tfidf

        graph = build_similarity_graph(tfidf_norm, ('a1', 'a2', 'a3'), edge_epsilon=0.0)

        for edge in graph.es:
            assert 0 <= edge['similarity'] <= 1

    def test_isolate_accounts_preserved(self):
        """Accounts with no edges still appear as vertices."""
        # a1 and a2 disjoint, a3 isolated
        data = np.array([1.0, 1.0])
        row = np.array([0, 1])
        col = np.array([0, 1])
        tfidf = csr_array((data, (row, col)), shape=(3, 2))

        graph = build_similarity_graph(tfidf, ('a1', 'a2', 'a3'), edge_epsilon=0.0)

        assert graph.vcount() == 3
        assert [v['name'] for v in graph.vs] == ['a1', 'a2', 'a3']

    def test_vertex_names_match_accounts(self):
        """Vertex names in order match matrix.accounts."""
        data = np.array([1.0, 1.0])
        row = np.array([0, 1])
        col = np.array([0, 1])
        tfidf = csr_array((data, (row, col)), shape=(2, 2))

        graph = build_similarity_graph(tfidf, ('acc1', 'acc2'), edge_epsilon=0.0)

        assert [v['name'] for v in graph.vs] == ['acc1', 'acc2']

class TestSimilarityNetwork:
    """End-to-end integration tests."""

    def test_small_end_to_end(self):
        """End-to-end: counts and graph structure correct.

        Setup: 3 accounts, all share u1 (ubiquitous, df=3) and u2 (niche, df=2).
        After TF-IDF, u1 gets zero weight (df=N), but u2 gives a1-a2 an edge.
        """
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),  # df=3 (all)
            UrlShareRow(did='a1', url='u2', share_count=1),  # df=2 (a1, a2)
            UrlShareRow(did='a2', url='u1', share_count=1),  # df=3 (all)
            UrlShareRow(did='a2', url='u2', share_count=1),  # df=2 (a1, a2)
            UrlShareRow(did='a3', url='u1', share_count=1),  # df=3 (all)
            UrlShareRow(did='a3', url='u3', share_count=1),  # df=1 (only a3)
        ]

        result = similarity_network(rows=rows, edge_epsilon=0.0)

        assert result.accounts_raw == 3
        assert result.accounts_eligible == 3
        assert result.urls_eligible == 3
        assert result.graph.vcount() == 3
        # a1 and a2 both share u2 (df=2, niche), so they form exactly one edge
        # a3 shares only u1 (ubiquitous) and u3, no edge to others
        assert result.graph.ecount() == 1

    def test_ac15_empty_input(self):
        """AC1.5: empty input → empty graph, zero counts, no error."""
        result = similarity_network(rows=[], edge_epsilon=0.0)

        assert result.accounts_raw == 0
        assert result.accounts_eligible == 0
        assert result.urls_eligible == 0
        assert result.graph.vcount() == 0
        assert result.graph.ecount() == 0

    def test_ac15_disjoint_accounts_form_no_edges(self):
        """AC1.5: accounts with no shared URLs → vertices present, zero edges."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
        ]

        result = similarity_network(rows=rows, edge_epsilon=0.0)

        assert result.accounts_raw == 2
        assert result.accounts_eligible == 2
        assert result.urls_eligible == 2
        assert result.graph.vcount() == 2
        assert result.graph.ecount() == 0

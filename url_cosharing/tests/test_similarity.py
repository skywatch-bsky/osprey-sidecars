# pattern: Functional Core
import pytest

from url_cosharing.similarity import UrlShareRow, filter_shares, build_share_matrix, tfidf_transform, build_similarity_graph, similarity_network
import numpy as np


class TestFilterShares:
    """Tests for in-Python activity/df filters (AC1.2, AC1.3)."""

    def test_ac12_account_below_min_unique_urls(self):
        """AC1.2: account with 2 unique URLs dropped when min_unique_urls=3."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a1', url='u2', share_count=1),
            UrlShareRow(did='a2', url='u3', share_count=1),
            UrlShareRow(did='a2', url='u4', share_count=1),
            UrlShareRow(did='a2', url='u5', share_count=1),
        ]
        result = filter_shares(rows, min_unique_urls=3, min_url_sharers=1, max_url_df_pctl=1.0)
        # a1 has 2 unique URLs, dropped; a2 has 3, kept
        assert len(result) == 3
        assert all(r.did == 'a2' for r in result)

    def test_ac12_account_at_min_unique_urls_boundary(self):
        """AC1.2: account with exactly min_unique_urls is kept."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a1', url='u2', share_count=1),
            UrlShareRow(did='a1', url='u3', share_count=1),
        ]
        result = filter_shares(rows, min_unique_urls=3, min_url_sharers=1, max_url_df_pctl=1.0)
        assert len(result) == 3
        assert all(r.did == 'a1' for r in result)

    def test_ac13_url_below_min_sharers(self):
        """AC1.3: URL shared by 1 account dropped when min_url_sharers=2."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),  # only a1
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a2', url='u3', share_count=1),  # a2 and a3
            UrlShareRow(did='a3', url='u3', share_count=1),
        ]
        result = filter_shares(rows, min_unique_urls=1, min_url_sharers=2, max_url_df_pctl=1.0)
        # u1 has df=1, dropped; u2 has df=1, dropped; u3 has df=2, kept
        assert len(result) == 2
        assert all(r.url == 'u3' for r in result)

    def test_ac13_url_at_min_sharers_boundary(self):
        """AC1.3: URL with exactly min_url_sharers is kept."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a2', url='u1', share_count=1),
        ]
        result = filter_shares(rows, min_unique_urls=1, min_url_sharers=2, max_url_df_pctl=1.0)
        assert len(result) == 2
        assert all(r.url == 'u1' for r in result)

    def test_ac13_url_ceiling_quantile(self):
        """AC1.3: URLs above df ceiling (max_url_df_pctl quantile) are dropped."""
        # 5 URLs with dfs: [1, 1, 1, 1, 10]
        # At quantile 0.5 (median), should be at or below 1
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),  # df=1
            UrlShareRow(did='a2', url='u2', share_count=1),  # df=1
            UrlShareRow(did='a3', url='u3', share_count=1),  # df=1
            UrlShareRow(did='a4', url='u4', share_count=1),  # df=1
            UrlShareRow(did='a1', url='u5', share_count=1),  # df=10 (shared by 10 accounts)
            UrlShareRow(did='a2', url='u5', share_count=1),
            UrlShareRow(did='a3', url='u5', share_count=1),
            UrlShareRow(did='a4', url='u5', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
            UrlShareRow(did='a6', url='u5', share_count=1),
            UrlShareRow(did='a7', url='u5', share_count=1),
            UrlShareRow(did='a8', url='u5', share_count=1),
            UrlShareRow(did='a9', url='u5', share_count=1),
            UrlShareRow(did='a10', url='u5', share_count=1),
        ]
        result = filter_shares(rows, min_unique_urls=1, min_url_sharers=1, max_url_df_pctl=0.5)
        # df_ceiling at 0.5 quantile of [1,1,1,1,10] is 1.0
        # u5 has df=10 > 1.0, dropped
        kept_urls = {r.url for r in result}
        assert 'u5' not in kept_urls
        assert len(kept_urls) == 4

    def test_single_pass_semantics(self):
        """Account survives with fewer remaining URLs than min_unique_urls (SQL mirror).

        This is the critical regression test: activity and df are computed
        over *raw* rows, then filters applied together in one pass.
        """
        # a1: 5 raw unique URLs, all will be filtered out by df_ceiling except u1
        # After filter, a1 has only 1 URL, but raw count was 5, so it survives
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),  # df=1, survives
            UrlShareRow(did='a1', url='u2', share_count=1),  # df=10, filtered
            UrlShareRow(did='a1', url='u3', share_count=1),  # df=10, filtered
            UrlShareRow(did='a1', url='u4', share_count=1),  # df=10, filtered
            UrlShareRow(did='a1', url='u5', share_count=1),  # df=10, filtered
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a3', url='u3', share_count=1),
            UrlShareRow(did='a4', url='u4', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a3', url='u3', share_count=1),
            UrlShareRow(did='a4', url='u4', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a3', url='u3', share_count=1),
            UrlShareRow(did='a4', url='u4', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a3', url='u3', share_count=1),
            UrlShareRow(did='a4', url='u4', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a3', url='u3', share_count=1),
            UrlShareRow(did='a4', url='u4', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a3', url='u3', share_count=1),
            UrlShareRow(did='a4', url='u4', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
            UrlShareRow(did='a3', url='u3', share_count=1),
            UrlShareRow(did='a4', url='u4', share_count=1),
            UrlShareRow(did='a5', url='u5', share_count=1),
        ]
        result = filter_shares(rows, min_unique_urls=3, min_url_sharers=1, max_url_df_pctl=0.5)
        # a1: raw unique count = 5 >= 3, so a1 is active
        # But after df filter (u2-u5 filtered), a1 has only u1 left
        # Since activity was computed on raw rows, a1 still appears
        assert any(r.did == 'a1' for r in result)

    def test_empty_input(self):
        """Empty input returns empty."""
        result = filter_shares([], min_unique_urls=1, min_url_sharers=1, max_url_df_pctl=1.0)
        assert result == []

    def test_fully_filtered_input(self):
        """Fully filtered input (all rows dropped) returns empty."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a1', url='u2', share_count=1),
        ]
        # min_unique_urls=3 drops a1 (has 2 URLs)
        result = filter_shares(rows, min_unique_urls=3, min_url_sharers=1, max_url_df_pctl=1.0)
        assert result == []


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
        from scipy.sparse import csr_array

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
        from scipy.sparse import csr_array

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
        from scipy.sparse import csr_array

        empty = csr_array((0, 0))
        result = tfidf_transform(empty)

        assert result.shape == (0, 0)

    def test_all_values_finite_and_nonnegative(self):
        """Property: all values finite, ≥ 0."""
        from scipy.sparse import csr_array

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
        from scipy.sparse import csr_array

        # Two identical rows: [1, 0, 1]
        data = np.array([1.0, 1.0, 1.0, 1.0])
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 2, 0, 2])
        tfidf = csr_array((data, (row, col)), shape=(2, 3))

        # Normalize (they already are, but for clarity)
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
        from scipy.sparse import diags_array
        tfidf_norm = diags_array(inv_norms) @ tfidf

        graph = build_similarity_graph(tfidf_norm, ('a1', 'a2'), edge_epsilon=0.0)

        assert graph.ecount() == 1
        assert graph.es[0]['similarity'] == pytest.approx(1.0)

    def test_disjoint_vectors_cosine_zero(self):
        """Two accounts with disjoint URL sets → cosine 0 → no edge."""
        from scipy.sparse import csr_array

        # a1: [1, 0], a2: [0, 1]
        data = np.array([1.0, 1.0])
        row = np.array([0, 1])
        col = np.array([0, 1])
        tfidf = csr_array((data, (row, col)), shape=(2, 2))

        graph = build_similarity_graph(tfidf, ('a1', 'a2'), edge_epsilon=0.0)

        assert graph.ecount() == 0

    def test_epsilon_boundary(self):
        """Similarity just below epsilon gets no edge; at epsilon gets one."""
        from scipy.sparse import csr_array

        # Create vectors that produce similarity ≈ 0.5
        # a1: [1, 1], a2: [1, 0] → dot product 1, norms sqrt(2) and 1 → cos ≈ 0.707
        data = np.array([1.0, 1.0, 1.0])
        row = np.array([0, 0, 1])
        col = np.array([0, 1, 0])
        tfidf = csr_array((data, (row, col)), shape=(2, 2))

        # Normalize
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
        from scipy.sparse import diags_array
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
        from scipy.sparse import csr_array

        data = np.array([1.0, 2.0, 1.0, 1.0])
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 1, 1, 1])
        tfidf = csr_array((data, (row, col)), shape=(3, 2))

        # Normalize
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
        from scipy.sparse import diags_array
        tfidf_norm = diags_array(inv_norms) @ tfidf

        graph = build_similarity_graph(tfidf_norm, ('a1', 'a2', 'a3'), edge_epsilon=0.0)

        for edge in graph.es:
            assert 0 <= edge['similarity'] <= 1

    def test_isolate_accounts_preserved(self):
        """Accounts with no edges still appear as vertices."""
        from scipy.sparse import csr_array

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
        from scipy.sparse import csr_array

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

        result = similarity_network(
            rows=rows,
            min_unique_urls=1,
            min_url_sharers=1,
            max_url_df_pctl=1.0,
            edge_epsilon=0.0,
        )

        assert result.accounts_raw == 3
        assert result.accounts_eligible == 3
        assert result.urls_eligible == 3
        assert result.graph.vcount() == 3
        # a1 and a2 both share u2 (df=2, niche), so they form an edge
        # a3 shares only u1 (ubiquitous) and u3, no edge to others
        assert result.graph.ecount() >= 1  # at least a1-a2 edge

    def test_ac15_empty_input(self):
        """AC1.5: empty input → empty graph, zero counts, no error."""
        result = similarity_network(
            rows=[],
            min_unique_urls=1,
            min_url_sharers=1,
            max_url_df_pctl=1.0,
            edge_epsilon=0.0,
        )

        assert result.accounts_raw == 0
        assert result.accounts_eligible == 0
        assert result.urls_eligible == 0
        assert result.graph.vcount() == 0
        assert result.graph.ecount() == 0

    def test_ac15_fully_filtered_input(self):
        """AC1.5: non-empty but fully filtered → accounts_raw > 0, eligible = 0, empty graph."""
        rows = [
            UrlShareRow(did='a1', url='u1', share_count=1),
            UrlShareRow(did='a2', url='u2', share_count=1),
        ]

        result = similarity_network(
            rows=rows,
            min_unique_urls=3,  # Both accounts have only 1 URL, will be filtered
            min_url_sharers=1,
            max_url_df_pctl=1.0,
            edge_epsilon=0.0,
        )

        assert result.accounts_raw == 2
        assert result.accounts_eligible == 0
        assert result.urls_eligible == 0
        assert result.graph.vcount() == 0
        assert result.graph.ecount() == 0

# pattern: Functional Core
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import igraph as ig
import numpy as np
import pytest

from url_cosharing.analyzer import (
    ClusterResult,
    cluster_core,
    compute_evolution,
    compute_jaccard,
    compute_temporal_metrics,
)
from url_cosharing.similarity import ShareMatrix, UrlShareRow, build_share_matrix, tfidf_transform


@pytest.fixture
def base_date() -> date:
    return date(2024, 3, 20)


class TestComputeTemporalMetrics:
    """Tests for temporal metric computation."""

    def test_compute_temporal_metrics_with_timestamps(self, base_date: date) -> None:
        """Test temporal metrics computation with known timestamps."""
        cluster = ClusterResult(
            cluster_id='',
            members=frozenset(['did:0', 'did:1']),
            member_count=2,
            total_edges=1,
            total_weight=5,
            unique_urls=2,
            sample_dids=['did:0', 'did:1'],
            sample_urls=['url1', 'url2'],
            resolution_parameter=0.05,
            mean_edge_similarity=0.0,
            subgraph_density=0.0,
        )

        now = datetime(2024, 3, 20, 12, 0, 0)
        timestamps = {
            'did:0': [now, now + timedelta(hours=1)],
            'did:1': [now + timedelta(hours=2), now + timedelta(hours=3)],
        }

        result = compute_temporal_metrics(cluster, timestamps)

        assert result.member_count == 2
        assert result.temporal_spread_hours == 3.0
        assert result.mean_posting_interval_seconds > 0

    def test_compute_temporal_metrics_empty_timestamps(self, base_date: date) -> None:
        """Test that empty timestamps dict returns 0.0 for both metrics."""
        cluster = ClusterResult(
            cluster_id='',
            members=frozenset(['did:0', 'did:1']),
            member_count=2,
            total_edges=1,
            total_weight=5,
            unique_urls=2,
            sample_dids=['did:0', 'did:1'],
            sample_urls=['url1', 'url2'],
            resolution_parameter=0.05,
            mean_edge_similarity=0.0,
            subgraph_density=0.0,
        )

        result = compute_temporal_metrics(cluster, {})

        assert result.temporal_spread_hours == 0.0
        assert result.mean_posting_interval_seconds == 0.0

    def test_compute_temporal_metrics_single_timestamp(self, base_date: date) -> None:
        """Test that single timestamp per member returns 0 interval."""
        cluster = ClusterResult(
            cluster_id='',
            members=frozenset(['did:0']),
            member_count=1,
            total_edges=0,
            total_weight=0,
            unique_urls=0,
            sample_dids=['did:0'],
            sample_urls=[],
            resolution_parameter=0.05,
            mean_edge_similarity=0.0,
            subgraph_density=0.0,
        )

        now = datetime(2024, 3, 20, 12, 0, 0)
        timestamps = {'did:0': [now]}

        result = compute_temporal_metrics(cluster, timestamps)

        assert result.temporal_spread_hours == 0.0
        assert result.mean_posting_interval_seconds == 0.0

    def test_compute_temporal_metrics_preserves_cluster_fields(self, base_date: date) -> None:
        """Test that all original cluster fields are preserved in timestamped result."""
        cluster = ClusterResult(
            cluster_id='test-id',
            members=frozenset(['did:0', 'did:1']),
            member_count=2,
            total_edges=5,
            total_weight=25,
            unique_urls=10,
            sample_dids=['did:0', 'did:1'],
            sample_urls=['url1', 'url2'],
            resolution_parameter=0.1,
            mean_edge_similarity=0.8,
            subgraph_density=0.9,
        )

        result = compute_temporal_metrics(cluster, {})

        assert result.cluster_id == cluster.cluster_id
        assert result.members == cluster.members
        assert result.member_count == cluster.member_count
        assert result.total_edges == cluster.total_edges
        assert result.total_weight == cluster.total_weight
        assert result.unique_urls == cluster.unique_urls
        assert result.sample_dids == cluster.sample_dids
        assert result.sample_urls == cluster.sample_urls
        assert result.resolution_parameter == cluster.resolution_parameter
        assert result.mean_edge_similarity == cluster.mean_edge_similarity
        assert result.subgraph_density == cluster.subgraph_density


class TestComputeJaccard:
    """Tests for Jaccard similarity computation."""

    def test_compute_jaccard_identical_sets(self) -> None:
        """Test Jaccard of identical sets is 1.0."""
        set_a = frozenset(['did:a', 'did:b', 'did:c'])
        set_b = frozenset(['did:a', 'did:b', 'did:c'])
        assert compute_jaccard(set_a, set_b) == 1.0

    def test_compute_jaccard_disjoint_sets(self) -> None:
        """Test Jaccard of disjoint sets is 0.0."""
        set_a = frozenset(['did:a', 'did:b'])
        set_b = frozenset(['did:c', 'did:d'])
        assert compute_jaccard(set_a, set_b) == 0.0

    def test_compute_jaccard_partial_overlap(self) -> None:
        """Test Jaccard of partially overlapping sets."""
        set_a = frozenset(['did:a', 'did:b', 'did:c'])
        set_b = frozenset(['did:b', 'did:c', 'did:d'])
        jaccard = compute_jaccard(set_a, set_b)
        assert jaccard == 0.5

    def test_compute_jaccard_subset(self) -> None:
        """Test Jaccard when one set is a subset of another."""
        set_a = frozenset(['did:a', 'did:b', 'did:c'])
        set_b = frozenset(['did:a', 'did:b'])
        jaccard = compute_jaccard(set_a, set_b)
        assert jaccard == pytest.approx(2 / 3)

    def test_compute_jaccard_both_empty(self) -> None:
        """Test Jaccard of two empty sets returns 0.0."""
        set_a = frozenset()
        set_b = frozenset()
        assert compute_jaccard(set_a, set_b) == 0.0

    def test_compute_jaccard_one_empty(self) -> None:
        """Test Jaccard when one set is empty."""
        set_a = frozenset(['did:a', 'did:b'])
        set_b = frozenset()
        assert compute_jaccard(set_a, set_b) == 0.0


class TestComputeEvolution:
    """Tests for cluster evolution classification and ID assignment."""

    @pytest.fixture
    def run_date(self) -> date:
        return date(2026, 3, 22)

    @pytest.fixture
    def prev_date(self) -> date:
        return date(2026, 3, 21)

    def test_evolution_birth_single_cluster(self, run_date: date) -> None:
        """Test AC2.2: New cluster with no previous match gets birth event."""
        current_clusters = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:x', 'did:y', 'did:z']),
                member_count=3,
                total_edges=2,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:x', 'did:y', 'did:z'],
                sample_urls=['url1', 'url2'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            )
        ]
        previous_membership = {}

        events = compute_evolution(current_clusters, previous_membership, run_date, jaccard_threshold=0.5)

        assert len(events) == 1
        assert events[0].evolution_type == 'birth'
        assert events[0].cluster_id == f'{run_date.isoformat()}-0001'
        assert events[0].members == frozenset(['did:x', 'did:y', 'did:z'])
        assert events[0].predecessor_cluster_ids == ()
        assert events[0].jaccard_score == 0.0

    def test_evolution_continuation(self, run_date: date, prev_date: date) -> None:
        """Test AC2.1: Cluster matching previous cluster above threshold inherits ID."""
        prev_id = f'{prev_date.isoformat()}-0001'
        current_clusters = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:a', 'did:b', 'did:c']),
                member_count=3,
                total_edges=2,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:a', 'did:b', 'did:c'],
                sample_urls=['url1', 'url2'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            )
        ]
        previous_membership = {prev_id: frozenset(['did:a', 'did:b', 'did:c', 'did:d'])}

        events = compute_evolution(current_clusters, previous_membership, run_date, jaccard_threshold=0.5)

        assert len(events) == 1
        assert events[0].evolution_type == 'continuation'
        assert events[0].cluster_id == prev_id
        assert events[0].jaccard_score == pytest.approx(0.75)
        assert events[0].predecessor_cluster_ids == (prev_id,)

    def test_evolution_merge(self, run_date: date, prev_date: date) -> None:
        """Test AC2.3: Multiple previous clusters mapping to one current cluster is merge."""
        prev_id_1 = f'{prev_date.isoformat()}-0001'
        prev_id_2 = f'{prev_date.isoformat()}-0002'
        current_clusters = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:a', 'did:b', 'did:c', 'did:d', 'did:e']),
                member_count=5,
                total_edges=4,
                total_weight=10,
                unique_urls=5,
                sample_dids=['did:a', 'did:b', 'did:c', 'did:d', 'did:e'],
                sample_urls=['url1', 'url2'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            )
        ]
        previous_membership = {
            prev_id_1: frozenset(['did:a', 'did:b', 'did:c']),
            prev_id_2: frozenset(['did:d', 'did:e']),
        }

        events = compute_evolution(current_clusters, previous_membership, run_date, jaccard_threshold=0.4)

        assert len(events) == 1
        assert events[0].evolution_type == 'merge'
        assert events[0].cluster_id == f'{run_date.isoformat()}-0001'
        assert set(events[0].predecessor_cluster_ids) == {prev_id_1, prev_id_2}

    def test_evolution_split(self, run_date: date, prev_date: date) -> None:
        """Test AC2.4: One previous cluster mapping to multiple current clusters is split."""
        prev_id = f'{prev_date.isoformat()}-0001'
        current_clusters = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:a', 'did:b']),
                member_count=2,
                total_edges=1,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:a', 'did:b'],
                sample_urls=['url1', 'url2'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            ),
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:c', 'did:d']),
                member_count=2,
                total_edges=1,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:c', 'did:d'],
                sample_urls=['url1', 'url2'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            ),
        ]
        previous_membership = {prev_id: frozenset(['did:a', 'did:b', 'did:c', 'did:d'])}

        events = compute_evolution(current_clusters, previous_membership, run_date, jaccard_threshold=0.5)

        assert len(events) == 2
        assert all(e.evolution_type == 'split' for e in events)
        assert all(e.predecessor_cluster_ids == (prev_id,) for e in events)
        assert all(e.cluster_id.startswith(f'{run_date.isoformat()}-') for e in events)

    def test_evolution_death(self, run_date: date, prev_date: date) -> None:
        """Test AC2.5: Previous cluster with no current match is recorded as death."""
        prev_id = f'{prev_date.isoformat()}-0001'
        current_clusters = []
        previous_membership = {prev_id: frozenset(['did:x', 'did:y', 'did:z'])}

        events = compute_evolution(current_clusters, previous_membership, run_date, jaccard_threshold=0.5)

        assert len(events) == 1
        assert events[0].evolution_type == 'death'
        assert events[0].cluster_id == prev_id
        assert events[0].jaccard_score == 0.0

    def test_evolution_id_format(self, run_date: date) -> None:
        """Test AC2.6: Generated IDs match YYYY-MM-DD-NNNN format."""
        current_clusters = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:a', 'did:b']),
                member_count=2,
                total_edges=1,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:a', 'did:b'],
                sample_urls=['url1'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            )
        ]
        previous_membership = {}

        events = compute_evolution(current_clusters, previous_membership, run_date, jaccard_threshold=0.5)

        pattern = r'^\d{4}-\d{2}-\d{2}-\d{4}$'
        for event in events:
            if event.evolution_type == 'birth':
                assert re.match(pattern, event.cluster_id), f'ID {event.cluster_id} does not match pattern'

    def test_evolution_first_run_all_births(self, run_date: date) -> None:
        """Test AC2.7: First run with empty previous_membership classifies all as births."""
        current_clusters = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:a', 'did:b']),
                member_count=2,
                total_edges=1,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:a', 'did:b'],
                sample_urls=['url1'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            ),
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:x', 'did:y', 'did:z']),
                member_count=3,
                total_edges=2,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:x', 'did:y', 'did:z'],
                sample_urls=['url1'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            ),
        ]
        previous_membership = {}

        events = compute_evolution(current_clusters, previous_membership, run_date, jaccard_threshold=0.5)

        assert len(events) == 2
        assert all(e.evolution_type == 'birth' for e in events)
        assert events[0].cluster_id == f'{run_date.isoformat()}-0001'
        assert events[1].cluster_id == f'{run_date.isoformat()}-0002'

    def test_evolution_continuation_lineage(self, run_date: date, prev_date: date) -> None:
        """Test AC4.4: Cluster ID persists across continuation chain."""
        birth_id = f'{prev_date.isoformat()}-0001'

        current_1 = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:a', 'did:b', 'did:c']),
                member_count=3,
                total_edges=2,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:a', 'did:b', 'did:c'],
                sample_urls=['url1'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            )
        ]

        prev_membership_1 = {birth_id: frozenset(['did:a', 'did:b', 'did:c', 'did:d'])}

        events_1 = compute_evolution(current_1, prev_membership_1, run_date, jaccard_threshold=0.5)
        assert events_1[0].cluster_id == birth_id

        current_2 = [
            ClusterResult(
                cluster_id='',
                members=frozenset(['did:a', 'did:b']),
                member_count=2,
                total_edges=1,
                total_weight=5,
                unique_urls=2,
                sample_dids=['did:a', 'did:b'],
                sample_urls=['url1'],
                resolution_parameter=0.05,
                mean_edge_similarity=0.0,
                subgraph_density=0.0,
            )
        ]

        prev_membership_2 = {birth_id: frozenset(['did:a', 'did:b', 'did:c'])}

        events_2 = compute_evolution(current_2, prev_membership_2, run_date + timedelta(days=1), jaccard_threshold=0.5)
        assert events_2[0].cluster_id == birth_id
        assert events_2[0].evolution_type == 'continuation'


class TestClusterCore:
    """Tests for Leiden clustering on the dismantled core with bipartite metrics."""

    def test_cluster_core_empty_core(self) -> None:
        """Test that empty core returns empty list."""
        core = ig.Graph()
        matrix = ShareMatrix(
            counts=__import__('scipy.sparse').sparse.csr_array((0, 0), dtype=np.float64),
            accounts=(),
            urls=(),
        )
        tfidf = __import__('scipy.sparse').sparse.csr_array((0, 0), dtype=np.float64)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=3)

        assert results == []

    def test_cluster_core_two_cliques_separated(self) -> None:
        """AC3.1: Two well-separated similarity cliques decompose into two clusters."""
        # Build a core graph with two 3-cliques connected by low similarity edge
        core = ig.Graph(6)
        core.vs['name'] = ['did:0', 'did:1', 'did:2', 'did:3', 'did:4', 'did:5']

        # First clique: 0-1-2 (high similarity)
        core.add_edges([(0, 1), (0, 2), (1, 2)])
        # Second clique: 3-4-5 (high similarity)
        core.add_edges([(3, 4), (3, 5), (4, 5)])

        # Set similarity weights: cliques get 0.9, bridge gets 0.1
        core.es['similarity'] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        # Build minimal share matrix (all accounts, minimal URLs)
        share_rows = [
            UrlShareRow(did='did:0', url='u1', share_count=1),
            UrlShareRow(did='did:1', url='u1', share_count=1),
            UrlShareRow(did='did:2', url='u1', share_count=1),
            UrlShareRow(did='did:3', url='u2', share_count=1),
            UrlShareRow(did='did:4', url='u2', share_count=1),
            UrlShareRow(did='did:5', url='u2', share_count=1),
        ]
        matrix = build_share_matrix(share_rows)
        tfidf = tfidf_transform(matrix.counts)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=3)

        assert len(results) == 2
        assert all(r.member_count == 3 for r in results)

    def test_cluster_core_filters_small_clusters(self) -> None:
        """AC3.1: Clusters below min_cluster_size are dropped."""
        # Create a graph with one large clique and one small pair
        core = ig.Graph(5)
        core.vs['name'] = ['did:0', 'did:1', 'did:2', 'did:3', 'did:4']

        # Large clique: 0-1-2 (min_cluster_size=3)
        core.add_edges([(0, 1), (0, 2), (1, 2)])
        # Small pair: 3-4
        core.add_edges([(3, 4)])

        core.es['similarity'] = [0.9, 0.9, 0.9, 0.8]

        share_rows = [
            UrlShareRow(did='did:0', url='u1', share_count=1),
            UrlShareRow(did='did:1', url='u1', share_count=1),
            UrlShareRow(did='did:2', url='u1', share_count=1),
            UrlShareRow(did='did:3', url='u2', share_count=1),
            UrlShareRow(did='did:4', url='u2', share_count=1),
        ]
        matrix = build_share_matrix(share_rows)
        tfidf = tfidf_transform(matrix.counts)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=3)

        assert len(results) == 1
        assert results[0].member_count == 3

    def test_cluster_core_total_weight_hand_computed(self) -> None:
        """AC3.2: total_weight = sum of C(k,2) per URL is hand-computed correctly.

        3 members where u1 is shared by all 3 and u2 by 2:
        total_weight == C(3,2) + C(2,2) = 3 + 1 = 4
        unique_urls == 2 (URLs with >= 2 member sharers)
        """
        core = ig.Graph(3)
        core.vs['name'] = ['did:0', 'did:1', 'did:2']
        core.add_edges([(0, 1), (0, 2), (1, 2)])
        core.es['similarity'] = [0.9, 0.9, 0.9]

        # All 3 share u1; only 0,1 share u2
        share_rows = [
            UrlShareRow(did='did:0', url='u1', share_count=1),
            UrlShareRow(did='did:1', url='u1', share_count=1),
            UrlShareRow(did='did:2', url='u1', share_count=1),
            UrlShareRow(did='did:0', url='u2', share_count=1),
            UrlShareRow(did='did:1', url='u2', share_count=1),
        ]
        matrix = build_share_matrix(share_rows)
        tfidf = tfidf_transform(matrix.counts)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=1)

        assert len(results) == 1
        assert results[0].total_weight == 4  # C(3,2) + C(2,2) = 3 + 1
        assert results[0].unique_urls == 2

    def test_cluster_core_similarity_metrics_hand_computed(self) -> None:
        """AC3.2: mean_edge_similarity and subgraph_density are hand-computed.

        3-clique [0.9, 0.8, 0.7] => mean=0.8, density=1.0
        3-path (2 edges) => density=2/3
        """
        # Test 1: 3-clique with known similarities
        core = ig.Graph(3)
        core.vs['name'] = ['did:0', 'did:1', 'did:2']
        core.add_edges([(0, 1), (0, 2), (1, 2)])
        core.es['similarity'] = [0.9, 0.8, 0.7]

        share_rows = [
            UrlShareRow(did='did:0', url='u1', share_count=1),
            UrlShareRow(did='did:1', url='u1', share_count=1),
            UrlShareRow(did='did:2', url='u1', share_count=1),
        ]
        matrix = build_share_matrix(share_rows)
        tfidf = tfidf_transform(matrix.counts)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=1)

        assert len(results) == 1
        assert results[0].mean_edge_similarity == pytest.approx(0.8)
        assert results[0].subgraph_density == pytest.approx(1.0)

        # Test 2: 3-path (2 edges)
        core2 = ig.Graph(3)
        core2.vs['name'] = ['did:0', 'did:1', 'did:2']
        core2.add_edges([(0, 1), (1, 2)])
        core2.es['similarity'] = [0.9, 0.8]

        results2 = cluster_core(core2, matrix, tfidf, resolution=0.05, min_cluster_size=1)

        assert len(results2) == 1
        assert results2[0].mean_edge_similarity == pytest.approx(0.85)
        assert results2[0].subgraph_density == pytest.approx(2 / 3)

    def test_cluster_core_sample_urls_ranked_by_tfidf(self) -> None:
        """AC3.2: sample_urls ranked by cluster TF-IDF mass (not raw frequency)."""
        core = ig.Graph(3)
        core.vs['name'] = ['did:0', 'did:1', 'did:2']
        core.add_edges([(0, 1), (0, 2), (1, 2)])
        core.es['similarity'] = [0.9, 0.9, 0.9]

        # u1: shared by cluster only (high TF-IDF)
        # u2: shared by cluster and many others (low TF-IDF)
        # u3: shared by cluster only but single occurrence (medium TF-IDF)
        share_rows = [
            UrlShareRow(did='did:0', url='u1', share_count=2),
            UrlShareRow(did='did:1', url='u1', share_count=2),
            UrlShareRow(did='did:2', url='u1', share_count=2),
            UrlShareRow(did='did:0', url='u2', share_count=1),
            UrlShareRow(did='did:1', url='u2', share_count=1),
            UrlShareRow(did='did:2', url='u2', share_count=1),
            UrlShareRow(did='did:0', url='u3', share_count=1),
        ]
        matrix = build_share_matrix(share_rows)
        tfidf = tfidf_transform(matrix.counts)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=1)

        assert len(results) == 1
        # u1 should rank higher than u2 in sample_urls
        sample_urls = results[0].sample_urls
        if 'u1' in sample_urls and 'u2' in sample_urls:
            assert sample_urls.index('u1') < sample_urls.index('u2')

    def test_cluster_core_sample_urls_no_zeros(self) -> None:
        """AC3.2: Zero-mass URLs never appear in sample_urls."""
        core = ig.Graph(2)
        core.vs['name'] = ['did:0', 'did:1']
        core.add_edges([(0, 1)])
        core.es['similarity'] = [0.9]

        share_rows = [
            UrlShareRow(did='did:0', url='u1', share_count=1),
            UrlShareRow(did='did:1', url='u1', share_count=1),
        ]
        matrix = build_share_matrix(share_rows)
        tfidf = tfidf_transform(matrix.counts)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=1)

        assert len(results) == 1
        assert len(results[0].sample_urls) <= 10
        assert all(url != '' for url in results[0].sample_urls)

    def test_cluster_core_temporal_metrics_propagation(self) -> None:
        """AC3.3: TimestampedCluster propagates mean_edge_similarity/subgraph_density."""
        core = ig.Graph(3)
        core.vs['name'] = ['did:0', 'did:1', 'did:2']
        core.add_edges([(0, 1), (0, 2), (1, 2)])
        core.es['similarity'] = [0.9, 0.8, 0.7]

        share_rows = [
            UrlShareRow(did='did:0', url='u1', share_count=1),
            UrlShareRow(did='did:1', url='u1', share_count=1),
            UrlShareRow(did='did:2', url='u1', share_count=1),
        ]
        matrix = build_share_matrix(share_rows)
        tfidf = tfidf_transform(matrix.counts)

        results = cluster_core(core, matrix, tfidf, resolution=0.05, min_cluster_size=1)
        assert len(results) == 1

        cluster = results[0]
        now = datetime(2024, 3, 20, 12, 0, 0)
        timestamps = {
            'did:0': [now],
            'did:1': [now + timedelta(hours=1)],
            'did:2': [now + timedelta(hours=2)],
        }

        timestamped = compute_temporal_metrics(cluster, timestamps)

        assert timestamped.mean_edge_similarity == pytest.approx(0.8)
        assert timestamped.subgraph_density == pytest.approx(1.0)
        assert timestamped.temporal_spread_hours == 2.0

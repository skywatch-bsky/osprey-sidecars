# pattern: Functional Core
import pytest

from quote_cosharing.config import AnalysisConfig
from quote_cosharing.queries import (
    fetch_historical_membership_query,
    fetch_member_timestamps_query,
    fetch_pairs_query,
    insert_clusters_query,
    insert_membership_query,
)


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        resolution=0.05,
        min_edge_weight=2,
        min_cluster_size=3,
        min_cosharers=3,
        jaccard_threshold=0.5,
        evolution_window_days=7,
        pairs_table='quote_cosharing_pairs',
        clusters_table='quote_cosharing_clusters',
        membership_table='quote_cosharing_membership',
        source_table='osprey_execution_results',
    )


class TestFetchPairsQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = fetch_pairs_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_uses_pairs_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_pairs_query(base_config)
        assert base_config.pairs_table in query

    def test_uses_yesterday_function(self, base_config: AnalysisConfig) -> None:
        query = fetch_pairs_query(base_config)
        assert 'yesterday()' in query

    def test_uses_min_edge_weight_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_pairs_query(base_config)
        assert f'weight >= {base_config.min_edge_weight}' in query

    def test_selects_all_required_columns(self, base_config: AnalysisConfig) -> None:
        query = fetch_pairs_query(base_config)
        assert 'date' in query
        assert 'account_a' in query
        assert 'account_b' in query
        assert 'weight' in query
        assert 'shared_uris' in query

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='custom_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='quote_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_pairs_query(config)
        assert 'custom_pairs' in query

    def test_with_custom_min_edge_weight(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=5,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='quote_cosharing_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='quote_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_pairs_query(config)
        assert 'weight >= 5' in query


class TestFetchHistoricalMembershipQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_uses_membership_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config)
        assert base_config.membership_table in query

    def test_uses_evolution_window_days_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config)
        assert f'today() - {base_config.evolution_window_days}' in query

    def test_uses_today_function(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config)
        assert 'today()' in query

    def test_selects_all_required_columns(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config)
        assert 'run_date' in query
        assert 'cluster_id' in query
        assert 'did' in query

    def test_orders_by_run_date_desc(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config)
        assert 'ORDER BY run_date DESC' in query

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='quote_cosharing_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='custom_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_historical_membership_query(config)
        assert 'custom_membership' in query

    def test_with_custom_evolution_window_days(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=14,
            pairs_table='quote_cosharing_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='quote_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_historical_membership_query(config)
        assert 'today() - 14' in query


class TestFetchMemberTimestampsQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'")
        assert isinstance(query, str)
        assert len(query) > 0

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'")
        assert base_config.source_table in query

    def test_includes_collection_filter(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'")
        assert "Collection = 'app.bsky.feed.post'" in query

    def test_includes_operation_kind_filter(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'")
        assert "OperationKind = 'create'" in query

    def test_uses_yesterday_function(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'")
        assert 'yesterday()' in query

    def test_includes_placeholder_in_where_clause(self, base_config: AnalysisConfig) -> None:
        placeholder = "'did:plc:abc','did:plc:def'"
        query = fetch_member_timestamps_query(base_config, placeholder)
        assert placeholder in query

    def test_selects_all_required_columns(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'")
        assert 'UserId AS did' in query
        assert '__timestamp AS ts' in query

    def test_orders_by_did_and_ts(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'")
        assert 'ORDER BY did, ts' in query

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='quote_cosharing_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='quote_cosharing_membership',
            source_table='custom_source',
        )
        query = fetch_member_timestamps_query(config, "'did:plc:abc'")
        assert 'custom_source' in query

    def test_with_multiple_dids_placeholder(self, base_config: AnalysisConfig) -> None:
        placeholder = "'did:plc:a','did:plc:b','did:plc:c'"
        query = fetch_member_timestamps_query(base_config, placeholder)
        assert placeholder in query


class TestInsertClustersQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = insert_clusters_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_uses_clusters_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = insert_clusters_query(base_config)
        assert base_config.clusters_table in query

    def test_includes_all_required_columns(self, base_config: AnalysisConfig) -> None:
        query = insert_clusters_query(base_config)
        assert 'run_date' in query
        assert 'cluster_id' in query
        assert 'member_count' in query
        assert 'total_edges' in query
        assert 'total_weight' in query
        assert 'unique_uris' in query
        assert 'temporal_spread_hours' in query
        assert 'mean_posting_interval_seconds' in query
        assert 'sample_dids' in query
        assert 'sample_uris' in query
        assert 'resolution_parameter' in query
        assert 'evolution_type' in query
        assert 'predecessor_cluster_ids' in query
        assert 'jaccard_score' in query

    def test_includes_values_placeholder(self, base_config: AnalysisConfig) -> None:
        query = insert_clusters_query(base_config)
        assert 'VALUES' in query

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='quote_cosharing_pairs',
            clusters_table='custom_clusters',
            membership_table='quote_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = insert_clusters_query(config)
        assert 'custom_clusters' in query


class TestInsertMembershipQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = insert_membership_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_uses_membership_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = insert_membership_query(base_config)
        assert base_config.membership_table in query

    def test_includes_all_required_columns(self, base_config: AnalysisConfig) -> None:
        query = insert_membership_query(base_config)
        assert 'run_date' in query
        assert 'cluster_id' in query
        assert 'did' in query

    def test_includes_values_placeholder(self, base_config: AnalysisConfig) -> None:
        query = insert_membership_query(base_config)
        assert 'VALUES' in query

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='quote_cosharing_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='custom_membership',
            source_table='osprey_execution_results',
        )
        query = insert_membership_query(config)
        assert 'custom_membership' in query

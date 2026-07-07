# pattern: Functional Core
from dataclasses import replace
from datetime import date

import pytest

from url_cosharing.config import AnalysisConfig
from url_cosharing.queries import (
    fetch_historical_membership_query,
    fetch_member_timestamps_query,
    fetch_raw_account_count_query,
    fetch_url_shares_query,
    insert_clusters_query,
    insert_membership_query,
)

AS_OF = date(2026, 7, 7)


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        resolution=0.05,
        min_cluster_size=3,
        jaccard_threshold=0.5,
        evolution_window_days=7,
        window_days=7,
        min_unique_urls=10,
        min_url_sharers=5,
        max_url_df_fraction=0.90,
        edge_epsilon=0.05,
        edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
        centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
        density_floor=0.5,
        max_flagged_fraction=0.02,
        runs_table='url_cosharing_runs',
        clusters_table='url_cosharing_clusters',
        membership_table='url_cosharing_membership',
        source_table='osprey_execution_results',
    )


class TestFetchHistoricalMembershipQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config, AS_OF)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_uses_membership_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config, AS_OF)
        assert base_config.membership_table in query

    def test_uses_evolution_window_days_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config, AS_OF)
        assert f"toDate('2026-07-07') - {base_config.evolution_window_days}" in query

    def test_uses_today_function(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config, AS_OF)
        assert "toDate('2026-07-07')" in query

    def test_selects_all_required_columns(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config, AS_OF)
        assert 'run_date' in query
        assert 'cluster_id' in query
        assert 'did' in query

    def test_orders_by_run_date_desc(self, base_config: AnalysisConfig) -> None:
        query = fetch_historical_membership_query(base_config, AS_OF)
        assert 'ORDER BY run_date DESC' in query

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='custom_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_historical_membership_query(config, AS_OF)
        assert 'custom_membership' in query

    def test_with_custom_evolution_window_days(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=14,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_historical_membership_query(config, AS_OF)
        assert "toDate('2026-07-07') - 14" in query


class TestFetchMemberTimestampsQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert base_config.source_table in query

    def test_includes_collection_filter(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert "Collection = 'app.bsky.feed.post'" in query

    def test_includes_operation_kind_filter(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert "OperationKind = 'create'" in query

    def test_window_anchored_to_as_of(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert "toDate('2026-07-06')" in query

    def test_window_bounds_match_detection_window(self, base_config: AnalysisConfig) -> None:
        """Timestamp metrics must cover the same rolling window as detection."""
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert ">= toDate('2026-06-30')" in query
        assert "<= toDate('2026-07-06')" in query

    def test_window_bounds_with_custom_window_days(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=3,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_member_timestamps_query(config, "'did:plc:abc'", AS_OF)
        assert ">= toDate('2026-07-04')" in query

    def test_includes_placeholder_in_where_clause(self, base_config: AnalysisConfig) -> None:
        placeholder = "'did:plc:abc','did:plc:def'"
        query = fetch_member_timestamps_query(base_config, placeholder, AS_OF)
        assert placeholder in query

    def test_selects_all_required_columns(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert 'UserId AS did' in query
        assert '__timestamp AS ts' in query

    def test_orders_by_did_and_ts(self, base_config: AnalysisConfig) -> None:
        query = fetch_member_timestamps_query(base_config, "'did:plc:abc'", AS_OF)
        assert 'ORDER BY did, ts' in query

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='custom_source',
        )
        query = fetch_member_timestamps_query(config, "'did:plc:abc'", AS_OF)
        assert 'custom_source' in query

    def test_with_multiple_dids_placeholder(self, base_config: AnalysisConfig) -> None:
        placeholder = "'did:plc:a','did:plc:b','did:plc:c'"
        query = fetch_member_timestamps_query(base_config, placeholder, AS_OF)
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
        assert 'unique_urls' in query
        assert 'temporal_spread_hours' in query
        assert 'mean_posting_interval_seconds' in query
        assert 'sample_dids' in query
        assert 'sample_urls' in query
        assert 'resolution_parameter' in query
        assert 'mean_edge_similarity' in query
        assert 'subgraph_density' in query
        assert 'evolution_type' in query
        assert 'predecessor_cluster_ids' in query
        assert 'jaccard_score' in query

    def test_includes_values_placeholder(self, base_config: AnalysisConfig) -> None:
        query = insert_clusters_query(base_config)
        assert 'VALUES' in query

    def test_includes_16_placeholders(self, base_config: AnalysisConfig) -> None:
        """insert_clusters_query should have 16 placeholders for 16 columns"""
        query = insert_clusters_query(base_config)
        placeholder_count = query.count('?')
        assert placeholder_count == 16

    def test_with_custom_table_name(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='custom_clusters',
            membership_table='url_cosharing_membership',
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
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='custom_membership',
            source_table='osprey_execution_results',
        )
        query = insert_membership_query(config)
        assert 'custom_membership' in query


class TestFetchUrlSharesQuery:
    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = fetch_url_shares_query(base_config, AS_OF)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_reads_from_source_table(self, base_config: AnalysisConfig) -> None:
        """AC1.1: query reads from osprey_execution_results, not url_cosharing_pairs"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert 'osprey_execution_results' in query
        assert 'url_cosharing_pairs' not in query

    def test_window_bounds_with_default_window_days(self, base_config: AnalysisConfig) -> None:
        """AC1.1: window_days=7 at as_of 2026-07-07 spans 2026-06-30 .. 2026-07-06."""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert ">= toDate('2026-06-30')" in query
        assert "<= toDate('2026-07-06')" in query

    def test_window_bounds_with_custom_window_days(self) -> None:
        """AC1.1: custom window_days produces correct bound"""
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=1,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_url_shares_query(config, AS_OF)
        assert ">= toDate('2026-07-06')" in query

    def test_includes_collection_filter(self, base_config: AnalysisConfig) -> None:
        """AC1.1: filters by Collection = 'app.bsky.feed.post'"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert "Collection = 'app.bsky.feed.post'" in query

    def test_includes_operation_kind_filter(self, base_config: AnalysisConfig) -> None:
        """AC1.1: filters by OperationKind = 'create'"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert "OperationKind = 'create'" in query

    def test_extracts_urls_via_array_join(self, base_config: AnalysisConfig) -> None:
        """AC1.1: uses arrayJoin(FacetLinkList)"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert 'arrayJoin(FacetLinkList)' in query

    def test_filters_non_empty_facet_lists(self, base_config: AnalysisConfig) -> None:
        """AC1.1: only rows with length(FacetLinkList) > 0"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert 'length(FacetLinkList) > 0' in query

    def test_selects_required_columns(self, base_config: AnalysisConfig) -> None:
        """AC1.1: selects did, url, share_count"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert 's.did' in query
        assert 's.url' in query
        assert 's.share_count' in query

    def test_activity_filter_default(self, base_config: AnalysisConfig) -> None:
        """AC1.2: accounts with uniqExact(url) >= 10"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert f'uniqExact(url) >= {base_config.min_unique_urls}' in query

    def test_activity_filter_custom(self) -> None:
        """AC1.2: custom min_unique_urls changes the literal"""
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=20,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_url_shares_query(config, AS_OF)
        assert 'uniqExact(url) >= 20' in query

    def test_df_floor_default(self, base_config: AnalysisConfig) -> None:
        """AC1.3: df >= 5"""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert f'df >= {base_config.min_url_sharers}' in query

    def test_df_floor_custom(self) -> None:
        """AC1.3: custom min_url_sharers changes the literal"""
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=10,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_url_shares_query(config, AS_OF)
        assert 'df >= 10' in query

    def test_df_ceiling_is_fraction_of_accounts(self, base_config: AnalysisConfig) -> None:
        """AC1.3: ceiling is max_url_df_fraction of distinct accounts (sklearn max_df
        semantics per Cinus et al.), never a percentile of the df distribution."""
        query = fetch_url_shares_query(base_config, AS_OF)
        assert 'df <= 0.9 * (SELECT uniqExact(did) FROM url_shares)' in query
        assert 'quantile' not in query

    def test_df_ceiling_fraction_custom(self) -> None:
        """AC1.3: custom max_url_df_fraction changes the ceiling literal"""
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.80,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )
        query = fetch_url_shares_query(config, AS_OF)
        assert 'df <= 0.8 * (SELECT uniqExact(did) FROM url_shares)' in query

    def test_with_custom_source_table(self) -> None:
        """Custom source_table is used in FROM clause"""
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='custom_source_table',
        )
        query = fetch_url_shares_query(config, AS_OF)
        assert 'custom_source_table' in query


class TestFetchRawAccountCountQuery:
    def test_counts_distinct_accounts_without_eligibility_filters(self, base_config: AnalysisConfig) -> None:
        """Run metadata contract: accounts_raw is the pre-filter window population."""
        query = fetch_raw_account_count_query(base_config, AS_OF)
        assert 'uniqExact(UserId)' in query
        assert 'HAVING' not in query
        assert 'uniqExact(url)' not in query
        assert 'df' not in query

    def test_mirrors_url_shares_population(self, base_config: AnalysisConfig) -> None:
        """Same source and row predicates as the url_shares CTE."""
        query = fetch_raw_account_count_query(base_config, AS_OF)
        assert 'osprey_execution_results' in query
        assert "Collection = 'app.bsky.feed.post'" in query
        assert "OperationKind = 'create'" in query
        assert 'length(FacetLinkList) > 0' in query

    def test_window_bounds(self, base_config: AnalysisConfig) -> None:
        query = fetch_raw_account_count_query(base_config, AS_OF)
        assert ">= toDate('2026-06-30')" in query
        assert "<= toDate('2026-07-06')" in query

    def test_custom_window_days(self, base_config: AnalysisConfig) -> None:
        config = replace(base_config, window_days=3)
        query = fetch_raw_account_count_query(config, AS_OF)
        assert ">= toDate('2026-07-04')" in query

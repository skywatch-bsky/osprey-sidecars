# pattern: Functional Core
import pytest

from account_entropy.config import AnalysisConfig
from account_entropy.queries import account_activity_query


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        window_days=7,
        min_posts=10,
        hourly_entropy_norm_threshold=0.85,
        interval_entropy_norm_threshold=0.53,
        cv_threshold=0.5,
        interval_bin_edges=(60, 300, 900, 3600, 14400, 86400),
        source_table='osprey_execution_results',
        output_table='account_entropy_results',
    )


class TestAccountActivityQuery:
    def test_includes_post_collection_filter(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert "Collection = 'app.bsky.feed.post'" in query

    def test_includes_create_operation_filter(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert "OperationKind = 'create'" in query

    def test_includes_user_id_not_null_filter(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert 'UserId IS NOT NULL' in query

    def test_uses_window_days_from_config(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert f'INTERVAL {base_config.window_days} DAY' in query

    def test_uses_min_posts_threshold_in_having(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert f'HAVING post_count >= {base_config.min_posts}' in query

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert base_config.source_table in query

    def test_uses_custom_source_table(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            window_days=7,
            min_posts=10,
            hourly_entropy_norm_threshold=0.85,
            interval_entropy_norm_threshold=0.53,
            cv_threshold=0.5,
            interval_bin_edges=(60, 300, 900, 3600, 14400, 86400),
            source_table='custom_execution_results',
            output_table='account_entropy_results',
        )
        query = account_activity_query(config)
        assert 'custom_execution_results' in query

    def test_uses_custom_window_days(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            window_days=14,
            min_posts=10,
            hourly_entropy_norm_threshold=0.85,
            interval_entropy_norm_threshold=0.53,
            cv_threshold=0.5,
            interval_bin_edges=(60, 300, 900, 3600, 14400, 86400),
            source_table='osprey_execution_results',
            output_table='account_entropy_results',
        )
        query = account_activity_query(config)
        assert 'INTERVAL 14 DAY' in query

    def test_uses_custom_min_posts(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            window_days=7,
            min_posts=25,
            hourly_entropy_norm_threshold=0.85,
            interval_entropy_norm_threshold=0.53,
            cv_threshold=0.5,
            interval_bin_edges=(60, 300, 900, 3600, 14400, 86400),
            source_table='osprey_execution_results',
            output_table='account_entropy_results',
        )
        query = account_activity_query(config)
        assert 'HAVING post_count >= 25' in query

    def test_includes_hourly_bins_array(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert 'groupArray(toUInt8(toHour(e.__timestamp))) AS hourly_bins' in query

    def test_includes_ordered_timestamps_array(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert 'arraySort(groupArray(toUnixTimestamp64Milli(e.__timestamp))) AS ordered_timestamps' in query

    def test_includes_sample_rkeys(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert 'arraySlice(groupArray(e.__action_id), 1, 5) AS sample_rkeys' in query

    def test_uses_inner_join_with_active_accounts(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert 'INNER JOIN active_accounts a ON e.UserId = a.user_id' in query

    def test_groups_by_user_id_and_post_count(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert 'GROUP BY e.UserId, a.post_count' in query

    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = account_activity_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

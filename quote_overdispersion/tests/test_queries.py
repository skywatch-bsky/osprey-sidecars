# pattern: Functional Core
import pytest

from quote_overdispersion.config import AnalysisConfig
from quote_overdispersion.queries import daily_aggregation_query, hourly_aggregation_query


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=900,
        volume_p_threshold=0.01,
        density_p_threshold=0.01,
        baseline_days=14,
        cold_start_min_days=3,
        min_sharers=3,
        source_table='osprey_execution_results',
        output_table='quote_overdispersion_results',
    )


class TestDailyAggregationQuery:
    def test_includes_collection_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "Collection = 'app.bsky.feed.post'" in query

    def test_includes_operation_kind_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "OperationKind = 'create'" in query

    def test_includes_having_min_sharers_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'HAVING unique_sharers >= {base_config.min_sharers}' in query

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert base_config.source_table in query

    def test_uses_baseline_days_from_config(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'ROWS BETWEEN {base_config.baseline_days} PRECEDING' in query

    def test_uses_cold_start_threshold_from_config(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'>= {base_config.cold_start_min_days}' in query

    def test_uses_date_bucketing(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'toDate(__timestamp)' in query

    def test_ac1_8_coalesces_quoted_uri(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "if(PostEmbedRecordUri != '', PostEmbedRecordUri, PostEmbedRecordWithMediaUri) AS quoted_uri" in query

    def test_ac1_8_filters_both_embed_columns(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "(PostEmbedRecordUri != '' OR PostEmbedRecordWithMediaUri != '')" in query

    def test_includes_sample_dids(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'arraySlice(groupArray(DISTINCT UserId), 1, 5)' in query

    def test_includes_sharer_density(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'toFloat64(unique_sharers) / total_shares AS sharer_density' in query

    def test_includes_rolling_averages(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'avg(total_shares) OVER w AS rolling_volume_mean' in query
        assert 'avg(toFloat64(unique_sharers) / total_shares) OVER w AS rolling_density_mean' in query

    def test_quoted_uri_in_group_by(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'GROUP BY quoted_uri, bucket' in query

    def test_quoted_uri_in_partition_by(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'PARTITION BY quoted_uri ORDER BY bucket' in query

    def test_with_custom_min_sharers(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=10,
            source_table='osprey_execution_results',
            output_table='quote_overdispersion_results',
        )
        query = daily_aggregation_query(config)
        assert 'HAVING unique_sharers >= 10' in query

    def test_with_custom_table_names(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=3,
            source_table='custom_execution_results',
            output_table='custom_output',
        )
        query = daily_aggregation_query(config)
        assert 'custom_execution_results' in query

    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0


class TestHourlyAggregationQuery:
    def test_includes_collection_filter(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "Collection = 'app.bsky.feed.post'" in query

    def test_includes_operation_kind_filter(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "OperationKind = 'create'" in query

    def test_includes_having_min_sharers_filter(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'HAVING unique_sharers >= {base_config.min_sharers}' in query

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert base_config.source_table in query

    def test_uses_baseline_days_for_hours_calculation(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'ROWS BETWEEN {base_config.baseline_days * 24} PRECEDING' in query

    def test_uses_cold_start_threshold_for_hours(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'>= {base_config.cold_start_min_days * 24}' in query

    def test_uses_hour_bucketing(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'toStartOfHour(__timestamp)' in query

    def test_ac1_8_coalesces_quoted_uri(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "if(PostEmbedRecordUri != '', PostEmbedRecordUri, PostEmbedRecordWithMediaUri) AS quoted_uri" in query

    def test_ac1_8_filters_both_embed_columns(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "(PostEmbedRecordUri != '' OR PostEmbedRecordWithMediaUri != '')" in query

    def test_includes_sample_dids(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'arraySlice(groupArray(DISTINCT UserId), 1, 5)' in query

    def test_includes_sharer_density(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'toFloat64(unique_sharers) / total_shares AS sharer_density' in query

    def test_includes_rolling_averages(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'avg(total_shares) OVER w AS rolling_volume_mean' in query
        assert 'avg(toFloat64(unique_sharers) / total_shares) OVER w AS rolling_density_mean' in query

    def test_converts_baseline_hours_to_days(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'toUInt16(intDiv(b.baseline_buckets_available, 24))' in query

    def test_quoted_uri_in_group_by(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'GROUP BY quoted_uri, bucket' in query

    def test_quoted_uri_in_partition_by(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'PARTITION BY quoted_uri ORDER BY bucket' in query

    def test_with_custom_min_sharers(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=10,
            source_table='osprey_execution_results',
            output_table='quote_overdispersion_results',
        )
        query = hourly_aggregation_query(config)
        assert 'HAVING unique_sharers >= 10' in query

    def test_with_custom_table_names(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=3,
            source_table='custom_execution_results',
            output_table='custom_output',
        )
        query = hourly_aggregation_query(config)
        assert 'custom_execution_results' in query

    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

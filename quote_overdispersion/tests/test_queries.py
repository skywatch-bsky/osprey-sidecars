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

    def test_min_sharers_in_scored_entities_and_final_where(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'HAVING unique_sharers' not in query
        assert f'AND b.unique_sharers >= {base_config.min_sharers}' in query

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

    def test_quoted_uri_in_group_by(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'GROUP BY quoted_uri, bucket' in query

    def test_quoted_uri_in_partition_by(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'PARTITION BY quoted_uri' in query

    def test_uses_densified_pipeline(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'raw_shares' in query
        assert 'scored_entities' in query
        assert 'entities' in query
        assert 'calendar' in query
        assert 'dense' in query

    def test_uses_median_exact_for_volume(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'medianExact(total_shares) OVER w' in query

    def test_includes_rolling_median_and_variance(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'rolling_volume_median' in query
        assert 'rolling_volume_variance' in query
        assert 'rolling_density_variance' in query

    def test_includes_population_dispersion(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'population_volume_dispersion' in query
        assert 'population_density_variance' in query

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
        assert 'HAVING unique_sharers' not in query
        assert 'AND b.unique_sharers >= 10' in query

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

    def test_min_sharers_in_scored_entities_and_final_where(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'HAVING unique_sharers' not in query
        assert f'AND b.unique_sharers >= {base_config.min_sharers}' in query

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert base_config.source_table in query

    def test_uses_baseline_days_unscaled_for_rows(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'ROWS BETWEEN {base_config.baseline_days} PRECEDING' in query
        assert f'ROWS BETWEEN {base_config.baseline_days * 24} PRECEDING' not in query

    def test_uses_cold_start_threshold_unscaled(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'>= {base_config.cold_start_min_days}' in query
        assert f'>= {base_config.cold_start_min_days * 24}' not in query

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

    def test_quoted_uri_in_group_by(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'GROUP BY quoted_uri, bucket' in query

    def test_uses_densified_pipeline(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'raw_shares' in query
        assert 'scored_entities' in query
        assert 'entities' in query
        assert 'calendar' in query
        assert 'dense' in query

    def test_hour_of_day_matching(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'PARTITION BY quoted_uri, toHour(bucket)' in query

    def test_no_intDiv_in_query(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'intDiv' not in query

    def test_baseline_days_available_unscaled(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'toUInt16(b.baseline_buckets_available)' in query

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
        assert 'HAVING unique_sharers' not in query
        assert 'AND b.unique_sharers >= 10' in query

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

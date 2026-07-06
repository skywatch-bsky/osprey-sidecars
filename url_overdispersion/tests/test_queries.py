# pattern: Functional Core
import pytest

from url_overdispersion.config import AnalysisConfig
from url_overdispersion.queries import daily_aggregation_query, hourly_aggregation_query


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=900,
        volume_p_threshold=0.01,
        density_p_threshold=0.01,
        baseline_days=14,
        cold_start_min_days=3,
        min_sharers=3,
        watchlist_domains=(),
        source_table='osprey_execution_results',
        output_table='url_overdispersion_results',
    )


class TestDailyAggregationQuery:
    def test_includes_collection_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "Collection = 'app.bsky.feed.post'" in query

    def test_includes_operation_kind_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "OperationKind = 'create'" in query

    def test_no_having_filter_on_history(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'HAVING' not in query

    def test_min_sharers_in_scored_entities(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'WHERE bucket = toDate(now()) AND unique_sharers >= {base_config.min_sharers}' in query

    def test_min_sharers_in_final_where(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'WHERE b.bucket = toDate(now()) AND b.unique_sharers >= {base_config.min_sharers}' in query

    def test_no_rolling_volume_mean_not_null_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'rolling_volume_mean IS NOT NULL' not in query

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

    def test_includes_domain_join(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'arrayJoin(assumeNotNull(PostAllDomains))' in query

    def test_includes_sample_dids(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'arraySlice(groupArray(DISTINCT UserId), 1, 5)' in query

    def test_includes_sample_urls(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'arraySlice(groupArray(DISTINCT arrayJoin(assumeNotNull(FacetLinkList))), 1, 5)' in query

    def test_includes_densification_cross_join(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'CROSS JOIN calendar c' in query
        assert 'LEFT JOIN raw_shares r' in query
        assert 'c.bucket >= e.first_seen' in query

    def test_includes_calendar_numbers(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'numbers({base_config.baseline_days + 1})' in query

    def test_includes_coalesce_total_shares(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'coalesce(r.total_shares, 0) AS total_shares' in query

    def test_null_density_on_zero_shares(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert (
            'if(coalesce(r.total_shares, 0) > 0, toFloat64(r.unique_sharers) / r.total_shares, NULL) AS sharer_density'
            in query
        )

    def test_includes_median_exact_volume(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'medianExact(total_shares) OVER w AS rolling_volume_median' in query

    def test_includes_rolling_statistics(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'avg(total_shares) OVER w AS rolling_volume_mean' in query
        assert 'varPop(total_shares) OVER w' in query
        assert 'avg(sharer_density) OVER w AS rolling_density_mean' in query
        assert 'varPop(sharer_density) OVER w' in query

    def test_population_dispersion_calculation(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert (
            'median(if(rolling_volume_mean > 0, rolling_volume_variance / rolling_volume_mean, NULL)) AS population_volume_dispersion'
            in query
        )

    def test_with_custom_min_sharers(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=10,
            watchlist_domains=(),
            source_table='osprey_execution_results',
            output_table='url_overdispersion_results',
        )
        query = daily_aggregation_query(config)
        assert 'unique_sharers >= 10' in query

    def test_with_custom_table_names(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=3,
            watchlist_domains=(),
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

    def test_no_having_filter_on_history(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'HAVING' not in query

    def test_min_sharers_in_scored_entities(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'WHERE bucket = toStartOfHour(now()) AND unique_sharers >= {base_config.min_sharers}' in query

    def test_min_sharers_in_final_where(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'WHERE b.bucket = toStartOfHour(now()) AND b.unique_sharers >= {base_config.min_sharers}' in query

    def test_hour_of_day_matching(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'PARTITION BY domain, toHour(bucket)' in query

    def test_uses_baseline_days_not_scaled(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'ROWS BETWEEN {base_config.baseline_days} PRECEDING' in query
        assert f'ROWS BETWEEN {base_config.baseline_days * 24} PRECEDING' not in query

    def test_uses_cold_start_threshold_not_scaled(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'>= {base_config.cold_start_min_days}' in query
        assert f'>= {base_config.cold_start_min_days * 24}' not in query

    def test_no_intdiv_conversion(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'intDiv' not in query

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert base_config.source_table in query

    def test_uses_hour_bucketing(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'toStartOfHour(__timestamp)' in query

    def test_includes_domain_join(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'arrayJoin(assumeNotNull(PostAllDomains))' in query

    def test_includes_sample_dids(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'arraySlice(groupArray(DISTINCT UserId), 1, 5)' in query

    def test_includes_sample_urls(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'arraySlice(groupArray(DISTINCT arrayJoin(assumeNotNull(FacetLinkList))), 1, 5)' in query

    def test_includes_densification_cross_join(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'CROSS JOIN calendar c' in query
        assert 'LEFT JOIN raw_shares r' in query
        assert 'c.bucket >= e.first_seen' in query

    def test_includes_calendar_numbers(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'numbers({(base_config.baseline_days + 1) * 24})' in query

    def test_includes_interval_hour(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'toIntervalHour(number)' in query

    def test_includes_coalesce_total_shares(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'coalesce(r.total_shares, 0) AS total_shares' in query

    def test_null_density_on_zero_shares(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert (
            'if(coalesce(r.total_shares, 0) > 0, toFloat64(r.unique_sharers) / r.total_shares, NULL) AS sharer_density'
            in query
        )

    def test_includes_median_exact_volume(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'medianExact(total_shares) OVER w AS rolling_volume_median' in query

    def test_includes_rolling_statistics(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'avg(total_shares) OVER w AS rolling_volume_mean' in query
        assert 'varPop(total_shares) OVER w' in query
        assert 'avg(sharer_density) OVER w AS rolling_density_mean' in query
        assert 'varPop(sharer_density) OVER w' in query

    def test_population_dispersion_calculation(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert (
            'median(if(rolling_volume_mean > 0, rolling_volume_variance / rolling_volume_mean, NULL)) AS population_volume_dispersion'
            in query
        )

    def test_with_custom_min_sharers(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=10,
            watchlist_domains=(),
            source_table='osprey_execution_results',
            output_table='url_overdispersion_results',
        )
        query = hourly_aggregation_query(config)
        assert 'unique_sharers >= 10' in query

    def test_with_custom_table_names(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=3,
            watchlist_domains=(),
            source_table='custom_execution_results',
            output_table='custom_output',
        )
        query = hourly_aggregation_query(config)
        assert 'custom_execution_results' in query

    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

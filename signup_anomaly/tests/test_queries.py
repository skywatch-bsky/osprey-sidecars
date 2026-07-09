# pattern: Functional Core
import pytest

from signup_anomaly.config import AnalysisConfig
from signup_anomaly.queries import daily_aggregation_query, hourly_aggregation_query


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        daily_p_value_threshold=0.01,
        hourly_p_value_threshold=0.05,
        baseline_days=7,
        cold_start_min_days=3,
        excluded_hosts=('bsky.network', 'bridgy-fed.appspot.com', 'mostr.pub'),
        source_table='osprey_execution_results',
        output_table='pds_signup_anomalies',
    )


class TestDailyAggregationQuery:
    def test_includes_identity_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "ActionName = 'identity'" in query

    def test_daily_includes_account_age_filter(self, base_config: AnalysisConfig) -> None:
        """Only count events where the account was created on the same day."""
        query = daily_aggregation_query(base_config)
        assert 'parseDateTime64BestEffortOrNull(AccountCreatedAt) >= toStartOfDay(__timestamp)' in query

    def test_daily_account_age_filter_upper_bound(self, base_config: AnalysisConfig) -> None:
        """AccountCreatedAt must be bounded above to exclude future-dated values."""
        query = daily_aggregation_query(base_config)
        assert 'parseDateTime64BestEffortOrNull(AccountCreatedAt) < toStartOfDay(__timestamp) + INTERVAL 1 DAY' in query

    def test_daily_signup_count_uses_distinct(self, base_config: AnalysisConfig) -> None:
        """signup_count must count distinct accounts, not raw events."""
        query = daily_aggregation_query(base_config)
        assert 'countDistinct(UserId) AS signup_count' in query

    def test_includes_pds_host_not_null_filter(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'PdsHost IS NOT NULL' in query

    def test_excludes_bsky_network(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "PdsHost NOT LIKE '%bsky.network'" in query

    def test_excludes_bridgy_fed(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "PdsHost != 'bridgy-fed.appspot.com'" in query

    def test_excludes_mostr_pub(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert "PdsHost != 'mostr.pub'" in query

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert base_config.source_table in query

    def test_uses_baseline_days_from_config(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'ROWS BETWEEN {base_config.baseline_days} PRECEDING' in query

    def test_uses_cold_start_threshold_from_config(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert f'>= {base_config.cold_start_min_days}' in query

    def test_with_minimal_excluded_hosts(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            daily_p_value_threshold=0.01,
            hourly_p_value_threshold=0.05,
            baseline_days=7,
            cold_start_min_days=3,
            excluded_hosts=('bsky.network',),
            source_table='osprey_execution_results',
            output_table='pds_signup_anomalies',
        )
        query = daily_aggregation_query(config)
        assert "PdsHost NOT LIKE '%bsky.network'" in query
        assert 'bridgy-fed.appspot.com' not in query

    def test_includes_distinct_accounts(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'countDistinct(UserId) AS distinct_accounts' in query
        assert 'b.distinct_accounts' in query

    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_includes_variance_window_function(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'ifNotFinite(varPop(signup_count) OVER' in query
        assert f'ROWS BETWEEN {base_config.baseline_days} PRECEDING AND 1 PRECEDING' in query

    def test_includes_densification(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'CROSS JOIN calendar' in query
        assert f'numbers({base_config.baseline_days + 1})' in query
        assert 'LEFT JOIN raw_counts' in query
        assert 'coalesce(r.signup_count, 0)' in query
        assert 'c.day >= h.first_seen' in query

    def test_includes_rolling_median(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'medianExact(signup_count) OVER w' in query

    def test_dispersion_index_no_longer_guarded_at_1_0(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'dispersion_index IS NOT NULL' not in query
        assert '>= 1.0' not in query

    def test_population_stats_computes_unbiased_median(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'median(rolling_median) AS population_median_lambda' in query
        assert (
            'median(if(rolling_mean > 0, rolling_variance / rolling_mean, NULL)) AS population_dispersion_index'
            in query
        )

    def test_includes_dispersion_index_computation(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'if(b.rolling_mean > 0, b.rolling_variance / b.rolling_mean, NULL) AS dispersion_index' in query

    def test_select_includes_rolling_variance_column(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'b.rolling_variance' in query

    def test_select_includes_rolling_median_column(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'b.rolling_median' in query

    def test_select_includes_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'if(b.rolling_mean > 0, b.rolling_variance / b.rolling_mean, NULL) AS dispersion_index' in query

    def test_select_includes_population_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'p.population_dispersion_index' in query

    def test_filters_zero_current_count_rows(self, base_config: AnalysisConfig) -> None:
        """Zero-signup hosts are excluded from the BH-FDR family to avoid inflating n."""
        query = daily_aggregation_query(base_config)
        assert 'AND b.signup_count > 0' in query


class TestHourlyAggregationQuery:
    def test_includes_distinct_accounts(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'countDistinct(UserId) AS distinct_accounts' in query
        assert 'b.distinct_accounts' in query

    def test_includes_identity_filter(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "ActionName = 'identity'" in query

    def test_hourly_includes_account_age_filter(self, base_config: AnalysisConfig) -> None:
        """Only count events where the account was created within the same hour."""
        query = hourly_aggregation_query(base_config)
        assert 'parseDateTime64BestEffortOrNull(AccountCreatedAt) >= toStartOfHour(__timestamp)' in query

    def test_hourly_account_age_filter_upper_bound(self, base_config: AnalysisConfig) -> None:
        """AccountCreatedAt must be bounded above to exclude future-dated values."""
        query = hourly_aggregation_query(base_config)
        assert 'parseDateTime64BestEffortOrNull(AccountCreatedAt) < toStartOfHour(__timestamp) + INTERVAL 1 HOUR' in query

    def test_hourly_signup_count_uses_distinct(self, base_config: AnalysisConfig) -> None:
        """signup_count must count distinct accounts, not raw events."""
        query = hourly_aggregation_query(base_config)
        assert 'countDistinct(UserId) AS signup_count' in query

    def test_includes_pds_host_not_null_filter(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'PdsHost IS NOT NULL' in query

    def test_excludes_bsky_network(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "PdsHost NOT LIKE '%bsky.network'" in query

    def test_excludes_bridgy_fed(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "PdsHost != 'bridgy-fed.appspot.com'" in query

    def test_excludes_mostr_pub(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "PdsHost != 'mostr.pub'" in query

    def test_uses_source_table_from_config(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert base_config.source_table in query

    def test_uses_baseline_days_from_config_unscaled(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'ROWS BETWEEN {base_config.baseline_days} PRECEDING AND 1 PRECEDING' in query

    def test_uses_cold_start_threshold_from_config_unscaled(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'>= {base_config.cold_start_min_days}' in query

    def test_old_continuous_window_removed(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert '168 PRECEDING' not in query
        assert 'intDiv' not in query

    def test_with_minimal_excluded_hosts(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            daily_p_value_threshold=0.01,
            hourly_p_value_threshold=0.05,
            baseline_days=7,
            cold_start_min_days=3,
            excluded_hosts=('bsky.network',),
            source_table='osprey_execution_results',
            output_table='pds_signup_anomalies',
        )
        query = hourly_aggregation_query(config)
        assert "PdsHost NOT LIKE '%bsky.network'" in query
        assert 'bridgy-fed.appspot.com' not in query

    def test_returns_string(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert isinstance(query, str)
        assert len(query) > 0

    def test_includes_variance_window_function(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'ifNotFinite(varPop(signup_count) OVER' in query
        assert f'ROWS BETWEEN {base_config.baseline_days} PRECEDING AND 1 PRECEDING' in query

    def test_includes_hour_of_day_matching(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'PARTITION BY pds_host, toHour(bucket)' in query

    def test_includes_hourly_densification(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'numbers({(base_config.baseline_days + 1) * 24})' in query
        assert 'toIntervalHour(number)' in query
        assert 'LEFT JOIN raw_counts' in query
        assert 'coalesce(r.signup_count, 0)' in query

    def test_includes_rolling_median(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'medianExact(signup_count) OVER w' in query

    def test_includes_population_dispersion_median_formula(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert (
            'median(if(rolling_mean > 0, rolling_variance / rolling_mean, NULL)) AS population_dispersion_index'
            in query
        )

    def test_includes_dispersion_index_computation(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'if(b.rolling_mean > 0, b.rolling_variance / b.rolling_mean, NULL) AS dispersion_index' in query

    def test_select_includes_rolling_variance_column(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'b.rolling_variance' in query

    def test_select_includes_rolling_median_column(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'b.rolling_median' in query

    def test_select_includes_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'if(b.rolling_mean > 0, b.rolling_variance / b.rolling_mean, NULL) AS dispersion_index' in query

    def test_select_includes_population_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'p.population_dispersion_index' in query

    def test_filters_zero_current_count_rows(self, base_config: AnalysisConfig) -> None:
        """Zero-signup hosts are excluded from the BH-FDR family to avoid inflating n."""
        query = hourly_aggregation_query(base_config)
        assert 'AND b.signup_count > 0' in query

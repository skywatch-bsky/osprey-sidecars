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

    def test_includes_dispersion_index_mean_guard(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'WHEN avg(signup_count) OVER' in query
        assert '>= 1.0' in query
        assert 'ELSE NULL' in query
        assert 'AS dispersion_index' in query

    def test_includes_dispersion_index_division(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'varPop(signup_count) OVER' in query
        assert 'NULLIF(avg(signup_count) OVER' in query
        assert 'AS dispersion_index' in query

    def test_includes_population_dispersion_median_formula(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'median(dispersion_index) AS population_dispersion_index' in query
        assert 'dispersion_index IS NOT NULL' in query

    def test_select_includes_rolling_variance_column(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'b.rolling_variance' in query

    def test_select_includes_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'b.dispersion_index' in query

    def test_select_includes_population_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = daily_aggregation_query(base_config)
        assert 'p.population_dispersion_index' in query


class TestHourlyAggregationQuery:
    def test_includes_distinct_accounts(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'countDistinct(UserId) AS distinct_accounts' in query
        assert 'b.distinct_accounts' in query

    def test_includes_identity_filter(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert "ActionName = 'identity'" in query

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

    def test_uses_baseline_days_for_hours_calculation(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'ROWS BETWEEN {base_config.baseline_days * 24} PRECEDING' in query

    def test_uses_cold_start_threshold_for_hours(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert f'>= {base_config.cold_start_min_days * 24}' in query

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
        assert f'ROWS BETWEEN {base_config.baseline_days * 24} PRECEDING AND 1 PRECEDING' in query

    def test_includes_dispersion_index_mean_guard(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'WHEN avg(signup_count) OVER' in query
        assert '>= 1.0' in query
        assert 'ELSE NULL' in query
        assert 'AS dispersion_index' in query

    def test_includes_dispersion_index_division(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'varPop(signup_count) OVER' in query
        assert 'NULLIF(avg(signup_count) OVER' in query
        assert 'AS dispersion_index' in query

    def test_includes_population_dispersion_median_formula(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'median(dispersion_index) AS population_dispersion_index' in query
        assert 'dispersion_index IS NOT NULL' in query

    def test_select_includes_rolling_variance_column(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'b.rolling_variance' in query

    def test_select_includes_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'b.dispersion_index' in query

    def test_select_includes_population_dispersion_index_column(self, base_config: AnalysisConfig) -> None:
        query = hourly_aggregation_query(base_config)
        assert 'p.population_dispersion_index' in query

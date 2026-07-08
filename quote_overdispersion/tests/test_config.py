# pattern: Functional Core
import pytest

from quote_overdispersion.config import AnalysisConfig, AppConfig, ClickHouseConfig


@pytest.fixture
def base_analysis_config() -> AnalysisConfig:
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


class TestClickHouseConfig:
    def test_from_env_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv('OSPREY_CLICKHOUSE_HOST', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_PORT', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_USER', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_PASSWORD', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_DB', raising=False)

        config = ClickHouseConfig.from_env()

        assert config.host == 'localhost'
        assert config.port == 8123
        assert config.user == 'default'
        assert config.password == 'clickhouse'
        assert config.database == 'default'

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'clickhouse.example.com')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_PORT', '9000')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_USER', 'admin')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_PASSWORD', 'secret')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_DB', 'mydb')

        config = ClickHouseConfig.from_env()

        assert config.host == 'clickhouse.example.com'
        assert config.port == 9000
        assert config.user == 'admin'
        assert config.password == 'secret'
        assert config.database == 'mydb'

    def test_is_frozen(self) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        with pytest.raises(Exception):
            config.host = 'newhost'  # type: ignore


class TestAnalysisConfig:
    def test_from_env_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv('QUOTE_OVERDISPERSION_INTERVAL_SECONDS', raising=False)
        monkeypatch.delenv('QUOTE_OVERDISPERSION_VOLUME_P_THRESHOLD', raising=False)
        monkeypatch.delenv('QUOTE_OVERDISPERSION_DENSITY_P_THRESHOLD', raising=False)
        monkeypatch.delenv('QUOTE_OVERDISPERSION_BASELINE_DAYS', raising=False)
        monkeypatch.delenv('QUOTE_OVERDISPERSION_COLD_START_MIN_DAYS', raising=False)
        monkeypatch.delenv('QUOTE_OVERDISPERSION_MIN_SHARERS', raising=False)
        monkeypatch.delenv('QUOTE_OVERDISPERSION_SOURCE_TABLE', raising=False)
        monkeypatch.delenv('QUOTE_OVERDISPERSION_OUTPUT_TABLE', raising=False)

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 900
        assert config.volume_p_threshold == 0.01
        assert config.density_p_threshold == 0.01
        assert config.baseline_days == 14
        assert config.cold_start_min_days == 3
        assert config.min_sharers == 3
        assert config.source_table == 'osprey_execution_results'
        assert config.output_table == 'quote_overdispersion_results'

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_OVERDISPERSION_INTERVAL_SECONDS', '1800')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_VOLUME_P_THRESHOLD', '0.05')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_DENSITY_P_THRESHOLD', '0.10')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_BASELINE_DAYS', '21')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_COLD_START_MIN_DAYS', '7')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_MIN_SHARERS', '5')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_SOURCE_TABLE', 'custom_results')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_OUTPUT_TABLE', 'custom_output')

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 1800
        assert config.volume_p_threshold == 0.05
        assert config.density_p_threshold == 0.10
        assert config.baseline_days == 21
        assert config.cold_start_min_days == 7
        assert config.min_sharers == 5
        assert config.source_table == 'custom_results'
        assert config.output_table == 'custom_output'

    def test_source_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_OVERDISPERSION_SOURCE_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_output_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_OVERDISPERSION_OUTPUT_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_is_frozen(self, base_analysis_config: AnalysisConfig) -> None:
        with pytest.raises(Exception):
            base_analysis_config.interval_seconds = 1200  # type: ignore

    def test_construct_directly(self) -> None:
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=3,
            source_table='osprey_execution_results',
            output_table='quote_overdispersion_results',
        )

        assert config.interval_seconds == 900
        assert config.output_table == 'quote_overdispersion_results'


class TestAppConfig:
    def test_from_env_composes_both_configs(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'ch.example.com')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_INTERVAL_SECONDS', '1800')

        config = AppConfig.from_env()

        assert config.clickhouse.host == 'ch.example.com'
        assert config.analysis.interval_seconds == 1800

    def test_is_frozen(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'ch.example.com')
        config = AppConfig.from_env()

        with pytest.raises(Exception):
            config.analysis = AnalysisConfig.from_env()  # type: ignore

    def test_construct_directly(self) -> None:
        ch_config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        analysis_config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.01,
            density_p_threshold=0.01,
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=3,
            source_table='osprey_execution_results',
            output_table='quote_overdispersion_results',
        )
        config = AppConfig(
            clickhouse=ch_config,
            analysis=analysis_config,
        )

        assert config.clickhouse.host == 'localhost'
        assert config.analysis.min_sharers == 3


class TestTelemetryConfig:
    def test_defaults_disabled(self, monkeypatch) -> None:
        from quote_overdispersion.config import TelemetryConfig

        for name in ['ENABLED', 'SERVICE_NAME', 'SERVICE_VERSION', 'ENVIRONMENT', 'TRACES_ENABLED', 'METRICS_ENABLED']:
            monkeypatch.delenv('QUOTE_OVERDISPERSION_OTEL_' + name, raising=False)
        monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
        config = TelemetryConfig.from_env()
        assert config.enabled is False
        assert config.service_name == 'quote-overdispersion'
        assert config.service_version == '0.1.0'
        assert config.environment == 'local'
        assert config.otlp_endpoint is None
        assert config.traces_enabled is False
        assert config.metrics_enabled is False

    def test_enabled_env_and_endpoint(self, monkeypatch) -> None:
        from quote_overdispersion.config import TelemetryConfig

        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_ENABLED', 'yes')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_SERVICE_NAME', 'custom-service')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_SERVICE_VERSION', '1.2.3')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_ENVIRONMENT', 'prod')
        monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://collector:4317')
        config = TelemetryConfig.from_env()
        assert config.enabled is True
        assert config.traces_enabled is True
        assert config.metrics_enabled is True
        assert config.service_name == 'custom-service'
        assert config.service_version == '1.2.3'
        assert config.environment == 'prod'
        assert config.otlp_endpoint == 'http://collector:4317'

    def test_trace_metric_overrides(self, monkeypatch) -> None:
        from quote_overdispersion.config import TelemetryConfig

        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_ENABLED', 'true')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_TRACES_ENABLED', 'off')
        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_METRICS_ENABLED', 'on')
        config = TelemetryConfig.from_env()
        assert config.enabled is True
        assert config.traces_enabled is False
        assert config.metrics_enabled is True

    @pytest.mark.parametrize('raw', ['1', 'true', 'TRUE', 'yes', 'On'])
    def test_bool_true_forms(self, monkeypatch, raw: str) -> None:
        from quote_overdispersion.config import TelemetryConfig

        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_ENABLED', raw)
        assert TelemetryConfig.from_env().enabled is True

    @pytest.mark.parametrize('raw', ['0', 'false', 'FALSE', 'no', 'Off'])
    def test_bool_false_forms(self, monkeypatch, raw: str) -> None:
        from quote_overdispersion.config import TelemetryConfig

        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_ENABLED', raw)
        assert TelemetryConfig.from_env().enabled is False

    def test_invalid_bool_names_env_var(self, monkeypatch) -> None:
        from quote_overdispersion.config import TelemetryConfig

        monkeypatch.setenv('QUOTE_OVERDISPERSION_OTEL_ENABLED', 'maybe')
        with pytest.raises(ValueError, match='QUOTE_OVERDISPERSION_OTEL_ENABLED'):
            TelemetryConfig.from_env()

    def test_direct_app_config_defaults_to_disabled_telemetry(self, base_analysis_config) -> None:
        from quote_overdispersion.config import AppConfig, ClickHouseConfig

        config = AppConfig(
            clickhouse=ClickHouseConfig('localhost', 8123, 'default', 'clickhouse', 'default'),
            analysis=base_analysis_config,
        )
        assert config.telemetry.enabled is False

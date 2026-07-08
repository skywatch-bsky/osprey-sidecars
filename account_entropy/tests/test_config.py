# pattern: Functional Core
import pytest

from account_entropy.config import AnalysisConfig, AppConfig, ClickHouseConfig


@pytest.fixture
def base_analysis_config() -> AnalysisConfig:
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
        monkeypatch.delenv('ACCOUNT_ENTROPY_INTERVAL_SECONDS', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_WINDOW_DAYS', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_MIN_POSTS', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_CV_THRESHOLD', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_INTERVAL_BIN_EDGES', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_SOURCE_TABLE', raising=False)
        monkeypatch.delenv('ACCOUNT_ENTROPY_OUTPUT_TABLE', raising=False)

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 3600
        assert config.window_days == 7
        assert config.min_posts == 10
        assert config.hourly_entropy_norm_threshold == 0.85
        assert config.interval_entropy_norm_threshold == 0.53
        assert config.cv_threshold == 0.5
        assert config.interval_bin_edges == (60, 300, 900, 3600, 14400, 86400)
        assert config.source_table == 'osprey_execution_results'
        assert config.output_table == 'account_entropy_results'

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv('ACCOUNT_ENTROPY_INTERVAL_SECONDS', '1800')
        monkeypatch.setenv('ACCOUNT_ENTROPY_WINDOW_DAYS', '14')
        monkeypatch.setenv('ACCOUNT_ENTROPY_MIN_POSTS', '20')
        monkeypatch.setenv('ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD', '0.9')
        monkeypatch.setenv('ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD', '0.4')
        monkeypatch.setenv('ACCOUNT_ENTROPY_CV_THRESHOLD', '0.3')
        monkeypatch.setenv('ACCOUNT_ENTROPY_INTERVAL_BIN_EDGES', '30,120,600,3600')
        monkeypatch.setenv('ACCOUNT_ENTROPY_SOURCE_TABLE', 'custom_results')
        monkeypatch.setenv('ACCOUNT_ENTROPY_OUTPUT_TABLE', 'custom_entropy')

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 1800
        assert config.window_days == 14
        assert config.min_posts == 20
        assert config.hourly_entropy_norm_threshold == 0.9
        assert config.interval_entropy_norm_threshold == 0.4
        assert config.cv_threshold == 0.3
        assert config.interval_bin_edges == (30, 120, 600, 3600)
        assert config.source_table == 'custom_results'
        assert config.output_table == 'custom_entropy'

    def test_interval_bin_edges_parsed_from_comma_separated(self, monkeypatch) -> None:
        monkeypatch.setenv('ACCOUNT_ENTROPY_INTERVAL_BIN_EDGES', '100,200,300')

        config = AnalysisConfig.from_env()

        assert config.interval_bin_edges == (100, 200, 300)

    def test_empty_bin_edges_string_produces_empty_tuple(self, monkeypatch) -> None:
        monkeypatch.setenv('ACCOUNT_ENTROPY_INTERVAL_BIN_EDGES', '')

        config = AnalysisConfig.from_env()

        assert config.interval_bin_edges == ()

    def test_bin_edges_strips_whitespace(self, monkeypatch) -> None:
        monkeypatch.setenv('ACCOUNT_ENTROPY_INTERVAL_BIN_EDGES', ' 100 , 200 , 300 ')

        config = AnalysisConfig.from_env()

        assert config.interval_bin_edges == (100, 200, 300)

    def test_source_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('ACCOUNT_ENTROPY_SOURCE_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_output_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('ACCOUNT_ENTROPY_OUTPUT_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_is_frozen(self, base_analysis_config: AnalysisConfig) -> None:
        with pytest.raises(Exception):
            base_analysis_config.interval_seconds = 1200  # type: ignore

    def test_construct_directly(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            window_days=7,
            min_posts=10,
            hourly_entropy_norm_threshold=0.85,
            interval_entropy_norm_threshold=0.53,
            cv_threshold=0.5,
            interval_bin_edges=(60, 300, 900),
            source_table='osprey_execution_results',
            output_table='account_entropy_results',
        )

        assert config.interval_seconds == 3600
        assert config.interval_bin_edges == (60, 300, 900)


class TestAppConfig:
    def test_from_env_composes_both_configs(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'ch.example.com')
        monkeypatch.setenv('ACCOUNT_ENTROPY_INTERVAL_SECONDS', '1800')

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
        config = AppConfig(
            clickhouse=ch_config,
            analysis=analysis_config,
        )

        assert config.clickhouse.host == 'localhost'
        assert config.analysis.min_posts == 10


class TestTelemetryConfig:
    def test_defaults_disabled(self, monkeypatch) -> None:
        from account_entropy.config import TelemetryConfig

        for name in ['ENABLED', 'SERVICE_NAME', 'SERVICE_VERSION', 'ENVIRONMENT', 'TRACES_ENABLED', 'METRICS_ENABLED']:
            monkeypatch.delenv('ACCOUNT_ENTROPY_OTEL_' + name, raising=False)
        monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
        config = TelemetryConfig.from_env()
        assert config.enabled is False
        assert config.service_name == 'account-entropy'
        assert config.service_version == '0.1.0'
        assert config.environment == 'local'
        assert config.otlp_endpoint is None
        assert config.traces_enabled is False
        assert config.metrics_enabled is False

    def test_enabled_env_and_endpoint(self, monkeypatch) -> None:
        from account_entropy.config import TelemetryConfig

        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_ENABLED', 'yes')
        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_SERVICE_NAME', 'custom-service')
        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_SERVICE_VERSION', '1.2.3')
        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_ENVIRONMENT', 'prod')
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
        from account_entropy.config import TelemetryConfig

        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_ENABLED', 'true')
        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_TRACES_ENABLED', 'off')
        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_METRICS_ENABLED', 'on')
        config = TelemetryConfig.from_env()
        assert config.enabled is True
        assert config.traces_enabled is False
        assert config.metrics_enabled is True

    @pytest.mark.parametrize('raw', ['1', 'true', 'TRUE', 'yes', 'On'])
    def test_bool_true_forms(self, monkeypatch, raw: str) -> None:
        from account_entropy.config import TelemetryConfig

        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_ENABLED', raw)
        assert TelemetryConfig.from_env().enabled is True

    @pytest.mark.parametrize('raw', ['0', 'false', 'FALSE', 'no', 'Off'])
    def test_bool_false_forms(self, monkeypatch, raw: str) -> None:
        from account_entropy.config import TelemetryConfig

        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_ENABLED', raw)
        assert TelemetryConfig.from_env().enabled is False

    def test_invalid_bool_names_env_var(self, monkeypatch) -> None:
        from account_entropy.config import TelemetryConfig

        monkeypatch.setenv('ACCOUNT_ENTROPY_OTEL_ENABLED', 'maybe')
        with pytest.raises(ValueError, match='ACCOUNT_ENTROPY_OTEL_ENABLED'):
            TelemetryConfig.from_env()

    def test_direct_app_config_defaults_to_disabled_telemetry(self, base_analysis_config) -> None:
        from account_entropy.config import AppConfig, ClickHouseConfig

        config = AppConfig(
            clickhouse=ClickHouseConfig('localhost', 8123, 'default', 'clickhouse', 'default'),
            analysis=base_analysis_config,
        )
        assert config.telemetry.enabled is False

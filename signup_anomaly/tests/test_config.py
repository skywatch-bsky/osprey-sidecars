# pattern: Functional Core
import pytest

from signup_anomaly.config import AnalysisConfig


@pytest.fixture
def base_analysis_config() -> AnalysisConfig:
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


class TestTelemetryConfig:
    def test_defaults_disabled(self, monkeypatch) -> None:
        from signup_anomaly.config import TelemetryConfig

        for name in ['ENABLED', 'SERVICE_NAME', 'SERVICE_VERSION', 'ENVIRONMENT', 'TRACES_ENABLED', 'METRICS_ENABLED']:
            monkeypatch.delenv('SIGNUP_ANOMALY_OTEL_' + name, raising=False)
        monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
        config = TelemetryConfig.from_env()
        assert config.enabled is False
        assert config.service_name == 'signup-anomaly'
        assert config.service_version == '0.1.0'
        assert config.environment == 'local'
        assert config.otlp_endpoint is None
        assert config.traces_enabled is False
        assert config.metrics_enabled is False

    def test_enabled_env_and_endpoint(self, monkeypatch) -> None:
        from signup_anomaly.config import TelemetryConfig

        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_ENABLED', 'yes')
        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_SERVICE_NAME', 'custom-service')
        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_SERVICE_VERSION', '1.2.3')
        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_ENVIRONMENT', 'prod')
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
        from signup_anomaly.config import TelemetryConfig

        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_ENABLED', 'true')
        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_TRACES_ENABLED', 'off')
        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_METRICS_ENABLED', 'on')
        config = TelemetryConfig.from_env()
        assert config.enabled is True
        assert config.traces_enabled is False
        assert config.metrics_enabled is True

    @pytest.mark.parametrize('raw', ['1', 'true', 'TRUE', 'yes', 'On'])
    def test_bool_true_forms(self, monkeypatch, raw: str) -> None:
        from signup_anomaly.config import TelemetryConfig

        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_ENABLED', raw)
        assert TelemetryConfig.from_env().enabled is True

    @pytest.mark.parametrize('raw', ['0', 'false', 'FALSE', 'no', 'Off'])
    def test_bool_false_forms(self, monkeypatch, raw: str) -> None:
        from signup_anomaly.config import TelemetryConfig

        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_ENABLED', raw)
        assert TelemetryConfig.from_env().enabled is False

    def test_invalid_bool_names_env_var(self, monkeypatch) -> None:
        from signup_anomaly.config import TelemetryConfig

        monkeypatch.setenv('SIGNUP_ANOMALY_OTEL_ENABLED', 'maybe')
        with pytest.raises(ValueError, match='SIGNUP_ANOMALY_OTEL_ENABLED'):
            TelemetryConfig.from_env()

    def test_direct_app_config_defaults_to_disabled_telemetry(self, base_analysis_config) -> None:
        from signup_anomaly.config import AppConfig, ClickHouseConfig

        config = AppConfig(
            clickhouse=ClickHouseConfig('localhost', 8123, 'default', 'clickhouse', 'default'),
            analysis=base_analysis_config,
        )
        assert config.telemetry.enabled is False

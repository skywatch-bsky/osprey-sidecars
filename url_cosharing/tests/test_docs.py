# pattern: Imperative Shell tests
from __future__ import annotations

from pathlib import Path

TELEMETRY_ENV_VARS = (
    'URL_COSHARING_OTEL_ENABLED',
    'URL_COSHARING_OTEL_SERVICE_NAME',
    'URL_COSHARING_OTEL_SERVICE_VERSION',
    'URL_COSHARING_OTEL_ENVIRONMENT',
    'URL_COSHARING_OTEL_TRACES_ENABLED',
    'URL_COSHARING_OTEL_METRICS_ENABLED',
    'OTEL_EXPORTER_OTLP_ENDPOINT',
)


def test_readme_documents_all_telemetry_env_vars() -> None:
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    content = readme.read_text()

    for env_var in TELEMETRY_ENV_VARS:
        assert env_var in content

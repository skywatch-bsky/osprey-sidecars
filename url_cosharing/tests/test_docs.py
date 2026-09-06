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


def test_readme_documents_exclusions_env_var() -> None:
    """The exclusions feature must be documented: env var, YAML schema, the
    dot-anchored matching rule, hot reload, and the audit columns."""
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    content = readme.read_text()

    assert 'URL_COSHARING_EXCLUSIONS_FILE' in content
    assert 'excluded_domains' in content
    assert 'excluded_dids' in content
    assert 'evil-klipy.com' in content  # dot-anchored suffix caveat
    assert 'exclusions_hash' in content
    assert 'excluded_shares_suppressed' in content

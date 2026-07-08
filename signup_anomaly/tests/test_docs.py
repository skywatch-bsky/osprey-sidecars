from __future__ import annotations

from pathlib import Path


def test_readme_documents_telemetry_env_vars() -> None:
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    for env_var in [
        'SIGNUP_ANOMALY_OTEL_ENABLED',
        'SIGNUP_ANOMALY_OTEL_SERVICE_NAME',
        'SIGNUP_ANOMALY_OTEL_SERVICE_VERSION',
        'SIGNUP_ANOMALY_OTEL_ENVIRONMENT',
        'SIGNUP_ANOMALY_OTEL_TRACES_ENABLED',
        'SIGNUP_ANOMALY_OTEL_METRICS_ENABLED',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
    ]:
        assert env_var in readme


def test_docs_mention_high_cardinality_guardrails() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / 'README.md').read_text() + '\n' + (root / 'CLAUDE.md').read_text()
    assert 'high-cardinality' in text
    for term in ['pds_host', 'sample_dids']:
        assert term in text

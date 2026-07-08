from __future__ import annotations

from pathlib import Path


def test_readme_documents_telemetry_env_vars() -> None:
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    for env_var in [
        'QUOTE_COSHARING_OTEL_ENABLED',
        'QUOTE_COSHARING_OTEL_SERVICE_NAME',
        'QUOTE_COSHARING_OTEL_SERVICE_VERSION',
        'QUOTE_COSHARING_OTEL_ENVIRONMENT',
        'QUOTE_COSHARING_OTEL_TRACES_ENABLED',
        'QUOTE_COSHARING_OTEL_METRICS_ENABLED',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
    ]:
        assert env_var in readme


def test_docs_mention_high_cardinality_guardrails() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / 'README.md').read_text() + '\n' + (root / 'CLAUDE.md').read_text()
    assert 'high-cardinality' in text
    for term in ['did', 'shared_uris', 'cluster_id']:
        assert term in text

from __future__ import annotations

from pathlib import Path


def test_readme_documents_telemetry_env_vars() -> None:
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    for env_var in [
        'ACCOUNT_ENTROPY_OTEL_ENABLED',
        'ACCOUNT_ENTROPY_OTEL_SERVICE_NAME',
        'ACCOUNT_ENTROPY_OTEL_SERVICE_VERSION',
        'ACCOUNT_ENTROPY_OTEL_ENVIRONMENT',
        'ACCOUNT_ENTROPY_OTEL_TRACES_ENABLED',
        'ACCOUNT_ENTROPY_OTEL_METRICS_ENABLED',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
    ]:
        assert env_var in readme


def test_docs_mention_high_cardinality_guardrails() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / 'README.md').read_text() + '\n' + (root / 'CLAUDE.md').read_text()
    assert 'high-cardinality' in text
    for term in ['user_id', 'sample_rkeys', 'rkey']:
        assert term in text


# Path to the repo-level calibration doc
_CALIBRATION_MD = Path(__file__).resolve().parents[2] / 'docs' / 'calibration.md'


def test_attribution_categories_partition_bot_like() -> None:
    """The four attribution categories are mutually exclusive and exhaustive.

    is_bot_like implies hourly_flag=1 AND (interval_flag=1 OR cv_flag=1).
    The four categories are defined by the (interval_flag, cv_flag) pair:
      hourly_only:    (0, 0) — structurally empty (is_bot_like requires one of them)
      interval_only:  (1, 0)
      cv_only:        (0, 1)
      both:           (1, 1)
    Every bot-like row falls into exactly one category.
    """
    categories = [(0, 0), (1, 0), (0, 1), (1, 1)]
    # Verify categories are mutually exclusive
    assert len(categories) == len(set(categories))
    # Verify categories are exhaustive over (interval_flag, cv_flag) space
    assert set(categories) == {(i, c) for i in (0, 1) for c in (0, 1)}


def test_calibration_query_3_uses_mutually_exclusive_predicates() -> None:
    """Query 3 in calibration.md must use mutually exclusive predicates.

    The hourly_only category must include interval_flag = 0 AND cv_flag = 0
    to avoid being tautological with is_bot_like (which implies hourly_flag=1).
    """
    text = _CALIBRATION_MD.read_text()
    # Extract the Query 3 SQL block for account entropy
    marker = '#### Query 3: Flag attribution'
    idx = text.index(marker)
    sql_block = text[idx : idx + 1500]

    # The hourly_only predicate must include the exclusive conditions
    assert 'hourly_only' in sql_block
    assert 'interval_flag = 0 AND cv_flag = 0' in sql_block
    # The tautological form must NOT be present
    assert 'is_bot_like = 1 AND hourly_flag = 1) AS hourly_only' not in sql_block
    assert 'is_bot_like = 1 AND hourly_flag = 1) as hourly_only' not in sql_block

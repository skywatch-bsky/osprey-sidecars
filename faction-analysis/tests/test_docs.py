import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_env_vars() -> None:
    config_source = (ROOT / 'src' / 'faction_analysis' / 'config.py').read_text()
    readme = (ROOT / 'README.md').read_text()
    names = sorted(set(re.findall(r'FACTION_ANALYSIS_[A-Z0-9_]+', config_source)))
    assert names, 'config.py should read at least one FACTION_ANALYSIS_* variable'
    missing = [name for name in names if f'`{name}`' not in readme]
    assert missing == [], f'undocumented variables: {missing}'
    for env_var in [
        'OSPREY_CLICKHOUSE_HOST',
        'OSPREY_CLICKHOUSE_PORT',
        'OSPREY_CLICKHOUSE_USER',
        'OSPREY_CLICKHOUSE_PASSWORD',
        'OSPREY_CLICKHOUSE_DB',
        'TYPESAFE_API_KEY',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
    ]:
        assert f'`{env_var}`' in readme, f'undocumented variable: {env_var}'


def test_docs_mention_high_cardinality_guardrails() -> None:
    text = (ROOT / 'README.md').read_text() + '\n' + (ROOT / 'CLAUDE.md').read_text()
    for term in ['high-cardinality', 'low-cardinality']:
        assert term in text
    for term in ['did', 'handle', 'faction_id', 'uri']:
        assert term in text


def test_docs_carry_verification_date_and_measurement_report() -> None:
    agents = (ROOT / 'AGENTS.md').read_text()
    assert re.search(r'^Last verified: \d{4}-\d{2}-\d{2}$', agents, re.MULTILINE), (
        'AGENTS.md must carry a dated verification line'
    )
    assert (ROOT / 'docs' / 'MEASUREMENT.md').is_file(), 'docs/MEASUREMENT.md must exist'

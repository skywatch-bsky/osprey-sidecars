import re
from pathlib import Path

from faction_analysis import schema

ROOT = Path(__file__).resolve().parents[1]
DDL = ROOT / 'sql' / '001_faction_analysis.sql'


def tables() -> dict[str, tuple[str, ...]]:
    """Table name to its column names, in file order."""
    text = DDL.read_text()
    found = re.findall(
        r'CREATE TABLE IF NOT EXISTS default\.(\w+) \((.*?)\n\) ENGINE = MergeTree\(\)(.*?);',
        text,
        re.DOTALL,
    )
    return {name: tuple(re.findall(r'^    (\w+) ', body, re.MULTILINE)) for name, body, _ in found}


def clauses() -> dict[str, str]:
    text = DDL.read_text()
    found = re.findall(
        r'CREATE TABLE IF NOT EXISTS default\.(\w+) \((.*?)\n\) ENGINE = MergeTree\(\)(.*?);', text, re.DOTALL
    )
    return {name: tail for name, _, tail in found}


def test_every_table_of_the_schema_module_is_defined_with_the_same_columns() -> None:
    defined = tables()
    for noun, columns in schema.COLUMNS.items():
        assert defined[f'faction_analysis_{noun}'] == columns, noun
    assert set(defined) == {f'faction_analysis_{noun}' for noun in schema.COLUMNS}


def test_every_table_except_runs_has_a_ttl() -> None:
    for name, tail in clauses().items():
        if name == 'faction_analysis_runs':
            assert 'TTL' not in tail
        else:
            assert re.search(r'^TTL .* DELETE$', tail, re.MULTILINE), name


def test_image_tables_are_partitioned_by_run_id() -> None:
    for noun in schema.IMAGE_TABLES:
        assert 'PARTITION BY run_id' in clauses()[f'faction_analysis_{noun}']
    for noun in ('runs', 'moves', 'conflict', schema.TEXT_SCORES):
        assert 'PARTITION BY' not in clauses()[f'faction_analysis_{noun}']


def test_no_table_stores_graph_edges() -> None:
    for columns in tables().values():
        for column in columns:
            assert column not in {'src', 'dst', 'edge_weight', 'u', 'v'}


def test_runs_table_order_is_run_id_and_members_order_is_did() -> None:
    assert 'ORDER BY run_id' in clauses()['faction_analysis_runs']
    assert 'ORDER BY did' in clauses()['faction_analysis_members']

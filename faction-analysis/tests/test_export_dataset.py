import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from faction_analysis import queries, train
from faction_analysis.config import AppConfig
from faction_analysis.export_dataset import EXPORT_FIELDS, export, latest_writes
from tests.training_world import HOSTILE_WORD, NOW_MS, FakeJev, FakeTrainingRepo

NOW = datetime.fromtimestamp(NOW_MS / 1000, tz=UTC)
WINDOW = queries.Window(NOW - timedelta(days=60), NOW)
PARENT = 'the parent post ' + HOSTILE_WORD


class ExportRepo:
    def __init__(self, writes: pd.DataFrame, parents_for: int) -> None:
        self.writes = writes
        self.parents_for = parents_for

    def fetch_dataset(self, dataset_version: str) -> pd.DataFrame:
        return self.writes

    def fetch_dataset_parent_texts(self, dataset_version: str, window: queries.Window) -> pd.DataFrame:
        first = self.writes['uri_hash'].iloc[: self.parents_for]
        return pd.DataFrame({'uri_hash': first, 'parent_text': [PARENT] * len(first)})


def test_export_writes_shards_manifest_and_schema(tmp_path: Path, fake_world: FakeTrainingRepo) -> None:
    config = AppConfig.from_mapping(
        {
            'FACTION_ANALYSIS_JEV_STRATUM_POSITIVES': '1000',
            'FACTION_ANALYSIS_JEV_STRATUM_RANDOM': '1000',
            'FACTION_ANALYSIS_MAX_WEAK_POSITIVES': '500',
            'FACTION_ANALYSIS_NEGATIVE_SAMPLE': '1000',
        }
    )
    train.refresh(fake_world, config, tmp_path / 'models', now=NOW, factory=lambda: FakeJev())
    written = fake_world.inserted['scorer_dataset'][0]
    # A retried write: the same rows again, with the older write first.
    retried = pd.concat([written, written]).sort_values('written_at', kind='stable').reset_index(drop=True)

    out = tmp_path / 'export'
    manifest = export(ExportRepo(retried, 10), out, str(written['dataset_version'].iloc[0]), WINDOW)
    assert manifest.rows == len(written)
    assert sum(shard['rows'] for shard in manifest.shards) == manifest.rows
    saved = json.loads((out / 'manifest.json').read_text())
    assert saved['rows'] == manifest.rows
    assert saved['fields'] == list(EXPORT_FIELDS)

    lines = [line for shard in manifest.shards for line in (out / shard['file']).read_text().splitlines()]
    rows = [json.loads(line) for line in lines]
    assert len(rows) == manifest.rows
    # A retried write survives exactly once.
    assert len({row['uri_hash'] for row in rows}) == manifest.rows
    assert set(rows[0]) == set(EXPORT_FIELDS) | {'parent_text'}
    for field in rows[0]:
        assert 'did' not in field and 'handle' not in field
    assert sum(row['parent_text'] == PARENT for row in rows) == 10
    assert sum(row['parent_text'] == '' for row in rows) == manifest.rows - 10
    assert all(isinstance(row['author_hash'], int) for row in rows)
    assert all(row['label_h'] is None or 0 <= row['label_h'] <= 1 for row in rows)
    assert all(isinstance(row['languages'], list) for row in rows)
    readme = (out / 'README.md').read_text()
    assert 'may name accounts' in readme and 'No DID and no handle' in readme


def test_shards_hold_at_most_the_shard_size(tmp_path: Path) -> None:
    import faction_analysis.export_dataset as export_module

    rows = 250_000
    writes = pd.DataFrame(
        {
            'uri_hash': range(rows),
            'kind': ['reply'] * rows,
            'split': ['train'] * rows,
            'label_source': ['random'] * rows,
            'target': [0] * rows,
            'weight': [1.0] * rows,
            'label_h': [None] * rows,
            'author_did': [1] * rows,
            'target_did': [2] * rows,
            'text': ['x'] * rows,
            'parent_uri': [''] * rows,
            'event_ms': [0] * rows,
            'languages': [[]] * rows,
            'text_length': [1] * rows,
            'uppercase': [0] * rows,
            'exclamations': [0] * rows,
            'questions': [0] * rows,
            'mentions': [0] * rows,
            'written_at': [0] * rows,
        }
    )
    original = export_module.SHARD_ROWS
    export_module.SHARD_ROWS = 100_000
    try:
        manifest = export(ExportRepo(writes, 0), tmp_path, 'ds', WINDOW)
    finally:
        export_module.SHARD_ROWS = original
    assert [shard['rows'] for shard in manifest.shards] == [100_000, 100_000, 50_000]


def test_an_empty_dataset_exports_no_shards(tmp_path: Path) -> None:
    manifest = export(ExportRepo(pd.DataFrame(), 0), tmp_path, 'ds-empty', WINDOW)
    assert manifest.rows == 0 and manifest.shards == ()
    assert list(tmp_path.glob('*.jsonl')) == []
    assert (tmp_path / 'manifest.json').is_file()


def test_latest_writes_keeps_the_last_write_of_each_row() -> None:
    rows = pd.DataFrame({'uri_hash': [7, 7, 8], 'text': ['first', 'second', 'only'], 'written_at': [1, 2, 3]})
    kept = latest_writes(rows)
    assert kept['uri_hash'].tolist() == [7, 8]
    assert kept['text'].tolist() == ['second', 'only']

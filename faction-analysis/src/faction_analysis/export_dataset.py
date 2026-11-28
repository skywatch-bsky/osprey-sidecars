# pattern: Imperative Shell
"""Export a training dataset as portable JSONL shards.

A transformer is trained later, on a GPU machine, from this export. Each row carries both texts, the kind,
the languages, the labels, the split, the target, the weight and the structural features. The sidecar adds
no identity: accounts are 64-bit hashes. Post text is exported as written, because it is the training data.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from faction_analysis import queries

logger = logging.getLogger(__name__)
SHARD_ROWS = 100_000
MANIFEST_FILE = 'manifest.json'
SCHEMA_FILE = 'README.md'

# The key set of a row. It names no DID and no handle, and a test asserts that.
EXPORT_FIELDS: tuple[str, ...] = (
    'uri_hash',
    'kind',
    'split',
    'label_source',
    'target',
    'weight',
    'label_h',
    'author_hash',
    'target_hash',
    'text',
    'parent_text',
    'event_ms',
    'languages',
    'text_length',
    'uppercase',
    'exclamations',
    'questions',
    'mentions',
)
FIELD_TYPES: dict[str, str] = {
    'uri_hash': 'integer, 64-bit hash of the post AT-URI',
    'kind': '"reply" or "quote"',
    'split': '"train", "calibration" or "test"',
    'label_source': '"jev", "block" or "random"',
    'target': '1 hostile, 0 not hostile',
    'weight': 'training weight',
    'label_h': 'hostility probability from Jev, or null',
    'author_hash': 'integer, 64-bit hash of the author DID',
    'target_hash': 'integer, 64-bit hash of the target DID',
    'text': 'the reply or the quote, as written',
    'parent_text': 'the parent post or the quoted post, or an empty string',
    'event_ms': 'epoch milliseconds of the interaction',
    'languages': 'list of language codes of the text',
    'text_length': 'characters in the text',
    'uppercase': 'ASCII upper-case characters in the text',
    'exclamations': 'exclamation marks in the text',
    'questions': 'question marks in the text',
    'mentions': 'at signs in the text',
}


@dataclass(frozen=True)
class ExportManifest:
    dataset_version: str
    rows: int
    shards: tuple[dict[str, Any], ...]
    fields: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            'dataset_version': self.dataset_version,
            'rows': self.rows,
            'shards': list(self.shards),
            'fields': list(self.fields),
        }


class ExportRepository(Protocol):
    def fetch_dataset(self, dataset_version: str) -> pd.DataFrame: ...

    def fetch_dataset_parent_texts(self, dataset_version: str, window: queries.Window) -> pd.DataFrame: ...


def latest_writes(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per URI hash: the last write, because the rows arrive oldest first."""
    return rows.drop_duplicates('uri_hash', keep='last').reset_index(drop=True)


def export(repo: ExportRepository, out_dir: Path, dataset_version: str, window: queries.Window) -> ExportManifest:
    """Write the shards, the manifest and a schema README. Returns the manifest."""
    writes = repo.fetch_dataset(dataset_version)
    rows = latest_writes(writes) if len(writes) else writes
    parents: dict[int, str] = {}
    if len(rows):
        parent_rows = repo.fetch_dataset_parent_texts(dataset_version, window)
        for uri_hash, text in zip(parent_rows['uri_hash'], parent_rows['parent_text'], strict=True):
            parents[int(uri_hash)] = '' if text is None else str(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    for number, start in enumerate(range(0, len(rows), SHARD_ROWS)):
        name = f'{dataset_version}-{number:04d}.jsonl'
        chunk = rows.iloc[start : start + SHARD_ROWS]
        with open(out_dir / name, 'w') as handle:
            for row in _records(chunk, parents):
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        shards.append({'file': name, 'rows': int(len(chunk))})
    manifest = ExportManifest(dataset_version, int(len(rows)), tuple(shards), EXPORT_FIELDS)
    _write_json(out_dir / MANIFEST_FILE, manifest.as_dict())
    (out_dir / SCHEMA_FILE).write_text(_schema_readme(dataset_version, manifest))
    logger.info('exported dataset', extra={'rows': manifest.rows, 'shards': len(shards)})
    return manifest


HASH_FIELDS = frozenset({'uri_hash', 'author_hash', 'target_hash'})


def _records(chunk: pd.DataFrame, parents: dict[int, str]) -> list[dict[str, Any]]:
    """One plain JSON object per row, in ``EXPORT_FIELDS`` plus ``parent_text``."""
    renamed = chunk.rename(columns={'author_did': 'author_hash', 'target_did': 'target_hash'})
    records: list[dict[str, Any]] = []
    for raw in renamed.to_dict('records'):
        record: dict[str, Any] = {}
        for name in EXPORT_FIELDS:
            value = raw.get(name)
            if isinstance(value, np.generic):
                value = value.item()
            if name in HASH_FIELDS:
                if isinstance(value, bool) or not isinstance(value, (int, float, np.integer)):
                    raise TypeError(f'{name} is not a hash: {value!r}')
                value = int(value)
            elif isinstance(value, float) and np.isnan(value):
                value = None
            elif name == 'languages' and value is None:
                value = []
            record[name] = value
        record['parent_text'] = parents.get(record['uri_hash'], '')
        records.append(record)
    return records


def _schema_readme(dataset_version: str, manifest: ExportManifest) -> str:
    lines = [
        f'# Dataset {dataset_version}',
        '',
        f'{manifest.rows} rows in {len(manifest.shards)} JSONL shards. One JSON object per row.',
        'The text is exported as written. It is public post text and may name accounts.',
        'Accounts are hashes. No DID and no handle is in this dataset.',
        '',
        '| Field | Meaning |',
        '|---|---|',
    ]
    lines += [f'| `{name}` | {FIELD_TYPES[name]} |' for name in EXPORT_FIELDS]
    return '\n'.join(lines) + '\n'


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from datetime import UTC, datetime, timedelta

    from faction_analysis.config import AppConfig
    from faction_analysis.db import FactionAnalysisDb

    parser = argparse.ArgumentParser(description='Export a training dataset as JSONL shards')
    parser.add_argument('--dataset-version', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    config = AppConfig.from_env()
    now = datetime.now(UTC)
    window = queries.Window(now - timedelta(days=config.graph.window_days), now)
    db = FactionAnalysisDb(config.clickhouse, config.query)
    try:
        manifest = export(db, args.out_dir, args.dataset_version, window)
    finally:
        db.close()
    print(json.dumps(manifest.as_dict(), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

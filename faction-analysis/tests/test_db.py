from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from faction_analysis import db as db_module
from faction_analysis import queries, schema
from faction_analysis.config import AppConfig
from faction_analysis.db import FactionAnalysisDb
from faction_analysis.graph import DeletedRecords
from faction_analysis.queries import Window
from faction_analysis.runstate import RunRow

WINDOW = Window(datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 3, tzinfo=UTC))
RUN_ID = '1764270000123.a1b2c3d4'


class FakeClient:
    """Records every call. Returns canned data shaped like the real client's."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []
        self.blocks: list[np.ndarray] = []
        self.rows: list[tuple[object, ...]] = []
        self.frame = pd.DataFrame()

    def query(
        self, query: str | None = None, parameters: object = None, settings: dict[str, int] | None = None
    ) -> object:
        self.calls.append(('query', query, settings))
        return self

    @property
    def result_rows(self) -> list[tuple[object, ...]]:
        return self.rows

    def query_df(
        self, query: str | None = None, parameters: object = None, settings: dict[str, int] | None = None
    ) -> pd.DataFrame:
        self.calls.append(('query_df', query, settings))
        return self.frame

    def query_np_stream(
        self, query: str | None = None, parameters: object = None, settings: dict[str, int] | None = None
    ) -> object:
        self.calls.append(('query_np_stream', query, settings))
        return self._stream(self.blocks)

    def query_df_stream(
        self, query: str | None = None, parameters: object = None, settings: dict[str, int] | None = None
    ) -> object:
        self.calls.append(('query_df_stream', query, settings))
        return self._stream([self.frame])

    @contextmanager
    def _stream(self, blocks: list) -> Generator[list]:
        yield blocks

    def insert(
        self,
        table: str | None = None,
        data: object = None,
        column_names: object = None,
        database: str | None = None,
        column_types: object = None,
        column_type_names: object = None,
        column_oriented: bool = False,
        settings: dict[str, int] | None = None,
    ) -> None:
        self.calls.append(('insert', (table, data, column_names), settings))

    def command(
        self, cmd: str, parameters: object = None, data: object = None, settings: dict[str, int] | None = None
    ) -> None:
        self.calls.append(('command', cmd, settings))

    def close(self) -> None:
        self.calls.append(('close', None, None))


def make_db(client: FakeClient | None = None) -> tuple[FactionAnalysisDb, FakeClient]:
    client = client or FakeClient()
    config = AppConfig.from_mapping({'FACTION_ANALYSIS_MAX_THREADS': '2'})
    return FactionAnalysisDb(config.clickhouse, config.query, client=client), client


def np_block(**columns: object) -> np.ndarray:
    """A block as clickhouse-connect hands it over.

    Columns of one NumPy type come as a plain two-dimensional array. Columns of several types come as an
    array with one field per column. The repository must read both, so the fake produces both.
    """
    arrays = [np.asarray(values) for values in columns.values()]
    if len({array.dtype for array in arrays}) == 1:
        return np.column_stack(arrays)
    block = np.empty(len(arrays[0]), dtype=[(name, array.dtype) for name, array in zip(columns, arrays, strict=True)])
    for name, array in zip(columns, arrays, strict=True):
        block[name] = array
    return block


def test_every_call_carries_the_query_settings() -> None:
    db, client = make_db()
    shapes = [
        (np_block(src=[1, 2], dst=[3, 4], uri_hash=[5, 6], event_ms=[7, 8]), db.stream_follow_creates),
        (np_block(src=[1, 2], dst=[3, 4]), db.stream_repost_creates),
    ]
    for block, stream in shapes:
        client.blocks = [block]
        list(stream(WINDOW))
    client.blocks = [np_block(uri_hash=[9], event_ms=[100])]
    db.fetch_deletes(WINDOW, queries.FOLLOW_COLLECTION)
    db.fetch_account_hashes(WINDOW)
    list(db.stream_interactions(WINDOW))
    db.fetch_daily_row_counts(WINDOW)
    db.fetch_run_states()
    client.blocks = [np_block(uri_hash=[9], h=[0.5])]
    db.fetch_text_scores('v1')
    client.blocks = [
        np_block(uri_hash=[9], kind=[0], event_ms=[100], author=[1], target=[2]),
        np_block(uri_hash=[9], event_ms=[100], blocker=[1], blocked=[2]),
    ]
    assert client.calls
    for _, _, settings in client.calls:
        assert settings == {
            'max_threads': 2,
            'max_execution_time': 3600,
            'max_memory_usage': 20_000_000_000,
            'max_query_size': db_module.MAX_QUERY_SIZE,
        }


def test_repository_settings_allow_the_largest_literal_query() -> None:
    repository, client = make_db()
    repository.fetch_account_hashes(WINDOW)

    settings = client.calls[-1][2]
    assert isinstance(settings, dict)
    assert settings['max_query_size'] == db_module.MAX_QUERY_SIZE
    assert db_module.MAX_QUERY_SIZE > db_module.HASH_CHUNK * 32
    assert db_module.MAX_QUERY_SIZE > db_module.URI_CHUNK * 4104


def test_stream_follow_creates_yields_typed_blocks() -> None:
    db, client = make_db()
    client.blocks = [np_block(src=[1, 2], dst=[3, 4], uri_hash=[5, 6], event_ms=[7, 8])]
    blocks = list(db.stream_follow_creates(WINDOW))
    assert len(blocks) == 1
    assert blocks[0]['src'].dtype == np.uint64
    assert blocks[0]['event_ms'].dtype == np.int64
    assert blocks[0]['src'].tolist() == [1, 2]
    assert client.calls[0][1] == queries.follow_creates_sql('osprey_execution_results', WINDOW)


def test_a_block_of_one_type_is_read_by_position() -> None:
    """Two UInt64 columns arrive as a plain array, without column names to index by."""
    db, client = make_db()
    assert np_block(src=[1, 2], dst=[3, 4]).dtype.names is None
    client.blocks = [np_block(src=[1, 2], dst=[3, 4])]
    blocks = list(db.stream_repost_creates(WINDOW))
    assert blocks[0]['src'].tolist() == [1, 2] and blocks[0]['dst'].tolist() == [3, 4]
    assert blocks[0]['src'].dtype == np.uint64


def test_fetch_deletes_keeps_the_latest_delete_per_uri() -> None:
    db, client = make_db()
    client.blocks = [np_block(uri_hash=[9, 9], event_ms=[100, 300]), np_block(uri_hash=[9, 4], event_ms=[200, 5])]
    deleted = db.fetch_deletes(WINDOW, queries.BLOCK_COLLECTION)
    assert isinstance(deleted, DeletedRecords)
    assert deleted.uri_hash.tolist() == [4, 9]
    assert deleted.deleted_ms.tolist() == [5, 300]


def test_fetch_run_states_resolves_each_run_from_its_rows() -> None:
    db, client = make_db()
    client.rows = [
        ('1764270000123.a1b2c3d4', 1, 0, 0, 0, datetime(2026, 11, 27, tzinfo=UTC)),
        ('1764270000124.b1b2c3d4', 0, 1, 0, 0, None),
    ]
    states = db.fetch_run_states()
    assert states['1764270000123.a1b2c3d4'].name == 'promoted'
    assert states['1764270000124.b1b2c3d4'].name == 'unresolved'


def test_latest_promoted_run_is_none_without_rows() -> None:
    db, client = make_db()
    client.rows = []
    assert db.latest_promoted_run() is None
    assert db.latest_promoted_run('full') is None
    assert "AND kind = 'full'" in str(client.calls[-1][1])


def test_insert_frame_requires_the_table_columns_in_order() -> None:
    db, client = make_db()
    with pytest.raises(ValueError, match='members columns must be'):
        db.insert_frame(schema.MEMBERS, pd.DataFrame({'did': ['a'], 'run_id': ['r']}))
    assert not client.calls


def test_cleanup_refuses_tables_that_are_not_run_owned_and_bad_run_ids() -> None:
    db, client = make_db()
    with pytest.raises(ValueError, match='not partitioned'):
        db.drop_partition(schema.MOVES, RUN_ID)
    with pytest.raises(ValueError, match='not a run-owned'):
        db.delete_rows(schema.TEXT_SCORES, RUN_ID)
    with pytest.raises(ValueError, match='invalid run_id'):
        db.delete_rows(schema.MOVES, "x' OR '1")
    assert not client.calls


def test_cleanup_drops_the_image_partition_and_deletes_the_other_rows() -> None:
    db, client = make_db()
    db.drop_partition(schema.MEMBERS, RUN_ID)
    db.delete_rows(schema.CONFLICT, RUN_ID)
    assert client.calls[0][1] == f"ALTER TABLE faction_analysis_members DROP PARTITION '{RUN_ID}'"
    # A lightweight delete, not the mutation ``ALTER TABLE ... DELETE``: it must not rewrite the parts.
    assert client.calls[1][1] == f"DELETE FROM faction_analysis_conflict WHERE run_id = '{RUN_ID}'"


def test_insert_run_row_uses_the_runs_column_order(run_row: RunRow) -> None:
    db, client = make_db()
    db.insert_run_row(run_row)
    _, payload, _ = client.calls[0]
    assert isinstance(payload, tuple)
    table, data, column_names = payload
    assert table == 'faction_analysis_runs'
    assert tuple(column_names) == schema.RUNS_COLUMNS
    assert data[0][0] == run_row.run_id

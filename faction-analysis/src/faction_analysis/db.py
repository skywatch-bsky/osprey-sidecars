# pattern: Imperative Shell
"""ClickHouse access: source reads, output writes and cleanup of failed runs.

Every statement carries the thread, time and memory settings from config. Large pulls stream one statement
for the whole window, because the source is not ordered by time: a query for one day reads a whole monthly
partition, so per-day queries would multiply the load on the production host.
"""

from collections.abc import Iterator, Sequence
from datetime import date
from typing import Any, Protocol

import clickhouse_connect
import numpy as np
import pandas as pd

from faction_analysis import queries, schema
from faction_analysis.config import ClickHouseConfig, QueryConfig
from faction_analysis.graph import DeletedRecords, fold_deletes
from faction_analysis.labels import LabelInput
from faction_analysis.queries import Window
from faction_analysis.runstate import (
    RunRow,
    RunState,
    is_valid_run_id,
    run_row_from_columns,
    run_row_to_columns,
)


class ClickHouseClient(Protocol):
    """The part of the clickhouse-connect client this sidecar uses, in the client's own parameter order."""

    def query(
        self, query: str | None = None, parameters: Any = None, settings: dict[str, Any] | None = None
    ) -> Any: ...

    def query_df(
        self, query: str | None = None, parameters: Any = None, settings: dict[str, Any] | None = None
    ) -> pd.DataFrame: ...

    def query_np_stream(
        self, query: str | None = None, parameters: Any = None, settings: dict[str, Any] | None = None
    ) -> Any: ...

    def query_df_stream(
        self, query: str | None = None, parameters: Any = None, settings: dict[str, Any] | None = None
    ) -> Any: ...

    def insert(
        self,
        table: str | None = None,
        data: Any = None,
        column_names: Any = None,
        database: str | None = None,
        column_types: Any = None,
        column_type_names: Any = None,
        column_oriented: bool = False,
        settings: dict[str, Any] | None = None,
    ) -> Any: ...

    def command(
        self,
        cmd: str,
        parameters: Any = None,
        data: Any = None,
        settings: dict[str, Any] | None = None,
    ) -> Any: ...

    def close(self) -> None: ...


class FactionAnalysisDb:
    def __init__(
        self, clickhouse: ClickHouseConfig, query: QueryConfig, *, client: ClickHouseClient | None = None
    ) -> None:
        self._clickhouse = clickhouse
        self._query = query
        if client is None:
            client = clickhouse_connect.get_client(
                host=clickhouse.host,
                port=clickhouse.port,
                username=clickhouse.user,
                password=clickhouse.password,
                database=clickhouse.database,
            )
        self._client: ClickHouseClient = client

    def _settings(self) -> dict[str, int]:
        return {**self._query.settings, 'max_query_size': MAX_QUERY_SIZE}

    def _table(self, noun: str) -> str:
        return self._query.table(noun)

    # ---- source reads

    def stream_follow_creates(self, window: Window) -> Iterator[dict[str, np.ndarray]]:
        """Blocks of ``src``, ``dst``, ``uri_hash`` (UInt64) and ``event_ms`` (Int64). One row per follow create."""
        sql = queries.follow_creates_sql(self._query.source_table, window)
        with self._client.query_np_stream(sql, settings=self._settings()) as stream:
            for block in stream:
                src, dst, uri_hash, event_ms = _columns(block)
                yield {
                    'src': np.asarray(src, dtype=np.uint64),
                    'dst': np.asarray(dst, dtype=np.uint64),
                    'uri_hash': np.asarray(uri_hash, dtype=np.uint64),
                    'event_ms': np.asarray(event_ms, dtype=np.int64),
                }

    def stream_repost_creates(self, window: Window) -> Iterator[dict[str, np.ndarray]]:
        """Blocks of ``src`` and ``dst`` (UInt64). One row per repost create."""
        sql = queries.repost_creates_sql(self._query.source_table, window)
        with self._client.query_np_stream(sql, settings=self._settings()) as stream:
            for block in stream:
                src, dst = _columns(block)
                yield {
                    'src': np.asarray(src, dtype=np.uint64),
                    'dst': np.asarray(dst, dtype=np.uint64),
                }

    def fetch_deletes(self, window: Window, collection: str) -> DeletedRecords:
        """Latest delete time per record URI hash, for the follow or the block collection."""
        sql = queries.deletes_sql(self._query.source_table, window, collection)
        hashes: list[np.ndarray] = []
        times: list[np.ndarray] = []
        with self._client.query_np_stream(sql, settings=self._settings()) as stream:
            for block in stream:
                uri_hash, event_ms = _columns(block)
                hashes.append(np.asarray(uri_hash, dtype=np.uint64))
                times.append(np.asarray(event_ms, dtype=np.int64))
        return fold_deletes(
            np.concatenate(hashes) if hashes else np.empty(0, dtype=np.uint64),
            np.concatenate(times) if times else np.empty(0, dtype=np.int64),
        )

    def fetch_account_hashes(self, window: Window) -> pd.DataFrame:
        """Columns ``h``, ``did`` and ``dids``: one row per account hash in the window."""
        sql = queries.account_hashes_sql(self._query.source_table, window)
        return self._client.query_df(sql, settings=self._settings())

    def stream_interactions(self, window: Window) -> Iterator[pd.DataFrame]:
        """Blocks of replies, quotes and blocks in ``interactions_sql`` columns."""
        sql = queries.interactions_sql(self._query.source_table, window)
        with self._client.query_df_stream(sql, settings=self._settings()) as stream:
            yield from stream

    def fetch_daily_row_counts(self, window: Window) -> dict[date, int]:
        """Source rows per UTC day. A day without rows is absent."""
        sql = queries.daily_row_counts_sql(self._query.source_table, window)
        frame = self._client.query_df(sql, settings=self._settings())
        if len(frame) == 0:
            return {}
        return {day: int(rows) for day, rows in zip(frame['day'], frame['rows'], strict=True)}

    def fetch_tags(self, run_id: str, window: Window) -> pd.DataFrame:
        sql = queries.faction_tags_sql(self._query.source_table, self._table(schema.MEMBERS), run_id, window)
        return self._client.query_df(sql, settings=self._settings())

    def fetch_languages(self, run_id: str, window: Window) -> pd.DataFrame:
        sql = queries.faction_languages_sql(self._query.source_table, self._table(schema.MEMBERS), run_id, window)
        return self._client.query_df(sql, settings=self._settings())

    def fetch_handles(self, run_id: str, window: Window) -> pd.DataFrame:
        sql = queries.faction_handles_sql(self._query.source_table, self._table(schema.MEMBER_FEATURES), run_id, window)
        return self._client.query_df(sql, settings=self._settings())

    def fetch_dataset_parent_texts(self, dataset_version: str, window: Window) -> pd.DataFrame:
        """Columns uri_hash and parent_text: the parent text of every dataset row that has one."""
        sql = queries.parent_texts_sql(
            self._query.source_table, self._table(schema.SCORER_DATASET), dataset_version, window
        )
        return self._client.query_df(sql, settings=self._settings())

    # ---- runs

    def insert_run_row(self, row: RunRow) -> None:
        self._client.insert(
            self._table(schema.RUNS),
            [run_row_to_columns(row)],
            column_names=list(schema.RUNS_COLUMNS),
            settings=self._settings(),
        )

    def fetch_run_states(self) -> dict[str, RunState]:
        result = self._client.query(queries.run_states_sql(self._table(schema.RUNS)), settings=self._settings())
        states: dict[str, RunState] = {}
        for values in result.result_rows:
            run_id = str(values[0])
            states[run_id] = RunState(
                run_id=run_id,
                promoted=bool(values[1]),
                promoting=bool(values[2]),
                failed=bool(values[3]),
                running=bool(values[4]),
                last_recorded_at=values[5],
            )
        return states

    def fetch_run_rows(self, run_id: str) -> list[RunRow]:
        result = self._client.query(queries.run_rows_sql(self._table(schema.RUNS), run_id), settings=self._settings())
        return [run_row_from_columns(row) for row in result.result_rows]

    def latest_promoted_run(self, kind: str | None = None) -> RunRow | None:
        result = self._client.query(
            queries.latest_promoted_sql(self._table(schema.RUNS), kind), settings=self._settings()
        )
        if not result.result_rows:
            return None
        return run_row_from_columns(result.result_rows[0])

    def fetch_image(self, run_id: str) -> pd.DataFrame:
        """The members image of one run: ``did``, ``faction_id``, ``since_run_id``, ``prev_faction_id``, ``confidence``."""
        return self._client.query_df(queries.image_sql(self._table(schema.MEMBERS), run_id), settings=self._settings())

    # ---- writes

    def insert_frame(self, noun: str, frame: pd.DataFrame) -> None:
        """Insert a frame whose columns are exactly the table's columns, in order."""
        expected = schema.COLUMNS[noun]
        if tuple(frame.columns) != expected:
            raise ValueError(f'{noun} columns must be {expected}: {tuple(frame.columns)}')
        if not len(frame):
            return
        self._client.insert(self._table(noun), frame, column_names=list(expected), settings=self._settings())

    # ---- caches

    def fetch_text_scores(self, model_version: str) -> tuple[np.ndarray, np.ndarray]:
        """Sorted ``uri_hash`` and the latest ``h`` for a model version. Each hash appears once."""
        sql = queries.text_scores_for_model_sql(self._table(schema.TEXT_SCORES), model_version)
        with self._client.query_np_stream(sql, settings=self._settings()) as stream:
            hashes: list[np.ndarray] = []
            scores: list[np.ndarray] = []
            for block in stream:
                uri_hash, h = _columns(block)
                hashes.append(np.asarray(uri_hash, dtype=np.uint64))
                scores.append(np.asarray(h, dtype=np.float64))
        return (
            np.concatenate(hashes) if hashes else np.empty(0, dtype=np.uint64),
            np.concatenate(scores) if scores else np.empty(0, dtype=np.float64),
        )

    def fetch_jev_labels(self, prompt_version: str) -> pd.DataFrame:
        sql = queries.jev_labels_sql(self._table(schema.JEV_LABELS), prompt_version)
        return self._client.query_df(sql, settings=self._settings())

    # ---- cleanup of failed runs

    def drop_partition(self, noun: str, run_id: str) -> None:
        """Remove the image partition of ``run_id`` from a table partitioned by ``run_id``."""
        if noun not in schema.IMAGE_TABLES:
            raise ValueError(f'{noun} is not partitioned by run_id')
        self._client.command(
            f'ALTER TABLE {queries.identifier(self._table(noun))} DROP PARTITION {queries.quote_literal(_run_id(run_id))}',
            settings=self._settings(),
        )

    def delete_rows(self, noun: str, run_id: str) -> None:
        """Remove every row of ``run_id`` from a run-owned table with a lightweight delete.

        ``DELETE FROM`` marks the rows at once and rewrites the parts in the background. ``ALTER TABLE ...
        DELETE``, which looks the same, is a mutation that rewrites the parts before it returns.
        """
        if noun not in schema.RUN_OWNED:
            raise ValueError(f'{noun} is not a run-owned table')
        self._client.command(
            f'DELETE FROM {queries.identifier(self._table(noun))} WHERE run_id = {queries.quote_literal(_run_id(run_id))}',
            settings=self._settings(),
        )

    # ---- classifier inputs

    def fetch_label_inputs(self, window: Window) -> LabelInput:
        """Every reply and quote of the window with hashed accounts, and every block create.

        Block creates that a later delete removed are dropped here, as the graph does for follows.
        """
        table = self._query.source_table
        with self._client.query_np_stream(
            queries.label_interactions_sql(table, window), settings=self._settings()
        ) as stream:
            parts = [_records(block, LABEL_DTYPE) for block in stream]
        interactions = np.concatenate(parts) if parts else np.empty(0, dtype=LABEL_DTYPE)
        with self._client.query_np_stream(queries.block_events_sql(table, window), settings=self._settings()) as stream:
            parts = [_records(block, BLOCK_DTYPE) for block in stream]
        blocks = np.concatenate(parts) if parts else np.empty(0, dtype=BLOCK_DTYPE)
        deleted = self.fetch_deletes(window, queries.BLOCK_COLLECTION)
        uri_hash = np.asarray(blocks['uri_hash'], dtype=np.uint64)
        live = np.ones(len(blocks), dtype=bool)
        if len(deleted.uri_hash) and len(blocks):
            position = np.searchsorted(deleted.uri_hash, uri_hash)
            clipped = np.minimum(position, len(deleted.uri_hash) - 1)
            live = ~((deleted.uri_hash[clipped] == uri_hash) & (deleted.deleted_ms[clipped] > blocks['event_ms']))
        blocks = blocks[live]
        return LabelInput(
            uri_hash=np.asarray(interactions['uri_hash'], dtype=np.uint64),
            kind=np.asarray(interactions['kind'], dtype=np.int8),
            event_ms=np.asarray(interactions['event_ms'], dtype=np.int64),
            author=np.asarray(interactions['author'], dtype=np.uint64),
            target=np.asarray(interactions['target'], dtype=np.uint64),
            block_ms=np.asarray(blocks['event_ms'], dtype=np.int64),
            block_blocker=np.asarray(blocks['blocker'], dtype=np.uint64),
            block_blocked=np.asarray(blocks['blocked'], dtype=np.uint64),
        )

    def fetch_interaction_texts(self, window: Window, uri_hashes: Sequence[int]) -> pd.DataFrame:
        """The interaction rows of ``interactions_sql`` for the given URI hashes, in chunks."""
        table = self._query.source_table
        frames = [
            self._client.query_df(
                queries.texts_for_hashes_sql(table, window, uri_hashes[start : start + HASH_CHUNK]),
                settings=self._settings(),
            )
            for start in range(0, len(uri_hashes), HASH_CHUNK)
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_parent_texts(self, window: Window, parent_uris: Sequence[str]) -> dict[str, str]:
        """The text of each parent post, for the URIs that exist."""
        table = self._query.source_table
        texts: dict[str, str] = {}
        for start in range(0, len(parent_uris), URI_CHUNK):
            frame = self._client.query_df(
                queries.parent_texts_for_uris_sql(table, window, parent_uris[start : start + URI_CHUNK]),
                settings=self._settings(),
            )
            for uri, text in zip(frame['parent_uri'], frame['parent_text'], strict=True):
                texts[str(uri)] = '' if text is None else str(text)
        return texts

    def fetch_dataset(self, dataset_version: str) -> pd.DataFrame:
        """Every write of a dataset row. The caller keeps the last write of each URI hash."""
        return self._client.query_df(
            queries.dataset_rows_sql(self._table(schema.SCORER_DATASET), dataset_version),
            settings=self._settings(),
        )

    def fetch_faction_registry(self) -> pd.DataFrame:
        """Columns faction_id, status, parent_faction_id and born_run_id: the current state of each faction."""
        return self._client.query_df(
            queries.faction_registry_sql(self._table(schema.FACTIONS), self._table(schema.RUNS)),
            settings=self._settings(),
        )

    def close(self) -> None:
        self._client.close()


LABEL_DTYPE = np.dtype([('uri_hash', '<u8'), ('kind', 'i1'), ('event_ms', '<i8'), ('author', '<u8'), ('target', '<u8')])
BLOCK_DTYPE = np.dtype([('uri_hash', '<u8'), ('event_ms', '<i8'), ('blocker', '<u8'), ('blocked', '<u8')])
HASH_CHUNK = 100_000
URI_CHUNK = 50_000
# 50,000 URIs of 4,096 characters plus quoting need about 205 MB; 100,000 UInt64 hashes
# plus separators need under 4 MB. 256 MiB covers either largest query with margin.
MAX_QUERY_SIZE = 256 * 1024 * 1024


def _columns(block: np.ndarray) -> tuple[np.ndarray, ...]:
    """The columns of a streamed NumPy block, in the order the statement selected them.

    clickhouse-connect hands over a plain two-dimensional array when every column has the same NumPy type,
    and an array with one field per column only when the types differ. Position is right in both.
    """
    if block.dtype.names is not None:
        return tuple(block[name] for name in block.dtype.names)
    return tuple(block[:, index] for index in range(block.shape[1]))


def _records(block: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """A streamed block as rows of ``dtype``, whose fields follow the order of the statement."""
    columns = _columns(block)
    records = np.empty(len(columns[0]) if columns else 0, dtype=dtype)
    for name, column in zip(dtype.names or (), columns, strict=True):
        records[name] = column
    return records


def _run_id(run_id: str) -> str:
    if not is_valid_run_id(run_id):
        raise ValueError(f'invalid run_id: {run_id!r}')
    return run_id

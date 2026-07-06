# pattern: Imperative Shell
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import clickhouse_connect

from account_entropy.config import ClickHouseConfig


@dataclass(frozen=True)
class AccountActivityRow:
    user_id: str
    post_count: int
    hourly_bins: list[int]
    ordered_timestamps: list[int]
    sample_rkeys: list[str]


@dataclass(frozen=True)
class ScoredResult:
    run_timestamp: datetime
    user_id: str
    window_start: datetime
    window_end: datetime
    post_count: int
    hourly_entropy: float
    interval_entropy: float
    mean_interval_seconds: float
    stddev_interval_seconds: float
    is_bot_like: int
    hourly_flag: int
    interval_flag: int
    sample_rkeys: list[str]


class AccountEntropyDb:
    def __init__(self, config: ClickHouseConfig):
        self._client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.user,
            password=config.password,
            database=config.database,
        )

    def fetch_account_rows(self, query: str) -> list[AccountActivityRow]:
        result = self._client.query(
            query,
            settings={'max_execution_time': 120},
        )
        rows = []
        for row in result.result_rows:
            rows.append(
                AccountActivityRow(
                    user_id=row[0],
                    post_count=int(row[1]),
                    hourly_bins=list(row[2]) if row[2] else [],
                    ordered_timestamps=list(row[3]) if row[3] else [],
                    sample_rkeys=[str(k) for k in row[4]] if row[4] else [],
                )
            )
        return rows

    def insert_results(self, table: str, results: Sequence[ScoredResult]) -> None:
        column_names = [
            'run_timestamp',
            'user_id',
            'window_start',
            'window_end',
            'post_count',
            'hourly_entropy',
            'interval_entropy',
            'mean_interval_seconds',
            'stddev_interval_seconds',
            'is_bot_like',
            'hourly_flag',
            'interval_flag',
            'sample_rkeys',
        ]
        data = [
            [
                r.run_timestamp,
                r.user_id,
                r.window_start,
                r.window_end,
                r.post_count,
                r.hourly_entropy,
                r.interval_entropy,
                r.mean_interval_seconds,
                r.stddev_interval_seconds,
                r.is_bot_like,
                r.hourly_flag,
                r.interval_flag,
                r.sample_rkeys,
            ]
            for r in results
        ]
        self._client.insert(table=table, data=data, column_names=column_names)

    def close(self) -> None:
        self._client.close()

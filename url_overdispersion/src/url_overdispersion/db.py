# pattern: Imperative Shell
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import clickhouse_connect

from url_overdispersion.config import ClickHouseConfig


@dataclass(frozen=True)
class AggregatedRow:
    domain: str
    bucket_start: datetime
    total_shares: int
    unique_sharers: int
    sharer_density: float
    rolling_volume_mean: float | None
    rolling_density_mean: float | None
    baseline_days_available: int
    sample_dids: list[str]
    sample_urls: list[str]
    population_volume_median: float | None
    population_density_median: float | None


@dataclass(frozen=True)
class ScoredResult:
    run_timestamp: datetime
    granularity: str
    domain: str
    bucket_start: datetime
    total_shares: int
    unique_sharers: int
    sharer_density: float
    expected_volume_lambda: float
    expected_density_lambda: float
    volume_p_value: float
    density_p_value: float
    is_anomaly: int
    baseline_source: str
    baseline_days_available: int
    sample_dids: list[str]
    sample_urls: list[str]
    on_watchlist: int


def _ensure_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


class UrlOverdispersionDb:
    def __init__(self, config: ClickHouseConfig):
        self._client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.user,
            password=config.password,
            database=config.database,
        )

    def fetch_aggregated_rows(self, query: str) -> list[AggregatedRow]:
        result = self._client.query(
            query,
            settings={'max_execution_time': 120},
        )
        rows = []
        for row in result.result_rows:
            rows.append(
                AggregatedRow(
                    domain=row[0],
                    bucket_start=_ensure_datetime(row[1]),
                    total_shares=int(row[2]),
                    unique_sharers=int(row[3]),
                    sharer_density=float(row[4]),
                    rolling_volume_mean=float(row[5]) if row[5] is not None else None,
                    rolling_density_mean=float(row[6]) if row[6] is not None else None,
                    baseline_days_available=int(row[7]),
                    sample_dids=list(row[8]) if row[8] else [],
                    sample_urls=list(row[9]) if row[9] else [],
                    population_volume_median=float(row[10]) if row[10] is not None else None,
                    population_density_median=float(row[11]) if row[11] is not None else None,
                )
            )
        return rows

    def insert_results(self, table: str, results: Sequence[ScoredResult]) -> None:
        column_names = [
            'run_timestamp',
            'granularity',
            'domain',
            'bucket_start',
            'total_shares',
            'unique_sharers',
            'sharer_density',
            'expected_volume_lambda',
            'expected_density_lambda',
            'volume_p_value',
            'density_p_value',
            'is_anomaly',
            'baseline_source',
            'baseline_days_available',
            'sample_dids',
            'sample_urls',
            'on_watchlist',
        ]
        data = [
            [
                r.run_timestamp,
                r.granularity,
                r.domain,
                r.bucket_start,
                r.total_shares,
                r.unique_sharers,
                r.sharer_density,
                r.expected_volume_lambda,
                r.expected_density_lambda,
                r.volume_p_value,
                r.density_p_value,
                r.is_anomaly,
                r.baseline_source,
                r.baseline_days_available,
                r.sample_dids,
                r.sample_urls,
                r.on_watchlist,
            ]
            for r in results
        ]
        self._client.insert(table=table, data=data, column_names=column_names)

    def close(self) -> None:
        self._client.close()

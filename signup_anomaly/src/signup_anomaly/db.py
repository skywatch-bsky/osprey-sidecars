# pattern: Imperative Shell
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import clickhouse_connect

from signup_anomaly.config import ClickHouseConfig


@dataclass(frozen=True)
class AggregatedRow:
    pds_host: str
    observed_count: int
    distinct_accounts: int
    rolling_mean: float | None
    baseline_days_available: int
    sample_dids: list[str]
    population_median_lambda: float | None
    rolling_variance: float | None
    dispersion_index: float | None
    population_dispersion_index: float | None


@dataclass(frozen=True)
class ScoredResult:
    run_timestamp: datetime
    granularity: str
    pds_host: str
    observed_count: int
    distinct_accounts: int
    expected_lambda: float
    p_value: float
    is_anomaly: int
    baseline_source: str
    baseline_days_available: int
    sample_dids: list[str]
    rolling_mean: float | None
    rolling_variance: float | None
    dispersion_index: float | None


class SignupAnomalyDb:
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
                    pds_host=row[0],
                    observed_count=int(row[1]),
                    distinct_accounts=int(row[2]),
                    rolling_mean=float(row[3]) if row[3] is not None else None,
                    baseline_days_available=int(row[4]),
                    sample_dids=list(row[5]) if row[5] else [],
                    population_median_lambda=float(row[6]) if row[6] is not None else None,
                    rolling_variance=float(row[7]) if row[7] is not None else None,
                    dispersion_index=float(row[8]) if row[8] is not None else None,
                    population_dispersion_index=float(row[9]) if row[9] is not None else None,
                )
            )
        return rows

    def insert_results(self, table: str, results: Sequence[ScoredResult]) -> None:
        column_names = [
            'run_timestamp',
            'granularity',
            'pds_host',
            'observed_count',
            'distinct_accounts',
            'expected_lambda',
            'p_value',
            'is_anomaly',
            'baseline_source',
            'baseline_days_available',
            'sample_dids',
            'rolling_mean',
            'rolling_variance',
            'dispersion_index',
        ]
        data = [
            [
                r.run_timestamp,
                r.granularity,
                r.pds_host,
                r.observed_count,
                r.distinct_accounts,
                r.expected_lambda,
                r.p_value,
                r.is_anomaly,
                r.baseline_source,
                r.baseline_days_available,
                r.sample_dids,
                r.rolling_mean,
                r.rolling_variance,
                r.dispersion_index,
            ]
            for r in results
        ]
        self._client.insert(table=table, data=data, column_names=column_names)

    def close(self) -> None:
        self._client.close()

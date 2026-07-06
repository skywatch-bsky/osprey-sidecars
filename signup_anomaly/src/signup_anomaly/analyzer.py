# pattern: Functional Core
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from signup_anomaly.config import AnalysisConfig
from signup_anomaly.counts import count_p_value
from signup_anomaly.db import AggregatedRow, ScoredResult
from signup_anomaly.fdr import bh_adjust


def determine_baseline(
    row: AggregatedRow,
    cold_start_min_days: int,
) -> tuple[float, str]:
    if row.baseline_days_available >= cold_start_min_days and row.rolling_median is not None and row.rolling_median > 0:
        return row.rolling_median, 'entity'

    if row.population_median_lambda is not None and row.population_median_lambda > 0:
        return row.population_median_lambda, 'population'

    return 0.0, 'population'


def determine_dispersion(
    row: AggregatedRow,
    cold_start_min_days: int,
) -> float | None:
    if row.baseline_days_available >= cold_start_min_days and row.dispersion_index is not None:
        return row.dispersion_index

    if row.population_dispersion_index is not None:
        return row.population_dispersion_index

    return None


def score_row(
    row: AggregatedRow,
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
) -> ScoredResult:
    expected_lambda, baseline_source = determine_baseline(
        row,
        config.cold_start_min_days,
    )

    phi = determine_dispersion(row, config.cold_start_min_days)
    variance = phi * expected_lambda if (phi is not None and phi > 1.0 and expected_lambda > 0) else None
    p_value = count_p_value(row.observed_count, expected_lambda, variance)

    return ScoredResult(
        run_timestamp=run_timestamp,
        granularity=granularity,
        pds_host=row.pds_host,
        observed_count=row.observed_count,
        distinct_accounts=row.distinct_accounts,
        expected_lambda=expected_lambda,
        p_value=p_value,
        q_value=1.0,
        is_anomaly=0,
        baseline_source=baseline_source,
        baseline_days_available=row.baseline_days_available,
        sample_dids=row.sample_dids,
        rolling_mean=row.rolling_mean,
        rolling_variance=row.rolling_variance,
        dispersion_index=phi,
    )


def score_rows(
    rows: list[AggregatedRow],
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
) -> list[ScoredResult]:
    provisional = [score_row(row, config, granularity, run_timestamp) for row in rows]
    q_values = bh_adjust([r.p_value for r in provisional])
    results = []
    for result, q_value in zip(provisional, q_values):
        threshold = (
            config.daily_p_value_threshold
            if granularity == 'daily' or result.baseline_source == 'population'
            else config.hourly_p_value_threshold
        )
        is_anomaly = 1 if (q_value < threshold and result.observed_count > 0) else 0
        results.append(replace(result, q_value=q_value, is_anomaly=is_anomaly))
    return results

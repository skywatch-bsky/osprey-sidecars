# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

from scipy.stats import poisson

from signup_anomaly.config import AnalysisConfig
from signup_anomaly.db import AggregatedRow, ScoredResult


def compute_p_value(observed: int, expected_lambda: float) -> float:
    if expected_lambda <= 0:
        return 1.0
    return float(poisson.sf(observed - 1, expected_lambda))


def determine_baseline(
    row: AggregatedRow,
    cold_start_min_days: int,
) -> tuple[float, str]:
    if row.baseline_days_available >= cold_start_min_days and row.rolling_mean is not None:
        return row.rolling_mean, 'entity'

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
    p_threshold = config.daily_p_value_threshold if granularity == 'daily' else config.hourly_p_value_threshold

    expected_lambda, baseline_source = determine_baseline(
        row,
        config.cold_start_min_days,
    )

    if baseline_source == 'population':
        p_threshold = config.daily_p_value_threshold

    p_value = compute_p_value(row.observed_count, expected_lambda)

    is_anomaly = 1 if (p_value < p_threshold and row.observed_count > 0) else 0

    resolved_dispersion = determine_dispersion(row, config.cold_start_min_days)

    return ScoredResult(
        run_timestamp=run_timestamp,
        granularity=granularity,
        pds_host=row.pds_host,
        observed_count=row.observed_count,
        distinct_accounts=row.distinct_accounts,
        expected_lambda=expected_lambda,
        p_value=p_value,
        is_anomaly=is_anomaly,
        baseline_source=baseline_source,
        baseline_days_available=row.baseline_days_available,
        sample_dids=row.sample_dids,
        rolling_mean=row.rolling_mean,
        rolling_variance=row.rolling_variance,
        dispersion_index=resolved_dispersion,
    )


def score_rows(
    rows: list[AggregatedRow],
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
) -> list[ScoredResult]:
    return [score_row(row, config, granularity, run_timestamp) for row in rows]

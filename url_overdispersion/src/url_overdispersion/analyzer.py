# pattern: Functional Core
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from url_overdispersion.config import AnalysisConfig
from url_overdispersion.counts import count_p_value
from url_overdispersion.db import AggregatedRow, ScoredResult
from url_overdispersion.density import density_p_value
from url_overdispersion.fdr import bh_adjust


def determine_baseline(
    row: AggregatedRow,
    cold_start_min_days: int,
) -> tuple[float, float, str]:
    """
    Select baseline for both volume and density signals.

    Returns (volume_lambda, density_lambda, baseline_source).

    Entity path when baseline_days_available >= cold_start_min_days AND
    rolling_volume_median is not None AND rolling_volume_median > 0 AND
    rolling_density_mean is not None AND rolling_density_mean > 0.
    Otherwise falls back to population path when both population medians are
    present and > 0. Else returns (0.0, 0.0, 'population').
    """
    if (
        row.baseline_days_available >= cold_start_min_days
        and row.rolling_volume_median is not None
        and row.rolling_volume_median > 0
        and row.rolling_density_mean is not None
        and row.rolling_density_mean > 0
    ):
        return row.rolling_volume_median, row.rolling_density_mean, 'entity'

    if (
        row.population_volume_median is not None
        and row.population_volume_median > 0
        and row.population_density_median is not None
        and row.population_density_median > 0
    ):
        return row.population_volume_median, row.population_density_median, 'population'

    return 0.0, 0.0, 'population'


def determine_variances(
    row: AggregatedRow,
    baseline_source: str,
) -> tuple[float | None, float | None]:
    """
    Compute volume and density variances for dispersion-aware tests.

    Returns (volume_variance, density_variance).

    Entity source: phi = rolling_volume_variance / rolling_volume_mean;
    if phi > 1, volume_variance = phi * rolling_volume_median, else None (Poisson fallback).
    density_variance = rolling_density_variance (may be None → binomial fallback).

    Population source: same logic using population_volume_dispersion and
    population_volume_median, population_density_variance.
    """
    if baseline_source == 'entity':
        volume_variance = None
        if (
            row.rolling_volume_mean is not None
            and row.rolling_volume_mean > 0
            and row.rolling_volume_variance is not None
        ):
            phi = row.rolling_volume_variance / row.rolling_volume_mean
            if phi > 1 and row.rolling_volume_median is not None:
                volume_variance = phi * row.rolling_volume_median
        return volume_variance, row.rolling_density_variance
    else:  # population source
        volume_variance = None
        if (
            row.population_volume_dispersion is not None
            and row.population_volume_median is not None
            and row.population_volume_median > 0
        ):
            phi = row.population_volume_dispersion
            if phi > 1:
                volume_variance = phi * row.population_volume_median
        return volume_variance, row.population_density_variance


def score_row(
    row: AggregatedRow,
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
    on_watchlist: int,
) -> ScoredResult:
    """
    Score one row, computing both p-values, returning provisional result with q=1.0, is_anomaly=0.

    Q-values and is_anomaly flag are determined in score_rows via two-pass BH adjustment.
    """
    expected_volume_lambda, expected_density_lambda, baseline_source = determine_baseline(
        row,
        config.cold_start_min_days,
    )

    volume_variance, density_variance = determine_variances(row, baseline_source)

    volume_p_value = count_p_value(row.total_shares, expected_volume_lambda, volume_variance)
    density_p_value_ = density_p_value(
        row.unique_sharers,
        row.total_shares,
        expected_density_lambda,
        density_variance,
    )

    return ScoredResult(
        run_timestamp=run_timestamp,
        granularity=granularity,
        domain=row.domain,
        bucket_start=row.bucket_start,
        total_shares=row.total_shares,
        unique_sharers=row.unique_sharers,
        sharer_density=row.sharer_density,
        expected_volume_lambda=expected_volume_lambda,
        expected_density_lambda=expected_density_lambda,
        rolling_volume_median=row.rolling_volume_median,
        rolling_volume_variance=row.rolling_volume_variance,
        rolling_density_mean=row.rolling_density_mean,
        rolling_density_variance=row.rolling_density_variance,
        volume_p_value=volume_p_value,
        volume_q_value=1.0,
        density_p_value=density_p_value_,
        density_q_value=1.0,
        is_anomaly=0,
        baseline_source=baseline_source,
        baseline_days_available=row.baseline_days_available,
        sample_dids=row.sample_dids,
        sample_urls=row.sample_urls,
        on_watchlist=on_watchlist,
    )


def score_rows(
    rows: list[AggregatedRow],
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
    watchlist_domains: tuple[str, ...],
) -> list[ScoredResult]:
    """
    Two-pass scoring with per-signal Benjamini–Hochberg FDR adjustment.

    Pass 1: score_row each row, computing raw p-values, provisional q=1.0, is_anomaly=0.
    Pass 2: adjust volume and density p-values as separate families via bh_adjust,
    then set q-values and is_anomaly = 1 iff (volume_q < threshold OR density_q < threshold).
    """
    provisional = [
        score_row(
            row,
            config,
            granularity,
            run_timestamp,
            on_watchlist=1 if row.domain in watchlist_domains else 0,
        )
        for row in rows
    ]

    volume_q = bh_adjust([r.volume_p_value for r in provisional])
    density_q = bh_adjust([r.density_p_value for r in provisional])

    results = []
    for result, vq, dq in zip(provisional, volume_q, density_q):
        is_anomaly = 1 if (vq < config.volume_p_threshold or dq < config.density_p_threshold) else 0
        results.append(replace(result, volume_q_value=vq, density_q_value=dq, is_anomaly=is_anomaly))
    return results

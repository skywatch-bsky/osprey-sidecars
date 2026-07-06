# pattern: Functional Core
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from quote_overdispersion.config import AnalysisConfig
from quote_overdispersion.counts import count_p_value
from quote_overdispersion.db import AggregatedRow, ScoredResult
from quote_overdispersion.density import density_p_value as compute_density_p_value
from quote_overdispersion.fdr import bh_adjust


def extract_quoted_author_did(quoted_uri: str) -> str:
    """Parse AT-URI format to extract the author DID.

    AT-URI format: at://did:plc:xxx/app.bsky.feed.post/yyy
    Returns the DID (e.g., "did:plc:xxx") or empty string if parsing fails.

    Edge cases handled:
    - Empty input → empty string
    - Missing "at://" prefix → empty string
    - Malformed URI → empty string
    - Any parsing error → empty string
    """
    if not quoted_uri:
        return ''

    try:
        if not quoted_uri.startswith('at://'):
            return ''

        remainder = quoted_uri[5:]
        parts = remainder.split('/')
        if not parts or not parts[0]:
            return ''

        return parts[0]
    except Exception:
        return ''


def determine_baseline(
    row: AggregatedRow,
    cold_start_min_days: int,
) -> tuple[float, float, str]:
    """Select baseline for both volume and density signals.

    Returns (volume_lambda, density_lambda, baseline_source).

    Uses rolling_volume_median and rolling_density_mean when entity baseline is available.
    Falls back to population medians when entity baseline is insufficient.
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
    """Derive volume and density variance estimates from row data.

    Returns (volume_variance, density_variance) for use in dispersion-aware tests.

    Entity path: volume_variance = rolling_volume_variance (already MoM-fit);
                 density_variance = rolling_density_variance (already scaled).
    Population path: volume_variance = population_volume_dispersion * population_volume_median;
                     density_variance = population_density_variance.
    """
    if baseline_source == 'entity':
        return row.rolling_volume_variance, row.rolling_density_variance

    if baseline_source == 'population':
        volume_variance = None
        if row.population_volume_dispersion is not None and row.population_volume_median is not None:
            volume_variance = row.population_volume_dispersion * row.population_volume_median

        return volume_variance, row.population_density_variance

    return None, None


def score_row(
    row: AggregatedRow,
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
) -> ScoredResult:
    """Score one row with provisional q-values and diagnostic columns.

    Computes volume and density p-values using dispersion-aware tests.
    q-values are set to 1.0 (provisional) — actual BH adjustment happens in score_rows.
    is_anomaly is set to 0 (provisional) — actual decision happens after BH adjustment.
    """
    expected_volume_lambda, expected_density_lambda, baseline_source = determine_baseline(
        row,
        config.cold_start_min_days,
    )

    volume_variance, density_variance = determine_variances(row, baseline_source)

    volume_p_value = count_p_value(row.total_shares, expected_volume_lambda, volume_variance)
    density_p_value_result = compute_density_p_value(
        row.unique_sharers,
        row.total_shares,
        expected_density_lambda,
        density_variance,
    )

    quoted_author_did = extract_quoted_author_did(row.quoted_uri)

    return ScoredResult(
        run_timestamp=run_timestamp,
        granularity=granularity,
        quoted_uri=row.quoted_uri,
        quoted_author_did=quoted_author_did,
        bucket_start=row.bucket_start,
        total_shares=row.total_shares,
        unique_sharers=row.unique_sharers,
        sharer_density=row.sharer_density,
        expected_volume_lambda=expected_volume_lambda,
        expected_density_lambda=expected_density_lambda,
        volume_p_value=volume_p_value,
        volume_q_value=1.0,  # Provisional: set by score_rows after BH adjustment
        density_p_value=density_p_value_result,
        density_q_value=1.0,  # Provisional: set by score_rows after BH adjustment
        is_anomaly=0,  # Provisional: set by score_rows after BH adjustment
        baseline_source=baseline_source,
        baseline_days_available=row.baseline_days_available,
        sample_dids=row.sample_dids,
        rolling_volume_median=row.rolling_volume_median,
        rolling_volume_variance=row.rolling_volume_variance,
        rolling_density_mean=row.rolling_density_mean,
        rolling_density_variance=row.rolling_density_variance,
    )


def score_rows(
    rows: list[AggregatedRow],
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
) -> list[ScoredResult]:
    """Score all rows with two-pass BH-FDR: separate families for volume and density.

    Pass 1: Compute all provisional p-values.
    Pass 2: Apply BH adjustment per signal (volume and density as separate families),
            then set is_anomaly = 1 iff (volume_q < threshold OR density_q < threshold).
    """
    provisional = [score_row(row, config, granularity, run_timestamp) for row in rows]

    volume_q = bh_adjust([r.volume_p_value for r in provisional])
    density_q = bh_adjust([r.density_p_value for r in provisional])

    results = []
    for result, vq, dq in zip(provisional, volume_q, density_q):
        is_anomaly = 1 if (vq < config.volume_p_threshold or dq < config.density_p_threshold) else 0
        results.append(replace(result, volume_q_value=vq, density_q_value=dq, is_anomaly=is_anomaly))
    return results

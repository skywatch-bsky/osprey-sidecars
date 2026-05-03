# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

from scipy.stats import norm, poisson

from quote_overdispersion.config import AnalysisConfig
from quote_overdispersion.db import AggregatedRow, ScoredResult


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


def compute_p_value(observed: int, expected_lambda: float) -> float:
    """Poisson survival function for volume counts.

    When expected_lambda <= 0, return 1.0 (AC1.5: zero baseline edge case).
    """
    if expected_lambda <= 0:
        return 1.0
    return float(poisson.sf(max(0, observed - 1), expected_lambda))


def compute_density_p_value(
    observed_density: float,
    expected_density: float,
    n_observations: int,
) -> float:
    """Normal approximation z-test for sharer density ratio (0-1 continuous value).

    Uses binomial proportion standard error.
    When expected_density <= 0 or n_observations < 2, return 1.0.
    """
    if expected_density <= 0 or n_observations < 2:
        return 1.0
    se = (expected_density * (1 - expected_density) / n_observations) ** 0.5
    if se <= 0:
        return 1.0
    z = (observed_density - expected_density) / se
    if z <= 0:
        return 1.0
    return float(norm.sf(z))


def determine_baseline(
    row: AggregatedRow,
    cold_start_min_days: int,
) -> tuple[float, float, str]:
    """Select baseline for both volume and density signals.

    Returns (volume_lambda, density_lambda, baseline_source).

    AC1.3: If baseline_days_available >= cold_start_min_days and rolling means are present,
            use entity baselines.
    AC1.4: Otherwise, if population medians are available and > 0, use population medians.
    AC1.5: Otherwise (no data), return (0.0, 0.0, 'population').
    """
    if (
        row.baseline_days_available >= cold_start_min_days
        and row.rolling_volume_mean is not None
        and row.rolling_density_mean is not None
    ):
        return row.rolling_volume_mean, row.rolling_density_mean, 'entity'

    if (
        row.population_volume_median is not None
        and row.population_volume_median > 0
        and row.population_density_median is not None
        and row.population_density_median > 0
    ):
        return row.population_volume_median, row.population_density_median, 'population'

    return 0.0, 0.0, 'population'


def score_row(
    row: AggregatedRow,
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
) -> ScoredResult:
    """Score one row against thresholds.

    Computes both p-values and determines is_anomaly flag.
    is_anomaly = 1 if EITHER volume_p_value < threshold OR density_p_value < threshold.
    """
    expected_volume_lambda, expected_density_lambda, baseline_source = determine_baseline(
        row,
        config.cold_start_min_days,
    )

    volume_p_value = compute_p_value(row.total_shares, expected_volume_lambda)
    density_p_value = compute_density_p_value(
        row.sharer_density,
        expected_density_lambda,
        row.unique_sharers,
    )

    quoted_author_did = extract_quoted_author_did(row.quoted_uri)

    is_anomaly = (
        1 if (volume_p_value < config.volume_p_threshold or density_p_value < config.density_p_threshold) else 0
    )

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
        density_p_value=density_p_value,
        is_anomaly=is_anomaly,
        baseline_source=baseline_source,
        baseline_days_available=row.baseline_days_available,
        sample_dids=row.sample_dids,
    )


def score_rows(
    rows: list[AggregatedRow],
    config: AnalysisConfig,
    granularity: str,
    run_timestamp: datetime,
) -> list[ScoredResult]:
    """Score all rows."""
    return [
        score_row(
            row,
            config,
            granularity,
            run_timestamp,
        )
        for row in rows
    ]

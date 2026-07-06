# pattern: Functional Core
from __future__ import annotations

from signup_anomaly.config import AnalysisConfig


def daily_aggregation_query(config: AnalysisConfig) -> str:
    """Daily signup baseline with densified zero-filled grid and median centre.

    Densification ensures every host gets one row per calendar day from first_seen,
    filling zero-signup days explicitly (coalesce(..., 0)) so they contribute 0 to the
    rolling window instead of vanishing. This prevents baseline bias from selective
    visibility and enables the cold-start guard to work correctly.

    Expected count is rolling_median (exact quantile over the window); dispersion_index
    (variance/mean ratio) is computed only for rolling_mean > 0, and population_stats
    no longer filters on dispersion_index IS NOT NULL, removing the bias toward
    overdispersed hosts in the population median estimate.
    """
    exclusion_clause = _build_exclusion_clause(config)

    return f"""
        WITH
            raw_counts AS (
                SELECT
                    PdsHost AS pds_host,
                    toDate(__timestamp) AS day,
                    count() AS signup_count,
                    countDistinct(UserId) AS distinct_accounts,
                    arraySlice(groupArray(UserId), 1, 5) AS sample_dids
                FROM {config.source_table}
                WHERE ActionName = 'identity'
                    AND PdsHost IS NOT NULL
                    {exclusion_clause}
                    AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
                GROUP BY pds_host, day
            ),
            hosts AS (
                SELECT pds_host, min(day) AS first_seen
                FROM raw_counts
                GROUP BY pds_host
            ),
            calendar AS (
                SELECT toDate(now()) - number AS day
                FROM numbers({config.baseline_days + 1})
            ),
            dense AS (
                SELECT
                    h.pds_host AS pds_host,
                    c.day AS day,
                    coalesce(r.signup_count, 0) AS signup_count,
                    coalesce(r.distinct_accounts, 0) AS distinct_accounts,
                    r.sample_dids AS sample_dids
                FROM hosts h
                CROSS JOIN calendar c
                LEFT JOIN raw_counts r ON r.pds_host = h.pds_host AND r.day = c.day
                WHERE c.day >= h.first_seen
            ),
            baseline AS (
                SELECT
                    pds_host,
                    day,
                    signup_count,
                    distinct_accounts,
                    sample_dids,
                    medianExact(signup_count) OVER w AS rolling_median,
                    avg(signup_count) OVER w AS rolling_mean,
                    ifNotFinite(varPop(signup_count) OVER w, NULL) AS rolling_variance,
                    count() OVER w AS baseline_days_available
                FROM dense
                WINDOW w AS (
                    PARTITION BY pds_host
                    ORDER BY day
                    ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                )
            ),
            population_stats AS (
                SELECT
                    median(rolling_median) AS population_median_lambda,
                    median(if(rolling_mean > 0, rolling_variance / rolling_mean, NULL)) AS population_dispersion_index
                FROM baseline
                WHERE day = toDate(now())
                    AND baseline_days_available >= {config.cold_start_min_days}
            )
        SELECT
            b.pds_host,
            b.signup_count AS observed_count,
            b.distinct_accounts,
            b.rolling_median,
            b.rolling_mean,
            b.rolling_variance,
            if(b.rolling_mean > 0, b.rolling_variance / b.rolling_mean, NULL) AS dispersion_index,
            toUInt16(b.baseline_days_available) AS baseline_days_available,
            b.sample_dids,
            p.population_median_lambda,
            p.population_dispersion_index
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.day = toDate(now())
    """


def hourly_aggregation_query(config: AnalysisConfig) -> str:
    """Hourly signup baseline with hour-of-day matching.

    Each partition isolates same-clock-hour observations across different days
    (PARTITION BY pds_host, toHour(bucket)), so the rolling window contains only
    observations from the same UTC hour, matching like-hour-to-like-hour. This
    accounts for diurnal signup patterns without losing granularity to hourly averages.

    Baseline_days_available is already in days (count() over a same-hour partition
    naturally counts distinct days); no 24x scaling needed. Trade-off: with
    baseline_days=7, only 7 same-hour observations per window; calibrate via
    SIGNUP_ANOMALY_BASELINE_DAYS if needed.
    """
    exclusion_clause = _build_exclusion_clause(config)

    return f"""
        WITH
            raw_counts AS (
                SELECT
                    PdsHost AS pds_host,
                    toStartOfHour(__timestamp) AS bucket,
                    count() AS signup_count,
                    countDistinct(UserId) AS distinct_accounts,
                    arraySlice(groupArray(UserId), 1, 5) AS sample_dids
                FROM {config.source_table}
                WHERE ActionName = 'identity'
                    AND PdsHost IS NOT NULL
                    {exclusion_clause}
                    AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
                GROUP BY pds_host, bucket
            ),
            hosts AS (
                SELECT pds_host, min(bucket) AS first_seen
                FROM raw_counts
                GROUP BY pds_host
            ),
            calendar AS (
                SELECT toStartOfHour(now()) - toIntervalHour(number) AS bucket
                FROM numbers({(config.baseline_days + 1) * 24})
            ),
            dense AS (
                SELECT
                    h.pds_host AS pds_host,
                    c.bucket AS bucket,
                    coalesce(r.signup_count, 0) AS signup_count,
                    coalesce(r.distinct_accounts, 0) AS distinct_accounts,
                    r.sample_dids AS sample_dids
                FROM hosts h
                CROSS JOIN calendar c
                LEFT JOIN raw_counts r ON r.pds_host = h.pds_host AND r.bucket = c.bucket
                WHERE c.bucket >= h.first_seen
            ),
            baseline AS (
                SELECT
                    pds_host,
                    bucket,
                    signup_count,
                    distinct_accounts,
                    sample_dids,
                    medianExact(signup_count) OVER w AS rolling_median,
                    avg(signup_count) OVER w AS rolling_mean,
                    ifNotFinite(varPop(signup_count) OVER w, NULL) AS rolling_variance,
                    count() OVER w AS baseline_days_available
                FROM dense
                WINDOW w AS (
                    PARTITION BY pds_host, toHour(bucket)
                    ORDER BY bucket
                    ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                )
            ),
            population_stats AS (
                SELECT
                    median(rolling_median) AS population_median_lambda,
                    median(if(rolling_mean > 0, rolling_variance / rolling_mean, NULL)) AS population_dispersion_index
                FROM baseline
                WHERE bucket = toStartOfHour(now())
                    AND baseline_days_available >= {config.cold_start_min_days}
            )
        SELECT
            b.pds_host,
            b.signup_count AS observed_count,
            b.distinct_accounts,
            b.rolling_median,
            b.rolling_mean,
            b.rolling_variance,
            if(b.rolling_mean > 0, b.rolling_variance / b.rolling_mean, NULL) AS dispersion_index,
            toUInt16(b.baseline_days_available) AS baseline_days_available,
            b.sample_dids,
            p.population_median_lambda,
            p.population_dispersion_index
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.bucket = toStartOfHour(now())
    """


def _build_exclusion_clause(config: AnalysisConfig) -> str:
    clauses = ["AND PdsHost NOT LIKE '%bsky.network'"]
    for host in config.excluded_hosts:
        if host == 'bsky.network':
            continue
        clauses.append(f"AND PdsHost != '{host}'")
    return '\n                    '.join(clauses)

# pattern: Functional Core
from __future__ import annotations

from signup_anomaly.config import AnalysisConfig


def daily_aggregation_query(config: AnalysisConfig) -> str:
    exclusion_clause = _build_exclusion_clause(config)

    return f"""
        WITH
            daily_counts AS (
                SELECT
                    PdsHost AS pds_host,
                    toDate(__timestamp) AS day,
                    count() AS signup_count,
                    countDistinct(UserId) AS distinct_accounts,
                    arraySlice(groupArray(UserId), 1, 5) AS sample_dids
                FROM {config.source_table}
                WHERE ActionName = 'identity'
                    AND PdsHost IS NOT NULL
                    AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
                    {exclusion_clause}
                GROUP BY PdsHost, toDate(__timestamp)
            ),
            baseline AS (
                SELECT
                    pds_host,
                    day,
                    signup_count,
                    distinct_accounts,
                    sample_dids,
                    avg(signup_count) OVER (
                        PARTITION BY pds_host
                        ORDER BY day
                        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                    ) AS rolling_mean,
                    ifNotFinite(varPop(signup_count) OVER (
                        PARTITION BY pds_host
                        ORDER BY day
                        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                    ), NULL) AS rolling_variance,
                    count() OVER (
                        PARTITION BY pds_host
                        ORDER BY day
                        ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                    ) AS baseline_days_available,
                    CASE
                        WHEN avg(signup_count) OVER (
                            PARTITION BY pds_host
                            ORDER BY day
                            ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                        ) >= 1.0
                        THEN varPop(signup_count) OVER (
                            PARTITION BY pds_host
                            ORDER BY day
                            ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                        ) / NULLIF(avg(signup_count) OVER (
                            PARTITION BY pds_host
                            ORDER BY day
                            ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                        ), 0)
                        ELSE NULL
                    END AS dispersion_index
                FROM daily_counts
            ),
            population_stats AS (
                SELECT
                    median(rolling_mean) AS population_median_lambda,
                    median(dispersion_index) AS population_dispersion_index
                FROM baseline
                WHERE day = toDate(now())
                    AND dispersion_index IS NOT NULL
                    AND baseline_days_available >= {config.cold_start_min_days}
            )
        SELECT
            b.pds_host,
            b.signup_count AS observed_count,
            b.distinct_accounts,
            b.rolling_mean,
            b.baseline_days_available,
            b.sample_dids,
            p.population_median_lambda,
            b.rolling_variance,
            b.dispersion_index,
            p.population_dispersion_index
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.day = toDate(now())
    """


def hourly_aggregation_query(config: AnalysisConfig) -> str:
    exclusion_clause = _build_exclusion_clause(config)

    return f"""
        WITH
            hourly_counts AS (
                SELECT
                    PdsHost AS pds_host,
                    toStartOfHour(__timestamp) AS hour,
                    count() AS signup_count,
                    countDistinct(UserId) AS distinct_accounts,
                    arraySlice(groupArray(UserId), 1, 5) AS sample_dids
                FROM {config.source_table}
                WHERE ActionName = 'identity'
                    AND PdsHost IS NOT NULL
                    AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
                    {exclusion_clause}
                GROUP BY PdsHost, toStartOfHour(__timestamp)
            ),
            baseline AS (
                SELECT
                    pds_host,
                    hour,
                    signup_count,
                    distinct_accounts,
                    sample_dids,
                    avg(signup_count) OVER (
                        PARTITION BY pds_host
                        ORDER BY hour
                        ROWS BETWEEN {config.baseline_days * 24} PRECEDING AND 1 PRECEDING
                    ) AS rolling_mean,
                    ifNotFinite(varPop(signup_count) OVER (
                        PARTITION BY pds_host
                        ORDER BY hour
                        ROWS BETWEEN {config.baseline_days * 24} PRECEDING AND 1 PRECEDING
                    ), NULL) AS rolling_variance,
                    count() OVER (
                        PARTITION BY pds_host
                        ORDER BY hour
                        ROWS BETWEEN {config.baseline_days * 24} PRECEDING AND 1 PRECEDING
                    ) AS baseline_hours_available,
                    CASE
                        WHEN avg(signup_count) OVER (
                            PARTITION BY pds_host
                            ORDER BY hour
                            ROWS BETWEEN {config.baseline_days * 24} PRECEDING AND 1 PRECEDING
                        ) >= 1.0
                        THEN varPop(signup_count) OVER (
                            PARTITION BY pds_host
                            ORDER BY hour
                            ROWS BETWEEN {config.baseline_days * 24} PRECEDING AND 1 PRECEDING
                        ) / NULLIF(avg(signup_count) OVER (
                            PARTITION BY pds_host
                            ORDER BY hour
                            ROWS BETWEEN {config.baseline_days * 24} PRECEDING AND 1 PRECEDING
                        ), 0)
                        ELSE NULL
                    END AS dispersion_index
                FROM hourly_counts
            ),
            population_stats AS (
                SELECT
                    median(rolling_mean) AS population_median_lambda,
                    median(dispersion_index) AS population_dispersion_index
                FROM baseline
                WHERE hour = toStartOfHour(now())
                    AND dispersion_index IS NOT NULL
                    AND baseline_hours_available >= {config.cold_start_min_days * 24}
            )
        SELECT
            b.pds_host,
            b.signup_count AS observed_count,
            b.distinct_accounts,
            b.rolling_mean,
            toUInt16(intDiv(b.baseline_hours_available, 24)) AS baseline_days_available,
            b.sample_dids,
            p.population_median_lambda,
            b.rolling_variance,
            b.dispersion_index,
            p.population_dispersion_index
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.hour = toStartOfHour(now())
    """


def _build_exclusion_clause(config: AnalysisConfig) -> str:
    clauses = ["AND PdsHost NOT LIKE '%bsky.network'"]
    for host in config.excluded_hosts:
        if host == 'bsky.network':
            continue
        clauses.append(f"AND PdsHost != '{host}'")
    return '\n                    '.join(clauses)

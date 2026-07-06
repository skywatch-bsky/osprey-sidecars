# pattern: Functional Core
from __future__ import annotations

import math
from datetime import datetime

from account_entropy.config import AnalysisConfig
from account_entropy.db import AccountActivityRow, ScoredResult


def compute_entropy(counts: list[int]) -> float:
    """
    Compute Shannon entropy from a histogram of counts.

    Shannon entropy formula: H = -Σ p(x) log2(p(x)) for each non-zero bin.

    Args:
        counts: List of non-negative integers representing bin counts.

    Returns:
        Float entropy value in bits. Zero when all counts are in a single bin.
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def normalized_entropy(counts: list[int], max_bins: int) -> float:
    """Bias-corrected Shannon entropy rescaled to [0, 1].

    Applies the Miller-Madow correction, H_mm = H + (K_occupied - 1) / (2 * N * ln 2)
    bits, where K_occupied is the number of non-zero bins and N the total count,
    then divides by the achievable maximum log2(min(N, max_bins)).

    Returns 0.0 when N < 2 or max_bins < 2 (no meaningful spread is measurable).
    Result is clamped to [0.0, 1.0] because the bias correction can push the
    corrected estimate above the achievable maximum for small N.
    """
    total = sum(counts)
    if total < 2 or max_bins < 2:
        return 0.0
    occupied = sum(1 for c in counts if c > 0)
    corrected = compute_entropy(counts) + (occupied - 1) / (2 * total * math.log(2))
    achievable = math.log2(min(total, max_bins))
    return min(1.0, max(0.0, corrected / achievable))


def coefficient_of_variation(mean: float, stddev: float) -> float:
    """Scale-free variability of inter-post intervals: stddev / mean.

    Returns 0.0 when mean <= 0 (fewer than two posts, or degenerate
    zero-length intervals) — maximally regular by convention.
    """
    if mean <= 0:
        return 0.0
    return stddev / mean


def compute_hourly_entropy(hourly_bins: list[int]) -> float:
    """
    Compute Shannon entropy of posting distribution across 24 hours.

    Builds a 24-bin histogram from array of hour values (0-23),
    then applies Shannon entropy formula.

    Args:
        hourly_bins: List of hour values (0-23) for each post.

    Returns:
        Float entropy value in bits. Range [0, log2(24)] ≈ [0, 4.585].
        Returns 0.0 if all posts are in same hour.
    """
    histogram = [0] * 24
    for hour in hourly_bins:
        histogram[hour] += 1
    return compute_entropy(histogram)


def compute_interval_entropy(
    ordered_timestamps_ms: list[int],
    bin_edges: tuple[int, ...],
) -> tuple[float, float, float]:
    """
    Compute Shannon entropy of inter-post intervals.

    Computes gaps between successive posts, bins them by configurable edges,
    and returns entropy plus mean and standard deviation of intervals.

    Bin structure: [0, edge1), [edge1, edge2), ..., [edgeN, inf)

    Args:
        ordered_timestamps_ms: Unix millisecond timestamps, sorted ascending.
        bin_edges: Tuple of bin edge boundaries in seconds.
                   Default: (60, 300, 900, 3600, 14400, 86400) for 7 bins.

    Returns:
        Tuple of (entropy_float, mean_interval_seconds, stddev_interval_seconds).
        Returns (0.0, 0.0, 0.0) if fewer than 2 timestamps (no intervals).
    """
    if len(ordered_timestamps_ms) < 2:
        return 0.0, 0.0, 0.0

    intervals_seconds = [
        (ordered_timestamps_ms[i + 1] - ordered_timestamps_ms[i]) / 1000.0
        for i in range(len(ordered_timestamps_ms) - 1)
    ]

    mean_interval = sum(intervals_seconds) / len(intervals_seconds)
    variance = sum((x - mean_interval) ** 2 for x in intervals_seconds) / len(intervals_seconds)
    stddev_interval = variance**0.5

    num_bins = len(bin_edges) + 1
    histogram = [0] * num_bins
    for interval in intervals_seconds:
        placed = False
        for i, edge in enumerate(bin_edges):
            if interval < edge:
                histogram[i] += 1
                placed = True
                break
        if not placed:
            histogram[-1] += 1

    return compute_entropy(histogram), mean_interval, stddev_interval


def score_account(
    row: AccountActivityRow,
    config: AnalysisConfig,
    run_timestamp: datetime,
    window_start: datetime,
    window_end: datetime,
) -> ScoredResult:
    """
    Score a single account for bot-like posting behaviour.

    Computes hourly and interval entropy, applies thresholds independently,
    and flags is_bot_like only when BOTH signals cross their thresholds (conjunction).

    Critical threshold directions:
    - Hourly entropy: HIGH entropy = bot-like. Flag when >= threshold (default 3.9).
    - Interval entropy: LOW entropy = bot-like. Flag when <= threshold (default 1.5).

    Args:
        row: Account activity data (hourly bins, sorted timestamps).
        config: Analysis configuration with entropy thresholds.
        run_timestamp: Timestamp when this analysis runs.
        window_start: Start of analysis window.
        window_end: End of analysis window.

    Returns:
        ScoredResult with entropy scores, flags, and bot-like determination.
    """
    hourly_entropy = compute_hourly_entropy(row.hourly_bins)
    interval_entropy, mean_interval, stddev_interval = compute_interval_entropy(
        row.ordered_timestamps, config.interval_bin_edges
    )

    # High hourly entropy (near uniform) = bot signal
    hourly_flag = 1 if hourly_entropy >= config.hourly_entropy_threshold else 0

    # Low interval entropy (regular gaps) = bot signal
    interval_flag = 1 if interval_entropy <= config.interval_entropy_threshold else 0

    # Conjunction: both signals must fire for is_bot_like
    is_bot_like = 1 if (hourly_flag == 1 and interval_flag == 1) else 0

    return ScoredResult(
        run_timestamp=run_timestamp,
        user_id=row.user_id,
        window_start=window_start,
        window_end=window_end,
        post_count=row.post_count,
        hourly_entropy=hourly_entropy,
        interval_entropy=interval_entropy,
        mean_interval_seconds=mean_interval,
        stddev_interval_seconds=stddev_interval,
        is_bot_like=is_bot_like,
        hourly_flag=hourly_flag,
        interval_flag=interval_flag,
        sample_rkeys=row.sample_rkeys,
    )


def score_accounts(
    rows: list[AccountActivityRow],
    config: AnalysisConfig,
    run_timestamp: datetime,
    window_start: datetime,
    window_end: datetime,
) -> list[ScoredResult]:
    """
    Score multiple accounts for bot-like behaviour.

    Args:
        rows: List of account activity rows.
        config: Analysis configuration.
        run_timestamp: When this analysis runs.
        window_start: Start of analysis window.
        window_end: End of analysis window.

    Returns:
        List of scored results, one per input row.
    """
    return [score_account(row, config, run_timestamp, window_start, window_end) for row in rows]

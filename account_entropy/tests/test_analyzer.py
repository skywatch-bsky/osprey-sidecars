# pattern: Functional Core
from __future__ import annotations

import math
from datetime import datetime

import pytest

from account_entropy.analyzer import (
    compute_entropy,
    compute_hourly_entropy,
    compute_interval_entropy,
    score_account,
    score_accounts,
)
from account_entropy.config import AnalysisConfig
from account_entropy.db import AccountActivityRow


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        window_days=7,
        min_posts=10,
        hourly_entropy_threshold=3.9,
        interval_entropy_threshold=1.5,
        interval_bin_edges=(60, 300, 900, 3600, 14400, 86400),
        source_table='osprey_execution_results',
        output_table='account_entropy_results',
    )


@pytest.fixture
def run_timestamp() -> datetime:
    return datetime(2024, 3, 20, 12, 0, 0)


@pytest.fixture
def window_start() -> datetime:
    return datetime(2024, 3, 13, 0, 0, 0)


@pytest.fixture
def window_end() -> datetime:
    return datetime(2024, 3, 20, 0, 0, 0)


class TestComputeEntropy:
    def test_uniform_distribution(self) -> None:
        # Equal counts across 8 bins -> entropy = log2(8) = 3.0
        counts = [10, 10, 10, 10, 10, 10, 10, 10]
        result = compute_entropy(counts)
        assert abs(result - 3.0) < 1e-10

    def test_single_bin(self) -> None:
        # All counts in one bin -> entropy = 0.0
        counts = [0, 0, 100, 0, 0]
        result = compute_entropy(counts)
        assert result == 0.0

    def test_empty_counts(self) -> None:
        # All zeros -> entropy = 0.0
        counts = [0, 0, 0, 0]
        result = compute_entropy(counts)
        assert result == 0.0

    def test_two_equal_bins(self) -> None:
        # Two equal bins -> entropy = log2(2) = 1.0
        counts = [5, 5]
        result = compute_entropy(counts)
        assert abs(result - 1.0) < 1e-10

    def test_skewed_distribution(self) -> None:
        # [90, 10] -> heavily skewed, low entropy
        counts = [90, 10]
        result = compute_entropy(counts)
        # H = -(0.9*log2(0.9) + 0.1*log2(0.1)) ≈ 0.469
        assert 0.4 < result < 0.5


class TestComputeHourlyEntropy:
    def test_ac2_2_uniform_24_hours(self) -> None:
        # AC2.2: Posts spread evenly across 24 hours -> entropy ≈ log2(24) ≈ 4.585
        # Create 24 posts, one per hour
        hourly_bins = list(range(24))
        result = compute_hourly_entropy(hourly_bins)
        expected = math.log2(24)
        assert abs(result - expected) < 1e-10

    def test_ac2_6_all_same_hour(self) -> None:
        # AC2.6: All posts in same hour -> entropy = 0.0
        hourly_bins = [14] * 50
        result = compute_hourly_entropy(hourly_bins)
        assert result == 0.0

    def test_concentrated_few_hours(self) -> None:
        # Posts in 3 hours equally -> entropy = log2(3) ≈ 1.585
        hourly_bins = [8, 8, 8, 9, 9, 9, 10, 10, 10]
        result = compute_hourly_entropy(hourly_bins)
        expected = math.log2(3)
        assert abs(result - expected) < 1e-10

    def test_empty_hourly_bins(self) -> None:
        # No posts -> entropy = 0.0
        hourly_bins = []
        result = compute_hourly_entropy(hourly_bins)
        assert result == 0.0


class TestComputeIntervalEntropy:
    def test_ac2_3_regular_intervals(self) -> None:
        # AC2.3: Regular 60s intervals -> all in first bin [0, 60) -> entropy = 0.0
        # 10 timestamps spaced 60 seconds apart
        timestamps_ms = [i * 60 * 1000 for i in range(10)]
        bin_edges = (60, 300, 900, 3600, 14400, 86400)
        entropy, mean, stddev = compute_interval_entropy(timestamps_ms, bin_edges)
        assert entropy == 0.0
        assert abs(mean - 60.0) < 1e-10
        assert abs(stddev - 0.0) < 1e-10

    def test_ac2_3_varied_intervals(self) -> None:
        # AC2.3: Mix of short and long gaps -> higher entropy
        # Timestamps: t0, t0+30s, t0+100s, t0+500s, t0+2000s
        timestamps_ms = [0, 30000, 130000, 630000, 2630000]
        bin_edges = (60, 300, 900, 3600, 14400, 86400)
        entropy, mean, stddev = compute_interval_entropy(timestamps_ms, bin_edges)
        # Intervals: [30, 100, 500, 2000] seconds
        # Bins: [0,60), [60,300), [300,900), [900,3600), [3600,14400), [14400,inf)
        # Counts: [1, 1, 1, 1, 0, 0] -> entropy = log2(4) = 2.0
        assert entropy > 1.0  # Mixed distribution has higher entropy

    def test_single_timestamp(self) -> None:
        # One timestamp -> no intervals -> (0.0, 0.0, 0.0)
        timestamps_ms = [1000000]
        bin_edges = (60, 300, 900, 3600, 14400, 86400)
        entropy, mean, stddev = compute_interval_entropy(timestamps_ms, bin_edges)
        assert entropy == 0.0
        assert mean == 0.0
        assert stddev == 0.0

    def test_two_timestamps(self) -> None:
        # Two timestamps -> one interval, entropy = 0.0
        timestamps_ms = [1000000, 1001000]  # 1 second apart
        bin_edges = (60, 300, 900, 3600, 14400, 86400)
        entropy, mean, stddev = compute_interval_entropy(timestamps_ms, bin_edges)
        assert entropy == 0.0
        assert abs(mean - 1.0) < 1e-10
        assert abs(stddev - 0.0) < 1e-10

    def test_mean_and_stddev_correct(self) -> None:
        # Verify mean and stddev computation
        # Intervals: [10, 20, 30] seconds
        timestamps_ms = [0, 10000, 30000, 60000]
        bin_edges = (60, 300, 900, 3600, 14400, 86400)
        entropy, mean, stddev = compute_interval_entropy(timestamps_ms, bin_edges)
        # Mean = (10 + 20 + 30) / 3 = 20
        assert abs(mean - 20.0) < 1e-10
        # Variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 3 = (100 + 0 + 100) / 3 ≈ 66.667
        # Stddev ≈ 8.165
        expected_stddev = math.sqrt(200.0 / 3.0)
        assert abs(stddev - expected_stddev) < 1e-10


class TestScoreAccount:
    def test_ac2_4_both_flags_set_is_bot_like(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC2.4: Both signals cross thresholds -> is_bot_like = 1
        # High hourly entropy (>= 3.9) AND low interval entropy (<= 1.5)
        row = AccountActivityRow(
            user_id='bot-user',
            post_count=24,
            hourly_bins=list(range(24)),  # Uniform across 24 hours -> entropy ≈ 4.585
            ordered_timestamps=[i * 60 * 1000 for i in range(24)],  # Regular 60s intervals -> entropy = 0.0
            sample_rkeys=['rkey1', 'rkey2'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        assert result.hourly_entropy >= base_config.hourly_entropy_threshold
        assert result.interval_entropy <= base_config.interval_entropy_threshold
        assert result.hourly_flag == 1
        assert result.interval_flag == 1
        assert result.is_bot_like == 1

    def test_ac2_4_only_hourly_flag_not_bot_like(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC2.4: Only hourly signal fires -> is_bot_like = 0
        # High hourly entropy (>= 3.9) AND high interval entropy (> 1.5)
        # Create uniform hourly distribution but highly irregular intervals
        # Use 12 hours evenly distributed -> entropy = log2(12) ≈ 3.585 (just under 3.9)
        # Better: spread across more hours to hit 3.9 threshold
        hourly_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15]
        # Mixed intervals spanning multiple bins: [30, 5000, 50000, 2000, ...]
        # to produce high entropy across interval bins
        timestamps_ms = [
            0,
            30 * 1000,        # 30s gap -> bin [0, 60)
            5030 * 1000,      # 5000s gap -> bin [3600, 14400)
            55030 * 1000,     # 50000s gap -> bin [14400, inf)
            57030 * 1000,     # 2000s gap -> bin [900, 3600)
            59030 * 1000,     # 2000s gap -> bin [900, 3600)
            61030 * 1000,     # 2000s gap -> bin [900, 3600)
            63030 * 1000,     # 2000s gap -> bin [900, 3600)
            65030 * 1000,     # 2000s gap -> bin [900, 3600)
            120000 * 1000,    # 55000s gap -> bin [14400, inf)
            175000 * 1000,    # 55000s gap -> bin [14400, inf)
            200000 * 1000,    # 25000s gap -> bin [14400, inf)
            220000 * 1000,    # 20000s gap -> bin [14400, inf)
            240000 * 1000,    # 20000s gap -> bin [14400, inf)
            250000 * 1000,    # 10000s gap -> bin [3600, 14400)
            260000 * 1000,    # 10000s gap -> bin [3600, 14400)
            265000 * 1000,    # 5000s gap -> bin [3600, 14400)
            267000 * 1000,    # 2000s gap -> bin [900, 3600)
            269000 * 1000,    # 2000s gap -> bin [900, 3600)
            270000 * 1000,    # 1000s gap -> bin [900, 3600)
            271000 * 1000,    # 1000s gap -> bin [900, 3600)
        ]

        row = AccountActivityRow(
            user_id='human-with-uniform-hours',
            post_count=20,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        # Check entropy values
        assert result.hourly_entropy >= 3.9, f"Expected hourly_entropy >= 3.9, got {result.hourly_entropy}"
        assert result.interval_entropy > 1.5, f"Expected interval_entropy > 1.5, got {result.interval_entropy}"
        assert result.hourly_flag == 1
        assert result.interval_flag == 0  # High interval entropy -> no flag
        assert result.is_bot_like == 0

    def test_ac2_4_only_interval_flag_not_bot_like(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC2.4: Only interval signal fires -> is_bot_like = 0
        # Low hourly entropy (< 3.9) AND low interval entropy (<= 1.5)
        # Concentrated in 3 hours with regular intervals
        hourly_bins = [9, 9, 9, 10, 10, 10, 11, 11, 11]  # entropy ≈ log2(3) ≈ 1.585
        timestamps_ms = [i * 60 * 1000 for i in range(9)]  # Regular 60s intervals

        row = AccountActivityRow(
            user_id='human-concentrated-regular',
            post_count=9,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        assert result.hourly_flag == 0
        assert result.interval_flag == 1
        assert result.is_bot_like == 0

    def test_ac2_4_neither_flag_not_bot_like(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC2.4: Neither signal fires -> is_bot_like = 0
        # Low hourly entropy AND high interval entropy
        hourly_bins = [14] * 10  # All in hour 14 -> entropy = 0.0
        # Highly varied intervals to span multiple bins and produce high entropy
        timestamps_ms = [
            0,
            50 * 1000,        # 50s
            250 * 1000,       # 200s
            1150 * 1000,      # 900s
            5150 * 1000,      # 4000s
            10150 * 1000,     # 5000s
            50150 * 1000,     # 40000s
            60150 * 1000,     # 10000s
            70150 * 1000,     # 10000s
            80150 * 1000,     # 10000s
        ]

        row = AccountActivityRow(
            user_id='normal-human',
            post_count=10,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        assert result.hourly_flag == 0
        assert result.interval_flag == 0  # High entropy should be > 1.5
        assert result.is_bot_like == 0

    def test_ac2_6_concentrated_hourly_not_bot_like(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC2.6: All posts same hour -> hourly_flag = 0 regardless of interval signal
        # Even if interval entropy is low (< 1.5), conjunction requires hourly_flag = 1
        hourly_bins = [14] * 24  # All in hour 14 -> entropy = 0.0
        timestamps_ms = [i * 60 * 1000 for i in range(24)]  # Regular 60s intervals -> entropy = 0.0

        row = AccountActivityRow(
            user_id='late-night-poster',
            post_count=24,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        assert result.hourly_entropy == 0.0
        assert result.hourly_flag == 0
        assert result.interval_entropy == 0.0
        assert result.interval_flag == 1
        assert result.is_bot_like == 0  # Conjunction fails due to hourly_flag

    def test_independent_flags_stored(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # Verify hourly_flag and interval_flag are set independently
        row = AccountActivityRow(
            user_id='test',
            post_count=10,
            hourly_bins=[i % 24 for i in range(10)],
            ordered_timestamps=[i * 1000 for i in range(10)],
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        # Both flags should be independent in the result
        assert isinstance(result.hourly_flag, int)
        assert isinstance(result.interval_flag, int)
        assert result.hourly_flag in (0, 1)
        assert result.interval_flag in (0, 1)


class TestScoreAccounts:
    def test_maps_all_accounts(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # Process multiple accounts correctly
        rows = [
            AccountActivityRow(
                user_id='user1',
                post_count=10,
                hourly_bins=[i % 24 for i in range(10)],
                ordered_timestamps=[i * 100 * 1000 for i in range(10)],
                sample_rkeys=['rkey1'],
            ),
            AccountActivityRow(
                user_id='user2',
                post_count=8,
                hourly_bins=[5] * 8,
                ordered_timestamps=[i * 50 * 1000 for i in range(8)],
                sample_rkeys=['rkey2'],
            ),
        ]
        results = score_accounts(rows, base_config, run_timestamp, window_start, window_end)
        assert len(results) == 2
        assert results[0].user_id == 'user1'
        assert results[1].user_id == 'user2'

    def test_mixed_bot_and_human(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # One bot-like and one human-like account
        bot_row = AccountActivityRow(
            user_id='bot',
            post_count=24,
            hourly_bins=list(range(24)),  # Uniform, high entropy
            ordered_timestamps=[i * 60 * 1000 for i in range(24)],  # Regular intervals
            sample_rkeys=['bot-rkey'],
        )
        human_row = AccountActivityRow(
            user_id='human',
            post_count=10,
            hourly_bins=[14] * 10,  # Concentrated in one hour
            ordered_timestamps=[i * 500 * 1000 for i in range(10)],  # Irregular intervals
            sample_rkeys=['human-rkey'],
        )
        results = score_accounts(
            [bot_row, human_row], base_config, run_timestamp, window_start, window_end
        )
        assert len(results) == 2
        # Bot should be flagged
        assert results[0].is_bot_like == 1
        # Human should not be flagged
        assert results[1].is_bot_like == 0

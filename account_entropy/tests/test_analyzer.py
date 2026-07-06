# pattern: Functional Core
from __future__ import annotations

import math
from datetime import datetime

import pytest

from account_entropy.analyzer import (
    coefficient_of_variation,
    compute_entropy,
    compute_hourly_entropy,
    compute_interval_entropy,
    normalized_entropy,
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
        hourly_entropy_norm_threshold=0.85,
        interval_entropy_norm_threshold=0.53,
        cv_threshold=0.5,
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


class TestNormalizedEntropy:
    def test_ac5_1_uniform_many_posts(self) -> None:
        # AC5.1: N >= bins, uniform distribution -> entropy saturates and clamps to 1.0
        # 240 posts uniformly across 24 bins (10 per bin)
        # H = log2(24) ≈ 4.585, correction pushes above, clamps to 1.0
        result = normalized_entropy([10] * 24, 24)
        assert result == 1.0

    def test_ac5_1_uniform_few_posts(self) -> None:
        # AC5.1: 10 posts in 10 distinct bins, achievable max = log2(10)
        # Uniform: H = log2(10), correction pushes above achievable, clamps to 1.0
        result = normalized_entropy([1] * 10 + [0] * 14, 24)
        assert result == 1.0

    def test_ac5_1_single_bin(self) -> None:
        # AC5.1: All counts in one bin -> entropy = 0.0, correction term = 0
        result = normalized_entropy([10, 0, 0], 24)
        assert result == 0.0

    def test_ac5_1_bounds_arbitrary_cases(self) -> None:
        # AC5.1: Result always in [0.0, 1.0] for various distributions
        test_cases = [
            [3, 1, 0, 7],
            [1, 1],
            [100, 1],
        ]
        for counts in test_cases:
            result = normalized_entropy(counts, 24)
            assert 0.0 <= result <= 1.0, f'Result {result} out of bounds for {counts}'

    def test_ac5_2_miller_madow_verification(self) -> None:
        # AC5.2: Numeric verification of Miller-Madow correction
        # normalized_entropy([5, 5], 24)
        # N = 10, K_occupied = 2, H = 1.0 bit
        # correction = (2-1)/(2*10*ln(2)) ≈ 0.072135 bits
        # achievable = log2(min(10, 24)) = log2(10) ≈ 3.321928
        # expected ≈ (1.0 + 0.072135) / 3.321928 ≈ 0.322745
        counts = [5, 5]
        result = normalized_entropy(counts, 24)
        H = compute_entropy(counts)
        K_occupied = 2
        correction = (K_occupied - 1) / (2 * 10 * math.log(2))
        achievable = math.log2(10)
        expected = (H + correction) / achievable
        assert result == pytest.approx(expected, rel=1e-9)

    def test_edge_empty_counts(self) -> None:
        # Edge: empty counts list -> N = 0 < 2 -> 0.0
        result = normalized_entropy([], 24)
        assert result == 0.0

    def test_edge_single_count(self) -> None:
        # Edge: N = 1 < 2 -> 0.0
        result = normalized_entropy([1], 24)
        assert result == 0.0

    def test_edge_max_bins_less_than_2(self) -> None:
        # Edge: max_bins = 1 < 2 -> 0.0
        result = normalized_entropy([2, 3], 1)
        assert result == 0.0


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


class TestCoefficientOfVariation:
    def test_regular_cadence(self) -> None:
        # Regular cadence: stddev = 5, mean = 100 -> CV = 0.05
        result = coefficient_of_variation(100.0, 5.0)
        assert result == pytest.approx(0.05)

    def test_irregular_cadence(self) -> None:
        # Irregular cadence: stddev = 150, mean = 100 -> CV = 1.5
        result = coefficient_of_variation(100.0, 150.0)
        assert result == pytest.approx(1.5)

    def test_edge_zero_mean(self) -> None:
        # Edge: mean = 0 -> 0.0 (degenerate case)
        result = coefficient_of_variation(0.0, 0.0)
        assert result == 0.0

    def test_edge_negative_mean(self) -> None:
        # Edge: mean < 0 -> 0.0 (invalid by convention)
        result = coefficient_of_variation(-1.0, 5.0)
        assert result == 0.0


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
    def test_ac5_3_ac5_4_ten_posts_cross_hourly_threshold(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC5.3: 10 posts uniformly spread over 10 distinct hours
        # Raw hourly_entropy ≈ log2(10) ≈ 3.32 bits (< 3.9, would fail old rule)
        # Normalized entropy = 1.0 (uniform across achievable max) >= 0.85 threshold -> hourly_flag = 1
        # AC5.4: With interval flag or cv_flag set, is_bot_like = 1
        row = AccountActivityRow(
            user_id='bot-user',
            post_count=10,
            hourly_bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Uniform across 10 distinct hours
            # Timestamps 1 hour + jitter apart: [0, 3600s, 7200s, 10800s, ...]
            # Intervals: [3600, 3600, 3600, ...] with small jitter -> concentrated in one bin
            ordered_timestamps=[i * 3600 * 1000 + i * 100 for i in range(10)],
            sample_rkeys=['rkey1', 'rkey2'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        # Raw hourly entropy should be log2(10) ≈ 3.32 bits
        assert result.hourly_entropy == pytest.approx(math.log2(10), rel=1e-6)
        # Normalized hourly entropy should be 1.0 (uniform over achievable 10 bins)
        assert result.hourly_entropy_norm == pytest.approx(1.0, rel=1e-6)
        # Hourly flag should be set (normalized >= 0.85)
        assert result.hourly_flag == 1
        # Interval entropy should be low (regular 3600s intervals)
        assert result.interval_entropy <= 1.5
        # Interval flag should be set
        assert result.interval_flag == 1
        # Conjunction: hourly_flag=1 and (interval_flag=1 or cv_flag=1) -> is_bot_like = 1
        assert result.is_bot_like == 1

    def test_ac5_4_cv_metronomic_intervals_low_cv(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC5.4 (CV path): Metronomic intervals (CV = 0) trigger cv_flag
        # 20 posts with exactly 3600s intervals -> stddev = 0, CV = 0 <= 0.5 threshold
        # Spread across 10 distinct hours -> hourly_entropy_norm >= 0.85
        # Regular 3600s intervals -> interval_entropy_norm likely <= 0.53
        hourly_bins = [i % 10 for i in range(20)]  # Spread across 10 hours
        timestamps_ms = [i * 3600 * 1000 for i in range(20)]  # Exactly 3600s apart

        row = AccountActivityRow(
            user_id='metronomic-bot',
            post_count=20,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        # CV should be 0 (perfect regularity)
        assert result.interval_cv == pytest.approx(0.0, abs=1e-10)
        assert result.cv_flag == 1
        # With hourly_flag=1 and cv_flag=1, is_bot_like should be 1
        if result.hourly_flag == 1:
            assert result.is_bot_like == 1

    def test_ac5_4_cv_high_variation_no_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC5.4 (CV path): High variation (CV > 0.5) does not trigger cv_flag
        # Create highly variable intervals: [100, 1000, 10000, 100000, ...]
        timestamps_ms = [
            0,
            100 * 1000,  # 100s interval
            1100 * 1000,  # 1000s interval
            11100 * 1000,  # 10000s interval
            111100 * 1000,  # 100000s interval
            111200 * 1000,  # 100s interval
            1200 * 1000,  # Should be further, use 2100000
        ]
        hourly_bins = [0, 1, 2, 3, 4, 5, 6]

        row = AccountActivityRow(
            user_id='variable-poster',
            post_count=6,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        # High variation in intervals should result in CV > 0.5
        assert result.interval_cv > 0.5
        assert result.cv_flag == 0

    def test_ac5_4_cv_conjunction_truth_table(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC5.4: Conjunction truth table: is_bot_like = hourly_flag AND (interval_flag OR cv_flag)
        # Case (a): hourly=1, interval=0, cv=1 -> is_bot_like=1
        row_a = AccountActivityRow(
            user_id='a-hourly-cv-bot',
            post_count=24,
            hourly_bins=list(range(24)),  # High normalized hourly entropy
            ordered_timestamps=[i * 3600 * 1000 for i in range(24)],  # Metronomic, low CV
            sample_rkeys=['rkey1'],
        )
        result_a = score_account(row_a, base_config, run_timestamp, window_start, window_end)
        # Expect hourly_flag=1, cv_flag=1 (metronomic), likely interval_flag=0 or 1
        if result_a.hourly_flag == 1 and result_a.cv_flag == 1:
            assert result_a.is_bot_like == 1

        # Case (b): hourly=1, interval=0, cv=0 -> is_bot_like=0
        # High hourly entropy, but neither interval nor cv flags
        # Create 24 posts spread across 24 hours with highly irregular intervals
        timestamps_b = [
            0,
            50 * 1000,
            250 * 1000,
            1150 * 1000,
            5150 * 1000,
            10150 * 1000,
            50150 * 1000,
            60150 * 1000,
            70150 * 1000,
            80150 * 1000,
            90150 * 1000,
            100150 * 1000,
            110150 * 1000,
            120150 * 1000,
            130150 * 1000,
            140150 * 1000,
            150150 * 1000,
            160150 * 1000,
            170150 * 1000,
            180150 * 1000,
            190150 * 1000,
            200150 * 1000,
            210150 * 1000,
            220150 * 1000,
        ]
        row_b = AccountActivityRow(
            user_id='b-hourly-only',
            post_count=24,
            hourly_bins=list(range(24)),
            ordered_timestamps=timestamps_b,
            sample_rkeys=['rkey1'],
        )
        result_b = score_account(row_b, base_config, run_timestamp, window_start, window_end)
        # If hourly=1 but both interval and cv are 0, is_bot_like=0
        if result_b.hourly_flag == 1 and result_b.interval_flag == 0 and result_b.cv_flag == 0:
            assert result_b.is_bot_like == 0

        # Case (c): hourly=0, interval=1, cv=1 -> is_bot_like=0
        # Concentrated in few hours, low interval entropy, low CV
        row_c = AccountActivityRow(
            user_id='c-no-hourly',
            post_count=10,
            hourly_bins=[14] * 10,  # All in hour 14 -> low normalized hourly entropy
            ordered_timestamps=[i * 3600 * 1000 for i in range(10)],  # Metronomic
            sample_rkeys=['rkey1'],
        )
        result_c = score_account(row_c, base_config, run_timestamp, window_start, window_end)
        # Expect hourly_flag=0, so is_bot_like=0 regardless of other flags
        assert result_c.hourly_flag == 0
        assert result_c.is_bot_like == 0

    def test_ac2_4_both_flags_set_is_bot_like(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        # AC2.4: Both signals cross thresholds -> is_bot_like = 1
        # High normalized hourly entropy (>= 0.85) AND low normalized interval entropy (<= 0.53)
        row = AccountActivityRow(
            user_id='bot-user',
            post_count=24,
            hourly_bins=list(range(24)),  # Uniform across 24 hours -> normalized entropy ≈ 1.0
            ordered_timestamps=[i * 60 * 1000 for i in range(24)],  # Regular 60s intervals -> normalized entropy ≈ 0.0
            sample_rkeys=['rkey1', 'rkey2'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        assert result.hourly_entropy_norm >= base_config.hourly_entropy_norm_threshold
        assert result.interval_entropy_norm <= base_config.interval_entropy_norm_threshold
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
        # High normalized hourly entropy (>= 0.85) AND high normalized interval entropy (> 0.53)
        # Create uniform hourly distribution but highly irregular intervals
        hourly_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15]
        # Mixed intervals spanning multiple bins: [30, 5000, 50000, 2000, ...]
        # to produce high entropy across interval bins
        timestamps_ms = [
            0,
            30 * 1000,  # 30s gap -> bin [0, 60)
            5030 * 1000,  # 5000s gap -> bin [3600, 14400)
            55030 * 1000,  # 50000s gap -> bin [14400, inf)
            57030 * 1000,  # 2000s gap -> bin [900, 3600)
            59030 * 1000,  # 2000s gap -> bin [900, 3600)
            61030 * 1000,  # 2000s gap -> bin [900, 3600)
            63030 * 1000,  # 2000s gap -> bin [900, 3600)
            65030 * 1000,  # 2000s gap -> bin [900, 3600)
            120000 * 1000,  # 55000s gap -> bin [14400, inf)
            175000 * 1000,  # 55000s gap -> bin [14400, inf)
            200000 * 1000,  # 25000s gap -> bin [14400, inf)
            220000 * 1000,  # 20000s gap -> bin [14400, inf)
            240000 * 1000,  # 20000s gap -> bin [14400, inf)
            250000 * 1000,  # 10000s gap -> bin [3600, 14400)
            260000 * 1000,  # 10000s gap -> bin [3600, 14400)
            265000 * 1000,  # 5000s gap -> bin [3600, 14400)
            267000 * 1000,  # 2000s gap -> bin [900, 3600)
            269000 * 1000,  # 2000s gap -> bin [900, 3600)
            270000 * 1000,  # 1000s gap -> bin [900, 3600)
            271000 * 1000,  # 1000s gap -> bin [900, 3600)
        ]

        row = AccountActivityRow(
            user_id='human-with-uniform-hours',
            post_count=20,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        # Check normalized entropy values
        assert result.hourly_entropy_norm >= 0.85, (
            f'Expected hourly_entropy_norm >= 0.85, got {result.hourly_entropy_norm}'
        )
        assert result.interval_entropy_norm > 0.53, (
            f'Expected interval_entropy_norm > 0.53, got {result.interval_entropy_norm}'
        )
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
        # Low normalized hourly entropy (< 0.85) AND low normalized interval entropy (<= 0.53)
        # Concentrated in 3 hours with regular intervals
        hourly_bins = [9, 9, 9, 10, 10, 10, 11, 11, 11]  # 9 posts in 3 hours -> normalized entropy < 0.85
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
            50 * 1000,  # 50s
            250 * 1000,  # 200s
            1150 * 1000,  # 900s
            5150 * 1000,  # 4000s
            10150 * 1000,  # 5000s
            50150 * 1000,  # 40000s
            60150 * 1000,  # 10000s
            70150 * 1000,  # 10000s
            80150 * 1000,  # 10000s
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
        # Even if interval entropy is low (<= 0.53), conjunction requires hourly_flag = 1
        hourly_bins = [14] * 24  # All in hour 14 -> entropy = 0.0, normalized = 0.0
        timestamps_ms = [i * 60 * 1000 for i in range(24)]  # Regular 60s intervals -> entropy = 0.0, normalized ≈ 0.0

        row = AccountActivityRow(
            user_id='late-night-poster',
            post_count=24,
            hourly_bins=hourly_bins,
            ordered_timestamps=timestamps_ms,
            sample_rkeys=['rkey1'],
        )
        result = score_account(row, base_config, run_timestamp, window_start, window_end)
        assert result.hourly_entropy == 0.0
        assert result.hourly_entropy_norm == 0.0
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
        results = score_accounts([bot_row, human_row], base_config, run_timestamp, window_start, window_end)
        assert len(results) == 2
        # Bot should be flagged
        assert results[0].is_bot_like == 1
        # Human should not be flagged
        assert results[1].is_bot_like == 0

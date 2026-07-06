# pattern: Functional Core
"""Tests for Benjamini-Hochberg FDR adjustment."""

from __future__ import annotations

import pytest

from url_overdispersion.fdr import bh_adjust


class TestBhAdjust:
    """Test Benjamini-Hochberg q-value adjustment."""

    def test_ac2_1_hand_computed_example(self) -> None:
        """AC2.1: Hand-computed BH example.

        Input: [0.01, 0.04, 0.03, 0.005]
        Sorted: [0.005, 0.01, 0.03, 0.04] at indices [3, 0, 2, 1]
        n = 4

        Backward pass (position n-1 down to 0):
        position=3 (rank=4, p=0.04): min(1.0, 0.04*4/4) = 0.04, running_min=0.04
        position=2 (rank=3, p=0.03): min(1.0, 0.03*4/3) = 0.04, running_min=min(0.04, 0.04)=0.04
        position=1 (rank=2, p=0.01): min(1.0, 0.01*4/2) = 0.02, running_min=min(0.04, 0.02)=0.02
        position=0 (rank=1, p=0.005): min(1.0, 0.005*4/1) = 0.02, running_min=min(0.02, 0.02)=0.02

        Mapped back to original positions:
        index 3 (input pos 1) -> 0.02
        index 0 (input pos 0) -> 0.02
        index 2 (input pos 2) -> 0.04
        index 1 (input pos 3) -> 0.04

        Expected output: [0.02, 0.04, 0.04, 0.02]
        """
        result = bh_adjust([0.01, 0.04, 0.03, 0.005])
        assert result == pytest.approx([0.02, 0.04, 0.04, 0.02], abs=1e-10)

    def test_ac2_1_monotone_property(self) -> None:
        """AC2.1: q-values are non-decreasing when ordered by p-values."""
        p_values = [0.01, 0.04, 0.03, 0.005]
        q_values = bh_adjust(p_values)

        # Sort both together
        sorted_pairs = sorted(zip(p_values, q_values), key=lambda x: x[0])
        sorted_q = [q for _, q in sorted_pairs]

        # Check monotonicity
        for i in range(len(sorted_q) - 1):
            assert sorted_q[i] <= sorted_q[i + 1]

    def test_ac2_1_monotone_messier_example(self) -> None:
        """AC2.1: Monotone property on messier example with more values."""
        p_values = [0.9, 0.001, 0.5, 0.02, 0.02]
        q_values = bh_adjust(p_values)

        # Sort both together
        sorted_pairs = sorted(zip(p_values, q_values), key=lambda x: x[0])
        sorted_q = [q for _, q in sorted_pairs]

        # Check monotonicity
        for i in range(len(sorted_q) - 1):
            assert sorted_q[i] <= sorted_q[i + 1]

        # Also check q >= p
        for p, q in zip(p_values, q_values):
            assert q >= p

    def test_ac2_1_order_preserved(self) -> None:
        """AC2.1: Input order is preserved in output."""
        p_values = [0.01, 0.04, 0.03, 0.005]
        q_values = bh_adjust(p_values)

        # Verify output length matches input
        assert len(q_values) == len(p_values)

        # Verify by comparing against adjusted sorted list
        sorted_p = sorted(p_values)
        adjusted_sorted = bh_adjust(sorted_p)

        # Un-sort adjusted_sorted back to original order
        sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
        expected = [0.0] * len(p_values)
        for sort_pos, orig_idx in enumerate(sorted_indices):
            expected[orig_idx] = adjusted_sorted[sort_pos]

        assert q_values == pytest.approx(expected, abs=1e-10)

    def test_ac2_4_empty_input(self) -> None:
        """AC2.4: Empty input returns empty output."""
        result = bh_adjust([])
        assert result == []

    def test_ac2_4_single_p_value(self) -> None:
        """AC2.4: Single p-value returns itself as q-value."""
        result = bh_adjust([0.03])
        assert result == pytest.approx([0.03], abs=1e-10)

    def test_cap_at_one(self) -> None:
        """Cap: q-values should not exceed 1.0."""
        result = bh_adjust([0.8, 0.9])
        for q in result:
            assert q <= 1.0

    def test_ties_handling(self) -> None:
        """Ties: Tied p-values should have appropriate q-values."""
        result = bh_adjust([0.05, 0.05])
        # Both have same p-value, should get same q-value
        # Sorted order: both are at positions 0,1 (or 1,2 depending on stability)
        # For a 2-element list with both 0.05:
        # position=1, rank=2: min(1.0, 0.05*2/2) = 0.05, running_min=0.05
        # position=0, rank=1: min(1.0, 0.05*2/1) = 0.10, running_min=min(0.05, 0.10)=0.05
        # Both should get 0.05
        assert result == pytest.approx([0.05, 0.05], abs=1e-10)

    def test_q_values_non_decreasing_with_p_ordering(self) -> None:
        """q-values preserve ordering: if p[i] <= p[j], then typically q[i] <= q[j]."""
        p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        q_values = bh_adjust(p_values)

        # Verify q >= p for all
        for p, q in zip(p_values, q_values):
            assert q >= p

        # Verify monotonicity in q when p is ordered
        for i in range(len(q_values) - 1):
            assert q_values[i] <= q_values[i + 1]

    def test_all_p_values_less_than_one(self) -> None:
        """All q-values should be in [0, 1]."""
        p_values = [0.001, 0.05, 0.1, 0.2, 0.5]
        q_values = bh_adjust(p_values)

        for q in q_values:
            assert 0.0 <= q <= 1.0

    def test_single_small_and_large_p(self) -> None:
        """Test mix of very small and very large p-values."""
        p_values = [0.0001, 0.9999]
        q_values = bh_adjust(p_values)

        # First: 0.0001 * 2 / 2 = 0.0001, then back with running_min
        # Second: 0.9999 * 2 / 1 = 1.9998 -> capped to 1.0, running_min stays from first
        # Actually: position=1 (rank=2): min(1.0, 0.9999*2/2) = 0.9999, running_min=0.9999
        #          position=0 (rank=1): min(1.0, 0.0001*2/1) = 0.0002, running_min=min(0.9999, 0.0002)=0.0002
        assert q_values[0] == pytest.approx(0.0002, abs=1e-10)
        assert q_values[1] == pytest.approx(0.9999, abs=1e-10)

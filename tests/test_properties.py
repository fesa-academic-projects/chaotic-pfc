"""Property-based tests using Hypothesis.

These tests verify mathematical invariants that hold across the input
domain, complementing the example-based tests in other test files.
"""

from __future__ import annotations

import unittest

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from chaotic_pfc.analysis.stats import (
    AreaSummary,
    area_summary,
    consolidate_kaiser,
    lmax_statistics,
)
from chaotic_pfc.analysis.sweep import SweepResult
from chaotic_pfc.dynamics.lyapunov import lyapunov_henon2d
from chaotic_pfc.dynamics.maps import henon_filtered, henon_standard
from chaotic_pfc.dynamics.signals import binary_message, sinusoidal_message

from ._hypothesis_strategies import (
    arrays_with_nan,
    finite_ndarrays,
    lowpass_fir_params,
    safe_henon_params,
    small_sweep_results,
)

# ── Global settings: fast dev profile ───────────────────────────────────
COMMON = settings(max_examples=100, deadline=5000, database=None)


class TestHenonStandardInvariants(unittest.TestCase):
    """Property tests for henon_standard()."""

    @given(params=safe_henon_params(), n=st.integers(5, 50))
    @COMMON
    def test_output_is_finite_or_monotonic_divergence(self, params, n):
        """For every (a,b) in safe range, henon_standard either produces
        all-finite output or diverges definitively (once NaN appears,
        everything after is also NaN/Inf — no recovery)."""
        a, b = params
        X, Y = henon_standard(n, x0=0.0, y0=0.0, a=a, b=b)

        # Shape contract
        self.assertEqual(X.shape, (n + 1,))
        self.assertEqual(Y.shape, (n + 1,))

        # Find the first non-finite index, if any
        finite_mask = np.isfinite(X) & np.isfinite(Y)
        first_bad = np.argmin(finite_mask)  # 0 if first is bad
        if not finite_mask[first_bad]:
            # Once divergence starts, it should be monotonic:
            # all subsequent points should also be non-finite
            after = finite_mask[first_bad:]
            self.assertFalse(
                np.any(after),
                f"henon_standard recovered to finite after divergence "
                f"at index {first_bad} for a={a:.4f}, b={b:.4f}",
            )

    @given(params=safe_henon_params(), n=st.integers(5, 100))
    @COMMON
    def test_length_equals_steps_plus_one(self, params, n):
        a, b = params
        X, Y = henon_standard(n, x0=0.0, y0=0.0, a=a, b=b)
        self.assertEqual(len(X), n + 1)
        self.assertEqual(len(Y), n + 1)


class TestHenonFilteredInvariants(unittest.TestCase):
    """Property tests for henon_filtered()."""

    @given(
        params=safe_henon_params(b_range=(0.1, 0.5)),
        fir=lowpass_fir_params(3, 15),
        n=st.integers(5, 50),
    )
    @COMMON
    def test_lowpass_filtered_output_has_correct_shape(self, params, fir, n):
        a, b = params
        N_filter, wc = fir
        from scipy.signal import firwin

        coeffs = firwin(N_filter + 1, wc)
        X, Y = henon_filtered(n, x0=0.0, y0=0.0, alpha=a, beta=b, c0=coeffs[0], c1=coeffs[1])
        self.assertEqual(len(X), n + 1)
        self.assertEqual(len(Y), n + 1)


class TestLyapunovInvariants(unittest.TestCase):
    """Property tests for Lyapunov functions."""

    @given(
        a=st.floats(1.38, 1.42, allow_nan=False, allow_infinity=False),
        b=st.floats(0.28, 0.32, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 100),
    )
    @settings(max_examples=30, deadline=10000, database=None)
    def test_standard_henon_yields_positive_lyapunov_max(self, a, b, seed):
        """For (a,b) very close to (1.4, 0.3). If bounded, λ_max > 0."""
        assume(abs(a - 1.4) < 0.06 and abs(b - 0.3) < 0.06)
        result = lyapunov_henon2d(alpha=a, beta=b, Nitera=500, Ndiscard=200, seed=seed)

        if not np.isfinite(result.lyapunov_max):
            return  # Orbit diverged — acceptable near chaotic boundary
        self.assertGreater(
            result.lyapunov_max, 0.0, f"λ_max ≤ 0 for a={a}, b={b}: {result.lyapunov_max}"
        )

    @given(
        a=st.floats(1.3, 1.5, allow_nan=False, allow_infinity=False),
        b=st.floats(0.2, 0.4, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 50),
    )
    @settings(max_examples=20, deadline=10000, database=None)
    def test_lyapunov_sum_equals_log_b(self, a, b, seed):
        """λ₁ + λ₂ = ln(b) holds for any (a,b) where orbit stays bounded."""
        result = lyapunov_henon2d(alpha=a, beta=b, Nitera=500, Ndiscard=200, seed=seed)
        # Only assert if orbit stayed bounded (both exponents finite)
        if np.all(np.isfinite(result.all_exponents)):
            computed_sum = float(np.sum(result.all_exponents))
            expected_sum = float(np.log(b))
            np.testing.assert_allclose(computed_sum, expected_sum, rtol=1e-4)


class TestSignalInvariants(unittest.TestCase):
    """Property tests for signal generation functions."""

    @given(n=st.integers(1, 200))
    @COMMON
    def test_binary_message_range(self, n):
        msg = binary_message(n)
        self.assertEqual(msg.shape, (n,))
        self.assertTrue(np.all((msg == -1) | (msg == 1)), f"Values not in {{-1,1}}: {set(msg)}")

    @given(n=st.integers(1, 200), freq=st.floats(0.001, 0.499, allow_nan=False, allow_infinity=False))
    @COMMON
    def test_sinusoidal_message_range(self, n, freq):
        msg = sinusoidal_message(n, normalised_freq=freq)
        self.assertEqual(msg.shape, (n,))
        self.assertTrue(np.all(np.isfinite(msg)))
        self.assertTrue(np.all(msg >= -1.0))
        self.assertTrue(np.all(msg <= 1.0))


class TestAreaSummaryInvariants(unittest.TestCase):
    """Property tests for area_summary()."""

    @given(data=small_sweep_results())
    @COMMON
    def test_counts_sum_to_total(self, data):
        h = data["h"]
        orders = np.arange(2, 2 + h.shape[0])
        cutoffs = np.linspace(0.1, 0.9, h.shape[1])
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=orders,
            cutoffs=cutoffs,
            window="hamming",
            filter_type="lowpass",
        )
        s: AreaSummary = area_summary(result)
        self.assertEqual(s["n_chaotic"] + s["n_periodic"] + s["n_divergent"], s["n_total"])

    @given(data=small_sweep_results())
    @COMMON
    def test_lmax_statistics_ci_contains_mean_when_enough_points(self, data):
        h = data["h"]
        orders = np.arange(2, 2 + h.shape[0])
        cutoffs = np.linspace(0.1, 0.9, h.shape[1])
        result = SweepResult(
            h=h,
            h_std=np.abs(h) * 0.1,
            orders=orders,
            cutoffs=cutoffs,
            window="hamming",
            filter_type="lowpass",
            metadata={"n_initial": 50},
        )
        stats = lmax_statistics(result, region="all_finite", n_bootstrap=100, seed=42)
        if stats["n_used"] >= 3 and not np.isnan(stats["ci_95_low"]):
            self.assertLessEqual(stats["ci_95_low"], stats["mean"])
            self.assertGreaterEqual(stats["ci_95_high"], stats["mean"])


class TestConsolidateKaiserInvariants(unittest.TestCase):
    """Property tests for consolidate_kaiser()."""

    def test_kaiser_entries_per_filter(self):
        """After consolidation, each filter type has at most 1 Kaiser entry."""
        from chaotic_pfc.analysis.stats import load_all_sweeps

        sweeps = load_all_sweeps("data/sweeps")
        consolidated = consolidate_kaiser(sweeps)

        for ft in ["lowpass", "highpass", "bandpass", "bandstop"]:
            kaiser_keys = [k for k in consolidated if k[0] == ft and "kaiser" in k[1]]
            self.assertLessEqual(
                len(kaiser_keys), 1, f"Expected ≤1 Kaiser for {ft}, got {len(kaiser_keys)}"
            )

    def test_non_kaiser_windows_preserved(self):
        """Non-Kaiser windows are passed through unchanged."""
        from chaotic_pfc.analysis.stats import load_all_sweeps

        sweeps = load_all_sweeps("data/sweeps")
        consolidated = consolidate_kaiser(sweeps)

        non_kaiser_original = {k for k in sweeps if sweeps[k].window != "kaiser"}
        non_kaiser_consolidated = {k for k in consolidated if consolidated[k].window != "kaiser"}
        self.assertEqual(non_kaiser_original, non_kaiser_consolidated)


if __name__ == "__main__":
    unittest.main()

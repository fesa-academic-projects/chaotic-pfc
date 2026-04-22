"""tests/test_sweep.py — Unit tests for the sweep module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chaotic_pfc.sweep import (
    FILTER_TYPES,
    WINDOW_DISPLAY_NAMES,
    WINDOWS,
    SweepResult,
    load_sweep,
    precompute_fir_bank,
    run_sweep,
    save_sweep,
)

# ═══════════════════════════════════════════════════════════════════════════
# Catalogue sanity checks
# ═══════════════════════════════════════════════════════════════════════════


class TestCatalogue(unittest.TestCase):
    def test_windows_are_unique(self):
        self.assertEqual(len(WINDOWS), len(set(WINDOWS)))

    def test_every_window_has_display_name(self):
        for w in WINDOWS:
            self.assertIn(w, WINDOW_DISPLAY_NAMES)

    def test_filter_types_values(self):
        self.assertEqual(set(FILTER_TYPES), {"lowpass", "highpass"})


# ═══════════════════════════════════════════════════════════════════════════
# FIR bank
# ═══════════════════════════════════════════════════════════════════════════


class TestFirBank(unittest.TestCase):
    def test_shape(self):
        orders = np.array([2, 3, 5])
        cutoffs = np.linspace(0.1, 0.9, 4)
        bank, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "hamming")
        self.assertEqual(bank.shape, (3, 4, 6))  # max_taps = max(orders) + 1
        self.assertEqual(gains.shape, (3, 4))

    def test_lowpass_dc_gain_approx_one(self):
        """Low-pass FIR filters should have DC gain close to 1."""
        orders = np.array([11, 21, 41])
        cutoffs = np.array([0.3, 0.5])
        _, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "hamming")
        np.testing.assert_allclose(gains, 1.0, atol=1e-6)

    def test_highpass_forces_odd_taps(self):
        """With pass_zero='highpass', scipy's firwin requires odd numtaps."""
        orders = np.array([4, 6, 8])  # all even → must be bumped to 5, 7, 9
        cutoffs = np.array([0.3])
        bank, _ = precompute_fir_bank(orders, cutoffs, "highpass", "hamming")
        # Odd-tap filters should have a non-zero coefficient at the bumped
        # position (index = Nss) — proof that Nss | 1 was applied.
        for i, Nss in enumerate(orders):
            self.assertNotEqual(
                bank[i, 0, int(Nss)], 0.0, f"expected coefficient at index {Nss} for order {Nss}"
            )

    def test_kaiser_window_accepted(self):
        orders = np.array([5])
        cutoffs = np.array([0.4])
        bank, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "kaiser")
        self.assertTrue(np.all(np.isfinite(bank)))
        self.assertTrue(np.all(np.isfinite(gains)))

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            precompute_fir_bank([3], np.array([0.5]), "lowpass", "not_a_window")

    def test_invalid_filter_raises(self):
        with self.assertRaises(ValueError):
            precompute_fir_bank([3], np.array([0.5]), "bandpass", "hamming")


# ═══════════════════════════════════════════════════════════════════════════
# run_sweep — tiny grid smoke tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Triggers Numba compilation once for the whole test class.
        cls.result = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.2, 0.8, 4),
            Nitera=30,
            Nmap=150,
            n_initial=3,
            seed=42,
            warmup=True,
        )

    def test_result_type_and_shape(self):
        r = self.result
        self.assertIsInstance(r, SweepResult)
        self.assertEqual(r.h.shape, (3, 4))
        self.assertEqual(r.h_std.shape, (3, 4))
        self.assertEqual(r.orders.shape, (3,))
        self.assertEqual(r.cutoffs.shape, (4,))

    def test_metadata_populated(self):
        meta = self.result.metadata
        self.assertEqual(meta["Nitera"], 30)
        self.assertEqual(meta["Nmap"], 150)
        self.assertEqual(meta["n_initial"], 3)
        self.assertEqual(meta["seed"], 42)

    def test_display_name(self):
        self.assertEqual(self.result.display_name, "Hamming (lowpass)")

    def test_some_points_finite(self):
        """At least some grid points should yield a finite λ_max."""
        self.assertTrue(np.any(np.isfinite(self.result.h)))

    def test_seed_produces_similar_statistics(self):
        """Two runs with the same seed produce statistically similar results.

        Note: exact bit-for-bit determinism is not guaranteed because the
        inner kernel runs under Numba's ``prange``, which has its own
        per-thread RNG state that is not controlled by ``np.random.seed``.
        What we can check is that the averaged λ_max over ``n_initial``
        ICs stays in the same ball-park between runs — the usual
        stability criterion for a stochastic Lyapunov estimate.
        """
        r2 = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.2, 0.8, 4),
            Nitera=30,
            Nmap=150,
            n_initial=3,
            seed=42,
            warmup=False,
        )
        # Compare only entries that are finite in both runs
        both_finite = np.isfinite(self.result.h) & np.isfinite(r2.h)
        self.assertTrue(
            np.any(both_finite),
            "expected at least one grid point finite in both runs",
        )
        diff = np.abs(self.result.h[both_finite] - r2.h[both_finite])
        # For a tiny sweep (n_initial=3, Nmap=150) we allow generous slack.
        self.assertLess(
            float(np.max(diff)),
            0.5,
            "per-point λ_max difference should be small between runs",
        )


# ═══════════════════════════════════════════════════════════════════════════
# save_sweep / load_sweep round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestIO(unittest.TestCase):
    def test_round_trip_preserves_arrays(self):
        result = run_sweep(
            window="hann",
            filter_type="highpass",
            orders=np.arange(2, 4),
            cutoffs=np.linspace(0.2, 0.8, 3),
            Nitera=20,
            Nmap=100,
            n_initial=2,
            seed=0,
            warmup=False,
        )
        with TemporaryDirectory() as td:
            path = Path(td) / "sweep.npz"
            save_sweep(result, path)
            self.assertTrue(path.exists())

            loaded = load_sweep(path)
            np.testing.assert_array_equal(loaded.orders, result.orders)
            np.testing.assert_array_equal(loaded.cutoffs, result.cutoffs)
            self.assertEqual(loaded.window, "hann")
            self.assertEqual(loaded.filter_type, "highpass")
            self.assertEqual(loaded.display_name, "Hann (highpass)")

            # NaN-aware array equality
            both_nan = np.isnan(loaded.h) & np.isnan(result.h)
            equal = both_nan | (loaded.h == result.h)
            self.assertTrue(np.all(equal))

    def test_legacy_format_inferred_from_path(self):
        """An .npz without explicit window/filter_type fields should still
        load correctly when the parent directory follows the naming
        convention 'Window (filter)/'."""
        with TemporaryDirectory() as td:
            subdir = Path(td) / "Blackman (lowpass)"
            subdir.mkdir()
            path = subdir / "variables_lyapunov.npz"

            np.savez(
                path,
                h=np.array([[0.1, 0.2]]),
                h_desvio=np.array([[0.01, 0.02]]),
                wcorte=np.array([0.3, 0.7]),
                coef=np.array([5]),
            )

            loaded = load_sweep(path)
            self.assertEqual(loaded.window, "blackman")
            self.assertEqual(loaded.filter_type, "lowpass")


if __name__ == "__main__":
    unittest.main()

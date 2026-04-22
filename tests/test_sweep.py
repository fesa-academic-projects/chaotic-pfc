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

    def test_determinism_with_seed(self):
        """Two runs with the same seed produce byte-identical arrays.

        This is stronger than just "similar" — because every stochastic
        sample used by the kernel is now generated on the Python side via
        ``np.random.seed``, ``_precompute_perturbations`` yields the same
        array twice, and the kernel turns that into the same outputs.
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
        # NaN-aware exact equality: identical NaN mask AND identical
        # values on every finite cell.
        nan_mask_1 = np.isnan(self.result.h)
        nan_mask_2 = np.isnan(r2.h)
        np.testing.assert_array_equal(nan_mask_1, nan_mask_2)
        np.testing.assert_array_equal(
            self.result.h[~nan_mask_1],
            r2.h[~nan_mask_2],
        )
        # Same thing for the standard deviation array.
        np.testing.assert_array_equal(
            np.isnan(self.result.h_std),
            np.isnan(r2.h_std),
        )
        np.testing.assert_array_equal(
            self.result.h_std[~np.isnan(self.result.h_std)],
            r2.h_std[~np.isnan(r2.h_std)],
        )

    def test_different_seeds_produce_different_results(self):
        """Sanity check that the seed actually does anything — without
        this, we can't tell apart 'deterministic kernel' from 'kernel
        always outputting the same thing regardless of seed'."""
        r_other = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.2, 0.8, 4),
            Nitera=30,
            Nmap=150,
            n_initial=3,
            seed=99,  # different from self.result's seed=42
            warmup=False,
        )
        finite_both = np.isfinite(self.result.h) & np.isfinite(r_other.h)
        self.assertTrue(
            np.any(finite_both),
            "at least one grid point should be finite in both runs",
        )
        self.assertFalse(
            np.array_equal(self.result.h[finite_both], r_other.h[finite_both]),
            "different seeds should produce different λ_max values",
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

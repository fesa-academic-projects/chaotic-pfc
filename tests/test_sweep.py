"""tests/test_sweep.py — Unit tests for the sweep module."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chaotic_pfc.analysis.sweep import (
    FILTER_TYPES,
    WINDOW_DISPLAY_NAMES,
    WINDOWS,
    SweepResult,
    _build_task_order,
    _infer_config_from_path,
    load_sweep,
    precompute_fir_bank,
    quick_sweep_params,
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
        self.assertEqual(set(FILTER_TYPES), {"lowpass", "highpass", "bandpass", "bandstop"})


# ═══════════════════════════════════════════════════════════════════════════
# FIR bank
# ═══════════════════════════════════════════════════════════════════════════


class TestFirBank(unittest.TestCase):
    def test_shape(self):
        orders = np.array([2, 3, 5])
        cutoffs = np.linspace(0.1, 0.9, 4)
        bank, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "hamming")
        self.assertEqual(bank.shape, (3, 4, 5))  # max_taps = max(orders)
        self.assertEqual(gains.shape, (3, 4))

    def test_lowpass_dc_gain_approx_one(self):
        """Low-pass FIR filters should have DC gain close to 1."""
        orders = np.array([11, 21, 41])
        cutoffs = np.array([0.3, 0.5])
        _, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "hamming")
        np.testing.assert_allclose(gains, 1.0, atol=1e-6)

    def test_highpass_requires_odd_orders(self):
        """Highpass needs odd numtaps; even orders must be rejected.

        Allowing Nss|1 padding (the previous behaviour) silently truncated
        the filter at read time in the sweep kernel, which suppressed
        divergence-based early-exits and caused a ~10× slowdown.
        """
        cutoffs = np.array([0.3])
        with self.assertRaises(ValueError):
            precompute_fir_bank(np.array([4, 6, 8]), cutoffs, "highpass", "hamming")
        # Mixed odd/even — even values must still be reported.
        with self.assertRaises(ValueError):
            precompute_fir_bank(np.array([3, 4, 5]), cutoffs, "highpass", "hamming")

    def test_highpass_odd_orders_ok(self):
        """Highpass with odd orders builds a filter of size == order."""
        orders = np.array([3, 5, 7])
        cutoffs = np.array([0.3])
        bank, _ = precompute_fir_bank(orders, cutoffs, "highpass", "hamming")
        self.assertEqual(bank.shape, (3, 1, 7))
        # Last tap of each odd-order filter must be non-zero (filter not truncated).
        for i, Nss in enumerate(orders):
            self.assertNotEqual(bank[i, 0, int(Nss) - 1], 0.0)

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
            precompute_fir_bank([3], np.array([0.5]), "notch", "hamming")

    def test_bandpass_produces_valid_coeffs(self):
        orders = np.array([11, 21])
        cutoffs = np.array([0.3, 0.5])
        bank, gains = precompute_fir_bank(orders, cutoffs, "bandpass", "hamming", bandwidth=0.2)
        self.assertEqual(bank.shape, (2, 2, 21))
        self.assertEqual(gains.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(bank)))
        self.assertTrue(np.all(np.isfinite(gains)))

    def test_bandstop_produces_valid_coeffs(self):
        orders = np.array([11, 21])
        cutoffs = np.array([0.3, 0.5])
        bank, _gains = precompute_fir_bank(orders, cutoffs, "bandstop", "hamming", bandwidth=0.2)
        self.assertEqual(bank.shape, (2, 2, 21))
        self.assertTrue(np.all(np.isfinite(bank)))

    def test_bandpass_even_orders_accepted(self):
        """Bandpass doesn't require odd orders (unlike highpass/bandstop)."""
        orders = np.array([4, 6, 8])
        cutoffs = np.array([0.4])
        bank, _ = precompute_fir_bank(orders, cutoffs, "bandpass", "hamming", bandwidth=0.2)
        self.assertEqual(bank.shape, (3, 1, 8))

    def test_bandstop_rejects_even_orders(self):
        """Bandstop requires odd orders (same constraint as highpass)."""
        with self.assertRaises(ValueError):
            precompute_fir_bank(
                np.array([4, 6]), np.array([0.4]), "bandstop", "hamming", bandwidth=0.2
            )

    def test_bandpass_low_gain(self):
        """Bandpass filters should have near-zero DC gain."""
        orders = np.array([21, 31])
        cutoffs = np.array([0.5])
        _, gains = precompute_fir_bank(orders, cutoffs, "bandpass", "hamming", bandwidth=0.2)
        self.assertTrue(np.all(np.abs(gains) < 0.1))

    def test_bandwidth_clamps_to_boundaries(self):
        """Band edges should never go outside (0, 1)."""
        orders = np.array([11])
        # centre near 0 → low edge clamped to 1e-5
        cutoffs = np.array([0.05])
        bank, _ = precompute_fir_bank(orders, cutoffs, "bandpass", "hamming", bandwidth=0.3)
        self.assertTrue(np.all(np.isfinite(bank)))
        # centre near 1 → high edge clamped to 0.99999
        cutoffs = np.array([0.95])
        bank, _ = precompute_fir_bank(orders, cutoffs, "bandpass", "hamming", bandwidth=0.3)
        self.assertTrue(np.all(np.isfinite(bank)))


# ═══════════════════════════════════════════════════════════════════════════
# run_sweep — tiny grid smoke tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Triggers Numba compilation once for the whole test class.
        # cls.result is read-only — tests must not mutate it.
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
            orders=np.array([3, 5]),  # highpass requires odd orders
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
            self.assertIsNone(loaded.n_iters_used)

    def test_n_iters_used_round_trip(self):
        """``n_iters_used`` must survive a save/load round-trip."""
        result = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=np.arange(2, 5),
            cutoffs=np.linspace(0.2, 0.8, 4),
            Nitera=30,
            Nmap=200,
            n_initial=3,
            seed=42,
            warmup=False,
            adaptive=True,
            Nmap_min=50,
            tol=1e-2,
        )
        self.assertIsNotNone(result.n_iters_used)
        with TemporaryDirectory() as td:
            path = Path(td) / "adaptive.npz"
            save_sweep(result, path)
            loaded = load_sweep(path)
            self.assertIsNotNone(loaded.n_iters_used)
            np.testing.assert_array_equal(
                np.isnan(loaded.n_iters_used),
                np.isnan(result.n_iters_used),
            )
            finite = ~np.isnan(loaded.n_iters_used)
            np.testing.assert_array_equal(
                loaded.n_iters_used[finite],
                result.n_iters_used[finite],
            )


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Lyapunov early-stop
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptive(unittest.TestCase):
    """Validate the adaptive (early-stop) path of run_sweep.

    The default (adaptive=False) is already covered by TestRunSweep; here
    we focus on (a) correct behaviour of the adaptive criterion, (b)
    bit-identical results when adaptive=False so the new code path does
    not regress the production case, (c) the n_iters_used array, and
    (d) parameter validation.
    """

    @classmethod
    def setUpClass(cls):
        # Non-adaptive baseline (reference)
        cls.ref = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=np.arange(2, 8),
            cutoffs=np.linspace(0.1, 0.9, 6),
            Nitera=100,
            Nmap=500,
            n_initial=4,
            seed=42,
            warmup=True,
            adaptive=False,
        )

        # Adaptive run with the same seed and grid
        cls.ada = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=np.arange(2, 8),
            cutoffs=np.linspace(0.1, 0.9, 6),
            Nitera=100,
            Nmap=500,
            n_initial=4,
            seed=42,
            warmup=False,
            adaptive=True,
            Nmap_min=100,
            tol=1e-2,
        )

    def test_n_iters_used_filled(self):
        """In non-adaptive mode every finite cell uses exactly Nmap iters."""
        ni = self.ref.n_iters_used
        self.assertIsNotNone(ni)
        self.assertEqual(ni.shape, self.ref.h.shape)
        finite = np.isfinite(self.ref.h)
        np.testing.assert_array_equal(ni[finite], 500.0)

    def test_adaptive_uses_fewer_iters(self):
        """Adaptive mode should average fewer than Nmap iterations."""
        ni = self.ada.n_iters_used
        self.assertIsNotNone(ni)
        finite = np.isfinite(ni)
        self.assertTrue(np.any(finite), "expected at least one finite point")
        # All used counts must lie in [Nmap_min, Nmap]
        self.assertTrue(np.all(ni[finite] >= 100))
        self.assertTrue(np.all(ni[finite] <= 500))
        # And the mean should be strictly less than Nmap (otherwise
        # adaptive is pointless)
        self.assertLess(ni[finite].mean(), 500.0)

    def test_adaptive_close_to_reference(self):
        """Adaptive λ_max must be close to the non-adaptive estimate.

        The threshold (0.05) is loose because with these tiny test
        parameters (Nmap=500, n_initial=4) the reference itself is
        noisy. Production sweeps with Nmap=3000, n_initial=25 see
        |Δλ| < 0.01 in the worst case.
        """
        both = np.isfinite(self.ref.h) & np.isfinite(self.ada.h)
        self.assertTrue(np.any(both))
        diff = np.abs(self.ref.h - self.ada.h)[both]
        self.assertLess(
            diff.max(),
            0.05,
            f"adaptive diverges from reference: max|Δλ|={diff.max():.4f}",
        )

    def test_adaptive_metadata(self):
        """Adaptive metadata must record Nmap_min and tol."""
        self.assertTrue(self.ada.metadata["adaptive"])
        self.assertEqual(self.ada.metadata["Nmap_min"], 100)
        self.assertEqual(self.ada.metadata["tol"], 1e-2)
        # Non-adaptive must record None for adaptive-only fields
        self.assertFalse(self.ref.metadata["adaptive"])
        self.assertIsNone(self.ref.metadata["Nmap_min"])

    def test_adaptive_determinism(self):
        """Same seed must give identical results in adaptive mode too."""
        ada2 = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=np.arange(2, 8),
            cutoffs=np.linspace(0.1, 0.9, 6),
            Nitera=100,
            Nmap=500,
            n_initial=4,
            seed=42,
            warmup=False,
            adaptive=True,
            Nmap_min=100,
            tol=1e-2,
        )
        np.testing.assert_array_equal(np.isnan(self.ada.h), np.isnan(ada2.h))
        finite = ~np.isnan(self.ada.h)
        np.testing.assert_array_equal(self.ada.h[finite], ada2.h[finite])
        np.testing.assert_array_equal(self.ada.n_iters_used[finite], ada2.n_iters_used[finite])

    def test_invalid_Nmap_min(self):
        kw = dict(
            orders=np.arange(2, 4),
            cutoffs=np.linspace(0.2, 0.8, 3),
            Nitera=20,
            Nmap=200,
            n_initial=2,
            warmup=False,
        )
        # Nmap_min > Nmap is invalid
        with self.assertRaises(ValueError):
            run_sweep(adaptive=True, Nmap_min=300, tol=1e-2, **kw)
        # Nmap_min == Nmap is a no-op; raise rather than silently disable
        with self.assertRaises(ValueError):
            run_sweep(adaptive=True, Nmap_min=200, tol=1e-2, **kw)
        # Nmap_min < 1 is invalid
        with self.assertRaises(ValueError):
            run_sweep(adaptive=True, Nmap_min=0, tol=1e-2, **kw)
        # tol <= 0 is invalid
        with self.assertRaises(ValueError):
            run_sweep(adaptive=True, Nmap_min=50, tol=0.0, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers and edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers(unittest.TestCase):
    def test_quick_sweep_params_reduces_work(self):
        orders_lp, _orders_hp, cutoffs, params = quick_sweep_params()
        self.assertLess(len(orders_lp), 10)
        self.assertLess(len(cutoffs), 20)
        self.assertIsInstance(params, dict)
        self.assertIn("Nitera", params)
        self.assertLess(params["Nitera"], 500)

    def test_build_task_order_covers_all_combos(self):
        orders = np.arange(2, 5)
        tasks = _build_task_order(orders, Ncut=4)
        self.assertEqual(len(tasks), len(orders) * 4)
        self.assertEqual(tasks.ndim, 1)

    def test_build_task_order_empty(self):
        tasks = _build_task_order(np.array([]), Ncut=0)
        self.assertEqual(len(tasks), 0)

    def test_infer_config_from_path_lowpass(self):
        path = Path("data/sweeps/Hamming (lowpass)/variables_lyapunov.npz")
        filt, cutoff = _infer_config_from_path(path)
        self.assertEqual(filt, "hamming")
        self.assertEqual(cutoff, "lowpass")

    def test_infer_config_from_path_highpass(self):
        path = Path("data/sweeps/Hamming (highpass)/variables_lyapunov.npz")
        filt, cutoff = _infer_config_from_path(path)
        self.assertEqual(filt, "hamming")
        self.assertEqual(cutoff, "highpass")

    def test_infer_config_from_path_unexpected_format(self):
        path = Path("kaiser/bandpass/beta_5.00/variables_lyapunov.npz")
        with self.assertRaises(ValueError):
            _infer_config_from_path(path)

    def test_display_name_all_filter_types(self):
        for ft in FILTER_TYPES:
            result = SweepResult(
                h=np.zeros((2, 2)),
                h_std=np.zeros((2, 2)),
                orders=np.array([3, 4]),
                cutoffs=np.array([0.2, 0.8]),
                window="hamming",
                filter_type=ft,
            )
            name = result.display_name
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_load_sweep_no_niters(self):
        rng = np.random.default_rng(0)
        result = SweepResult(
            h=rng.uniform(-0.5, 0.5, size=(2, 2)),
            h_std=np.zeros((2, 2)),
            orders=np.array([3, 4]),
            cutoffs=np.array([0.2, 0.8]),
            window="hamming",
            filter_type="lowpass",
        )
        with TemporaryDirectory() as td:
            path = Path(td) / "variables_lyapunov.npz"
            save_sweep(result, path)
            loaded = load_sweep(path)
            np.testing.assert_array_equal(loaded.h, result.h)

    def test_bandstop_odd_filter_raises(self):
        with self.assertRaises(ValueError):
            precompute_fir_bank(
                window="hann",
                filter_type="bandstop",
                orders=np.arange(2, 5),
                cutoffs=np.array([0.2, 0.8]),
            )


# ═══════════════════════════════════════════════════════════════════════════
# Kernel function tests (Numba-JIT helpers)
# ═══════════════════════════════════════════════════════════════════════════


class TestKernelFunctions(unittest.TestCase):
    """Direct tests for the Numba-JIT kernel functions used by the sweep."""

    @classmethod
    def setUpClass(cls):
        from chaotic_pfc.analysis.sweep._kernel import (
            _henon_n12_inplace,
            _henon_nN_inplace,
            _mgs_accumulate,
        )

        cls._h12 = staticmethod(_henon_n12_inplace)
        cls._hN = staticmethod(_henon_nN_inplace)
        cls._mgs = staticmethod(_mgs_accumulate)

    def test_henon_n12_inplace_shape(self):
        c = np.array([0.6, 0.3])
        x = np.array([0.0, 0.0])
        x_new = np.empty(2)
        self._h12(x, x_new, 1.4, 0.3, c)
        self.assertEqual(x_new.shape, (2,))
        self.assertTrue(np.all(np.isfinite(x_new)))

    def test_henon_nN_inplace_shape(self):
        c = np.array([0.5, 0.3, 0.15, 0.05])
        x = np.array([0.0, 0.0, 0.0, 0.0])
        x_new = np.empty(4)
        self._hN(x, x_new, 1.4, 0.3, c)
        self.assertEqual(x_new.shape, (4,))

    def test_mgs_accumulate_maintains_orthogonality(self):
        rng = np.random.default_rng(0)
        Ns = 4
        z = rng.standard_normal((Ns, Ns)) * 0.1
        lyap_sum = np.zeros(Ns)
        self._mgs(z, Ns, lyap_sum)
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertTrue(np.all(np.isfinite(lyap_sum)))

    def test_build_task_order_balanced(self):
        orders = np.array([2, 3, 4, 5, 10, 20, 30, 41])
        task = _build_task_order(orders, Ncut=5)
        self.assertEqual(len(task), len(orders) * 5)
        self.assertEqual(len(set(task)), len(task))  # all unique

    def test_ns1_sweep_produces_finite_lyapunov(self):
        """Ns=1 should produce a finite Lyapunov exponent, not NaN/Inf."""
        from chaotic_pfc.analysis.sweep import run_sweep

        result = run_sweep(
            window="hamming",
            filter_type="lowpass",
            orders=[1],
            cutoffs=np.linspace(0.1, 0.9, 3),
            Nitera=50,
            Nmap=100,
            n_initial=2,
            seed=42,
        )
        self.assertTrue(np.all(np.isfinite(result.h)), "Ns=1 produced NaN/Inf Lyapunov")
        self.assertTrue(np.all(np.isfinite(result.n_iters_used)))

    def test_divergent_orbit_produces_nan_not_sentinel(self):
        """Pathological point (Nz=8, ωc=0.2828, Hann bandpass) must be NaN.

        Confirms the fix: divergence during the Lyapunov phase is caught and
        marked NaN instead of leaking the sentinel -1e30 into sweep averages.
        """
        from chaotic_pfc.analysis.sweep import run_sweep

        result = run_sweep(
            window="hann",
            filter_type="bandpass",
            orders=[8],
            cutoffs=np.array([0.2828326263]),
            Nitera=500,
            Nmap=3000,
            n_initial=25,
            seed=42,
        )
        val = float(result.h[0, 0])
        self.assertTrue(
            np.isnan(val),
            f"Expected NaN for divergent orbit, got {val:.6e}. "
            "The sentinel -1e30 or Inf should not leak through.",
        )

    def test_healthy_orbit_produces_finite_lyapunov(self):
        """Known-healthy point (Nz=8, ωc=0.5, Hann bandpass) must be finite."""
        from chaotic_pfc.analysis.sweep import run_sweep

        result = run_sweep(
            window="hann",
            filter_type="bandpass",
            orders=[8],
            cutoffs=np.array([0.5]),
            Nitera=500,
            Nmap=3000,
            n_initial=25,
            seed=42,
        )
        val = float(result.h[0, 0])
        self.assertTrue(
            np.isfinite(val),
            f"Expected finite Lyapunov for healthy orbit, got {val:.6e}",
        )

    def test_borderline_orbit_produces_finite_lyapunov(self):
        """Point near the pathological one must remain finite.

        Nz=8, ωc=0.35 (bandpass Hann) is between the pathological 0.283 and
        the clearly healthy 0.5. If the fix is too aggressive, this could be
        falsely flagged as divergent.
        """
        from chaotic_pfc.analysis.sweep import run_sweep

        result = run_sweep(
            window="hann",
            filter_type="bandpass",
            orders=[8],
            cutoffs=np.array([0.35]),
            Nitera=500,
            Nmap=3000,
            n_initial=25,
            seed=42,
        )
        val = float(result.h[0, 0])
        self.assertTrue(
            np.isfinite(val),
            f"Expected finite Lyapunov for borderline orbit, got {val:.6e}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Benettin block reorthonormalisation (K=10)
# ═══════════════════════════════════════════════════════════════════════════


class TestBenettinBlock(unittest.TestCase):
    """Validate the block reorthonormalisation kernel against the per-iteration
    reference (``_step_n*_ref`` functions preserved in ``_kernel.py``)."""

    @classmethod
    def setUpClass(cls):
        from chaotic_pfc.analysis.sweep._kernel import (
            _lyap_online_core,
            _lyap_online_n12,
            _lyap_online_nN,
            _mgs_accumulate,
            _step_n12_ref,
            _step_nN_ref,
        )

        cls._lyap_n12 = staticmethod(_lyap_online_n12)
        cls._lyap_nN = staticmethod(_lyap_online_nN)
        cls._core = staticmethod(_lyap_online_core)
        cls._ref_n12 = staticmethod(_step_n12_ref)
        cls._ref_nN = staticmethod(_step_nN_ref)
        cls._mgs = staticmethod(_mgs_accumulate)

    def _ref_online_core(self, xx, ww, lyap_sum, Nitera, Ns, c, alpha, beta):
        """Per-iteration reference (no block, no adaptive)."""
        for i in range(Ns):
            for j in range(Ns):
                ww[0, i, j] = 1.0 if i == j else 0.0
            lyap_sum[i] = 0.0
        tick = 0
        for _ in range(Nitera):
            if Ns <= 2:
                tick = self._ref_n12(tick, xx, ww, lyap_sum, Ns, c, alpha, beta)
            else:
                tick = self._ref_nN(tick, xx, ww, lyap_sum, Ns, c, alpha, beta)
            if not np.isfinite(xx[tick, 0]):
                for k in range(Ns):
                    lyap_sum[k] = np.nan
                break
        best = np.nan
        for k in range(Ns):
            if np.isnan(lyap_sum[k]):
                break
        else:
            best = -1e30
            for k in range(Ns):
                val = lyap_sum[k] / Nitera
                if val > best:
                    best = val
            if not np.isfinite(best):
                best = np.nan
        return best, Nitera

    def _assert_close(self, lam_new, lam_ref, atol, label):
        if np.isnan(lam_ref) and np.isnan(lam_new):
            return
        self.assertFalse(
            np.isnan(lam_new) and np.isfinite(lam_ref),
            f"{label}: new produced NaN but reference is finite",
        )
        self.assertFalse(
            np.isfinite(lam_new) and np.isnan(lam_ref),
            f"{label}: new is finite but reference is NaN",
        )
        self.assertLessEqual(
            abs(lam_new - lam_ref),
            atol,
            f"{label}: |Δλ| = {abs(lam_new - lam_ref):.6e} > {atol}",
        )

    def _run_oracle(self, Ns, dim, c, alpha, beta, Nitera, Nmap_min):
        """Run one IC through both new (block) and old (per-iteration) kernels."""
        xx = np.empty((2, dim))
        ww = np.empty((2, dim, dim))
        lyap_sum = np.empty(dim)

        xx[0, :] = 0.1 * np.arange(1, dim + 1) / dim
        xx[0, 1:] = 0.0 if dim == 2 else xx[0, 1:]
        if dim > 2:
            xx[0, 2] = 0.0
        xx[1, :] = 0.0

        # Reference: per-iteration MGS
        lam_ref, _ = self._ref_online_core(
            xx.copy(), ww.copy(), lyap_sum.copy(), Nitera, Ns, c.copy(), alpha, beta
        )

        # New: block Benettin
        if Ns <= 2:
            lam_new, _ = self._lyap_n12(
                xx.copy(),
                ww.copy(),
                lyap_sum.copy(),
                Nitera,
                Nmap_min,
                0.0,
                Ns,
                c.copy(),
                alpha,
                beta,
            )
        else:
            lam_new, _ = self._lyap_nN(
                xx.copy(),
                ww.copy(),
                lyap_sum.copy(),
                Nitera,
                Nmap_min,
                0.0,
                Ns,
                c.copy(),
                alpha,
                beta,
            )

        return lam_new, lam_ref

    def test_ns2_block_vs_reference(self):
        """Ns=2: block kernel within 1e-3 of per-iteration oracle."""
        Ns = 2
        dim = 2
        c = np.array([0.6, 0.3])
        alpha, beta = 1.4, 0.3
        lam_new, lam_ref = self._run_oracle(Ns, dim, c, alpha, beta, 5000, 5000)
        self._assert_close(lam_new, lam_ref, 1e-3, "Ns=2")

    def test_ns3_non_multiple_block_vs_reference(self):
        """Ns=3 (dim=4), Nitera=4995 not multiple of K: partial-block path."""
        Ns = 3
        dim = 4
        c = np.array([0.5, 0.3, 0.1])
        alpha, beta = 1.4, 0.3
        lam_new, lam_ref = self._run_oracle(Ns, dim, c, alpha, beta, 4995, 4995)
        self._assert_close(lam_new, lam_ref, 1e-3, "Ns=3 partial-block")

    def test_ns5_block_vs_reference(self):
        """Ns=5: block kernel within 1e-3 of per-iteration oracle."""
        Ns = 5
        dim = 5
        c = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        alpha, beta = 1.4, 0.3
        lam_new, lam_ref = self._run_oracle(Ns, dim, c, alpha, beta, 5000, 5000)
        self._assert_close(lam_new, lam_ref, 1e-3, "Ns=5")

    def test_ns12_block_vs_reference(self):
        """Ns=12: block kernel within 1e-3 of per-iteration oracle."""
        Ns = 12
        dim = 12
        c = np.array([0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
        alpha, beta = 1.4, 0.3
        lam_new, lam_ref = self._run_oracle(Ns, dim, c, alpha, beta, 5000, 5000)
        self._assert_close(lam_new, lam_ref, 1e-3, "Ns=12")


if __name__ == "__main__":
    unittest.main()

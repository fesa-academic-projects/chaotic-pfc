"""Sweep orchestration and adaptive iteration."""

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


class TestRunSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Triggers Numba compilation once for the whole test class.
        # cls.result is read-only — tests must not mutate it.
        cls.result = run_sweep(
            window="hamming",
            filter_type="lowpass",
            num_taps_grid=np.arange(2, 5),
            cutoff_grid=np.linspace(0.2, 0.8, 4),
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
        self.assertEqual(r.num_taps_grid.shape, (3,))
        self.assertEqual(r.cutoff_grid.shape, (4,))

    def test_metadata_populated(self):
        meta = self.result.metadata
        self.assertEqual(meta["Nitera"], 30)
        self.assertEqual(meta["Nmap"], 150)
        self.assertEqual(meta["n_initial"], 3)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["bandwidth"], 0.05)

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
            num_taps_grid=np.arange(2, 5),
            cutoff_grid=np.linspace(0.2, 0.8, 4),
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
            num_taps_grid=np.arange(2, 5),
            cutoff_grid=np.linspace(0.2, 0.8, 4),
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
            num_taps_grid=np.arange(2, 8),
            cutoff_grid=np.linspace(0.1, 0.9, 6),
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
            num_taps_grid=np.arange(2, 8),
            cutoff_grid=np.linspace(0.1, 0.9, 6),
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
            num_taps_grid=np.arange(2, 8),
            cutoff_grid=np.linspace(0.1, 0.9, 6),
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
            num_taps_grid=np.arange(2, 4),
            cutoff_grid=np.linspace(0.2, 0.8, 3),
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


class TestWarmupKernel(unittest.TestCase):
    def test_run_sweep_with_1_and_2_cutoffs_and_warmup(self):
        for ncut in (1, 2):
            cutoffs = np.linspace(0.25, 0.5, ncut)
            res_warm = run_sweep(
                "hamming",
                "lowpass",
                num_taps_grid=np.array([2, 3]),
                cutoff_grid=cutoffs,
                Nitera=20,
                Nmap=50,
                n_initial=2,
                seed=42,
                warmup=True,
            )
            res_no_warm = run_sweep(
                "hamming",
                "lowpass",
                num_taps_grid=np.array([2, 3]),
                cutoff_grid=cutoffs,
                Nitera=20,
                Nmap=50,
                n_initial=2,
                seed=42,
                warmup=False,
            )
            self.assertEqual(res_warm.h.shape, (2, ncut))
            np.testing.assert_array_equal(res_warm.h, res_no_warm.h)

    def test_warmup_passes_int64_and_inbounds_task_order(self):
        from unittest.mock import patch

        import chaotic_pfc.analysis.sweep._orchestration as orch

        captured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        real_kernel = orch._sweep_kernel

        def spy_kernel(num_taps_grid, cutoff_grid, *args):
            task_order = args[3]
            captured.append((num_taps_grid.copy(), cutoff_grid.copy(), task_order.copy()))
            return real_kernel(num_taps_grid, cutoff_grid, *args)

        with patch.object(orch, "_sweep_kernel", side_effect=spy_kernel):
            run_sweep(
                "hamming",
                "lowpass",
                num_taps_grid=np.array([2, 3, 4]),
                cutoff_grid=np.array([0.25, 0.5]),
                Nitera=10,
                Nmap=50,
                n_initial=2,
                seed=42,
                warmup=True,
            )

        self.assertEqual(len(captured), 2)
        warm_orders, warm_cutoffs, warm_task_order = captured[0]
        self.assertEqual(warm_orders.dtype, np.int64)
        self.assertEqual(len(warm_cutoffs), 2)
        self.assertTrue(np.all(warm_task_order >= 0))
        self.assertTrue(np.all(warm_task_order < len(warm_orders) * len(warm_cutoffs)))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers and edge cases
# ═══════════════════════════════════════════════════════════════════════════

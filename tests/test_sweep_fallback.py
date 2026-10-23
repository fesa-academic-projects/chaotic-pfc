"""Hybrid fallback for marginal cells."""

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


class TestHybridFallback(unittest.TestCase):
    """Validate the hybrid estimator: v-only for strong cells, full-spectrum
    fallback for cells near ``LAMBDA_CHAOS_THRESHOLD``."""

    def _marginal_cell_setup(self):
        """Return the known marginal cell (Ns=5, ωc=0.8182, Bartlett lowpass).

        Its single-vector mean is about 8.7e-3, inside the band centred on
        ``LAMBDA_CHAOS_THRESHOLD``.
        """
        from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

        num_taps_grid = np.array([5], dtype=np.int64)
        cutoff_grid = np.array([0.8181754545454546])
        bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "bartlett")
        pert = _precompute_perturbations(num_taps_grid, 1, 25, 42)
        Ns = 5
        c = bank[0, 0, :Ns]
        gain = gains[0, 0]
        alpha, beta = 1.4, 0.3
        disc = (1.0 - beta) ** 2 + 4.0 * alpha * (gain**2)
        p1 = 2.0 * alpha / ((1.0 - beta) + disc**0.5)
        return num_taps_grid, cutoff_grid, bank, gains, pert, Ns, c, alpha, beta, p1, gain * p1

    def test_fallback_returns_full_spectrum_value(self):
        """Marginal cell: hybrid output equals direct full-spectrum call (assertEqual)."""
        from chaotic_pfc.analysis.sweep._kernel import (
            _henon_nN_inplace,
            _lyap_online_nN,
            _sweep_kernel,
        )

        num_taps_grid, cutoff_grid, bank, gains, pert, Ns, c, alpha, beta, p1, p3 = (
            self._marginal_cell_setup()
        )
        p2 = p1
        Nmap, Nitera = 3000, 500
        dim = Ns

        # Direct full-spectrum: average over 25 ICs
        lam_f_samples = np.full(25, np.nan)
        for ci in range(25):
            noise = pert[0, 0, ci, :Ns]
            xx = np.empty((2, dim))
            xx[0, 0] = 0.1 * noise[0] + p1
            xx[0, 1] = 0.1 * noise[1] + p2
            xx[0, 2] = 0.1 * noise[2] + p3
            for k in range(3, Ns):
                xx[0, k] = 0.1 * noise[k] + p1
            tick = 0
            for it_t in range(Nitera):
                _henon_nN_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
                tick = 1 - tick
                if (it_t & 63) == 63 and not np.isfinite(xx[tick, 0]):
                    break
            if tick == 1:
                for k in range(Ns):
                    xx[0, k] = xx[1, k]
            if not np.isfinite(xx[0, 0]):
                continue
            ww = np.empty((2, dim, dim))
            lf = np.empty(dim)
            lam_f, _ = _lyap_online_nN(
                xx.copy(), ww, lf, Nmap, Nmap, 0.0, Ns, c.copy(), alpha, beta
            )
            lam_f_samples[ci] = lam_f
        total_f = 0.0
        count_f = 0
        for v in lam_f_samples:
            if not np.isnan(v):
                total_f += v
                count_f += 1
        mean_f = total_f / count_f

        # Hybrid sweep kernel
        task = np.zeros(1, dtype=np.int64)
        h, _, _, n_fallback, _ = _sweep_kernel(
            num_taps_grid.astype(np.float64),
            cutoff_grid,
            bank,
            gains,
            pert,
            task,
            Nitera,
            Nmap,
            Nmap,
            0.0,
            25,
            alpha,
            beta,
        )

        self.assertEqual(n_fallback[0, 0], 1, "Marginal cell must trigger fallback")
        self.assertEqual(
            h[0, 0],
            mean_f,
            f"Hybrid output {h[0, 0]:.20e} must match full spectrum {mean_f:.20e}",
        )

    def test_near_zero_negative_cell_does_not_fall_back(self):
        """The old band centred on zero no longer triggers near zero."""
        from chaotic_pfc.analysis.sweep._kernel import _sweep_kernel
        from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

        num_taps_grid = np.array([10], dtype=np.int64)
        cutoff_grid = np.array([0.212127])
        bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "hamming")
        pert = _precompute_perturbations(num_taps_grid, 1, 25, 42)
        task = np.array([0], dtype=np.int64)
        h, _, _, n_fallback, v_mean = _sweep_kernel(
            num_taps_grid.astype(np.float64),
            cutoff_grid,
            bank,
            gains,
            pert,
            task,
            500,
            3000,
            3000,
            0.0,
            25,
            1.4,
            0.3,
        )
        self.assertEqual(n_fallback[0, 0], 0)
        self.assertEqual(h[0, 0], v_mean[0, 0])
        self.assertLess(h[0, 0], 0.0)

    def test_no_fallback_on_non_marginal_cell(self):
        """A cell far from the threshold uses the v-only result directly."""
        from chaotic_pfc.analysis.sweep._kernel import _sweep_kernel
        from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

        num_taps_grid = np.array([3], dtype=np.int64)
        cutoff_grid = np.array([0.5])
        bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "hamming")
        pert = _precompute_perturbations(num_taps_grid, 1, 4, 42)
        task = np.array([0], dtype=np.int64)

        _, _, _, n_fallback, _ = _sweep_kernel(
            num_taps_grid.astype(np.float64),
            cutoff_grid,
            bank,
            gains,
            pert,
            task,
            50,
            200,
            200,
            0.0,
            4,
            1.4,
            0.3,
        )
        # lambda_1 for Ns=3, wc=0.5 is well above the threshold; no fallback
        self.assertEqual(n_fallback[0, 0], 0)


if __name__ == "__main__":
    unittest.main()

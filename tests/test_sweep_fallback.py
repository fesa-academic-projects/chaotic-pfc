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
    fallback for marginal cells (|λ| < 5e-3)."""

    def _marginal_cell_setup(self):
        """Return (orders, cutoffs, bank, gains, pert, Ns, c, alpha, beta, p1, p2, p3)
        for the known marginal cell (order=10, ωc=0.212127, hamming lowpass)."""
        from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

        orders = np.array([10], dtype=np.int64)
        cutoffs = np.array([0.212127])
        bank, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "hamming")
        pert = _precompute_perturbations(orders, 1, 25, 42)
        Ns = 10
        c = bank[0, 0, :Ns]
        gain = gains[0, 0]
        alpha, beta = 1.4, 0.3
        disc = (1.0 - beta) ** 2 + 4.0 * alpha * (gain**2)
        p1 = 2.0 * alpha / ((1.0 - beta) + disc**0.5)
        return orders, cutoffs, bank, gains, pert, Ns, c, alpha, beta, p1, gain * p1

    def test_fallback_returns_full_spectrum_value(self):
        """Marginal cell: hybrid output equals direct full-spectrum call (assertEqual)."""
        from chaotic_pfc.analysis.sweep._kernel import (
            _henon_nN_inplace,
            _lyap_online_nN,
            _sweep_kernel,
        )

        orders, cutoffs, bank, gains, pert, Ns, c, alpha, beta, p1, p3 = self._marginal_cell_setup()
        p2 = p1
        Nmap, Nitera = 3000, 500
        dim = 10

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
        h, _, _, n_fallback = _sweep_kernel(
            orders.astype(np.float64),
            cutoffs,
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

    def test_no_fallback_on_non_marginal_cell(self):
        """A cell with |λ| > 5e-3 uses the v-only result directly."""
        from chaotic_pfc.analysis.sweep._kernel import _sweep_kernel
        from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

        orders = np.array([3], dtype=np.int64)
        cutoffs = np.array([0.5])
        bank, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "hamming")
        pert = _precompute_perturbations(orders, 1, 4, 42)
        task = np.array([0], dtype=np.int64)

        _, _, _, n_fallback = _sweep_kernel(
            orders.astype(np.float64),
            cutoffs,
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
        # |λ| for Ns=3, ωc=0.5 is well above 5e-3; no fallback
        self.assertEqual(n_fallback[0, 0], 0)


if __name__ == "__main__":
    unittest.main()

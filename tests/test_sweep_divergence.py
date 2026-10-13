"""Single-vector exactness and the divergence rule."""

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


class TestSingleVectorExact(unittest.TestCase):
    """Validate the single-vector Benettin estimator (``_lyap_online_core_v``).

    For most (Ns, c) pairs the estimator is bit-identical to the
    full-spectrum ``_lyap_online_core`` because vector 0 is MGS-independent
    and carries λ_max under exact arithmetic.

    In ~0.4 % of cells the finite Benettin block (K=10) causes adjacent
    exponents to cross, so ``lyap_sum[k] > lyap_sum[0]`` for some k > 0.
    Those cells are documented at the sweep level; at the single-IC level
    the sign is always preserved.  The tests below choose (Ns, c) pairs
    where crossing does not occur, confirming bit-identity for the standard
    case.
    """

    @classmethod
    def setUpClass(cls):
        from chaotic_pfc.analysis.sweep._kernel import (
            _lyap_online_core_v,
            _lyap_online_n12,
            _lyap_online_nN,
        )

        cls._v = staticmethod(_lyap_online_core_v)
        cls._full_n12 = staticmethod(_lyap_online_n12)
        cls._full_nN = staticmethod(_lyap_online_nN)

    def _run_exact_comparison(self, Ns, dim, c, alpha, beta, Nitera, Nmap_min):
        """Run both kernels on identical ICs; return (lam_v, lam_full)."""
        xx_v = np.empty((2, dim))
        xx_full = np.empty((2, dim))
        vv = np.empty((2, dim))
        ww = np.empty((2, dim, dim))
        lyap_sum_v = np.empty(1)
        lyap_sum_full = np.empty(dim)

        ic = 0.1 * np.arange(1, dim + 1) / dim
        xx_v[0, :] = ic
        xx_full[0, :] = ic.copy()
        if dim == 2:
            xx_v[0, 1] = 0.0
            xx_full[0, 1] = 0.0
        elif dim > 2:
            xx_v[0, 2] = 0.0
            xx_full[0, 2] = 0.0
        xx_v[1, :] = 0.0
        xx_full[1, :] = 0.0

        # Full-spectrum Benettin-K
        if Ns <= 2:
            lam_full, _ = self._full_n12(
                xx_full,
                ww,
                lyap_sum_full,
                Nitera,
                Nmap_min,
                0.0,
                Ns,
                c.copy(),
                alpha,
                beta,
            )
        else:
            lam_full, _ = self._full_nN(
                xx_full,
                ww,
                lyap_sum_full,
                Nitera,
                Nmap_min,
                0.0,
                Ns,
                c.copy(),
                alpha,
                beta,
            )

        # Single-vector estimator
        step_id = 0 if Ns <= 2 else 1
        lam_v, _ = self._v(
            xx_v,
            vv,
            lyap_sum_v,
            Nitera,
            Nmap_min,
            0.0,
            Ns,
            step_id,
            c.copy(),
            alpha,
            beta,
        )

        return lam_v, lam_full

    def test_ns2_exact(self):
        Ns = 2
        dim = 2
        c = np.array([0.6, 0.3])
        alpha, beta = 1.4, 0.3
        lam_v, lam_full = self._run_exact_comparison(Ns, dim, c, alpha, beta, 5000, 5000)
        if np.isnan(lam_full) and np.isnan(lam_v):
            return
        self.assertEqual(
            lam_v,
            lam_full,
            f"Ns=2: λ_max differs (v={lam_v:.20e}, full={lam_full:.20e})",
        )

    def test_ns3_exact(self):
        """Ns=3 (dim=4), Nitera=4995 — exercises partial-block path."""
        Ns = 3
        dim = 4
        c = np.array([0.5, 0.3, 0.1])
        alpha, beta = 1.4, 0.3
        lam_v, lam_full = self._run_exact_comparison(Ns, dim, c, alpha, beta, 4995, 4995)
        if np.isnan(lam_full) and np.isnan(lam_v):
            return
        self.assertEqual(
            lam_v,
            lam_full,
            f"Ns=3: λ_max differs (v={lam_v:.20e}, full={lam_full:.20e})",
        )

    def test_ns5_exact(self):
        Ns = 5
        dim = 5
        c = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        alpha, beta = 1.4, 0.3
        lam_v, lam_full = self._run_exact_comparison(Ns, dim, c, alpha, beta, 5000, 5000)
        if np.isnan(lam_full) and np.isnan(lam_v):
            return
        self.assertEqual(
            lam_v,
            lam_full,
            f"Ns=5: λ_max differs (v={lam_v:.20e}, full={lam_full:.20e})",
        )

    def test_ns12_exact(self):
        Ns = 12
        dim = 12
        c = np.array([0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
        alpha, beta = 1.4, 0.3
        lam_v, lam_full = self._run_exact_comparison(Ns, dim, c, alpha, beta, 5000, 5000)
        if np.isnan(lam_full) and np.isnan(lam_v):
            return
        self.assertEqual(
            lam_v,
            lam_full,
            f"Ns=12: λ_max differs (v={lam_v:.20e}, full={lam_full:.20e})",
        )

    def test_ns10_known_crossing_preserves_sign(self):
        """Known-discrepancy cell (order=10, lowpass Hamming, ωc≈0.2121).

        In the full sweep ~2 of 25 ICs have ``lyap_sum[1] > lyap_sum[0]``
        due to finite-block MGS.  This test verifies the discrepancy is
        expected: no single IC flips sign, and the v-only estimator tracks
        lyap_sum[0] while the full estimator picks max(lyap_sum[0..Ns-1]).
        """
        from chaotic_pfc.analysis.sweep._kernel import (
            _henon_nN_inplace,
            _lyap_online_core_v,
            _lyap_online_nN,
        )
        from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

        num_taps_grid = np.array([10], dtype=np.int64)
        cutoff_grid = np.array([0.212127])
        bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "hamming")
        pert = _precompute_perturbations(num_taps_grid, 1, 25, 42)

        Ns, dim = 10, 10
        c = bank[0, 0, :Ns]
        gain = gains[0, 0]
        alpha, beta = 1.4, 0.3
        Nitera, Nmap = 500, 3000

        disc = (1.0 - beta) ** 2 + 4.0 * alpha * (gain**2)
        p1 = 2.0 * alpha / ((1.0 - beta) + disc**0.5)
        p2, p3 = p1, gain * p1

        lam_v_all = np.empty(25)
        lam_full_all = np.empty(25)
        lyap0_all = np.empty(25)
        n_crossings = 0

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
                lam_v_all[ci] = np.nan
                lam_full_all[ci] = np.nan
                lyap0_all[ci] = np.nan
                continue

            # Full spectrum
            ww = np.empty((2, dim, dim))
            lf = np.empty(dim)
            lam_f, _ = _lyap_online_nN(
                xx.copy(), ww, lf, Nmap, Nmap, 0.0, Ns, c.copy(), alpha, beta
            )

            # Single vector
            vv = np.empty((2, dim))
            lv = np.empty(1)
            lam_v, _ = _lyap_online_core_v(
                xx.copy(), vv, lv, Nmap, Nmap, 0.0, Ns, 1, c.copy(), alpha, beta
            )

            lam_v_all[ci] = lam_v
            lam_full_all[ci] = lam_f
            lyap0_all[ci] = lv[0]

            # Check for sign flip
            if np.isfinite(lam_v) and np.isfinite(lam_f):
                if lf[1] > lf[0]:
                    n_crossings += 1
                self.assertEqual(
                    np.sign(lam_v),
                    np.sign(lam_f),
                    f"IC={ci}: sign flip at λ≈{lam_v:.3e} (v) vs {lam_f:.3e} (full)",
                )

        # At least 1 IC should have lyap_sum[1] > lyap_sum[0]
        self.assertGreater(
            n_crossings,
            0,
            "Expected at least one IC with lyap_sum[1] > lyap_sum[0] in known-discrepancy cell",
        )
        # The v-only mean may differ from full mean (documented limitation)
        # but individual ICs never flip sign (tested in the loop above).


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid single-vector / full-spectrum fallback
# ═══════════════════════════════════════════════════════════════════════════

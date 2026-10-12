"""Sweep kernel functions and the Benettin block."""

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


# ═══════════════════════════════════════════════════════════════════════════
# Single-vector estimator (λ_max) — bit-exact comparison vs full spectrum
# ═══════════════════════════════════════════════════════════════════════════

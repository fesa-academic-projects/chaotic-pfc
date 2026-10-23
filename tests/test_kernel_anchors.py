"""Anchor the sweep kernel in independent, literature and analytic values.

The sweeps use a single-tangent-vector estimator (Algorithm 1 of the
manuscript), which is the path exercised through :func:`run_sweep`. Both checks
below drive that path with ``N_s = 1`` (a single tap, so the FIR filter is a
unit DC gain and the kernel reduces to the bare two-dimensional Hénon map).

* Standard Hénon map, ``alpha = 1.4``, ``beta = 0.3``: the largest Lyapunov
  exponent is the value reported by Sprott (2003) and Wolf et al. (1985),
  ``lambda_1 ~ 0.419``. The tolerance is 5e-3 because the kernel estimates the
  exponent from a finite orbit of ``Nmap`` iterations; the deviation of the
  finite-time estimate is bounded by the convergence rate documented in the
  algorithm page (well below 1% at ``Nmap = 20000``).

* Analytic case ``alpha = 2``, ``beta = 0``: the map becomes
  ``x -> 2 - x^2``, topologically conjugate to the logistic map at ``r = 4``,
  for which the Lyapunov exponent is exactly ``ln 2``. The tolerance is 1e-3
  because the conjugate map preserves the exponent exactly and the remaining
  error is only the finite-time truncation.
"""

from __future__ import annotations

import math

import numpy as np

from chaotic_pfc.analysis.sweep import precompute_fir_bank, run_sweep


def _single_tap_sweep(alpha: float, beta: float) -> float:
    result = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=[1],
        cutoff_grid=[0.5],
        Nitera=1000,
        Nmap=20000,
        n_initial=10,
        adaptive=False,
        seed=42,
        alpha=alpha,
        beta=beta,
    )
    return float(result.h[0, 0])


def test_single_tap_is_the_unfiltered_map() -> None:
    """``N_s = 1`` reduces the filtered map to the bare Hénon map (DC gain 1)."""
    _, gains = precompute_fir_bank(np.array([1]), np.array([0.5]), "lowpass", "hamming")
    assert gains[0, 0] == 1.0


def test_kernel_standard_henon_matches_sprott() -> None:
    """The kernel reproduces the standard Hénon exponent (~0.419)."""
    lam = _single_tap_sweep(alpha=1.4, beta=0.3)
    assert abs(lam - 0.419) < 5e-3, f"lambda_1 = {lam}, expected ~0.419 (Sprott 2003)"


def test_kernel_analytic_quadratic_map_is_ln2() -> None:
    """``alpha = 2``, ``beta = 0`` is conjugate to the r=4 logistic map: ln 2."""
    lam = _single_tap_sweep(alpha=2.0, beta=0.0)
    assert abs(lam - math.log(2)) < 1e-3, f"lambda_1 = {lam}, expected ln 2 = {math.log(2)}"


def test_production_path_adaptive_single_vector() -> None:
    """Production estimator on the unfiltered map: adaptive, 25 ICs, seed 42.

    The reference values are 0.4185 (full-spectrum estimate already quoted in
    the manuscript) and the literature value 0.42.
    """
    result = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=[1],
        cutoff_grid=[0.5],
        Nitera=500,
        Nmap=3000,
        n_initial=25,
        adaptive=True,
        Nmap_min=500,
        tol=1e-3,
        seed=42,
    )
    mean = float(result.h[0, 0])
    assert abs(mean - 0.4185) / 0.4185 < 0.01, f"mean = {mean:.12f}"
    assert abs(mean - 0.42) / 0.42 < 0.01, f"mean = {mean:.12f}"
    assert result.metadata["n_fallback"] == 0
    assert result.n_iters_used is not None
    assert float(result.n_iters_used[0, 0]) < 3000.0

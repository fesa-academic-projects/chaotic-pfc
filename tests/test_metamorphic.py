"""Metamorphic and extended-precision reference tests.

Three properties are pinned here:

* the sum of the Lyapunov spectrum equals the mean of ``log|det J(x_n)|``
  along the orbit (``ln|beta|`` for the 2-D Hénon map);
* the cancellation-free positive fixed point matches a 50-digit ``mpmath``
  evaluation to within 4 ulp;
* permuting ``cutoff_grid`` permutes the columns of the sweep grid without
  changing any value, because the perturbation tensor is drawn in cutoff order.
"""

from __future__ import annotations

import numpy as np
import pytest

from chaotic_pfc import lyapunov_spectrum, run_sweep
from chaotic_pfc._formulas import fixed_point_positive

ULP = float(np.finfo(np.float64).eps)


def _henon2d(alpha: float = 1.4, beta: float = 0.3):
    def iterate(x):
        return np.array([alpha - x[0] ** 2 + beta * x[1], x[0]])

    def jacobian(x):
        return np.array([[-2.0 * x[0], beta], [1.0, 0.0]])

    return iterate, jacobian


def _henon_n(coeffs: np.ndarray, alpha: float = 1.4, beta: float = 0.3):
    """N-th order filtered Hénon map and its analytic Jacobian."""
    c = np.asarray(coeffs, dtype=float)
    Ns = len(c)

    def iterate(x):
        x0_new = alpha - x[2] ** 2 + beta * x[1]
        out = np.empty(Ns)
        out[0] = x0_new
        out[1] = x[0]
        out[2] = c[0] * x0_new + c[1] * x[0] + c[2] * x[1] + float(np.dot(c[3:], x[3:]))
        if Ns > 3:
            out[3] = x[1]
        if Ns > 4:
            out[4:] = x[3 : Ns - 1]
        return out

    def jacobian(x):
        J = np.zeros((Ns, Ns))
        J[0, 1] = beta
        J[0, 2] = -2.0 * x[2]
        J[1, 0] = 1.0
        J[2, 0] = c[1]
        J[2, 1] = c[0] * beta + c[2]
        J[2, 2] = -2.0 * c[0] * x[2]
        if Ns > 3:
            J[2, 3:] = c[3:]
            J[3, 1] = 1.0
        for k in range(4, Ns):
            J[k, k - 1] = 1.0
        return J

    return iterate, jacobian


def _mean_log_abs_det(iterate, jacobian, x0, n: int = 2000, discard: int = 500) -> float:
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(discard):
        x = iterate(x)
    logs = np.empty(n)
    for i in range(n):
        logs[i] = np.log(abs(float(np.linalg.det(jacobian(x)))))
        x = iterate(x)
    return float(np.mean(logs))


def test_spectrum_sum_equals_mean_log_det_henon2d() -> None:
    iterate, jacobian = _henon2d()
    x0 = np.array([0.1, 0.1])
    spectrum = lyapunov_spectrum(iterate, jacobian, x0, Nitera=5000, Ndiscard=1000)
    assert np.all(np.isfinite(spectrum))
    np.testing.assert_allclose(spectrum.sum(), np.log(0.3), atol=1e-6)
    mean_det = _mean_log_abs_det(iterate, jacobian, x0)
    np.testing.assert_allclose(spectrum.sum(), mean_det, atol=1e-2)


@pytest.mark.parametrize("num_taps", [3, 5, 9])
def test_spectrum_sum_equals_mean_log_det_filtered(num_taps: int) -> None:
    """The identity holds for the expanded N-th order filtered Hénon system."""
    rng = np.random.default_rng(42)
    coeffs = rng.random(num_taps)
    coeffs /= coeffs.sum()
    iterate, jacobian = _henon_n(coeffs)
    x0 = np.concatenate([rng.uniform(0.05, 0.4, size=num_taps)])
    spectrum = lyapunov_spectrum(iterate, jacobian, x0, Nitera=3000, Ndiscard=1000)
    assert np.all(np.isfinite(spectrum)), "orbit diverged for the test coefficients"
    mean_det = _mean_log_abs_det(iterate, jacobian, x0)
    np.testing.assert_allclose(
        spectrum.sum(),
        mean_det,
        atol=5e-3,
        err_msg=f"Ns={num_taps}: metamorphic identity failed",
    )


def test_fixed_point_positive_matches_mpmath() -> None:
    mpmath = pytest.importorskip("mpmath", reason="mpmath ships in the [test] extra")
    mpmath.mp.dps = 50
    alpha, beta = 1.4, 0.3
    for gain in (0.0, 1e-12, 1e-6, 0.5, 1.0, 2.0):
        disc = (1 - beta) ** 2 + 4 * alpha * gain**2
        expected = 2 * alpha / ((1 - beta) + mpmath.sqrt(disc))
        got = float(fixed_point_positive(gain, alpha, beta))
        rel = abs(got - float(expected)) / float(expected)
        assert rel <= 4 * ULP, f"G={gain:g}: relative error {rel:.3e} exceeds 4 ulp ({4 * ULP:.3e})"


def test_cutoff_grid_permutation_permutes_kernel_columns() -> None:
    """Reordering cutoffs with consistently reordered inputs permutes ``h`` exactly.

    The perturbation tensor is drawn in cutoff order, so with a fixed seed a
    permuted ``cutoff_grid`` receives a *different* (statistically equivalent)
    draw. The positional invariance that does hold, and that the kernel
    guarantees, is: permuting cutoffs, FIR coefficients and perturbations
    together permutes the ``h`` columns bit for bit.
    """
    from chaotic_pfc.analysis.sweep import precompute_fir_bank
    from chaotic_pfc.analysis.sweep._kernel import _build_task_order, _sweep_kernel
    from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

    orders = np.array([2, 3, 4], dtype=np.int64)
    cutoffs = np.array([0.15, 0.35, 0.55, 0.75])
    perm = np.array([2, 0, 3, 1])
    bank, gains = precompute_fir_bank(orders, cutoffs, "lowpass", "hamming")
    perturbations = _precompute_perturbations(orders, len(cutoffs), 2, seed=42)
    task_order = _build_task_order(orders, len(cutoffs))
    args = (20, 100, 100, 0.0, 2, 1.4, 0.3, 1)

    h, h_std, n_iters, _ = _sweep_kernel(
        orders, cutoffs, bank, gains, perturbations, task_order, *args
    )
    h_p, h_std_p, n_iters_p, _ = _sweep_kernel(
        orders,
        cutoffs[perm],
        bank[:, perm, :],
        gains[:, perm],
        perturbations[:, perm, :, :],
        task_order,
        *args,
    )
    both_nan = np.isnan(h[:, perm]) & np.isnan(h_p)
    assert np.all(both_nan | (h[:, perm] == h_p))
    both_nan_std = np.isnan(h_std[:, perm]) & np.isnan(h_std_p)
    assert np.all(both_nan_std | (h_std[:, perm] == h_std_p))
    both_nan_it = np.isnan(n_iters[:, perm]) & np.isnan(n_iters_p)
    assert np.all(both_nan_it | (n_iters[:, perm] == n_iters_p))


def test_run_sweep_is_deterministic_per_cutoff_position() -> None:
    """Same seed and same grid reproduce the grid bit for bit (positional seeds)."""
    common = dict(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=[2, 3],
        cutoff_grid=np.array([0.15, 0.35, 0.55]),
        Nitera=20,
        Nmap=100,
        n_initial=2,
        seed=42,
        warmup=False,
    )
    first = run_sweep(**common)
    second = run_sweep(**common)
    np.testing.assert_array_equal(np.isnan(first.h), np.isnan(second.h))
    finite = ~np.isnan(first.h)
    np.testing.assert_array_equal(first.h[finite], second.h[finite])

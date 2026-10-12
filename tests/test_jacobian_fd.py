"""tests/test_jacobian_fd.py: analytic Jacobian versus central finite differences.

The analytic side is the package's own linearisation, read directly from the
sweep kernel (``_propagate_n12`` / ``_propagate_nN``, which compute ``J @ W``
with ``W = I``). The numeric side is central finite differences of the
package's expanded map (``_henon_n12_inplace`` / ``_henon_nN_inplace``). The
FD step is swept and, per matrix entry, the smallest absolute error across
steps is kept, so truncation error (which shrinks with the step) is separated
from a genuine plateau (a real discrepancy).

Two extra checks close the chain: ``test_eq4_matches_package_map`` implements
equation (4) of the article literally in NumPy and compares it with the
package map, and the named ``N_s = 4`` dump prints one single evaluation
(coefficients, state, analytic matrix, numeric matrix, and the ``(3,2)``
entry against ``beta c_0`` and ``beta c_0 + c_2``).

The test imports the package and runs the Numba kernels, so it is marked
``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import firwin

from chaotic_pfc.analysis.sweep import _kernel as K
from chaotic_pfc.config import InternalFIRConfig

NS_VALUES = (1, 2, 3, 4, 5, 8, 12)
STEPS = (1e-4, 1e-5, 1e-6, 1e-7)
EARLY_UPTO = 80
EARLY_STRIDE = 5
ALPHA = 1.4
BETA = 0.3
TOL_ABS = 1e-9
TOL_REL = 1e-6
BOUND = 1e6


def _state_dim(Ns: int) -> int:
    return 2 if Ns <= 2 else max(Ns, 4)


def _random_coeffs(Ns: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.random(Ns)
    return c / c.sum()


def _real_coeffs(Ns: int, kind: str) -> np.ndarray | None:
    try:
        if kind == "lowpass":
            if Ns <= 2:
                return np.asarray(
                    firwin(Ns, 0.5, window="hamming", pass_zero=True, fs=2.0), dtype=float
                )
            return np.asarray(InternalFIRConfig(num_taps=Ns, cutoff=0.5).fir_coeffs(), dtype=float)
        if kind == "highpass":
            return np.asarray(
                firwin(Ns, 0.5, window="hamming", pass_zero=False, fs=2.0), dtype=float
            )
        if kind == "bandstop":
            return np.asarray(
                firwin(Ns, [0.3, 0.7], window="hamming", pass_zero=True, fs=2.0), dtype=float
            )
    except ValueError:
        return None
    return None


def _coeff_sets(Ns: int) -> list[tuple[str, np.ndarray]]:
    x0 = np.array([0.1, -0.05, 0.03, 0.02] + [0.0] * 20)[: _state_dim(Ns)]
    sets: list[tuple[str, np.ndarray]] = [
        (f"random seed={seed}", _random_coeffs(Ns, seed)) for seed in (0, 1, 2)
    ]
    for kind in ("lowpass", "highpass", "bandstop"):
        c = _real_coeffs(Ns, kind)
        if c is not None:
            sets.append((f"real {kind}", c))
    return sets, x0


def _step(x: np.ndarray, c: np.ndarray, Ns: int) -> np.ndarray:
    x_in = np.asarray(x, dtype=float).copy()
    x_out = np.empty_like(x_in)
    if Ns <= 2:
        K._henon_n12_inplace(x_in, x_out, ALPHA, BETA, c)
    else:
        K._henon_nN_inplace(x_in, x_out, ALPHA, BETA, c)
    return x_out


def _early_states(c: np.ndarray, Ns: int, x0: np.ndarray) -> tuple[list[np.ndarray], int]:
    dim = _state_dim(Ns)
    x = np.zeros(dim)
    x[: min(len(x0), dim)] = x0[: min(len(x0), dim)]
    good: list[np.ndarray] = []
    discarded = 0
    for step in range(1, EARLY_UPTO + 1):
        x = _step(x, c, Ns)
        if step % EARLY_STRIDE:
            continue
        if np.all(np.isfinite(x)) and np.max(np.abs(x)) < BOUND:
            good.append(x.copy())
        else:
            discarded += 1
    return good, discarded


def _analytic(x: np.ndarray, c: np.ndarray, Ns: int) -> np.ndarray:
    if Ns <= 2:
        xx = np.empty((2, 2))
        ww = np.empty((2, 2, 2))
        xx[0] = x[:2]
        ww[0] = np.eye(2)
        tick = K._propagate_n12(0, xx, ww, Ns, c, ALPHA, BETA)
        return ww[tick].copy()
    dim = _state_dim(Ns)
    xx = np.empty((2, dim))
    ww = np.empty((2, dim, Ns))
    xx[0] = x[:dim]
    ww[0] = np.vstack([np.eye(Ns), np.zeros((dim - Ns, Ns))])
    tick = K._propagate_nN(0, xx, ww, Ns, c, ALPHA, BETA)
    return ww[tick][:Ns, :Ns].copy()


def _fd_by_step(x: np.ndarray, c: np.ndarray, Ns: int) -> list[np.ndarray]:
    dim = 2 if Ns <= 2 else Ns
    buf = np.zeros(_state_dim(Ns))
    buf[:dim] = x[:dim]
    jacobians = []
    for step in STEPS:
        jac = np.zeros((dim, dim))
        for k in range(dim):
            xp = buf.copy()
            xp[k] += step
            xm = buf.copy()
            xm[k] -= step
            fp = _step(xp, c, Ns)[:dim]
            fm = _step(xm, c, Ns)[:dim]
            jac[:, k] = (fp - fm) / (2.0 * step)
        jacobians.append(jac)
    return jacobians


def _aggregate(
    Ns: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[tuple[int, int], float, float, float]], int]:
    dim = 2 if Ns <= 2 else Ns
    worst_err = np.zeros((dim, dim))
    worst_ana = np.zeros((dim, dim))
    worst_num = np.zeros((dim, dim))
    have = np.zeros((dim, dim), dtype=bool)
    discarded = 0
    sets, x0 = _coeff_sets(Ns)
    for _, c in sets:
        states, disc = _early_states(c, Ns, x0)
        discarded += disc
        for state in states:
            analytic = _analytic(state, c, Ns)
            stack = np.stack(_fd_by_step(state, c, Ns))
            errors = np.abs(analytic[None, :, :] - stack)
            best_idx = errors.argmin(axis=0)
            best_err = errors.min(axis=0)
            for i in range(dim):
                for j in range(dim):
                    if not have[i, j] or best_err[i, j] > worst_err[i, j]:
                        worst_err[i, j] = best_err[i, j]
                        worst_ana[i, j] = analytic[i, j]
                        worst_num[i, j] = stack[best_idx[i, j], i, j]
                        have[i, j] = True
    entries = [
        ((i + 1, j + 1), float(worst_err[i, j]), float(worst_ana[i, j]), float(worst_num[i, j]))
        for i in range(dim)
        for j in range(dim)
    ]
    entries.sort(key=lambda item: item[1], reverse=True)
    return worst_err, worst_ana, entries[:5], discarded


def _eq4_map(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Equation (4) of the article, implemented literally in NumPy."""
    Ns = len(c)
    x1, x2, x3 = x[0], x[1], x[2]
    new = np.empty(Ns)
    new[0] = ALPHA - x3**2 + BETA * x2
    new[1] = x1
    new[2] = c[0] * (ALPHA - x3**2 + BETA * x2) + c[1] * x1 + c[2] * x2
    for j in range(3, Ns):
        new[2] += c[j] * x[j]
    if Ns > 3:
        new[3] = x2
    for j in range(4, Ns):
        new[j] = x[j - 1]
    return new


def _print_named_evaluation(Ns: int = 4) -> None:
    sets, x0 = _coeff_sets(Ns)
    name, c = sets[0]
    states, _ = _early_states(c, Ns, x0)
    state = states[0]
    analytic = _analytic(state, c, Ns)
    stack = _fd_by_step(state, c, Ns)
    best = int(np.argmin([np.abs(analytic - j).max() for j in stack]))
    numeric = stack[best]
    print(f"\n=== named evaluation, N_s = {Ns} ({name}) ===")
    print("  coefficients c_0..c_3:", np.array2string(c, precision=8))
    print("  state x:", np.array2string(state, precision=8))
    print("  analytic J:")
    print(np.array2string(analytic, precision=10))
    print(f"  numeric J (central FD, step={STEPS[best]:.0e}):")
    print(np.array2string(numeric, precision=10))
    print(f"  beta*c_0           = {BETA * c[0]:.10f}")
    print(f"  beta*c_0 + c_2     = {BETA * c[0] + c[2]:.10f}")
    print(f"  analytic (3,2)     = {analytic[2, 1]:.10f}")
    print(f"  numeric  (3,2)     = {numeric[2, 1]:.10f}")
    print(f"  |numeric - (beta*c0)|      = {abs(numeric[2, 1] - BETA * c[0]):.3e}")
    print(f"  |numeric - (beta*c0+c2)|   = {abs(numeric[2, 1] - (BETA * c[0] + c[2])):.3e}")


@pytest.mark.slow
def test_jacobian_matches_finite_differences():
    _print_named_evaluation(4)
    failures: list[tuple[int, float]] = []
    total_discarded = 0
    for Ns in NS_VALUES:
        worst_err, worst_ana, top5, discarded = _aggregate(Ns)
        total_discarded += discarded
        allowed = TOL_ABS + TOL_REL * np.abs(worst_ana)
        ratio = float(np.max(worst_err / allowed))
        print(f"\n=== N_s = {Ns} (K = {worst_err.shape[0]}) max error per entry ===")
        print(np.array2string(worst_err, precision=3, suppress_small=False))
        print(f"  max(err / (abs + rel*|analytic|)) = {ratio:.3e}  (limit 1.0)")
        print(f"  candidates discarded by the bound/finite guard: {discarded}")
        print("  five worst entries (i, j) [1-based], error, analytic, numeric:")
        for (i, j), err, ana, num in top5:
            print(f"    ({i},{j})  err={err:.3e}  analytic={ana:.8f}  numeric={num:.8f}")
        if ratio > 1.0:
            failures.append((Ns, ratio))
    print(f"\ntotal candidates discarded by the guard: {total_discarded}")
    assert not failures, f"Jacobian mismatch (mixed tolerance): {failures}"


@pytest.mark.slow
def test_eq4_matches_package_map():
    rng = np.random.default_rng(12345)
    worst = 0.0
    for Ns in (4, 5, 8, 12):
        for _ in range(50):
            c = rng.standard_normal(Ns)
            x = rng.standard_normal(_state_dim(Ns))
            reference = np.asarray(_eq4_map(x[:Ns], c), dtype=float)
            package = _step(x, c, Ns)[:Ns]
            worst = max(worst, float(np.max(np.abs(reference - package))))
    print(f"\nmax |eq(4) literal - package map| = {worst:.3e}")
    assert worst < 1e-12, f"equation (4) and the package map disagree: {worst}"

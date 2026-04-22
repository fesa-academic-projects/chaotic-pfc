"""
maps.py
=======
Hénon map variants used in chaotic communication systems.

Standard Hénon map (1976):
    x[n+1] = 1 − a·x[n]² + y[n]
    y[n+1] = b·x[n]

Generalised form:
    x₁[n+1] = α − x₁[n]² + β·x₂[n]
    x₂[n+1] = x₁[n]

Filtered form:
    x₁[n+1] = α − (c₀·x₁[n] + c₁·x₂[n])² + β·x₂[n]
    x₂[n+1] = x₁[n]

N-th order with internal FIR filter.
"""

import numpy as np
from numpy.typing import NDArray


# ── 2-D maps ────────────────────────────────────────────────────────────────

def henon_standard(
    steps: int, x0: float = 0.0, y0: float = 0.0,
    a: float = 1.4, b: float = 0.3,
) -> tuple[NDArray, NDArray]:
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    for i in range(steps):
        X[i + 1] = 1.0 - a * X[i] ** 2 + Y[i]
        Y[i + 1] = b * X[i]
    return X, Y


def henon_generalised(
    steps: int, x0: float = 0.0, y0: float = 0.0,
    alpha: float = 1.4, beta: float = 0.3,
) -> tuple[NDArray, NDArray]:
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    for i in range(steps):
        X[i + 1] = alpha - X[i] ** 2 + beta * Y[i]
        Y[i + 1] = X[i]
    return X, Y


def henon_filtered(
    steps: int, x0: float = 0.0, y0: float = 0.0,
    alpha: float = 1.4, beta: float = 0.3,
    c0: float = 0.7, c1: float = 0.3,
) -> tuple[NDArray, NDArray]:
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    for i in range(steps):
        z = c0 * X[i] + c1 * Y[i]
        X[i + 1] = alpha - z ** 2 + beta * Y[i]
        Y[i + 1] = X[i]
    return X, Y


# ── N-th order map (vectorised state) ──────────────────────────────────────

def _henon_n4_step(
    x: NDArray, s: float, a: float, b: float, c: NDArray,
) -> NDArray:
    """Single step of the N-th order Hénon map.

    State vector x has length Nc (= len(c)).
    Output of the oscillator is x[2].
    """
    Nc = len(c)
    out = np.empty(Nc)

    x1_new = a - s ** 2 + b * x[1]
    x2_new = x[0]

    v = c[0] * x1_new + c[1] * x[0] + c[2] * x[1]
    if Nc > 3:
        v += float(np.dot(c[3:], x[3:]))

    out[0] = x1_new
    out[1] = x2_new
    out[2] = v
    if Nc > 3:
        out[3] = x[1]
    if Nc > 4:
        out[4:] = x[3:Nc - 1]
    return out


def henon_order_n(
    steps: int, fir_coeffs: NDArray,
    x0: NDArray | None = None,
    a: float = 1.4, b: float = 0.3,
    driving: NDArray | None = None,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    Nc = len(fir_coeffs)
    rng = np.random.default_rng(seed)
    c = np.asarray(fir_coeffs, dtype=float)

    state = np.zeros((Nc, steps + 1))
    state[:, 0] = x0 if x0 is not None else 0.5 * rng.random(Nc)
    output = np.empty(steps)

    for i in range(steps):
        s_i = float(driving[i]) if driving is not None else state[2, i]
        output[i] = state[2, i]
        state[:, i + 1] = _henon_n4_step(state[:, i], s_i, a, b, c)

    return state, output

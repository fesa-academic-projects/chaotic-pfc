"""
transmitter.py
==============
Chaos-based transmitter (modulator).

Standard:
    s[n] = x₁[n] + μ·m[n]
    x₁[n+1] = a − s[n]² + b·x₂[n]
    x₂[n+1] = x₁[n]

N-th order:
    s[n] = v[n] + μ·m[n]   where v[n] = x[2][n]
"""

import numpy as np
from numpy.typing import NDArray
from .maps import _henon_n4_step


def transmit(
    message: NDArray, mu: float = 0.01,
    a: float = 1.4, b: float = 0.3,
    x0: float = 0.0, y0: float = 0.0,
) -> NDArray:
    N = len(message)
    x1 = np.empty(N + 1)
    x2 = np.empty(N + 1)
    x1[0], x2[0] = x0, y0
    s = np.empty(N)

    for k in range(N):
        s[k] = x1[k] + mu * message[k]
        x1[k + 1] = a - s[k] ** 2 + b * x2[k]
        x2[k + 1] = x1[k]
    return s


def transmit_order_n(
    message: NDArray, fir_coeffs: NDArray,
    mu: float = 0.01, a: float = 1.4, b: float = 0.3,
    x0: NDArray | None = None, seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    Nc = len(fir_coeffs)
    rng = np.random.default_rng(seed)
    c = np.asarray(fir_coeffs, dtype=float)

    N = len(message)
    state = np.zeros((Nc, N + 1))
    state[:, 0] = x0 if x0 is not None else 0.5 * rng.random(Nc)
    s = np.empty(N)

    for i in range(N):
        v = state[2, i]
        s[i] = v + mu * message[i]
        state[:, i + 1] = _henon_n4_step(state[:, i], s[i], a, b, c)
    return s, state

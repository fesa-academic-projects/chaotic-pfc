"""
receiver.py
===========
Chaos-synchronisation demodulator.

Standard:
    y₁[n+1] = a − r[n]² + b·y₂[n]
    y₂[n+1] = y₁[n]
    m̂[n]    = (r[n] − y₁[n]) / μ

N-th order: carrier from filtered state index 2.
"""

import numpy as np
from numpy.typing import NDArray
from .maps import _henon_n4_step


def receive(
    r: NDArray, mu: float = 0.01,
    a: float = 1.4, b: float = 0.3,
    y0: float = 0.0, z0: float = 0.0,
) -> NDArray:
    N = len(r)
    y1 = np.empty(N + 1)
    y2 = np.empty(N + 1)
    y1[0], y2[0] = y0, z0
    m_hat = np.empty(N)

    for k in range(N):
        y1[k + 1] = a - r[k] ** 2 + b * y2[k]
        y2[k + 1] = y1[k]
        m_hat[k] = (r[k] - y1[k]) / mu
    return m_hat


def receive_order_n(
    r: NDArray, fir_coeffs: NDArray,
    mu: float = 0.01, a: float = 1.4, b: float = 0.3,
    y0: NDArray | None = None, seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    N = len(r)
    Nc = len(fir_coeffs)
    rng = np.random.default_rng(seed)
    c = np.asarray(fir_coeffs, dtype=float)

    state = np.zeros((Nc, N + 1))
    state[:, 0] = y0 if y0 is not None else 0.5 * rng.random(Nc)
    m_hat = np.empty(N)

    for i in range(N):
        v = state[2, i]
        m_hat[i] = (r[i] - v) / mu
        state[:, i + 1] = _henon_n4_step(state[:, i], r[i], a, b, c)
    return m_hat, state

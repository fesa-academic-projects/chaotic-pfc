"""Consistência do núcleo técnico do PFC com o texto do artigo.

Duas verificações pedidas na revisão externa, mantidas como testes:

a) a jacobiana simbólica (SymPy) do sistema (4) para ``Nz = 5`` é comparada,
   entrada a entrada, com a matriz (5);
b) a jacobiana usada no código é comparada com diferenças finitas centrais do
   mapa do código, em pontos aleatórios, para ``Nz = 1, 2, 8 e 40``.

O mapeamento entre o texto e o código é ``Ns = Nz + 1`` (ordem do filtro contra
número de coeficientes).
"""

from __future__ import annotations

import numpy as np
import pytest

from chaotic_pfc.analysis.sweep import _kernel as K

ALPHA = 1.4
BETA = 0.3
STEP = 1e-6


def _state_dim(Ns: int) -> int:
    return 2 if Ns <= 2 else max(Ns, 4)


def _coeffs(Ns: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.random(Ns)
    return c / c.sum()


def _step(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    Ns = len(c)
    x_in = np.asarray(x, dtype=float).copy()
    x_out = np.empty_like(x_in)
    if Ns <= 2:
        K._henon_n12_inplace(x_in, x_out, ALPHA, BETA, c)
    else:
        K._henon_nN_inplace(x_in, x_out, ALPHA, BETA, c)
    return x_out


def _analytic(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    Ns = len(c)
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
    return ww[tick][:Ns, :].copy()


@pytest.mark.parametrize("Nz", [1, 2, 8, 40])
def test_code_jacobian_matches_finite_differences(Nz: int) -> None:
    """A jacobiana do kernel reproduz o mapa do código por diferenças finitas."""
    Ns = Nz + 1
    dim = Ns
    worst = 0.0
    for seed in range(5):
        c = _coeffs(Ns, seed)
        rng = np.random.default_rng(100 + seed)
        x = 0.1 * rng.standard_normal(_state_dim(Ns))
        x[:dim] = np.clip(x[:dim], -0.5, 0.5)
        analytic = _analytic(x, c)[:dim, :dim]
        numeric = np.zeros((dim, dim))
        for k in range(dim):
            xp = x.copy()
            xm = x.copy()
            xp[k] += STEP
            xm[k] -= STEP
            fp = _step(xp, c)[:dim]
            fm = _step(xm, c)[:dim]
            numeric[:, k] = (fp - fm) / (2.0 * STEP)
        scale = np.maximum(np.abs(analytic), 1.0)
        worst = max(worst, float(np.max(np.abs(analytic - numeric) / scale)))
    assert worst < 1e-6, f"Nz={Nz}: erro relativo máximo {worst:.2e}"


def test_symbolic_jacobian_matches_matrix_eq5() -> None:
    """A jacobiana simbólica de (4) é a matriz (5), para Nz = 5."""
    sp = pytest.importorskip("sympy")

    Ns = 6
    x = sp.symbols(f"x1:{Ns + 1}")
    c = sp.symbols(f"c0:{Ns}")
    f1 = ALPHA - x[2] ** 2 + BETA * x[1]
    f = [
        f1,
        x[0],
        c[0] * f1 + c[1] * x[0] + c[2] * x[1] + sum(c[j] * x[j] for j in range(3, Ns)),
        x[1],
    ]
    f.extend(x[j - 1] for j in range(4, Ns))
    symbolic = sp.Matrix(f).jacobian(x)

    matrix = sp.zeros(Ns, Ns)
    matrix[0, 1] = BETA
    matrix[0, 2] = -2 * x[2]
    matrix[1, 0] = 1
    matrix[2, 0] = c[1]
    matrix[2, 1] = c[0] * BETA + c[2]
    matrix[2, 2] = -2 * c[0] * x[2]
    for j in range(3, Ns):
        matrix[2, j] = c[j]
    matrix[3, 1] = 1
    for row in range(4, Ns):
        matrix[row, row - 1] = 1

    assert sp.simplify(symbolic - matrix) == sp.zeros(Ns, Ns)

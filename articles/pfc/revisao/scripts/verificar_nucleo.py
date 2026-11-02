#!/usr/bin/env python3
"""Verificação do núcleo técnico do PFC contra o texto do artigo.

Quatro verificações, cada uma imprime PASS/FAIL e um resumo numérico:

a) jacobiana simbólica (SymPy) do sistema (4) para ``Nz = 5`` comparada, entrada
   a entrada, com a matriz (5);
b) jacobiana usada no código contra diferenças finitas centrais do mapa do
   código, em pontos aleatórios, para ``Nz = 1, 2, 8 e 40``;
c) mapa do código contra (4) para ``Nz >= 2`` e contra o texto da Seção 2.4 para
   ``Nz <= 1``;
d) estimador do código contra a descrição das Seções 2.5 e 3.3: vetor único,
   renormalização a cada 10 iterações, soma dos logaritmos dividida pelo número
   de iterações acumuladas, parada adaptativa e recálculo com Gram-Schmidt
   modificado nas células de fronteira.

O mapeamento entre o texto e o código é ``Ns = Nz + 1``: ``Nz`` é a ordem do
filtro e ``Ns`` é o número de coeficientes (taps).

Uso::

    python articles/pfc/revisao/scripts/verificar_nucleo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from chaotic_pfc._formulas import fixed_point_positive
from chaotic_pfc.analysis.sweep import _kernel as K
from chaotic_pfc.analysis.sweep._types import (
    _ADAPTIVE_CHECKPOINT_EVERY,
    _ADAPTIVE_STREAK,
    _BENETTIN_K,
    _VONLY_MARGIN,
    LAMBDA_CHAOS_THRESHOLD,
)

ALPHA = 1.4
BETA = 0.3
ROOT = Path(__file__).resolve().parents[4]

_RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    """Guarda e imprime o resultado de uma verificação."""
    _RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def state_dim(Ns: int) -> int:
    """Dimensão do estado usado pelo kernel (inclui o slot de rascunho)."""
    return 2 if Ns <= 2 else max(Ns, 4)


def random_coeffs(Ns: int, seed: int) -> np.ndarray:
    """Coeficientes positivos normalizados, como um filtro passa-baixas."""
    rng = np.random.default_rng(seed)
    c = rng.random(Ns)
    return c / c.sum()


def eq4_map(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Sistema (4) do artigo, implementado literalmente."""
    Ns = len(c)
    out = np.empty(Ns)
    out[0] = ALPHA - x[2] ** 2 + BETA * x[1]
    out[1] = x[0]
    out[2] = c[0] * (ALPHA - x[2] ** 2 + BETA * x[1]) + c[1] * x[0] + c[2] * x[1]
    for j in range(3, Ns):
        out[2] += c[j] * x[j]
    if Ns > 3:
        out[3] = x[1]
    for j in range(4, Ns):
        out[j] = x[j - 1]
    return out


def eq5_jacobian(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Matriz (5) do artigo, montada a partir do estado e dos coeficientes."""
    Ns = len(c)
    jac = np.zeros((Ns, Ns))
    jac[0, 1] = BETA
    jac[0, 2] = -2.0 * x[2]
    jac[1, 0] = 1.0
    jac[2, 0] = c[1]
    jac[2, 1] = c[0] * BETA + c[2]
    jac[2, 2] = -2.0 * c[0] * x[2]
    if Ns > 3:
        jac[2, 3:] = c[3:]
        jac[3, 1] = 1.0
        for row in range(4, Ns):
            jac[row, row - 1] = 1.0
    return jac


def code_step(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Um passo do mapa do código, devolvendo o estado completo do buffer."""
    Ns = len(c)
    x_in = np.asarray(x, dtype=float).copy()
    x_out = np.empty_like(x_in)
    if Ns <= 2:
        K._henon_n12_inplace(x_in, x_out, ALPHA, BETA, c)
    else:
        K._henon_nN_inplace(x_in, x_out, ALPHA, BETA, c)
    return x_out


def code_jacobian(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Jacobiana do kernel, com ``W = I`` e a dimensão descrita no artigo."""
    Ns = len(c)
    if Ns <= 2:
        xx = np.empty((2, 2))
        ww = np.empty((2, 2, 2))
        xx[0] = x[:2]
        ww[0] = np.eye(2)
        tick = K._propagate_n12(0, xx, ww, Ns, c, ALPHA, BETA)
        return ww[tick].copy()
    dim = state_dim(Ns)
    xx = np.empty((2, dim))
    ww = np.empty((2, dim, Ns))
    xx[0] = x[:dim]
    ww[0] = np.vstack([np.eye(Ns), np.zeros((dim - Ns, Ns))])
    tick = K._propagate_nN(0, xx, ww, Ns, c, ALPHA, BETA)
    return ww[tick][:Ns, :].copy()


def check_a() -> None:
    """Jacobsiana simbólica de (4) contra a matriz (5), para Nz = 5."""
    import sympy as sp

    Nz = 5
    Ns = Nz + 1
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
    if Ns > 3:
        for j in range(3, Ns):
            matrix[2, j] = c[j]
        matrix[3, 1] = 1
        for row in range(4, Ns):
            matrix[row, row - 1] = 1

    diff = sp.simplify(symbolic - matrix)
    nonzero = [(i + 1, j + 1, diff[i, j]) for i in range(Ns) for j in range(Ns) if diff[i, j] != 0]
    report(
        "a) jacobiana simbólica (4) x matriz (5), Nz=5",
        not nonzero,
        f"{Ns}x{Ns} entradas, {len(nonzero)} divergências"
        + (f": {nonzero[:3]}" if nonzero else ""),
    )


def check_b() -> None:
    """Jacobiana do código contra diferenças finitas centrais do mapa do código."""
    step = 1e-6
    for Nz in (1, 2, 8, 40):
        Ns = Nz + 1
        dim = Ns
        worst = 0.0
        for seed in range(5):
            c = random_coeffs(Ns, seed)
            rng = np.random.default_rng(100 + seed)
            x = 0.1 * rng.standard_normal(state_dim(Ns))
            x[:dim] = np.clip(x[:dim], -0.5, 0.5)
            analytic = code_jacobian(x, c)[:dim, :dim]
            numeric = np.zeros((dim, dim))
            for k in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[k] += step
                xm[k] -= step
                fp = code_step(xp, c)[:dim]
                fm = code_step(xm, c)[:dim]
                numeric[:, k] = (fp - fm) / (2.0 * step)
            scale = np.maximum(np.abs(analytic), 1.0)
            worst = max(worst, float(np.max(np.abs(analytic - numeric) / scale)))
        report(
            f"b) jacobiana do código x diferenças finitas, Nz={Nz}",
            worst < 1e-6,
            f"erro relativo máximo {worst:.2e}",
        )


def check_c() -> None:
    """Mapa do código contra (4) e o caso Nz <= 1 contra a Seção 2.4."""
    worst_eq4 = 0.0
    for Nz in (2, 3, 5, 8, 40):
        Ns = Nz + 1
        for seed in range(10):
            c = random_coeffs(Ns, seed)
            rng = np.random.default_rng(200 + seed)
            x = 0.1 * rng.standard_normal(state_dim(Ns))
            reference = eq4_map(x[:Ns], c)
            package = code_step(x, c)[:Ns]
            worst_eq4 = max(worst_eq4, float(np.max(np.abs(reference - package))))
    report(
        "c1) mapa do código x sistema (4), Nz >= 2",
        worst_eq4 < 1e-12,
        f"erro absoluto máximo {worst_eq4:.2e}",
    )

    worst_n1 = 0.0
    for Nz in (0, 1):
        Ns = Nz + 1
        for seed in range(10):
            c = random_coeffs(Ns, seed)
            if Ns == 1:
                c = np.array([c[0]])
            rng = np.random.default_rng(300 + seed)
            x = 0.1 * rng.standard_normal(2)
            x3 = c[0] * x[0] + (c[1] * x[1] if Ns > 1 else 0.0)
            reference = np.array([ALPHA - x3**2 + BETA * x[1], x[0]])
            package = code_step(x, c)
            worst_n1 = max(worst_n1, float(np.max(np.abs(reference - package))))
    report(
        "c2) mapa do código x texto da 2.4, Nz <= 1",
        worst_n1 < 1e-12,
        f"erro absoluto máximo {worst_n1:.2e}",
    )


def reference_single_vector(x0: np.ndarray, c: np.ndarray, iterations: int) -> tuple[float, int]:
    """Estimador de vetor único escrito à parte, seguindo as Seções 2.5 e 3.3."""
    Ns = len(c)
    x = x0.copy()
    v = np.zeros(Ns)
    v[0] = 1.0
    total = 0.0
    for it in range(1, iterations + 1):
        jac = eq5_jacobian(x, c)
        v = jac @ v
        x = eq4_map(x, c)
        if it % _BENETTIN_K == 0:
            nrm = float(np.linalg.norm(v))
            total += np.log(nrm)
            v /= nrm
    return total / iterations, iterations


def check_d() -> None:
    """Estimador do código contra as Seções 2.5 e 3.3."""
    constants_ok = (
        _BENETTIN_K == 10
        and _ADAPTIVE_CHECKPOINT_EVERY == 100
        and _ADAPTIVE_STREAK == 2
        and LAMBDA_CHAOS_THRESHOLD == 5e-3
        and _VONLY_MARGIN == 5e-3
    )
    report(
        "d1) constantes do estimador",
        constants_ok,
        f"K={_BENETTIN_K}, checkpoint={_ADAPTIVE_CHECKPOINT_EVERY}, "
        f"streak={_ADAPTIVE_STREAK}, theta={LAMBDA_CHAOS_THRESHOLD:g}, "
        f"margem={_VONLY_MARGIN:g}",
    )

    Ns = 9
    c = random_coeffs(Ns, 7)
    gain = float(c.sum())
    p = float(fixed_point_positive(gain, ALPHA, BETA))
    x0 = np.full(Ns, p)
    iterations = 200

    reference, n_ref = reference_single_vector(x0, c, iterations)
    xx = np.empty((2, Ns))
    vv = np.empty((2, Ns))
    lyap_sum = np.empty(1)
    xx[0] = x0
    code, n_code = K._lyap_online_core_v(
        xx, vv, lyap_sum, iterations, iterations, 1e-3, Ns, 1, c, ALPHA, BETA
    )
    error = abs(float(code) - reference)
    report(
        "d2) vetor único: soma dos logaritmos / iterações",
        error < 1e-9 and n_code == n_ref,
        f"referência {reference:.12f}, código {float(code):.12f}, erro {error:.2e}, n={n_code}",
    )

    xx_full = np.empty((2, Ns))
    ww = np.empty((2, Ns, Ns))
    lyap_full = np.empty(Ns)
    xx_full[0] = x0
    full, n_full = K._lyap_online_core(
        xx_full, ww, lyap_full, iterations, iterations, 1e-3, Ns, 1, c, ALPHA, BETA, 1
    )
    full_error = abs(float(full) - float(code))
    report(
        "d3) vetor único = coluna 0 do espectro completo (n_use=1)",
        full_error < 1e-9 and n_full == n_code,
        f"espectro {float(full):.12f}, vetor {float(code):.12f}, erro {full_error:.2e}",
    )

    xx_ad = np.empty((2, Ns))
    vv_ad = np.empty((2, Ns))
    lyap_ad = np.empty(1)
    xx_ad[0] = x0
    _, n_adapt = K._lyap_online_core_v(
        xx_ad, vv_ad, lyap_ad, 3000, 100, 1e-3, Ns, 1, c, ALPHA, BETA
    )
    report(
        "d4) parada adaptativa: para em checkpoint de 100",
        n_adapt < 3000 and n_adapt >= 100 and n_adapt % _ADAPTIVE_CHECKPOINT_EVERY == 0,
        f"n_used={n_adapt} (limite 3000)",
    )

    numbers = json.loads((ROOT / "build/pfc/pfc_numbers.json").read_text())
    n_fallback = int(numbers["RECALC.n_fallback"])
    report(
        "d5) recálculo nas células de fronteira",
        n_fallback == 23000,
        f"n_fallback={n_fallback} no build/pfc/pfc_numbers.json (texto: 23.000 de 552.000)",
    )


def main() -> int:
    """Executa as quatro verificações e devolve o código de saída."""
    check_a()
    check_b()
    check_c()
    check_d()
    failed = [name for name, ok, _ in _RESULTS if not ok]
    print()
    if failed:
        print(f"{len(failed)} verificação(ões) falharam:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"{len(_RESULTS)} verificações: todas PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

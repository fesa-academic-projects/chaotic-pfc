#!/usr/bin/env python3
"""Hierarquia dos fatores sobre a fracao de celulas caoticas.

Le os checkpoints versionados em ``data/sweeps`` (nenhuma varredura nova) e
classifica cada celula exatamente como o artigo: ``NaN`` em ``h`` e divergente,
``h > theta`` e caotica, o resto e nao caotica, com ``theta`` vindo de
``chaotic_pfc.analysis.sweep.LAMBDA_CHAOS_THRESHOLD`` (5e-3).

Para cada configuracao ``rho_c = N_c / N_grade``. A influencia de um fator e a
amplitude de ``rho_c`` ao variar so aquele fator com os outros fixos, media
sobre as combinacoes. Como o numero de niveis difere entre fatores (4 tipos, 7
janelas, 40 valores de beta_K), o script tambem calcula o desvio padrao medio e
um eta quadrado com blocos, que nao cresce so porque o fator tem mais niveis.

A comparacao entre tipos usa o suporte comum (ordens pares apenas), porque os
filtros passa-altas e rejeita-faixa so admitem ordens pares no metodo da
janela; a media no grid completo fica ao lado, para comparar com o texto.

Uso::

    python articles/pfc/revisao/scripts/hierarquia.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD as THETA

ROOT = Path(__file__).resolve().parents[4]
SWEEPS = ROOT / "data/sweeps"

TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
TYPE_PT = {
    "lowpass": "passa-baixas",
    "highpass": "passa-altas",
    "bandpass": "passa-faixa",
    "bandstop": "rejeita-faixa",
}
FIXED = (
    ("Rectangular", "retangular"),
    ("Bartlett", "Bartlett"),
    ("Hann", "Hann"),
    ("Hamming", "Hamming"),
    ("Blackman", "Blackman"),
    ("Blackman-Harris", "Blackman-Harris"),
)
BETAS = tuple(round(0.5 + 0.5 * k, 2) for k in range(40))


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the exponent grid and the number of taps per row."""
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["h"], dtype=float), np.asarray(data["num_taps_grid"])


def configs() -> dict[tuple[str, str, float | None], tuple[np.ndarray, np.ndarray]]:
    """Map ``(tipo, janela, beta)`` to the checkpoint arrays of all 184 configs."""
    out: dict[tuple[str, str, float | None], tuple[np.ndarray, np.ndarray]] = {}
    for folder, label in FIXED:
        for ftype in TYPES:
            path = SWEEPS / f"{folder} ({ftype})" / "variables_lyapunov.npz"
            out[(ftype, label, None)] = load(path)
    for ftype in TYPES:
        for beta in BETAS:
            path = SWEEPS / "kaiser" / ftype / f"beta_{beta:.2f}" / "variables_lyapunov.npz"
            out[(ftype, "Kaiser", beta)] = load(path)
    return out


def rho(h: np.ndarray, even_only: bool, taps: np.ndarray) -> float:
    """Fraction of chaotic cells, optionally restricted to odd numbers of taps."""
    cells = h[taps % 2 == 1] if even_only else h
    return float(np.mean(np.isfinite(cells) & (cells > THETA)))


def report(name: str, cells: np.ndarray) -> None:
    """Print the amplitude, dispersion and blocked eta squared of one design."""
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    widths = np.ptp(cells, axis=1)
    m = float(cells.mean())
    ss_factor = cells.shape[0] * float(np.sum((cells.mean(axis=0) - m) ** 2))
    resid = cells - cells.mean(axis=0, keepdims=True) - cells.mean(axis=1, keepdims=True) + m
    ss_error = float(np.sum(resid**2))
    print(
        f"  {name:<26} blocos {cells.shape[0]:3d}  niveis {cells.shape[1]:2d}  "
        f"amplitude media {100 * widths.mean():5.2f}  maxima {100 * widths.max():5.2f}  "
        f"desvio medio {100 * cells.std(axis=1).mean():5.2f}  "
        f"eta2 {ss_factor / (ss_factor + ss_error):.3f}"
    )


def main() -> int:
    """Print the factor hierarchy and the per-configuration table."""
    data = configs()
    if len(data) != 184:
        raise SystemExit(f"esperava 184 configuracoes, encontrei {len(data)}")
    total_cells = sum(h.size for h, _ in data.values())
    if total_cells != 552_000:
        raise SystemExit(f"esperava 552000 celulas, encontrei {total_cells}")

    values = {key: rho(h, True, taps) for key, (h, taps) in data.items()}
    full = {key: rho(h, False, taps) for key, (h, taps) in data.items()}

    print(f"configuracoes: {len(data)}, celulas: {total_cells}, theta={THETA}")
    print()
    print("rho_c por tipo (suporte comum, ordens pares) e grid completo (%):")
    for ftype in TYPES:
        sub = [v for k, v in values.items() if k[0] == ftype]
        sub_full = [v for k, v in full.items() if k[0] == ftype]
        print(
            f"  {TYPE_PT[ftype]:<14} media {100 * np.mean(sub):5.1f}  "
            f"min {100 * np.min(sub):5.1f}  max {100 * np.max(sub):5.1f}  "
            f"| grid completo: media {100 * np.mean(sub_full):5.1f}"
        )
    print()

    type_design = [[values[(ftype, label, None)] for ftype in TYPES] for _, label in FIXED] + [
        [values[(ftype, "Kaiser", beta)] for ftype in TYPES] for beta in BETAS
    ]

    window_design = [
        [values[(ftype, label, None)] for _, label in FIXED] + [values[(ftype, "Kaiser", beta)]]
        for ftype in TYPES
        for beta in BETAS
    ]
    window_fixed = [[values[(ftype, label, None)] for _, label in FIXED] for ftype in TYPES]

    beta_design = [[values[(ftype, "Kaiser", beta)] for beta in BETAS] for ftype in TYPES]

    print("influencia dos fatores (amplitude de rho_c em pontos percentuais):")
    report("tipo (4 niveis)", type_design)
    report("janela (7 niveis)", window_design)
    report("janela (6 fixas)", window_fixed)
    report("beta_K (40 niveis)", beta_design)
    print()

    print("janela dentro de cada tipo (Kaiser na media dos betas, %):")
    for ftype in TYPES:
        row = [f"{label} {100 * values[(ftype, label, None)]:.1f}" for _, label in FIXED]
        kaiser = [v for k, v in values.items() if k[0] == ftype and k[1] == "Kaiser"]
        row.append(
            f"Kaiser {100 * np.mean(kaiser):.1f} ({100 * np.min(kaiser):.1f} a "
            f"{100 * np.max(kaiser):.1f})"
        )
        print(f"  {TYPE_PT[ftype]:<14} " + "; ".join(row))
    print()

    print("caso do enlace (Hamming passa-baixas, Nz=8, corte 0,52 pi):")
    h, taps = data[("lowpass", "Hamming", None)]
    line = int(np.flatnonzero(taps == 9)[0])
    cutoff = np.linspace(1e-5, 1 - 1e-5, h.shape[1])
    nearest = int(np.argmin(np.abs(cutoff - 0.52)))
    chaotics = np.flatnonzero(np.isfinite(h[line]) & (h[line] > THETA))
    print(
        f"  Nz={taps[line] - 1}, rho_c={100 * values[('lowpass', 'Hamming', None)]:.1f}%, "
        f"celulas caoticas na linha {chaotics.size} de {h.shape[1]}, "
        f"corte maximo caotico {cutoff[chaotics.max()]:.4f}"
    )
    print(
        f"  ponto mais proximo: omega/pi={cutoff[nearest]:.4f}, "
        f"lambda1={h[line, nearest]:.6f}, {h[line, nearest] / THETA:.1f} x theta"
    )
    print()

    print("tabela completa (tipo | janela | beta | rho_c % | celulas | divergentes %):")
    for key in sorted(data, key=lambda k: (TYPES.index(k[0]), k[1], k[2] or 0)):
        h, _ = data[key]
        beta = "" if key[2] is None else f"{key[2]:.2f}"
        print(
            f"  {TYPE_PT[key[0]]:<14} {key[1]:<16} {beta:<6} "
            f"{100 * full[key]:5.1f} {h.size:6d} {100 * float(np.mean(np.isnan(h))):6.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

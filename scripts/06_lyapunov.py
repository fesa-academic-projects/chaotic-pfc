#!/usr/bin/env python3
"""
06_lyapunov.py — Lyapunov exponent computation (text output + optional CSV).

Computes:
  Part A: Pure 2-D Hénon — single perturbed IC, both fixed points analysed
  Part B: 4-D Pole-filtered Hénon — single perturbed IC
  Part C: 2-D Hénon ensemble — 20 ICs drawn uniformly in ±10% around x_f⁺
  Part D: 4-D Pole-filtered ensemble — same protocol, 4-D system

Parts A/B are quick sanity checks (single estimator call). Parts C/D
implement the full experimental protocol used in the TCC: multiple ICs
are sampled around the fixed point, the Lyapunov spectrum is estimated
for each, and aggregated statistics (mean spectrum, mean/max λ_max,
chaotic count) are reported.

With ``--save`` the per-IC tables are also written to
``data/lyapunov/henon2d_ensemble.csv`` and
``data/lyapunov/henon4d_ensemble.csv``.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--Nitera", type=int, default=2000)
    p.add_argument("--Ndiscard", type=int, default=1000)
    p.add_argument("--pole-radius", type=float, default=0.975)
    p.add_argument("--w0", type=float, default=0.0)
    p.add_argument(
        "--n-ci", type=int, default=20, help="Number of ICs for parts C and D (default: 20)"
    )
    p.add_argument(
        "--perturbation",
        type=float,
        default=0.1,
        help="Half-width of sampling box around fixed point (default: 0.1 = ±10%%)",
    )
    p.add_argument(
        "--data-dir",
        default="data/lyapunov",
        help="Directory for CSV output (default: data/lyapunov)",
    )
    p.add_argument(
        "--save", action="store_true", help="Write per-IC tables to CSV under --data-dir"
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="(accepted for CLI consistency; this script has no UI)",
    )
    return p.parse_args()


def _print_ensemble_summary(label: str, result, dim: int) -> None:
    """Compact summary of an EnsembleResult."""
    print(f"     Ponto fixo (centro da amostragem): {result.fixed_point}")
    print(f"     Autovalores: {result.eigenvalues}")
    print(f"     |λ|:         {np.abs(result.eigenvalues)}")
    print(f"     Estável:     {result.stable}")
    print(f"     Número de CIs: {len(result.lmax_per_ci)}")

    mean_str = "  ".join(f"{v:+.6f}" for v in result.mean_exponents)
    print(f"     Média dos expoentes (λ_1..λ_{dim}): {mean_str}")
    print(f"     Média de λ_max:                     {result.mean_lmax:+.6f}")
    print(f"     Maior  λ_max:                       {result.max_lmax:+.6f}")

    total = result.n_chaotic + result.n_stable
    print(f"     Trajetórias caóticas: {result.n_chaotic}/{total}")
    print(f"     Trajetórias estáveis: {result.n_stable}/{total}")

    verdict = "CAÓTICO" if result.mean_lmax > 0 else "ESTÁVEL"
    print(f"     Diagnóstico: {verdict} (baseado na média de λ_max)")


def main():
    args = parse_args()
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.lyapunov import (
        lyapunov_henon2d,
        lyapunov_henon2d_ensemble,
        lyapunov_max,
        lyapunov_max_ensemble,
    )

    alpha, beta = cfg.comm.henon.a, cfg.comm.henon.b
    data_dir = Path(args.data_dir)

    # ─── Part A: Pure 2-D Hénon (single IC) ────────────────────────────────
    print(
        f"\n[06a] Lyapunov — Hénon puro 2-D (1 IC perturbada)  |  "
        f"Nitera={args.Nitera}  Ndiscard={args.Ndiscard}"
    )

    res2d = lyapunov_henon2d(
        alpha=alpha,
        beta=beta,
        Nitera=args.Nitera,
        Ndiscard=args.Ndiscard,
        perturbation=cfg.lyapunov.perturbation,
        seed=cfg.seed,
    )

    print(f"     Ponto fixo (+): {res2d['fixed_point_p']}")
    print(f"     Ponto fixo (−): {res2d['fixed_point_n']}")
    for label, key in [("(+)", "eigenvalues_p"), ("(−)", "eigenvalues_n")]:
        eigs = res2d[key]
        print(f"     Autovalores {label}: {eigs}")
        print(f"     |λ| {label}:         {np.abs(eigs)}")
    print(f"     Estável (+): {res2d['stable_p']}   (−): {res2d['stable_n']}")
    print(f"     Expoentes de Lyapunov: {res2d['all_exponents']}")
    print(
        f"     λ_max = {res2d['lyapunov_max']:.6f}  "
        f"→ {'Caótico' if res2d['lyapunov_max'] > 0 else 'Não caótico'}"
    )

    # ─── Part B: 4-D Pole-filtered Hénon (single IC) ───────────────────────
    print(
        f"\n[06b] Lyapunov — Hénon filtrado 4-D (1 IC perturbada)  |  "
        f"r={args.pole_radius}  w0={args.w0}"
    )

    res4d = lyapunov_max(
        alpha=alpha,
        beta=beta,
        Gz=cfg.lyapunov.Gz,
        pole_radius=args.pole_radius,
        w0=args.w0,
        Nitera=args.Nitera,
        Ndiscard=args.Ndiscard,
        perturbation=cfg.lyapunov.perturbation,
        seed=cfg.seed,
    )

    print(f"     Ponto fixo:  {res4d['fixed_point']}")
    print(f"     Autovalores: {res4d['eigenvalues']}")
    print(f"     |λ|:         {np.abs(res4d['eigenvalues'])}")
    print(f"     Estável:     {res4d['stable']}")
    print(f"     Expoentes de Lyapunov: {res4d['all_exponents']}")
    print(
        f"     λ_max = {res4d['lyapunov_max']:.6f}  "
        f"→ {'Caótico' if res4d['lyapunov_max'] > 0 else 'Não caótico'}"
    )

    # ─── Part C: 2-D Hénon ensemble (N_ci ICs ±perturbation) ───────────────
    print(
        f"\n[06c] Lyapunov — Hénon puro 2-D (ensemble)  |  "
        f"N_ci={args.n_ci}  ±{args.perturbation * 100:.0f}%"
    )

    ens2d = lyapunov_henon2d_ensemble(
        alpha=alpha,
        beta=beta,
        Nitera=args.Nitera,
        Ndiscard=args.Ndiscard,
        perturbation=args.perturbation,
        n_initial=args.n_ci,
        seed=cfg.seed,
    )
    _print_ensemble_summary("2-D", ens2d, dim=2)

    if args.save:
        out = ens2d.to_csv(data_dir / "henon2d_ensemble.csv")
        print(f"     Tabela por CI salva em: {out}")

    # ─── Part D: 4-D Pole-filtered ensemble ────────────────────────────────
    print(
        f"\n[06d] Lyapunov — Hénon filtrado 4-D (ensemble)  |  "
        f"N_ci={args.n_ci}  ±{args.perturbation * 100:.0f}%  "
        f"r={args.pole_radius}  w0={args.w0}"
    )

    ens4d = lyapunov_max_ensemble(
        alpha=alpha,
        beta=beta,
        Gz=cfg.lyapunov.Gz,
        pole_radius=args.pole_radius,
        w0=args.w0,
        Nitera=args.Nitera,
        Ndiscard=args.Ndiscard,
        perturbation=args.perturbation,
        n_initial=args.n_ci,
        seed=cfg.seed,
    )
    _print_ensemble_summary("4-D", ens4d, dim=4)

    # Dissipativity check (4-D): sum of exponents should approach ln|β|·(dim/2)
    expected_sum = np.log(abs(beta)) * 2.0  # 2 of the 4 exponents "feel" β
    actual_sum = float(ens4d.mean_exponents.sum())
    print(f"     Soma dos expoentes (média): {actual_sum:+.6f}")
    print(f"     2·ln|β| esperado (dissipação): {expected_sum:+.6f}")

    if args.save:
        out = ens4d.to_csv(data_dir / "henon4d_ensemble.csv")
        print(f"     Tabela por CI salva em: {out}")


if __name__ == "__main__":
    main()

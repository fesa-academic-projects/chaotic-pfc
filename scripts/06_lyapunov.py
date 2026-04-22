#!/usr/bin/env python3
"""
06_lyapunov.py — Lyapunov exponent computation (text output only, no figures).

Computes:
  Part A: Pure 2-D Hénon — both fixed points, eigenvalues, Lyapunov exponents
  Part B: 4-D Pole-filtered Hénon — fixed point, eigenvalues, Lyapunov exponents
"""
import argparse, sys
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--Nitera", type=int, default=2000)
    p.add_argument("--Ndiscard", type=int, default=1000)
    p.add_argument("--pole-radius", type=float, default=0.975)
    p.add_argument("--w0", type=float, default=0.0)
    p.add_argument("--save", action="store_true")       # kept for CLI compat
    p.add_argument("--no-display", action="store_true")  # kept for CLI compat
    return p.parse_args()


def main():
    args = parse_args()
    from chaotic_pfc.lyapunov import lyapunov_max, lyapunov_henon2d
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg

    a, b = cfg.comm.henon.a, cfg.comm.henon.b

    # ── Part A: Pure 2-D Hénon ──────────────────────────────────────────
    print(f"\n[06a] Lyapunov — Hénon puro 2-D  |  "
          f"Nitera={args.Nitera}  Ndiscard={args.Ndiscard}")

    res2d = lyapunov_henon2d(
        alpha=a, beta=b,
        Nitera=args.Nitera, Ndiscard=args.Ndiscard,
        perturbation=cfg.lyapunov.perturbation, seed=cfg.seed,
    )

    print(f"     Ponto fixo (+): {res2d['fixed_point_p']}")
    print(f"     Ponto fixo (−): {res2d['fixed_point_n']}")
    for label, key in [("(+)", "eigenvalues_p"), ("(−)", "eigenvalues_n")]:
        eigs = res2d[key]
        mods = np.abs(eigs)
        print(f"     Autovalores {label}: {eigs}")
        print(f"     |λ| {label}:         {mods}")
    print(f"     Estável (+): {res2d['stable_p']}   (−): {res2d['stable_n']}")
    print(f"     Expoentes de Lyapunov: {res2d['all_exponents']}")
    print(f"     λ_max = {res2d['lyapunov_max']:.6f}  "
          f"→ {'Caótico' if res2d['lyapunov_max'] > 0 else 'Não caótico'}")

    # ── Part B: 4-D Pole-filtered Hénon ─────────────────────────────────
    print(f"\n[06b] Lyapunov — Hénon filtrado 4-D  |  "
          f"r={args.pole_radius}  w0={args.w0}")

    res4d = lyapunov_max(
        alpha=a, beta=b,
        Gz=cfg.lyapunov.Gz, pole_radius=args.pole_radius, w0=args.w0,
        Nitera=args.Nitera, Ndiscard=args.Ndiscard,
        perturbation=cfg.lyapunov.perturbation, seed=cfg.seed,
    )

    print(f"     Ponto fixo:  {res4d['fixed_point']}")
    print(f"     Autovalores: {res4d['eigenvalues']}")
    print(f"     |λ|:         {np.abs(res4d['eigenvalues'])}")
    print(f"     Estável:     {res4d['stable']}")
    print(f"     Expoentes de Lyapunov: {res4d['all_exponents']}")
    print(f"     λ_max = {res4d['lyapunov_max']:.6f}  "
          f"→ {'Caótico' if res4d['lyapunov_max'] > 0 else 'Não caótico'}")


if __name__ == "__main__":
    main()

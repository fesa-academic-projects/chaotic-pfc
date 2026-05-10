"""Lyapunov exponent computation (text output + optional CSV).


* **Part A** — pure 2-D Henon, single perturbed IC, both fixed points.
* **Part B** — 4-D Pole-filtered Henon, single perturbed IC.
* **Part C** — 2-D Henon ensemble, ``n_ci`` ICs drawn uniformly in
  ``±perturbation`` around the fixed point.
* **Part D** — 4-D Pole-filtered ensemble, same protocol, 4-D system.

Parts A/B are quick sanity checks; parts C/D implement the full
experimental protocol. With ``--save`` the per-IC
tables are written to ``data/lyapunov/henon2d_ensemble.csv`` and
``data/lyapunov/henon4d_ensemble.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run lyapunov`` subcommand."""
    p = subparsers.add_parser(
        "lyapunov",
        help="Lyapunov spectra for single IC and N-IC ensemble (2-D and 4-D).",
        description=(
            "Lyapunov exponent computation: single perturbed IC (parts A, B) "
            "plus the N-IC ensemble protocol (parts C, D)."
        ),
    )
    p.add_argument("--Nitera", type=int, default=2000)
    p.add_argument("--Ndiscard", type=int, default=1000)
    p.add_argument("--pole-radius", type=float, default=0.975)
    p.add_argument("--w0", type=float, default=0.0)
    p.add_argument(
        "--n-ci",
        type=int,
        default=20,
        help="Number of ICs for parts C and D (default: 20)",
    )
    p.add_argument(
        "--perturbation",
        type=float,
        default=0.1,
        help="Half-width of sampling box around fixed point (default: 0.1 = +/-10%%)",
    )
    p.add_argument(
        "--data-dir",
        default="data/lyapunov",
        help="Directory for CSV output (default: data/lyapunov)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Write per-IC tables to CSV under --data-dir",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="(accepted for CLI consistency; this command has no UI)",
    )
    p.set_defaults(_run=run)


def _print_ensemble_summary(result, dim: int) -> None:
    """Compact summary of an EnsembleResult."""
    import numpy as np

    print(f"     Fixed point (sampling centre): {result.fixed_point}")
    print(f"     Eigenvalues: {result.eigenvalues}")
    print(f"     |lambda|:    {np.abs(result.eigenvalues)}")
    print(f"     Stable:      {result.stable}")
    print(f"     Number of ICs: {len(result.lmax_per_ci)}")

    mean_str = "  ".join(f"{v:+.6f}" for v in result.mean_exponents)
    print(f"     Mean exponents (lambda_1..lambda_{dim}): {mean_str}")
    print(f"     Mean lambda_max:                         {result.mean_lmax:+.6f}")
    print(f"     Max  lambda_max:                         {result.max_lmax:+.6f}")

    total = result.n_chaotic + result.n_stable
    print(f"     Chaotic trajectories: {result.n_chaotic}/{total}")
    print(f"     Stable trajectories:  {result.n_stable}/{total}")

    verdict = "CHAOTIC" if result.mean_lmax > 0 else "STABLE"
    print(f"     Diagnosis: {verdict} (based on mean lambda_max)")


def run(args: argparse.Namespace) -> int:
    """Execute the ``lyapunov`` experiment."""
    import numpy as np

    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.dynamics.lyapunov import (
        lyapunov_henon2d,
        lyapunov_henon2d_ensemble,
        lyapunov_max,
        lyapunov_max_ensemble,
    )

    alpha, beta = cfg.comm.henon.a, cfg.comm.henon.b
    data_dir = Path(args.data_dir)

    # ─── Part A: Pure 2-D Henon (single IC) ────────────────────────────────
    print(
        f"\n[06a] Lyapunov — pure Henon 2-D (1 perturbed IC)  |  "
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

    print(f"     Fixed point (+): {res2d.fixed_point_p}")
    print(f"     Fixed point (-): {res2d.fixed_point_n}")
    for label, eigs in [("(+)", res2d.eigenvalues_p), ("(-)", res2d.eigenvalues_n)]:
        assert eigs is not None
        print(f"     Eigenvalues {label}: {eigs}")
        print(f"     |lambda| {label}:     {np.abs(eigs)}")
    print(f"     Stable (+): {res2d.stable_p}   (-): {res2d.stable_n}")
    print(f"     Lyapunov exponents: {res2d.all_exponents}")
    print(
        f"     lambda_max = {res2d.lyapunov_max:.6f}  "
        f"→ {'Chaotic' if res2d.lyapunov_max > 0 else 'Non-chaotic'}"
    )

    # ─── Part B: 4-D Pole-filtered Henon (single IC) ───────────────────────
    print(
        f"\n[06b] Lyapunov — filtered Henon 4-D (1 perturbed IC)  |  "
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

    print(f"     Fixed point:  {res4d.fixed_point}")
    print(f"     Eigenvalues:  {res4d.eigenvalues}")
    assert res4d.eigenvalues is not None
    print(f"     |lambda|:     {np.abs(res4d.eigenvalues)}")
    print(f"     Stable:       {res4d.stable}")
    print(f"     Lyapunov exponents: {res4d.all_exponents}")
    print(
        f"     lambda_max = {res4d.lyapunov_max:.6f}  "
        f"→ {'Chaotic' if res4d.lyapunov_max > 0 else 'Non-chaotic'}"
    )

    # ─── Part C: 2-D Henon ensemble (N_ci ICs +/-perturbation) ─────────────
    print(
        f"\n[06c] Lyapunov — pure Henon 2-D (ensemble)  |  "
        f"N_ci={args.n_ci}  +/-{args.perturbation * 100:.0f}%"
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
    _print_ensemble_summary(ens2d, dim=2)

    if args.save:
        out = ens2d.to_csv(data_dir / "henon2d_ensemble.csv")
        print(f"     Per-IC table saved to: {out}")

    # ─── Part D: 4-D Pole-filtered ensemble ────────────────────────────────
    print(
        f"\n[06d] Lyapunov — filtered Henon 4-D (ensemble)  |  "
        f"N_ci={args.n_ci}  +/-{args.perturbation * 100:.0f}%  "
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
    _print_ensemble_summary(ens4d, dim=4)

    expected_sum = np.log(abs(beta)) * 2.0  # 2 of the 4 exponents "feel" beta
    actual_sum = float(ens4d.mean_exponents.sum())
    print(f"     Sum of exponents (mean): {actual_sum:+.6f}")
    print(f"     2*ln|beta| expected (dissipation): {expected_sum:+.6f}")

    if args.save:
        out = ens4d.to_csv(data_dir / "henon4d_ensemble.csv")
        print(f"     Per-IC table saved to: {out}")
    return 0

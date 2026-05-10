"""Sensitive Dependence on Initial Conditions (SDIC)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import add_lang_flag, add_save_display_flags, pick_backend


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run sensitivity`` subcommand."""
    p = subparsers.add_parser(
        "sensitivity",
        help="Sensitive Dependence on Initial Conditions (SDIC).",
        description=(
            "Overlay two Hénon trajectories with infinitesimally different "
            "initial conditions to visualise exponential divergence."
        ),
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--epsilon", type=float, default=1e-4)
    add_save_display_flags(p)
    add_lang_flag(p)
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``sensitivity`` experiment."""
    headless = pick_backend(args.no_display)

    import matplotlib.pyplot as plt
    import numpy as np

    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.dynamics.maps import henon_standard
    from chaotic_pfc.plotting.figures import plot_sensitivity

    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    fmt = cfg.plot.fmt
    fdir = Path(cfg.plot.figures_dir)
    if args.save:
        fdir.mkdir(parents=True, exist_ok=True)

    print(f"[02] SDIC  |  steps={args.steps}  ε={args.epsilon:.0e}")

    X1, _ = henon_standard(args.steps, x0=0.0, y0=0.0, a=a, b=b)
    X2, _ = henon_standard(args.steps, x0=args.epsilon, y0=args.epsilon, a=a, b=b)
    n = np.arange(args.steps + 1)

    save_path = str(fdir / f"sensitivity.{fmt}") if args.save else None
    fig = plot_sensitivity(n, X1, X2, save_path=save_path, lang=args.lang)

    if headless:
        plt.close(fig)
        if args.save:
            print(f"    Saved -> {save_path}")
    else:
        plt.show()
    return 0

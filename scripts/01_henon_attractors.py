#!/usr/bin/env python3
"""01_henon_attractors.py — Phase-space attractors for three Hénon variants."""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-display", dest="no_display", action="store_true")
    return p.parse_args()


def _backend(nd):
    hl = nd or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
    if hl:
        import matplotlib

        matplotlib.use("Agg")
    return hl


def main():
    args = parse_args()
    headless = _backend(args.no_display)
    import matplotlib.pyplot as plt

    from chaotic_pfc.config import DEFAULT_CONFIG as cfg
    from chaotic_pfc.maps import henon_filtered, henon_generalised, henon_standard
    from chaotic_pfc.plotting import plot_attractor

    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    steps, fmt = args.steps, cfg.plot.fmt
    fdir = Path(cfg.plot.figures_dir)
    if args.save:
        fdir.mkdir(parents=True, exist_ok=True)

    def sp(name):
        return str(fdir / name) if args.save else None

    print(f"[01] Hénon attractors  |  steps={steps:,}")

    X, Y = henon_standard(steps, a=a, b=b)
    f1 = plot_attractor(
        X,
        Y,
        title=r"Atrator de Hénon Padrão ($a=1.4,\; b=0.3$)",
        xlabel=r"$x[n]$",
        ylabel=r"$y[n]$",
        save_path=sp(f"attractor_standard.{fmt}"),
    )

    X, Y = henon_generalised(steps, alpha=a, beta=b)
    f2 = plot_attractor(
        X,
        Y,
        title=r"Atrator de Hénon Generalizado ($\alpha=1.4,\; \beta=0.3$)",
        xlabel=r"$x_1[n]$",
        ylabel=r"$x_2[n]$",
        save_path=sp(f"attractor_generalised.{fmt}"),
    )

    X, Y = henon_filtered(steps, alpha=a, beta=b, c0=1.0, c1=0.0)
    f3 = plot_attractor(
        X,
        Y,
        title=r"Atrator de Hénon Filtrado ($c_0=1,\; c_1=0$)",
        xlabel=r"$x_1[n]$",
        ylabel=r"$x_2[n]$",
        save_path=sp(f"attractor_filtered.{fmt}"),
    )

    if headless:
        for f in [f1, f2, f3]:
            plt.close(f)
        if args.save:
            print(f"    Saved -> {fdir}/attractor_*.{fmt}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

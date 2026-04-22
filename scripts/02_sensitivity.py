#!/usr/bin/env python3
"""02_sensitivity.py — Sensitive Dependence on Initial Conditions (SDIC)."""
import argparse, os, sys
from pathlib import Path
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--epsilon", type=float, default=1e-4)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-display", dest="no_display", action="store_true")
    return p.parse_args()

def _backend(nd):
    hl = nd or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
    if hl: import matplotlib; matplotlib.use("Agg")
    return hl

def main():
    args = parse_args(); headless = _backend(args.no_display)
    import matplotlib.pyplot as plt
    from chaotic_pfc.maps import henon_standard
    from chaotic_pfc.plotting import plot_sensitivity
    from chaotic_pfc.config import DEFAULT_CONFIG as cfg

    a, b = cfg.comm.henon.a, cfg.comm.henon.b
    fmt = cfg.plot.fmt
    fdir = Path(cfg.plot.figures_dir)
    if args.save: fdir.mkdir(parents=True, exist_ok=True)

    print(f"[02] SDIC  |  steps={args.steps}  ε={args.epsilon:.0e}")

    X1, _ = henon_standard(args.steps, x0=0.0, y0=0.0, a=a, b=b)
    X2, _ = henon_standard(args.steps, x0=args.epsilon, y0=args.epsilon, a=a, b=b)
    n = np.arange(args.steps + 1)

    sp = str(fdir / f"sensitivity.{fmt}") if args.save else None
    fig = plot_sensitivity(n, X1, X2, save_path=sp)

    if headless:
        plt.close(fig)
        if args.save: print(f"    Saved -> {sp}")
    else: plt.show()

if __name__ == "__main__": main()

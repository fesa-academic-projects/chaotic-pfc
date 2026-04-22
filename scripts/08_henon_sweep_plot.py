#!/usr/bin/env python3
"""
08_henon_sweep_plot.py
======================
Generate the four standard classification figures for one or more sweeps
previously produced by ``07_henon_sweep_compute.py``.

By default, both PNG and SVG versions are generated side-by-side: PNG for
quick browsing and GitHub preview, SVG for the paper.

Examples
--------
Plot a single configuration in both formats (default)::

    python scripts/08_henon_sweep_plot.py --window hamming --filter lowpass

Plot every sweep found under the data directory::

    python scripts/08_henon_sweep_plot.py --all

Restrict to a single format::

    python scripts/08_henon_sweep_plot.py --all --fmt svg

Custom input/output roots::

    python scripts/08_henon_sweep_plot.py --all \\
        --data-dir data/sweeps --figures-dir figures/sweeps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    from chaotic_pfc.sweep import FILTER_TYPES, WINDOWS

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--window", choices=WINDOWS, default="hamming", help="FIR window (default: hamming)"
    )
    p.add_argument(
        "--filter",
        choices=FILTER_TYPES,
        default="lowpass",
        dest="filter_type",
        help="Filter pass-zero configuration (default: lowpass)",
    )
    p.add_argument("--all", action="store_true", help="Plot every .npz found under --data-dir")
    p.add_argument(
        "--data-dir",
        default="data/sweeps",
        help="Root directory with .npz files (default: data/sweeps)",
    )
    p.add_argument(
        "--figures-dir",
        default="figures/sweeps",
        help="Root directory for output figures (default: figures/sweeps)",
    )
    p.add_argument(
        "--fmt",
        nargs="+",
        default=["png", "svg"],
        choices=("png", "svg", "pdf"),
        help="Output format(s). Pass multiple values to generate "
        "several at once, e.g. '--fmt png svg'. "
        "Default: png svg",
    )
    # Compatibility flags — kept consistent with the other scripts.
    p.add_argument(
        "--save",
        action="store_true",
        help="(accepted for CLI consistency; figures are always saved)",
    )
    p.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="(accepted for CLI consistency; matplotlib runs headless)",
    )
    return p.parse_args()


def _force_headless_if_needed(no_display: bool) -> None:
    """Pick a headless matplotlib backend before pyplot is imported."""
    headless = no_display or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
    if headless:
        import matplotlib

        matplotlib.use("Agg")


def _discover_sweeps(data_dir: Path) -> list[Path]:
    """Find every variables_lyapunov.npz under data_dir, sorted."""
    return sorted(data_dir.rglob("variables_lyapunov.npz"))


def _target_dir(figures_dir: Path, npz_path: Path, data_dir: Path) -> Path:
    """Mirror the data-dir subpath into figures-dir.

    ``data/sweeps/Hamming (lowpass)/variables_lyapunov.npz`` →
    ``figures/sweeps/Hamming (lowpass)/``.
    """
    try:
        rel = npz_path.parent.relative_to(data_dir)
    except ValueError:
        rel = Path(npz_path.parent.name)
    return figures_dir / rel


def main() -> None:
    args = parse_args()
    _force_headless_if_needed(args.no_display)

    from chaotic_pfc.sweep import WINDOW_DISPLAY_NAMES, load_sweep
    from chaotic_pfc.sweep_plotting import plot_all

    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)

    if args.all:
        npz_paths = _discover_sweeps(data_dir)
        if not npz_paths:
            print(
                f"[08] No .npz files found under {data_dir}/. Run 07_henon_sweep_compute.py first."
            )
            return
    else:
        pretty = WINDOW_DISPLAY_NAMES.get(args.window, args.window.capitalize())
        subdir = f"{pretty} ({args.filter_type})"
        candidate = data_dir / subdir / "variables_lyapunov.npz"
        if not candidate.exists():
            print(f"[08] Not found: {candidate}")
            print(
                f"     Run 07_henon_sweep_compute.py "
                f"--window {args.window} --filter {args.filter_type}"
            )
            sys.exit(1)
        npz_paths = [candidate]

    fmts_str = ", ".join(f".{f}" for f in args.fmt)
    print(f"[08] Plotting {len(npz_paths)} sweep(s) (formats: {fmts_str})")

    for npz_path in npz_paths:
        result = load_sweep(npz_path)
        out_dir = _target_dir(figures_dir, npz_path, data_dir)
        print(f"\n     {result.display_name}")
        print(f"     ← {npz_path}")
        print(f"     → {out_dir}")
        for fmt in args.fmt:
            paths = plot_all(result, out_dir, fmt=fmt)
            for p in paths:
                print(f"       {p.name}")

    print("\n[08] done")


if __name__ == "__main__":
    main()

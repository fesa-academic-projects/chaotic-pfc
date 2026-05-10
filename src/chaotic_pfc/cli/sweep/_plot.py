from __future__ import annotations

import argparse
from pathlib import Path

from .._common import pick_backend


def _add_plot_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``run sweep plot``."""
    from chaotic_pfc.analysis.sweep import FILTER_TYPES, WINDOWS

    p = subparsers.add_parser(
        "plot",
        help="Plot the four standard classification figures from a saved sweep.",
        description="Generate classification figures from previously saved sweep .npz files.",
    )
    p.add_argument(
        "--window",
        choices=WINDOWS,
        default="hamming",
        help="FIR window (default: hamming)",
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
        help="Output format(s). Multiple values allowed, e.g. '--fmt png svg'.",
    )
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
    p.set_defaults(_run=run_plot)


# ════════════════════════════════════════════════════════════════════════════
# run sweep plot
# ════════════════════════════════════════════════════════════════════════════


def _discover_sweeps(data_dir: Path) -> list[Path]:
    """Find every ``variables_lyapunov.npz`` under ``data_dir``, sorted."""
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


def run_plot(args: argparse.Namespace) -> int:
    """Execute ``run sweep plot``."""
    pick_backend(args.no_display)

    from chaotic_pfc.analysis.sweep import WINDOW_DISPLAY_NAMES, load_sweep
    from chaotic_pfc.analysis.sweep_plotting import plot_all

    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)

    if args.all:
        npz_paths = _discover_sweeps(data_dir)
        if not npz_paths:
            print(
                f"[08] No .npz files found under {data_dir}/. "
                "Run 'chaotic-pfc run sweep compute' first."
            )
            return 0
    else:
        pretty = WINDOW_DISPLAY_NAMES.get(args.window, args.window.capitalize())
        subdir = f"{pretty} ({args.filter_type})"
        candidate = data_dir / subdir / "variables_lyapunov.npz"
        if not candidate.exists():
            print(f"[08] Not found: {candidate}")
            print(
                f"     Run: chaotic-pfc run sweep compute "
                f"--window {args.window} --filter {args.filter_type}"
            )
            return 1
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
    return 0

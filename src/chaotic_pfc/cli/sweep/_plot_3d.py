from __future__ import annotations

import argparse
from pathlib import Path


def _add_plot_3d_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``run sweep plot-3d``."""
    p = subparsers.add_parser(
        "plot-3d",
        help="Render an interactive 3-D surface stack of Kaiser β-sweeps.",
        description=(
            "Aggregate all per-β .npz files under --data-dir into a single "
            "3-D volume λ_max(N_z, ω_c, β) and save an interactive HTML figure."
        ),
    )
    p.add_argument(
        "--data-dir",
        default="data/sweeps/kaiser",
        help="Root directory with per-β .npz files (default: data/sweeps/kaiser)",
    )
    p.add_argument(
        "--figures-dir",
        default="figures/sweeps",
        help="Root directory for the output HTML (default: figures/sweeps)",
    )
    p.add_argument(
        "--filter-type",
        default="lowpass",
        dest="filter_type",
        help="Filter type subdirectory under --data-dir (default: lowpass)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Generate 3-D plots for every filter type found under --data-dir",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="(accepted for CLI consistency; figure is always saved)",
    )
    p.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        help="(accepted for CLI consistency; plotly runs headless)",
    )
    p.set_defaults(_run=run_plot_3d)


def run_plot_3d(args: argparse.Namespace) -> int:
    """Execute ``run sweep plot-3d``."""
    from chaotic_pfc.analysis.sweep_plotting_3d import (
        aggregate_beta_sweeps,
        plot_3d_beta_volume,
    )

    base_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        filter_types = [d.name for d in sorted(base_dir.iterdir()) if d.is_dir()]
        if not filter_types:
            print(f"[plot-3d] No filter type directories found under {base_dir}")
            return 1
    else:
        filter_types = [args.filter_type]

    for ft in filter_types:
        data_dir = base_dir / ft
        if not data_dir.is_dir():
            print(f"[plot-3d] Skipping {ft}: directory not found")
            continue

        out_html = figures_dir / f"beta_sweep_3d_{ft}.html"
        print(f"[plot-3d] Aggregating {ft} sweeps from {data_dir}")
        h_volume, betas, orders, cutoffs = aggregate_beta_sweeps(data_dir)
        print(f"[plot-3d] Loaded {len(betas)} β values: {betas[0]:.1f}..{betas[-1]:.1f}")
        print(f"[plot-3d] Volume shape: {h_volume.shape}")

        plot_3d_beta_volume(h_volume, betas, orders, cutoffs, save_path=out_html)
        print(f"[plot-3d] Saved → {out_html}")

    print("[plot-3d] done")
    return 0

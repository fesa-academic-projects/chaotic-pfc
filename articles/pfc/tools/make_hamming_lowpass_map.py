#!/usr/bin/env python3
"""Render the Hamming lowpass classification map with the case marker."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chaotic_pfc.analysis.sweep import load_sweep
from chaotic_pfc.analysis.sweep_plotting import plot_classification_interleaved
from chaotic_pfc.paper.figures import JCIS_FIG

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP = ROOT / "data/sweeps/Hamming (lowpass)/variables_lyapunov.npz"
DEFAULT_OUT = ROOT / "articles/pfc/figures/hamming_lowpass_classification.pdf"


def main() -> int:
    """Run the figure generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sweep = load_sweep(args.sweep)
    opts = {
        **JCIS_FIG,
        "figsize": (6.0, 3.8),
        "label_fontsize": 11.0,
        "tick_fontsize": 9.0,
        "legend_fontsize": 8.0,
        "ytick_step": 0.2,
        "xtick_step": 10,
        "legend": True,
        "lang": "en",
    }
    fig = plot_classification_interleaved(sweep, **opts)

    target_nz = 8
    target_cutoff = 0.52
    row = int(np.flatnonzero(np.asarray(sweep.num_taps_grid) == target_nz + 1)[0])
    slot_total = 4
    x = row * slot_total - 0.5 + 1.5
    ax = fig.axes[0]
    ax.plot(
        [x],
        [target_cutoff],
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.2,
        linestyle="none",
        zorder=20,
    )
    ax.annotate(
        "case",
        xy=(x, target_cutoff),
        xytext=(x - 7.0, target_cutoff + 0.12),
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "lw": 0.5},
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "black"},
        zorder=21,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.out,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "chaotic-pfc", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    nearest = int(np.argmin(np.abs(np.asarray(sweep.cutoff_grid) - target_cutoff)))
    cutoff = float(sweep.cutoff_grid[nearest])
    value = float(sweep.h[row, nearest])
    print(f"wrote {args.out}")
    print(
        f"Nz={target_nz} cutoff={cutoff:.10f} distance={abs(cutoff - target_cutoff):.10f} "
        f"lambda1={value:.10f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

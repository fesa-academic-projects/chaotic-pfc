"""Camera-ready figure generation for the SIMAC and JCIS manuscripts.

Figures are rendered at their *final printed size*, so the font sizes
requested here are the font sizes that end up on the page. This is the
only way to keep figure typography consistent with the body text: a
12 x 5 in figure squeezed into a 3.5 in column shrinks a 16 pt label
down to 4.7 pt.

Page geometry
-------------
SIMAC  ``\\documentclass[12pt,a4paper]{article}`` with 1.5 cm side
       margins -> ``\\textwidth`` = 18 cm = 7.087 in. Panels are laid
       out two per row at ``0.48\\linewidth`` = 3.40 in.
JCIS   ``\\documentclass[journal]{IEEEtran}`` two-column ->
       ``\\columnwidth`` = 3.5 in.

Exposed through the CLI as ``chaotic-pfc run paper-figures`` (see
:func:`add_parser`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict


class _PanelOptions(TypedDict, total=False):
    figsize: tuple[float, float]
    label_fontsize: float
    tick_fontsize: float
    legend_fontsize: float
    ytick_step: float
    xtick_step: int
    legend: bool
    lang: str
    panel_label_loc: str


# ── SIMAC: 0.48\linewidth of an 18 cm text block ────────────────────────────
SIMAC_W = 3.40
SIMAC_PANEL: _PanelOptions = dict(
    figsize=(SIMAC_W, 2.10),
    label_fontsize=10.0,
    tick_fontsize=8.5,
    ytick_step=0.2,
    xtick_step=10,
    legend=False,
    lang="pt",
    panel_label_loc="above left",
)

# ── JCIS: one IEEEtran column ───────────────────────────────────────────────
JCIS_W = 3.50
JCIS_FIG: _PanelOptions = dict(
    figsize=(JCIS_W, 2.30),
    label_fontsize=9.0,
    tick_fontsize=8.0,
    legend_fontsize=7.5,
    ytick_step=0.2,
    xtick_step=10,
    legend=True,
    lang="en",
)

SIMAC_PANELS = [
    ("Hamming (highpass)", "simac_fig1a_hamming_highpass", "(a)"),
    ("Hamming (bandpass)", "simac_fig1b_hamming_bandpass", "(b)"),
    ("Hamming (bandstop)", "simac_fig1c_hamming_bandstop", "(c)"),
]

JCIS_FIGURES = [
    ("kaiser/lowpass/beta_13.50", "kaiser_b13p5_lowpass_classification_interleaved"),
    ("kaiser/highpass/beta_11.50", "kaiser_b11p5_highpass_classification_interleaved"),
    ("kaiser/bandpass/beta_13.50", "kaiser_b13p5_bandpass_classification_interleaved"),
    ("Rectangular (bandstop)", "rectangular_bandstop_classification_interleaved"),
]

FORMATS = ("pdf", "svg")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run paper-figures`` subcommand."""
    p = subparsers.add_parser(
        "paper-figures",
        help="Render camera-ready SIMAC and JCIS figures at final printed size.",
        description=(
            "Render the SIMAC and JCIS figures at their final printed size. "
            "Requires the sweep checkpoints under --data-dir."
        ),
    )
    p.add_argument(
        "--data-dir",
        default="data/sweeps",
        help="Root sweep directory (default: data/sweeps)",
    )
    p.add_argument(
        "--output-dir",
        default="paper-figures",
        help="Output directory (default: paper-figures)",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=list(FORMATS),
        help="Output formats to emit (default: pdf svg)",
    )
    p.add_argument(
        "--only",
        choices=["all", "simac", "jcis"],
        default="all",
        help="Limit generation to one manuscript (default: all)",
    )
    p.set_defaults(_run=run)


def run(args: argparse.Namespace) -> int:
    """Execute the ``paper-figures`` experiment."""
    import matplotlib.pyplot as plt

    from chaotic_pfc.analysis.sweep import load_sweep
    from chaotic_pfc.analysis.sweep_plotting import (
        make_classification_legend,
        plot_chaotic_density,
        plot_classification_interleaved,
    )
    from chaotic_pfc.plotting.figures import setup_rc

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    formats = tuple(args.formats)

    def _sweep(name: str):
        path = data_dir / name / "variables_lyapunov.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        return load_sweep(path)

    def _emit(fig, out_dir: Path, stem: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fig.savefig(
                out_dir / f"{stem}.{fmt}",
                dpi=600,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
            )
        plt.close(fig)

    setup_rc()
    if args.only in ("all", "simac"):
        out_dir = out / "simac"
        for sweep_name, stem, label in SIMAC_PANELS:
            fig = plot_classification_interleaved(
                _sweep(sweep_name), panel_label=label, **SIMAC_PANEL
            )
            _emit(fig, out_dir, stem)
        fig = make_classification_legend(lang="pt", figsize=(SIMAC_W, 1.60), fontsize=10.0, ncol=1)
        _emit(fig, out_dir, "simac_fig1_legenda")

    if args.only in ("all", "jcis"):
        out_dir = out / "jcis"
        for sweep_name, stem in JCIS_FIGURES:
            fig = plot_classification_interleaved(_sweep(sweep_name), **JCIS_FIG)
            _emit(fig, out_dir, stem)

        # Titles are carried by the LaTeX \caption, not baked into the
        # artwork: an embedded title duplicates the caption, sets it in
        # the wrong typeface, and cannot be edited without re-running Python.
        fig = plot_chaotic_density(
            data_dir,
            lang="en",
            figsize=(JCIS_W, 2.30),
            label_fontsize=9.0,
            tick_fontsize=8.0,
            cbar_fontsize=8.0,
            show_title=False,
            marker_size=6.0,
            ytick_step=0.2,
            xtick_step=10,
        )
        _emit(fig, out_dir, "fig_chaotic_density")

    print(f"figures written to {out}")
    return 0

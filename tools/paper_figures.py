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

Run from the repository root::

    python tools/paper_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CHAOTIC_PFC_FORCE_LATEX", "0")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from chaotic_pfc.analysis.sweep import load_sweep
from chaotic_pfc.analysis.sweep_plotting import (
    make_classification_legend,
    plot_chaotic_density,
    plot_classification_interleaved,
)
from chaotic_pfc.plotting.figures import setup_rc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sweeps"
OUT = ROOT / "paper-figures"

# ── SIMAC: 0.48\linewidth of an 18 cm text block ────────────────────────────
SIMAC_W = 3.40
SIMAC_PANEL = dict(
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
JCIS_FIG = dict(
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


def _emit(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        fig.savefig(
            out_dir / f"{stem}.{fmt}",
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


def _sweep(name: str):
    path = DATA / name / "variables_lyapunov.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_sweep(path)


def build_simac() -> None:
    out = OUT / "simac"
    for sweep_name, stem, label in SIMAC_PANELS:
        fig = plot_classification_interleaved(_sweep(sweep_name), panel_label=label, **SIMAC_PANEL)
        _emit(fig, out, stem)
    fig = make_classification_legend(lang="pt", figsize=(SIMAC_W, 1.60), fontsize=10.0, ncol=1)
    _emit(fig, out, "simac_fig1_legenda")


def build_jcis() -> None:
    out = OUT / "jcis"
    for sweep_name, stem in JCIS_FIGURES:
        fig = plot_classification_interleaved(_sweep(sweep_name), **JCIS_FIG)
        _emit(fig, out, stem)

    # Titles are carried by the LaTeX \caption, not baked into the
    # artwork: an embedded title duplicates the caption, sets it in the
    # wrong typeface, and cannot be edited without re-running Python.
    fig = plot_chaotic_density(
        DATA,
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
    _emit(fig, out, "fig_chaotic_density")


def main() -> None:
    setup_rc()
    build_simac()
    build_jcis()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()

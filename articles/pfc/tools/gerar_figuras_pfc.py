#!/usr/bin/env python3
r"""Generate the PFC data figures in Portuguese, with comma decimals.

The figures are rebuilt from the versioned sweeps in ``data/sweeps`` and from
the communication pipeline of ``chaotic_pfc``. Nothing is recomputed from a
different data set: the sweeps and the modulator/receiver are the same used by
the manuscript. Output names match the ``\\includegraphics`` calls of
``artigo.tex``.

Usage::

    python articles/pfc/tools/gerar_figuras_pfc.py
    python articles/pfc/tools/gerar_figuras_pfc.py --only classification
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.signal import firwin

from chaotic_pfc._rounding import round_decimal
from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep
from chaotic_pfc.analysis.sweep_plotting import (
    plot_chaotic_density_norm,
    plot_classification_interleaved,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    COLOR_CHAOTIC,
    COLOR_PERIODIC,
    COLOR_UNBOUNDED,
)
from chaotic_pfc.comms.channel import fir_channel
from chaotic_pfc.comms.receiver import receive, receive_order_n
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.config import DEFAULT_CONFIG as CFG
from chaotic_pfc.dynamics.signals import text_message
from chaotic_pfc.dynamics.spectral import psd_normalised

ROOT = Path(__file__).resolve().parents[3]
SWEEPS = ROOT / "data/sweeps"
OUT = ROOT / "articles/pfc/figures"

CLASSIFICATION = (
    ("kaiser/lowpass/beta_17.00", "kaiser_b17p0_lowpass_classification_interleaved"),
    ("kaiser/highpass/beta_15.50", "kaiser_b15p5_highpass_classification_interleaved"),
    ("kaiser/bandpass/beta_15.50", "kaiser_b15p5_bandpass_classification_interleaved"),
    ("Rectangular (bandstop)", "rectangular_bandstop_classification_interleaved"),
)

TYPE_LABELS = {
    "lowpass": "Passa-baixas",
    "highpass": "Passa-altas",
    "bandpass": "Passa-faixa",
    "bandstop": "Rejeita-faixa",
}

COMM_WINDOW = (2000, 2400)
FILTER_CUTOFF = 0.52
FILTER_TAPS = 9


def setup() -> None:
    """Configure a serif face close to the article body and vector PDFs."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Liberation Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )


def comma(value: float, _pos: float | None = None) -> str:
    """Format a tick with a decimal comma and a typographic minus."""
    text = f"{value:g}"
    return text.replace(".", ",").replace("-", "\u2212")


def save(fig: plt.Figure, path: Path) -> None:
    """Write a deterministic PDF without metadata dates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "chaotic-pfc", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def legend_handles() -> list[Patch]:
    """Return the classification key in the order requested by the text."""
    return [
        Patch(facecolor=COLOR_CHAOTIC, edgecolor="gray", label="Caótico"),
        Patch(facecolor=COLOR_PERIODIC, edgecolor="gray", label="Não caótico"),
        Patch(facecolor=COLOR_UNBOUNDED, edgecolor="gray", label="Divergente"),
    ]


def style_classification(fig: plt.Figure, *, legend: bool) -> None:
    """Apply comma ticks and the Portuguese legend to a classification panel."""
    ax = fig.axes[0]
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    if legend:
        ax.legend(
            handles=legend_handles(),
            loc="upper right",
            fontsize=6.5,
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
            handlelength=1.2,
            handleheight=0.8,
            borderpad=0.4,
        )


def classification() -> None:
    """Write the four classification maps."""
    for k, (name, stem) in enumerate(CLASSIFICATION):
        sweep = load_sweep(SWEEPS / name / "variables_lyapunov.npz")
        fig = plot_classification_interleaved(
            sweep,
            lang="pt",
            figsize=(3.6, 2.4),
            label_fontsize=9.0,
            tick_fontsize=8.0,
            legend=False,
            ytick_step=0.2,
            xtick_step=10,
        )
        style_classification(fig, legend=k == 0)
        save(fig, OUT / f"{stem}.pdf")


def density() -> None:
    """Write the normalised chaotic-density map."""
    fig = plot_chaotic_density_norm(
        SWEEPS,
        lang="pt",
        figsize=(6.2, 3.4),
        label_fontsize=9.0,
        tick_fontsize=8.0,
        cbar_fontsize=8.0,
        show_title=False,
        ytick_step=0.2,
        xtick_step=10,
        strict=True,
    )
    ax = fig.axes[0]
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    for line in list(ax.lines):
        line.remove()
    if len(fig.axes) > 1:
        cbar = fig.axes[1]
        cbar.yaxis.set_major_formatter(FuncFormatter(comma))
        cbar.set_ylabel("porcentagem das configurações (%)", fontsize=8.0)
    fig.tight_layout()
    save(fig, OUT / "fig_chaotic_density_norm.pdf")


def kaiser() -> None:
    """Write the chaotic coverage as a function of the Kaiser beta."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for ft in ("lowpass", "highpass", "bandpass", "bandstop"):
        rows = []
        for path in sorted((SWEEPS / "kaiser" / ft).glob("beta_*/variables_lyapunov.npz")):
            sweep = load_sweep(path)
            beta = float(sweep.metadata["kaiser_beta"])
            h = np.asarray(sweep.h, dtype=np.float64)
            chaotic = int(np.count_nonzero(h > LAMBDA_CHAOS_THRESHOLD))
            rows.append((beta, 100.0 * chaotic / h.size))
        rows.sort()
        betas = np.array([row[0] for row in rows])
        coverage = np.array([row[1] for row in rows])
        ax.plot(betas, coverage, marker="o", ms=2.4, lw=0.9, label=TYPE_LABELS[ft])
        best = int(np.argmax(coverage))
        print(
            f"[kaiser] {ft}: max {round_decimal(float(coverage[best]), 1)}% at beta_K={betas[best]:g}; "
            f"beta=0.5 -> {round_decimal(float(coverage[0]), 1)}%, "
            f"beta=20 -> {round_decimal(float(coverage[-1]), 1)}%"
        )
    ax.set_xlabel(r"$\beta_K$")
    ax.set_ylabel("cobertura caótica (%)")
    ax.set_xlim(0.0, 20.5)
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, OUT / "fig_kaiser_beta.pdf")


def hamming() -> None:
    """Write the Hamming lowpass map with the chosen project marked."""
    sweep = load_sweep(SWEEPS / "Hamming (lowpass)" / "variables_lyapunov.npz")
    fig = plot_classification_interleaved(
        sweep,
        lang="pt",
        figsize=(3.6, 1.9),
        label_fontsize=8.5,
        tick_fontsize=7.0,
        legend=False,
        ytick_step=0.2,
        xtick_step=10,
    )
    style_classification(fig, legend=True)
    ax = fig.axes[0]
    target = 8
    row = int(np.flatnonzero(np.asarray(sweep.num_taps_grid) == target + 1)[0])
    x = row * 4 + 1
    ax.plot(
        [x],
        [FILTER_CUTOFF],
        marker="o",
        markersize=5.5,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.2,
        linestyle="none",
        zorder=20,
    )
    fig.tight_layout()
    save(fig, OUT / "hamming_lowpass_classification.pdf")

    nearest = int(np.argmin(np.abs(np.asarray(sweep.cutoff_grid) - FILTER_CUTOFF)))
    print(
        f"[hamming] Nz={target} cutoff={round_decimal(float(sweep.cutoff_grid[nearest]), 10)} "
        f"lambda1={round_decimal(float(sweep.h[row, nearest]), 10)}"
    )


def _psd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = CFG.spectral
    omega, values = psd_normalised(
        signal,
        spec.nfft,
        spec.window_length,
        window=spec.window,
        kaiser_beta=spec.kaiser_beta,
    )
    return np.asarray(omega), np.asarray(values)


def _panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
    xlabel: str,
    reference: np.ndarray | None = None,
    reference_label: str = "",
    solid_label: str = "mensagem recuperada",
    ylim: tuple[float, float] | None = None,
) -> None:
    if reference is not None:
        ax.plot(x, reference, color="black", lw=0.6, ls="--", label=reference_label)
    ax.plot(x, y, color="tab:blue", lw=0.7, label=solid_label if reference is not None else None)
    ax.set_title(title, loc="left", fontsize=6.5, pad=2)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.set_xlabel(xlabel, fontsize=6.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.grid(alpha=0.25, lw=0.4)


def _comm_figure(
    message: np.ndarray,
    sent: np.ndarray,
    received: np.ndarray,
    recovered: np.ndarray,
    path: Path,
    *,
    clip_recovered: bool,
) -> None:
    delay = (CFG.channel.num_taps - 1) // 2
    start, stop = COMM_WINDOW
    n = np.arange(start, stop)
    delayed = message[n - delay]

    fig, axes = plt.subplots(4, 2, figsize=(5.2, 3.8))
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")
    rows = (
        ("mensagem", message, False),
        ("sinal transmitido", sent, False),
        ("sinal recebido", received, False),
        ("mensagem recuperada", recovered, clip_recovered),
    )
    for row, (label, signal, clipped) in enumerate(rows):
        ylim = (-3.0, 3.0) if clipped else None
        suffix = " (limitada a ±3)" if clipped else ""
        reference = delayed if row == 3 else None
        _panel(
            axes[row, 0],
            n,
            signal[n],
            title=f"{panel_labels[row]} {label}{suffix}",
            ylabel="amplitude",
            xlabel="amostra",
            reference=reference,
            reference_label="mensagem enviada (atrasada)",
            ylim=ylim,
        )
        omega, values = _psd(signal)
        _panel(
            axes[row, 1],
            omega,
            values,
            title=f"{panel_labels[row + 4]} espectro de {label}",
            ylabel="densidade espectral",
            xlabel=r"$\omega/\pi$",
        )
    handles, labels = axes[3, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=6.0,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def comm() -> None:
    """Write the failure and the filtered-link eight-panel figures."""
    comm_cfg = CFG.comm
    message = text_message(comm_cfg.message_text, comm_cfg.N, bit_period=comm_cfg.message_period)

    rng = np.random.default_rng(CFG.seed)
    sent = transmit(message, mu=comm_cfg.mu, alpha=1.4, beta=0.3, x0=0.0, y0=0.0)
    received, _ = fir_channel(sent, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    recovered = receive(
        received,
        mu=comm_cfg.mu,
        alpha=1.4,
        beta=0.3,
        y0=rng.random(),
        z0=rng.random(),
    )
    _comm_figure(
        message, sent, received, recovered, OUT / "fig_comm_falha.pdf", clip_recovered=True
    )

    coeffs = firwin(
        numtaps=FILTER_TAPS,
        cutoff=FILTER_CUTOFF,
        window="hamming",
        pass_zero=True,
        fs=2.0,
    )
    rng = np.random.default_rng(CFG.seed)
    x0 = 0.5 * rng.random(len(coeffs))
    y0 = 0.5 * rng.random(len(coeffs))
    sent_f, _ = transmit_order_n(message, coeffs, mu=comm_cfg.mu, alpha=1.4, beta=0.3, x0=x0)
    received_f, _ = fir_channel(sent_f, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    recovered_f, _ = receive_order_n(received_f, coeffs, mu=comm_cfg.mu, alpha=1.4, beta=0.3, y0=y0)
    _comm_figure(
        message, sent_f, received_f, recovered_f, OUT / "fig_comm_filtro.pdf", clip_recovered=False
    )


GENERATORS = {
    "classification": classification,
    "density": density,
    "kaiser": kaiser,
    "hamming": hamming,
    "comm": comm,
}


def main() -> int:
    """Run the requested generators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["all", *GENERATORS],
        default="all",
        help="Generate only one group of figures.",
    )
    args = parser.parse_args()
    setup()
    names = list(GENERATORS) if args.only == "all" else [args.only]
    for name in names:
        GENERATORS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
r"""Generate the PFC data figures in Portuguese, with comma decimals.

Every figure is rebuilt from the versioned sweeps in ``data/sweeps`` and from
the communication pipeline of ``chaotic_pfc``; nothing is edited by hand.
The style comes from :mod:`tools.paper_style`, which fixes the advisor
standard shared with the JCIS: Times face, text at the caption size, figures
produced already at their final printed width, red/blue/gray classification
and comma decimals.

Output names match the ``\includegraphics`` calls of ``artigo.tex``.

Usage::

    python articles/pfc/tools/gerar_figuras_pfc.py
    python articles/pfc/tools/gerar_figuras_pfc.py --only classification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.signal import firwin, freqz

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chaotic_pfc._rounding import round_decimal  # noqa: E402
from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, load_sweep  # noqa: E402
from chaotic_pfc.analysis.sweep_plotting import (  # noqa: E402
    plot_chaotic_density_norm,
    plot_classification_interleaved,
)
from chaotic_pfc.comms.channel import fir_channel  # noqa: E402
from chaotic_pfc.comms.receiver import receive, receive_order_n  # noqa: E402
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n  # noqa: E402
from chaotic_pfc.config import DEFAULT_CONFIG as CFG  # noqa: E402
from chaotic_pfc.dynamics.maps import henon_generalised  # noqa: E402
from chaotic_pfc.dynamics.signals import text_message  # noqa: E402
from chaotic_pfc.dynamics.spectral import psd_normalised  # noqa: E402
from tools.paper_style import (  # noqa: E402
    LINK_COLORS,
    SERIES,
    Style,
    apply,
    class_handles,
    install_sweep_palette,
    save,
)

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

WINDOW_LABELS = {
    "rectangular": "Retangular",
    "hamming": "Hamming",
    "blackmanharris": "Blackman-Harris",
    "kaiser": r"Kaiser, $\beta_K=17$",
}

COMM_WINDOW = (2000, 2400)
FILTER_CUTOFF = 0.52
FILTER_TAPS = 9
FIR_CUTOFF = 0.4


def setup() -> Style:
    """Apply the shared figure style and point the sweep palette at it."""
    st = apply("pfc")
    install_sweep_palette()
    return st


def series(index: int) -> dict[str, Any]:
    """Colour and line style of series ``index`` in the advisor order."""
    color, ls = SERIES[index % len(SERIES)]
    return {"color": color, "linestyle": ls}


def legend_above(fig: plt.Figure, handles: list[Any], *, ncol: int, pad: float = 1.0) -> None:
    """Place the legend above the axes, outside the plotting area."""
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, pad),
        ncol=ncol,
        handlelength=1.5,
        handleheight=0.8,
        columnspacing=1.4,
    )


# ── Fig. 1: Hénon signals ──────────────────────────────────────────────────


def henon(st: Style) -> None:
    """Write the two-panel Hénon trace of Fig. 1."""
    n_steps = 99
    n = np.arange(n_steps + 1)
    fig, axes = st.subplots(2, 1, height_in=2.4, sharex=True)
    for ax, (label, alpha) in zip(axes, (("a", 1.4), ("b", 0.9)), strict=True):
        x1_a, _ = henon_generalised(n_steps, x0=0.0, y0=0.0, alpha=alpha, beta=0.3)
        x1_b, _ = henon_generalised(n_steps, x0=0.0, y0=0.001, alpha=alpha, beta=0.3)
        ax.plot(n, x1_a, lw=1.0, label=r"$x_1(0)=x_2(0)=0$", **series(0))
        ax.plot(n, x1_b, lw=1.0, label=r"$x_1(0)=0,\ x_2(0)=0{,}001$", **series(1))
        ax.set_title(
            f"({label})  "
            + rf"$\alpha={st.number(alpha).replace(',', '{,}')}$, "
            + rf"$\beta={st.number(0.3).replace(',', '{,}')}$",
            loc="left",
            pad=3,
        )
        ax.set_ylabel(r"$x_1(n)$")
        ax.set_yticks([-1.0, 0.0, 1.0] if label == "a" else [0.0, 1.0])
        ax.yaxis.set_major_formatter(st.formatter)
        ax.grid(alpha=0.3)
    axes[1].set_xticks([0, 20, 40, 60, 80, 100])
    axes[1].xaxis.set_major_formatter(st.formatter)
    axes[1].set_xlabel("$n$")
    handles, _ = axes[0].get_legend_handles_labels()
    legend_above(fig, handles, ncol=2, pad=0.995)
    fig.subplots_adjust(left=0.088, right=0.995, top=0.80, bottom=0.17, hspace=0.7)
    save(fig, OUT / "fig_henon_signals_h.pdf")
    print(f"wrote {(OUT / 'fig_henon_signals_h.pdf').relative_to(ROOT)}")


# ── Fig. 2: FIR responses ──────────────────────────────────────────────────


def _response(numtaps: int, window: str | tuple[str, float]) -> tuple[np.ndarray, np.ndarray]:
    coeffs = firwin(numtaps=numtaps, cutoff=FIR_CUTOFF, window=window, pass_zero=True, fs=2.0)
    w, h = freqz(coeffs, worN=2048, fs=2.0)
    return w, 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))


def fir(st: Style) -> None:
    """Write the two-panel frequency response of Fig. 2."""
    fig, axes = st.subplots(1, 2, height_in=2.55, sharey=True)
    panels = (
        (
            "(a) janelas diferentes, $N_z=20$",
            [
                ("rectangular", 21),
                ("hamming", 21),
                ("blackmanharris", 21),
                (("kaiser", 17.0), 21),
            ],
            ("Retangular", "Hamming", "Blackman-Harris", r"Kaiser, $\beta_K=17$"),
        ),
        (
            "(b) janela de Hamming, ordens diferentes",
            [("hamming", 5), ("hamming", 9), ("hamming", 21), ("hamming", 41)],
            (r"$N_z=4$", r"$N_z=8$", r"$N_z=20$", r"$N_z=40$"),
        ),
    )
    styles = (
        ("#1F77B4", "-", "o"),
        ("#D62728", "--", "s"),
        ("#2CA02C", "-.", "^"),
        ("#000000", ":", "D"),
    )
    for ax, (title, curves, labels) in zip(axes, panels, strict=True):
        for index, ((window, taps), name) in enumerate(zip(curves, labels, strict=True)):
            w, mag = _response(taps, window)
            color, ls, marker = styles[index]
            ax.plot(
                w, mag, color=color, ls=ls, lw=1.2, marker=marker, markevery=180, ms=3.0, label=name
            )
        ax.axvline(FIR_CUTOFF, color="0.35", lw=0.6, ls=":")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-160.0, 5.0)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel(r"$\omega/\pi$")
        ax.xaxis.set_major_formatter(st.formatter)
        ax.yaxis.set_major_formatter(st.formatter)
        ax.grid(alpha=0.3)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=2,
            handlelength=1.4,
            columnspacing=1.2,
        )
    axes[0].set_ylabel(r"$|H(e^{j\omega})|$ (dB)")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.36, wspace=0.12)
    save(fig, OUT / "fig_fir_respostas.pdf")
    print(f"wrote {(OUT / 'fig_fir_respostas.pdf').relative_to(ROOT)}")


# ── Fig. 9: classification maps ────────────────────────────────────────────


def classification(st: Style) -> None:
    """Write the four classification maps and their shared key."""
    for name, stem in CLASSIFICATION:
        sweep = load_sweep(SWEEPS / name / "variables_lyapunov.npz")
        fig = plot_classification_interleaved(
            sweep,
            lang="pt",
            figsize=st.figsize(0.49, aspect=0.67),
            label_fontsize=st.caption_pt,
            tick_fontsize=st.caption_pt,
            legend=False,
            ytick_step=0.2,
            xtick_step=10,
        )
        ax = fig.axes[0]
        ax.yaxis.set_major_formatter(FuncFormatter(st.number))
        ax.grid(False)
        save(fig, OUT / f"{stem}.pdf")
        print(f"wrote {(OUT / f'{stem}.pdf').relative_to(ROOT)}")

    fig, ax = st.subplots(1, 1, fraction=1.0, height_in=0.32)
    ax.axis("off")
    fig.legend(
        handles=class_handles("pt"),
        loc="center",
        ncol=3,
        handlelength=1.6,
        handleheight=0.8,
        columnspacing=2.4,
    )
    save(fig, OUT / "legenda_classificacao.pdf")
    print(f"wrote {(OUT / 'legenda_classificacao.pdf').relative_to(ROOT)}")


# ── Fig. 10: aggregate chaotic density ─────────────────────────────────────


def density(st: Style) -> None:
    """Write the normalised chaotic-density map."""
    fig = plot_chaotic_density_norm(
        SWEEPS,
        lang="pt",
        figsize=st.figsize(0.70, aspect=0.57),
        label_fontsize=st.caption_pt,
        tick_fontsize=st.caption_pt,
        cbar_fontsize=st.caption_pt,
        show_title=False,
        ytick_step=0.2,
        xtick_step=10,
        strict=True,
    )
    ax = fig.axes[0]
    ax.yaxis.set_major_formatter(FuncFormatter(st.number))
    if len(fig.axes) > 1:
        cbar = fig.axes[1]
        cbar.yaxis.set_major_formatter(FuncFormatter(st.number))
        cbar.set_ylabel("porcentagem das configurações (%)", fontsize=st.caption_pt)
    fig.tight_layout(pad=0.2)
    save(fig, OUT / "fig_chaotic_density_norm.pdf")
    print(f"wrote {(OUT / 'fig_chaotic_density_norm.pdf').relative_to(ROOT)}")


# ── Fig. 11: chaotic coverage against the Kaiser beta ──────────────────────


def kaiser(st: Style) -> None:
    """Write the chaotic coverage as a function of the Kaiser beta."""
    fig, ax = st.subplots(1, 1, fraction=0.8, aspect=0.50)
    styles = (
        ("#1B4F9C", "-", "o"),
        ("#B2182B", "--", "s"),
        ("#1B7837", "-.", "^"),
        ("#6A3D9A", ":", "D"),
    )
    for index, ft in enumerate(("lowpass", "highpass", "bandpass", "bandstop")):
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
        color, ls, marker = styles[index]
        ax.plot(
            betas,
            coverage,
            color=color,
            ls=ls,
            marker=marker,
            ms=2.8,
            lw=1.2,
            label=TYPE_LABELS[ft],
        )
        best = int(np.argmax(coverage))
        print(
            f"[kaiser] {ft}: max {round_decimal(float(coverage[best]), 1)}% at beta_K={betas[best]:g}; "
            f"beta=0.5 -> {round_decimal(float(coverage[0]), 1)}%, "
            f"beta=20 -> {round_decimal(float(coverage[-1]), 1)}%"
        )
    ax.set_xlabel(r"$\beta_K$")
    ax.set_ylabel("cobertura caótica (%)")
    ax.set_xlim(0.0, 20.5)
    ax.xaxis.set_major_formatter(FuncFormatter(st.number))
    ax.yaxis.set_major_formatter(FuncFormatter(st.number))
    handles, _ = ax.get_legend_handles_labels()
    legend_above(fig, handles, ncol=4, pad=0.995)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.79, bottom=0.19)
    save(fig, OUT / "fig_kaiser_beta.pdf")
    print(f"wrote {(OUT / 'fig_kaiser_beta.pdf').relative_to(ROOT)}")


# ── Fig. 12: Hamming lowpass map with the chosen design ────────────────────


def hamming(st: Style) -> None:
    """Write the Hamming lowpass map with the chosen project circled."""
    sweep = load_sweep(SWEEPS / "Hamming (lowpass)" / "variables_lyapunov.npz")
    fig = plot_classification_interleaved(
        sweep,
        lang="pt",
        figsize=st.figsize(0.70, aspect=0.63),
        label_fontsize=st.caption_pt,
        tick_fontsize=st.caption_pt,
        legend=False,
        ytick_step=0.2,
        xtick_step=10,
    )
    ax = fig.axes[0]
    ax.yaxis.set_major_formatter(FuncFormatter(st.number))
    ax.grid(False)
    target = 8
    row = int(np.flatnonzero(np.asarray(sweep.num_taps_grid) == target + 1)[0])
    ax.plot(
        [row * 4 + 1],
        [FILTER_CUTOFF],
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.2,
        linestyle="none",
        zorder=20,
    )
    legend_above(fig, class_handles("pt"), ncol=3, pad=0.99)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.80, bottom=0.17)
    save(fig, OUT / "hamming_lowpass_classification.pdf")
    print(f"wrote {(OUT / 'hamming_lowpass_classification.pdf').relative_to(ROOT)}")

    nearest = int(np.argmin(np.abs(np.asarray(sweep.cutoff_grid) - FILTER_CUTOFF)))
    print(
        f"[hamming] Nz={target} cutoff={round_decimal(float(sweep.cutoff_grid[nearest]), 10)} "
        f"lambda1={round_decimal(float(sweep.h[row, nearest]), 10)}"
    )


# ── Figs. 8 and 13: eight-panel communication figures ──────────────────────


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
    st: Style,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    color: str,
    reference: np.ndarray | None = None,
    reference_label: str = "",
    solid_label: str = "mensagem recuperada",
    ylim: tuple[float, float] | None = None,
) -> None:
    if reference is not None:
        ax.plot(x, reference, color="black", ls="--", lw=1.0, label=reference_label)
    ax.plot(x, y, color=color, lw=1.2, label=solid_label if reference is not None else None)
    ax.set_title(title, loc="left", pad=2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FuncFormatter(st.number))
    ax.yaxis.set_major_formatter(FuncFormatter(st.number))
    ax.grid(alpha=0.3)


def _comm_figure(
    st: Style,
    message: np.ndarray,
    sent: np.ndarray,
    received: np.ndarray,
    recovered: np.ndarray,
    path: Path,
    *,
    fraction: float,
    clip_recovered: bool,
) -> None:
    delay = (CFG.channel.num_taps - 1) // 2
    start, stop = COMM_WINDOW
    n = np.arange(start, stop)
    # Same alignment as the synchronisation criterion: the channel is a
    # linear-phase FIR, so m_hat[n] reproduces m[n - delay] and the received
    # samples are compared at n + delay.  Plot every received/recovered panel
    # on that shifted grid so (a) and (d) show the same message bits.
    received_aligned = received[n + delay]
    recovered_aligned = recovered[n + delay]
    message_window = message[n]
    sent_window = sent[n]

    aspect = 0.48
    fig, axes = st.subplots(4, 2, fraction=fraction, aspect=aspect, sharex="col")
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")
    rows = (
        (
            "mensagem",
            "mensagem",
            message_window,
            message,
            LINK_COLORS["message"],
            LINK_COLORS["message_spec"],
            False,
        ),
        (
            "transmitido",
            "transmitido",
            sent_window,
            sent,
            LINK_COLORS["carrier"],
            LINK_COLORS["carrier_spec"],
            False,
        ),
        (
            "recebido",
            "recebido",
            received_aligned,
            received,
            LINK_COLORS["carrier"],
            LINK_COLORS["carrier_spec"],
            False,
        ),
        (
            "recuperada",
            "recuperado",
            recovered_aligned,
            recovered,
            LINK_COLORS["message"],
            LINK_COLORS["message_spec"],
            clip_recovered,
        ),
    )
    for row, (label, _short, signal, psd_signal, color, spec_color, clipped) in enumerate(rows):
        ylim = (-3.0, 3.0) if clipped else None
        suffix = ""
        reference = message_window if row == 3 else None
        _panel(
            axes[row, 0],
            st,
            n,
            signal,
            title=f"{panel_labels[row]} {label}{suffix}",
            color=color,
            reference=reference,
            reference_label="mensagem enviada",
            ylim=ylim,
        )
        omega, values = _psd(psd_signal)
        _panel(
            axes[row, 1],
            st,
            omega,
            values,
            title=f"{panel_labels[row + 4]} espectro: {label}",
            color=spec_color,
        )
    for row in (1, 2):
        axes[row, 0].set_yticks([-1, 1])
    for ax in axes[-1, :]:
        ax.set_xlabel("amostra")
    axes[-1, 1].set_xlabel(r"$\omega/\pi$")
    handles, _ = axes[3, 0].get_legend_handles_labels()
    legend_above(fig, handles, ncol=2, pad=0.995)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.80, bottom=0.14, hspace=0.82, wspace=0.28)
    # One y label per column, centred on the four rows and clear of the tick
    # labels of both columns.
    y_mid = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)
    left_x = axes[0, 0].get_position().x0 - 0.045
    right_x = axes[0, 1].get_position().x0 - 0.060
    fig.text(
        left_x,
        y_mid,
        "amplitude",
        rotation=90,
        ha="center",
        va="center",
        fontsize=st.caption_pt,
    )
    fig.text(
        right_x,
        y_mid,
        "densidade espectral",
        rotation=90,
        ha="center",
        va="center",
        fontsize=st.caption_pt,
    )
    save(fig, path)
    print(f"wrote {path.relative_to(ROOT)}")


def comm(st: Style) -> None:
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
        st,
        message,
        sent,
        received,
        recovered,
        OUT / "fig_comm_falha.pdf",
        fraction=1.0,
        clip_recovered=True,
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
        st,
        message,
        sent_f,
        received_f,
        recovered_f,
        OUT / "fig_comm_filtro.pdf",
        fraction=1.0,
        clip_recovered=False,
    )


GENERATORS = {
    "henon": henon,
    "fir": fir,
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
    st = setup()
    names = list(GENERATORS) if args.only == "all" else [args.only]
    for name in names:
        GENERATORS[name](st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

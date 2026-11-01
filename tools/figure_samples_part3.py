#!/usr/bin/env python3
"""Amostras das figuras da revisão de 01/11 (Parte 3). Nada é aplicado ao artigo.

* Fig. 2: quatro séries por painel, distinguíveis por cor, traço e marcador.
* Figs. 8 e 13: títulos de painel curtos, sem quebra; a nota de escala sai do
  painel e vai para a legenda (a legenda é do .tex e não é alterada aqui).
* Fig. 10: a mesma função do JCIS, sem sobrescrever o mapa de cores nem o
  marcador, para sair idêntica à Fig. 4 do JCIS.
* Fig. 1: vírgula decimal sem espaço no mathtext.

Saída em ``revisao/amostras-estilo/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.signal import firwin, freqz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chaotic_pfc.analysis.sweep_plotting import plot_chaotic_density_norm  # noqa: E402
from chaotic_pfc.comms.channel import fir_channel  # noqa: E402
from chaotic_pfc.comms.receiver import receive, receive_order_n  # noqa: E402
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n  # noqa: E402
from chaotic_pfc.config import DEFAULT_CONFIG as CFG  # noqa: E402
from chaotic_pfc.dynamics.maps import henon_generalised  # noqa: E402
from chaotic_pfc.dynamics.signals import text_message  # noqa: E402
from chaotic_pfc.dynamics.spectral import psd_normalised  # noqa: E402
from tools.paper_style import LINK_COLORS, apply, save  # noqa: E402

OUT = ROOT / "revisao/amostras-estilo"
SWEEPS = ROOT / "data/sweeps"
COMM_WINDOW = (2000, 2400)
FILTER_CUTOFF = 0.52
FILTER_TAPS = 9
FIR_CUTOFF = 0.4

# Quatro séries com cor, traço e marcador distintos (Fig. 2).
SERIES_STYLE = (
    ("#1F77B4", "-", "o"),
    ("#D62728", "--", "s"),
    ("#2CA02C", "-.", "^"),
    ("#000000", ":", "D"),
)
WINDOW_NAMES = ("Retangular", "Hamming", "Blackman-Harris", r"Kaiser, $\beta_K=17$")


def num(st, value: float) -> str:
    """Número com vírgula, agrupada para não abrir espaço no mathtext."""
    return st.number(value).replace(",", "{,}")


def fig1(st) -> None:
    """Fig. 1 com vírgula decimal sem espaço no mathtext."""
    n_steps = 99
    n = np.arange(n_steps + 1)
    fig, axes = st.subplots(2, 1, height_in=2.4, sharex=True)
    for ax, (label, alpha) in zip(axes, (("a", 1.4), ("b", 0.9)), strict=True):
        x1_a, _ = henon_generalised(n_steps, x0=0.0, y0=0.0, alpha=alpha, beta=0.3)
        x1_b, _ = henon_generalised(n_steps, x0=0.0, y0=0.001, alpha=alpha, beta=0.3)
        ax.plot(n, x1_a, color="#1F77B4", lw=1.0, ls="-", label=r"$x_1(0)=x_2(0)=0$")
        ax.plot(n, x1_b, color="#D62728", lw=1.0, ls="--", label=r"$x_1(0)=0,\ x_2(0)=0{,}001$")
        ax.set_title(
            rf"({label})  $\alpha={num(st, alpha)}$, $\beta={num(st, 0.3)}$", loc="left", pad=3
        )
        ax.set_ylabel(r"$x_1(n)$")
        ax.set_yticks([-1.0, 0.0, 1.0] if label == "a" else [0.0, 1.0])
        ax.yaxis.set_major_formatter(st.formatter)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("$n$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2)
    fig.subplots_adjust(left=0.088, right=0.995, top=0.80, bottom=0.17, hspace=0.7)
    save(fig, OUT / "pfc-fig1-henon-virgula.pdf")
    print("wrote pfc-fig1-henon-virgula.pdf")


def fig2(st) -> None:
    """Fig. 2 com quatro séries distinguíveis por cor, traço e marcador."""
    fig, axes = st.subplots(1, 2, height_in=3.05, sharey=True)
    panels = (
        (
            "(a) janelas, $N_z=20$",
            [("rectangular", 21), ("hamming", 21), ("blackmanharris", 21), (("kaiser", 17.0), 21)],
            ("Retangular", "Hamming", "Blackman-Harris", r"Kaiser, $\beta_K=17$"),
        ),
        (
            "(b) Hamming, ordens diferentes",
            [("hamming", 5), ("hamming", 9), ("hamming", 21), ("hamming", 41)],
            (r"$N_z=4$", r"$N_z=8$", r"$N_z=20$", r"$N_z=40$"),
        ),
    )
    for ax, (title, curves, names) in zip(axes, panels, strict=True):
        for index, ((window, taps), (color, ls, marker)) in enumerate(
            zip(curves, SERIES_STYLE, strict=True)
        ):
            coeffs = firwin(numtaps=taps, cutoff=FIR_CUTOFF, window=window, pass_zero=True, fs=2.0)
            w, h = freqz(coeffs, worN=2048, fs=2.0)
            label = names[index]
            ax.plot(
                w,
                20.0 * np.log10(np.maximum(np.abs(h), 1e-12)),
                color=color,
                ls=ls,
                lw=1.2,
                marker=marker,
                markevery=180,
                ms=3.0,
                label=label,
            )
        ax.axvline(FIR_CUTOFF, color="0.35", lw=0.6, ls=":")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-160.0, 5.0)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel(r"$\omega/\pi$")
        ax.xaxis.set_major_formatter(FuncFormatter(st.number))
        ax.yaxis.set_major_formatter(FuncFormatter(st.number))
        ax.grid(alpha=0.3)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=2,
            handlelength=2.4,
            columnspacing=1.0,
        )
    axes[0].set_ylabel(r"$|H(e^{j\omega})|$ (dB)")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.36, wspace=0.12)
    save(fig, OUT / "pfc-fig2-series.pdf")
    print("wrote pfc-fig2-series.pdf")


def psd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = CFG.spectral
    omega, values = psd_normalised(
        signal, spec.nfft, spec.window_length, window=spec.window, kaiser_beta=spec.kaiser_beta
    )
    return np.asarray(omega), np.asarray(values)


def comm_figure(st, message, sent, received, recovered, path: Path) -> None:
    """Figs. 8 e 13 com títulos curtos, sem a nota de escala no painel."""
    delay = (CFG.channel.num_taps - 1) // 2
    start, stop = COMM_WINDOW
    n = np.arange(start, stop)
    delayed = message[n - delay]
    fig, axes = st.subplots(4, 2, fraction=1.0, aspect=0.68, sharex="col")
    labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")
    rows = (
        ("mensagem", "mensagem", message, LINK_COLORS["message"], LINK_COLORS["message_spec"]),
        ("transmitido", "transmitido", sent, LINK_COLORS["carrier"], LINK_COLORS["carrier_spec"]),
        ("recebido", "recebido", received, LINK_COLORS["carrier"], LINK_COLORS["carrier_spec"]),
        (
            "recuperada",
            "recuperado",
            recovered,
            LINK_COLORS["message"],
            LINK_COLORS["message_spec"],
        ),
    )
    for row, (label, _short, signal, color, spec_color) in enumerate(rows):
        ax = axes[row, 0]
        ax.plot(
            n, signal[n], color=color, lw=1.2, label="mensagem recuperada" if row == 3 else None
        )
        if row == 3:
            ax.plot(n, delayed, color="black", ls="--", lw=1.0, label="mensagem enviada (atrasada)")
        ax.set_title(f"{labels[row]} {label}", loc="left", pad=2)
        if row == 3:
            ax.set_ylim(-3.0, 3.0)
        ax.xaxis.set_major_formatter(FuncFormatter(st.number))
        ax.yaxis.set_major_formatter(FuncFormatter(st.number))
        ax.grid(alpha=0.3)
        ax = axes[row, 1]
        omega, values = psd(signal)
        ax.plot(omega, values, color=spec_color, lw=1.2)
        ax.set_title(f"{labels[row + 4]} espectro: {label}", loc="left", pad=2)
        ax.xaxis.set_major_formatter(FuncFormatter(st.number))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("amplitude")
    axes[0, 1].set_ylabel("densidade espectral")
    axes[-1, 0].set_xlabel("amostra")
    axes[-1, 1].set_xlabel(r"$\omega/\pi$")
    handles, labels_ = axes[3, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.09, hspace=0.62, wspace=0.22)
    save(fig, path)
    print(f"wrote {path.name}")


def comm_pair(st) -> None:
    cfg = CFG.comm
    message = text_message(cfg.message_text, cfg.N, bit_period=cfg.message_period)
    rng = np.random.default_rng(CFG.seed)
    sent = transmit(message, mu=cfg.mu, alpha=1.4, beta=0.3, x0=0.0, y0=0.0)
    received, _ = fir_channel(sent, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    recovered = receive(received, mu=cfg.mu, alpha=1.4, beta=0.3, y0=rng.random(), z0=rng.random())
    comm_figure(st, message, sent, received, recovered, OUT / "pfc-fig8-paineis.pdf")

    coeffs = firwin(
        numtaps=FILTER_TAPS, cutoff=FILTER_CUTOFF, window="hamming", pass_zero=True, fs=2.0
    )
    rng = np.random.default_rng(CFG.seed)
    x0 = 0.5 * rng.random(len(coeffs))
    y0 = 0.5 * rng.random(len(coeffs))
    sent_f, _ = transmit_order_n(message, coeffs, mu=cfg.mu, alpha=1.4, beta=0.3, x0=x0)
    received_f, _ = fir_channel(sent_f, cutoff=CFG.channel.cutoff, num_taps=CFG.channel.num_taps)
    recovered_f, _ = receive_order_n(received_f, coeffs, mu=cfg.mu, alpha=1.4, beta=0.3, y0=y0)
    comm_figure(st, message, sent_f, received_f, recovered_f, OUT / "pfc-fig13-paineis.pdf")


def fig10(st) -> None:
    """Fig. 10 com a função e os padrões do JCIS (viridis, marcador no máximo)."""
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
        fig.axes[1].yaxis.set_major_formatter(FuncFormatter(st.number))
        fig.axes[1].set_ylabel("porcentagem das configurações (%)", fontsize=st.caption_pt)
    fig.tight_layout(pad=0.2)
    save(fig, OUT / "pfc-fig10-jcis.pdf")
    print("wrote pfc-fig10-jcis.pdf")


def main() -> int:
    """Gera as amostras da Parte 3."""
    st = apply("pfc")
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(st)
    fig2(st)
    comm_pair(st)
    fig10(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

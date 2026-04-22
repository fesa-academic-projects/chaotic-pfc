"""
plotting.py
===========
Publication-quality SVG figures with LaTeX-style labels.

All text uses matplotlib's mathtext engine (no external LaTeX needed).
Figures are saved as .svg by default for vector-quality output.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.signal import freqz

# ── Global RC params for LaTeX-like rendering ───────────────────────────────


def setup_rc():
    """Configure matplotlib for publication-quality LaTeX-style SVG output.

    Uses STIX fonts (the standard for scientific publishing, very close
    to Computer Modern) and converts all text to vector paths so that
    SVGs render identically on any system without requiring font installation.
    """
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",  # STIX ≈ Computer Modern
            "font.family": "STIXGeneral",  # matching text font
            "svg.fonttype": "path",  # text → vector paths (portable)
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "lines.linewidth": 1.5,
        }
    )


setup_rc()


# ── Colour palette ──────────────────────────────────────────────────────────

C = {
    "msg_t": "#0073BD",  # blue (time-domain message)
    "msg_f": "#804000",  # brown (freq-domain message)
    "sig_t": "#E00000",  # red (time-domain signal)
    "sig_f": "#660066",  # purple (freq-domain signal)
    "traj": "#000000",  # black (attractor)
    "traj2": "#E00000",  # red (second trajectory)
}


def _style(ax: Axes, ts: int = 12) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=ts, width=1.2, direction="in")
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)


def _save(fig: Figure, path: str | None) -> None:
    if path:
        fig.savefig(path)


# ── 1. Attractor ────────────────────────────────────────────────────────────


def plot_attractor(
    X: NDArray,
    Y: NDArray,
    title: str = "",
    xlabel: str = r"$x_1[n]$",
    ylabel: str = r"$x_2[n]$",
    save_path: str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(X, Y, ",", color=C["traj"], alpha=0.8, markersize=0.3)
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    _style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 2. Sensitivity (SDIC) ──────────────────────────────────────────────────


def plot_sensitivity(
    n: NDArray,
    X1: NDArray,
    X2: NDArray,
    save_path: str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(n, X1, label=r"$x^{(1)}[n],\; x_0=0$", color="steelblue", linewidth=1.2)
    ax.plot(n, X2, label=r"$x^{(2)}[n],\; x_0=10^{-4}$", color="tomato", linewidth=1.2, alpha=0.85)
    ax.set_xlabel(r"$n$", fontsize=12)
    ax.set_ylabel(r"$x[n]$", fontsize=12)
    ax.set_title(
        r"Sensibilidade às Condições Iniciais — Mapa de Hénon",
        fontsize=13,
    )
    ax.legend(fontsize=11, framealpha=0.9)
    _style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 3. Communication 4×2 grid (time + PSD) ─────────────────────────────────


def plot_comm_grid(
    n: NDArray,
    m: NDArray,
    s: NDArray,
    r: NDArray,
    m_hat: NDArray,
    omega: NDArray,
    psd_m: NDArray,
    psd_s: NDArray,
    psd_r: NDArray,
    psd_mhat: NDArray,
    time_window: slice = slice(0, 300),
    suptitle: str = "",
    y_lim_msg: tuple = (-1.5, 1.5),
    y_lim_sig: tuple = (-2.5, 2.5),
    y_lim_mhat: tuple | None = None,
    h_channel: NDArray | None = None,
    save_path: str | None = None,
) -> Figure:
    """
    4×2 grid: left = time domain, right = PSD.
    Rows: m[n], s[n], r[n], m̂[n].
    If h_channel is provided, its frequency response is overlaid on PSD_s.
    """
    if y_lim_mhat is None:
        y_lim_mhat = y_lim_msg

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.995)

    nn = n[time_window]

    # Row configs: (signal, ylabel_t, ylabel_f, color_t, color_f, dots?, ylim_t)
    rows = [
        (
            m,
            r"$(a)\; m[n]$",
            r"$(e)\; \mathcal{M}(\omega)$",
            C["msg_t"],
            C["msg_f"],
            True,
            y_lim_msg,
        ),
        (s, r"$(b)\; s[n]$", r"$(f)\; S(\omega)$", C["sig_t"], C["sig_f"], False, y_lim_sig),
        (r, r"$(c)\; r[n]$", r"$(g)\; R(\omega)$", C["sig_t"], C["sig_f"], False, y_lim_sig),
        (
            m_hat,
            r"$(d)\; \hat{m}[n]$",
            r"$(h)\; \hat{\mathcal{M}}(\omega)$",
            C["msg_t"],
            C["msg_f"],
            True,
            y_lim_mhat,
        ),
    ]
    psds = [psd_m, psd_s, psd_r, psd_mhat]

    for i, ((sig, lbl_t, lbl_f, ct, cf, dots, ylim), psd) in enumerate(
        zip(rows, psds, strict=False)
    ):
        # ---- Time domain (left column) ----
        ax_t = axes[i, 0]
        if dots:
            ax_t.plot(nn, sig[time_window], ".", markersize=5, color=ct)
        else:
            ax_t.plot(nn, sig[time_window], color=ct, linewidth=1.2)
        ax_t.set_ylabel(lbl_t, fontsize=12)
        ax_t.set_xlim([nn[0], nn[-1]])
        ax_t.set_ylim(ylim)
        if i < 3:
            ax_t.set_xticklabels([])
        else:
            ax_t.set_xlabel(r"$n$", fontsize=12)
        _style(ax_t)

        # ---- PSD (right column) ----
        ax_f = axes[i, 1]
        ax_f.plot(omega, psd, color=cf, linewidth=1.2)
        ax_f.set_ylabel(lbl_f, fontsize=12)
        ax_f.set_ylim([-0.05, 1.08])
        ax_f.set_xlim([0, 1.0])
        if i < 3:
            ax_f.set_xticklabels([])
        else:
            ax_f.set_xlabel(r"$\omega / \pi$", fontsize=12)
        _style(ax_f)

        # Overlay channel response on PSD_s panel
        if i == 1 and h_channel is not None:
            w_h, H = freqz(h_channel, worN=1024, whole=False)
            ax_f.plot(
                w_h / np.pi,
                np.abs(H),
                "k--",
                linewidth=1.5,
                label=r"$|H_{ch}(e^{j\omega})|$",
                alpha=0.7,
            )
            ax_f.legend(fontsize=9, loc="upper right")

    # Column titles
    axes[0, 0].set_title(r"Domínio do Tempo", fontsize=12)
    axes[0, 1].set_title(r"PSD Normalizada (Welch)", fontsize=12)

    fig.tight_layout()
    _save(fig, save_path)
    return fig

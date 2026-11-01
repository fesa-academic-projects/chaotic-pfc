#!/usr/bin/env python3
"""Power spectral density without the loop filter and with the link filter.

Uses the same communication pipeline and seeds as the channel verification
script (``articles/pfc/revisao/scripts/verificacao_canal.py``): the unfiltered
carrier is ``x1`` of the transmitted signal, and the filtered carrier is ``x3``
of the same link. Both spectra share one normalisation so the levels compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.signal import firwin, welch

from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.config import DEFAULT_CONFIG as CFG
from chaotic_pfc.dynamics.signals import text_message

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.paper_style import apply, save  # noqa: E402

OUT = ROOT / "articles/pfc/figures/fig_psd_laco.pdf"
FILTER_CUTOFF = 0.52
FILTER_TAPS = 9
CHANNEL_EDGE = 0.99
N_PSD = 65536


def spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD in cycles per sample, returned as a function of ``omega/pi``."""
    freqs, power = welch(signal, fs=1.0, nperseg=1024, noverlap=512, detrend=False)
    return 2.0 * freqs, power


def main() -> int:
    """Write the two-spectrum figure."""
    cfg = CFG.comm
    message = text_message(cfg.message_text, cfg.N, bit_period=cfg.message_period)

    sent = transmit(message, mu=cfg.mu, alpha=1.4, beta=0.3, x0=0.0, y0=0.0)
    x1 = sent - cfg.mu * message

    coeffs = firwin(
        numtaps=FILTER_TAPS, cutoff=FILTER_CUTOFF, window="hamming", pass_zero=True, fs=2.0
    )
    rng = np.random.default_rng(CFG.seed)
    x0 = 0.5 * rng.random(len(coeffs))
    _, state = transmit_order_n(message, coeffs, mu=cfg.mu, alpha=1.4, beta=0.3, x0=x0)
    x3 = state[2]

    w1, p1 = spectrum(x1[:N_PSD])
    w3, p3 = spectrum(x3[:N_PSD])
    ref = p1.max()

    st = apply("pfc")
    fig, ax = st.subplots(1, 1, fraction=0.62, aspect=0.45)
    ax.axvspan(CHANNEL_EDGE, 1.0, color="0.92", zorder=0, lw=0)
    ax.semilogy(w1, p1 / ref, color="black", lw=1.0, ls="-", label="sem filtro")
    ax.semilogy(
        w3,
        p3 / ref,
        color="#D62728",
        lw=1.0,
        ls="--",
        label="com o filtro do enlace",
    )
    ax.axvline(CHANNEL_EDGE, color="0.35", lw=0.8, ls=":")
    ax.plot([], [], color="0.35", lw=0.8, ls=":", label="limite do canal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-9, 5.0)
    ax.set_xlabel(r"$\omega/\pi$")
    ax.set_ylabel("densidade espectral")
    ax.xaxis.set_major_formatter(st.formatter)
    ax.grid(True, which="major")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.155, right=0.985, top=0.83, bottom=0.235)
    save(fig, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

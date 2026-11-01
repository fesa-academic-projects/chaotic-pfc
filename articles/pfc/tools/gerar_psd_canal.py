#!/usr/bin/env python3
"""Power spectral density without the loop filter and with the link filter.

Uses the same communication pipeline and seeds as the channel verification
script (``articles/pfc/revisao/scripts/verificacao_canal.py``): the unfiltered
carrier is ``x1`` of the transmitted signal, and the filtered carrier is ``x3``
of the same link. Both spectra share one normalisation so the levels compare.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, welch

from chaotic_pfc.comms.transmitter import transmit, transmit_order_n
from chaotic_pfc.config import DEFAULT_CONFIG as CFG
from chaotic_pfc.dynamics.signals import text_message
from chaotic_pfc.plotting.figures import setup_rc

ROOT = Path(__file__).resolve().parents[3]
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
    setup_rc()
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

    fig, ax = plt.subplots(figsize=(3.9, 1.9))
    ax.axvspan(CHANNEL_EDGE, 1.0, color="0.88", zorder=0)
    ax.semilogy(w1, p1 / ref, color="black", linewidth=1.0, label="sem filtro")
    ax.semilogy(
        w3, p3 / ref, color="#d55e00", linewidth=1.0, linestyle="--", label="com o filtro do enlace"
    )
    ax.axvline(CHANNEL_EDGE, color="0.35", linewidth=0.8, linestyle=":")
    ax.text(
        CHANNEL_EDGE - 0.01,
        2e-6,
        "limite do canal",
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-9, 2.0)
    ax.set_xlabel(r"$\omega/\pi$", fontsize=9)
    ax.set_ylabel("densidade espectral", fontsize=9)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUT, format="pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

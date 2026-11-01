#!/usr/bin/env python3
"""Render the style samples for the advisor's approval.

Four figures, one per style decision:

* ``pfc-fig1-henon``      PFC Fig. 1 - two series, blue solid and red dashed
* ``pfc-fig9-mapa``       PFC Fig. 9 - classification map, Portuguese, comma
* ``pfc-fig14-psd``       PFC Fig. 14 - spectra, Portuguese, comma
* ``jcis-fig3-mapa``      JCIS Fig. 3 - classification map, English, point

Each one is written as a vector PDF and as a PNG rendered from that PDF, in
``revisao/amostras-estilo/``, together with a grayscale copy of the two maps.
The PNG is what the reviewer sees in the chat; the PDF is what goes into the
manuscript.

Nothing is written into ``articles/``: the samples are generated from the same
sweeps and the same pipeline, only changing type, colours and size.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image
from scipy.signal import firwin, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chaotic_pfc.analysis.sweep import load_sweep  # noqa: E402
from chaotic_pfc.analysis.sweep_plotting import plot_classification_interleaved  # noqa: E402
from chaotic_pfc.comms.transmitter import transmit, transmit_order_n  # noqa: E402
from chaotic_pfc.config import DEFAULT_CONFIG as CFG  # noqa: E402
from chaotic_pfc.dynamics.maps import henon_generalised  # noqa: E402
from chaotic_pfc.dynamics.signals import text_message  # noqa: E402
from tools.paper_style import (  # noqa: E402
    CLASS_COLORS,
    SERIES,
    apply,
    class_handles,
    install_sweep_palette,
    luminance,
    save,
)

OUT = ROOT / "revisao/amostras-estilo"
PDFTOPPM = shutil.which("pdftoppm") or "/usr/bin/pdftoppm"
SWEEPS = ROOT / "data/sweeps"
CHANNEL_EDGE = 0.99
JCIS_LOWPASS = "kaiser/lowpass/beta_17.00"


# ── helpers ────────────────────────────────────────────────────────────────


def png_from_pdf(pdf: Path, dpi: int = 200) -> Path:
    """Render the vector sample to PNG so it can be inspected in the chat."""
    png = pdf.with_suffix(".png")
    subprocess.run(  # noqa: S603 - fixed argv, no user input
        [PDFTOPPM, "-r", str(dpi), "-png", "-singlefile", str(pdf), str(pdf.with_suffix(""))],
        check=True,
        capture_output=True,
    )
    produced = pdf.with_suffix(".png")
    if produced != png:  # pragma: no cover - defensive
        produced.rename(png)
    return png


def grayscale_copy(png: Path) -> tuple[Path, dict[str, int]]:
    """Write a grayscale copy and measure the gray value of every class.

    The values are read from the rendered image, not computed from the hex
    codes: they are what a grayscale printer produces.
    """
    image = Image.open(png).convert("RGB")
    gray = image.convert("L")
    out = png.with_name(png.stem + "-cinza.png")
    gray.save(out)

    pixels = np.asarray(image).reshape(-1, 3)
    levels = np.asarray(gray).reshape(-1)
    measured: dict[str, int] = {}
    for name, hex_color in CLASS_COLORS.items():
        target = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)])
        mask = np.all(np.abs(pixels.astype(int) - target) <= 2, axis=1)
        if mask.any():
            measured[name] = int(np.median(levels[mask]))
    return out, measured


def series_style(index: int) -> dict[str, str]:
    """Colour and line style of series ``index`` in the advisor order."""
    color, ls = SERIES[index % len(SERIES)]
    return {"color": color, "linestyle": ls}


# ── samples ────────────────────────────────────────────────────────────────


def pfc_fig1() -> None:
    """PFC Fig. 1: x1(n) for two close initial conditions."""
    st = apply("pfc")
    n_steps = 99
    fig, axes = st.subplots(2, 1, fraction=1.0, height_in=2.4, sharex=True)
    n = np.arange(n_steps + 1)
    for ax, (label, alpha) in zip(axes, (("a", 1.4), ("b", 0.9)), strict=True):
        ax.set_yticks([-1.0, 0.0, 1.0] if label == "a" else [0.0, 1.0])
        ax.yaxis.set_major_formatter(st.formatter)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.xaxis.set_major_formatter(st.formatter)
        x1_a, _ = henon_generalised(n_steps, x0=0.0, y0=0.0, alpha=alpha, beta=0.3)
        x1_b, _ = henon_generalised(n_steps, x0=0.0, y0=0.001, alpha=alpha, beta=0.3)
        ax.plot(n, x1_a, lw=1.0, label=r"$x_1(0)=x_2(0)=0$", **series_style(0))
        ax.plot(
            n,
            x1_b,
            lw=1.0,
            label=r"$x_1(0)=0,\ x_2(0)=0{,}001$",
            **series_style(1),
        )
        title = f"({label})  " + rf"$\alpha={st.number(alpha)}$, $\beta={st.number(0.3)}$"
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylabel(r"$x_1(n)$")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
    )
    axes[1].set_xlabel("$n$")
    fig.subplots_adjust(left=0.088, right=0.995, top=0.80, bottom=0.17, hspace=0.7)
    save(fig, OUT / "pfc-fig1-henon.pdf")


def _classification(
    article: str, fraction: float, aspect: float, name: str, *, legend: str = "inside"
) -> Path:
    st = apply(article)
    install_sweep_palette()
    sweep = load_sweep(SWEEPS / JCIS_LOWPASS / "variables_lyapunov.npz")
    fig = plot_classification_interleaved(
        sweep,
        lang=st.lang,
        figsize=st.figsize(fraction, aspect=aspect),
        label_fontsize=st.caption_pt,
        tick_fontsize=st.caption_pt,
        legend=False,
        ytick_step=0.2,
        xtick_step=10,
    )
    ax = fig.axes[0]
    ax.yaxis.set_major_formatter(st.formatter)
    ax.grid(False)
    handles = class_handles(st.lang)
    if legend == "inside":
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=st.caption_pt,
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
            handlelength=1.4,
            handleheight=0.7,
            handletextpad=0.5,
            borderpad=0.35,
            labelspacing=0.3,
        )
        fig.tight_layout(pad=0.2)
    else:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            columnspacing=1.2,
            handlelength=1.2,
            handleheight=0.9,
        )
        fig.subplots_adjust(left=0.13, right=0.99, top=0.80, bottom=0.17)
    pdf = save(fig, OUT / f"{name}.pdf")
    return pdf


def pfc_map() -> None:
    """PFC Fig. 9, one panel: Portuguese, comma, 11 pt."""
    _classification("pfc", 0.49, 0.67, "pfc-fig9-mapa", legend="outside")


def jcis_map() -> None:
    """JCIS Fig. 3: English, point, 8 pt."""
    _classification("jcis", 1.0, 0.66, "jcis-fig3-mapa", legend="inside")


def pfc_psd() -> None:
    """PFC Fig. 14: spectra with and without the link filter."""
    st = apply("pfc")
    cfg = CFG.comm
    message = text_message(cfg.message_text, cfg.N, bit_period=cfg.message_period)
    sent = transmit(message, mu=cfg.mu, alpha=1.4, beta=0.3, x0=0.0, y0=0.0)
    x1 = sent - cfg.mu * message

    coeffs = firwin(numtaps=9, cutoff=0.52, window="hamming", pass_zero=True, fs=2.0)
    rng = np.random.default_rng(CFG.seed)
    x0 = 0.5 * rng.random(len(coeffs))
    _, state = transmit_order_n(message, coeffs, mu=cfg.mu, alpha=1.4, beta=0.3, x0=x0)
    x3 = state[2]

    def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD in cycles per sample, as in the PFC generator."""
        freqs, power = welch(x[:65536], fs=1.0, nperseg=1024, noverlap=512, detrend=False)
        return 2.0 * freqs, power

    w1, p1 = spectrum(x1)
    w3, p3 = spectrum(x3)
    ref = p1.max()

    fig, ax = st.subplots(1, 1, fraction=0.62, aspect=0.45)
    ax.axvspan(CHANNEL_EDGE, 1.0, color="0.92", zorder=0, lw=0)
    ax.semilogy(w1, p1 / ref, lw=1.0, label="sem filtro", **series_style(2))
    ax.semilogy(w3, p3 / ref, lw=1.0, label="com o filtro do enlace", **series_style(1))
    ax.plot([], [], color="0.35", lw=0.8, ls=":", label="limite do canal")
    ax.axvline(CHANNEL_EDGE, color="0.35", lw=0.8, ls=":")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-9, 5.0)
    ax.set_xlabel(r"$\omega/\pi$")
    ax.set_ylabel("densidade espectral")
    ax.xaxis.set_major_formatter(st.formatter)
    ax.grid(True, which="major")
    ax.legend(loc="lower left", borderaxespad=0.4)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.96, bottom=0.235)
    save(fig, OUT / "pfc-fig14-psd.pdf")


# ── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pfc_fig1()
    pfc_map()
    pfc_psd()
    jcis_map()

    report = []
    for pdf in sorted(OUT.glob("*.pdf")):
        png = png_from_pdf(pdf)
        line = f"{pdf.name:24s} -> {png.name}"
        if "mapa" in pdf.name:
            gray, levels = grayscale_copy(png)
            line += f", {gray.name}, cinza=" + ", ".join(f"{k} {v}" for k, v in levels.items())
            for key, value in levels.items():
                expected = luminance(CLASS_COLORS[key])[2]
                if abs(value - expected) > 2:
                    raise SystemExit(f"{key}: cinza medido {value}, esperado {expected}")
        report.append(line)
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

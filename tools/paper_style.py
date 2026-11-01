#!/usr/bin/env python3
r"""Single figure style for the PFC and the JCIS manuscripts.

One module, two articles.  Every figure of both articles should be produced
through :func:`apply` so that type, colours and line weights are identical
across the two texts; the only differences are the ones the page geometry and
the language impose: text width, caption size and the decimal separator.

Provenance of each choice
-------------------------
Type face
    Both manuscripts are typeset with ``mathptmx`` (Times clone), the PFC
    through ``pfccefsa.cls`` and the JCIS through IEEEtran plus
    ``mathptmx``.  Figures therefore use ``Nimbus Roman`` (the URW Times that
    ``mathptmx`` itself selects) with the STIX math font, which is
    Times-compatible, so figure and body text share one family.
Type size
    The figure text is set to the caption size *in print*, and the figure is
    generated already at its final printed width, so no scaling is applied by
    ``\includegraphics``:

    * PFC: ``pfccefsa.cls`` sets body 12 bp and caption 11 bp (``\tamonze``);
      text width is 510.22 bp = 7.086 in.
    * JCIS: IEEEtran journal sets captions in ``\footnotesize`` = 8 pt; the
      column is 3.5 in wide.

Classification colours
    The advisor's convention, as approved in the JCIS: chaotic red, non-chaotic
    light blue, divergent light gray.  The exact values are the ones in the
    JCIS figures (``#FF0000``, ``#4DBEEE``, ``#E0E0E0``); the luminance audit
    below shows the three separate in grayscale.

Line series
    Advisor's MATLAB sources (``figure_1.m``, ``figure_5.m``) and the approved
    JCIS Fig. 1 use a blue solid curve followed by a red dashed curve, with
    black and gray reserved for references.  :data:`SERIES` fixes that order.

Line weights and markers
    The MATLAB figures use ``LineWidth`` 2 on curves and axes, ``FontSize``
    20-24 on ticks and ``MarkerSize`` 1 for orbit points.  Those figures are
    authored at roughly twice the printed size, so the print equivalents used
    here are 1.0 pt curves, 0.8 pt frame, 0.4 pt grid and 1.5-3 pt markers.

Usage
-----
>>> from tools.paper_style import apply, save
>>> st = apply("pfc")                       # sets rcParams, returns geometry
>>> fig, ax = st.subplots(1, 1, fraction=0.49, aspect=0.67)
>>> ...
>>> save(fig, "articles/pfc/figures/example.pdf")

Run ``python tools/paper_style.py`` to print the audit (widths, sizes and the
grayscale luminance of the classification palette).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter

# ═══════════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════════

#: Advisor/JCIS classification colours (see the module docstring).
CLASS_COLORS: dict[str, str] = {
    "chaotic": "#FF0000",
    "nonchaotic": "#4DBEEE",
    "divergent": "#E0E0E0",
}
COLOR_GAP = "white"

#: Line-series order of the advisor: blue solid, red dashed, then black and
#: gray for references (JCIS Fig. 1 and the MATLAB sources).
SERIES: tuple[tuple[str, str], ...] = (
    ("#1F77B4", "-"),
    ("#D62728", "--"),
    ("#000000", "-"),
    ("#7F7F7F", "--"),
)

#: Order of the classification key: the same order of Eq. (9) of both
#: manuscripts (divergent, chaotic, non-chaotic, in the ``cases`` block).
CLASS_ORDER: tuple[str, ...] = ("divergent", "chaotic", "nonchaotic")

#: Labels of the classification key, in the order used by both manuscripts.
CLASS_LABELS: dict[str, dict[str, str]] = {
    "pt": {"chaotic": "Caótico", "nonchaotic": "Não caótico", "divergent": "Divergente"},
    "en": {"chaotic": "Chaotic", "nonchaotic": "Non-chaotic", "divergent": "Divergent"},
}

# ═══════════════════════════════════════════════════════════════════════════
# Page geometry and type size of each article
# ═══════════════════════════════════════════════════════════════════════════

PFC_TEXTWIDTH_IN = 510.22 / 72.0  # 510.22 bp of pfccefsa minus the margins
JCIS_COLUMN_IN = 3.5  # IEEEtran journal column

ARTICLES: dict[str, dict[str, Any]] = {
    "pfc": {
        "textwidth_in": PFC_TEXTWIDTH_IN,
        "caption_pt": 11.0,  # \tamonze: 11 bp
        "lang": "pt",
        "decimal": ",",
        "font_serif": "Liberation Serif",  # body font of pfccefsa under XeLaTeX
    },
    "jcis": {
        "textwidth_in": JCIS_COLUMN_IN,
        "caption_pt": 8.0,  # IEEEtran \footnotesize
        "lang": "en",
        "decimal": ".",
        # Times design in TrueType.  Nimbus Roman is the exact mathptmx face,
        # but it is CFF-only here and would have to be embedded as Type 3.
        "font_serif": "STIXGeneral",
    },
}


@dataclass(frozen=True)
class Style:
    """Geometry and typography of one article, after :func:`apply`."""

    article: str
    textwidth_in: float
    caption_pt: float
    lang: str
    decimal: str

    def width(self, fraction: float = 1.0) -> float:
        """Final printed width of a figure that occupies ``fraction`` of the text."""
        return self.textwidth_in * fraction

    def figsize(
        self,
        fraction: float = 1.0,
        *,
        height_in: float | None = None,
        aspect: float | None = None,
    ) -> tuple[float, float]:
        """Figure size in inches at the final printed width.

        Pass exactly one of ``height_in`` (absolute printed height) or
        ``aspect`` (height/width of the axes area).
        """
        width = self.width(fraction)
        if (height_in is None) == (aspect is None):
            raise ValueError("pass exactly one of height_in or aspect")
        return (width, height_in if height_in is not None else width * float(aspect))

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        *,
        fraction: float = 1.0,
        height_in: float | None = None,
        aspect: float | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """``plt.subplots`` already sized for the page."""
        return plt.subplots(
            nrows,
            ncols,
            figsize=self.figsize(fraction, height_in=height_in, aspect=aspect),
            **kwargs,
        )

    def number(self, value: float, _pos: float | None = None) -> str:
        """Format a tick with the article's decimal separator."""
        text = f"{value:g}"
        if self.decimal == ",":
            text = text.replace(".", ",")
        return text.replace("-", "\u2212")

    @property
    def formatter(self) -> FuncFormatter:
        """Tick formatter that honours the article's decimal separator."""
        return FuncFormatter(self.number)


def apply(article: str) -> Style:
    """Set the global rcParams of ``article`` and return its :class:`Style`."""
    if article not in ARTICLES:
        raise ValueError(f"article must be one of {sorted(ARTICLES)}")
    cfg = ARTICLES[article]
    pt = float(cfg["caption_pt"])
    plt.rcParams.update(
        {
            # Type: Times through the URW clone used by mathptmx, STIX maths.
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": [str(cfg["font_serif"]), "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            # Every text element is the caption size on the printed page.
            "font.size": pt,
            "axes.labelsize": pt,
            "axes.titlesize": pt,
            "xtick.labelsize": pt,
            "ytick.labelsize": pt,
            "legend.fontsize": pt,
            "figure.titlesize": pt,
            # Lines and marks (print equivalents of the MATLAB 2 pt curves).
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "grid.color": "black",
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "patch.linewidth": 0.5,
            # Output: vector PDF with embedded Type 42 fonts, no dates.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "axes.grid": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
        }
    )
    return Style(
        article=article,
        textwidth_in=float(cfg["textwidth_in"]),
        caption_pt=pt,
        lang=str(cfg["lang"]),
        decimal=str(cfg["decimal"]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Classification palette for the sweep plotting pipeline
# ═══════════════════════════════════════════════════════════════════════════


def class_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete colormap of the three classes plus the interleaved gap."""
    cmap = ListedColormap(
        [
            CLASS_COLORS["nonchaotic"],
            CLASS_COLORS["chaotic"],
            CLASS_COLORS["divergent"],
            COLOR_GAP,
        ]
    )
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 2.5, 3.5], cmap.N)
    return cmap, norm


def install_sweep_palette() -> None:
    """Point the sweep plotting module at :data:`CLASS_COLORS`.

    The pipeline keeps its own constants in
    ``chaotic_pfc.analysis.sweep_plotting._common``; figure scripts call this
    once so the whole sweep machinery uses the advisor palette.
    """
    from chaotic_pfc.analysis.sweep_plotting import _classification, _common

    _common.COLOR_CHAOTIC = CLASS_COLORS["chaotic"]
    _common.COLOR_PERIODIC = CLASS_COLORS["nonchaotic"]
    _common.COLOR_UNBOUNDED = CLASS_COLORS["divergent"]
    _classification.COLOR_CHAOTIC = CLASS_COLORS["chaotic"]
    _classification.COLOR_PERIODIC = CLASS_COLORS["nonchaotic"]
    _classification.COLOR_UNBOUNDED = CLASS_COLORS["divergent"]
    _classification._CLASS_COLORS.update(
        {
            -1.0: CLASS_COLORS["nonchaotic"],
            0.0: CLASS_COLORS["chaotic"],
            2.0: CLASS_COLORS["divergent"],
        }
    )


def class_handles(lang: str = "pt") -> list[Any]:
    """Legend patches of the classification key, in the order of Eq. (9)."""
    from matplotlib.patches import Patch

    labels = CLASS_LABELS[lang]
    return [
        Patch(
            facecolor=CLASS_COLORS[key],
            edgecolor="gray",
            linewidth=0.5,
            label=labels[key],
        )
        for key in CLASS_ORDER
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════


def save(fig: Any, path: str | Path) -> Path:
    r"""Write ``fig`` as a deterministic PDF of exactly ``figsize`` inches.

    ``bbox_inches`` is deliberately *not* tight: the page keeps the nominal
    size, so ``\includegraphics[width=\linewidth]`` inserts the artwork at
    1:1 and the type keeps the size it was authored with.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        facecolor="white",
        metadata={"Creator": None, "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Grayscale audit
# ═══════════════════════════════════════════════════════════════════════════


def _srgb_to_linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> tuple[float, float, int]:
    """Return ``(Y, L*, gray8)`` of an sRGB colour.

    ``Y`` is the relative luminance of WCAG 2.x, ``L*`` the CIE 1976 lightness
    (0-100) and ``gray8`` the 8-bit value produced by the ITU-R BT.601 luma
    used by grayscale conversion (PIL, most printers), that is
    ``0.299 R + 0.587 G + 0.114 B`` on the gamma-encoded channels.
    """
    h = hex_color.lstrip("#")
    r, g, b = (_srgb_to_linear(int(h[i : i + 2], 16) / 255.0) for i in (0, 2, 4))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    lstar = 116.0 * y ** (1.0 / 3.0) - 16.0 if y > 0.008856 else 903.3 * y
    ri, gi, bi = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    gray = round(0.299 * ri + 0.587 * gi + 0.114 * bi)
    return y, lstar, int(gray)


def audit() -> str:
    """Human-readable audit of widths, sizes and the classification palette."""
    lines = ["Figure style audit", ""]
    for name, cfg in ARTICLES.items():
        lines.append(
            f"{name:5s} text width {cfg['textwidth_in']:.3f} in, "
            f"caption {cfg['caption_pt']:.1f} pt, decimal '{cfg['decimal']}'"
        )
    lines += ["", "Classification palette (advisor convention)", ""]
    lines.append(f"{'class':12s} {'hex':9s} {'Y':>6s} {'L*':>6s} {'gray':>5s}")
    values = []
    for key, hex_color in CLASS_COLORS.items():
        y, lstar, gray = luminance(hex_color)
        values.append((key, lstar, gray))
        lines.append(f"{key:12s} {hex_color:9s} {y:6.3f} {lstar:6.1f} {gray:5d}")
    ls = [v[1] for v in values]
    gs = [v[2] for v in values]
    lines += [
        "",
        f"L* steps: chaotic->nonchaotic {abs(ls[0] - ls[1]):.0f}, "
        f"nonchaotic->divergent {abs(ls[1] - ls[2]):.0f}, "
        f"chaotic->divergent {abs(ls[0] - ls[2]):.0f}",
        f"BT.601 gray steps: {abs(gs[0] - gs[1])} and {abs(gs[1] - gs[2])} of 255",
        "",
        "Line series: " + ", ".join(f"{c} {s}" for c, s in SERIES),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the figure style audit.")
    parser.parse_args()
    print(audit())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

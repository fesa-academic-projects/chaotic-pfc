"""
sweep_plotting.py
=================
Figures for Lyapunov classification maps produced by
:mod:`chaotic_pfc.sweep`.

Four plot types are provided:

1. :func:`plot_heatmap_continuous` — raw λ_max heatmap.
2. :func:`plot_classification_simple` — discrete periodic / chaotic /
   unbounded classification.
3. :func:`plot_classification_separated` — same classification with
   per-order vertical separators.
4. :func:`plot_classification_interleaved` — paper-style layout with
   gaps between orders.

All four accept a :class:`~chaotic_pfc.sweep.SweepResult` (or its
individual arrays) and optionally a ``save_path``. They return the
:class:`matplotlib.figure.Figure` so callers can compose or display
them. The module also re-uses the RC params from
:mod:`chaotic_pfc.plotting`, so sweep figures look consistent with the
rest of the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

# Pull in global RC params (STIX fonts, vector SVG, etc.)
from .plotting import setup_rc
from .sweep import SweepResult

setup_rc()


# ═══════════════════════════════════════════════════════════════════════════
# Palette and discrete classification
# ═══════════════════════════════════════════════════════════════════════════

COLOR_PERIODIC: str = "#4DBEEE"
COLOR_CHAOTIC: str = "red"
COLOR_UNBOUNDED: str = "#E0E0E0"
COLOR_GAP: str = "white"

_cmap_disc = mcolors.ListedColormap(
    [
        COLOR_PERIODIC,
        COLOR_CHAOTIC,
        COLOR_UNBOUNDED,
        COLOR_GAP,
    ]
)
_bounds_disc = [-1.5, -0.5, 0.5, 2.5, 3.5]
_norm_disc = mcolors.BoundaryNorm(_bounds_disc, _cmap_disc.N)

_cmap_3 = mcolors.ListedColormap(
    [
        COLOR_PERIODIC,
        COLOR_CHAOTIC,
        COLOR_UNBOUNDED,
    ]
)
_bounds_3 = [-1.5, -0.5, 0.5, 2.5]
_norm_3 = mcolors.BoundaryNorm(_bounds_3, _cmap_3.N)

_LEGEND_HANDLES = [
    Patch(facecolor=COLOR_PERIODIC, edgecolor="gray", label="Periodic orbits"),
    Patch(facecolor=COLOR_CHAOTIC, edgecolor="gray", label="Chaotic orbits"),
    Patch(facecolor=COLOR_UNBOUNDED, edgecolor="gray", label="Unbounded orbits"),
]

_YTICKS = np.arange(0.0, 1.01, 0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def classify(h: NDArray) -> NDArray:
    """Map raw λ_max values to discrete classes.

    Returns an array with the same shape as ``h`` where each entry is:

    * ``-1`` — periodic orbit (λ_max ≤ 0)
    * ``0``  — chaotic orbit (λ_max > 0)
    * ``2``  — unbounded / divergent (NaN in ``h``)

    The unusual integer codes match the :class:`matplotlib.colors.BoundaryNorm`
    bins used below.
    """
    out = np.full_like(h, np.nan, dtype=np.float64)
    mask_diverged = np.isnan(h)
    mask_periodic = (~mask_diverged) & (h <= 0.0)
    mask_chaotic = (~mask_diverged) & (h > 0.0)

    out[mask_diverged] = 2
    out[mask_periodic] = -1
    out[mask_chaotic] = 0
    return out


def _axis_cosmetics(ax, ylabel_fs: int = 24) -> None:
    ax.set_xlabel(r"$N_z$", fontsize=ylabel_fs)
    ax.set_ylabel(r"$\omega_c/\pi$", fontsize=ylabel_fs)
    ax.set_yticks(_YTICKS)
    ax.set_ylim(0.0, 1.0)
    for yt in _YTICKS:
        ax.axhline(y=yt, color="black", linewidth=0.4)


def _save(fig: Figure, path: str | Path | None) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Continuous heatmap
# ═══════════════════════════════════════════════════════════════════════════


def plot_heatmap_continuous(
    result: SweepResult | None = None,
    *,
    h: NDArray | None = None,
    orders: NDArray | None = None,
    cutoffs: NDArray | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Continuous λ_max heatmap over the (N_z, ω_c/π) plane."""
    h, Nz, cutoffs = _unpack(result, h, orders, cutoffs)

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(Nz, cutoffs, h.T, shading="nearest")
    fig.colorbar(pcm, ax=ax, label=r"$\lambda_{\max}$")
    _axis_cosmetics(ax)
    ax.grid(True, axis="x", color="gray", linewidth=0.3)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. Simple discrete classification
# ═══════════════════════════════════════════════════════════════════════════


def plot_classification_simple(
    result: SweepResult | None = None,
    *,
    h: NDArray | None = None,
    orders: NDArray | None = None,
    cutoffs: NDArray | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Discrete classification map (periodic / chaotic / unbounded)."""
    h, Nz, cutoffs = _unpack(result, h, orders, cutoffs)
    h_color = classify(h)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(
        Nz,
        cutoffs,
        h_color.T,
        cmap=_cmap_3,
        norm=_norm_3,
        shading="nearest",
    )
    _axis_cosmetics(ax)
    ax.grid(True, axis="x", color="gray", linewidth=0.3)
    ax.legend(
        handles=_LEGEND_HANDLES,
        fontsize=12,
        loc="upper right",
        framealpha=0.9,
        edgecolor="gray",
    )
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. Classification with vertical separators per order
# ═══════════════════════════════════════════════════════════════════════════


def plot_classification_separated(
    result: SweepResult | None = None,
    *,
    h: NDArray | None = None,
    orders: NDArray | None = None,
    cutoffs: NDArray | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Classification map with a black line between consecutive orders."""
    h, Nz, cutoffs = _unpack(result, h, orders, cutoffs)
    h_color = classify(h)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pcolormesh(
        Nz,
        cutoffs,
        h_color.T,
        cmap=_cmap_3,
        norm=_norm_3,
        shading="nearest",
    )
    _axis_cosmetics(ax)
    ax.grid(False)
    ax.legend(
        handles=_LEGEND_HANDLES,
        fontsize=12,
        loc="upper right",
        framealpha=0.9,
        edgecolor="gray",
    )
    for xval in Nz:
        ax.axvline(x=xval - 0.5, color="black", linewidth=0.5)
    ax.axvline(x=Nz[-1] + 0.5, color="black", linewidth=0.5)
    ax.set_xticks(Nz[::2])
    ax.set_xticklabels(Nz[::2], fontsize=12)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. Paper-style interleaved bars (3 data slots + 1 gap per order)
# ═══════════════════════════════════════════════════════════════════════════


def plot_classification_interleaved(
    result: SweepResult | None = None,
    *,
    h: NDArray | None = None,
    orders: NDArray | None = None,
    cutoffs: NDArray | None = None,
    save_path: str | Path | None = None,
    data_slots: int = 3,
    gap_slots: int = 1,
) -> Figure:
    """Publication-style layout with gaps between adjacent orders.

    Each order occupies ``data_slots`` columns of coloured data followed
    by ``gap_slots`` blank columns, producing the striped appearance
    used in Baptista et al.
    """
    h, Nz, cutoffs = _unpack(result, h, orders, cutoffs)
    h_color = classify(h)

    slot_total = data_slots + gap_slots
    Ncoef = len(Nz)
    Ncut = len(cutoffs)
    total_slots = Ncoef * slot_total
    h_color_exp = np.full((total_slots, Ncut), 3.0)

    for i in range(Ncoef):
        start = i * slot_total
        for s in range(data_slots):
            h_color_exp[start + s, :] = h_color[i, :]

    x_exp = np.arange(total_slots)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.pcolormesh(
        x_exp,
        cutoffs,
        h_color_exp.T,
        cmap=_cmap_disc,
        norm=_norm_disc,
        shading="nearest",
    )

    ax.set_xlabel(r"$N_z$", fontsize=16)
    ax.set_ylabel(r"$\omega_c/\pi$", fontsize=16)
    ax.set_yticks(_YTICKS)
    ax.set_ylim(0.0, 1.0)
    for yt in _YTICKS:
        ax.axhline(y=yt, color="black", linewidth=0.4)

    ax.grid(False)
    for i in range(Ncoef + 1):
        ax.axvline(x=i * slot_total - 0.5, color="black", linewidth=0.6)

    tick_vals: Iterable[int] = [1, *list(range(5, int(Nz[-1]) + 1, 5))]
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for nz_val in tick_vals:
        idx = np.where(Nz == nz_val)[0]
        if len(idx) > 0:
            center = idx[0] * slot_total + (data_slots - 1) / 2.0
            tick_positions.append(center)
            tick_labels.append(str(nz_val))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.set_xlim(-0.5, total_slots - 0.5)

    ax.legend(
        handles=_LEGEND_HANDLES,
        fontsize=9,
        loc="upper right",
        framealpha=0.95,
        edgecolor="gray",
        fancybox=False,
        handlelength=1.2,
        handleheight=0.8,
        borderpad=0.4,
    )

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: produce the full four-figure set
# ═══════════════════════════════════════════════════════════════════════════

FIGURE_FILENAMES: tuple[str, ...] = (
    "fig1_heatmap_continuous",
    "fig2_classification",
    "fig3_classification_separated",
    "fig4_classification_interleaved",
)


def plot_all(
    result: SweepResult,
    out_dir: str | Path,
    *,
    fmt: str = "png",
    close_figures: bool = True,
) -> list[Path]:
    """Generate the four standard figures for a sweep and save them to
    ``out_dir/<fig>.{fmt}``. Returns the list of written paths.

    The output directory is created if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotters = (
        plot_heatmap_continuous,
        plot_classification_simple,
        plot_classification_separated,
        plot_classification_interleaved,
    )

    paths: list[Path] = []
    for fname, plotter in zip(FIGURE_FILENAMES, plotters, strict=False):
        path = out_dir / f"{fname}.{fmt}"
        fig = plotter(result, save_path=path)
        if close_figures:
            plt.close(fig)
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# Private: argument unpacking
# ═══════════════════════════════════════════════════════════════════════════


def _unpack(
    result: SweepResult | None,
    h: NDArray | None,
    orders: NDArray | None,
    cutoffs: NDArray | None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Resolve the two accepted calling conventions into (h, Nz, cutoffs)."""
    if result is not None:
        h = result.h
        orders = result.orders
        cutoffs = result.cutoffs
    if h is None or orders is None or cutoffs is None:
        raise ValueError("Provide either a SweepResult or the (h, orders, cutoffs) triple.")
    Nz = np.asarray(orders) - 1
    return np.asarray(h), Nz, np.asarray(cutoffs)

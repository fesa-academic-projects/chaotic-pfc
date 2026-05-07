"""
sweep_plotting.py
=================
Figures for Lyapunov classification maps produced by
:mod:`chaotic_pfc.sweep`.

Three plot types are provided:

1. :func:`plot_heatmap_continuous` — raw λ_max heatmap.
2. :func:`plot_classification_interleaved` — publication-style discrete
   classification (periodic / chaotic / unbounded) with gaps between
   orders.
3. :func:`plot_difficulty_map` — heatmap of the number of Lyapunov
   iterations actually used at each grid point. Only meaningful when
   the sweep was run with ``adaptive=True``; shows where the spectrum
   converges quickly (light) vs. where it needs the full budget
   (dark) — a "difficulty map" of the parameter space.

All three accept a :class:`~chaotic_pfc.sweep.SweepResult` (or its
individual arrays for the first two) and optionally a ``save_path``.
They return the :class:`matplotlib.figure.Figure` so callers can
compose or display them. The module also re-uses the RC params from
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
from ..plotting.figures import setup_rc
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
# 2. Paper-style interleaved bars (3 data slots + 1 gap per order)
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
# 3. Difficulty map: how many iterations each grid point needed
# ═══════════════════════════════════════════════════════════════════════════
#
# This figure only carries information for sweeps run with adaptive=True.
# When adaptive=False every finite cell equals Nmap, so the heatmap would
# be a single colour and we'd be misleading the reader. We require
# ``n_iters_used`` to be present and to actually vary across cells; the
# helper raises if the input is non-adaptive.


def plot_difficulty_map(
    result: SweepResult,
    *,
    save_path: str | Path | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Heatmap of Lyapunov iterations actually used at each grid point.

    A "difficulty map" of the parameter space: low values mean the
    spectrum estimate converged quickly (strongly chaotic or strongly
    stable points), high values mean the running estimate stayed within
    the convergence tolerance only after many iterations (typically
    fronteira points where \\|λ_max\\| ≈ 0). Diverged grid points are shown
    in the same light grey used for unbounded orbits in the
    classification figures, so the two layers can be visually overlaid.

    Parameters
    ----------
    result
        A :class:`~chaotic_pfc.sweep.SweepResult` produced with
        ``adaptive=True``. The function relies on
        ``result.n_iters_used`` and on ``result.metadata['Nmap_min']``
        / ``result.metadata['Nmap']`` for the colour-bar limits.
    save_path
        Optional path to write the figure to.
    cmap
        Sequential matplotlib colormap name. ``viridis`` is
        perceptually uniform and prints well in greyscale.

    Raises
    ------
    ValueError
        If ``result`` was produced with ``adaptive=False`` (the heatmap
        would be a single colour, which is misleading rather than
        informative). The error message points the user to the
        ``adaptive=True`` flag in :func:`run_sweep`.
    """
    if result.n_iters_used is None:
        raise ValueError(
            "result.n_iters_used is None: the sweep was loaded from a "
            "legacy .npz that did not record the iteration count, or "
            "the in-memory result predates the adaptive feature. "
            "Re-run with run_sweep(..., adaptive=True) to produce one."
        )
    if not result.metadata.get("adaptive", False):
        raise ValueError(
            "Difficulty map is only meaningful for sweeps with "
            "adaptive=True. The provided SweepResult was run with "
            "adaptive=False, so every finite cell trivially equals "
            "Nmap and the figure would carry no information. "
            "Pass adaptive=True to run_sweep() to enable early-stop "
            "and produce a non-trivial iteration map."
        )

    Nz = np.asarray(result.orders) - 1
    cutoffs = np.asarray(result.cutoffs)
    n_iters = np.asarray(result.n_iters_used, dtype=np.float64)

    # Colour-bar bounds: anchor to the (Nmap_min, Nmap) range used at
    # sweep time so different sweeps with the same parameters share a
    # comparable scale. Fall back to the data range if the metadata
    # is incomplete (e.g. legacy result).
    Nmap_min = result.metadata.get("Nmap_min")
    Nmap = result.metadata.get("Nmap")
    if Nmap_min is None or Nmap is None:
        finite = n_iters[np.isfinite(n_iters)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
    else:
        vmin = float(Nmap_min)
        vmax = float(Nmap)

    # Render diverged cells (NaN) in the same light grey used by the
    # classification plots for unbounded orbits, so the two figures
    # stay visually consistent.
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(COLOR_UNBOUNDED)

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(
        Nz,
        cutoffs,
        n_iters.T,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
    )
    cbar = fig.colorbar(pcm, ax=ax, label="Lyapunov iterations used")
    cbar.ax.tick_params(labelsize=12)

    _axis_cosmetics(ax)
    ax.grid(True, axis="x", color="gray", linewidth=0.3)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: produce the full figure set
# ═══════════════════════════════════════════════════════════════════════════
#
# ``plot_all`` always emits the two classification figures (1, 2) and
# *additionally* emits the difficulty map (3) when the sweep was run
# with ``adaptive=True``. The non-adaptive case is detected by
# inspecting ``result.metadata['adaptive']`` (and falling back to a
# ``None`` n_iters_used check, for legacy results loaded from disk).
#
# ``FIGURE_FILENAMES`` lists the names that are *always* produced;
# difficulty-map filename is appended at runtime when applicable. This
# keeps the constant useful as a stable contract for callers that want
# to predict the always-present outputs while still allowing the third
# file to appear when it carries information.

FIGURE_FILENAMES: tuple[str, ...] = (
    "fig1_heatmap_continuous",
    "fig2_classification_interleaved",
)

DIFFICULTY_FIGURE_FILENAME: str = "fig3_difficulty_map"


def _has_difficulty_data(result: SweepResult) -> bool:
    """Return True iff a difficulty map can be plotted from ``result``.

    A difficulty map is informative only when the sweep was run with
    ``adaptive=True``. Otherwise every finite cell trivially equals
    ``Nmap`` and the figure would carry no information.
    """
    if result.n_iters_used is None:
        return False
    return bool(result.metadata.get("adaptive", False))


def plot_all(
    result: SweepResult,
    out_dir: str | Path,
    *,
    fmt: str = "png",
    close_figures: bool = True,
) -> list[Path]:
    """Generate the standard figures for a sweep and save them to
    ``out_dir/<fig>.{fmt}``. Returns the list of written paths.

    Always produces the two classification figures listed in
    :data:`FIGURE_FILENAMES`. Additionally produces
    ``fig3_difficulty_map.{fmt}`` (see :data:`DIFFICULTY_FIGURE_FILENAME`)
    when ``result`` was generated with ``adaptive=True`` — the figure is
    silently skipped for non-adaptive sweeps because it would be
    monochromatic and therefore uninformative.

    The output directory is created if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotters: tuple = (
        plot_heatmap_continuous,
        plot_classification_interleaved,
    )

    paths: list[Path] = []
    for fname, plotter in zip(FIGURE_FILENAMES, plotters, strict=True):
        path = out_dir / f"{fname}.{fmt}"
        fig = plotter(result, save_path=path)
        if close_figures:
            plt.close(fig)
        paths.append(path)

    if _has_difficulty_data(result):
        path = out_dir / f"{DIFFICULTY_FIGURE_FILENAME}.{fmt}"
        fig = plot_difficulty_map(result, save_path=path)
        if close_figures:
            plt.close(fig)
        paths.append(path)

    return paths


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

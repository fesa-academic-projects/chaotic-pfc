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
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

from chaotic_pfc._i18n import t
from chaotic_pfc.analysis.sweep import SweepResult, load_sweep

# Pull in global RC params (STIX fonts, vector SVG, etc.)
from chaotic_pfc.plotting.figures import _save as _figures_save

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


def _build_legend_handles(lang: str = "pt") -> list:
    """Build the discrete-classification legend in the requested language."""
    return [
        Patch(
            facecolor=COLOR_PERIODIC,
            edgecolor="gray",
            label=t("sweep.legend.periodic", lang=lang),
        ),
        Patch(
            facecolor=COLOR_CHAOTIC,
            edgecolor="gray",
            label=t("sweep.legend.chaotic", lang=lang),
        ),
        Patch(
            facecolor=COLOR_UNBOUNDED,
            edgecolor="gray",
            label=t("sweep.legend.unbounded", lang=lang),
        ),
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


def _pick_xticks(
    Nz: NDArray,
    slot_total: int,
    data_slots: int,
    step: int,
) -> tuple[list[float], list[str]]:
    """Choose x-tick positions that exist in the swept order grid.

    Highpass and bandstop sweeps only contain odd tap counts, so a
    naive ``range(step, ..., step)`` with ``step`` even silently yields
    an axis with no numeric labels at all. The step is halved until at
    least three ticks land on real grid points.
    """
    nz_int = np.asarray(Nz).astype(int)
    while step >= 1:
        candidates: Iterable[int] = [1, *list(range(step, int(nz_int[-1]) + 1, step))]
        positions: list[float] = []
        labels: list[str] = []
        for nz_val in candidates:
            idx = np.where(nz_int == nz_val)[0]
            if len(idx) > 0:
                positions.append(idx[0] * slot_total + (data_slots - 1) / 2.0)
                labels.append(str(nz_val))
        if len(positions) >= 3 or step == 1:
            return positions, labels
        step //= 2
    return [], []


_PANEL_LABEL_CORNERS: dict[str, tuple[float, float, str, str]] = {
    "upper left": (0.015, 0.975, "left", "top"),
    "upper right": (0.985, 0.975, "right", "top"),
    "lower left": (0.015, 0.025, "left", "bottom"),
    "lower right": (0.985, 0.025, "right", "bottom"),
}

# Sit just above the top spine, flush with it. No opaque patch: there is
# no data up here to hide, so a box would only add visual noise.
_PANEL_LABEL_ABOVE: dict[str, tuple[float, str]] = {
    "above left": (0.0, "left"),
    "above": (0.5, "center"),
    "above right": (1.0, "right"),
}


def _draw_panel_label(
    fig: Figure,
    ax: plt.Axes,
    text: str,
    fontsize: float,
    loc: str,
) -> None:
    """Draw a sub-panel identifier such as ``"(a)"`` on *ax*."""
    if loc in _PANEL_LABEL_ABOVE:
        x, ha = _PANEL_LABEL_ABOVE[loc]
        ax.text(
            x,
            1.0,
            text,
            transform=ax.transAxes,
            ha=ha,
            va="bottom",
            fontsize=fontsize,
            zorder=20,
        )
        ax.margins(y=0.0)
        return

    if loc in _PANEL_LABEL_CORNERS:
        x, y, ha, va = _PANEL_LABEL_CORNERS[loc]
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            zorder=20,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.6,
                "boxstyle": "square,pad=0.25",
            },
        )
        return

    if loc != "below":
        raise ValueError(
            f"panel_label_loc must be 'below', one of "
            f"{sorted(_PANEL_LABEL_ABOVE)}, or one of "
            f"{sorted(_PANEL_LABEL_CORNERS)}; got {loc!r}"
        )

    # Anchor the identifier a fixed number of points below the real ink
    # of the x-axis (ticks + label), measured after a draw pass.
    # Reserving a fraction of the figure height instead would leave a
    # gap that scales with the panel, which reads as a stray caption.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.xaxis.get_tightbbox(renderer)
    y_fig = fig.transFigure.inverted().transform((0.0, bbox.y0))[1]
    gap = 5.0 / (fig.get_figheight() * 72.0)
    box = ax.get_position()
    fig.text(
        box.x0 + box.width / 2.0,
        y_fig - gap,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
    )


def make_classification_legend(
    *,
    lang: str = "pt",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (3.35, 1.0),
    fontsize: float = 9.0,
    ncol: int = 1,
) -> Figure:
    """Render the periodic/chaotic/unbounded legend as a standalone figure.

    Used when several classification panels are composed into a single
    LaTeX float and share one legend, so the colour key is not repeated
    three times.

    Parameters
    ----------
    lang
        ``"pt"`` or ``"en"``.
    save_path
        Optional output path.
    figsize
        Size in inches; make it match the panel width it sits next to.
    fontsize
        Type size in points at ``figsize`` scale.
    ncol
        Number of legend columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.legend(
        handles=_build_legend_handles(lang),
        loc="center",
        fontsize=fontsize,
        framealpha=1.0,
        edgecolor="gray",
        fancybox=False,
        handlelength=1.4,
        handleheight=0.9,
        borderpad=0.6,
        labelspacing=0.6,
        ncol=ncol,
    )
    _save(fig, save_path)
    return fig


def _save(fig: Figure, path: str | Path | None) -> None:
    """Save figure with sweep-plotting defaults (dpi=600, fixed bbox).

    ``bbox_inches=fig.bbox_inches`` prevents matplotlib from doing a
    second full render to compute the tight bounding box — every plot
    function already calls ``fig.tight_layout()`` before saving.
    """
    _figures_save(
        fig,
        path,
        dpi=600,
        bbox_inches=fig.bbox_inches,
        facecolor="white",
    )


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
    data_slots: int = 3,
    gap_slots: int = 1,
) -> Figure:
    """Continuous λ_max heatmap over the (N_z, ω_c/π) plane.

    Uses the same interleaved-column layout as
    :func:`plot_classification_interleaved` for visual consistency
    across the full figure set.

    Parameters
    ----------
    result
        :class:`SweepResult` to plot.  Mutually exclusive with the
        ``(h, orders, cutoffs)`` triple.
    h
        Lyapunov grid, shape ``(len(orders), len(cutoffs))``.
    orders
        Filter orders (x-axis), shape ``(len(orders),)``.
    cutoffs
        Cutoff frequencies (y-axis), shape ``(len(cutoffs),)``.
    save_path
        If provided, the figure is saved to this path via ``_save()``.
    data_slots, gap_slots
        Interleaved-column parameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    h, Nz, cutoffs = _unpack(result, h, orders, cutoffs)

    h_expanded = _interleaved_expand(h, data_slots, gap_slots)
    x_exp = np.arange(h_expanded.shape[0])

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Rasterized mesh at savefig dpi (600): in vector backends the mesh layer
    # becomes an embedded image while axes, ticks, labels, grid, colorbar and
    # legend remain 100 % vector. At 12 × 5 in → 7200 × 3000 px the raster
    # resolution far exceeds the data grid, so there is zero information loss.
    # antialiased=False prevents faint half-pixel lines between adjacent quads.
    pcm = ax.pcolormesh(
        x_exp, cutoffs, h_expanded.T, shading="nearest", rasterized=True, antialiased=False
    )
    fig.colorbar(pcm, ax=ax, label=r"$\lambda_{\max}$")
    _setup_interleaved_axes(ax, Nz, cutoffs, data_slots, gap_slots)
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
    lang: str = "pt",
    figsize: tuple[float, float] = (12.0, 5.0),
    label_fontsize: float = 16.0,
    tick_fontsize: float = 12.0,
    legend_fontsize: float = 10.0,
    legend: bool = True,
    ytick_step: float = 0.1,
    xtick_step: int = 5,
    panel_label: str | None = None,
    panel_label_fontsize: float = 11.0,
    panel_label_loc: str = "below",
) -> Figure:
    """Publication-style layout with gaps between adjacent orders.

    Each order occupies ``data_slots`` columns of coloured data followed
    by ``gap_slots`` blank columns, producing the striped appearance
    used in Baptista et al.

    Parameters
    ----------
    result
        :class:`SweepResult` to plot.  Mutually exclusive with the
        ``(h, orders, cutoffs)`` triple.
    h
        Lyapunov grid, shape ``(len(orders), len(cutoffs))``.
    orders
        Filter orders (x-axis), shape ``(len(orders),)``.
    cutoffs
        Cutoff frequencies (y-axis), shape ``(len(cutoffs),)``.
    save_path
        If provided, the figure is saved to this path via ``_save()``.
    data_slots
        Number of colormap quantisation levels for chaotic/periodic.
    gap_slots
        Number of white/yellow slots interleaved between classes.
    lang
        Language for axis labels (``"pt"`` or ``"en"``).
    figsize
        Figure size in inches. For camera-ready output set this to the
        *final printed size* of the figure so that ``label_fontsize``
        and friends are expressed directly in typographic points on
        the page — no scaling correction needed.
    label_fontsize, tick_fontsize, legend_fontsize
        Type sizes in points, interpreted at ``figsize`` scale.
    legend
        Draw the classification legend inside the axes. Set to
        ``False`` when several panels share one external legend (see
        :func:`make_classification_legend`).
    ytick_step
        Spacing of *labelled* y ticks. Horizontal rules are always
        drawn every 0.1, independently of this value, so a compact
        panel can keep the fine grid while thinning the labels.
    xtick_step
        Preferred spacing of x ticks. Only values actually present in
        ``orders`` are labelled; if fewer than three survive the
        filter, the step is progressively halved.
    panel_label
        Text such as ``"(a)"`` drawn centred beneath the axes, baking
        the sub-panel identifier into the artwork itself. Leave as
        ``None`` when the identifier is supplied by LaTeX (e.g. via
        ``subcaption``), otherwise it will appear twice.
    panel_label_fontsize
        Type size of ``panel_label`` in points at ``figsize`` scale.
    panel_label_loc
        Where the identifier goes: ``"above left"``, ``"above"`` and
        ``"above right"`` sit flush on top of the plot box;
        ``"below"`` places it under the x-label; ``"upper left"``,
        ``"upper right"``, ``"lower left"`` and ``"lower right"`` place
        it inside the axes, on an opaque patch so it stays readable
        over the data.

    Returns
    -------
    matplotlib.figure.Figure
    """
    h, Nz, cutoffs = _unpack(result, h, orders, cutoffs)
    h_color = classify(h)

    slot_total = data_slots + gap_slots
    Ncoef = len(Nz)
    Ncut = len(cutoffs)
    total_slots = Ncoef * slot_total

    # Convert cutoff centres to edges for rectangle drawing
    cut = np.asarray(cutoffs, dtype=np.float64)
    if cut.size == 1:
        y_edges = np.array([cut[0] - 0.5, cut[0] + 0.5])
    else:
        mid = 0.5 * (cut[:-1] + cut[1:])
        y_edges = np.concatenate(
            [
                [cut[0] - (mid[0] - cut[0])],
                mid,
                [cut[-1] + (cut[-1] - mid[-1])],
            ]
        )

    _CLASS_COLORS: dict[float, str] = {
        -1.0: COLOR_PERIODIC,
        0.0: COLOR_CHAOTIC,
        2.0: COLOR_UNBOUNDED,
    }

    verts: list[list[tuple[float, float]]] = []
    colors: list[str] = []

    for i in range(Ncoef):
        x0 = i * slot_total - 0.5
        x1 = x0 + data_slots

        col = h_color[i, :]
        diffs = np.diff(col)
        change_idx = np.flatnonzero(diffs != 0)
        starts = np.concatenate([[0], change_idx + 1])
        ends = np.concatenate([change_idx, [Ncut - 1]])
        run_values = col[starts]

        for s, e, code in zip(starts, ends, run_values, strict=True):
            try:
                fc = _CLASS_COLORS[code]
            except KeyError:
                continue
            verts.append(
                [
                    (x0, y_edges[s]),
                    (x1, y_edges[s]),
                    (x1, y_edges[e + 1]),
                    (x0, y_edges[e + 1]),
                ]
            )
            colors.append(fc)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    coll = PolyCollection(
        verts,
        facecolors=colors,
        edgecolors="face",
        linewidths=0.0,
        snap=False,
        zorder=0,
    )
    ax.add_collection(coll)

    ax.set_xlabel(r"$N_z$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\omega_c/\pi$", fontsize=label_fontsize)
    ax.set_yticks(np.arange(0.0, 1.0 + 1e-9, ytick_step))
    ax.set_ylim(0.0, 1.0)
    for yt in _YTICKS:
        ax.axhline(y=yt, color="black", linewidth=0.4)

    ax.grid(False)
    for i in range(Ncoef + 1):
        ax.axvline(x=i * slot_total - 0.5, color="black", linewidth=0.6)

    tick_positions, tick_labels = _pick_xticks(Nz, slot_total, data_slots, xtick_step)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_xlim(-0.5, total_slots - 0.5)

    if legend:
        ax.legend(
            handles=_build_legend_handles(lang),
            fontsize=legend_fontsize,
            loc="upper right",
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
            handlelength=1.2,
            handleheight=0.8,
            borderpad=0.4,
        )

    fig.tight_layout()

    if panel_label is not None:
        _draw_panel_label(fig, ax, panel_label, panel_label_fontsize, panel_label_loc)

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
    data_slots: int = 3,
    gap_slots: int = 1,
) -> Figure:
    """Heatmap of Lyapunov iterations actually used at each grid point.

    A "difficulty map" of the parameter space: low values mean the
    spectrum estimate converged quickly (strongly chaotic or strongly
    stable points), high values mean the running estimate stayed within
    the convergence tolerance only after many iterations (typically
    fronteira points where \\|λ_max\\| ≈ 0). Diverged grid points are shown
    in the same light grey used for unbounded orbits in the
    classification figures, so the two layers can be visually overlaid.

    Uses the same interleaved-column layout as
    :func:`plot_classification_interleaved`.

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
    data_slots, gap_slots
        Interleaved-column parameters.

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

    # Render diverged cells (NaN) in grey; gap columns get a sentinel
    # below *vmin* so they render in white via ``set_under``.
    sentinel = vmin - 1.0
    n_expanded = _interleaved_expand(n_iters, data_slots, gap_slots, fill_value=sentinel)
    x_exp = np.arange(n_expanded.shape[0])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(COLOR_UNBOUNDED)
    cmap_obj.set_under("white")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_exp,
        cutoffs,
        n_expanded.T,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
        rasterized=True,
        antialiased=False,
    )
    cbar = fig.colorbar(pcm, ax=ax, label="Lyapunov iterations used")
    cbar.ax.tick_params(labelsize=12)

    _setup_interleaved_axes(ax, Nz, cutoffs, data_slots, gap_slots)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. Chaotic-region union & density maps — all windows × filters overlaid
# ═══════════════════════════════════════════════════════════════════════════


def _discover_sweeps(sweep_dir: Path) -> tuple[list[SweepResult], NDArray, NDArray]:
    """Load all sweeps under *sweep_dir* and unify to the finest grid.

    Returns (sweeps, Nz, cutoffs) where every sweep has been resampled
    (nearest-neighbour for orders) to match the largest grid found.
    """
    raw: list[SweepResult] = []
    for npz_path in sorted(sweep_dir.rglob("variables_lyapunov.npz")):
        try:
            raw.append(load_sweep(npz_path))
        except Exception:
            continue
    if not raw:
        raise FileNotFoundError(
            f"No sweeps found under {sweep_dir}. "
            f"Expected subdirectories like '<Window> (<ft>)/variables_lyapunov.npz'."
        )

    # Determine finest grid (largest number of orders)
    ref = max(raw, key=lambda sr: len(sr.orders))
    Nz_fine = np.asarray(ref.orders) - 1
    cutoffs = np.asarray(ref.cutoffs)

    unified: list[SweepResult] = []
    for sr in raw:
        if sr.orders.shape == ref.orders.shape and np.array_equal(sr.orders, ref.orders):
            unified.append(sr)
        else:
            # Resample to fine grid via nearest-neighbour in order space
            Nz_coarse = np.asarray(sr.orders) - 1
            h_coarse = np.asarray(sr.h, dtype=np.float64)
            h_fine = np.full((len(Nz_fine), len(cutoffs)), np.nan, dtype=np.float64)
            for j, nz in enumerate(Nz_fine):
                idx = np.argmin(np.abs(Nz_coarse - nz))
                h_fine[j, :] = h_coarse[idx, :]
            unified.append(
                SweepResult(
                    h=h_fine,
                    h_std=np.full_like(h_fine, np.nan),
                    orders=ref.orders.copy(),
                    cutoffs=cutoffs.copy(),
                    window=sr.window,
                    filter_type=sr.filter_type,
                    metadata=sr.metadata,
                )
            )
    return unified, Nz_fine, cutoffs


def _interleaved_expand(
    data_2d: NDArray,
    data_slots: int = 3,
    gap_slots: int = 1,
    fill_value: float = np.nan,
) -> NDArray:
    """Expand (Ncoef, Ncut) into (total_slots, Ncut) with gap columns.

    Gap slots are filled with *fill_value* (default ``NaN``).  Callers
    that need a visually distinguishable gap colour (e.g. a sentinel
    below the colour-bar minimum) can pass a custom fill.
    """
    Ncoef = data_2d.shape[0]
    Ncut = data_2d.shape[1]
    slot_total = data_slots + gap_slots
    total_slots = Ncoef * slot_total
    expanded = np.full((total_slots, Ncut), fill_value)
    for i in range(Ncoef):
        start = i * slot_total
        for s in range(data_slots):
            expanded[start + s, :] = data_2d[i, :]
    return expanded


def _setup_interleaved_axes(
    ax: plt.Axes,
    Nz: NDArray,
    cutoffs: NDArray,
    data_slots: int = 3,
    gap_slots: int = 1,
    label_fontsize: float = 16.0,
    tick_fontsize: float = 12.0,
    ytick_step: float = 0.1,
    xtick_step: int = 5,
) -> None:
    """Apply tick marks, vlines, and hlines for an interleaved heatmap."""
    Ncoef = len(Nz)
    slot_total = data_slots + gap_slots
    total_slots = Ncoef * slot_total

    ax.set_xlabel(r"$N_z$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\omega_c/\pi$", fontsize=label_fontsize)
    ax.set_yticks(np.arange(0.0, 1.0 + 1e-9, ytick_step))
    ax.set_ylim(0.0, 1.0)
    for yt in _YTICKS:
        ax.axhline(y=yt, color="black", linewidth=0.4)

    ax.grid(False)
    for i in range(Ncoef + 1):
        ax.axvline(x=i * slot_total - 0.5, color="black", linewidth=0.6)

    tick_positions, tick_labels = _pick_xticks(Nz, slot_total, data_slots, xtick_step)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_xlim(-0.5, total_slots - 0.5)


def _save_svg(fig: Figure, stem: Path) -> None:
    """Save *fig* as SVG from *stem*."""
    _save(fig, stem.with_suffix(".svg"))


def plot_chaotic_map(
    sweep_dir: str | Path = "data/sweeps",
    *,
    save_path: str | Path | None = None,
    color: str = COLOR_CHAOTIC,
    data_slots: int = 3,
    gap_slots: int = 1,
    lang: str = "pt",
    figsize: tuple[float, float] = (12.0, 5.0),
    label_fontsize: float = 16.0,
    tick_fontsize: float = 12.0,
    title_fontsize: float = 16.0,
    show_title: bool = True,
    ytick_step: float = 0.1,
    xtick_step: int = 5,
) -> Figure:
    """Binary union of chaotic regions across all window × filter sweeps.

    At each grid point, if **any** sweep has λ_max > 0 the cell is
    coloured; otherwise it is white.  The result is a single-colour
    silhouette of the full chaotic envelope.  The interleaved-column
    layout matches :func:`plot_classification_interleaved`.

    Parameters
    ----------
    sweep_dir
        Root directory with ``<Window> (<ft>)/variables_lyapunov.npz``
        subdirectories.
    save_path
        If provided, the figure is saved as SVG via ``_save_svg()``.
    color
        Matplotlib colour for chaotic cells (default: project red).
    data_slots, gap_slots
        Interleaved-column parameters.
    lang
        Language for labels (``"pt"`` or ``"en"``).

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    plot_chaotic_density
        How many configurations agree on chaos at each grid point.
    plot_classification_interleaved
        Discrete-classification counterpart for a single sweep.
    """
    sweep_dir = Path(sweep_dir)
    loaded, Nz, cutoffs = _discover_sweeps(sweep_dir)

    # Binary union: 1 where any sweep is chaotic, NaN elsewhere
    binary = np.zeros_like(loaded[0].h, dtype=np.float64)
    for sr in loaded:
        h_arr = np.asarray(sr.h, dtype=np.float64)
        binary = np.where(h_arr > 0.0, 1.0, binary)
    binary = np.where(binary > 0.0, binary, np.nan)

    h_expanded = _interleaved_expand(binary, data_slots, gap_slots)
    x_exp = np.arange(h_expanded.shape[0])

    cmap_obj = mcolors.ListedColormap([color])
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.pcolormesh(
        x_exp,
        cutoffs,
        h_expanded.T,
        cmap=cmap_obj,
        shading="nearest",
        rasterized=True,
        antialiased=False,
    )

    _setup_interleaved_axes(
        ax,
        Nz,
        cutoffs,
        data_slots,
        gap_slots,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        ytick_step=ytick_step,
        xtick_step=xtick_step,
    )
    if show_title:
        ax.set_title(t("sweep.chaotic_map.title", lang=lang), fontsize=title_fontsize, pad=14)

    fig.tight_layout()
    if save_path is not None:
        _save_svg(fig, Path(save_path))
    return fig


def plot_chaotic_density(
    sweep_dir: str | Path = "data/sweeps",
    *,
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    data_slots: int = 3,
    gap_slots: int = 1,
    lang: str = "pt",
    figsize: tuple[float, float] = (12.0, 5.0),
    label_fontsize: float = 16.0,
    tick_fontsize: float = 12.0,
    title_fontsize: float = 16.0,
    cbar_fontsize: float = 13.0,
    show_title: bool = True,
    marker_size: float = 10.0,
    ytick_step: float = 0.1,
    xtick_step: int = 5,
) -> Figure:
    """Density map: how many window × filter configurations are chaotic.

    At each grid point counts the number of sweeps (across all
    windows, filter types, and Kaiser β values) for which
    λ_max > 0.  Darker colours mean more configurations agree on
    chaos; white means none do.  The interleaved-column layout matches
    :func:`plot_classification_interleaved`.

    Parameters
    ----------
    sweep_dir
        Root directory with ``<Window> (<ft>)/variables_lyapunov.npz``
        subdirectories.
    save_path
        If provided, the figure is saved as SVG via ``_save_svg()``.
    cmap
        Sequential matplotlib colormap (default ``"viridis"``).
    data_slots, gap_slots
        Interleaved-column parameters.
    lang
        Language for labels (``"pt"`` or ``"en"``).

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    plot_chaotic_map
        Binary union map (chaotic or not).
    """
    sweep_dir = Path(sweep_dir)
    loaded, Nz, cutoffs = _discover_sweeps(sweep_dir)

    # Count how many sweeps are chaotic at each grid point
    count = np.zeros_like(loaded[0].h, dtype=np.float64)
    for sr in loaded:
        h_arr = np.asarray(sr.h, dtype=np.float64)
        count += (h_arr > 0.0).astype(np.float64)

    # Where count == 0 → white (NaN so set_bad kicks in)
    density = np.where(count > 0, count, np.nan)
    max_density = float(np.nanmax(density))
    idx_flat = np.nanargmax(density)
    idx_order, idx_cutoff = np.unravel_index(idx_flat, density.shape)

    h_expanded = _interleaved_expand(density, data_slots, gap_slots)
    x_exp = np.arange(h_expanded.shape[0])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_exp,
        cutoffs,
        h_expanded.T,
        cmap=cmap_obj,
        vmin=0.0,
        vmax=max_density,
        shading="nearest",
        rasterized=True,
        antialiased=False,
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(t("sweep.chaotic_density.cbar", lang=lang), fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=cbar_fontsize - 1.0)

    _setup_interleaved_axes(
        ax,
        Nz,
        cutoffs,
        data_slots,
        gap_slots,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        ytick_step=ytick_step,
        xtick_step=xtick_step,
    )
    if show_title:
        ax.set_title(t("sweep.chaotic_density.title", lang=lang), fontsize=title_fontsize, pad=14)

    # Mark point of maximum density
    slot_total = data_slots + gap_slots
    x_max = idx_order * slot_total + (data_slots - 1) / 2.0
    y_max = cutoffs[idx_cutoff]
    ax.plot(
        x_max,
        y_max,
        marker="o",
        markersize=marker_size,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=max(1.0, marker_size / 5.0),
        linestyle="none",
        zorder=10,
    )

    fig.tight_layout()
    if save_path is not None:
        _save_svg(fig, Path(save_path))
    return fig


CHAOTIC_MAP_FILENAME: str = "fig_chaotic_map"
CHAOTIC_DENSITY_FILENAME: str = "fig_chaotic_density"


def plot_chaotic_all(
    sweep_dir: str | Path = "data/sweeps",
    *,
    out_dir: str | Path = "figures/sweeps",
    close_figures: bool = True,
    lang: str = "pt",
) -> list[Path]:
    """Generate both cross-sweep chaotic figures and save to *out_dir*.

    Produces :data:`CHAOTIC_MAP_FILENAME` and
    :data:`CHAOTIC_DENSITY_FILENAME` as SVG.
    Returns the list of written paths.

    Parameters
    ----------
    sweep_dir
        Root directory with ``<Window> (<ft>)/variables_lyapunov.npz``
        subdirectories.
    out_dir
        Output directory (created if missing).
    close_figures
        If True, close each figure after saving.
    lang
        Language for labels (``"pt"`` or ``"en"``).

    Returns
    -------
    list[Path]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for stem in (CHAOTIC_MAP_FILENAME, CHAOTIC_DENSITY_FILENAME):
        paths.append(out_dir / f"{stem}.svg")

    fig = plot_chaotic_map(sweep_dir, save_path=out_dir / CHAOTIC_MAP_FILENAME, lang=lang)
    if close_figures:
        plt.close(fig)

    fig = plot_chaotic_density(sweep_dir, save_path=out_dir / CHAOTIC_DENSITY_FILENAME, lang=lang)
    if close_figures:
        plt.close(fig)

    return paths


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
    fmt: str = "svg",
    close_figures: bool = True,
    lang: str = "pt",
) -> list[Path]:
    """Generate the standard figures for a sweep and save them to
    ``out_dir/<fig>.{fmt}``. Returns the list of written paths.

    Always produces the two classification figures listed in
    :data:`FIGURE_FILENAMES`. Additionally produces
    ``fig3_difficulty_map.{fmt}`` (see :data:`DIFFICULTY_FIGURE_FILENAME`)
    when ``result`` was generated with ``adaptive=True``: the figure is
    silently skipped for non-adaptive sweeps because it would be
    monochromatic and therefore uninformative.

    The output directory is created if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    path = out_dir / f"fig1_heatmap_continuous.{fmt}"
    fig = plot_heatmap_continuous(result, save_path=path)
    if close_figures:
        plt.close(fig)
    paths.append(path)

    path = out_dir / f"fig2_classification_interleaved.{fmt}"
    fig = plot_classification_interleaved(result, save_path=path, lang=lang)
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

"""2-D classification maps (heatmap, interleaved classification, difficulty)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chaotic_pfc._i18n import t
from chaotic_pfc.analysis.sweep import SweepResult
from chaotic_pfc.analysis.sweep_plotting._common import (
    _YTICKS,
    COLOR_CHAOTIC,
    COLOR_PERIODIC,
    COLOR_UNBOUNDED,
    _build_legend_handles,
    _draw_panel_label,
    _interleaved_expand,
    _pick_xticks,
    _save,
    _setup_interleaved_axes,
    _unpack,
    classify,
)

# Pull in global RC params (STIX fonts, vector SVG, etc.)


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


def plot_heatmap_continuous(
    result: SweepResult | None = None,
    *,
    h: NDArray | None = None,
    num_taps_grid: NDArray | None = None,
    cutoff_grid: NDArray | None = None,
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
        :class:`~chaotic_pfc.analysis.sweep.SweepResult` to plot.  Mutually exclusive with the
        ``(h, num_taps_grid, cutoff_grid)`` triple.
    h
        Lyapunov grid, shape ``(len(num_taps_grid), len(cutoff_grid))``.
    num_taps_grid
        Filter num_taps_grid (x-axis), shape ``(len(num_taps_grid),)``.
    cutoff_grid
        Cutoff frequencies (y-axis), shape ``(len(cutoff_grid),)``.
    save_path
        If provided, the figure is saved to this path via ``_save()``.
    data_slots, gap_slots
        Interleaved-column parameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    h, Nz, cutoff_grid = _unpack(result, h, num_taps_grid, cutoff_grid)

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
        x_exp, cutoff_grid, h_expanded.T, shading="nearest", rasterized=True, antialiased=False
    )
    fig.colorbar(pcm, ax=ax, label=r"$\lambda_{\max}$")
    _setup_interleaved_axes(ax, Nz, cutoff_grid, data_slots, gap_slots)
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
    num_taps_grid: NDArray | None = None,
    cutoff_grid: NDArray | None = None,
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
    """Publication-style layout with gaps between adjacent num_taps_grid.

    Each order occupies ``data_slots`` columns of coloured data followed
    by ``gap_slots`` blank columns, producing the striped appearance
    used in Baptista et al.

    Parameters
    ----------
    result
        :class:`~chaotic_pfc.analysis.sweep.SweepResult` to plot.  Mutually exclusive with the
        ``(h, num_taps_grid, cutoff_grid)`` triple.
    h
        Lyapunov grid, shape ``(len(num_taps_grid), len(cutoff_grid))``.
    num_taps_grid
        Filter num_taps_grid (x-axis), shape ``(len(num_taps_grid),)``.
    cutoff_grid
        Cutoff frequencies (y-axis), shape ``(len(cutoff_grid),)``.
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
        ``num_taps_grid`` are labelled; if fewer than three survive the
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
    h, Nz, cutoff_grid = _unpack(result, h, num_taps_grid, cutoff_grid)
    h_color = classify(h)

    slot_total = data_slots + gap_slots
    Ncoef = len(Nz)
    Ncut = len(cutoff_grid)
    total_slots = Ncoef * slot_total

    # Convert cutoff centres to edges for rectangle drawing
    cut = np.asarray(cutoff_grid, dtype=np.float64)
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
    lang: str = "pt",
) -> Figure:
    r"""Heatmap of Lyapunov iterations actually used at each grid point.

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
        A :class:`~chaotic_pfc.analysis.sweep.SweepResult` produced with
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
    lang
        Language for the colour-bar label (``"pt"`` or ``"en"``).

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

    Nz = np.asarray(result.num_taps_grid) - 1
    cutoff_grid = np.asarray(result.cutoff_grid)
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
        cutoff_grid,
        n_expanded.T,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
        rasterized=True,
        antialiased=False,
    )
    cbar = fig.colorbar(pcm, ax=ax, label=t("sweep.difficulty.cbar", lang=lang))
    cbar.ax.tick_params(labelsize=12)

    _setup_interleaved_axes(ax, Nz, cutoff_grid, data_slots, gap_slots)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. Chaotic-region union & density maps — all windows × filters overlaid
# ═══════════════════════════════════════════════════════════════════════════

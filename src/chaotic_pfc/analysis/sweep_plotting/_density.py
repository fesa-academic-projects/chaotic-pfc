"""Chaotic-region union and density maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chaotic_pfc._i18n import t
from chaotic_pfc.analysis.sweep import SweepResult, load_sweep
from chaotic_pfc.analysis.sweep_plotting._common import (
    COLOR_CHAOTIC,
    _discover_sweeps,
    _interleaved_expand,
    _save_svg,
    _setup_interleaved_axes,
)

# Pull in global RC params (STIX fonts, vector SVG, etc.)


def _binary_union(loaded: Any) -> NDArray[Any]:
    """Binary union over sweeps: 1 where any sweep is chaotic, NaN elsewhere."""
    binary = np.zeros_like(loaded[0].h, dtype=np.float64)
    for sr in loaded:
        binary = np.where(np.asarray(sr.h, dtype=np.float64) > 0.0, 1.0, binary)
    return np.where(binary > 0.0, binary, np.nan)


def _chaotic_count(loaded: Any) -> NDArray[Any]:
    """Count the sweeps that are chaotic at each grid point."""
    count = np.zeros_like(loaded[0].h, dtype=np.float64)
    for sr in loaded:
        count += (np.asarray(sr.h, dtype=np.float64) > 0.0).astype(np.float64)
    return count


def _marker_of_max(
    density: NDArray[Any], cutoff_grid: NDArray[Any], data_slots: int, gap_slots: int
) -> tuple[float, float]:
    """Interleaved ``(x, y)`` of the grid point with the highest density."""
    idx_order, idx_cutoff = np.unravel_index(np.nanargmax(density), density.shape)
    return idx_order * (data_slots + gap_slots) + (data_slots - 1) / 2.0, float(
        cutoff_grid[idx_cutoff]
    )


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
    loaded, Nz, cutoff_grid = _discover_sweeps(sweep_dir)

    # Binary union: 1 where any sweep is chaotic, NaN elsewhere
    binary = _binary_union(loaded)

    h_expanded = _interleaved_expand(binary, data_slots, gap_slots)
    x_exp = np.arange(h_expanded.shape[0])

    cmap_obj = mcolors.ListedColormap([color])
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.pcolormesh(
        x_exp,
        cutoff_grid,
        h_expanded.T,
        cmap=cmap_obj,
        shading="nearest",
        rasterized=True,
        antialiased=False,
    )

    _setup_interleaved_axes(
        ax,
        Nz,
        cutoff_grid,
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
    loaded, Nz, cutoff_grid = _discover_sweeps(sweep_dir)

    # Count how many sweeps are chaotic at each grid point
    count = _chaotic_count(loaded)

    # Where count == 0 → white (NaN so set_bad kicks in)
    density = np.where(count > 0, count, np.nan)
    max_density = float(np.nanmax(density))

    h_expanded = _interleaved_expand(density, data_slots, gap_slots)
    x_exp = np.arange(h_expanded.shape[0])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_exp,
        cutoff_grid,
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
        cutoff_grid,
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
    x_max, y_max = _marker_of_max(density, cutoff_grid, data_slots, gap_slots)
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


def chaotic_density_norm_table(
    sweep_dir: str | Path = "data/sweeps",
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Percentage of the sweeps that contain each grid point and find it chaotic.

    ``Nz = order - 1``. A sweep contains ``Nz`` when its num_taps_grid include
    ``Nz + 1``; highpass and bandstop only have odd num_taps_grid, so even ``Nz``
    is contained in all sweeps and odd ``Nz`` only in lowpass and bandpass.
    Membership is counted order by order, without resampling.

    Parameters
    ----------
    sweep_dir
        Root directory with the sweep ``.npz`` checkpoints.

    Returns
    -------
    tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]
        ``(Nz, cutoff_grid, percent, n_containing)``.  ``percent[k, j]`` is the
        percentage of the sweeps containing ``Nz[k]`` in which the point
        ``(Nz[k], cutoff_grid[j])`` has ``lambda_max > 0``; it is NaN where no
        sweep contains ``Nz[k]``.
    """
    sweep_dir = Path(sweep_dir)
    raw: list[SweepResult] = []
    for npz_path in sorted(sweep_dir.rglob("variables_lyapunov.npz")):
        try:
            raw.append(load_sweep(npz_path))
        except Exception:
            continue
    if not raw:
        raise FileNotFoundError(f"No sweeps found under {sweep_dir}.")

    cutoff_grid = np.asarray(raw[0].cutoff_grid, dtype=np.float64)
    for sr in raw:
        if not np.array_equal(np.asarray(sr.cutoff_grid, dtype=np.float64), cutoff_grid):
            raise ValueError("sweeps do not share the same cutoff grid")

    Nz: NDArray[np.int64] = np.array(
        sorted({int(o) - 1 for sr in raw for o in np.asarray(sr.num_taps_grid)})
    )
    index = {int(nz): k for k, nz in enumerate(Nz)}
    n_chaotic = np.zeros((len(Nz), len(cutoff_grid)))
    n_containing = np.zeros(len(Nz))
    for sr in raw:
        h = np.asarray(sr.h, dtype=np.float64)
        for i, order in enumerate(np.asarray(sr.num_taps_grid)):
            k = index[int(order) - 1]
            n_containing[k] += 1.0
            n_chaotic[k] += h[i] > 0.0

    percent = np.full_like(n_chaotic, np.nan)
    has = n_containing > 0
    percent[has] = 100.0 * n_chaotic[has] / n_containing[has, None]
    return Nz, cutoff_grid, percent, n_containing


def plot_chaotic_density_norm(
    sweep_dir: str | Path = "data/sweeps",
    *,
    cmap: str = "viridis",
    data_slots: int = 3,
    gap_slots: int = 1,
    lang: str = "en",
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
    """Normalised density map: chaotic percentage over the containing sweeps.

    Same layout as :func:`plot_chaotic_density`, with the colour giving the
    percentage of the sweeps that contain each grid point and find it
    chaotic (see :func:`chaotic_density_norm_table`). The circle marks the
    global maximum.

    Parameters
    ----------
    sweep_dir
        Root directory with the sweep ``.npz`` checkpoints.
    cmap
        Matplotlib colormap name.
    data_slots, gap_slots
        Interleaving layout of the x-axis (data and gap slots per column).
    lang
        Figure language (``"en"`` or ``"pt"``).
    figsize
        Figure size in inches.
    label_fontsize, tick_fontsize, title_fontsize, cbar_fontsize
        Font sizes for the labels, ticks, title and colour bar.
    show_title
        Draw the figure title when ``True``.
    marker_size
        Size of the maximum marker.
    ytick_step, xtick_step
        Axis tick spacing.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    Nz, cutoff_grid, percent, _ = chaotic_density_norm_table(sweep_dir)

    h_expanded = _interleaved_expand(percent, data_slots, gap_slots)
    x_exp = np.arange(h_expanded.shape[0])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_exp,
        cutoff_grid,
        h_expanded.T,
        cmap=cmap_obj,
        vmin=0.0,
        vmax=100.0,
        shading="nearest",
        rasterized=True,
        antialiased=False,
    )
    cbar = fig.colorbar(pcm, ax=ax)
    label = r"Chaotic configurations (\%)" if lang == "en" else r"Configurações caóticas (\%)"
    cbar.set_label(label, fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=cbar_fontsize - 1.0)

    _setup_interleaved_axes(
        ax,
        Nz,
        cutoff_grid,
        data_slots,
        gap_slots,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        ytick_step=ytick_step,
        xtick_step=xtick_step,
    )
    if show_title:
        ax.set_title(
            r"Chaotic configurations (\%)" if lang == "en" else r"Configurações caóticas (\%)",
            fontsize=title_fontsize,
            pad=14,
        )

    k_max, j_max = np.unravel_index(np.nanargmax(percent), percent.shape)
    slot_total = data_slots + gap_slots
    ax.plot(
        k_max * slot_total + (data_slots - 1) / 2.0,
        cutoff_grid[j_max],
        marker="o",
        markersize=marker_size,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=max(1.0, marker_size / 5.0),
        linestyle="none",
        zorder=10,
    )

    fig.tight_layout()
    return fig


CHAOTIC_MAP_FILENAME: str = "fig_chaotic_map"
CHAOTIC_DENSITY_FILENAME: str = "fig_chaotic_density"

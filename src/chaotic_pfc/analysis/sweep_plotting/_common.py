"""Shared palette, classification and helpers for the sweep figures."""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

# Pull in global RC params (STIX fonts, vector SVG, etc.)
from chaotic_pfc._figure_io import save_figure as _figures_save
from chaotic_pfc._i18n import t
from chaotic_pfc.analysis.sweep import SweepResult, load_sweep
from chaotic_pfc.errors import InvalidParameterError

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


def _build_legend_handles(lang: str = "pt") -> list[Any]:
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


class OrbitClass(IntEnum):
    """Discrete classification of an orbit on a grid point.

    The integer values are the ones the discrete colour map expects and are
    unchanged from the bare ``-1``/``0``/``2`` literals they replace, so the
    classification figures are byte-identical.
    """

    PERIODIC = -1
    CHAOTIC = 0
    DIVERGENT = 2


def classify(h: NDArray[np.float64]) -> NDArray[np.int64]:
    """Map raw λ_max values to discrete classes.

    Parameters
    ----------
    h
        Raw λ_max array (NaN where the orbit diverged).

    Returns
    -------
    NDArray[Any]
        Array of the same shape as ``h`` where each entry is an
        :class:`OrbitClass` value:

        * :attr:`OrbitClass.PERIODIC` — periodic orbit (λ_max ≤ 0)
        * :attr:`OrbitClass.CHAOTIC` — chaotic orbit (λ_max > 0)
        * :attr:`OrbitClass.DIVERGENT` — unbounded / divergent (NaN in ``h``)

    The integer values match the :class:`matplotlib.colors.BoundaryNorm` bins
    used below.

    Examples
    --------
    >>> classify(np.array([np.nan, -0.5, 0.5])).tolist()
    [2, -1, 0]

    Raises
    ------
    InvalidParameterError
        If ``h`` is not a non-empty real numeric array or contains ``+/-inf``.
    """
    if isinstance(h, (str, bytes, dict, set)):
        raise InvalidParameterError(f"h must be a numeric array, got {type(h).__name__}")
    try:
        raw = np.asarray(h)
    except Exception as exc:
        raise InvalidParameterError(f"h could not be converted to an array: {exc}") from exc
    if raw.dtype == bool or np.issubdtype(raw.dtype, np.complexfloating) or raw.dtype == object:
        raise InvalidParameterError(f"h must be a real numeric array, got dtype {raw.dtype}")
    try:
        h_arr = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(f"h must contain real numbers: {exc}") from exc
    if h_arr.ndim == 0 or h_arr.size == 0:
        raise InvalidParameterError(f"h must be a non-empty array, got shape {h_arr.shape}")
    if np.any(np.isinf(h_arr)):
        raise InvalidParameterError("h must not contain +/-inf (use NaN for divergent cells)")
    out = np.empty_like(h_arr, dtype=np.int64)
    mask_diverged = np.isnan(h_arr)
    mask_periodic = (~mask_diverged) & (h_arr <= 0.0)
    mask_chaotic = (~mask_diverged) & (h_arr > 0.0)

    out[mask_diverged] = int(OrbitClass.DIVERGENT)
    out[mask_periodic] = int(OrbitClass.PERIODIC)
    out[mask_chaotic] = int(OrbitClass.CHAOTIC)
    return out


def _pick_xticks(
    Nz: NDArray[Any],
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
    ax: Axes,
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
    # get_renderer lives on the concrete canvas (e.g. FigureCanvasAgg),
    # not on FigureCanvasBase, so the matplotlib stubs miss it.
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # only on the Agg canvas
    bbox = ax.xaxis.get_tightbbox(renderer)
    # get_tightbbox returns None only for an empty axis; the draw pass
    # above guarantees the x-axis has a measurable tight bbox.
    y_fig = fig.transFigure.inverted().transform((0.0, bbox.y0))[1]  # type: ignore[union-attr]  # non-None after the draw pass
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


def _discover_sweeps(sweep_dir: Path) -> tuple[list[SweepResult], NDArray[Any], NDArray[Any]]:
    """Load all sweeps under *sweep_dir* and unify to the finest grid.

    Returns (sweeps, Nz, cutoff_grid) where every sweep has been resampled
    (nearest-neighbour for num_taps_grid) to match the largest grid found.
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

    # Determine finest grid (largest number of taps)
    ref = max(raw, key=lambda sr: len(sr.num_taps_grid))
    Nz_fine = np.asarray(ref.num_taps_grid) - 1
    cutoff_grid = np.asarray(ref.cutoff_grid)

    unified: list[SweepResult] = []
    for sr in raw:
        if sr.num_taps_grid.shape == ref.num_taps_grid.shape and np.array_equal(
            sr.num_taps_grid, ref.num_taps_grid
        ):
            unified.append(sr)
        else:
            # Resample to fine grid via nearest-neighbour in order space
            Nz_coarse = np.asarray(sr.num_taps_grid) - 1
            h_coarse = np.asarray(sr.h, dtype=np.float64)
            h_fine = np.full((len(Nz_fine), len(cutoff_grid)), np.nan, dtype=np.float64)
            for j, nz in enumerate(Nz_fine):
                idx = np.argmin(np.abs(Nz_coarse - nz))
                h_fine[j, :] = h_coarse[idx, :]
            unified.append(
                SweepResult(
                    h=h_fine,
                    h_std=np.full_like(h_fine, np.nan),
                    num_taps_grid=ref.num_taps_grid.copy(),
                    cutoff_grid=cutoff_grid.copy(),
                    window=sr.window,
                    filter_type=sr.filter_type,
                    metadata=sr.metadata,
                )
            )
    return unified, Nz_fine, cutoff_grid


def _interleaved_expand(
    data_2d: NDArray[Any],
    data_slots: int = 3,
    gap_slots: int = 1,
    fill_value: float = np.nan,
) -> NDArray[Any]:
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
    ax: Axes,
    Nz: NDArray[Any],
    cutoff_grid: NDArray[Any],
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


def _unpack(
    result: SweepResult | None,
    h: NDArray[Any] | None,
    num_taps_grid: NDArray[Any] | None,
    cutoff_grid: NDArray[Any] | None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Resolve the two accepted calling conventions into (h, Nz, cutoff_grid)."""
    if result is not None:
        if not isinstance(result, SweepResult):
            raise InvalidParameterError(
                f"result must be a SweepResult, got {type(result).__name__}"
            )
        h = result.h
        num_taps_grid = result.num_taps_grid
        cutoff_grid = result.cutoff_grid
    if h is None or num_taps_grid is None or cutoff_grid is None:
        raise InvalidParameterError(
            "Provide either a SweepResult or the (h, num_taps_grid, cutoff_grid) triple."
        )
    Nz = np.asarray(num_taps_grid) - 1
    return np.asarray(h), Nz, np.asarray(cutoff_grid)

"""Multi-panel figure bundles: per-sweep and chaotic-region."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from chaotic_pfc.analysis.sweep import SweepResult
from chaotic_pfc.analysis.sweep_plotting._classification import (
    plot_classification_interleaved,
    plot_difficulty_map,
    plot_heatmap_continuous,
)
from chaotic_pfc.analysis.sweep_plotting._density import (
    CHAOTIC_DENSITY_FILENAME,
    CHAOTIC_MAP_FILENAME,
    plot_chaotic_density,
    plot_chaotic_map,
)

# Pull in global RC params (STIX fonts, vector SVG, etc.)


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
    paths: list[Path] = [
        out_dir / f"{stem}.svg" for stem in (CHAOTIC_MAP_FILENAME, CHAOTIC_DENSITY_FILENAME)
    ]

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
    """Generate the standard figures for a sweep, save them, and return their paths.

    Always produces the two classification figures listed in
    :data:`FIGURE_FILENAMES`. Additionally produces
    ``fig3_difficulty_map.{fmt}`` (see :data:`DIFFICULTY_FIGURE_FILENAME`)
    when ``result`` was generated with ``adaptive=True``: the figure is
    silently skipped for non-adaptive sweeps because it would be
    monochromatic and therefore uninformative.

    Parameters
    ----------
    result
        Sweep result to plot.
    out_dir
        Directory for the output files (created if missing).
    fmt
        Output format (``"svg"``, ``"png"`` or ``"pdf"``).
    close_figures
        Close each figure after saving it.
    lang
        Figure language (``"en"`` or ``"pt"``).

    Returns
    -------
    list[Path]
        Paths of the files written.

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
        fig = plot_difficulty_map(result, save_path=path, lang=lang)
        if close_figures:
            plt.close(fig)
        paths.append(path)

    return paths

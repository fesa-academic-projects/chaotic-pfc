"""Figures for the Lyapunov classification maps of :mod:`chaotic_pfc.analysis.sweep`.

Three plot types are provided:

1. :func:`plot_heatmap_continuous` — raw λ_max heatmap.
2. :func:`plot_classification_interleaved` — publication-style discrete
   classification (periodic / chaotic / unbounded) with gaps between
   num_taps_grid.
3. :func:`plot_difficulty_map` — heatmap of the number of Lyapunov
   iterations actually used at each grid point. Only meaningful when
   the sweep was run with ``adaptive=True``; shows where the spectrum
   converges quickly (light) vs. where it needs the full budget
   (dark) — a "difficulty map" of the parameter space.

All three accept a :class:`~chaotic_pfc.analysis.sweep.SweepResult` (or its
individual arrays for the first two) and optionally a ``save_path``.
They return the :class:`matplotlib.figure.Figure` so callers can
compose or display them. The module also re-uses the RC params set by
:func:`~chaotic_pfc.plotting.figures.setup_rc`, so sweep figures look consistent
with the rest of the pipeline.

.. rubric:: Functions

.. autosummary::

   chaotic_density_norm_table
   classify
   make_classification_legend
   plot_all
   plot_chaotic_all
   plot_chaotic_density
   plot_chaotic_density_norm
   plot_chaotic_map
   plot_classification_interleaved
   plot_difficulty_map
   plot_heatmap_continuous
"""

# Implementation is split, by responsibility, into the private submodules
# ``_common`` (palette, classification, shared helpers), ``_classification``
# (2-D maps), ``_density`` (chaotic-region maps) and ``_bundle`` (multi-panel
# bundles). This package re-exports the public surface unchanged; ``__all__``
# is exactly the previously public set, in the original source order.

from __future__ import annotations

__all__ = [  # noqa: RUF022
    "OrbitClass",
    "classify",
    "make_classification_legend",
    "plot_heatmap_continuous",
    "plot_classification_interleaved",
    "plot_difficulty_map",
    "plot_chaotic_map",
    "plot_chaotic_density",
    "chaotic_density_norm_table",
    "plot_chaotic_density_norm",
    "plot_chaotic_all",
    "plot_all",
]

from chaotic_pfc.analysis.sweep_plotting._bundle import (
    DIFFICULTY_FIGURE_FILENAME as DIFFICULTY_FIGURE_FILENAME,
)
from chaotic_pfc.analysis.sweep_plotting._bundle import (
    FIGURE_FILENAMES as FIGURE_FILENAMES,
)
from chaotic_pfc.analysis.sweep_plotting._bundle import (
    plot_all as plot_all,
)
from chaotic_pfc.analysis.sweep_plotting._bundle import (
    plot_chaotic_all as plot_chaotic_all,
)
from chaotic_pfc.analysis.sweep_plotting._classification import (
    make_classification_legend as make_classification_legend,
)
from chaotic_pfc.analysis.sweep_plotting._classification import (
    plot_classification_interleaved as plot_classification_interleaved,
)
from chaotic_pfc.analysis.sweep_plotting._classification import (
    plot_difficulty_map as plot_difficulty_map,
)
from chaotic_pfc.analysis.sweep_plotting._classification import (
    plot_heatmap_continuous as plot_heatmap_continuous,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    _YTICKS as _YTICKS,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    COLOR_CHAOTIC as COLOR_CHAOTIC,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    COLOR_GAP as COLOR_GAP,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    COLOR_PERIODIC as COLOR_PERIODIC,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    COLOR_UNBOUNDED as COLOR_UNBOUNDED,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    OrbitClass as OrbitClass,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    _interleaved_expand as _interleaved_expand,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    _unpack as _unpack,
)
from chaotic_pfc.analysis.sweep_plotting._common import (
    classify as classify,
)
from chaotic_pfc.analysis.sweep_plotting._density import (
    CHAOTIC_DENSITY_FILENAME as CHAOTIC_DENSITY_FILENAME,
)
from chaotic_pfc.analysis.sweep_plotting._density import (
    CHAOTIC_MAP_FILENAME as CHAOTIC_MAP_FILENAME,
)
from chaotic_pfc.analysis.sweep_plotting._density import (
    chaotic_density_norm_table as chaotic_density_norm_table,
)
from chaotic_pfc.analysis.sweep_plotting._density import (
    plot_chaotic_density as plot_chaotic_density,
)
from chaotic_pfc.analysis.sweep_plotting._density import (
    plot_chaotic_density_norm as plot_chaotic_density_norm,
)
from chaotic_pfc.analysis.sweep_plotting._density import (
    plot_chaotic_map as plot_chaotic_map,
)

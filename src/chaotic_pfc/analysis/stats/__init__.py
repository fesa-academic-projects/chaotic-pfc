"""
Statistical analysis of Lyapunov sweep results.

Provides quick programmatic access to the full sweep dataset:
summary tables, regime classification, parameter ranking, and
comparisons across windows, filter types, and Kaiser beta sweeps.

Return types
------------
All public functions that return ``dict`` have corresponding
:class:`~typing.TypedDict` definitions in this module so that IDE
autocompletion and mypy can statically verify the result shape:

* :class:`SummaryRow` — one row of :func:`summary_table`
* :class:`FilterTypeAggregate` — output of :func:`compare_filter_types`
* :class:`OptimalParams` — one entry of :func:`optimal_parameters`
* :class:`LmaxDistribution` — output of :func:`lmax_distribution`
* :class:`CorrelationMatrix` — output of :func:`correlation_matrix`
* :class:`BootstrapConfidence` — one entry of :func:`bootstrap_confidence`

.. rubric:: Functions

.. autosummary::

   area_summary
   best_chaos_preserving
   beta_curve
   beta_summary
   bootstrap_confidence
   chaos_margin
   compare_filter_types
   consolidate_kaiser
   correlation_matrix
   export_summary_json
   kaiser_beta_optimal
   lmax_distribution
   lmax_statistics
   load_all_sweeps
   optimal_parameters
   rank_configurations
   summary_table
   sweet_spot_per_filter
   top_k_per_filter
   transition_boundary

.. rubric:: Classes

.. autosummary::

   AreaSummary
   BootstrapConfidence
   ConfigRank
   CorrelationMatrix
   FilterTypeAggregate
   KaiserBetaOptimal
   LmaxDistribution
   LmaxStats
   OptimalParams
   SummaryRow
   SweetSpot
"""

# Implementation is split, by responsibility, into the private submodules
# ``_common``, ``_types``, ``_summary``, ``_ranking`` and ``_statistics``.
# This package re-exports the public surface unchanged; ``__all__`` is exactly
# the previously public set, in the original source order.

from __future__ import annotations

__all__ = [  # noqa: RUF022
    "SummaryRow",
    "FilterTypeAggregate",
    "OptimalParams",
    "LmaxDistribution",
    "CorrelationMatrix",
    "BootstrapConfidence",
    "AreaSummary",
    "LmaxStats",
    "ConfigRank",
    "SweetSpot",
    "KaiserBetaOptimal",
    "summary_table",
    "area_summary",
    "lmax_statistics",
    "load_all_sweeps",
    "rank_configurations",
    "top_k_per_filter",
    "consolidate_kaiser",
    "kaiser_beta_optimal",
    "sweet_spot_per_filter",
    "best_chaos_preserving",
    "compare_filter_types",
    "optimal_parameters",
    "export_summary_json",
    "beta_summary",
    "beta_curve",
    "lmax_distribution",
    "transition_boundary",
    "chaos_margin",
    "correlation_matrix",
    "bootstrap_confidence",
]

from chaotic_pfc.analysis.stats._ranking import (
    best_chaos_preserving as best_chaos_preserving,
)
from chaotic_pfc.analysis.stats._ranking import (
    compare_filter_types as compare_filter_types,
)
from chaotic_pfc.analysis.stats._ranking import (
    consolidate_kaiser as consolidate_kaiser,
)
from chaotic_pfc.analysis.stats._ranking import (
    kaiser_beta_optimal as kaiser_beta_optimal,
)
from chaotic_pfc.analysis.stats._ranking import (
    optimal_parameters as optimal_parameters,
)
from chaotic_pfc.analysis.stats._ranking import (
    rank_configurations as rank_configurations,
)
from chaotic_pfc.analysis.stats._ranking import (
    sweet_spot_per_filter as sweet_spot_per_filter,
)
from chaotic_pfc.analysis.stats._ranking import (
    top_k_per_filter as top_k_per_filter,
)
from chaotic_pfc.analysis.stats._statistics import (
    beta_curve as beta_curve,
)
from chaotic_pfc.analysis.stats._statistics import (
    beta_summary as beta_summary,
)
from chaotic_pfc.analysis.stats._statistics import (
    bootstrap_confidence as bootstrap_confidence,
)
from chaotic_pfc.analysis.stats._statistics import (
    chaos_margin as chaos_margin,
)
from chaotic_pfc.analysis.stats._statistics import (
    correlation_matrix as correlation_matrix,
)
from chaotic_pfc.analysis.stats._statistics import (
    lmax_distribution as lmax_distribution,
)
from chaotic_pfc.analysis.stats._statistics import (
    transition_boundary as transition_boundary,
)
from chaotic_pfc.analysis.stats._summary import (
    area_summary as area_summary,
)
from chaotic_pfc.analysis.stats._summary import (
    export_summary_json as export_summary_json,
)
from chaotic_pfc.analysis.stats._summary import (
    lmax_statistics as lmax_statistics,
)
from chaotic_pfc.analysis.stats._summary import (
    load_all_sweeps as load_all_sweeps,
)
from chaotic_pfc.analysis.stats._summary import (
    summary_table as summary_table,
)
from chaotic_pfc.analysis.stats._types import (
    AreaSummary as AreaSummary,
)
from chaotic_pfc.analysis.stats._types import (
    BootstrapConfidence as BootstrapConfidence,
)
from chaotic_pfc.analysis.stats._types import (
    ConfigRank as ConfigRank,
)
from chaotic_pfc.analysis.stats._types import (
    CorrelationMatrix as CorrelationMatrix,
)
from chaotic_pfc.analysis.stats._types import (
    FilterTypeAggregate as FilterTypeAggregate,
)
from chaotic_pfc.analysis.stats._types import (
    KaiserBetaOptimal as KaiserBetaOptimal,
)
from chaotic_pfc.analysis.stats._types import (
    LmaxDistribution as LmaxDistribution,
)
from chaotic_pfc.analysis.stats._types import (
    LmaxStats as LmaxStats,
)
from chaotic_pfc.analysis.stats._types import (
    OptimalParams as OptimalParams,
)
from chaotic_pfc.analysis.stats._types import (
    SummaryRow as SummaryRow,
)
from chaotic_pfc.analysis.stats._types import (
    SweetSpot as SweetSpot,
)

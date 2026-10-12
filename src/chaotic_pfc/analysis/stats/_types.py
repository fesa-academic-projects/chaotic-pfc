"""TypedDict return types for :mod:`chaotic_pfc.analysis.stats`."""

from __future__ import annotations

from typing import TypedDict


class SummaryRow(TypedDict):
    """One row of :func:`summary_table`."""

    window: str
    filter_type: str
    n_orders: int
    n_cutoffs: int
    pct_chaotic: float
    pct_periodic: float
    pct_divergent: float
    mean_lmax: float
    max_lmax: float
    beta: float | None


class FilterTypeAggregate(TypedDict, total=False):
    """Aggregate statistics for a single filter type."""

    n_sweeps: int
    mean_pct_chaotic: float
    mean_pct_periodic: float
    mean_pct_divergent: float
    mean_lmax: float


class OptimalParams(TypedDict):
    """One optimal (order, cutoff) pair from :func:`optimal_parameters`."""

    window: str
    filter_type: str
    order: int
    cutoff: float
    lmax: float


class LmaxDistribution(TypedDict):
    """Distribution statistics for lambda_max per filter type.

    Returned by :func:`lmax_distribution`.
    """

    hist: list[float]
    edges: list[float]
    mean: float
    std: float
    skewness: float
    n: int


class CorrelationMatrix(TypedDict):
    """Spearman correlation results from :func:`correlation_matrix`."""

    order_vs_lmax: float
    cutoff_vs_lmax: float
    n: int


class BootstrapConfidence(TypedDict, total=False):
    """Bootstrap 95% CI entry for one filter type."""

    mean: float
    ci_low: float
    ci_high: float
    n: int


class AreaSummary(TypedDict):
    """Chaotic-area breakdown for a single :class:`~.sweep.SweepResult`.

    Returned by :func:`area_summary`.
    """

    n_chaotic: int
    n_periodic: int
    n_divergent: int
    n_total: int
    pct_chaotic: float
    pct_chaotic_finite: float


class LmaxStats(TypedDict):
    r"""Descriptive statistics of :math:`\\lambda_{\\max}` for a subset of points.

    Returned by :func:`lmax_statistics`.
    """

    mean: float
    median: float
    std: float
    min: float
    max: float
    ci_95_low: float
    ci_95_high: float
    n_used: int


class ConfigRank(TypedDict):
    """One entry of :func:`rank_configurations`.

    Ranks a (filter_type, window) combination by chaotic-area coverage.
    Kaiser-beta sweeps include the beta value and a synthetic window key.
    """

    rank: int
    filter_type: str
    window: str
    n_chaotic: int
    pct_chaotic: float
    pct_chaotic_finite: float
    lmax_mean: float
    lmax_max: float
    lmax_std: float
    lmax_ci_95_low: float
    lmax_ci_95_high: float
    beta: float | None


class SweetSpot(TypedDict):
    r"""Single grid point with the highest :math:`\\lambda_{\\max}` for one filter type.

    Returned per filter type by :func:`sweet_spot_per_filter`.
    """

    filter_type: str
    window: str
    n_z: int
    omega_c: float
    lmax: float
    lmax_ci_95_low: float | None
    lmax_ci_95_high: float | None


class KaiserBetaOptimal(TypedDict):
    """Best Kaiser β per filter type (by chaotic area).

    Returned by :func:`kaiser_beta_optimal`.
    """

    filter_type: str
    beta: float
    n_chaotic: int
    pct_chaotic_finite: float
    lmax_mean: float
    lmax_max: float

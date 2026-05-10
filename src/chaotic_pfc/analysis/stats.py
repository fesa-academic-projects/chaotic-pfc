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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .sweep import FILTER_TYPES, SweepResult, load_sweep


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
    """Distribution statistics for lambda_max per filter type."""

    n: int
    mean: float
    std: float
    skewness: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float


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


def _discover_all(data_dir: str | Path = "data/sweeps") -> list[SweepResult]:
    """Load every ``variables_lyapunov.npz`` under *data_dir*."""
    results: list[SweepResult] = []
    for path in sorted(Path(data_dir).rglob("variables_lyapunov.npz")):
        results.append(load_sweep(path))
    return results


def _safe_stats(h: NDArray) -> tuple[float, float]:
    """Mean and max of *h* ignoring NaN and Inf."""
    finite = h[np.isfinite(h)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.max(finite))


def summary_table(
    data_dir: str | Path = "data/sweeps",
) -> list[dict]:
    """Return one row per sweep with key statistics.

    Each row is a :class:`SummaryRow` with: ``window``, ``filter_type``,
    ``n_orders``, ``n_cutoffs``, ``pct_chaotic``, ``pct_periodic``,
    ``pct_divergent``, ``mean_lmax``, ``max_lmax``, ``beta`` (Kaiser only).

    This is the foundation of all downstream analyses — every other
    public function in this module either calls or derives from this table.
    """
    rows: list[dict] = []
    for result in _discover_all(data_dir):
        h = result.h
        chaotic = np.sum(h > 0)
        periodic = np.sum((~np.isnan(h)) & (h <= 0))
        divergent = np.sum(np.isnan(h))
        total = h.size

        mean_lmax, max_lmax = _safe_stats(h)

        row: dict = {
            "window": result.window,
            "filter_type": result.filter_type,
            "n_orders": len(result.orders),
            "n_cutoffs": len(result.cutoffs),
            "pct_chaotic": round(100 * chaotic / total, 1),
            "pct_periodic": round(100 * periodic / total, 1),
            "pct_divergent": round(100 * divergent / total, 1),
            "mean_lmax": round(mean_lmax, 4),
            "max_lmax": round(max_lmax, 4),
            "beta": result.metadata.get("kaiser_beta"),
        }
        rows.append(row)
    return rows


def best_chaos_preserving(
    data_dir: str | Path = "data/sweeps",
    top_n: int = 5,
) -> list[dict]:
    """Rank sweeps by percentage of chaotic grid points (descending).

    Returns the *top_n* entries with the most chaotic coverage.
    """
    rows = summary_table(data_dir)
    rows.sort(key=lambda r: r["pct_chaotic"], reverse=True)
    return rows[:top_n]


def compare_filter_types(
    data_dir: str | Path = "data/sweeps",
) -> dict[str, dict]:
    """Aggregate statistics per filter type across all windows.

    Returns a ``dict`` keyed by filter type (``"lowpass"``, ...) with
    each value being a :class:`FilterTypeAggregate` containing mean
    percentages and lambda_max across all windows that use that filter type.
    """
    rows = summary_table(data_dir)
    agg: dict[str, list[dict]] = {ft: [] for ft in FILTER_TYPES}
    for row in rows:
        agg[row["filter_type"]].append(row)

    out: dict[str, dict] = {}
    for ft, entries in agg.items():
        if not entries:
            out[ft] = {}
            continue
        out[ft] = {
            "n_sweeps": len(entries),
            "mean_pct_chaotic": round(float(np.mean([e["pct_chaotic"] for e in entries])), 1),
            "mean_pct_periodic": round(float(np.mean([e["pct_periodic"] for e in entries])), 1),
            "mean_pct_divergent": round(float(np.mean([e["pct_divergent"] for e in entries])), 1),
            "mean_lmax": round(float(np.mean([e["mean_lmax"] for e in entries])), 4),
        }
    return out


def optimal_parameters(
    data_dir: str | Path = "data/sweeps",
    window: str | None = None,
    filter_type: str | None = None,
    top_n: int = 3,
) -> list[dict]:
    """Find the (order, cutoff) pairs with the highest λ_max.

    Filters results by *window* and *filter_type* if given.
    """
    best: list[dict] = []
    for result in _discover_all(data_dir):
        if window is not None and result.window != window:
            continue
        if filter_type is not None and result.filter_type != filter_type:
            continue

        h, orders, cutoffs = result.h, result.orders, result.cutoffs
        for i in range(len(orders)):
            for j in range(len(cutoffs)):
                val = h[i, j]
                if np.isnan(val) or np.isinf(val):
                    continue
                best.append(
                    {
                        "window": result.window,
                        "filter_type": result.filter_type,
                        "order": int(orders[i]),
                        "cutoff": round(float(cutoffs[j]), 4),
                        "lmax": round(float(val), 6),
                    }
                )

    best.sort(key=lambda x: x["lmax"], reverse=True)
    return best[:top_n]


def export_summary_json(
    data_dir: str | Path = "data/sweeps",
    output: str | Path = "data/analysis_summary.json",
) -> Path:
    """Write the full summary table to a JSON file."""
    output = Path(output)
    rows = summary_table(data_dir)
    output.write_text(json.dumps(rows, indent=2, default=str))
    return output


# ── Kaiser β-sweep analysis ──────────────────────────────────────────────────


def beta_summary(
    data_dir: str | Path = "data/sweeps/kaiser",
) -> dict[str, dict[float, dict]]:
    """Aggregate per-β statistics for each filter type.

    Returns a nested dict: ``{filter_type: {beta: {pct_chaotic, mean_lmax, ...}}}``
    """
    out: dict[str, dict[float, dict]] = {}
    for ft in FILTER_TYPES:
        ft_dir = Path(data_dir) / ft
        if not ft_dir.is_dir():
            continue
        out[ft] = {}
        for path in sorted(ft_dir.rglob("variables_lyapunov.npz")):
            result = load_sweep(path)
            beta = float(result.metadata.get("kaiser_beta", float("nan")))
            if np.isnan(beta):
                continue
            h = result.h
            chaotic = np.sum((~np.isnan(h)) & (h > 0))
            total = h.size
            mn, mx = _safe_stats(h)
            out[ft][beta] = {
                "pct_chaotic": round(100 * chaotic / total, 1),
                "mean_lmax": round(mn, 4),
                "max_lmax": round(mx, 4),
            }
    return out


def beta_curve(
    data_dir: str | Path = "data/sweeps/kaiser",
    filter_type: str = "lowpass",
) -> tuple[NDArray, NDArray]:
    """Return (betas, pct_chaotic) arrays for a single filter type.

    Useful for plotting the β-dependence of chaotic coverage.
    """
    summary = beta_summary(data_dir)
    if filter_type not in summary:
        return np.array([]), np.array([])
    betas = np.array(sorted(summary[filter_type]))
    pct = np.array([summary[filter_type][b]["pct_chaotic"] for b in betas])
    return betas, pct


# ── Advanced statistical analyses ────────────────────────────────────────────


def lmax_distribution(
    data_dir: str | Path = "data/sweeps",
    bins: int = 50,
) -> dict[str, dict]:
    """Histogram of λ_max values per filter type.

    Returns
    -------
    dict
        ``{filter_type: {"hist": counts, "edges": bin_edges, "mean": float, "std": float, "skewness": float}}``
    """
    from scipy.stats import skew

    all_vals: dict[str, list[float]] = {ft: [] for ft in FILTER_TYPES}
    for result in _discover_all(data_dir):
        vals = result.h.ravel()
        vals = vals[np.isfinite(vals)]
        all_vals[result.filter_type].extend(vals.tolist())

    out: dict[str, dict] = {}
    for ft, val_list in all_vals.items():
        arr = np.array(val_list)
        if len(arr) == 0:
            out[ft] = {}
            continue
        hist, edges = np.histogram(arr, bins=bins)
        out[ft] = {
            "hist": hist.tolist(),
            "edges": edges.tolist(),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "skewness": round(float(skew(arr)), 4),
            "n": len(arr),
        }
    return out


def transition_boundary(
    data_dir: str | Path = "data/sweeps",
    window: str | None = None,
    filter_type: str = "lowpass",
) -> tuple[NDArray, NDArray]:
    """Find the cutoff where each order transitions from non-chaotic to chaotic.

    For each order, returns the *first* cutoff (lowest frequency) where
    λ_max > 0, or NaN if no chaotic point exists.

    Returns
    -------
    orders : ndarray
        Filter orders.
    cutoffs : ndarray
        First chaotic cutoff per order (NaN if never chaotic).
    """
    best: dict[int, tuple[float, float]] = {}
    for result in _discover_all(data_dir):
        if window is not None and result.window != window:
            continue
        if result.filter_type != filter_type:
            continue

        h, orders, cutoffs = result.h, result.orders, result.cutoffs
        for i in range(len(orders)):
            for j in range(len(cutoffs)):
                val = h[i, j]
                if not np.isfinite(val):
                    continue
                if val > 0:
                    order = int(orders[i])
                    wc = float(cutoffs[j])
                    if order not in best or wc < best[order][0]:
                        best[order] = (wc, float(val))

    if not best:
        return np.array([]), np.array([])
    orders_sorted = np.array(sorted(best))
    cutoffs_sorted = np.array([best[o][0] for o in orders_sorted])
    return orders_sorted, cutoffs_sorted


def chaos_margin(
    data_dir: str | Path = "data/sweeps",
    window: str | None = None,
    filter_type: str = "lowpass",
) -> tuple[NDArray, NDArray]:
    """For each order, compute the width of the chaotic region in cutoff space.

    Returns
    -------
    orders : ndarray
    widths : ndarray
        Fraction of cutoffs where λ_max > 0 per order.
    """
    accum: dict[int, list[float]] = {}
    for result in _discover_all(data_dir):
        if window is not None and result.window != window:
            continue
        if result.filter_type != filter_type:
            continue

        h, orders = result.h, result.orders
        for i in range(len(orders)):
            order = int(orders[i])
            row = h[i, :]
            chaotic_count = int(np.sum(np.isfinite(row) & (row > 0)))
            total = int(np.sum(np.isfinite(row)))
            if total > 0:
                accum.setdefault(order, []).append(chaotic_count / total)

    if not accum:
        return np.array([]), np.array([])
    orders_sorted = np.array(sorted(accum))
    widths = np.array([float(np.mean(accum[o])) for o in orders_sorted])
    return orders_sorted, widths


def correlation_matrix(
    data_dir: str | Path = "data/sweeps",
) -> dict:
    """Spearman correlation between (order, cutoff, λ_max) across all sweeps.

    Returns
    -------
    dict
        ``{"order_vs_lmax": rho, "cutoff_vs_lmax": rho, "n": int}``
    """
    from scipy.stats import spearmanr

    orders_all: list[float] = []
    cutoffs_all: list[float] = []
    lmax_all: list[float] = []

    for result in _discover_all(data_dir):
        h, orders, cutoffs = result.h, result.orders, result.cutoffs
        for i in range(len(orders)):
            for j in range(len(cutoffs)):
                val = h[i, j]
                if not np.isfinite(val):
                    continue
                orders_all.append(float(orders[i]))
                cutoffs_all.append(float(cutoffs[j]))
                lmax_all.append(float(val))

    if len(lmax_all) < 3:
        return {"order_vs_lmax": 0.0, "cutoff_vs_lmax": 0.0, "n": len(lmax_all)}

    rho_o, _ = spearmanr(orders_all, lmax_all)
    rho_c, _ = spearmanr(cutoffs_all, lmax_all)
    return {
        "order_vs_lmax": round(float(rho_o), 4),
        "cutoff_vs_lmax": round(float(rho_c), 4),
        "n": len(lmax_all),
    }


def bootstrap_confidence(
    data_dir: str | Path = "data/sweeps",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict]:
    """Bootstrap 95% CI for mean λ_max per filter type.

    Returns
    -------
    dict
        ``{filter_type: {"mean": float, "ci_low": float, "ci_high": float}}``
    """
    rng = np.random.default_rng(seed)
    all_vals: dict[str, list[float]] = {ft: [] for ft in FILTER_TYPES}
    for result in _discover_all(data_dir):
        vals = result.h.ravel()
        vals = vals[np.isfinite(vals)]
        all_vals[result.filter_type].extend(vals.tolist())

    out: dict[str, dict] = {}
    for ft, val_list in all_vals.items():
        arr = np.array(val_list)
        if len(arr) == 0:
            out[ft] = {}
            continue
        means = np.array(
            [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
        )
        out[ft] = {
            "mean": round(float(np.mean(arr)), 4),
            "ci_low": round(float(np.percentile(means, 2.5)), 4),
            "ci_high": round(float(np.percentile(means, 97.5)), 4),
            "n": len(arr),
        }
    return out

"""Distribution, boundary, correlation and bootstrap statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc.analysis.stats._common import (
    _discover_all,
    _safe_stats,
)
from chaotic_pfc.analysis.stats._types import (
    BetaPoint,
    BootstrapConfidence,
    CorrelationMatrix,
    LmaxDistribution,
)
from chaotic_pfc.analysis.sweep import FILTER_TYPES, load_sweep


def beta_summary(
    data_dir: str | Path = "data/sweeps/kaiser",
) -> dict[str, dict[float, BetaPoint]]:
    """Aggregate per-β statistics for each filter type.

    Returns a nested dict: ``{filter_type: {beta: BetaPoint}}``

    Parameters
    ----------
    data_dir
        Root directory of the Kaiser sweep checkpoints.

    Returns
    -------
    dict[str, dict[float, BetaPoint]]
        Aggregated per-β statistics for each filter type.
    """
    out: dict[str, dict[float, BetaPoint]] = {}
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
    Returns empty arrays when *filter_type* is absent from the data.

    Parameters
    ----------
    data_dir
        Root directory of the Kaiser sweep checkpoints.
    filter_type
        Filter type to extract (default ``"lowpass"``).

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(betas, pct_chaotic)``; empty arrays when absent.
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
) -> dict[str, LmaxDistribution]:
    """Histogram of λ_max values per filter type.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.
    bins
        Number of histogram bins (default 50).

    Returns
    -------
    dict
        ``{filter_type: {"hist": counts, "edges": bin_edges, "mean": float,
        "std": float, "skewness": float, "n": int}}``
    """
    from scipy.stats import skew

    all_vals: dict[str, list[float]] = {ft: [] for ft in FILTER_TYPES}
    for result in _discover_all(data_dir):
        vals = result.h.ravel()
        vals = vals[np.isfinite(vals)]
        all_vals[result.filter_type].extend(vals.tolist())

    out: dict[str, LmaxDistribution] = {}
    for ft, val_list in all_vals.items():
        arr = np.array(val_list)
        if len(arr) == 0:
            out[ft] = {"hist": [], "edges": [], "mean": 0.0, "std": 0.0, "skewness": 0.0, "n": 0}
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

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.
    window
        Restrict to one window key; ``None`` uses every window.
    filter_type
        Filter type to analyse (default ``"lowpass"``).

    Returns
    -------
    num_taps_grid : ndarray
        Filter num_taps_grid.
    cutoff_grid : ndarray
        First chaotic cutoff per order (NaN if never chaotic).
    """
    best: dict[int, tuple[float, float]] = {}
    for result in _discover_all(data_dir):
        if window is not None and result.window != window:
            continue
        if result.filter_type != filter_type:
            continue

        h, num_taps_grid, cutoff_grid = result.h, result.num_taps_grid, result.cutoff_grid
        for i in range(len(num_taps_grid)):
            for j in range(len(cutoff_grid)):
                val = h[i, j]
                if not np.isfinite(val):
                    continue
                if val > 0:
                    order = int(num_taps_grid[i])
                    wc = float(cutoff_grid[j])
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

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.
    window
        Restrict to one window key; ``None`` uses every window.
    filter_type
        Filter type to analyse (default ``"lowpass"``).

    Returns
    -------
    num_taps_grid : ndarray
    widths : ndarray
        Fraction of cutoff_grid where λ_max > 0 per order.
    """
    accum: dict[int, list[float]] = {}
    for result in _discover_all(data_dir):
        if window is not None and result.window != window:
            continue
        if result.filter_type != filter_type:
            continue

        h, num_taps_grid = result.h, result.num_taps_grid
        for i in range(len(num_taps_grid)):
            order = int(num_taps_grid[i])
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
) -> CorrelationMatrix:
    """Spearman correlation between (order, cutoff, λ_max) across all sweeps.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.

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
        h, num_taps_grid, cutoff_grid = result.h, result.num_taps_grid, result.cutoff_grid
        for i in range(len(num_taps_grid)):
            for j in range(len(cutoff_grid)):
                val = h[i, j]
                if not np.isfinite(val):
                    continue
                orders_all.append(float(num_taps_grid[i]))
                cutoffs_all.append(float(cutoff_grid[j]))
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
) -> dict[str, BootstrapConfidence]:
    """Bootstrap 95% CI for mean λ_max per filter type.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.
    n_bootstrap
        Number of bootstrap resamples (default 1000).
    seed
        RNG seed for the resampling (default 42).

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
    return out  # type: ignore[return-value]

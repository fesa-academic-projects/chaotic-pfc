"""Rankings, comparisons and optimal-parameter selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc._validate import require_positive_int, require_seed
from chaotic_pfc.analysis.stats._common import _discover_all, round_half_up
from chaotic_pfc.analysis.stats._summary import (
    area_summary,
    lmax_statistics,
    summary_table,
)
from chaotic_pfc.analysis.stats._types import (
    ConfigRank,
    FilterTypeAggregate,
    KaiserBetaOptimal,
    OptimalParams,
    SummaryRow,
    SweetSpot,
)
from chaotic_pfc.analysis.sweep import FILTER_TYPES, WINDOWS, SweepResult
from chaotic_pfc.errors import InvalidParameterError


def rank_configurations(
    sweep_results: dict[tuple[str, str], SweepResult],
    bootstrap_seed: int = 42,
) -> list[ConfigRank]:
    """Rank every (filter_type, window) combination by chaotic grid-point count.

    Parameters
    ----------
    sweep_results
        Mapping from ``(filter_type, window_key)`` to a loaded sweep.
        Typically obtained via :func:`load_all_sweeps`.
    bootstrap_seed
        RNG seed forwarded to :func:`lmax_statistics` (default 42).

    Returns
    -------
    list of ConfigRank
        Descending by ``n_chaotic``. Rank is 1-indexed.

    Raises
    ------
    InvalidParameterError
        If *sweep_results* is not a dict or *bootstrap_seed* is invalid.
    """
    if not isinstance(sweep_results, dict):
        raise InvalidParameterError(
            f"sweep_results must be a dict, got {type(sweep_results).__name__}"
        )
    require_seed("bootstrap_seed", bootstrap_seed)
    entries: list[ConfigRank] = []
    for (ft, win), result in sweep_results.items():
        area = area_summary(result)
        lmax = lmax_statistics(result, region="chaotic", seed=bootstrap_seed)
        beta_raw = result.metadata.get("kaiser_beta")
        beta = round(float(beta_raw), 2) if beta_raw is not None else None
        entries.append(
            {
                "rank": 0,
                "filter_type": ft,
                "window": win,
                "n_chaotic": area["n_chaotic"],
                "pct_chaotic": area["pct_chaotic"],
                "pct_chaotic_finite": area["pct_chaotic_finite"],
                "lmax_mean": lmax["mean"],
                "lmax_max": lmax["max"],
                "lmax_std": lmax["std"],
                "lmax_ci_95_low": lmax["ci_95_low"],
                "lmax_ci_95_high": lmax["ci_95_high"],
                "beta": beta,
            }
        )

    entries.sort(key=lambda e: e["n_chaotic"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return entries


def top_k_per_filter(
    sweep_results: dict[tuple[str, str], SweepResult],
    k: int = 3,
    bootstrap_seed: int = 42,
) -> dict[str, list[ConfigRank]]:
    """Return the top-*k* configurations for each filter type.

    Calls :func:`rank_configurations` internally, then groups the
    global ranking by ``filter_type``, keeping at most *k* entries
    per group.

    Parameters
    ----------
    sweep_results
        Mapping from ``(filter_type, window_key)`` to loaded sweeps.
    k
        Maximum number of entries per filter type (default 3).
    bootstrap_seed
        Forwarded to :func:`rank_configurations`.

    Returns
    -------
    dict of str → list of ConfigRank
        Keys are filter types; values are lists of at most *k*
        :class:`ConfigRank` entries, ordered by descending
        ``n_chaotic``.

    Raises
    ------
    InvalidParameterError
        If *k* is not a positive integer, *sweep_results* is not a dict,
        or *bootstrap_seed* is invalid.
    """
    k = require_positive_int("k", k, min_value=1)
    ranking = rank_configurations(sweep_results, bootstrap_seed=bootstrap_seed)
    out: dict[str, list[ConfigRank]] = {}
    for entry in ranking:
        ft = entry["filter_type"]
        out.setdefault(ft, []).append(entry)
    for ft in list(out):
        entries = out[ft][:k]
        for i, e in enumerate(entries):
            e["rank"] = i + 1
        out[ft] = entries
    return out


def consolidate_kaiser(
    sweep_results: dict[tuple[str, str], SweepResult],
) -> dict[tuple[str, str], SweepResult]:
    """Keep only the best Kaiser β per filter type, by chaotic point count.

    Non-Kaiser windows are passed through unchanged. This produces a
    balanced dataset where each window family appears once per filter
    type, making window-to-window comparisons fair.

    Parameters
    ----------
    sweep_results
        Mapping from :func:`load_all_sweeps`.

    Returns
    -------
    dict
        Same shape as *sweep_results*, but with at most one Kaiser
        entry per ``filter_type`` (the β with the most chaotic points).

    Raises
    ------
    InvalidParameterError
        If *sweep_results* is not a dict.
    """
    if not isinstance(sweep_results, dict):
        raise InvalidParameterError(
            f"sweep_results must be a dict, got {type(sweep_results).__name__}"
        )
    consolidated: dict[tuple[str, str], SweepResult] = {}
    best_kaiser: dict[str, tuple[str, int]] = {}

    for (ft, win), result in sweep_results.items():
        if result.window == "kaiser":
            area = area_summary(result)
            n_chaotic = area["n_chaotic"]
            if ft not in best_kaiser or n_chaotic > best_kaiser[ft][1]:
                best_kaiser[ft] = (win, n_chaotic)
        else:
            consolidated[(ft, win)] = result

    for ft, (win, _) in best_kaiser.items():
        consolidated[(ft, win)] = sweep_results[(ft, win)]

    return consolidated


def kaiser_beta_optimal(
    sweep_results: dict[tuple[str, str], SweepResult],
    bootstrap_seed: int = 42,
) -> dict[str, KaiserBetaOptimal]:
    """Report the best Kaiser β per filter type with extended statistics.

    Parameters
    ----------
    sweep_results
        Mapping from :func:`load_all_sweeps`.
    bootstrap_seed
        Forwarded to :func:`lmax_statistics`.

    Returns
    -------
    dict of str → KaiserBetaOptimal

    Raises
    ------
    InvalidParameterError
        If *sweep_results* is not a dict or *bootstrap_seed* is invalid.
    """
    if not isinstance(sweep_results, dict):
        raise InvalidParameterError(
            f"sweep_results must be a dict, got {type(sweep_results).__name__}"
        )
    require_seed("bootstrap_seed", bootstrap_seed)
    out: dict[str, KaiserBetaOptimal] = {}
    for (ft, _win), result in sweep_results.items():
        if result.window != "kaiser":
            continue
        beta = float(result.metadata.get("kaiser_beta", 0.0))
        area = area_summary(result)
        lmax = lmax_statistics(result, region="chaotic", seed=bootstrap_seed)
        if ft not in out or area["n_chaotic"] > out[ft]["n_chaotic"]:
            out[ft] = {
                "filter_type": ft,
                "beta": beta,
                "n_chaotic": area["n_chaotic"],
                "pct_chaotic_finite": area["pct_chaotic_finite"],
                "lmax_mean": lmax["mean"],
                "lmax_max": lmax["max"],
            }
    return out


def sweet_spot_per_filter(
    sweep_results: dict[tuple[str, str], SweepResult],
) -> dict[str, SweetSpot]:
    r"""Identify the grid point with the highest :math:`\\lambda_{\\max}` per filter type.

    For each filter type, scans all available window sweeps and returns
    the exact ``(N_z, omega_c, window)`` where ``lambda_max`` is
    globally maximal.

    The 95% confidence interval is computed from the per-point standard
    deviation ``h_std`` and ``n_initial`` metadata via the
    t-distribution. If ``h_std`` is NaN or ``n_initial`` is unavailable,
    the CI fields are ``None``.

    Parameters
    ----------
    sweep_results
        Mapping from ``(filter_type, window_key)`` to loaded sweeps.

    Returns
    -------
    dict of str → SweetSpot
        One entry per filter type that has at least one finite point.

    Raises
    ------
    InvalidParameterError
        If *sweep_results* is not a dict.
    """
    from scipy.stats import t as t_dist

    if not isinstance(sweep_results, dict):
        raise InvalidParameterError(
            f"sweep_results must be a dict, got {type(sweep_results).__name__}"
        )
    best: dict[str, SweetSpot] = {}

    for (ft, win), result in sweep_results.items():
        h, h_std, num_taps_grid, cutoff_grid = (
            result.h,
            result.h_std,
            result.num_taps_grid,
            result.cutoff_grid,
        )
        n_initial = result.metadata.get("n_initial", None)
        for i in range(len(num_taps_grid)):
            for j in range(len(cutoff_grid)):
                val = h[i, j]
                if not np.isfinite(val):
                    continue
                if ft not in best or val > best[ft]["lmax"]:
                    std_val = h_std[i, j]
                    if (
                        n_initial is not None
                        and n_initial > 1
                        and np.isfinite(std_val)
                        and std_val > 0
                    ):
                        se = std_val / np.sqrt(n_initial)
                        t_crit = t_dist.ppf(0.975, n_initial - 1)
                        ci_low = round(float(val - t_crit * se), 6)
                        ci_high = round(float(val + t_crit * se), 6)
                    else:
                        ci_low = None
                        ci_high = None
                    best[ft] = {
                        "filter_type": ft,
                        "window": win,
                        "n_z": int(num_taps_grid[i]) - 1,
                        "omega_c": round(float(cutoff_grid[j]), 4),
                        "lmax": round(float(val), 6),
                        "lmax_ci_95_low": ci_low,
                        "lmax_ci_95_high": ci_high,
                    }
    return best


def best_chaos_preserving(
    data_dir: str | Path = "data/sweeps",
    top_n: int = 5,
) -> list[SummaryRow]:
    """Rank sweeps by percentage of chaotic grid points (descending).

    Returns the *top_n* entries with the most chaotic coverage.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.
    top_n
        Number of top entries to return (default 5).

    Returns
    -------
    list[SummaryRow]
        The ``top_n`` sweeps with the highest chaotic coverage.

    Raises
    ------
    InvalidParameterError
        If *top_n* is not a positive integer.
    """
    top_n = require_positive_int("top_n", top_n, min_value=1)
    rows = summary_table(data_dir)
    rows.sort(key=lambda r: r["pct_chaotic"], reverse=True)
    return rows[:top_n]


def compare_filter_types(
    data_dir: str | Path = "data/sweeps",
) -> dict[str, FilterTypeAggregate]:
    """Aggregate statistics per filter type across all windows.

    Returns a ``dict`` keyed by filter type (``"lowpass"``, ...) with
    each value being a :class:`FilterTypeAggregate` containing mean
    percentages and lambda_max across all windows that use that filter type.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.

    Returns
    -------
    dict[str, FilterTypeAggregate]
        Aggregated statistics keyed by filter type.
    """
    rows = summary_table(data_dir)
    agg: dict[str, list[SummaryRow]] = {ft: [] for ft in FILTER_TYPES}
    for row in rows:
        agg[row["filter_type"]].append(row)

    out: dict[str, FilterTypeAggregate] = {}
    for ft, entries in agg.items():
        if not entries:
            out[ft] = {}
            continue
        out[ft] = {
            "n_sweeps": len(entries),
            "mean_pct_chaotic": round_half_up(
                float(np.mean([e["pct_chaotic"] for e in entries])), 1
            ),
            "mean_pct_periodic": round_half_up(
                float(np.mean([e["pct_periodic"] for e in entries])), 1
            ),
            "mean_pct_divergent": round_half_up(
                float(np.mean([e["pct_divergent"] for e in entries])), 1
            ),
            "mean_lmax": round(float(np.mean([e["mean_lmax"] for e in entries])), 4),
        }
    return out


def optimal_parameters(
    data_dir: str | Path = "data/sweeps",
    window: str | None = None,
    filter_type: str | None = None,
    top_n: int = 3,
) -> list[OptimalParams]:
    """Find the (order, cutoff) pairs with the highest λ_max.

    Filters results by *window* and *filter_type* if given.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.
    window
        Restrict to one window key; ``None`` uses every window.
    filter_type
        Restrict to one filter type; ``None`` uses every type.
    top_n
        Number of (order, cutoff) pairs to return (default 3).

    Returns
    -------
    list[OptimalParams]
        The ``top_n`` points with the highest ``lambda_max``.

    Raises
    ------
    InvalidParameterError
        If *top_n* is not a positive integer or *window* / *filter_type*
        is unknown.
    """
    top_n = require_positive_int("top_n", top_n, min_value=1)
    if window is not None and window not in WINDOWS:
        raise InvalidParameterError(f"window must be one of {WINDOWS}, got {window!r}")
    if filter_type is not None and filter_type not in FILTER_TYPES:
        raise InvalidParameterError(
            f"filter_type must be one of {FILTER_TYPES}, got {filter_type!r}"
        )
    best: list[dict[str, Any]] = []
    for result in _discover_all(data_dir):
        if window is not None and result.window != window:
            continue
        if filter_type is not None and result.filter_type != filter_type:
            continue

        h, num_taps_grid, cutoff_grid = result.h, result.num_taps_grid, result.cutoff_grid
        for i in range(len(num_taps_grid)):
            for j in range(len(cutoff_grid)):
                val = h[i, j]
                if np.isnan(val) or np.isinf(val):
                    continue
                best.append(
                    {
                        "window": result.window,
                        "filter_type": result.filter_type,
                        "order": int(num_taps_grid[i]) - 1,
                        "cutoff": round(float(cutoff_grid[j]), 4),
                        "lmax": round(float(val), 6),
                    }
                )

    best.sort(key=lambda x: x["lmax"], reverse=True)
    return best[:top_n]  # type: ignore[return-value]  # list[ConfigRank]

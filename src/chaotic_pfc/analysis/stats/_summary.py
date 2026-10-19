"""Summary tables, area/exponent statistics and sweep loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc.analysis.stats._common import (
    _DEFAULT_N_BOOTSTRAP,
    _discover_all,
    _safe_stats,
    pct_round,
)
from chaotic_pfc.analysis.stats._types import AreaSummary, LmaxStats, SummaryRow
from chaotic_pfc.analysis.sweep import SweepResult, load_sweep


def summary_table(
    data_dir: str | Path = "data/sweeps",
) -> list[SummaryRow]:
    """Return one row per sweep with key statistics.

    Each row is a :class:`SummaryRow` with: ``window``, ``filter_type``,
    ``n_orders``, ``n_cutoffs``, ``pct_chaotic``, ``pct_periodic``,
    ``pct_divergent``, ``mean_lmax``, ``max_lmax``, ``beta`` (Kaiser only).

    This is the foundation of all downstream analyses — every other
    public function in this module either calls or derives from this table.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.

    Returns
    -------
    list[SummaryRow]
        One entry per sweep with the statistics listed above.
    """
    rows: list[dict[str, Any]] = []
    for result in _discover_all(data_dir):
        h = result.h
        chaotic = np.sum((~np.isnan(h)) & (h > 0))
        periodic = np.sum((~np.isnan(h)) & (h <= 0))
        divergent = np.sum(np.isnan(h))
        total = h.size

        mean_lmax, max_lmax = _safe_stats(h)

        row: dict[str, Any] = {
            "window": result.window,
            "filter_type": result.filter_type,
            "n_orders": len(result.num_taps_grid),
            "n_cutoffs": len(result.cutoff_grid),
            "pct_chaotic": pct_round(int(chaotic), total, 1),
            "pct_periodic": pct_round(int(periodic), total, 1),
            "pct_divergent": pct_round(int(divergent), total, 1),
            "mean_lmax": round(mean_lmax, 4),
            "max_lmax": round(max_lmax, 4),
            "beta": result.metadata.get("kaiser_beta"),
        }
        rows.append(row)
    return rows  # type: ignore[return-value]  # list[SummaryRow]


def area_summary(sweep_result: SweepResult) -> AreaSummary:
    r"""Compute chaotic-area coverage for a single sweep.

    Classifies each grid point (order, cutoff) as chaotic
    (:math:`\\lambda_{\\max} > 0`, finite), periodic
    (:math:`\\lambda_{\\max} \\le 0`, finite), or divergent
    (NaN or Inf). Returns absolute counts and two complementary
    percentages: one over the full grid, another excluding
    divergent points so the chaotic proportion is not
    artificially deflated by parameter regions where the
    FIR filter itself becomes ill-conditioned.

    Parameters
    ----------
    sweep_result : SweepResult
        A loaded sweep.

    Returns
    -------
    AreaSummary
        Counts and percentages for the three dynamical regimes.
    """
    h = sweep_result.h
    finite = np.isfinite(h)
    chaotic = np.sum(finite & (h > 0))
    periodic = np.sum(finite & (h <= 0))
    divergent = np.sum(~finite)
    total = h.size

    pct_chaotic = pct_round(int(chaotic), total, 1)
    n_finite = total - int(divergent)
    pct_chaotic_finite = pct_round(int(chaotic), n_finite, 1)

    return {
        "n_chaotic": int(chaotic),
        "n_periodic": int(periodic),
        "n_divergent": int(divergent),
        "n_total": int(total),
        "pct_chaotic": pct_chaotic,
        "pct_chaotic_finite": pct_chaotic_finite,
    }


def lmax_statistics(
    sweep_result: SweepResult,
    region: str = "chaotic",
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> LmaxStats:
    r"""Compute descriptive statistics of :math:`\\lambda_{\\max}` for a subset of grid points.

    Filters the sweep grid by dynamical regime then computes mean,
    median, standard deviation, min, max, and a bootstrap 95%
    confidence interval for the mean.

    Parameters
    ----------
    sweep_result : SweepResult
    region : {"chaotic", "all_finite", "periodic"}
        Subset of grid points to include:
        * ``"chaotic"`` — only points with :math:`\\lambda_{\\max} > 0` (finite)
        * ``"all_finite"`` — all finite points (excludes NaN/Inf)
        * ``"periodic"`` — only points with :math:`\\lambda_{\\max} \\le 0` (finite)
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    seed : int
        RNG seed for reproducibility (default 42).

    Returns
    -------
    LmaxStats
        Descriptive statistics including bootstrap 95% CI.
        When fewer than 3 valid points exist, the CI fields are NaN.

    Notes
    -----
    The bootstrap treats the grid points as **independent** samples. Neighbouring
    cells are correlated, so the interval underestimates the true uncertainty and
    is best read as a description of the dispersion over the region, not as a
    hypothesis-test interval (see the article's statistical section).
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    h = sweep_result.h.ravel()
    finite = np.isfinite(h)

    if region == "chaotic":
        mask = finite & (h > 0)
    elif region == "periodic":
        mask = finite & (h <= 0)
    elif region == "all_finite":
        mask = finite
    else:
        raise ValueError(
            f"Unknown region {region!r}; expected 'chaotic', 'all_finite', or 'periodic'"
        )

    vals = h[mask]
    n_used = len(vals)

    if n_used < 3:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci_95_low": float("nan"),
            "ci_95_high": float("nan"),
            "n_used": n_used,
        }

    ci_result = scipy_bootstrap(
        (vals,),
        np.mean,
        n_resamples=n_bootstrap,
        confidence_level=0.95,
        random_state=seed,
        method="BCa",
    )

    return {
        "mean": round(float(np.mean(vals)), 6),
        "median": round(float(np.median(vals)), 6),
        "std": round(float(np.std(vals, ddof=1)), 6),
        "min": round(float(np.min(vals)), 6),
        "max": round(float(np.max(vals)), 6),
        "ci_95_low": round(float(ci_result.confidence_interval.low), 6),
        "ci_95_high": round(float(ci_result.confidence_interval.high), 6),
        "n_used": n_used,
    }


# ── Ranking helpers ─────────────────────────────────────────────────────────

_LOADED_SWEEPS_CACHE: dict[Path, dict[tuple[str, str], SweepResult]] = {}


def load_all_sweeps(
    sweep_dir: str | Path = "data/sweeps",
) -> dict[tuple[str, str], SweepResult]:
    """Load every ``variables_lyapunov.npz`` under *sweep_dir*.

    Returns a dict keyed by ``(filter_type, window)`` for non-Kaiser
    sweeps or ``(filter_type, "kaiser_<beta>")`` for Kaiser  beta-sweeps.
    Results are cached in memory so repeated calls are free.

    Parameters
    ----------
    sweep_dir
        Root directory containing per-configuration subdirectories
        and an optional ``kaiser/<filter_type>/beta_<N>`` subtree.

    Returns
    -------
    dict
        Mapping from ``(filter_type, window_key)`` to :class:`~chaotic_pfc.analysis.sweep.SweepResult`.
    """
    sweep_dir = Path(sweep_dir).resolve()
    if sweep_dir in _LOADED_SWEEPS_CACHE:
        return _LOADED_SWEEPS_CACHE[sweep_dir]

    loaded: dict[tuple[str, str], SweepResult] = {}
    for path in sorted(sweep_dir.rglob("variables_lyapunov.npz")):
        result = load_sweep(path)
        if result.window == "kaiser":
            beta = result.metadata.get("kaiser_beta", 0.0)
            key = (result.filter_type, f"kaiser_{float(beta):.2f}")
        else:
            key = (result.filter_type, result.window)
        loaded[key] = result

    _LOADED_SWEEPS_CACHE[sweep_dir] = loaded
    return loaded


def export_summary_json(
    data_dir: str | Path = "data/sweeps",
    output: str | Path = "data/analysis_summary.json",
) -> Path:
    """Write the full summary table to a JSON file.

    Parameters
    ----------
    data_dir
        Root directory containing the sweep ``.npz`` checkpoints.
    output
        Destination JSON path.

    Returns
    -------
    Path
        The path written.
    """
    output = Path(output)
    rows = summary_table(data_dir)
    output.write_text(json.dumps(rows, indent=2, default=str))
    return output


# ── Kaiser β-sweep analysis ──────────────────────────────────────────────────

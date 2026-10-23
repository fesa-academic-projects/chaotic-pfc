"""Summary tables, area/exponent statistics and sweep loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc._validate import require_positive_int, require_seed
from chaotic_pfc.analysis.stats._common import (
    _DEFAULT_N_BOOTSTRAP,
    _discover_all,
    _safe_stats,
    pct_round,
    round_half_up,
)
from chaotic_pfc.analysis.stats._types import AreaSummary, LmaxStats, SummaryRow
from chaotic_pfc.analysis.sweep import LAMBDA_CHAOS_THRESHOLD, SweepResult, load_sweep
from chaotic_pfc.errors import InvalidParameterError


def summary_table(
    data_dir: str | Path = "data/sweeps",
) -> list[SummaryRow]:
    """Return one row per sweep with key statistics.

    Each row is a :class:`SummaryRow` with: ``window``, ``filter_type``,
    ``n_orders``, ``n_cutoffs``, ``pct_chaotic``, ``pct_periodic`` (the
    non-chaotic fraction), ``pct_divergent``, ``mean_lmax``, ``max_lmax``, ``beta`` (Kaiser only).

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
        chaotic = np.sum((~np.isnan(h)) & (h > LAMBDA_CHAOS_THRESHOLD))
        periodic = np.sum((~np.isnan(h)) & (h <= LAMBDA_CHAOS_THRESHOLD))
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
            "mean_lmax": round_half_up(mean_lmax, 4),
            "max_lmax": round_half_up(max_lmax, 4),
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

    Raises
    ------
    InvalidParameterError
        If *sweep_result* is not a :class:`~.sweep.SweepResult` or its
        ``h`` array is not 2-D.
    """
    if not isinstance(sweep_result, SweepResult):
        raise InvalidParameterError(
            f"sweep_result must be a SweepResult, got {type(sweep_result).__name__}"
        )
    h = np.asarray(sweep_result.h, dtype=np.float64)
    if h.ndim != 2:
        raise InvalidParameterError(f"sweep_result.h must be 2-D, got shape {h.shape}")
    finite = np.isfinite(h)
    chaotic = np.sum(finite & (h > LAMBDA_CHAOS_THRESHOLD))
    periodic = np.sum(finite & (h <= LAMBDA_CHAOS_THRESHOLD))
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

    Raises
    ------
    InvalidParameterError
        If *sweep_result* is not a :class:`~.sweep.SweepResult`,
        *region* is unknown, or *n_bootstrap* / *seed* is invalid.
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    if not isinstance(sweep_result, SweepResult):
        raise InvalidParameterError(
            f"sweep_result must be a SweepResult, got {type(sweep_result).__name__}"
        )
    n_bootstrap = require_positive_int("n_bootstrap", n_bootstrap, min_value=1)
    seed_val = require_seed("seed", seed)

    h = np.asarray(sweep_result.h, dtype=np.float64).ravel()
    finite = np.isfinite(h)

    if region == "chaotic":
        mask = finite & (h > LAMBDA_CHAOS_THRESHOLD)
    elif region == "periodic":
        mask = finite & (h <= LAMBDA_CHAOS_THRESHOLD)
    elif region == "all_finite":
        mask = finite
    else:
        raise InvalidParameterError(
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

    if np.all(vals == vals[0]):
        # A constant sample has zero variance: BCa's acceleration term is 0/0
        # and scipy emits a RuntimeWarning. The confidence interval of a
        # constant sample is the constant itself.
        ci_low = ci_high = float(vals[0])
        return {
            "mean": round_half_up(float(vals[0]), 6),
            "median": round_half_up(float(vals[0]), 6),
            "std": 0.0,
            "min": round_half_up(float(vals[0]), 6),
            "max": round_half_up(float(vals[0]), 6),
            "ci_95_low": round_half_up(ci_low, 6),
            "ci_95_high": round_half_up(ci_high, 6),
            "n_used": n_used,
        }

    # scipy's BCa implementation computes a skewness correction that can emit
    # an invalid-divide RuntimeWarning for near-degenerate resamples; the
    # interval itself is still well defined, so only that internal arithmetic
    # is silenced here.
    with np.errstate(invalid="ignore", divide="ignore"):
        ci_result = scipy_bootstrap(
            (vals,),
            np.mean,
            n_resamples=n_bootstrap,
            confidence_level=0.95,
            random_state=seed_val,
            method="BCa",
        )

    return {
        "mean": round_half_up(float(np.mean(vals)), 6),
        "median": round_half_up(float(np.median(vals)), 6),
        "std": round_half_up(float(np.std(vals, ddof=1)), 6),
        "min": round_half_up(float(np.min(vals)), 6),
        "max": round_half_up(float(np.max(vals)), 6),
        "ci_95_low": round_half_up(float(ci_result.confidence_interval.low), 6),
        "ci_95_high": round_half_up(float(ci_result.confidence_interval.high), 6),
        "n_used": n_used,
    }


# ── Ranking helpers ─────────────────────────────────────────────────────────

_CacheKey = tuple[Path, tuple[tuple[str, int, int], ...]]
_LOADED_SWEEPS_CACHE: dict[_CacheKey, dict[tuple[str, str], SweepResult]] = {}


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
    paths = sorted(sweep_dir.rglob("variables_lyapunov.npz"))
    files_key = tuple(
        (path.relative_to(sweep_dir).as_posix(), st.st_mtime_ns, st.st_size)
        for path in paths
        for st in (path.stat(),)
    )
    cache_key: _CacheKey = (sweep_dir, files_key)
    if cache_key in _LOADED_SWEEPS_CACHE:
        return dict(_LOADED_SWEEPS_CACHE[cache_key])

    loaded: dict[tuple[str, str], SweepResult] = {}
    for path in paths:
        result = load_sweep(path)
        if result.window == "kaiser":
            beta = result.metadata.get("kaiser_beta", 0.0)
            key = (result.filter_type, f"kaiser_{float(beta):.2f}")
        else:
            key = (result.filter_type, result.window)
        loaded[key] = result

    _LOADED_SWEEPS_CACHE[cache_key] = loaded
    return dict(loaded)


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

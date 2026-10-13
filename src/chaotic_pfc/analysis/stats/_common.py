"""Shared helpers for the sweep statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc.analysis.sweep import SweepResult, load_sweep
from chaotic_pfc.config import DEFAULT_CONFIG

_DEFAULT_N_BOOTSTRAP = DEFAULT_CONFIG.sweep.n_bootstrap


def _discover_all(data_dir: str | Path = "data/sweeps") -> list[SweepResult]:
    """Load every ``variables_lyapunov.npz`` under *data_dir*."""
    results: list[SweepResult] = []
    for path in sorted(Path(data_dir).rglob("variables_lyapunov.npz")):
        results.append(load_sweep(path))
    return results


def _safe_stats(h: NDArray[Any]) -> tuple[float, float]:
    """Mean and max of *h* ignoring NaN and Inf.

    Returns ``(NaN, NaN)`` when all entries of *h* are non-finite
    (e.g. a fully-divergent sweep).  Callers are expected to handle
    this gracefully when formatting output.
    """
    finite = h[np.isfinite(h)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.max(finite))

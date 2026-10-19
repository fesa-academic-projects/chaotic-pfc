"""Shared helpers for the sweep statistics."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc.analysis.sweep import SweepResult, load_sweep
from chaotic_pfc.config import DEFAULT_CONFIG

_DEFAULT_N_BOOTSTRAP = DEFAULT_CONFIG.sweep.n_bootstrap


def pct(n: int, d: int, places: int = 1) -> str:
    """Percentage computed from integer counts, quantized with ROUND_HALF_UP."""
    if d == 0:
        return "n/a"
    q = Decimal(1).scaleb(-places)
    return str((Decimal(100 * int(n)) / Decimal(int(d))).quantize(q, rounding=ROUND_HALF_UP))


def pct_round(n: int, d: int, places: int = 1) -> float:
    """Percentage from integer counts as a float, quantized with ROUND_HALF_UP."""
    if d <= 0:
        return 0.0
    return float(pct(int(n), int(d), places))


def round_half_up(val: float, places: int = 1) -> float:
    """Round a float to *places* decimals with ROUND_HALF_UP."""
    q = Decimal(1).scaleb(-places)
    return float(Decimal(repr(float(val))).quantize(q, rounding=ROUND_HALF_UP))


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

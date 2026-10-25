#!/usr/bin/env python3
"""Production-path reproduction helper for the manuscript diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from chaotic_pfc.analysis.sweep._kernel import _build_task_order, _sweep_kernel


def reproduce_cell(
    sr: Any,
    bank: np.ndarray,
    gains: np.ndarray,
    perturbations: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Recompute one cell with the production kernel and its exact ICs."""
    orders = np.array([sr.num_taps_grid[i]], dtype=np.float64)
    cutoffs = np.array([sr.cutoff_grid[j]], dtype=np.float64)
    task = np.array([0], dtype=np.int64)
    h, _, _, _, _ = _sweep_kernel(
        orders,
        cutoffs,
        bank[i : i + 1, j : j + 1, :],
        gains[i : i + 1, j : j + 1],
        perturbations[i : i + 1, j : j + 1, :, :],
        task,
        int(sr.metadata["Nitera"]),
        int(sr.metadata["Nmap"]),
        int(sr.metadata.get("Nmap_min") or 500),
        float(sr.metadata.get("tol") or 1e-3),
        int(sr.metadata["n_initial"]),
        float(sr.metadata["alpha"]),
        float(sr.metadata["beta"]),
        1,
    )
    return float(h[0, 0])


def reproduce_grid(
    sr: Any, bank: np.ndarray, gains: np.ndarray, perturbations: np.ndarray
) -> np.ndarray:
    """Recompute a full grid with the production kernel and its exact ICs."""
    orders = np.asarray(sr.num_taps_grid, dtype=np.float64)
    cutoffs = np.asarray(sr.cutoff_grid, dtype=np.float64)
    task = _build_task_order(orders.astype(np.int64), len(cutoffs))
    h, _, _, _, _ = _sweep_kernel(
        orders,
        cutoffs,
        bank,
        gains,
        perturbations,
        task,
        int(sr.metadata["Nitera"]),
        int(sr.metadata["Nmap"]),
        int(sr.metadata.get("Nmap_min") or 500),
        float(sr.metadata.get("tol") or 1e-3),
        int(sr.metadata["n_initial"]),
        float(sr.metadata["alpha"]),
        float(sr.metadata["beta"]),
        1,
    )
    return h


def max_difference(stored: np.ndarray, reproduced: np.ndarray) -> tuple[float, int]:
    """Return the NaN-aware max abs difference and the NaN-pattern mismatch count."""
    stored = np.asarray(stored, dtype=np.float64)
    reproduced = np.asarray(reproduced, dtype=np.float64)
    nan_diff = int(np.count_nonzero(np.isnan(stored) != np.isnan(reproduced)))
    both = ~np.isnan(stored) & ~np.isnan(reproduced)
    diff = float(np.max(np.abs(stored[both] - reproduced[both]))) if both.any() else 0.0
    if nan_diff:
        diff = float("inf")
    return diff, nan_diff

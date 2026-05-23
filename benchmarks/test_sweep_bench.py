"""Benchmarks for sweep and FIR precomputation — orchestration overhead."""

from __future__ import annotations

import numpy as np

from chaotic_pfc.analysis.sweep import run_sweep


def test_mini_sweep_30_points(benchmark):
    """Mini sweep (30 grid points) — typical quick mode operation."""
    orders = np.arange(2, 8)
    cutoffs = np.linspace(0.1, 0.9, 5)
    result = benchmark(
        run_sweep,
        window="hamming",
        filter_type="lowpass",
        orders=orders,
        cutoffs=cutoffs,
        Nitera=50,
        Nmap=200,
        n_initial=3,
        seed=42,
    )
    assert result.h.shape == (len(orders), len(cutoffs))

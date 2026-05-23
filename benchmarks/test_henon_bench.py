"""Benchmarks for Henon map iteration — the innermost hot path."""

from __future__ import annotations

import numpy as np

from chaotic_pfc.dynamics.maps import (
    henon_filtered,
    henon_generalised,
    henon_order_n,
    henon_standard,
)


def test_henon_standard_1000_iters(benchmark):
    """Hénon standard map, 1000 iterations."""
    X, _Y = benchmark(henon_standard, steps=1000, x0=0.0, y0=0.0, a=1.4, b=0.3)
    assert len(X) == 1001
    assert len(_Y) == 1001


def test_henon_standard_10000_iters(benchmark):
    """Hénon standard map, 10 000 iterations — typical sweep point."""
    X, _Y = benchmark(henon_standard, steps=10000, x0=0.0, y0=0.0, a=1.4, b=0.3)
    assert len(X) == 10001


def test_henon_generalised_1000_iters(benchmark):
    """Hénon generalised map (α=1.4, β=0.3), 1000 iterations."""
    X, _Y = benchmark(henon_generalised, steps=1000, x0=0.0, y0=0.0, alpha=1.4, beta=0.3)
    assert len(X) == 1001


def test_henon_filtered_1000_iters(benchmark):
    """Hénon filtered map (c0=1, c1=0 — identity filter), 1000 iterations."""
    X, _Y = benchmark(henon_filtered, steps=1000, x0=0.0, y0=0.0, c0=1.0, c1=0.0)
    assert len(X) == 1001


def test_henon_order_n_1000_iters(benchmark):
    """Order-N Hénon (Nc=4), 1000 iterations."""
    coeffs = np.array([0.5, 0.3, 0.15, 0.05])
    X, _Y = benchmark(henon_order_n, steps=1000, fir_coeffs=coeffs)
    assert X.shape == (4, 1001)

"""Benchmarks for Henon map iteration — the innermost hot path."""

from __future__ import annotations

from chaotic_pfc.dynamics.maps import henon_standard


def test_henon_standard_1000_iters(benchmark):
    """Hénon standard map, 1000 iterations."""
    X, _Y = benchmark(henon_standard, steps=1000, x0=0.0, y0=0.0, a=1.4, b=0.3)
    assert len(X) == 1001
    assert len(_Y) == 1001


def test_henon_standard_10000_iters(benchmark):
    """Hénon standard map, 10 000 iterations — typical sweep point."""
    X, _Y = benchmark(henon_standard, steps=10000, x0=0.0, y0=0.0, a=1.4, b=0.3)
    assert len(X) == 10001

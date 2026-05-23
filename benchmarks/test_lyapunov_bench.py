"""Benchmarks for Lyapunov exponent computation — 71% of sweep time."""

from __future__ import annotations

from chaotic_pfc.dynamics.lyapunov import (
    lyapunov_henon2d,
    lyapunov_henon2d_ensemble,
)


def test_lyapunov_henon2d_2000_iters(benchmark):
    """Single Lyapunov calculation, 2000 iterations."""
    result = benchmark(
        lyapunov_henon2d,
        alpha=1.4,
        beta=0.3,
        Nitera=2000,
        Ndiscard=1000,
        seed=42,
    )
    assert result.lyapunov_max > 0


def test_lyapunov_ensemble_25_ics(benchmark):
    """Ensemble Lyapunov with 25 ICs — typical sweep grid point load."""
    result = benchmark(
        lyapunov_henon2d_ensemble,
        alpha=1.4,
        beta=0.3,
        Nitera=500,
        Ndiscard=200,
        n_initial=25,
        seed=42,
        perturbation=0.1,
    )
    assert result.n_chaotic > 0

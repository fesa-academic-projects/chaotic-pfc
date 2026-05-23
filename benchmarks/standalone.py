#!/usr/bin/env python3
"""Quick performance benchmarks for core operations.

Usage:  python scripts/benchmark.py

All benchmarks run with Numba JIT acceleration when available
(``pip install -e '.[fast]'``). Without Numba, the same code
runs in pure Python — correct but ~80-100x slower.
"""

import time

import numpy as np

from chaotic_pfc._compat import HAS_NUMBA
from chaotic_pfc.analysis.sweep import precompute_fir_bank

# ── Imports ──────────────────────────────────────────────────────────────
from chaotic_pfc.dynamics.lyapunov import lyapunov_henon2d, lyapunov_max
from chaotic_pfc.dynamics.maps import (
    henon_filtered,
    henon_generalised,
    henon_order_n,
    henon_standard,
)

REPEAT = 3


def _timeit(label: str, fn, *args, **kwargs) -> float:
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    print(f"  {label:<50} {best:8.4f} s")
    return best


def main() -> None:
    print("=" * 65)
    print("  chaotic-pfc benchmarks")
    print(f"  Numba JIT: {'available' if HAS_NUMBA else 'NOT available (pure Python)'}")
    print("=" * 65)

    # ── Henon maps ───────────────────────────────────────────────────────
    print("\nHenon maps (500_000 iterations):")
    _timeit("henon_standard", henon_standard, 500_000)
    _timeit("henon_generalised", henon_generalised, 500_000)
    _timeit("henon_filtered (c0=1, c1=0)", henon_filtered, 500_000, c0=1.0, c1=0.0)
    _timeit(
        "henon_order_n (Nc=4)", henon_order_n, 50_000, fir_coeffs=np.array([0.5, 0.3, 0.15, 0.05])
    )

    # ── FIR bank ─────────────────────────────────────────────────────────
    print("\nFIR coefficient bank:")
    orders = np.arange(2, 42)
    cutoffs = np.linspace(0.01, 0.99, 100)
    _timeit(
        "precompute_fir_bank (40x100 lowpass)",
        precompute_fir_bank,
        orders,
        cutoffs,
        "lowpass",
        "hamming",
    )

    # ── Lyapunov ─────────────────────────────────────────────────────────
    print("\nLyapunov exponents:")
    _timeit("lyapunov_henon2d (Nitera=2000)", lyapunov_henon2d, Nitera=2000, Ndiscard=500)
    _timeit("lyapunov_max 4-D (Nitera=2000)", lyapunov_max, Nitera=2000, Ndiscard=500)
    _timeit("lyapunov_max 4-D (Nitera=10000)", lyapunov_max, Nitera=10000, Ndiscard=1000)

    print()


if __name__ == "__main__":
    main()

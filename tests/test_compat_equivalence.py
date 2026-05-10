"""tests/test_compat_equivalence.py — Verify Numba JIT vs pure-Python equivalence."""

import unittest

import numpy as np

from chaotic_pfc._compat import HAS_NUMBA, njit


class TestCompatEquivalence(unittest.TestCase):
    def test_jit_and_raw_produce_same_result(self):
        """njit-decorated function must match undecorated output exactly."""

        def f(a, b, c):
            x = a + b * c
            for _ in range(10):
                x = x * 0.9 + a
            return x

        f_jit = njit(f)
        a, b, c = 1.5, 3.0, 2.0
        self.assertAlmostEqual(f_jit(a, b, c), f(a, b, c))

    @unittest.skipUnless(HAS_NUMBA, "Numba not installed")
    def test_parallel_dot_matches_sequential(self):
        """prange-based dot product matches sequential with Numba."""
        from chaotic_pfc._compat import prange

        @njit(parallel=True)
        def parallel_sum(arr):
            s = 0.0
            for i in prange(len(arr)):
                s += arr[i]
            return s

        @njit
        def sequential_sum(arr):
            s = 0.0
            for i in range(len(arr)):
                s += arr[i]
            return s

        arr = np.random.default_rng(42).random(1000)
        # Parallel sum has floating-point non-determinism from reduction order
        ps = parallel_sum(arr)
        ss = sequential_sum(arr)
        self.assertAlmostEqual(ps, ss, places=10)

    def test_njit_with_fastmath_preserves_identity(self):
        """njit with fastmath=True should still compute correctly."""

        @njit(fastmath=True)
        def g(x):
            return np.sin(x) ** 2 + np.cos(x) ** 2

        for x in (0.0, 0.5, 1.0, 2.0):
            self.assertAlmostEqual(g(x), 1.0, places=12)

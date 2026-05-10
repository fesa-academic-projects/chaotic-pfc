"""tests/test_compat.py — Unit tests for the Numba compatibility layer."""

import unittest

from chaotic_pfc._compat import get_num_threads, njit, prange


class TestCompat(unittest.TestCase):
    def test_get_num_threads_returns_int(self):
        n = get_num_threads()
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 1)

    def test_njit_noop_passthrough(self):
        @njit
        def f(x):
            return x + 1

        self.assertEqual(f(41), 42)

    def test_njit_with_kwargs(self):
        @njit(cache=True, parallel=False)
        def f(x):
            return x * 2

        self.assertEqual(f(21), 42)

    def test_prange_behaves_like_range(self):
        self.assertEqual(list(prange(5)), [0, 1, 2, 3, 4])

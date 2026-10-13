"""tests/test_compat.py — Unit tests for the Numba compatibility layer."""

import unittest
from pathlib import Path

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


def test_compat_without_numba(monkeypatch):
    """Exercise the pure-Python fallback branch of ``_compat`` (Numba absent).

    ``_compat`` only takes its fallback branch when ``import numba`` fails, so
    the module is re-executed in-process with ``numba`` masked to ``None`` in
    ``sys.modules`` (which makes the import raise). This is the branch the CI
    selects with ``NUMBA_DISABLE_JIT=1`` plus a Numba-free install.
    """
    import importlib.util
    import sys

    import chaotic_pfc

    monkeypatch.setitem(sys.modules, "numba", None)
    path = Path(chaotic_pfc.__file__).with_name("_compat.py")
    spec = importlib.util.spec_from_file_location("_compat_no_numba_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.HAS_NUMBA is False
    assert module.get_num_threads() == 1
    assert list(module.prange(2, 7, 2)) == [2, 4, 6]

    @module.njit(cache=True, parallel=True)
    def inc(x):
        return x + 1

    assert inc(41) == 42

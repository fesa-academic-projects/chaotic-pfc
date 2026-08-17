"""Compatibility layer that makes Numba optional.

When Numba is installed, ``njit``, ``prange``, and ``get_num_threads``
delegate to the real library. When it is not, ``njit`` becomes a no-op
passthrough, ``prange`` falls back to the built-in ``range``, and
``get_num_threads()`` returns 1.

This keeps the import-time contract clean: the package installs and
imports regardless of whether ``pip install chaotic-pfc[fast]`` was
used. The sweep kernels will run without Numba — just much slower.
"""

from __future__ import annotations

import builtins
from typing import Any

try:
    from numba import get_num_threads as _get_num_threads
    from numba import njit as _njit
    from numba import prange as _prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _njit_slow(*args: Any, **kwargs: Any) -> Any:
    """No-op fallback: returns the decorated function unchanged.

    Accepts all Numba keyword arguments (``cache``, ``parallel``,
    ``inline``, ``fastmath``) silently to keep call sites compatible.
    """
    if args and callable(args[0]):
        return args[0]
    return lambda fn: fn


def _prange_slow(stop: int, *args: Any, **kwargs: Any) -> range:
    """Fallback for ``numba.prange`` — behaves like ``builtins.range``."""
    return builtins.range(stop, *args)


def _get_num_threads_slow() -> int:
    """Fallback for ``numba.get_num_threads`` — single-threaded."""
    return 1


njit: Any
prange: Any
get_num_threads: Any

if HAS_NUMBA:
    njit = _njit
    prange = _prange
    get_num_threads = _get_num_threads
else:
    njit = _njit_slow
    prange = _prange_slow
    get_num_threads = _get_num_threads_slow

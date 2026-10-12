"""Parameter-sweep computation of Lyapunov exponents for the filtered Henon map.

This module sweeps the 2-D grid of filter orders ``N_s in {2, ..., 41}``
against normalised cutoffs ``omega_c in (0, 1)`` and estimates the largest
Lyapunov exponent ``lambda_max`` at each grid point.

All hot paths are JIT-compiled with Numba. The outer 2-D loop is flattened
into a single ``prange`` so every grid point is a task available to the
thread pool.

The public API is small:

- :data:`SweepResult` — container for the output of one sweep.
- :func:`precompute_fir_bank` — builds the FIR coefficient tensor.
- :func:`run_sweep` — full end-to-end computation for a (window, filter) pair.
- :func:`save_sweep` / :func:`load_sweep` — round-trip to ``.npz``.
- ``WINDOWS``, ``FILTER_TYPES``, ``WINDOW_DISPLAY_NAMES`` — the catalogue of
  supported configurations (see the API reference page).
"""

from __future__ import annotations

from ._io import _infer_config_from_path, load_sweep, save_sweep  # noqa: F401
from ._kernel import _build_task_order  # noqa: F401
from ._orchestration import (
    precompute_fir_bank,
    quick_sweep_params,
    run_sweep,
)
from ._types import (
    FILTER_TYPES,
    WINDOW_DISPLAY_NAMES,
    WINDOWS,
    SweepResult,
)

__all__ = [
    "FILTER_TYPES",
    "WINDOWS",
    "WINDOW_DISPLAY_NAMES",
    "SweepResult",
    "load_sweep",
    "precompute_fir_bank",
    "quick_sweep_params",
    "run_sweep",
    "save_sweep",
]

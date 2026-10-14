"""Single source of the mathematical formulas shared across the package.

The positive fixed point of the filtered Hénon map and the two-dimensional Hénon
recurrence appear both in the Numba sweep kernel (``analysis.sweep._kernel``)
and in the pure-Python code (``dynamics.lyapunov``, ``dynamics.maps``). They are
defined once here — Numba-compilable — and imported by the sweep kernel. The
pure-Python validation wrappers keep their own NumPy copies on purpose (see
below).

The modified Gram-Schmidt (MGS) orthogonalisation has two shapes that must stay
bit-compatible with the versioned sweeps: the in-place, log-accumulating variant
in :func:`chaotic_pfc.analysis.sweep._kernel._mgs_accumulate` (the reference used
by the full-spectrum fallback) and the NumPy variant in
``dynamics.lyapunov._gram_schmidt`` used by the validation wrappers. They are
kept as two implementations of the same algorithm because the fallback's
bit-exactness depends on the scalar loop; ``tests/test_formulas.py`` pins that
the two agree on the orthonormality property.

The pure-Python validation wrappers in ``dynamics.lyapunov`` deliberately keep
their **own NumPy copies** of :func:`fixed_point_positive` and
:func:`henon2d_step`: the Numba-compiled versions here differ from the NumPy
evaluation by ~1 ULP, which the chaotic map amplifies (the two trajectories
first diverge around iteration 6750), so the single-seed estimator would move
from the published ``0.41756661`` to ``0.42280037``.  The shared compiled
formulas are the reference for the **sweeps** (the versioned ``.npz`` and the
``paper numbers``); the NumPy copies are the reference for the single-seed
**validation** numbers (``docs/validation.rst``), pinned by
``tests/test_lyapunov_regression.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc._compat import njit


@njit(cache=True, inline="always")
def fixed_point_positive(gain: Any, alpha: Any, beta: Any) -> Any:
    """Positive fixed point of the filtered Hénon map (cancellation-free).

    Solves ``G^2 p^2 + (1 - beta) p - alpha = 0`` with ``G = gain`` and returns
    its positive root ``2 alpha / ((1 - beta) + sqrt((1 - beta)^2 + 4 alpha G^2))``,
    the form that stays accurate as ``G -> 0``.
    """
    disc = (1.0 - beta) ** 2 + 4.0 * alpha * gain**2
    return 2.0 * alpha / ((1.0 - beta) + disc**0.5)


@njit(cache=True)
def henon2d_step(x: Any, alpha: Any, beta: Any) -> Any:
    """One iterate of the generalised 2-D Hénon map.

    ``x = [x1, x2]`` -> ``[alpha - x1**2 + beta * x2, x1]``.
    """
    out = np.empty(2)
    out[0] = alpha - x[0] * x[0] + beta * x[1]
    out[1] = x[0]
    return out


def gram_schmidt_python(Z: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Orthonormalise the columns of ``Z`` with modified Gram-Schmidt (NumPy path).

    Returns the orthonormal columns ``Q`` and the column norms before
    normalising. The sweep kernel uses its own scalar variant; see the module
    docstring.
    """
    _d, n = Z.shape
    Q = np.zeros_like(Z)
    norms = np.zeros(n)
    for i in range(n):
        v = Z[:, i].copy()
        for j in range(i):
            v -= np.dot(Q[:, j], v) * Q[:, j]
        norms[i] = np.linalg.norm(v)
        if norms[i] > 0:
            Q[:, i] = v / norms[i]
    return Q, norms

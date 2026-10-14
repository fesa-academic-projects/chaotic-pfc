"""Regression pins for the Lyapunov estimators.

These values are deterministic and must not change silently.  In particular,
replacing the pure-NumPy Hénon step in :mod:`chaotic_pfc.dynamics.lyapunov` by
the Numba-compiled ``_formulas.henon2d_step`` changes the single-seed result:
the two differ by ~1 ULP, which the chaotic map amplifies, so the finite-time
estimate moves from 0.41756661 to 0.42280037.  The asymptotic exponent is the
same, but the published validation value is the former, so the pure path is
kept pinned here (the sweep kernel uses the compiled ``_formulas`` version).
"""

from __future__ import annotations

import numpy as np
import pytest

from chaotic_pfc.dynamics.lyapunov import lyapunov_henon2d, lyapunov_max, lyapunov_spectrum

# Reference values (computed with the pure-NumPy Hénon step).
_HENON2D_SEED42 = [0.41756661229176445, -1.6215394166177004]
_GENERIC_SPECTRUM = [0.41830591556930247, -1.6222787198952386]
_LAMBDA1 = 0.41830591556930247


def _henon_iterate(x: np.ndarray) -> np.ndarray:
    return np.array([1.4 - x[0] ** 2 + 0.3 * x[1], x[0]])


def _henon_jacobian(x: np.ndarray) -> np.ndarray:
    return np.array([[-2.0 * x[0], 0.3], [1.0, 0.0]])


def test_lyapunov_henon2d_seed42_is_pinned() -> None:
    """The single-seed estimator reproduces the published validation value."""
    result = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=20000, Ndiscard=1000, seed=42)
    np.testing.assert_allclose(result.all_exponents, _HENON2D_SEED42, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        float(result.lyapunov_max), _HENON2D_SEED42[0], rtol=1e-12, atol=1e-12
    )


def test_lyapunov_spectrum_henon_is_pinned() -> None:
    """The generic full-spectrum estimator on the standard Hénon map."""
    spectrum = lyapunov_spectrum(
        _henon_iterate, _henon_jacobian, np.array([0.1, 0.1]), Nitera=20000, Ndiscard=1000
    )
    np.testing.assert_allclose(spectrum, _GENERIC_SPECTRUM, rtol=1e-12, atol=1e-12)


def test_lyapunov_max_henon_is_pinned() -> None:
    """The generic largest-exponent estimator on the standard Hénon map."""
    value = lyapunov_max(
        _henon_iterate, _henon_jacobian, np.array([0.1, 0.1]), Nitera=20000, Ndiscard=1000
    )
    assert value == pytest.approx(_LAMBDA1, rel=1e-12, abs=1e-12)

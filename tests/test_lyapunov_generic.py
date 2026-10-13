"""Generic public Lyapunov estimator.

``lyapunov_spectrum`` and ``lyapunov_max`` accept any map and its Jacobian, so
they are checked here against the two independent references used across the
test suite: the standard Hénon map (Sprott 2003) and the analytic
``alpha=2, beta=0`` map (conjugate to the logistic map at r=4, exponent ln 2).
"""

from __future__ import annotations

import math

import numpy as np

from chaotic_pfc import lyapunov_max, lyapunov_spectrum


def _henon(alpha: float = 1.4, beta: float = 0.3):
    iterate = lambda x: np.array([alpha - x[0] ** 2 + beta * x[1], x[0]])  # noqa: E731
    jacobian = lambda x: np.array([[-2.0 * x[0], beta], [1.0, 0.0]])  # noqa: E731
    return iterate, jacobian


def _quadratic():
    iterate = lambda x: np.array([2.0 - x[0] ** 2, x[0]])  # noqa: E731
    jacobian = lambda x: np.array([[-2.0 * x[0], 0.0], [1.0, 0.0]])  # noqa: E731
    return iterate, jacobian


def test_generic_henon_matches_literature() -> None:
    iterate, jacobian = _henon()
    lam = lyapunov_max(iterate, jacobian, np.array([0.1, 0.1]), Nitera=20000, Ndiscard=1000)
    assert abs(lam - 0.419) < 5e-3, f"lambda_1 = {lam}, expected ~0.419 (Sprott 2003)"


def test_generic_analytic_quadratic_is_ln2() -> None:
    iterate, jacobian = _quadratic()
    lam = lyapunov_max(iterate, jacobian, np.array([0.1, 0.1]), Nitera=20000, Ndiscard=1000)
    assert abs(lam - math.log(2)) < 1e-3, f"lambda_1 = {lam}, expected ln 2"


def test_generic_spectrum_returns_every_exponent() -> None:
    iterate, jacobian = _henon()
    spectrum = lyapunov_spectrum(
        iterate, jacobian, np.array([0.1, 0.1]), Nitera=5000, Ndiscard=1000
    )
    assert spectrum.shape == (2,)
    # Arnold identity for the Hénon map: lambda_1 + lambda_2 = ln|beta|.
    assert abs(float(spectrum.sum()) - math.log(0.3)) < 1e-2


def test_generic_matches_the_henon_wrapper() -> None:
    from chaotic_pfc import lyapunov_henon2d

    iterate, jacobian = _henon()
    generic = lyapunov_max(iterate, jacobian, np.array([0.1, 0.1]), Nitera=5000, Ndiscard=1000)
    wrapped = lyapunov_henon2d(Nitera=5000, Ndiscard=1000, seed=42)
    assert abs(generic - wrapped.lyapunov_max) < 0.05

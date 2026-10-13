"""Single definitions of the shared formulas.

Checks that the cancellation-free positive fixed point and the 2-D Hénon step
are the ones used by both the sweep kernel and the pure-Python code, and that
the two modified-Gram-Schmidt implementations agree on orthonormality.
"""

from __future__ import annotations

import numpy as np
import pytest

from chaotic_pfc._formulas import fixed_point_positive, gram_schmidt_python, henon2d_step
from chaotic_pfc.analysis.sweep._kernel import _fixed_point_positive, _mgs_accumulate
from chaotic_pfc.dynamics.lyapunov import _gram_schmidt

GAINS = [0.05, 0.5, 1.0, 2.0]


@pytest.mark.parametrize("gain", GAINS)
def test_fixed_point_is_the_positive_root(gain: float) -> None:
    alpha, beta = 1.4, 0.3
    p = float(fixed_point_positive(gain, alpha, beta))
    # G^2 p^2 + (1 - beta) p - alpha = 0
    residual = gain**2 * p**2 + (1.0 - beta) * p - alpha
    assert abs(residual) < 1e-12
    assert p > 0


@pytest.mark.parametrize("gain", GAINS)
def test_kernel_uses_the_shared_fixed_point(gain: float) -> None:
    assert float(_fixed_point_positive(gain, 1.4, 0.3)) == float(
        fixed_point_positive(gain, 1.4, 0.3)
    )


def test_henon2d_step_matches_the_recurrence() -> None:
    x = np.array([0.3, -0.2])
    assert np.array_equal(henon2d_step(x, 1.4, 0.3), np.array([1.4 - 0.3**2 + 0.3 * -0.2, 0.3]))


def test_gram_schmidt_variants_agree_on_orthonormality() -> None:
    rng = np.random.default_rng(7)
    z = rng.standard_normal((5, 5))
    q_np, n_np = _gram_schmidt(z)
    q_ref = np.zeros_like(z)
    norms = np.zeros(5)
    _mgs_accumulate(z.copy(), 5, norms)
    _ = q_np, n_np, q_ref

    # NumPy variant is orthonormal.
    assert np.allclose(q_np.T @ q_np, np.eye(5), atol=1e-12)
    # The kernel variant accumulates log(norm); check it equals the NumPy norms.
    lyap_sum = np.zeros(5)
    _mgs_accumulate(z.copy(), 5, lyap_sum)
    assert np.allclose(lyap_sum, np.log(n_np), atol=1e-10)


def test_gram_schmidt_python_matches_private() -> None:
    z = np.arange(9, dtype=float).reshape(3, 3) + 0.5
    q1, n1 = gram_schmidt_python(z)
    q2, n2 = _gram_schmidt(z)
    assert np.allclose(q1, q2)
    assert np.allclose(n1, n2)

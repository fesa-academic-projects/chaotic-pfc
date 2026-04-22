"""
lyapunov.py
===========
Lyapunov exponent computation for the Hénon map with pole filter.

System (dimension 4):
    x₁[n+1] = α − x₃[n]² + β·x₂[n]
    x₂[n+1] = x₁[n]
    x₃[n+1] = b₀·(α − x₃[n]² + β·x₂[n]) + a₁·x₃[n] + a₂·x₄[n]
    x₄[n+1] = x₃[n]

where b = [b₀, 0, 0] and a = −[1, −2r·cos(w₀), r²]  (pole-filter coeffs).
"""

import numpy as np
from numpy.typing import NDArray


# ── Fixed-point computation ─────────────────────────────────────────────────

def _fixed_point(alpha: float, beta: float, b: NDArray, a: NDArray) -> NDArray:
    """Compute the stable fixed point of the 4-D pole-filtered Hénon map."""
    H = np.sum(a[1:])   # sum of a coeffs (excluding a[0])
    G = np.sum(b)        # sum of b coeffs

    ratio = G / (1.0 - H)
    denom = ratio ** 2

    disc = (1.0 - beta) ** 2 + 4.0 * alpha * denom
    xf = ((beta - 1.0) + np.sqrt(disc)) / (2.0 * denom)

    vf = G * xf / (1.0 - H)
    return np.array([xf, xf, vf, vf])


# ── Jacobian ────────────────────────────────────────────────────────────────

def _jacobian(beta: float, b: NDArray, a: NDArray, x: NDArray) -> NDArray:
    """Jacobian of the 4-D pole-filtered Hénon map at state x."""
    return np.array([
        [0.0,        beta,                -2.0 * x[2],              0.0   ],
        [1.0,        0.0,                  0.0,                     0.0   ],
        [0.0,        b[0] * beta,         -2.0 * b[0] * x[2] + a[1], a[2]],
        [0.0,        0.0,                  1.0,                     0.0   ],
    ])


# ── Map iteration ──────────────────────────────────────────────────────────

def _iterate(alpha: float, beta: float, b: NDArray, a: NDArray,
             x: NDArray) -> NDArray:
    """Single iteration of the 4-D pole-filtered Hénon map."""
    x1 = alpha - x[2] ** 2 + beta * x[1]
    x2 = x[0]
    x3 = b[0] * (alpha - x[2] ** 2 + beta * x[1]) + a[1] * x[2] + a[2] * x[3]
    x4 = x[2]
    return np.array([x1, x2, x3, x4])


# ── Gram-Schmidt orthogonalisation ─────────────────────────────────────────

def _gram_schmidt(Z: NDArray) -> tuple[NDArray, NDArray]:
    """Modified Gram-Schmidt. Returns (Q, norms)."""
    d, n = Z.shape
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


# ── Lyapunov exponent ──────────────────────────────────────────────────────

def lyapunov_max(
    alpha: float = 1.4, beta: float = 0.3,
    Gz: float = 1.0, pole_radius: float = 0.975, w0: float = 0.0,
    Nitera: int = 2000, Ndiscard: int = 1000,
    perturbation: float = 0.1, seed: int = 42,
) -> dict:
    """
    Compute the maximum Lyapunov exponent of the 4-D pole-filtered Hénon map.

    Returns a dict with keys:
        'lyapunov_max'  : float — largest Lyapunov exponent
        'all_exponents' : ndarray of shape (4,)
        'fixed_point'   : ndarray of shape (4,)
        'eigenvalues'   : ndarray of shape (4,) — eigenvalues at fixed point
        'stable'        : bool
    """
    dim = 4
    b = Gz * np.array([1.0, 0.0, 0.0])
    a = -np.array([1.0, -2.0 * pole_radius * np.cos(w0), pole_radius ** 2])

    # Fixed point
    xf = _fixed_point(alpha, beta, b, a)
    J_fp = _jacobian(beta, b, a, xf)
    eigs = np.linalg.eigvals(J_fp)
    stable = bool(np.all(np.abs(eigs) < 1.0))

    # Perturbed initial condition
    rng = np.random.default_rng(seed)
    x = xf * (1.0 + perturbation * (2.0 * rng.random(dim) - 1.0))

    # Discard transient
    for _ in range(Ndiscard):
        x = _iterate(alpha, beta, b, a, x)

    # Lyapunov computation via QR / Gram-Schmidt
    W = np.eye(dim)
    log_r = np.zeros((dim, Nitera))

    for i in range(Nitera):
        J = _jacobian(beta, b, a, x)
        Z = J @ W
        W, norms = _gram_schmidt(Z)
        for k in range(dim):
            log_r[k, i] = np.log(max(norms[k], 1e-300))
        x = _iterate(alpha, beta, b, a, x)

    exponents = np.array([np.mean(log_r[k, :]) for k in range(dim)])

    return {
        "lyapunov_max": float(np.max(exponents)),
        "all_exponents": exponents,
        "fixed_point": xf,
        "eigenvalues": eigs,
        "stable": stable,
    }


def fixed_point_stability(
    alpha: float = 1.4, beta: float = 0.3,
    Gz: float = 1.0, pole_radius: float = 0.975, w0: float = 0.0,
) -> dict:
    """Quick check: fixed point, eigenvalues, stability (4-D filtered)."""
    b = Gz * np.array([1.0, 0.0, 0.0])
    a = -np.array([1.0, -2.0 * pole_radius * np.cos(w0), pole_radius ** 2])
    xf = _fixed_point(alpha, beta, b, a)
    J_fp = _jacobian(beta, b, a, xf)
    eigs = np.linalg.eigvals(J_fp)
    return {
        "fixed_point": xf,
        "eigenvalues": eigs,
        "stable": bool(np.all(np.abs(eigs) < 1.0)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pure 2-D Hénon map (no filter)
# ═══════════════════════════════════════════════════════════════════════════
#
#   x₁[n+1] = α − x₁[n]² + β·x₂[n]
#   x₂[n+1] = x₁[n]
#
# Fixed points: x* = α − x*² + β·x*  ⟹  x*² + (1−β)·x* − α = 0


def _henon2d_fixed_points(alpha: float, beta: float):
    """Both fixed points of the 2-D Hénon map (positive and negative roots)."""
    disc = (1.0 - beta) ** 2 + 4.0 * alpha
    x_p = ((beta - 1.0) + np.sqrt(disc)) / 2.0
    x_n = ((beta - 1.0) - np.sqrt(disc)) / 2.0
    return np.array([x_p, x_p]), np.array([x_n, x_n])


def _henon2d_jacobian(beta: float, x: NDArray) -> NDArray:
    """Jacobian of the 2-D Hénon map at state x."""
    return np.array([
        [-2.0 * x[0], beta],
        [1.0,          0.0 ],
    ])


def _henon2d_iterate(alpha: float, beta: float, x: NDArray) -> NDArray:
    """Single iteration of the 2-D Hénon map."""
    return np.array([
        alpha - x[0] ** 2 + beta * x[1],
        x[0],
    ])


def lyapunov_henon2d(
    alpha: float = 1.4, beta: float = 0.3,
    Nitera: int = 2000, Ndiscard: int = 1000,
    perturbation: float = 0.1, seed: int = 42,
) -> dict:
    """Compute Lyapunov exponents for the standard 2-D Hénon map.

    Returns dict with keys:
        'lyapunov_max'   : float
        'all_exponents'  : ndarray of shape (2,)
        'fixed_point_p'  : ndarray — positive fixed point
        'fixed_point_n'  : ndarray — negative fixed point
        'eigenvalues_p'  : ndarray — eigenvalues at (+) fixed point
        'eigenvalues_n'  : ndarray — eigenvalues at (−) fixed point
        'stable_p'       : bool
        'stable_n'       : bool
    """
    dim = 2

    # ── Fixed points & stability ──
    xf_p, xf_n = _henon2d_fixed_points(alpha, beta)

    J_p = _henon2d_jacobian(beta, xf_p)
    eigs_p = np.linalg.eigvals(J_p)
    stable_p = bool(np.all(np.abs(eigs_p) < 1.0))

    J_n = _henon2d_jacobian(beta, xf_n)
    eigs_n = np.linalg.eigvals(J_n)
    stable_n = bool(np.all(np.abs(eigs_n) < 1.0))

    # ── IC perturbada em torno do ponto fixo (+) ──
    rng = np.random.default_rng(seed)
    x = xf_p * (1.0 + perturbation * (2.0 * rng.random(dim) - 1.0))

    # ── Descarte transiente ──
    for _ in range(Ndiscard):
        x = _henon2d_iterate(alpha, beta, x)

    # ── Lyapunov via Gram-Schmidt ──
    W = np.eye(dim)
    log_r = np.zeros((dim, Nitera))

    for i in range(Nitera):
        J = _henon2d_jacobian(beta, x)
        Z = J @ W
        W, norms = _gram_schmidt(Z)
        for k in range(dim):
            log_r[k, i] = np.log(max(norms[k], 1e-300))
        x = _henon2d_iterate(alpha, beta, x)

    exponents = np.array([np.mean(log_r[k, :]) for k in range(dim)])

    return {
        "lyapunov_max": float(np.max(exponents)),
        "all_exponents": exponents,
        "fixed_point_p": xf_p,
        "fixed_point_n": xf_n,
        "eigenvalues_p": eigs_p,
        "eigenvalues_n": eigs_n,
        "stable_p": stable_p,
        "stable_n": stable_n,
    }

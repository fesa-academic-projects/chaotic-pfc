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

Two flavours of computation are provided:

* ``lyapunov_max`` / ``lyapunov_henon2d`` — single perturbed initial
  condition, returning one estimate of the spectrum.
* ``lyapunov_max_ensemble`` / ``lyapunov_henon2d_ensemble`` — full
  experimental protocol: sample N_ci initial conditions uniformly
  around the fixed point (±``perturbation``), compute the spectrum for
  each, and aggregate statistics (mean, max, chaotic/stable count).
  Results can be exported to CSV via :meth:`EnsembleResult.to_csv`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ── Fixed-point computation ─────────────────────────────────────────────────


def _fixed_point(alpha: float, beta: float, b: NDArray, a: NDArray) -> NDArray:
    """Compute the stable fixed point of the 4-D pole-filtered Hénon map."""
    H = np.sum(a[1:])  # sum of a coeffs (excluding a[0])
    G = np.sum(b)  # sum of b coeffs

    ratio = G / (1.0 - H)
    denom = ratio**2

    disc = (1.0 - beta) ** 2 + 4.0 * alpha * denom
    xf = ((beta - 1.0) + np.sqrt(disc)) / (2.0 * denom)

    vf = G * xf / (1.0 - H)
    return np.array([xf, xf, vf, vf])


# ── Jacobian ────────────────────────────────────────────────────────────────


def _jacobian(beta: float, b: NDArray, a: NDArray, x: NDArray) -> NDArray:
    """Jacobian of the 4-D pole-filtered Hénon map at state x."""
    return np.array(
        [
            [0.0, beta, -2.0 * x[2], 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, b[0] * beta, -2.0 * b[0] * x[2] + a[1], a[2]],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


# ── Map iteration ──────────────────────────────────────────────────────────


def _iterate(alpha: float, beta: float, b: NDArray, a: NDArray, x: NDArray) -> NDArray:
    """Single iteration of the 4-D pole-filtered Hénon map."""
    x1 = alpha - x[2] ** 2 + beta * x[1]
    x2 = x[0]
    x3 = b[0] * (alpha - x[2] ** 2 + beta * x[1]) + a[1] * x[2] + a[2] * x[3]
    x4 = x[2]
    return np.array([x1, x2, x3, x4])


# ── Gram-Schmidt orthogonalisation ─────────────────────────────────────────


def _gram_schmidt(Z: NDArray) -> tuple[NDArray, NDArray]:
    """Modified Gram-Schmidt. Returns (Q, norms)."""
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


# ── Lyapunov exponent ──────────────────────────────────────────────────────


def lyapunov_max(
    alpha: float = 1.4,
    beta: float = 0.3,
    Gz: float = 1.0,
    pole_radius: float = 0.975,
    w0: float = 0.0,
    Nitera: int = 2000,
    Ndiscard: int = 1000,
    perturbation: float = 0.1,
    seed: int = 42,
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
    a = -np.array([1.0, -2.0 * pole_radius * np.cos(w0), pole_radius**2])

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
    alpha: float = 1.4,
    beta: float = 0.3,
    Gz: float = 1.0,
    pole_radius: float = 0.975,
    w0: float = 0.0,
) -> dict:
    """Quick check: fixed point, eigenvalues, stability (4-D filtered)."""
    b = Gz * np.array([1.0, 0.0, 0.0])
    a = -np.array([1.0, -2.0 * pole_radius * np.cos(w0), pole_radius**2])
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
    return np.array(
        [
            [-2.0 * x[0], beta],
            [1.0, 0.0],
        ]
    )


def _henon2d_iterate(alpha: float, beta: float, x: NDArray) -> NDArray:
    """Single iteration of the 2-D Hénon map."""
    return np.array(
        [
            alpha - x[0] ** 2 + beta * x[1],
            x[0],
        ]
    )


def lyapunov_henon2d(
    alpha: float = 1.4,
    beta: float = 0.3,
    Nitera: int = 2000,
    Ndiscard: int = 1000,
    perturbation: float = 0.1,
    seed: int = 42,
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


# ═══════════════════════════════════════════════════════════════════════════
# Ensemble protocol: multiple ICs sampled around the fixed point
# ═══════════════════════════════════════════════════════════════════════════
#
# For the 4-D pole-filtered system the experimental protocol is:
#   1. Compute the analytic fixed point x_f.
#   2. Draw N_ci initial conditions uniformly in [x_f·(1−p), x_f·(1+p)],
#      independently per component.
#   3. For each IC, discard a transient of length Ndiscard and then run
#      the Lyapunov estimator over Nitera iterations.
#   4. Report per-IC exponents + λ_max, plus aggregated statistics.
#
# The 2-D variant uses the positive fixed point x_f^+ as the centre of
# the sampling box (matching the convention used in the TCC scripts).


@dataclass
class EnsembleResult:
    """Outcome of the ensemble Lyapunov protocol.

    Attributes
    ----------
    fixed_point : ndarray, shape (dim,)
        Fixed point used as the sampling centre.
    eigenvalues : ndarray, shape (dim,)
        Eigenvalues of the Jacobian at ``fixed_point``.
    stable : bool
        ``True`` iff ``all(|eigenvalues| < 1)``.
    initial_conditions : ndarray, shape (n_initial, dim)
        Initial conditions drawn for each run.
    exponents_per_ci : ndarray, shape (n_initial, dim)
        Full Lyapunov spectrum for each IC.
    lmax_per_ci : ndarray, shape (n_initial,)
        Largest Lyapunov exponent for each IC.
    mean_exponents : ndarray, shape (dim,)
        Per-component mean of ``exponents_per_ci``.
    mean_lmax, max_lmax : float
        Aggregates of ``lmax_per_ci``.
    n_chaotic, n_stable : int
        Counts of ICs with ``λ_max > 0`` and ``λ_max ≤ 0``.
    metadata : dict
        Free-form metadata (parameters used for the run).
    """

    fixed_point: NDArray
    eigenvalues: NDArray
    stable: bool
    initial_conditions: NDArray
    exponents_per_ci: NDArray
    lmax_per_ci: NDArray
    mean_exponents: NDArray
    mean_lmax: float
    max_lmax: float
    n_chaotic: int
    n_stable: int
    metadata: dict = field(default_factory=dict)

    # ── CSV export ─────────────────────────────────────────────────────

    def to_csv(self, path: str | Path) -> Path:
        """Write the per-IC table to CSV.

        Columns: ``ci, x0_0, x0_1, ..., lambda_1, lambda_2, ..., lmax, status``.
        Parent directories are created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dim = self.initial_conditions.shape[1]
        header = (
            ["ci"]
            + [f"x0_{k}" for k in range(dim)]
            + [f"lambda_{k + 1}" for k in range(dim)]
            + ["lmax", "status"]
        )

        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(self.lmax_per_ci)):
                status = "chaotic" if self.lmax_per_ci[i] > 0 else "stable"
                row = (
                    [i + 1]
                    + [f"{v:.10g}" for v in self.initial_conditions[i]]
                    + [f"{v:.10g}" for v in self.exponents_per_ci[i]]
                    + [f"{self.lmax_per_ci[i]:.10g}", status]
                )
                w.writerow(row)

            # Trailing blank line + aggregate summary block.
            w.writerow([])
            w.writerow(["# mean_exponents"] + [f"{v:.10g}" for v in self.mean_exponents])
            w.writerow(["# mean_lmax", f"{self.mean_lmax:.10g}"])
            w.writerow(["# max_lmax", f"{self.max_lmax:.10g}"])
            w.writerow(["# n_chaotic", self.n_chaotic])
            w.writerow(["# n_stable", self.n_stable])
        return path


# ── Sampling helper (shared between 2-D and 4-D variants) ──────────────────


def _sample_ics(
    fixed_point: NDArray,
    perturbation: float,
    n_initial: int,
    seed: int | None,
) -> NDArray:
    """Draw ``n_initial`` ICs uniformly in ``[fp·(1−p), fp·(1+p)]``.

    The sampling formula matches the TCC experimental protocol exactly:
    ``np.random.uniform(low=fp*(1-p), high=fp*(1+p), size=(n_initial, dim))``.
    It uses the legacy ``np.random.seed`` + ``np.random.uniform`` API so
    that results are byte-for-byte reproducible against the original
    standalone scripts, not just statistically similar.
    """
    if seed is not None:
        np.random.seed(seed)
    low = fixed_point * (1.0 - perturbation)
    high = fixed_point * (1.0 + perturbation)
    return np.random.uniform(low=low, high=high, size=(n_initial, len(fixed_point)))


# ── 4-D pole-filtered ensemble ─────────────────────────────────────────────


def lyapunov_max_ensemble(
    alpha: float = 1.4,
    beta: float = 0.3,
    Gz: float = 1.0,
    pole_radius: float = 0.975,
    w0: float = 0.0,
    Nitera: int = 2000,
    Ndiscard: int = 1000,
    perturbation: float = 0.1,
    n_initial: int = 20,
    seed: int | None = 42,
) -> EnsembleResult:
    """Ensemble Lyapunov protocol for the 4-D pole-filtered Hénon map.

    See :class:`EnsembleResult` for the output schema. Samples
    ``n_initial`` initial conditions uniformly in a box of half-width
    ``perturbation`` around the fixed point, then runs the single-IC
    Gram-Schmidt Lyapunov estimator on each.
    """
    dim = 4
    b = Gz * np.array([1.0, 0.0, 0.0])
    a = -np.array([1.0, -2.0 * pole_radius * np.cos(w0), pole_radius**2])

    xf = _fixed_point(alpha, beta, b, a)
    J_fp = _jacobian(beta, b, a, xf)
    eigs = np.linalg.eigvals(J_fp)
    stable = bool(np.all(np.abs(eigs) < 1.0))

    ics = _sample_ics(xf, perturbation, n_initial, seed)

    exponents_per_ci = np.empty((n_initial, dim))
    lmax_per_ci = np.empty(n_initial)

    for i in range(n_initial):
        x = ics[i].copy()
        for _ in range(Ndiscard):
            x = _iterate(alpha, beta, b, a, x)

        W = np.eye(dim)
        log_r = np.zeros((dim, Nitera))
        for k in range(Nitera):
            J = _jacobian(beta, b, a, x)
            Z = J @ W
            W, norms = _gram_schmidt(Z)
            for d in range(dim):
                log_r[d, k] = np.log(max(norms[d], 1e-300))
            x = _iterate(alpha, beta, b, a, x)

        exps = np.array([np.mean(log_r[d, :]) for d in range(dim)])
        exponents_per_ci[i] = exps
        lmax_per_ci[i] = float(np.max(exps))

    return EnsembleResult(
        fixed_point=xf,
        eigenvalues=eigs,
        stable=stable,
        initial_conditions=ics,
        exponents_per_ci=exponents_per_ci,
        lmax_per_ci=lmax_per_ci,
        mean_exponents=exponents_per_ci.mean(axis=0),
        mean_lmax=float(lmax_per_ci.mean()),
        max_lmax=float(lmax_per_ci.max()),
        n_chaotic=int(np.sum(lmax_per_ci > 0)),
        n_stable=int(np.sum(lmax_per_ci <= 0)),
        metadata={
            "system": "henon4d_pole_filtered",
            "alpha": alpha,
            "beta": beta,
            "Gz": Gz,
            "pole_radius": pole_radius,
            "w0": w0,
            "Nitera": Nitera,
            "Ndiscard": Ndiscard,
            "perturbation": perturbation,
            "n_initial": n_initial,
            "seed": seed,
        },
    )


# ── 2-D ensemble (positive fixed point, matching the TCC script) ──────────


def lyapunov_henon2d_ensemble(
    alpha: float = 1.4,
    beta: float = 0.3,
    Nitera: int = 2000,
    Ndiscard: int = 1000,
    perturbation: float = 0.1,
    n_initial: int = 20,
    seed: int | None = 42,
) -> EnsembleResult:
    """Ensemble Lyapunov protocol for the standard 2-D Hénon map.

    Samples around the *positive* fixed point (the one usually found in
    the strange attractor basin for α = 1.4, β = 0.3). See
    :class:`EnsembleResult` for the output schema.
    """
    dim = 2
    xf_p, xf_n = _henon2d_fixed_points(alpha, beta)

    J_p = _henon2d_jacobian(beta, xf_p)
    eigs_p = np.linalg.eigvals(J_p)
    stable_p = bool(np.all(np.abs(eigs_p) < 1.0))

    ics = _sample_ics(xf_p, perturbation, n_initial, seed)

    exponents_per_ci = np.empty((n_initial, dim))
    lmax_per_ci = np.empty(n_initial)

    for i in range(n_initial):
        x = ics[i].copy()
        for _ in range(Ndiscard):
            x = _henon2d_iterate(alpha, beta, x)

        W = np.eye(dim)
        log_r = np.zeros((dim, Nitera))
        for k in range(Nitera):
            J = _henon2d_jacobian(beta, x)
            Z = J @ W
            W, norms = _gram_schmidt(Z)
            for d in range(dim):
                log_r[d, k] = np.log(max(norms[d], 1e-300))
            x = _henon2d_iterate(alpha, beta, x)

        exps = np.array([np.mean(log_r[d, :]) for d in range(dim)])
        exponents_per_ci[i] = exps
        lmax_per_ci[i] = float(np.max(exps))

    return EnsembleResult(
        fixed_point=xf_p,
        eigenvalues=eigs_p,
        stable=stable_p,
        initial_conditions=ics,
        exponents_per_ci=exponents_per_ci,
        lmax_per_ci=lmax_per_ci,
        mean_exponents=exponents_per_ci.mean(axis=0),
        mean_lmax=float(lmax_per_ci.mean()),
        max_lmax=float(lmax_per_ci.max()),
        n_chaotic=int(np.sum(lmax_per_ci > 0)),
        n_stable=int(np.sum(lmax_per_ci <= 0)),
        metadata={
            "system": "henon2d_standard",
            "alpha": alpha,
            "beta": beta,
            "Nitera": Nitera,
            "Ndiscard": Ndiscard,
            "perturbation": perturbation,
            "n_initial": n_initial,
            "seed": seed,
            "fixed_point_n": xf_n.tolist(),  # negative fp, for reference
        },
    )

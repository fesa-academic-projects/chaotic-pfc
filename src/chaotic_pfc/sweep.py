"""
sweep.py
========
Parameter-sweep computation of Lyapunov exponents for the Hénon map with
an internal FIR filter.

This module sweeps the 2-D grid of filter orders ``N_s ∈ {2, …, 41}``
against normalised cutoffs ``ω_c ∈ (0, 1)`` and, for each grid point,
estimates the largest Lyapunov exponent ``λ_max`` of the corresponding
filtered Hénon system.

All hot paths are JIT-compiled with Numba. The outer 2-D loop is
flattened into a single ``prange`` so that every grid point is a task
available to the thread pool, which keeps the cores busy even when the
cost of different orders is uneven.

The public API is small:

- :class:`SweepResult` — container for the output of one sweep.
- :func:`precompute_fir_bank` — builds the FIR coefficient tensor.
- :func:`run_sweep` — full end-to-end computation for a (window, filter)
  pair.
- :func:`save_sweep` / :func:`load_sweep` — round-trip to ``.npz``.
- :data:`WINDOWS`, :data:`FILTER_TYPES`, :data:`WINDOW_DISPLAY_NAMES`
  — the catalogue of supported configurations.

Notes
-----
The inner kernels are split across two state-dimension regimes:

* ``_henon_n12_inplace`` / ``_lyap_online_n12`` — Ns ∈ {1, 2}, state
  dim = 2. A single kernel handles both tap counts via a tiny loop
  over ``c``; with Ns = 1 the second-tap term collapses to zero.
* ``_henon_nN_inplace`` / ``_lyap_online_nN`` — Ns ≥ 3, state dim = Ns
  (or 4 when Ns = 3, see the ``dim`` allocation in ``_sweep_kernel``).
  Generic N-tap routine with O(Ns²) structured ``J @ W``.

The dim-2 and dim-N regimes are kept separate because they exercise
different parts of the Jacobian and would otherwise need an extra
branch in every iteration of the hot loop.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.signal import firwin

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Catalogue of supported FIR windows and filter types
# ═══════════════════════════════════════════════════════════════════════════

WINDOWS: tuple[str, ...] = (
    "hamming",
    "hann",
    "blackman",
    "kaiser",
    "blackmanharris",
    "boxcar",
    "bartlett",
)

FILTER_TYPES: tuple[str, ...] = ("lowpass", "highpass")

WINDOW_DISPLAY_NAMES: dict[str, str] = {
    "hamming": "Hamming",
    "hann": "Hann",
    "blackman": "Blackman",
    "kaiser": "Kaiser",
    "blackmanharris": "Blackman-Harris",
    "boxcar": "Rectangular",
    "bartlett": "Bartlett",
}

# Kaiser window needs a β parameter (attenuation trade-off).
# β = 5.0 gives ~50 dB stop-band attenuation, a common default.
_KAISER_BETA: float = 5.0

# Adaptive Lyapunov: how often to evaluate the convergence criterion.
# Each "checkpoint" computes λ_max from the running lyap_sum, compares it
# to the previous checkpoint, and stops if the difference is below ``tol``
# for ``_ADAPTIVE_STREAK`` consecutive checkpoints. 100 is a sweet spot:
# small enough that early-stop fires close to the true convergence point,
# large enough that the eval overhead is negligible (~Ns ops per check vs
# Ns³ in the inner MGS loop).
_ADAPTIVE_CHECKPOINT_EVERY: int = 100
_ADAPTIVE_STREAK: int = 2


# ═══════════════════════════════════════════════════════════════════════════
# Hénon map variants — in-place, buffer-reusing, @njit (no fastmath)
# ═══════════════════════════════════════════════════════════════════════════
#
# Every iterate writes the next state into a pre-allocated ``x_new`` buffer
# instead of returning a fresh ndarray. The caller swaps ``x`` and ``x_new``
# by reference. Removing per-iteration allocations is critical: the sweep
# kernel calls these tens of millions of times.


@njit(cache=True)
def _henon_n12_inplace(x, x_new, alpha, beta, c):
    """1- or 2-tap filtered Hénon iterate (Ns ∈ {1, 2}). Writes into ``x_new``.

    Subsumes the previous ``_henon_n1`` / ``_henon_n2`` pair via a tiny
    loop over ``c``. With Ns = 1 the loop runs once and ``x[1]`` (which
    the caller initialises to 0) is unused; with Ns = 2 it runs twice.
    State dimension is 2 in both cases.

    Note: ``fastmath`` is intentionally **not** enabled. The Hénon map
    is chaotic; fastmath reassociations and division-approximations
    perturb the bit-exact arithmetic enough that long trajectories that
    used to overflow to Inf/NaN (triggering the divergence early-exit)
    instead stay finite, dramatically changing the set of "valid" grid
    points and the runtime profile.
    """
    Ns = len(c)
    z = 0.0
    for k in range(Ns):
        z += c[k] * x[k]
    x_new[0] = alpha - z * z + beta * x[1]
    x_new[1] = x[0]


@njit(cache=True)
def _henon_nN_inplace(x, x_new, alpha, beta, c):
    """General N-tap (N ≥ 3) filtered Hénon iterate. Writes into ``x_new``.

    Subsumes the previous ``_henon_n3``/``_henon_n4`` pair: with Ns=3
    the ``range(3, Ns)`` and ``range(4, Ns)`` loops are empty, and
    ``x_new[3] = x[1]`` is written into a scratch slot the caller
    allocates as part of the ``dim = max(Ns, 4)`` working buffer.
    The MGS in the Lyapunov kernel reads only rows ``[0, Ns)`` so this
    slot is never observed.
    """
    Ns = len(c)
    x_new[0] = alpha - x[2] ** 2 + beta * x[1]
    x_new[1] = x[0]
    head = c[0] * x_new[0] + c[1] * x[0] + c[2] * x[1]
    tail = 0.0
    for k in range(3, Ns):
        tail += c[k] * x[k]
    x_new[2] = head + tail
    x_new[3] = x[1]
    for k in range(4, Ns):
        x_new[k] = x[k - 1]


# ═══════════════════════════════════════════════════════════════════════════
# Lyapunov kernels — buffer-reusing, structure-aware J@W
# ═══════════════════════════════════════════════════════════════════════════
#
# Two algorithmic improvements over the previous version:
#
# 1. **No allocations in the hot loop.** Each task receives pre-allocated
#    work buffers (``w``, ``z``, ``x_buf``, ``x_buf2``, ``lyap_sum``). The
#    Jacobian is never materialised — its action on ``w`` is fused into
#    the loop that fills ``z``.
#
# 2. **Structured J @ W in O(Ns²) for n ≥ 4.** The Jacobian of the
#    filtered Hénon map is sparse (companion-like with a single dense
#    row at index 2). Computing ``J @ W`` row-by-row exploiting this
#    structure costs O(Ns²) instead of O(Ns³), which dominates the
#    per-iteration cost for large filter orders.
#
# The Modified Gram-Schmidt inner block is unchanged (manual loops, no
# BLAS overhead for tiny matrices). The kernels return the largest
# Lyapunov exponent ``λ_max = max_k (sum log_norm_k) / Nitera``.


@njit(cache=True, inline="always")
def _mgs_accumulate(z, Ns, lyap_sum):
    """Modified Gram-Schmidt + log-norm accumulation, in-place on ``z``."""
    for k in range(Ns):
        for p in range(k):
            dot = 0.0
            for row in range(Ns):
                dot += z[row, p] * z[row, k]
            for row in range(Ns):
                z[row, k] -= dot * z[row, p]
        nrm = 0.0
        for row in range(Ns):
            nrm += z[row, k] * z[row, k]
        nrm = nrm**0.5
        if nrm > 0.0:
            lyap_sum[k] += np.log(nrm)
            inv = 1.0 / nrm
            for row in range(Ns):
                z[row, k] *= inv


@njit(cache=True)
def _lyap_online_n12(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, c, alpha, beta):
    """Lyapunov kernel for Ns ∈ {1, 2}. State dimension is 2.

    See :func:`_lyap_online_nN` for the adaptive early-stop protocol;
    the logic is identical, only the kernel body differs.

    Subsumes the previous ``_lyap_online_n1``/``_lyap_online_n2`` pair.
    Let ``z = Σ c[k] · x[k]`` (over k < Ns). The Jacobian rows are:

        row 0: [-2 c[0] z,  β + (-2 c[1] z if Ns ≥ 2 else 0)]
        row 1: [1,          0]

    With Ns = 1 the contribution of ``c[1]`` collapses to zero and we
    recover the old n1 kernel exactly. With Ns = 2 we recover n2.
    """
    Ns_full = 2  # state dim is 2 for both n1 and n2
    for i in range(Ns_full):
        for j in range(Ns_full):
            ww[0, i, j] = 1.0 if i == j else 0.0
        lyap_sum[i] = 0.0

    c0 = c[0]
    c1 = c[1] if Ns >= 2 else 0.0

    adaptive = Nitera_min < Nitera
    prev_est = 0.0
    streak = 0
    n_used = Nitera

    tick = 0
    for it in range(Nitera):
        w_in = ww[tick]
        z_out = ww[1 - tick]
        x_in = xx[tick]
        x_out = xx[1 - tick]
        # z = c0*x0 (+ c1*x1 if Ns=2). For Ns=1, x[1] is the unused
        # second state slot; the n12 iterate ignores it, so its value
        # may carry stale data. We multiply by c1=0 above, which makes
        # the term vanish regardless of the buffer contents.
        z_lin = c0 * x_in[0] + c1 * x_in[1]
        a = -2.0 * c0 * z_lin
        b = -2.0 * c1 * z_lin + beta
        for col in range(Ns_full):
            z_out[0, col] = a * w_in[0, col] + b * w_in[1, col]
            z_out[1, col] = w_in[0, col]
        _mgs_accumulate(z_out, Ns_full, lyap_sum)
        _henon_n12_inplace(x_in, x_out, alpha, beta, c)
        tick = 1 - tick

        if adaptive:
            n_done = it + 1
            if n_done >= Nitera_min and (n_done - Nitera_min) % _ADAPTIVE_CHECKPOINT_EVERY == 0:
                cur = -1e30
                for k in range(Ns_full):
                    val = lyap_sum[k] / n_done
                    if val > cur:
                        cur = val
                if n_done > Nitera_min:
                    if abs(cur - prev_est) < tol:
                        streak += 1
                        if streak >= _ADAPTIVE_STREAK:
                            n_used = n_done
                            break
                    else:
                        streak = 0
                prev_est = cur

    best = -1e30
    for k in range(Ns_full):
        val = lyap_sum[k] / n_used
        if val > best:
            best = val
    return best, n_used


@njit(cache=True)
def _lyap_online_nN(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, c, alpha, beta):
    """N-tap (N ≥ 3) Lyapunov kernel with O(Ns²) structured J @ W.

    Returns
    -------
    (best, n_used) : (float, int)
        ``best`` is the largest Lyapunov exponent estimated from the
        running spectrum; ``n_used`` is the actual number of iterations
        performed (= Nitera in non-adaptive mode, possibly less when
        ``Nitera_min < Nitera`` and the convergence criterion fires).

    Adaptive early-stop
    -------------------
    When ``Nitera_min < Nitera``, after the ``Nitera_min``-th iteration
    we evaluate ``best = max_k(lyap_sum[k] / n)`` every
    ``_ADAPTIVE_CHECKPOINT_EVERY`` steps. If the change between
    consecutive checkpoints stays below ``tol`` for ``_ADAPTIVE_STREAK``
    checkpoints in a row, we conclude that the spectrum has stabilised
    and return early. When ``Nitera_min == Nitera`` (the default,
    non-adaptive path), the checkpoint logic is dead code and the loop
    runs to completion — preserving bit-exactness against the previous
    fixed-Nitera implementation.

    Subsumes the previous ``_lyap_online_n3``/``_lyap_online_n4`` pair.
    With Ns=3 the ``range(3, Ns)`` and ``range(4, Ns)`` loops are empty,
    making the body identical to the old ``n3`` kernel — except that
    we additionally write into ``z_out[3, col]``, which is a scratch
    slot reserved by the caller (see ``dim = max(Ns, 4)``). The MGS
    accumulator only reads rows ``[0, Ns)``, so the extra write is
    invisible to downstream computation.

    Jacobian structure (N = Ns):
        row 0:    [0, β, -2 x₂, 0, …, 0]              — 2 non-zeros
        row 1:    e_0
        row 2:    [c₁, c₀β+c₂, -2c₀x₂, c₃, c₄, …, c_{N-1}]  — dense
        row 3:    e_1
        row k≥4:  e_{k-1}

    Each output column of ``z = J @ w`` therefore costs O(Ns) flops for
    rows {0, 1, 3, 4, …, Ns-1} and O(Ns) for row 2 — total O(Ns²).

    Buffer protocol
    ---------------
    ``ww`` has shape ``(2, Ns, Ns)`` and ``xx`` has shape ``(2, dim)``.
    On entry the "current" state is ``xx[0]`` (the caller wrote the
    burn-in result there). The kernel ping-pongs between ``ww[0] / ww[1]``
    and ``xx[0] / xx[1]`` to avoid copying ``w <- z`` and ``x <- x_new``
    on every Lyapunov iteration.
    """
    # Initialise ww[0] = I (the input side at tick=0) and lyap_sum = 0
    for i in range(Ns):
        for j in range(Ns):
            ww[0, i, j] = 1.0 if i == j else 0.0
        lyap_sum[i] = 0.0

    c0 = c[0]
    c1 = c[1]
    c2 = c[2]
    cb02 = c0 * beta + c2  # row-2 coefficient at column 1

    # Adaptive state: previous checkpoint estimate and consecutive-stable
    # counter. When ``adaptive`` is False, ``Nitera_min == Nitera`` and
    # the inner ``if`` test never fires.
    adaptive = Nitera_min < Nitera
    prev_est = 0.0
    streak = 0
    n_used = Nitera

    tick = 0  # which slice currently holds w (input)
    for it in range(Nitera):
        w_in = ww[tick]
        z_out = ww[1 - tick]
        x_in = xx[tick]
        x_out = xx[1 - tick]

        m02 = -2.0 * x_in[2]
        m22 = -2.0 * c0 * x_in[2]
        # Compute z = J @ w one column at a time, exploiting the sparse rows.
        for col in range(Ns):
            w0 = w_in[0, col]
            w1 = w_in[1, col]
            w2 = w_in[2, col]
            # Dense row 2: c1*w0 + (c0β+c2)*w1 + (-2 c0 x2)*w2 + Σ_{k≥3} c[k]*w[k,col]
            row2 = c1 * w0 + cb02 * w1 + m22 * w2
            for k in range(3, Ns):
                row2 += c[k] * w_in[k, col]
            z_out[0, col] = beta * w1 + m02 * w2
            z_out[1, col] = w0
            z_out[2, col] = row2
            z_out[3, col] = w1
            for k in range(4, Ns):
                z_out[k, col] = w_in[k - 1, col]

        _mgs_accumulate(z_out, Ns, lyap_sum)

        # x_out = henon(x_in); next iteration reads from the "1-tick" side.
        _henon_nN_inplace(x_in, x_out, alpha, beta, c)
        tick = 1 - tick

        # Adaptive checkpoint: only after warmup, only at multiples.
        # The (it + 1) is because we have completed (it + 1) iterations.
        if adaptive:
            n_done = it + 1
            if n_done >= Nitera_min and (n_done - Nitera_min) % _ADAPTIVE_CHECKPOINT_EVERY == 0:
                cur = -1e30
                for k in range(Ns):
                    val = lyap_sum[k] / n_done
                    if val > cur:
                        cur = val
                if n_done > Nitera_min:  # need a previous checkpoint to compare
                    if abs(cur - prev_est) < tol:
                        streak += 1
                        if streak >= _ADAPTIVE_STREAK:
                            n_used = n_done
                            break
                    else:
                        streak = 0
                prev_est = cur

    best = -1e30
    for k in range(Ns):
        val = lyap_sum[k] / n_used
        if val > best:
            best = val
    return best, n_used


# ═══════════════════════════════════════════════════════════════════════════
# Flattened 2-D sweep kernel — one prange covers every (order, cutoff)
# ═══════════════════════════════════════════════════════════════════════════


def _build_task_order(orders_arr: NDArray, Ncut: int) -> NDArray:
    """Produce a permutation of ``[0, Ncoef * Ncut)`` that balances thread load.

    Numba's ``prange`` schedule is static: each thread receives a
    contiguous block of the iteration space. Per-task cost is dominated
    by the O(Ns³) Modified Gram-Schmidt; with ``Ns ∈ {3, …, 41}`` the
    heaviest task is ~700× more expensive than the cheapest, so a
    row-major ``flat = i * Ncut + j`` layout dumps all the heavy
    ``Ns ≈ 41`` tasks on the last thread, which then runs alone for
    minutes after the others finish.

    Algorithm: sort tasks by descending cost (``Ns³``), then deal them
    round-robin across ``Nt`` threads. Equivalent to interpreting the
    sorted list as an ``(Ncut · Ncoef // Nt + 1) × Nt`` matrix (filled
    row-by-row), transposing, and flattening — every thread gets
    "round 1's heaviest task", "round 2's heaviest task", etc., so all
    threads see the same total cost up to one task's worth.

    Determinism: the order is a deterministic function of the inputs
    (uses ``np.argsort(kind="stable")``), so byte-exact reproducibility
    under a fixed seed is preserved.
    """
    from numba import get_num_threads

    Ncoef = len(orders_arr)
    Ntot = Ncoef * Ncut
    Nt = max(1, get_num_threads())

    # Cost model: O(Ns³). Tasks with the same Ns get a stable tie-break
    # by flat index so the result is bit-exact reproducible.
    flat_idx = np.arange(Ntot, dtype=np.int64)
    i_of_flat = flat_idx // Ncut
    cost = orders_arr[i_of_flat].astype(np.float64) ** 3
    sorted_desc = np.argsort(-cost, kind="stable")  # heaviest first

    # Round-robin deal: pad to a multiple of Nt so reshape/transpose works.
    pad = (-Ntot) % Nt
    if pad:
        # Pad with a sentinel that we drop after transposing. Use Ntot
        # itself as the sentinel value; we filter it out below.
        sentinel = Ntot
        padded = np.empty(Ntot + pad, dtype=np.int64)
        padded[:Ntot] = sorted_desc
        padded[Ntot:] = sentinel
    else:
        padded = sorted_desc

    rounds = padded.size // Nt  # = ceil(Ntot / Nt)
    # Reshape (rounds, Nt) row-major → transpose → (Nt, rounds) → flatten.
    # After transpose: row t holds [task at round 0 for thread t, round 1, …].
    # Flatten in C-order: thread t's contiguous block in prange == row t.
    interleaved = padded.reshape(rounds, Nt).T.reshape(-1)
    if pad:
        interleaved = interleaved[interleaved != Ntot]
    return interleaved.astype(np.int64)


@njit(parallel=True, cache=True)
def _sweep_kernel(
    orders: NDArray,
    cutoffs: NDArray,
    fir_bank: NDArray,
    gains: NDArray,
    perturbations: NDArray,
    task_order: NDArray,
    Nitera: int,
    Nmap: int,
    Nmap_min: int,
    tol: float,
    n_initial: int,
    alpha: float,
    beta: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Parallel inner sweep. Returns (h, h_std, n_iters_mean), all (Ncoef, Ncut).

    ``perturbations`` has shape ``(Ncoef, Ncut, n_initial, max_Ns)`` and
    provides the stochastic seed values for every grid point. Generating
    them outside the kernel keeps the sweep deterministic under
    ``np.random.seed``: inside the ``prange`` loop, Numba uses a separate
    per-thread RNG state that does not honour the global seed, so calling
    ``np.random.rand`` here would make repeat runs byte-different.

    ``Nmap`` is the maximum number of Lyapunov iterations per IC.
    ``Nmap_min`` is the minimum; when ``Nmap_min < Nmap`` the inner
    Lyapunov kernels enable adaptive early-stop and may return after
    fewer iterations once the spectrum estimate stabilises within
    ``tol`` (see :func:`_lyap_online_nN` for the criterion). When
    ``Nmap_min == Nmap`` the early-stop branch never fires and behaviour
    is bit-identical to the pre-adaptive implementation.

    The third returned array, ``n_iters_mean``, holds the average
    number of Lyapunov iterations actually used per grid point (across
    its ``n_initial`` ICs). It equals ``Nmap`` everywhere in
    non-adaptive mode and is useful as a "difficulty map" of the
    parameter space in adaptive mode.

    ``task_order`` is a permutation of ``[0, Ntot)`` whose layout is
    chosen to balance work across threads. Numba's ``prange`` uses a
    static block schedule (each thread gets a contiguous slice of the
    iteration space). Per-task cost grows roughly as ``Ns³`` from the
    O(Ns³) Modified Gram-Schmidt inside the Lyapunov kernel, so a naive
    ``flat = i*Ncut + j`` layout assigns all heavy ``Ns ≈ 41`` tasks to
    the last thread, which then runs alone for minutes after the others
    finish. ``task_order`` interleaves heavy and light tasks across the
    static blocks so every thread gets a similar total cost.
    """
    Ncoef = len(orders)
    Ncut = len(cutoffs)
    Ntot = Ncoef * Ncut

    h = np.full((Ncoef, Ncut), np.nan)
    h_std = np.full((Ncoef, Ncut), np.nan)
    n_iters_mean = np.full((Ncoef, Ncut), np.nan)

    for tk in prange(Ntot):
        flat = task_order[tk]
        i = flat // Ncut
        j = flat % Ncut

        Ns = int(orders[i])
        c = fir_bank[i, j, :Ns]
        gain = gains[i, j]

        # Analytic fixed point (used to seed random ICs around it)
        p1 = 0.0
        p2 = 0.0
        p3 = 0.0
        if Ns >= 2 and gain != 0.0:
            disc = (1.0 - beta) ** 2 + 4.0 * alpha * (gain**2)
            p1 = (-(1.0 - beta) + disc**0.5) / (2.0 * gain**2)
            p2 = p1
            p3 = gain * p1

        # ``h_samples`` is filled NaN-first so that an early ``break``
        # on divergence leaves the trailing slots as NaN, which the
        # mean/std accumulator below ignores via ``if not np.isnan(v)``.
        # ``iters_samples`` parallels h_samples and tracks the actual
        # iteration count returned by the adaptive kernel for each IC.
        h_samples = np.full(n_initial, np.nan)
        iters_samples = np.zeros(n_initial, dtype=np.int64)
        diverged = False

        # Per-task work buffers: allocated once, reused across all ICs.
        # ``xx`` (2, dim) and ``ww`` (2, dim, dim) hold double buffers for
        # ping-pong: the kernel alternates which slice is "input" vs
        # "output" each iteration to avoid any per-iteration array copy.
        #
        # ``dim`` is the size of the working state buffer, which is not
        # always equal to Ns:
        #   Ns ∈ {1, 2}: state is 2-D (n1/n2 kernels operate on (x[0], x[1])).
        #   Ns == 3:     the generic n4 kernel writes into x_new[3] but only
        #                rows [0, Ns) are observed downstream. Allocate dim=4
        #                to give it a scratch slot.
        #   Ns >= 4:     dim = Ns, the natural state dimension.
        if Ns <= 2:
            dim = 2
        elif Ns == 3:
            dim = 4
        else:
            dim = Ns
        xx = np.empty((2, dim))
        ww = np.empty((2, dim, dim))
        lyap_sum = np.empty(dim)

        for ci in range(n_initial):
            # Pre-generated noise for this (i, j, ci); slice to Ns components.
            noise = perturbations[i, j, ci, :Ns]

            # Burn-in: ping-pong via ``tick`` indexing into xx[0] / xx[1].
            # After Nitera iterations the "current" state is in xx[Nitera & 1];
            # the Lyapunov kernel reads xx[0] as the initial state, so we
            # arrange the burn-in to land there.
            if Ns <= 2:
                # 1- or 2-tap branch (state dim = 2).
                if Ns == 1:
                    xx[0, 0] = 0.1 * noise[0]
                    xx[0, 1] = 0.0  # unused by n12 iterate, kept defined
                else:
                    xx[0, 0] = 0.1 * noise[0] + p1
                    xx[0, 1] = 0.1 * noise[1] + p2
                tick = 0
                for _ in range(Nitera):
                    _henon_n12_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
                    tick = 1 - tick
                if tick == 1:
                    xx[0, 0] = xx[1, 0]
                    xx[0, 1] = xx[1, 1]
                if np.isnan(xx[0, 0]) or np.isinf(xx[0, 0]):
                    diverged = True
                    break
                lam, n_used = _lyap_online_n12(
                    xx, ww, lyap_sum, Nmap, Nmap_min, tol, Ns, c, alpha, beta
                )
                h_samples[ci] = lam
                iters_samples[ci] = n_used

            else:
                # Ns >= 3: generic N-tap kernel.
                xx[0, 0] = 0.1 * noise[0] + p1
                xx[0, 1] = 0.1 * noise[1] + p2
                xx[0, 2] = 0.1 * noise[2] + p3
                for k in range(3, Ns):
                    xx[0, k] = 0.1 * noise[k] + p1
                tick = 0
                for _ in range(Nitera):
                    _henon_nN_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
                    tick = 1 - tick
                if tick == 1:
                    for k in range(Ns):
                        xx[0, k] = xx[1, k]
                if np.isnan(xx[0, 0]) or np.isinf(xx[0, 0]):
                    diverged = True
                    break
                lam, n_used = _lyap_online_nN(
                    xx, ww, lyap_sum, Nmap, Nmap_min, tol, Ns, c, alpha, beta
                )
                h_samples[ci] = lam
                iters_samples[ci] = n_used

        if not diverged:
            total = 0.0
            iters_total = 0
            count = 0
            for ci in range(n_initial):
                v = h_samples[ci]
                if not np.isnan(v):
                    total += v
                    iters_total += iters_samples[ci]
                    count += 1
            if count > 0:
                mean = total / count
                h[i, j] = mean
                var = 0.0
                for v in h_samples:
                    if not np.isnan(v):
                        var += (v - mean) ** 2
                h_std[i, j] = (var / count) ** 0.5
                n_iters_mean[i, j] = iters_total / count

    return h, h_std, n_iters_mean


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SweepResult:
    """Result of a single (window, filter) sweep.

    Attributes
    ----------
    h : ndarray, shape (Ncoef, Ncut)
        Mean of λ_max across ``n_initial`` random ICs, per grid point.
        NaN where the trajectory diverged for all ICs.
    h_std : ndarray, shape (Ncoef, Ncut)
        Standard deviation of λ_max samples at each grid point.
    orders : ndarray, shape (Ncoef,)
        Filter orders N_s actually swept.
    cutoffs : ndarray, shape (Ncut,)
        Normalised cutoff frequencies ω_c ∈ (0, 1) swept.
    window : str
        FIR window used (lower-case, e.g. ``"hamming"``).
    filter_type : {"lowpass", "highpass"}
        Filter pass-zero configuration.
    n_iters_used : ndarray, shape (Ncoef, Ncut), optional
        Average number of Lyapunov iterations actually used per grid
        point (across non-divergent ICs). In non-adaptive sweeps every
        finite cell equals ``Nmap``; in adaptive sweeps it ranges from
        ``Nmap_min`` to ``Nmap``, providing a "difficulty map" of the
        parameter space. ``None`` when loaded from a legacy ``.npz``
        without this field.
    metadata : dict
        Free-form metadata (simulation parameters, timing, etc.).
    """

    h: NDArray
    h_std: NDArray
    orders: NDArray
    cutoffs: NDArray
    window: str
    filter_type: str
    n_iters_used: NDArray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Human-readable name used for output directories."""
        pretty = WINDOW_DISPLAY_NAMES.get(self.window, self.window.capitalize())
        return f"{pretty} ({self.filter_type})"


# ───────────────────────────────────────────────────────────────────────────


def _precompute_perturbations(
    orders: NDArray,
    n_cutoffs: int,
    n_initial: int,
    seed: int | None,
) -> NDArray:
    """Build the deterministic perturbation tensor consumed by the kernel.

    Returns an array of shape ``(Ncoef, n_cutoffs, n_initial, max_Ns)``
    filled with uniform ``[0, 1)`` samples, where ``max_Ns = orders.max()``.
    Only the first ``Ns_i`` components along the last axis are actually
    used for each row ``i`` of ``orders``; the remainder is padding that
    the kernel never reads.

    Using :func:`np.random.seed` with an integer seed guarantees
    byte-identical output across runs, which propagates determinism to
    the whole sweep. Passing ``seed=None`` leaves the RNG state
    untouched (useful for warm-up runs whose output is discarded).
    """
    Ncoef = len(orders)
    max_Ns = int(orders.max()) if Ncoef > 0 else 0

    if seed is not None:
        np.random.seed(seed)

    return np.random.rand(Ncoef, n_cutoffs, n_initial, max_Ns)


# ───────────────────────────────────────────────────────────────────────────


def precompute_fir_bank(
    orders: Sequence[int] | NDArray,
    cutoffs: NDArray,
    filter_type: str,
    window: str,
    *,
    kaiser_beta: float = _KAISER_BETA,
) -> tuple[NDArray, NDArray]:
    """Build the FIR coefficient tensor and gain matrix for a sweep.

    Parameters
    ----------
    orders
        Filter orders to sweep. Values will be cast to ``int``.
    cutoffs
        Normalised cutoff frequencies in (0, 1).
    filter_type
        ``"lowpass"`` or ``"highpass"``.
    window
        Window name in :data:`WINDOWS`.
    kaiser_beta
        Shape parameter of the Kaiser window. Ignored unless
        ``window == "kaiser"``. Must be ``>= 0``.

    Returns
    -------
    fir_bank : ndarray, shape (Ncoef, Ncut, max_order)
        ``fir_bank[i, j, :order_i]`` holds the FIR coefficients for the
        ``(order_i, cutoff_j)`` pair, zero-padded on the right.
    gains : ndarray, shape (Ncoef, Ncut)
        DC gain (sum of coefficients) of each filter.

    Notes
    -----
    For highpass filters every order in ``orders`` must be odd. This is
    required by :func:`scipy.signal.firwin`: a highpass filter with
    ``pass_zero=False`` must have an odd number of taps. We enforce
    ``numtaps == Nss`` so that the kernel slice ``fir_bank[i, j, :Nss]``
    sees the full filter; allowing even highpass orders would require
    storing ``Nss + 1`` taps and silently truncate the filter at read
    time, breaking both the frequency response and the divergence-based
    early-exit in the sweep kernel.

    ``scipy.signal.firwin`` returns *exactly* ``numtaps`` coefficients,
    which differs from MATLAB's ``fir1`` by one — callers converting
    MATLAB code should pass ``numtaps = Nss`` (not ``Nss - 1``).
    """
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"filter_type must be one of {FILTER_TYPES}, got {filter_type!r}")
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    if window == "kaiser" and kaiser_beta < 0:
        raise ValueError(f"kaiser_beta must be >= 0, got {kaiser_beta!r}")

    orders_arr = np.asarray(orders, dtype=np.int64)
    if filter_type == "highpass":
        even = orders_arr[orders_arr % 2 == 0]
        if even.size:
            raise ValueError(
                "highpass requires odd orders; got even values "
                f"{even.tolist()}. Use np.arange(start|1, stop, 2) or "
                "filter your orders to odd integers."
            )
    Ncoef = len(orders_arr)
    Ncut = len(cutoffs)
    max_taps = int(orders_arr.max())

    fir_bank = np.zeros((Ncoef, Ncut, max_taps))
    gains = np.zeros((Ncoef, Ncut))

    win_arg = ("kaiser", float(kaiser_beta)) if window == "kaiser" else window

    for i, Nss in enumerate(orders_arr):
        numtaps = int(Nss)
        for j, wc in enumerate(cutoffs):
            c = firwin(numtaps, wc, pass_zero=filter_type, window=win_arg)
            fir_bank[i, j, :numtaps] = c.astype(np.float64)
            gains[i, j] = float(np.sum(c))

    return fir_bank, gains


# ───────────────────────────────────────────────────────────────────────────


def quick_sweep_params() -> tuple[NDArray, NDArray, NDArray, dict[str, int]]:
    """Return the (orders_lp, orders_hp, cutoffs, params) for quick-sweep mode.

    These are small grids suitable for testing and CI smoke runs.
    """
    import numpy as np

    return (
        np.arange(2, 8),
        np.arange(3, 9, 2),
        np.linspace(0.1, 0.9, 10),
        dict(Nitera=50, Nmap=200, n_initial=3),
    )


def run_sweep(
    window: str = "hamming",
    filter_type: str = "lowpass",
    *,
    orders: Sequence[int] | NDArray | None = None,
    cutoffs: NDArray | None = None,
    Nitera: int = 500,
    Nmap: int = 3000,
    n_initial: int = 25,
    alpha: float = 1.4,
    beta: float = 0.3,
    seed: int | None = 42,
    warmup: bool = True,
    kaiser_beta: float = _KAISER_BETA,
    adaptive: bool = False,
    Nmap_min: int = 500,
    tol: float = 1e-3,
) -> SweepResult:
    """Run a full Lyapunov sweep for one (window, filter) combination.

    Parameters
    ----------
    window, filter_type
        FIR configuration (see :data:`WINDOWS`, :data:`FILTER_TYPES`).
    orders
        Filter orders to sweep. Defaults to ``np.arange(2, 42)`` for
        lowpass and ``np.arange(3, 43, 2)`` for highpass (which requires
        odd tap counts).
    cutoffs
        Cutoff frequencies. Defaults to 100 points linearly spaced in
        ``(0, 1)``, with the endpoints nudged away from 0 and 1 to avoid
        ``firwin`` edge cases.
    Nitera
        Burn-in iterations applied to the map before starting the
        Lyapunov accumulation.
    Nmap
        Maximum number of iterations used to estimate each Lyapunov
        exponent. Also the exact iteration count when ``adaptive=False``.
    n_initial
        Number of random initial conditions averaged per grid point.
    alpha, beta
        Hénon parameters (α, β).
    seed
        Seed for the RNG used to shuffle initial conditions. ``None``
        leaves Numba's RNG state untouched.
    warmup
        Run a tiny 2×3 sweep first to trigger Numba compilation. This is
        recommended before timing a real sweep because the first call
        includes several seconds of JIT overhead.
    kaiser_beta
        Shape parameter of the Kaiser window. Ignored unless
        ``window == "kaiser"``.
    adaptive
        When ``True``, enable adaptive early-stop in the Lyapunov
        kernels: after ``Nmap_min`` iterations, the running estimate
        of λ_max is checked every 100 iterations and the loop exits
        early if it has stabilised within ``tol``. Default ``False``
        preserves bit-exactness with the pre-adaptive implementation.
        Calibration data on a representative grid (40×100, n_initial=25)
        suggests ``adaptive=True`` with ``Nmap_min=500``, ``tol=1e-3``
        gives a 2-3× speedup with mean |Δλ| < 0.001 vs the fixed-Nmap
        reference.
    Nmap_min
        Minimum iterations before the adaptive criterion may fire.
        Ignored when ``adaptive=False``. Smaller values give larger
        speedups but worse worst-case accuracy at fronteira points
        (|λ| ≈ 0). 500 is a safe default.
    tol
        Convergence tolerance for the adaptive criterion. The loop
        exits when ``|λ_t − λ_{t-1}| < tol`` for two consecutive
        checkpoints. Ignored when ``adaptive=False``.

    Returns
    -------
    SweepResult
        Full result object (see :class:`SweepResult`).
    """
    if adaptive:
        if Nmap_min < 1:
            raise ValueError(f"Nmap_min must be >= 1, got {Nmap_min}")
        if Nmap_min > Nmap:
            raise ValueError(f"Nmap_min ({Nmap_min}) must be <= Nmap ({Nmap})")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        # Inside the kernel, ``Nmap_min < Nmap`` is the gate that enables
        # the adaptive branch. If a caller asks for adaptive=True but
        # Nmap_min == Nmap, that would silently fall back to non-adaptive;
        # raise instead so the user notices.
        if Nmap_min == Nmap:
            raise ValueError(
                "adaptive=True with Nmap_min == Nmap is a no-op; "
                "set Nmap_min < Nmap or pass adaptive=False."
            )
        kernel_Nmap_min = Nmap_min
        kernel_tol = float(tol)
    else:
        # Disable the adaptive branch by setting Nmap_min == Nmap, which
        # makes ``adaptive = Nitera_min < Nitera`` evaluate False inside
        # the kernel. The tol value is unused in that path.
        kernel_Nmap_min = Nmap
        kernel_tol = 0.0
    if orders is None:
        # Highpass requires odd taps; use 20 odd orders to match the
        # cardinality of the lowpass default (np.arange(2, 42) → 40 pts,
        # but highpass cannot use even orders — see precompute_fir_bank).
        orders = np.arange(3, 43, 2) if filter_type == "highpass" else np.arange(2, 42)
    if cutoffs is None:
        cutoffs = np.linspace(1e-5, 1.0 - 1e-5, 100)

    orders_arr = np.asarray(orders, dtype=np.int64)
    cutoffs_arr = np.asarray(cutoffs, dtype=np.float64)

    fir_bank, gains = precompute_fir_bank(
        orders_arr,
        cutoffs_arr,
        filter_type,
        window,
        kaiser_beta=kaiser_beta,
    )

    # Pre-generate every perturbation array on the Python side so the
    # kernel is deterministic. The Numba prange uses per-thread RNG
    # states that do not honour np.random.seed, so drawing inside the
    # kernel would make repeat runs byte-different.
    perturbations = _precompute_perturbations(
        orders_arr,
        len(cutoffs_arr),
        n_initial,
        seed,
    )

    task_order = _build_task_order(orders_arr, len(cutoffs_arr))

    if warmup:
        # Trigger Numba compilation on a tiny slice. Output discarded.
        # Use a fresh perturbation slice so the warmup does not consume
        # the deterministic noise used by the real run.
        _warmup_perturbations = _precompute_perturbations(
            orders_arr[:2],
            3,
            2,
            seed=None,
        )
        _warmup_order = _build_task_order(orders_arr[:2], 3)
        _sweep_kernel(
            orders_arr[:2].astype(np.float64),
            cutoffs_arr[:3],
            fir_bank[:2, :3, :],
            gains[:2, :3],
            _warmup_perturbations,
            _warmup_order,
            10,  # Nitera
            50,  # Nmap
            50,  # Nmap_min == Nmap → non-adaptive (warmup compiles both paths)
            0.0,  # tol (unused)
            2,  # n_initial
            alpha,
            beta,
        )

    h, h_std, n_iters_used = _sweep_kernel(
        orders_arr.astype(np.float64),
        cutoffs_arr,
        fir_bank,
        gains,
        perturbations,
        task_order,
        Nitera,
        Nmap,
        kernel_Nmap_min,
        kernel_tol,
        n_initial,
        alpha,
        beta,
    )

    return SweepResult(
        h=h,
        h_std=h_std,
        orders=orders_arr,
        cutoffs=cutoffs_arr,
        window=window,
        filter_type=filter_type,
        n_iters_used=n_iters_used,
        metadata={
            "Nitera": Nitera,
            "Nmap": Nmap,
            "n_initial": n_initial,
            "alpha": alpha,
            "beta": beta,
            "seed": seed,
            "kaiser_beta": kaiser_beta,
            "adaptive": adaptive,
            "Nmap_min": Nmap_min if adaptive else None,
            "tol": tol if adaptive else None,
        },
    )


# ───────────────────────────────────────────────────────────────────────────


def save_sweep(result: SweepResult, path: str | Path) -> Path:
    """Save a :class:`SweepResult` to a compressed ``.npz``.

    The on-disk schema matches the legacy ``variables_lyapunov.npz``
    format (keys ``h``, ``h_desvio``, ``wcorte``, ``coef``) so older
    plotting scripts keep working, and adds ``window``, ``filter_type``,
    ``metadata`` and (optionally) ``n_iters_used`` for round-tripping
    into :class:`SweepResult`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "h": result.h,
        "h_desvio": result.h_std,
        "wcorte": result.cutoffs,
        "coef": result.orders,
        "window": result.window,
        "filter_type": result.filter_type,
        "metadata": np.array(list(result.metadata.items()), dtype=object),
    }
    if result.n_iters_used is not None:
        payload["n_iters_used"] = result.n_iters_used
    np.savez(path, **payload)  # type: ignore[arg-type]
    return path


def load_sweep(path: str | Path) -> SweepResult:
    """Load a sweep from ``.npz`` produced by :func:`save_sweep`.

    Also accepts the legacy format (only ``h``, ``h_desvio``, ``wcorte``,
    ``coef``), in which case ``window`` and ``filter_type`` are inferred
    from the parent directory name, e.g.
    ``data/sweeps/Hamming (lowpass)/variables_lyapunov.npz``. The
    ``n_iters_used`` field is optional; older saves load with
    ``n_iters_used=None``.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    if "window" in data.files and "filter_type" in data.files:
        window = str(data["window"])
        filter_type = str(data["filter_type"])
    else:
        window, filter_type = _infer_config_from_path(path)

    metadata: dict = {}
    if "metadata" in data.files:
        try:
            metadata = dict(data["metadata"].tolist())
        except (TypeError, ValueError):
            metadata = {}

    n_iters_used = data["n_iters_used"] if "n_iters_used" in data.files else None

    return SweepResult(
        h=data["h"],
        h_std=data["h_desvio"],
        orders=data["coef"],
        cutoffs=data["wcorte"],
        window=window,
        filter_type=filter_type,
        n_iters_used=n_iters_used,
        metadata=metadata,
    )


def _infer_config_from_path(path: Path) -> tuple[str, str]:
    """Infer (window, filter_type) from a directory name like
    ``Hamming (lowpass)``. Falls back to ("unknown", "lowpass") if the
    pattern does not match.
    """
    name = path.parent.name
    for key, pretty in WINDOW_DISPLAY_NAMES.items():
        for ft in FILTER_TYPES:
            if name == f"{pretty} ({ft})":
                return key, ft
    return "unknown", "lowpass"

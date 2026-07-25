"""Numba-JIT kernel functions for the Lyapunov sweep.

All hot paths are JIT-compiled. The inner kernels use pre-allocated
buffer arrays and avoid per-iteration allocations — critical for
performance when called tens of millions of times.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc._compat import njit, prange

from ._types import _ADAPTIVE_CHECKPOINT_EVERY, _ADAPTIVE_STREAK, _BENETTIN_K

# ═══════════════════════════════════════════════════════════════════════════
# Henon map variants — in-place, buffer-reusing, @njit (no fastmath)
# ═══════════════════════════════════════════════════════════════════════════
#
# Every iterate writes the next state into a pre-allocated ``x_new`` buffer
# instead of returning a fresh ndarray. The caller swaps ``x`` and ``x_new``
# by reference. Removing per-iteration allocations is critical: the sweep
# kernel calls these tens of millions of times.


@njit(cache=True)
def _henon_n12_inplace(x, x_new, alpha, beta, c):
    """1- or 2-tap filtered Henon iterate (Ns in {1, 2}). Writes into ``x_new``.

    Subsumes the previous ``_henon_n1`` / ``_henon_n2`` pair via a tiny
    loop over ``c``. With Ns = 1 the loop runs once and ``x[1]`` (which
    the caller initialises to 0) is unused; with Ns = 2 it runs twice.
    State dimension is 2 in both cases.

    Note: ``fastmath`` is intentionally **not** enabled. The Henon map
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
    """General N-tap (N >= 3) filtered Henon iterate. Writes into ``x_new``.

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
# The two Lyapunov estimator variants (_n12 for Ns ∈ {1, 2},
# _nN for Ns ≥ 3) share the same adaptive early-stop protocol.
# The common loop body (init → iterate → checkpoint → finalise) is
# factored into ``_lyap_online_core`` so that only the kernel-specific
# Jacobian-vector product and state update differ between the two.
# This eliminates ~60 duplicated lines and makes the adaptive logic
# testable independently of the map dimension.


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
        nrm = max(nrm**0.5, 1e-300)
        lyap_sum[k] += np.log(nrm)
        if nrm > 1e-300:
            inv = 1.0 / nrm
            for row in range(Ns):
                z[row, k] *= inv


# Adaptive early-stop: shared by both n12 and nN kernels.
@njit(cache=True, inline="always")
def _adaptive_checkpoint(lyap_sum, Ns, n_done, Nitera_min, tol, prev_est, streak, n_used):
    """Evaluate convergence at one checkpoint. Returns (prev_est, streak, n_used, should_break)."""
    cur = -1e30
    for k in range(Ns):
        val = lyap_sum[k] / n_done
        if val > cur:
            cur = val
    if n_done > Nitera_min:
        if abs(cur - prev_est) < tol:
            streak += 1
            if streak >= _ADAPTIVE_STREAK:
                return cur, streak, n_done, True
        else:
            streak = 0
    return cur, streak, n_used, False


@njit(cache=True)
def _lyap_online_core(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, _step_id, c, alpha, beta):
    """Shared adaptive Lyapunov loop with Benettin block reorthonormalisation.

    ``_step_id`` is 0 for the n12 kernel (Ns ∈ {1, 2}) or 1 for nN (Ns ≥ 3).

    The tangent vectors are propagated without MGS for ``_BENETTIN_K``
    iterations, then reorthonormalised once per block.  This reduces the
    O(Ns^3) MGS cost by a factor ~K while preserving the Lyapunov exponent
    estimates within early-stop tolerance (typically |Δλ| < 1e-3).
    See Benettin et al., Meccanica 15:9–20, 1980.
    """
    for i in range(Ns):
        for j in range(Ns):
            ww[0, i, j] = 1.0 if i == j else 0.0
        lyap_sum[i] = 0.0

    adaptive = Nitera_min < Nitera
    prev_est = 0.0
    streak = 0
    n_used = Nitera
    tick = 0

    for it in range(Nitera):
        # Propagate tangent vectors + map (J@W only, no MGS)
        if _step_id == 0:
            tick = _propagate_n12(tick, xx, ww, Ns, c, alpha, beta)
        else:
            tick = _propagate_nN(tick, xx, ww, Ns, c, alpha, beta)

        # O(1) state divergence check — every iteration.
        if not np.isfinite(xx[tick, 0]):
            for k in range(Ns):
                lyap_sum[k] = np.nan
            n_used = it + 1
            break

        # Benettin block: reorthonormalise every K iterations.
        # The ww scan and adaptive checkpoint are both aligned to multiples
        # of _BENETTIN_K (since _ADAPTIVE_CHECKPOINT_EVERY % K == 0), so
        # they land on MGS ticks and see an orthonormal slab.
        if (it + 1) % _BENETTIN_K == 0:
            _mgs_accumulate(ww[tick], Ns, lyap_sum)

            n_done = it + 1
            do_scan = (n_done % _ADAPTIVE_CHECKPOINT_EVERY == 0) or (
                adaptive
                and n_done >= Nitera_min
                and (n_done - Nitera_min) % _ADAPTIVE_CHECKPOINT_EVERY == 0
            )
            if do_scan:
                w_diverged = False
                w_slab = ww[tick]
                for w_i in range(Ns):
                    for w_j in range(Ns):
                        if not np.isfinite(w_slab[w_i, w_j]):
                            w_diverged = True
                            break
                    if w_diverged:
                        break
                if w_diverged:
                    for k in range(Ns):
                        lyap_sum[k] = np.nan
                    n_used = it + 1
                    break

            if (
                adaptive
                and n_done >= Nitera_min
                and (n_done - Nitera_min) % _ADAPTIVE_CHECKPOINT_EVERY == 0
            ):
                prev_est, streak, n_used, should_break = _adaptive_checkpoint(
                    lyap_sum, Ns, n_done, Nitera_min, tol, prev_est, streak, n_used
                )
                if should_break:
                    break

    # Final partial block: if the loop ended mid-block and the state did
    # not diverge, capture the growth of the unfinished block.
    # lyap_sum has size dim >= Ns; slots [Ns, dim) are never initialised
    # and may contain heap garbage. Only check the live Ns entries.
    if np.all(np.isfinite(lyap_sum[:Ns])) and n_used % _BENETTIN_K != 0:
        _mgs_accumulate(ww[tick], Ns, lyap_sum)

    best = np.nan
    for k in range(Ns):
        if np.isnan(lyap_sum[k]):
            break
    else:
        best = -1e30
        for k in range(Ns):
            val = lyap_sum[k] / n_used
            if val > best:
                best = val
        if not np.isfinite(best):
            best = np.nan
    return best, n_used


@njit(cache=True, inline="always")
def _propagate_n12(tick, xx, ww, Ns, c, alpha, beta):
    """Propagate tangent vectors + map for Ns ∈ {1, 2} (no MGS)."""
    Ns_full = 2
    w_in = ww[tick]
    z_out = ww[1 - tick]
    x_in = xx[tick]
    x_out = xx[1 - tick]

    c0 = c[0]
    c1 = c[1] if len(c) >= 2 else 0.0
    z_lin = c0 * x_in[0] + c1 * x_in[1]
    a = -2.0 * c0 * z_lin
    b_val = -2.0 * c1 * z_lin + beta
    for col in range(Ns_full):
        z_out[0, col] = a * w_in[0, col] + b_val * w_in[1, col]
        z_out[1, col] = w_in[0, col]
    _henon_n12_inplace(x_in, x_out, alpha, beta, c)
    return 1 - tick


@njit(cache=True, inline="always")
def _step_n12_ref(tick, xx, ww, lyap_sum, Ns, c, alpha, beta):
    """Per-iteration reference step (J@W + MGS + map) for Ns ∈ {1, 2}."""
    tick = _propagate_n12(tick, xx, ww, Ns, c, alpha, beta)
    _mgs_accumulate(ww[tick], 2, lyap_sum)
    return tick


@njit(cache=True)
def _lyap_online_n12(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, c, alpha, beta):
    """Lyapunov kernel for Ns ∈ {1, 2}. Delegates to :func:`_lyap_online_core`."""
    return _lyap_online_core(xx, ww, lyap_sum, Nitera, Nitera_min, tol, 2, 0, c, alpha, beta)


@njit(cache=True, inline="always")
def _propagate_nN(tick, xx, ww, Ns, c, alpha, beta):
    """Propagate tangent vectors + map for Ns ≥ 3 (no MGS)."""
    w_in = ww[tick]
    z_out = ww[1 - tick]
    x_in = xx[tick]
    x_out = xx[1 - tick]

    c0 = c[0]
    c1 = c[1]
    c2 = c[2]
    cb02 = c0 * beta + c2
    m02 = -2.0 * x_in[2]
    m22 = -2.0 * c0 * x_in[2]
    for col in range(Ns):
        w0 = w_in[0, col]
        w1 = w_in[1, col]
        w2 = w_in[2, col]
        row2 = c1 * w0 + cb02 * w1 + m22 * w2
        for k in range(3, Ns):
            row2 += c[k] * w_in[k, col]
        z_out[0, col] = beta * w1 + m02 * w2
        z_out[1, col] = w0
        z_out[2, col] = row2
        z_out[3, col] = w1
        for k in range(4, Ns):
            z_out[k, col] = w_in[k - 1, col]

    _henon_nN_inplace(x_in, x_out, alpha, beta, c)
    return 1 - tick


@njit(cache=True, inline="always")
def _step_nN_ref(tick, xx, ww, lyap_sum, Ns, c, alpha, beta):
    """Per-iteration reference step (J@W + MGS + map) for Ns ≥ 3."""
    tick = _propagate_nN(tick, xx, ww, Ns, c, alpha, beta)
    _mgs_accumulate(ww[tick], Ns, lyap_sum)
    return tick


@njit(cache=True)
def _lyap_online_nN(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, c, alpha, beta):
    """Lyapunov kernel for Ns ≥ 3. Delegates to :func:`_lyap_online_core`."""
    return _lyap_online_core(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, 1, c, alpha, beta)


def _build_task_order(orders_arr: NDArray, Ncut: int) -> NDArray:
    """Produce a permutation of ``[0, Ncoef * Ncut)`` that balances thread load.

    Numba's ``prange`` schedule is static: each thread receives a
    contiguous block of the iteration space. Per-task cost is dominated
    by the O(Ns^3) Modified Gram-Schmidt. Task order interleaves heavy
    and light tasks across the static blocks so every thread gets a
    similar total cost.
    """
    from chaotic_pfc._compat import get_num_threads

    Ncoef = len(orders_arr)
    Ntot = Ncoef * Ncut
    Nt = max(1, get_num_threads())

    flat_idx = np.arange(Ntot, dtype=np.int64)
    i_of_flat = flat_idx // Ncut
    cost = orders_arr[i_of_flat].astype(np.float64) ** 3
    sorted_desc = np.argsort(-cost, kind="stable")

    pad = (-Ntot) % Nt
    if pad:
        sentinel = Ntot
        padded = np.empty(Ntot + pad, dtype=np.int64)
        padded[:Ntot] = sorted_desc
        padded[Ntot:] = sentinel
    else:
        padded = sorted_desc

    rounds = padded.size // Nt
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
    """Parallel inner sweep. Returns (h, h_std, n_iters_mean), all (Ncoef, Ncut)."""
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

        # Fixed-point approximation for IC centring.
        #
        # The true fixed point of the N>=3 filtered Henon system has
        # state components that are not all equal when G != 1 (the
        # filter recurrence x_new[2] = c0*x_new[0] + ... forces
        # x[2] = G*x[0] at equilibrium).  We use the simplified
        # formula below — exact when G=1, approximate otherwise.
        # The perturbation spread (0.1 * |noise| per component) is
        # wide enough to cover the true fixed-point neighbourhood
        # in practice.  When G=0 the fixed point is ill-defined and
        # we fall through with p1=p2=p3=0.0.
        p1 = 0.0
        p2 = 0.0
        p3 = 0.0
        if abs(gain) > 1e-15:
            disc = (1.0 - beta) ** 2 + 4.0 * alpha * (gain**2)
            p1 = (-(1.0 - beta) + disc**0.5) / (2.0 * gain**2)
            p2 = p1
            p3 = gain * p1

        h_samples = np.full(n_initial, np.nan)
        iters_samples = np.zeros(n_initial, dtype=np.int64)
        diverged = False

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
            noise = perturbations[i, j, ci, :Ns]

            if Ns <= 2:
                if Ns == 1:
                    xx[0, 0] = 0.1 * noise[0] + p1
                    xx[0, 1] = 0.0
                else:
                    xx[0, 0] = 0.1 * noise[0] + p1
                    xx[0, 1] = 0.1 * noise[1] + p2
                tick = 0
                for it_t in range(Nitera):
                    _henon_n12_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
                    tick = 1 - tick
                    # Non-finite states are absorbing in the Henon map:
                    # once component 0 is Inf/NaN, subsequent iterates
                    # remain non-finite, and the post-loop check below
                    # catches it with the same outcome.
                    if (it_t & 63) == 63 and not np.isfinite(xx[tick, 0]):
                        break
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
                xx[0, 0] = 0.1 * noise[0] + p1
                xx[0, 1] = 0.1 * noise[1] + p2
                xx[0, 2] = 0.1 * noise[2] + p3
                for k in range(3, Ns):
                    xx[0, k] = 0.1 * noise[k] + p1
                tick = 0
                for it_t in range(Nitera):
                    _henon_nN_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
                    tick = 1 - tick
                    if (it_t & 63) == 63 and not np.isfinite(xx[tick, 0]):
                        break
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

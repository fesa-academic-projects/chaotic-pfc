"""Numba-JIT kernel functions for the Lyapunov sweep.

All hot paths are JIT-compiled. The inner kernels use pre-allocated
buffer arrays and avoid per-iteration allocations — critical for
performance when called tens of millions of times.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc._compat import njit, prange
from chaotic_pfc._formulas import fixed_point_positive as _fixed_point_positive

from ._types import (
    _ADAPTIVE_CHECKPOINT_EVERY,
    _ADAPTIVE_STREAK,
    _BENETTIN_K,
    _VONLY_MARGIN,
)

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
    """Full-spectrum reference MGS (all columns).

    This is the bit-exact reference used by the versioned sweeps: the fallback
    recomputation must reproduce these last bits. The NumPy variant
    ``chaotic_pfc.dynamics.lyapunov._gram_schmidt`` is the same algorithm for
    the validation wrappers but is not unified with this one on purpose —
    ``np.dot``/``np.linalg.norm`` may contract the sum in a different order, so
    sharing a single implementation would risk the bit-for-bit reproduction.
    """
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


@njit(cache=True, inline="always")
def _norm_accumulate_v(v, Ns, lyap_sum):
    """Single-vector norm + log accumulation (k=0 of MGS)."""
    nrm = 0.0
    for row in range(Ns):
        nrm += v[row] * v[row]
    nrm = max(nrm**0.5, 1e-300)
    lyap_sum[0] += np.log(nrm)
    if nrm > 1e-300:
        inv = 1.0 / nrm
        for row in range(Ns):
            v[row] *= inv


# Adaptive early-stop: shared by both n12 and nN kernels.
@njit(cache=True, inline="always")
def _adaptive_checkpoint(lyap_sum, Ns, n_use, n_done, Nitera_min, tol, prev_est, streak, n_used):
    """Evaluate convergence at one checkpoint. Returns (prev_est, streak, n_used, should_break).

    ``n_use`` is the number of leading ``lyap_sum`` columns folded into the
    running estimate: ``Ns`` for the max-over-spectrum modes and ``1`` for
    the single-column (col-0) mode.
    """
    cur = -1e30
    for k in range(n_use):
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
def _lyap_online_core(
    xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, _step_id, c, alpha, beta, n_use=0
):
    """Full-spectrum reference: shared adaptive Lyapunov loop with Benettin block.

    ``_step_id`` is 0 for the n12 kernel (Ns ∈ {1, 2}) or 1 for nN (Ns ≥ 3).

    ``n_use`` selects how many leading columns are reduced at the adaptive
    checkpoints and in the returned estimate: ``0`` (default) means ``Ns``,
    the maximum over the columns, and ``1`` means column 0 only.  Column 0 is
    the single-vector estimate of ``_lyap_online_core_v``: both propagate the
    same tangent vector and accumulate the same norms, so ``n_use=1`` gives
    that estimate in exact arithmetic.

    The tangent vectors are propagated without MGS for ``_BENETTIN_K``
    iterations, then reorthonormalised once per block.  In exact arithmetic
    this gives the same exponents as reorthonormalising at every iteration,
    and it reduces the O(Ns^3) MGS cost by a factor ~K.
    See Benettin et al., Meccanica 15:9–20, 1980.

    .. note::
       At finite horizon the maximum over columns can exceed column 0, because
       a column k > 0 may show a larger average growth over the run.  The
       difference between the maximum and column 0 is this finite-horizon
       effect, not a different exponent.  The sweep kernel uses the
       single-vector estimator and recomputes marginal cells with this kernel
       through the hybrid fallback (see ``_sweep_kernel``).
    """
    if n_use <= 0:
        n_use = Ns

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
                    lyap_sum, Ns, n_use, n_done, Nitera_min, tol, prev_est, streak, n_used
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
    for k in range(n_use):
        if np.isnan(lyap_sum[k]):
            break
    else:
        best = -1e30
        for k in range(n_use):
            val = lyap_sum[k] / n_used
            if val > best:
                best = val
        if not np.isfinite(best):
            best = np.nan
    return best, n_used


@njit(cache=True)
def _lyap_online_core_v(xx, vv, lyap_sum, Nitera, Nitera_min, tol, Ns, _step_id, c, alpha, beta):
    """Single-vector Lyapunov loop (tangent column 0 only), Benettin block K.

    Propagates one tangent vector and normalises it every ``_BENETTIN_K``
    iterations, accumulating the log-growth in ``lyap_sum[0]``.  The
    per-iteration cost is O(Ns) instead of O(Ns^2) for the full spectrum.

    In exact arithmetic this is column 0 of the full-spectrum kernel
    (``_lyap_online_core`` with ``n_use=1``).  The full-spectrum estimate used
    by the hybrid fallback takes the maximum over the columns, and at finite
    horizon a column k > 0 can show a larger average growth than column 0.
    The two estimates then differ by that finite-horizon effect.
    """
    for i in range(Ns):
        vv[0, i] = 1.0 if i == 0 else 0.0
    lyap_sum[0] = 0.0

    adaptive = Nitera_min < Nitera
    prev_est = 0.0
    streak = 0
    n_used = Nitera
    tick = 0

    for it in range(Nitera):
        if _step_id == 0:
            tick = _propagate_v_n12(tick, xx, vv, Ns, c, alpha, beta)
        else:
            tick = _propagate_v_nN(tick, xx, vv, Ns, c, alpha, beta)

        if not np.isfinite(xx[tick, 0]):
            lyap_sum[0] = np.nan
            n_used = it + 1
            break

        if (it + 1) % _BENETTIN_K == 0:
            _norm_accumulate_v(vv[tick], Ns, lyap_sum)

            n_done = it + 1
            do_scan = (n_done % _ADAPTIVE_CHECKPOINT_EVERY == 0) or (
                adaptive
                and n_done >= Nitera_min
                and (n_done - Nitera_min) % _ADAPTIVE_CHECKPOINT_EVERY == 0
            )
            if do_scan:
                v_slab = vv[tick]
                w_diverged = False
                for w_i in range(Ns):
                    if not np.isfinite(v_slab[w_i]):
                        w_diverged = True
                        break
                if w_diverged:
                    lyap_sum[0] = np.nan
                    n_used = it + 1
                    break

            if (
                adaptive
                and n_done >= Nitera_min
                and (n_done - Nitera_min) % _ADAPTIVE_CHECKPOINT_EVERY == 0
            ):
                prev_est, streak, n_used, should_break = _adaptive_checkpoint(
                    lyap_sum, 1, 1, n_done, Nitera_min, tol, prev_est, streak, n_used
                )
                if should_break:
                    break

    if np.isfinite(lyap_sum[0]) and n_used % _BENETTIN_K != 0:
        _norm_accumulate_v(vv[tick], Ns, lyap_sum)

    if np.isfinite(lyap_sum[0]):
        best = lyap_sum[0] / n_used
        if not np.isfinite(best):
            best = np.nan
    else:
        best = np.nan
    return best, n_used


@njit(cache=True, inline="always")
def _propagate_n12(tick, xx, ww, Ns, c, alpha, beta):
    """Full-spectrum reference: propagate tangent vectors + map for Ns ∈ {1, 2} (no MGS)."""
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
def _propagate_v_n12(tick, xx, vv, Ns, c, alpha, beta):
    """Single-vector propagate + map for Ns ∈ {1, 2} (col 0 only, no MGS)."""
    w_in = vv[tick]
    z_out = vv[1 - tick]
    x_in = xx[tick]
    x_out = xx[1 - tick]
    c0 = c[0]
    c1 = c[1] if len(c) >= 2 else 0.0
    z_lin = c0 * x_in[0] + c1 * x_in[1]
    a = -2.0 * c0 * z_lin
    b_val = -2.0 * c1 * z_lin + beta
    z_out[0] = a * w_in[0] + b_val * w_in[1]
    z_out[1] = w_in[0]
    _henon_n12_inplace(x_in, x_out, alpha, beta, c)
    return 1 - tick


@njit(cache=True, inline="always")
def _step_n12_ref(tick, xx, ww, lyap_sum, Ns, c, alpha, beta):
    """Per-iteration reference step (J@W + MGS + map) for Ns ∈ {1, 2}."""
    tick = _propagate_n12(tick, xx, ww, Ns, c, alpha, beta)
    _mgs_accumulate(ww[tick], 2, lyap_sum)
    return tick


@njit(cache=True)
def _lyap_online_n12(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, c, alpha, beta, n_use=0):
    """Lyapunov kernel for Ns ∈ {1, 2}. Delegates to :func:`_lyap_online_core`."""
    return _lyap_online_core(xx, ww, lyap_sum, Nitera, Nitera_min, tol, 2, 0, c, alpha, beta, n_use)


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
def _propagate_v_nN(tick, xx, vv, Ns, c, alpha, beta):
    """Single-vector propagate + map for Ns ≥ 3 (col 0 only, no MGS)."""
    w_in = vv[tick]
    z_out = vv[1 - tick]
    x_in = xx[tick]
    x_out = xx[1 - tick]
    c0 = c[0]
    c1 = c[1]
    c2 = c[2]
    cb02 = c0 * beta + c2
    m02 = -2.0 * x_in[2]
    m22 = -2.0 * c0 * x_in[2]
    w0 = w_in[0]
    w1 = w_in[1]
    w2 = w_in[2]
    row2 = c1 * w0 + cb02 * w1 + m22 * w2
    for k in range(3, Ns):
        row2 += c[k] * w_in[k]
    z_out[0] = beta * w1 + m02 * w2
    z_out[1] = w0
    z_out[2] = row2
    # Scratch slot: written for Ns≥3 (required for Ns≥4 shift chain).
    # For Ns=3 (dim=4) this value is written but never read.
    z_out[3] = w1
    for k in range(4, Ns):
        z_out[k] = w_in[k - 1]
    _henon_nN_inplace(x_in, x_out, alpha, beta, c)
    return 1 - tick


@njit(cache=True, inline="always")
def _step_nN_ref(tick, xx, ww, lyap_sum, Ns, c, alpha, beta):
    """Per-iteration reference step (J@W + MGS + map) for Ns ≥ 3."""
    tick = _propagate_nN(tick, xx, ww, Ns, c, alpha, beta)
    _mgs_accumulate(ww[tick], Ns, lyap_sum)
    return tick


@njit(cache=True)
def _lyap_online_nN(xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, c, alpha, beta, n_use=0):
    """Lyapunov kernel for Ns ≥ 3. Delegates to :func:`_lyap_online_core`."""
    return _lyap_online_core(
        xx, ww, lyap_sum, Nitera, Nitera_min, tol, Ns, 1, c, alpha, beta, n_use
    )


def _build_task_order(orders_arr: NDArray, Ncut: int) -> NDArray:
    """Produce a permutation of ``[0, Ncoef * Ncut)`` that balances thread load.

    Numba's ``prange`` schedule is static: each thread receives a
    contiguous block of the iteration space. Task order interleaves heavy
    and light tasks across the static blocks so every thread gets a
    similar total cost.

    .. note::
       With the single-vector estimator the per-iteration cost scales as
       O(Ns) instead of O(Ns^3); the cost heuristic below uses Ns directly.
    """
    from chaotic_pfc._compat import get_num_threads

    Ncoef = len(orders_arr)
    Ntot = Ncoef * Ncut
    Nt = max(1, get_num_threads())

    flat_idx = np.arange(Ntot, dtype=np.int64)
    i_of_flat = flat_idx // Ncut
    cost = orders_arr[i_of_flat].astype(np.float64)
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


@njit(cache=True)
def _burn_in_n12(xx, Nitera, alpha, beta, c):
    """Discard the Ns<=2 transient in place, leaving the state in row 0.

    Non-finite states are absorbing, so the loop stops early (checked every 64
    iterations); the post-condition check is left to the caller.
    """
    tick = 0
    for it_t in range(Nitera):
        _henon_n12_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
        tick = 1 - tick
        if (it_t & 63) == 63 and not np.isfinite(xx[tick, 0]):
            break
    if tick == 1:
        xx[0, 0] = xx[1, 0]
        xx[0, 1] = xx[1, 1]


@njit(cache=True)
def _burn_in_nN(xx, Ns, Nitera, alpha, beta, c):
    """Discard the Ns>=3 transient in place, leaving the state in row 0."""
    tick = 0
    for it_t in range(Nitera):
        _henon_nN_inplace(xx[tick], xx[1 - tick], alpha, beta, c)
        tick = 1 - tick
        if (it_t & 63) == 63 and not np.isfinite(xx[tick, 0]):
            break
    if tick == 1:
        for k in range(Ns):
            xx[0, k] = xx[1, k]


@njit(cache=True)
def _samples_summary(samples, iters_samples, n_initial):
    """Mean, std and mean iteration count over the finite samples.

    Returns ``(nan, nan, nan)`` when every sample is non-finite.
    """
    total = 0.0
    iters_total = 0
    count = 0
    for ci in range(n_initial):
        v = samples[ci]
        if not np.isnan(v):
            total += v
            iters_total += iters_samples[ci]
            count += 1
    if count == 0:
        return np.nan, np.nan, np.nan
    mean = total / count
    var = 0.0
    for v in samples:
        if not np.isnan(v):
            var += (v - mean) ** 2
    return mean, (var / count) ** 0.5, iters_total / count


@njit(parallel=True, cache=True)
def _sweep_kernel(
    num_taps_grid: NDArray,
    cutoff_grid: NDArray,
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
    fallback_mode: int = 1,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Parallel inner sweep. Returns (h, h_std, n_iters_mean, n_fallback), all (Ncoef, Ncut).

    ``fallback_mode`` selects the recomputation policy for marginal cells
    (|mean| < ``_VONLY_MARGIN``):

    * ``0`` — no recomputation (``n_fallback`` stays 0 everywhere).
    * ``1`` — recompute with the full-spectrum kernel and return the maximum
      over the spectrum (default; unchanged behaviour).
    * ``2`` — recompute with the full-spectrum kernel but return and adapt on
      column 0 only.

    Each cell is first computed with the single-vector estimator (O(Ns) per
    iteration). Under modes 1 and 2 marginal cells are recomputed with the
    full-spectrum kernel (O(Ns^2) per iteration) to eliminate the tiny
    probability of classification-sign flips near the transition boundary.
    """
    Ncoef = len(num_taps_grid)
    Ncut = len(cutoff_grid)
    Ntot = Ncoef * Ncut

    h = np.full((Ncoef, Ncut), np.nan)
    h_std = np.full((Ncoef, Ncut), np.nan)
    n_iters_mean = np.full((Ncoef, Ncut), np.nan)
    n_fallback = np.zeros((Ncoef, Ncut), dtype=np.int64)

    for tk in prange(Ntot):
        flat = task_order[tk]
        i = flat // Ncut
        j = flat % Ncut

        Ns = int(num_taps_grid[i])
        c = fir_bank[i, j, :Ns]
        gain = gains[i, j]

        # Fixed point used to centre the initial conditions.
        #
        # At equilibrium the filtered Henon system has x1 = x2 = p and
        # x[2] = G*p (the filter recurrence x_new[2] = c0*x_new[0] + ...
        # forces x[2] = G*x[0]), where p is the positive root of
        #     G^2 p^2 + (1 - beta) p - alpha = 0.
        # It is evaluated in a cancellation-free rationalised form so it
        # stays accurate as G -> 0, with limiting value alpha / (1 - beta)
        # at G = 0.  This form is exact for every G.
        p1 = _fixed_point_positive(gain, alpha, beta)
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
        vv = np.empty((2, dim))
        lyap_sum = np.empty(1)

        for ci in range(n_initial):
            noise = perturbations[i, j, ci, :Ns]

            if Ns <= 2:
                if Ns == 1:
                    xx[0, 0] = 0.1 * noise[0] + p1
                    xx[0, 1] = 0.0
                else:
                    xx[0, 0] = 0.1 * noise[0] + p1
                    xx[0, 1] = 0.1 * noise[1] + p2
                _burn_in_n12(xx, Nitera, alpha, beta, c)
                if np.isnan(xx[0, 0]) or np.isinf(xx[0, 0]):
                    diverged = True
                    break
                lam, n_used = _lyap_online_core_v(
                    xx, vv, lyap_sum, Nmap, Nmap_min, tol, 2, 0, c, alpha, beta
                )
                h_samples[ci] = lam
                iters_samples[ci] = n_used
            else:
                xx[0, 0] = 0.1 * noise[0] + p1
                xx[0, 1] = 0.1 * noise[1] + p2
                xx[0, 2] = 0.1 * noise[2] + p3
                for k in range(3, Ns):
                    xx[0, k] = 0.1 * noise[k] + p1
                _burn_in_nN(xx, Ns, Nitera, alpha, beta, c)
                if np.isnan(xx[0, 0]) or np.isinf(xx[0, 0]):
                    diverged = True
                    break
                lam, n_used = _lyap_online_core_v(
                    xx, vv, lyap_sum, Nmap, Nmap_min, tol, Ns, 1, c, alpha, beta
                )
                h_samples[ci] = lam
                iters_samples[ci] = n_used

        if not diverged:
            mean, std, iters_mean = _samples_summary(h_samples, iters_samples, n_initial)
            if not np.isnan(mean):
                h[i, j] = mean
                h_std[i, j] = std
                n_iters_mean[i, j] = iters_mean

                # Hybrid fallback: recompute marginal cells with full spectrum
                if fallback_mode != 0 and abs(mean) < _VONLY_MARGIN:
                    ww = np.empty((2, dim, dim))
                    lyap_sum_full = np.empty(dim)
                    full_samples = np.full(n_initial, np.nan)
                    full_iters = np.zeros(n_initial, dtype=np.int64)
                    n_use_full = 1 if fallback_mode == 2 else 0

                    for ci in range(n_initial):
                        noise = perturbations[i, j, ci, :Ns]

                        if Ns <= 2:
                            if Ns == 1:
                                xx[0, 0] = 0.1 * noise[0] + p1
                                xx[0, 1] = 0.0
                            else:
                                xx[0, 0] = 0.1 * noise[0] + p1
                                xx[0, 1] = 0.1 * noise[1] + p2
                            _burn_in_n12(xx, Nitera, alpha, beta, c)
                            if np.isnan(xx[0, 0]) or np.isinf(xx[0, 0]):
                                continue
                            lam, n_used = _lyap_online_n12(
                                xx,
                                ww,
                                lyap_sum_full,
                                Nmap,
                                Nmap_min,
                                tol,
                                2,
                                c,
                                alpha,
                                beta,
                                n_use_full,
                            )
                            full_samples[ci] = lam
                            full_iters[ci] = n_used
                        else:
                            xx[0, 0] = 0.1 * noise[0] + p1
                            xx[0, 1] = 0.1 * noise[1] + p2
                            xx[0, 2] = 0.1 * noise[2] + p3
                            for k in range(3, Ns):
                                xx[0, k] = 0.1 * noise[k] + p1
                            _burn_in_nN(xx, Ns, Nitera, alpha, beta, c)
                            if np.isnan(xx[0, 0]) or np.isinf(xx[0, 0]):
                                continue
                            lam, n_used = _lyap_online_nN(
                                xx,
                                ww,
                                lyap_sum_full,
                                Nmap,
                                Nmap_min,
                                tol,
                                Ns,
                                c,
                                alpha,
                                beta,
                                n_use_full,
                            )
                            full_samples[ci] = lam
                            full_iters[ci] = n_used

                    mean_f, std_f, iters_f = _samples_summary(full_samples, full_iters, n_initial)
                    if not np.isnan(mean_f):
                        h[i, j] = mean_f
                        h_std[i, j] = std_f
                        n_iters_mean[i, j] = iters_f

                    n_fallback[i, j] = 1

    return h, h_std, n_iters_mean, n_fallback

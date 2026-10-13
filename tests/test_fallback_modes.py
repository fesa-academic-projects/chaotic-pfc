"""Tests for the fixed-point formula and the ``fallback_mode`` policy of the sweep kernel.

The fixed-point expression is inlined in ``_sweep_kernel``, so the tests
repeat it here and check it against the positive root of
``G^2 p^2 + (1 - beta) p - alpha = 0`` and against the fixed point of
``_henon_nN_inplace``.
"""

import numpy as np
import pytest

from chaotic_pfc.analysis.sweep import load_sweep, precompute_fir_bank, run_sweep, save_sweep
from chaotic_pfc.analysis.sweep._kernel import (
    _adaptive_checkpoint,
    _fixed_point_positive,
    _henon_n12_inplace,
    _henon_nN_inplace,
    _lyap_online_core_v,
    _lyap_online_n12,
    _lyap_online_nN,
    _sweep_kernel,
)
from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

ALPHA, BETA = 1.4, 0.3
NITERA, NMAP = 500, 3000


def _dims(Ns):
    if Ns <= 2:
        return 2
    if Ns == 3:
        return 4
    return Ns


def _burn_in(noise, Ns, c, gain):
    """Replicates the initial state and burn-in of ``_sweep_kernel``.

    Returns the state (row 0 of a (2, dim) array) or None if the burn-in
    produced a non-finite state.
    """
    p1 = _fixed_point_positive(gain, ALPHA, BETA)
    p2, p3 = p1, gain * p1
    dim = _dims(Ns)
    xx = np.empty((2, dim))
    if Ns <= 2:
        xx[0, 0] = 0.1 * noise[0] + p1
        xx[0, 1] = 0.1 * noise[1] + p2 if Ns == 2 else 0.0
        step = _henon_n12_inplace
    else:
        xx[0, 0] = 0.1 * noise[0] + p1
        xx[0, 1] = 0.1 * noise[1] + p2
        xx[0, 2] = 0.1 * noise[2] + p3
        for k in range(3, Ns):
            xx[0, k] = 0.1 * noise[k] + p1
        step = _henon_nN_inplace
    tick = 0
    for _ in range(NITERA):
        step(xx[tick], xx[1 - tick], ALPHA, BETA, c)
        tick = 1 - tick
        if not np.isfinite(xx[tick, 0]):
            return None
    if tick == 1:
        xx[0, :] = xx[1, :]
    if not np.isfinite(xx[0, 0]):
        return None
    return xx


def _lam_vector(xx, Ns, c):
    ns_arg, step_id = (2, 0) if Ns <= 2 else (Ns, 1)
    lv = np.empty(1)
    vv = np.empty((2, _dims(Ns)))
    lam, _ = _lyap_online_core_v(
        xx.copy(), vv, lv, NMAP, NMAP, 0.0, ns_arg, step_id, c.copy(), ALPHA, BETA
    )
    return lam


def _lam_full_col0(xx, Ns, c):
    dim = _dims(Ns)
    ww = np.empty((2, dim, dim))
    lf = np.empty(dim)
    if Ns <= 2:
        lam, _ = _lyap_online_n12(xx.copy(), ww, lf, NMAP, NMAP, 0.0, Ns, c.copy(), ALPHA, BETA, 1)
    else:
        lam, _ = _lyap_online_nN(xx.copy(), ww, lf, NMAP, NMAP, 0.0, Ns, c.copy(), ALPHA, BETA, 1)
    return lam


def _seq_mean(values):
    """Mean skipping NaN, summed in the same order as the kernel."""
    total, count = 0.0, 0
    for v in values:
        if not np.isnan(v):
            total += v
            count += 1
    return total / count


# Marginal cell used by the hybrid-fallback tests (order 10, hamming lowpass).
def _marginal_cell():
    num_taps_grid = np.array([10], dtype=np.int64)
    cutoff_grid = np.array([0.212127])
    bank, gains = precompute_fir_bank(num_taps_grid, cutoff_grid, "lowpass", "hamming")
    pert = _precompute_perturbations(num_taps_grid, 1, 25, 42)
    return num_taps_grid, cutoff_grid, bank, gains, pert


def _marginal_sweep(fallback_mode=None):
    num_taps_grid, cutoff_grid, bank, gains, pert = _marginal_cell()
    task = np.zeros(1, dtype=np.int64)
    args = (
        num_taps_grid.astype(np.float64),
        cutoff_grid,
        bank,
        gains,
        pert,
        task,
        NITERA,
        NMAP,
        NMAP,
        0.0,
        25,
        ALPHA,
        BETA,
    )
    if fallback_mode is None:
        return _sweep_kernel(*args)
    return _sweep_kernel(*args, fallback_mode)


def _marginal_states():
    num_taps_grid, _, bank, gains, pert = _marginal_cell()
    Ns = int(num_taps_grid[0])
    c = bank[0, 0, :Ns]
    gain = gains[0, 0]
    states = [_burn_in(pert[0, 0, ci, :Ns], Ns, c, gain) for ci in range(25)]
    return Ns, c, states


# ---------------------------------------------------------------------------
# Fixed point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gain", [0.0, 1e-15, 1e-12, 1e-8, 1e-4, 0.5, 1.0, 2.0])
def test_fixed_point_is_positive_root(gain):
    p = _fixed_point_positive(gain, ALPHA, BETA)
    if gain == 0.0:
        ref = ALPHA / (1.0 - BETA)
    else:
        roots = np.roots([gain**2, 1.0 - BETA, -ALPHA])
        ref = max(r.real for r in roots if abs(r.imag) < 1e-14)
    np.testing.assert_allclose(p, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("gain", [0.0, 1e-15, 1e-12, 1e-8, 1e-4, 0.5, 1.0, 2.0])
def test_fixed_point_is_fixed_by_henon_map(gain):
    # 5-tap filter with DC gain equal to ``gain``.
    c = gain * np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    p = _fixed_point_positive(gain, ALPHA, BETA)
    x = np.array([p, p, gain * p, p, p])
    x_new = np.empty(5)
    _henon_nN_inplace(x, x_new, ALPHA, BETA, c)
    np.testing.assert_allclose(x_new, x, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Column 0 of the full-spectrum kernel versus the single-vector estimator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Ns", [2, 3, 5, 12])
def test_column0_equals_single_vector(Ns):
    from chaotic_pfc.analysis.sweep import precompute_fir_bank as bank_fn

    num_taps_grid = np.array([Ns], dtype=np.int64)
    cutoff_grid = np.array([0.3])
    bank, gains = bank_fn(num_taps_grid, cutoff_grid, "lowpass", "hamming")
    pert = _precompute_perturbations(num_taps_grid, 1, 25, 42)
    c = bank[0, 0, :Ns]
    gain = gains[0, 0]

    diffs = []
    for ci in range(25):
        xx = _burn_in(pert[0, 0, ci, :Ns], Ns, c, gain)
        if xx is None:
            continue
        diffs.append(abs(_lam_vector(xx, Ns, c) - _lam_full_col0(xx, Ns, c)))
    diffs = np.array(diffs)
    max_diff = float(diffs.max())
    print(f"Ns={Ns}: finite ICs={diffs.size}, max|col0 - single-vector| = {max_diff:.3e}")
    assert diffs.size > 0
    assert max_diff <= 1e-12


# ---------------------------------------------------------------------------
# fallback_mode in the kernel (marginal cell)
# ---------------------------------------------------------------------------


def test_default_fallback_mode_is_one():
    h_default, _, _, n_default = _marginal_sweep()
    h_one, _, _, n_one = _marginal_sweep(1)
    assert n_default[0, 0] == 1
    assert n_one[0, 0] == 1
    assert h_default[0, 0] == h_one[0, 0]


def test_mode0_skips_recomputation_and_returns_single_vector_mean():
    h0, _, _, n0 = _marginal_sweep(0)
    assert n0.sum() == 0

    Ns, c, states = _marginal_states()
    lam_v = [np.nan if xx is None else _lam_vector(xx, Ns, c) for xx in states]
    assert h0[0, 0] == _seq_mean(lam_v)


def test_mode2_recomputes_and_returns_column0_mean():
    h2, _, _, n2 = _marginal_sweep(2)
    assert n2[0, 0] == 1

    Ns, c, states = _marginal_states()
    lam_c0 = [np.nan if xx is None else _lam_full_col0(xx, Ns, c) for xx in states]
    assert h2[0, 0] == _seq_mean(lam_c0)


def test_adaptive_checkpoint_column_selection():
    # Columns >= n_use must not influence the checkpoint decision.
    base = np.array([100.0, 50.0, 25.0])
    other = np.array([100.0, 1.0e6, 1.0e6])
    tail = (0.0, 0, 500, 1e-3)
    res_a = _adaptive_checkpoint(base, 3, 1, 500, 500, *tail)
    res_b = _adaptive_checkpoint(other, 3, 1, 500, 500, *tail)
    assert res_a == res_b
    res_c = _adaptive_checkpoint(base, 3, 3, 500, 500, *tail)
    res_d = _adaptive_checkpoint(other, 3, 3, 500, 500, *tail)
    assert res_c != res_d


# ---------------------------------------------------------------------------
# run_sweep: validation, default and metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 3, 2.5])
def test_invalid_fallback_mode_raises(bad):
    with pytest.raises(ValueError, match="fallback_mode"):
        run_sweep(
            "hamming",
            "lowpass",
            num_taps_grid=np.array([3]),
            cutoff_grid=np.array([0.3]),
            Nitera=20,
            Nmap=50,
            n_initial=2,
            warmup=False,
            fallback_mode=bad,
        )


def _small_sweep(**kwargs):
    return run_sweep(
        "hamming",
        "lowpass",
        num_taps_grid=np.array([3, 4]),
        cutoff_grid=np.array([0.2, 0.6]),
        Nitera=50,
        Nmap=200,
        n_initial=3,
        warmup=False,
        **kwargs,
    )


def test_run_sweep_default_equals_explicit_mode_one():
    default = _small_sweep()
    explicit = _small_sweep(fallback_mode=1)
    np.testing.assert_array_equal(default.h, explicit.h)
    assert default.metadata["fallback_mode"] == 1
    assert explicit.metadata["fallback_mode"] == 1


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_fallback_mode_is_recorded_in_npz(mode, tmp_path):
    result = _small_sweep(fallback_mode=mode)
    assert result.metadata["fallback_mode"] == mode
    path = save_sweep(result, tmp_path / f"mode{mode}.npz")
    loaded = load_sweep(path)
    assert loaded.metadata["fallback_mode"] == mode

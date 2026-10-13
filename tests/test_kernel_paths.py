"""Tests for sweep-kernel paths that are easy to break silently.

Three behaviours from the manuscript are pinned here:

* the third step of the divergence rule — a cell whose initial conditions are
  mixed (some non-finite only after the burn-in) keeps the mean over the
  surviving initial conditions;
* the short symmetric Hann/Bartlett windows at ``N_s = 2`` produce non-finite
  coefficients and a divergent cell;
* the adaptive early-stop requires two consecutive stable checkpoints and never
  fires before iteration 500.
"""

from __future__ import annotations

import numpy as np
import pytest

from chaotic_pfc.analysis.sweep import precompute_fir_bank, run_sweep
from chaotic_pfc.analysis.sweep._kernel import (
    _adaptive_checkpoint,
    _fixed_point_positive,
    _henon_n12_inplace,
    _henon_nN_inplace,
    _lyap_online_core_v,
)
from chaotic_pfc.analysis.sweep._orchestration import _precompute_perturbations

ALPHA, BETA = 1.4, 0.3
NITERA, NMAP = 500, 3000
# Hann bandpass, N_s = 9 (N_z = 8), omega_c = 0.16: 24 of the 25 initial
# conditions survive the burn-in and only 24 yield a finite exponent.
PARTIAL = dict(window="hann", filter_type="bandpass", ns=9, cutoff=0.16)


def _dims(ns: int) -> int:
    if ns <= 2:
        return 2
    return 4 if ns == 3 else ns


def _per_ic_lambdas(window, filter_type, ns, cutoff, kaiser_beta=5.0):
    """Independent oracle: burn in + single-vector exponent per IC (NaN if it dies)."""
    bank, gains = precompute_fir_bank(
        np.array([ns]), np.array([cutoff]), filter_type, window, kaiser_beta=kaiser_beta
    )
    pert = _precompute_perturbations(np.array([ns]), 1, 25, 42)
    c = bank[0, 0, :ns]
    gain = gains[0, 0]
    p1 = _fixed_point_positive(gain, ALPHA, BETA)
    p2, p3 = p1, gain * p1
    dim = _dims(ns)
    lams = np.full(25, np.nan)
    for ci in range(25):
        noise = pert[0, 0, ci, :ns]
        xx = np.empty((2, dim))
        if ns <= 2:
            xx[0, 0] = 0.1 * noise[0] + p1
            xx[0, 1] = 0.1 * noise[1] + p2 if ns == 2 else 0.0
            step = _henon_n12_inplace
        else:
            xx[0, 0] = 0.1 * noise[0] + p1
            xx[0, 1] = 0.1 * noise[1] + p2
            xx[0, 2] = 0.1 * noise[2] + p3
            for k in range(3, ns):
                xx[0, k] = 0.1 * noise[k] + p1
            step = _henon_nN_inplace
        tick = 0
        diverged = False
        for _ in range(NITERA):
            step(xx[tick], xx[1 - tick], ALPHA, BETA, c)
            tick = 1 - tick
            if not np.isfinite(xx[tick, 0]):
                diverged = True
                break
        if diverged:
            continue
        if tick == 1:
            xx[0, :] = xx[1, :]
        if not np.isfinite(xx[0, 0]):
            continue
        vv = np.empty((2, dim))
        lv = np.empty(1)
        ns_arg, step_id = (2, 0) if ns <= 2 else (ns, 1)
        lam, _ = _lyap_online_core_v(
            xx.copy(), vv, lv, NMAP, NMAP, 0.0, ns_arg, step_id, c.copy(), ALPHA, BETA
        )
        if np.isfinite(lam):
            lams[ci] = lam
    return lams


def test_partial_divergence_averages_survivors():
    """Mixed ICs: the cell value is the mean over the non-divergent ones."""
    lams = _per_ic_lambdas(**PARTIAL)
    finite = np.isfinite(lams)
    assert 0 < int(finite.sum()) < 25, f"expected mixed divergence, got {finite.sum()}/25"
    expected = float(lams[finite].mean())

    res = run_sweep(
        window=PARTIAL["window"],
        filter_type=PARTIAL["filter_type"],
        num_taps_grid=[PARTIAL["ns"]],
        cutoff_grid=np.array([PARTIAL["cutoff"]]),
        Nitera=NITERA,
        Nmap=NMAP,
        n_initial=25,
        seed=42,
        warmup=False,
        fallback_mode=1,
    )
    assert np.isfinite(res.h[0, 0]), "the cell must stay finite when some ICs survive"
    np.testing.assert_allclose(res.h[0, 0], expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("window", ["hann", "bartlett"])
@pytest.mark.parametrize("filter_type", ["lowpass", "bandpass"])
def test_short_symmetric_window_is_divergent(window, filter_type):
    """N_s = 2 Hann/Bartlett taps are non-finite and the cell is divergent."""
    bank, gains = precompute_fir_bank(np.array([2]), np.array([0.3]), filter_type, window)
    assert not np.all(np.isfinite(bank[0, 0, :2]))
    assert not np.isfinite(gains[0, 0])

    res = run_sweep(
        window=window,
        filter_type=filter_type,
        num_taps_grid=[2],
        cutoff_grid=np.array([0.3]),
        Nitera=NITERA,
        Nmap=NMAP,
        n_initial=25,
        seed=42,
        warmup=False,
        fallback_mode=1,
    )
    assert np.isnan(res.h[0, 0])


def test_adaptive_checkpoint_needs_two_checks_and_waits_for_500():
    """The stop fires only on the second consecutive stable check, never at 500."""
    ns = 3
    lyap_sum = np.array([0.30, 0.10, 0.05])
    cur, streak, _, brk = _adaptive_checkpoint(lyap_sum, ns, ns, 500, 500, 1e-3, 0.0, 0, 500)
    assert not brk and streak == 0, "must not fire at n_done == Nitera_min"
    cur, streak, _, brk = _adaptive_checkpoint(lyap_sum, ns, ns, 600, 500, 1e-3, cur, streak, 600)
    assert not brk and streak == 1, "one stable check is not enough"
    _, streak, _, brk = _adaptive_checkpoint(lyap_sum, ns, ns, 700, 500, 1e-3, cur, streak, 700)
    assert brk and streak == 2, "two consecutive stable checks must break"


def test_adaptive_checkpoint_resets_streak_when_estimate_moves():
    ns = 3
    lyap_sum = np.array([0.30, 0.10, 0.05])
    cur, streak, _, brk = _adaptive_checkpoint(lyap_sum, ns, ns, 600, 500, 1e-3, 0.0, 0, 600)
    assert streak == 1 and not brk
    moved = np.array([0.90, 0.10, 0.05])
    _, streak, _, brk = _adaptive_checkpoint(moved, ns, ns, 700, 500, 1e-9, cur, streak, 700)
    assert not brk and streak == 0, "a moved estimate must reset the streak"

"""User-supplied FIR banks and designs in ``run_sweep``.

The sweep accepts a pre-computed coefficient bank or a design callable in place
of the built-in window catalogue. A custom bank that equals the catalogue must
reproduce the catalogue sweep bit for bit; a design callable is checked against
the same bank; and malformed input is rejected with a clear message.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import firwin

from chaotic_pfc.analysis.sweep import precompute_fir_bank, run_sweep

ORDERS = np.array([3, 5])
CUTOFFS = np.array([0.2, 0.4])
KERNEL = dict(Nitera=50, Nmap=200, n_initial=3, seed=1, adaptive=False, warmup=False)


def _catalogue() -> np.ndarray:
    bank, _ = precompute_fir_bank(ORDERS, CUTOFFS, "lowpass", "hamming")
    return bank


def test_custom_fir_bank_reproduces_the_catalogue() -> None:
    reference = run_sweep(
        window="hamming", filter_type="lowpass", num_taps_grid=ORDERS, cutoff_grid=CUTOFFS, **KERNEL
    )
    custom = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=ORDERS,
        cutoff_grid=CUTOFFS,
        fir_bank=_catalogue(),
        **KERNEL,
    )
    assert np.array_equal(reference.h, custom.h, equal_nan=True)
    assert np.array_equal(reference.h_std, custom.h_std, equal_nan=True)


def test_custom_fir_design_matches_the_bank() -> None:
    def design(num_taps: int, cutoff: float) -> np.ndarray:
        return firwin(num_taps, cutoff, pass_zero="lowpass", window="hamming")

    by_design = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=ORDERS,
        cutoff_grid=CUTOFFS,
        fir_design=design,
        **KERNEL,
    )
    by_bank = run_sweep(
        window="hamming",
        filter_type="lowpass",
        num_taps_grid=ORDERS,
        cutoff_grid=CUTOFFS,
        fir_bank=_catalogue(),
        **KERNEL,
    )
    assert np.array_equal(by_design.h, by_bank.h, equal_nan=True)


def test_bad_fir_bank_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="fir_bank must have shape"):
        run_sweep(
            num_taps_grid=ORDERS,
            cutoff_grid=CUTOFFS,
            fir_bank=np.zeros((2, 2, 4)),
            **KERNEL,
        )


def test_bad_fir_design_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="fir_design"):
        run_sweep(
            num_taps_grid=ORDERS,
            cutoff_grid=CUTOFFS,
            fir_design=lambda num_taps, cutoff: np.zeros(num_taps + 1),
            **KERNEL,
        )


def test_fir_bank_and_design_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        run_sweep(
            num_taps_grid=ORDERS,
            cutoff_grid=CUTOFFS,
            fir_bank=_catalogue(),
            fir_design=lambda num_taps, cutoff: np.zeros(num_taps),
            **KERNEL,
        )

"""Regression tests for the Lyapunov sweep kernel.

These tests pin behaviours that are otherwise easy to break silently:
the fixed-point centring (cancellation-free form), the column-0
equivalence between the full-spectrum and single-vector kernels, and a
frozen-fixture baseline of ``fallback_mode=1`` against the versioned
``data/sweeps`` checkpoints.

The baseline test (``test_mode1_bitexact_baseline``) is opt-in: it
compares against the ``.npz`` checkpoints regenerated with the current
kernel, so it must reproduce them bit for bit.  Run it with
``CHAOTIC_PFC_RUN_BASELINE=1``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from chaotic_pfc.analysis.sweep import load_sweep, run_sweep

# The 4 frozen-fixture configurations used for the baseline check.
_BASELINE_CONFIGS = [
    ("kaiser", "lowpass", 13.5),
    ("kaiser", "highpass", 11.5),
    ("kaiser", "bandpass", 13.5),
    ("kaiser", "bandstop", 4.0),
]


def _nan_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-exact array equality treating NaN as equal to NaN."""
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(
        a[~np.isnan(a)], b[~np.isnan(b)]
    )


def _baseline_path(window: str, filter_type: str, kaiser_beta: float) -> Path:
    return (
        Path("data/sweeps/kaiser")
        / filter_type
        / f"beta_{kaiser_beta:.2f}"
        / "variables_lyapunov.npz"
    )


@pytest.mark.skipif(
    os.environ.get("CHAOTIC_PFC_RUN_BASELINE") != "1",
    reason="frozen-fixture baseline; run with CHAOTIC_PFC_RUN_BASELINE=1",
)
@pytest.mark.parametrize(("window", "filter_type", "kaiser_beta"), _BASELINE_CONFIGS)
def test_mode1_bitexact_baseline(window, filter_type, kaiser_beta):
    """fallback_mode=1 must reproduce the versioned .npz bit-for-bit."""
    path = _baseline_path(window, filter_type, kaiser_beta)
    old = load_sweep(path)
    meta = old.metadata

    result = run_sweep(
        window=window,
        filter_type=filter_type,
        kaiser_beta=kaiser_beta,
        bandwidth=0.05,
        Nitera=meta["Nitera"],
        Nmap=meta["Nmap"],
        n_initial=meta["n_initial"],
        seed=meta["seed"],
        adaptive=meta["adaptive"],
        Nmap_min=meta["Nmap_min"],
        tol=meta["tol"],
        fallback_mode=1,
        warmup=True,
    )

    assert _nan_equal(result.h, old.h), f"h differs for {window}/{filter_type}/{kaiser_beta}"
    assert _nan_equal(result.h_std, old.h_std)
    assert _nan_equal(result.n_iters_used, old.n_iters_used)
    assert result.metadata["n_fallback"] == meta["n_fallback"]

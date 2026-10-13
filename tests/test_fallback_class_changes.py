"""Class stability across the ``fallback_mode`` policy.

The hybrid estimator can (``1``, ``2``) or cannot (``0``) recompute marginal
cells (``|lambda_1| < 5e-3``) with the full spectrum. This module runs the four
baseline configurations of the manuscript under all three modes on a reduced
grid and records how many grid cells change class (chaotic / periodic /
divergent). The counts are quoted in the documentation as evidence
that the fallback does not move the classification.

The full-resolution version of this measurement is the opt-in bit-exact
baseline (``CHAOTIC_PFC_RUN_BASELINE=1``); here the grid is reduced so the
comparison stays in the normal suite. The test still requires the fallback to
fire, so a mode-1 run that never recomputed anything would fail the premise.
"""

from __future__ import annotations

import numpy as np
import pytest

from chaotic_pfc.analysis.sweep import run_sweep

# The four baseline configurations (the best Kaiser beta per filter type).
BASELINE_CONFIGS = (
    ("lowpass", 13.5),
    ("highpass", 11.5),
    ("bandpass", 13.5),
    ("bandstop", 4.0),
)
CUTOFFS = np.array([0.1, 0.3, 0.5, 0.7])
ORDERS_BAND = np.array([3, 5, 9, 11])  # odd taps required by highpass/bandstop
ORDERS_PLAIN = np.array([2, 5, 8, 11])


def _classes(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return np.where(np.isnan(h), -1, np.where(h > 0, 1, 0))


def _sweep(filter_type: str, kaiser_beta: float, fallback_mode: int):
    orders = ORDERS_BAND if filter_type in ("highpass", "bandstop") else ORDERS_PLAIN
    return run_sweep(
        window="kaiser",
        filter_type=filter_type,
        kaiser_beta=kaiser_beta,
        orders=orders,
        cutoffs=CUTOFFS,
        Nitera=100,
        Nmap=800,
        n_initial=6,
        adaptive=False,
        seed=42,
        fallback_mode=fallback_mode,
    )


@pytest.mark.parametrize(("filter_type", "kaiser_beta"), BASELINE_CONFIGS)
def test_fallback_mode_preserves_the_classification(filter_type: str, kaiser_beta: float) -> None:
    reference = _sweep(filter_type, kaiser_beta, fallback_mode=1)
    ref_classes = _classes(reference.h)
    for mode in (0, 2):
        other = _sweep(filter_type, kaiser_beta, fallback_mode=mode)
        changes = int(np.count_nonzero(_classes(other.h) != ref_classes))
        assert changes == 0, (
            f"{filter_type}/kaiser(beta={kaiser_beta}): fallback_mode={mode} changed the "
            f"class of {changes} grid cells relative to mode 1"
        )


def test_fallback_fires_on_the_reduced_grid() -> None:
    """The comparison above is only meaningful if the fallback actually runs.

    Across the four baseline configurations the reduced grid must contain at
    least one marginal cell; otherwise the three modes would agree trivially.
    """
    counts = {
        f"{filter_type}/kaiser(beta={kaiser_beta})": int(
            _sweep(filter_type, kaiser_beta, 1).metadata["n_fallback"]
        )
        for filter_type, kaiser_beta in BASELINE_CONFIGS
    }
    assert any(count > 0 for count in counts.values()), (
        f"no marginal cells anywhere in the reduced comparison grid "
        f"(n_fallback = {counts}); the class-stability check is vacuous"
    )

"""Unit tests for the classification labels of ``paper.numbers.section_maps``.

The lowpass labels must use the manuscript criterion ``h > theta`` (theta =
``LAMBDA_CHAOS_THRESHOLD``), not merely ``h > 0``: finite-time estimates below
the threshold are non-chaotic and must not create a chaotic boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from chaotic_pfc.paper.numbers import section_maps


def _sweep(h: list[list[float]], taps: list[int], cutoffs: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        h=np.asarray(h, dtype=np.float64),
        num_taps_grid=np.asarray(taps, dtype=np.int64),
        cutoff_grid=np.asarray(cutoffs, dtype=np.float64),
    )


def test_lowpass_labels_use_chaos_threshold() -> None:
    """Positive estimates below theta are not chaotic."""
    cutoffs = [0.0, 0.5, 1.0]
    lowpass = _sweep(
        [
            [0.001, 0.001, 0.001],
            [0.001, 0.020, 0.005],
            [0.001, 0.001, 0.001],
        ],
        taps=[2, 3, 4],
        cutoffs=cutoffs,
    )
    best = {
        "lowpass": [{"sr": lowpass}],
        "highpass": [{"sr": _sweep([[0.010, 0.001, 0.001]], taps=[5], cutoffs=cutoffs)}],
        "bandpass": [{"sr": _sweep([[0.010, 0.001, 0.001]], taps=[2], cutoffs=cutoffs)}],
        "bandstop": [{"sr": _sweep([[0.010, 0.001, 0.001]], taps=[2], cutoffs=cutoffs)}],
    }

    out: dict[str, str] = {}
    section_maps(best, out)

    assert out["MAP.lowpass.max_wc_by_Nz"] == "2:0.50"
    assert out["MAP.lowpass.Nz_without_chaos"] == "1, 3"

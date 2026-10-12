"""Frequency-response checks for the window-designed FIR filters (article D1d).

The four filter types are built with the same helper the sweeps use and their
transfer function is evaluated with :func:`scipy.signal.freqz`.  Each type must
pass its passband and reject its stopband, with tolerances consistent with the
window method.  A final check confirms that a low-side-lobe window attenuates
more than a rectangular one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz

from chaotic_pfc.analysis.sweep import precompute_fir_bank

NUMTAPS = 41
CUTOFF = 0.4
BANDWIDTH = 0.2


def _mag(filter_type: str, window: str = "hamming") -> tuple[np.ndarray, np.ndarray]:
    bank, _ = precompute_fir_bank(
        np.array([NUMTAPS]), np.array([CUTOFF]), filter_type, window, bandwidth=BANDWIDTH
    )
    c = bank[0, 0, :NUMTAPS]
    w, h = freqz(c, worN=8192)
    return w / np.pi, np.abs(h)


def _at(freq: np.ndarray, mag: np.ndarray, x: float) -> float:
    return float(np.interp(x, freq, mag))


@pytest.mark.parametrize("window", ["hamming", "blackmanharris"])
def test_lowpass_passes_dc_and_stops_high(window):
    f, h = _mag("lowpass", window)
    assert _at(f, h, 0.05) > 0.9
    assert _at(f, h, 0.9) < 0.05


def test_highpass_stops_low_and_passes_high():
    f, h = _mag("highpass")
    assert _at(f, h, 0.9) > 0.9
    assert _at(f, h, 0.05) < 0.05


def test_bandpass_passes_the_band_only():
    f, h = _mag("bandpass")
    assert _at(f, h, CUTOFF) > 0.9
    assert _at(f, h, 0.1) < 0.05
    assert _at(f, h, 0.8) < 0.05


def test_bandstop_rejects_the_band_only():
    f, h = _mag("bandstop")
    assert _at(f, h, CUTOFF) < 0.05
    assert _at(f, h, 0.1) > 0.9
    assert _at(f, h, 0.8) > 0.9


def test_lower_side_lobes_give_a_deeper_stopband():
    f_rect, h_rect = _mag("lowpass", "boxcar")
    f_bh, h_bh = _mag("lowpass", "blackmanharris")
    assert _at(f_bh, h_bh, 0.7) < _at(f_rect, h_rect, 0.7)

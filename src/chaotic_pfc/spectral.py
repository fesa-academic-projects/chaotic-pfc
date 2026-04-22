"""
spectral.py
===========
PSD estimation via Welch's method (Hamming window).
Returns normalised PSD (peak = 1) and ω/π axis in [0, 1].
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch, windows


def psd_normalised(
    x: NDArray,
    nfft: int = 4096,
    window_length: int = 1024,
    fs: float = 1.0,
    remove_dc: bool = True,
) -> tuple[NDArray, NDArray]:
    if remove_dc:
        x = x - np.mean(x)
    win = windows.hamming(window_length)
    f, Pxx = welch(
        x,
        fs=fs,
        window=win,
        nperseg=window_length,
        noverlap=None,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
    )
    peak = Pxx.max()
    if peak > 0:
        Pxx = Pxx / peak
    omega_norm = 2.0 * f / fs  # ω/π ∈ [0, 1]
    return omega_norm, Pxx

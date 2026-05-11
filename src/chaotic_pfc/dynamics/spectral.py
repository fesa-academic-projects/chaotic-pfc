"""
spectral.py
===========
Power-spectral-density estimation utilities used by the plotting layer.

The single public function, :func:`psd_normalised`, wraps
:func:`scipy.signal.welch` with the conventions adopted throughout the
project: peak-normalised magnitude, and the frequency axis expressed as
``ω/π ∈ [0, 1]`` rather than cycles or hertz.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch, windows

_WINDOWS = {
    "hamming": windows.hamming,
    "hann": windows.hann,
    "blackman": windows.blackman,
    "kaiser": windows.kaiser,
    "blackmanharris": windows.blackmanharris,
    "boxcar": windows.boxcar,
    "bartlett": windows.bartlett,
}


def psd_normalised(
    x: NDArray,
    nfft: int = 4096,
    window_length: int = 1024,
    fs: float = 1.0,
    remove_dc: bool = True,
    *,
    window: str = "hamming",
    kaiser_beta: float = 5.0,
) -> tuple[NDArray, NDArray]:
    """Estimate the normalised power-spectral density of ``x`` via Welch.

    The PSD is one-sided, peak-normalised to ``1.0``, and the frequency
    axis is returned as ``ω/π`` for consistency with the notation used
    in the project figures (x-axis of every PSD panel).

    Parameters
    ----------
    x
        Input signal, shape ``(N,)``. Can be any dtype coercible to
        ``float``.
    nfft
        Length of the FFT. Larger values give smoother spectra at the
        cost of more computation. Must be ``>= window_length``.
    window_length
        Length of the Welch segment (aka ``nperseg``). Controls the
        frequency resolution.
    fs
        Sampling frequency. Leave at the default ``1.0`` to work in
        normalised units — the output axis becomes ``ω/π`` directly.
    remove_dc
        If ``True`` (default), the DC component of ``x`` is subtracted
        before the FFT to avoid a spurious spike at ``ω = 0``. Set to
        ``False`` only if you explicitly want to see the DC bin.
    window
        Welch window. Supported values: ``"hamming"`` (default),
        ``"hann"``, ``"blackman"``, ``"kaiser"``, ``"blackmanharris"``,
        ``"boxcar"``, ``"bartlett"``.
    kaiser_beta
        Shape parameter of the Kaiser window. Must be ``>= 0``;
        ignored when ``window != "kaiser"``.

    Returns
    -------
    omega_norm : ndarray, shape (nfft // 2 + 1,)
        Normalised frequency axis, ``ω/π ∈ [0, 1]``.
    psd : ndarray, shape (nfft // 2 + 1,)
        Peak-normalised PSD. ``psd.max() == 1.0`` unless the input is
        identically zero.

    Notes
    -----
    Welch's method divides ``x`` into overlapping segments, applies the
    window, takes the periodogram of each, and averages the result.
    This trades frequency resolution (controlled by ``window_length``)
    for variance reduction, which is the right trade-off for visualising
    the smooth broadband spectrum of a chaotic carrier.
    """
    if remove_dc:
        x = x - np.mean(x)
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {sorted(_WINDOWS)}, got {window!r}")
    if window == "kaiser":
        if kaiser_beta < 0:
            raise ValueError(f"kaiser_beta must be >= 0, got {kaiser_beta!r}")
        win = windows.kaiser(window_length, beta=kaiser_beta)
    else:
        win = _WINDOWS[window](window_length)
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
    omega_norm = 2.0 * f / fs  # Convert Welch f (cycles/sample) to ω/π: ω = 2πf, so ω/π = 2f/fs
    return omega_norm, Pxx

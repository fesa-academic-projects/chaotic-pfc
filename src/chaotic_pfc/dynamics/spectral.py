"""Power-spectral-density estimation utilities used by the plotting layer.

The single public function, :func:`psd_normalised`, wraps
:func:`scipy.signal.welch` with the conventions adopted throughout the
project: peak-normalised magnitude, and the frequency axis expressed as
``ω/π ∈ [0, 1]`` rather than cycles or hertz.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch, windows

from chaotic_pfc._validate import require_1d_nonempty, require_in_range, require_positive_int
from chaotic_pfc.errors import InvalidParameterError

_WINDOWS = {
    "hamming": windows.hamming,
    "hann": windows.hann,
    "blackman": windows.blackman,
    "blackmanharris": windows.blackmanharris,
    "boxcar": windows.boxcar,
    "bartlett": windows.bartlett,
}


def psd_normalised(
    x: NDArray[np.float64],
    nfft: int = 4096,
    window_length: int = 1024,
    fs: float = 1.0,
    remove_dc: bool = True,
    *,
    window: str = "hamming",
    kaiser_beta: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        Peak-normalised PSD. ``psd.max() == 1.0``.  Raises ``ValueError``
        when the input is identically zero.

    Notes
    -----
    Welch's method divides ``x`` into overlapping segments, applies the
    window, takes the periodogram of each, and averages the result.
    This trades frequency resolution (controlled by ``window_length``)
    for variance reduction, which is the right trade-off for visualising
    the smooth broadband spectrum of a chaotic carrier.

    Segment overlap is left at the scipy default of 50 %
    (``noverlap = nperseg // 2``).

    Raises
    ------
    InvalidParameterError
        If ``x`` is empty or non-finite, ``nfft``/``window_length``/``fs``
        are out of range, ``window``/``kaiser_beta`` is invalid, or the
        signal has zero power.
    """
    x_arr = require_1d_nonempty("x", x)
    window_length = require_positive_int("window_length", window_length, min_value=1)
    nfft = require_positive_int("nfft", nfft, min_value=window_length)
    fs = require_in_range("fs", fs, 0.0, 1e150, closed="neither")
    if not isinstance(remove_dc, bool):
        raise InvalidParameterError(f"remove_dc must be a bool, got {remove_dc!r}")
    if remove_dc:
        x_arr = x_arr - np.mean(x_arr)
    if not isinstance(window, str) or (window not in _WINDOWS and window != "kaiser"):
        raise InvalidParameterError(
            f"window must be one of {[*sorted(_WINDOWS), 'kaiser']}, got {window!r}"
        )
    if window == "kaiser":
        kaiser_beta = require_in_range("kaiser_beta", kaiser_beta, 0.0, 700.0, closed="both")
        win = windows.kaiser(window_length, beta=kaiser_beta)
    else:
        win = _WINDOWS[window](window_length)
    f, Pxx = welch(
        x_arr,
        fs=fs,
        window=win,
        nperseg=window_length,
        noverlap=None,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
    )
    peak = float(Pxx.max())
    if np.isfinite(peak) and peak > 0:
        Pxx = Pxx / peak
    else:
        raise InvalidParameterError(
            f"input signal has zero or non-finite power (peak={peak}); cannot normalise PSD to max=1"
        )
    omega_norm = 2.0 * f / fs  # Convert Welch f (cycles/sample) to ω/π: ω = 2πf, so ω/π = 2f/fs
    return omega_norm, Pxx

"""
channel.py
==========
Transmission-channel models sitting between the chaotic transmitter and
the receiver.

Eight channels are provided:

* :func:`ideal_channel` — noiseless pass-through used to establish a
  best-case baseline for chaos synchronisation.
* :func:`fir_channel` — a symmetric FIR low-pass filter built on top of
  :func:`scipy.signal.firwin`, representing a band-limited physical
  link.
* :func:`awgn` — additive white Gaussian noise.
* :func:`channel_impulsive` — AWGN with Middleton Class-A impulsive noise.
* :func:`channel_multipath` — multipath channel with configurable tap
  delays and gains.
* :func:`channel_interferers` — composite channel (AWGN + DCSK interferer
  + narrow-band WiFi-like interferer), lives in :mod:`chaotic_pfc.comms.dcsk`.
* :func:`channel_urban` — combined urban channel with impulsive noise,
  multipath, and interferers, lives in :mod:`chaotic_pfc.comms.dcsk`.

All accept the transmitted signal ``s`` as a 1-D array and return the
received signal ``r`` of the same length. The FIR channel additionally
returns its filter coefficients so they can be overlaid on the PSD
plots produced by :mod:`chaotic_pfc.plotting`.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin, lfilter


def ideal_channel(s: NDArray) -> NDArray:
    """Transmit ``s`` through a noiseless, distortion-free channel.

    This is literally a copy of the input. It is still defined as a
    function (rather than aliased to ``numpy.ndarray.copy``) so that the
    receiver-side code can always call ``ideal_channel(s)`` consistently
    regardless of the channel configuration chosen by the caller.

    Parameters
    ----------
    s
        Transmitted signal, shape ``(N,)``.

    Returns
    -------
    ndarray, shape (N,)
        An independent copy of ``s``. Mutating the output does not
        affect the input.

    Implements: :class:`~chaotic_pfc.comms.protocols.Channel`."""
    return s.copy()


def fir_channel(
    s: NDArray,
    cutoff: float = 0.99,
    num_taps: int = 201,
    window: str = "hamming",
) -> tuple[NDArray, NDArray]:
    """Transmit ``s`` through a symmetric FIR low-pass channel.

    The channel is built with :func:`scipy.signal.firwin` using
    ``pass_zero=True`` (true low-pass) and unit sampling frequency
    normalisation (``fs=2.0``). The DC gain of the resulting filter is
    normalised to ``1.0``.

    Parameters
    ----------
    s
        Transmitted signal, shape ``(N,)``.
    cutoff
        Normalised cutoff frequency ``ω_c / π ∈ (0, 1)``. Defaults to
        ``0.99`` — i.e. the channel passes almost everything, mimicking
        a very lightly band-limited link.
    num_taps
        Length of the FIR filter in samples. Must be a positive integer.
    window
        Window function passed through to ``firwin``. Any name accepted
        by :func:`scipy.signal.get_window` is valid — common choices are
        ``"hamming"``, ``"hann"``, ``"blackman"``.

    Returns
    -------
    r : ndarray, shape (N,)
        Received signal, ``s`` filtered through the FIR coefficients.
    h : ndarray, shape (num_taps,)
        Filter coefficients, useful for overlaying the channel response
        on PSD plots.

    Implements: :class:`~chaotic_pfc.comms.protocols.Channel`."""
    h = firwin(numtaps=num_taps, cutoff=cutoff, window=window, pass_zero=True, fs=2.0)
    r = lfilter(h, [1.0], s)
    return r, h


# ── Channel models ───────────────────────────────────────────────────────────


def awgn(sig: NDArray, snr_db: float, rng: np.random.Generator | None = None) -> NDArray:
    """Add white Gaussian noise to *sig* for a given SNR in dB.

    Parameters
    ----------
    sig
        Signal samples.
    snr_db
        Signal-to-noise ratio in dB.
    rng
        Random generator (uses ``np.random.default_rng`` if None).
    """
    if rng is None:
        rng = np.random.default_rng()
    p = float(np.mean(sig**2)) / 10 ** (snr_db / 10)
    return sig + rng.normal(0.0, float(np.sqrt(p)), sig.shape)


def channel_impulsive(
    sig: NDArray,
    snr_db: float,
    prob_impulso: float = 0.01,
    amp_fator: float = 10.0,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """AWGN channel with Middleton Class-A impulsive noise.

    Parameters
    ----------
    prob_impulso
        Probability a sample is hit by an impulse (e.g. 0.01 = 1 %).
    amp_fator
        Impulse amplitude in multiples of the signal std.
    """
    if rng is None:
        rng = np.random.default_rng()
    rx = awgn(sig, snr_db, rng)
    std = float(np.std(sig))
    mask = rng.random(len(sig)) < prob_impulso
    rx[mask] += amp_fator * std * rng.choice(np.array([-1.0, 1.0]), mask.sum())
    return rx


def channel_multipath(
    sig: NDArray,
    snr_db: float,
    delays: list[int] | None = None,
    gains: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Multipath channel with configurable tap delays and gains.

    Parameters
    ----------
    delays
        Delay of each path in samples (default: ``[0, 3, 7, 15]``).
    gains
        Attenuation per path (default: ``[1.0, 0.6, 0.4, 0.2]``).
    """
    if delays is None:
        delays = [0, 3, 7, 15]
    if gains is None:
        gains = [1.0, 0.6, 0.4, 0.2]

    N = len(sig)
    out = np.zeros(N)
    for d, g in zip(delays, gains, strict=True):
        if d == 0:
            out += g * sig
        else:
            out[d:] += g * sig[: N - d]
    out *= float(np.sqrt(np.mean(sig**2))) / (float(np.std(out)) + 1e-12)
    return awgn(out, snr_db, rng)


def _wifi_interferer(
    N: int, fc: float = 0.2, bw: float = 0.08, rng: np.random.Generator | None = None
) -> NDArray:
    """Synthetic narrow-band interferer (noise filtered in a sub-band)."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, 1, N)
    h_bp = _wifi_fir(fc, bw)
    return lfilter(h_bp, 1.0, noise)


@lru_cache(maxsize=4)
def _wifi_fir(fc: float, bw: float) -> NDArray:
    """Cached FIR band-pass filter for the WiFi-like interferer."""
    return firwin(101, [fc - bw / 2, fc + bw / 2], pass_zero=False)

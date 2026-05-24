"""
channel.py
==========
Transmission-channel models sitting between the chaotic transmitter and
the receiver.

Six channels are provided:

* :func:`ideal_channel` — noiseless pass-through used to establish a
  best-case baseline for chaos synchronisation.
* :func:`fir_channel` — a symmetric FIR low-pass filter built on top of
  :func:`scipy.signal.firwin`, representing a band-limited physical
  link.
* :func:`awgn` — additive white Gaussian noise.
* :func:`channel_impulsive` — AWGN with Middleton Class-A impulsive noise.
* :func:`channel_multipath` — multipath channel with configurable tap
  delays and gains.

All accept the transmitted signal ``s`` as a 1-D array and return the
received signal ``r`` of the same length. The FIR channel additionally
returns its filter coefficients so they can be overlaid on the PSD
plots produced by :mod:`chaotic_pfc.plotting`.
"""

from __future__ import annotations

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

    Notes
    -----
    Unlike :func:`ideal_channel`, this function returns the filter
    taps alongside the filtered signal.  It therefore does **not**
    satisfy :class:`~chaotic_pfc.comms.protocols.Channel` (which
    expects a single ``NDArray`` return)."""
    h = firwin(numtaps=num_taps, cutoff=cutoff, window=window, pass_zero=True, fs=2.0)
    r = lfilter(h, [1.0], s)
    return r, h


# ── Channel models ───────────────────────────────────────────────────────────


def awgn(sig: NDArray, snr_db: float, rng: np.random.Generator | None = None) -> NDArray:
    """Add white Gaussian noise to *sig* for a given SNR in dB.

    The noise power is ``E[|sig|²] / 10^(SNR/10)``, so a higher SNR
    means less noise.

    Parameters
    ----------
    sig
        Signal samples, shape ``(N,)``.
    snr_db
        Signal-to-noise ratio in dB.  ``snr_db → ∞`` gives the
        original signal; ``snr_db = 0`` gives equal-power noise.
    rng
        NumPy ``Generator``.  ``None`` (default) creates a fresh
        ``default_rng()`` with no fixed seed, so results are not
        reproducible across calls.  Pass a seeded generator for
        deterministic noise.

    Returns
    -------
    ndarray, shape (N,)
        Signal with additive white Gaussian noise.

    Notes
    -----
    The SNR is defined as :math:`E[|s|^2] / 10^{SNR/10}` [Haykin01]_.

    References
    ----------
    .. [Haykin01] S. Haykin. "Communication Systems." 4th ed., Wiley, 2001.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = float(np.mean(sig**2)) / 10 ** (snr_db / 10)
    return sig + rng.normal(0.0, float(np.sqrt(p)), sig.shape)


def channel_impulsive(
    sig: NDArray,
    snr_db: float,
    prob_impulse: float = 0.01,
    amp_factor: float = 10.0,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """AWGN channel with Middleton Class-A impulsive noise.

    A fraction *prob_impulse* of samples receive an additional impulse
    of randomised sign and amplitude *amp_factor* × ``std(sig)`` on top
    of the AWGN background.

    Parameters
    ----------
    sig
        Signal samples, shape ``(N,)``.
    snr_db
        Background AWGN signal-to-noise ratio in dB.
    prob_impulse
        Probability a sample is hit by an impulse (e.g. 0.01 = 1 %).
    amp_factor
        Impulse amplitude in multiples of the signal std.
    rng
        Random generator.  ``None`` creates a fresh ``default_rng()``.

    Returns
    -------
    ndarray, shape (N,)
        Signal with AWGN + Middleton Class-A impulsive noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    rx = awgn(sig, snr_db, rng)
    std = float(np.std(sig))
    mask = rng.random(len(sig)) < prob_impulse
    rx[mask] += amp_factor * std * rng.choice(np.array([-1.0, 1.0]), mask.sum())
    return rx


def channel_multipath(
    sig: NDArray,
    snr_db: float,
    delays: list[int] | None = None,
    gains: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Multipath channel with configurable tap delays and gains.

    Simulates a frequency-selective channel by summing delayed copies
    of the input.  Output power is normalised to the input power, then
    AWGN is added at the requested SNR.

    Parameters
    ----------
    sig
        Signal samples, shape ``(N,)``.
    snr_db
        AWGN signal-to-noise ratio after multipath combining, in dB.
    delays
        Delay of each path in samples (default: ``[0, 3, 7, 15]``).
    gains
        Attenuation per path (default: ``[1.0, 0.6, 0.4, 0.2]``).
    rng
        Random generator.  ``None`` creates a fresh ``default_rng()``.

    Returns
    -------
    ndarray, shape (N,)
        Multipath-combined signal with AWGN.
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

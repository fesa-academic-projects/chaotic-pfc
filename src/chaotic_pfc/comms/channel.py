"""Transmission-channel models between the chaotic transmitter and receiver.

Five channels are provided:

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
plots produced by :mod:`chaotic_pfc.plotting.figures`.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin, lfilter

from chaotic_pfc._validate import (
    require_1d_nonempty,
    require_finite,
    require_in_range,
    require_positive_int,
    require_probability,
)
from chaotic_pfc.errors import InvalidParameterError


def ideal_channel(s: NDArray[np.float64]) -> NDArray[np.float64]:
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

    Raises
    ------
    InvalidParameterError
        If ``s`` is not a non-empty 1-D finite array.

    Implements: :class:`~chaotic_pfc.comms.protocols.Channel`.
    """
    s_arr = require_1d_nonempty("s", s)
    return s_arr.copy()


def fir_channel(
    s: NDArray[np.float64],
    *,
    num_taps: int = 201,
    cutoff: float = 0.99,
    window: str = "hamming",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transmit ``s`` through a symmetric FIR low-pass channel.

    The channel is built with :func:`scipy.signal.firwin` using
    ``pass_zero=True`` (true low-pass) and unit sampling frequency
    normalisation (``fs=2.0``). The DC gain of the resulting filter is
    normalised to ``1.0``.

    Parameters
    ----------
    s
        Transmitted signal, shape ``(N,)``.
    num_taps
        Length of the FIR filter in samples. Must be a positive integer.
    cutoff
        Normalised cutoff frequency ``ω_c / π ∈ (0, 1)``. Defaults to
        ``0.99`` — i.e. the channel passes almost everything, mimicking
        a very lightly band-limited link.
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
    expects a single ``NDArray[np.float64]`` return).

    Raises
    ------
    InvalidParameterError
        If ``s`` is not a non-empty 1-D finite array, ``num_taps < 1``,
        ``cutoff`` is outside ``(0, 1)``, or ``window`` is invalid.
    """
    s_arr = require_1d_nonempty("s", s)
    num_taps = require_positive_int("num_taps", num_taps, min_value=1)
    cutoff = require_in_range("cutoff", cutoff, 0.0, 1.0, closed="neither")
    if not isinstance(window, str):
        raise InvalidParameterError(f"window must be a string, got {window!r}")
    try:
        # Symmetric all-zero windows (Hann/Bartlett with two taps) make scipy
        # divide by a zero gain; the non-finite result is rejected below.
        with np.errstate(invalid="ignore", divide="ignore"):
            h = firwin(numtaps=num_taps, cutoff=cutoff, window=window, pass_zero=True, fs=2.0)
    except Exception as exc:
        raise InvalidParameterError(f"invalid FIR channel parameters: {exc}") from exc
    if not np.all(np.isfinite(h)):
        raise InvalidParameterError(
            f"firwin produced non-finite coefficients for num_taps={num_taps}, window={window!r}"
        )
    r = lfilter(h, [1.0], s_arr)
    return r, h


# ── Channel models ───────────────────────────────────────────────────────────


def awgn(
    sig: NDArray[np.float64], snr_db: float, rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
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

    Raises
    ------
    InvalidParameterError
        If ``sig`` is not a non-empty 1-D finite array or ``snr_db`` is
        non-finite or out of numerical range.

    Notes
    -----
    The SNR is defined as :math:`E[|s|^2] / 10^{SNR/10}` [Haykin01]_.

    References
    ----------
    .. [Haykin01] S. Haykin. "Communication Systems." 4th ed., Wiley, 2001.
    """
    sig_arr = require_1d_nonempty("sig", sig)
    snr_val = require_in_range("snr_db", snr_db, -1000.0, 1000.0, closed="both")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise InvalidParameterError(f"rng must be a numpy.random.Generator or None, got {rng!r}")
    if rng is None:
        rng = np.random.default_rng()
    with np.errstate(over="ignore", invalid="ignore"):
        sig_power = float(np.mean(sig_arr**2))
        p = sig_power / (10.0 ** (snr_val / 10.0))
    if not math.isfinite(p) or p < 0.0:
        raise InvalidParameterError(
            f"noise power overflowed for sig and snr_db={snr_db!r} (power={p})"
        )
    return sig_arr + rng.normal(0.0, float(np.sqrt(p)), sig_arr.shape)


def channel_impulsive(
    sig: NDArray[np.float64],
    snr_db: float,
    prob_impulse: float = 0.01,
    amp_factor: float = 10.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
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

    Raises
    ------
    InvalidParameterError
        If ``sig`` is invalid, ``snr_db`` is non-finite, ``prob_impulse``
        is outside ``[0, 1]``, or ``amp_factor`` is negative or non-finite.
    """
    sig_arr = require_1d_nonempty("sig", sig)
    snr_val = require_in_range("snr_db", snr_db, -1000.0, 1000.0, closed="both")
    prob = require_probability("prob_impulse", prob_impulse)
    amp = require_in_range("amp_factor", amp_factor, 0.0, 1e150, closed="both")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise InvalidParameterError(f"rng must be a numpy.random.Generator or None, got {rng!r}")
    if rng is None:
        rng = np.random.default_rng()
    rx = awgn(sig_arr, snr_val, rng)
    with np.errstate(over="ignore", invalid="ignore"):
        std = float(np.std(sig_arr))
        scale = amp * std
    if not math.isfinite(scale):
        raise InvalidParameterError(
            f"impulse amplitude overflowed for sig and amp_factor={amp_factor!r}"
        )
    mask = rng.random(len(sig_arr)) < prob
    rx[mask] += scale * rng.choice(np.array([-1.0, 1.0]), mask.sum())
    return rx


def channel_multipath(
    sig: NDArray[np.float64],
    snr_db: float,
    delays: list[int] | None = None,
    gains: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
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

    Raises
    ------
    InvalidParameterError
        If ``sig``, ``snr_db``, ``delays``, or ``gains`` is invalid.
    """
    sig_arr = require_1d_nonempty("sig", sig)
    snr_val = require_in_range("snr_db", snr_db, -1000.0, 1000.0, closed="both")
    if delays is None:
        delays = [0, 3, 7, 15]
    if gains is None:
        gains = [1.0, 0.6, 0.4, 0.2]
    if isinstance(delays, (str, bytes)) or not hasattr(delays, "__len__") or len(delays) == 0:
        raise InvalidParameterError("delays must be a non-empty sequence of non-negative integers")
    if isinstance(gains, (str, bytes)) or not hasattr(gains, "__len__") or len(gains) == 0:
        raise InvalidParameterError("gains must be a non-empty sequence of finite floats")
    if len(delays) != len(gains):
        raise InvalidParameterError(
            f"delays and gains must have the same length, got {len(delays)} and {len(gains)}"
        )

    N = len(sig_arr)
    validated_delays = [
        require_positive_int(f"delays[{i}]", d, min_value=0) for i, d in enumerate(delays)
    ]
    validated_gains = [require_finite(f"gains[{i}]", g) for i, g in enumerate(gains)]
    for d in validated_delays:
        if d >= N:
            raise InvalidParameterError(f"delay {d} must be strictly less than signal length {N}")

    out = np.zeros(N)
    with np.errstate(over="ignore", invalid="ignore"):
        for d, g in zip(validated_delays, validated_gains, strict=True):
            if d == 0:
                out += g * sig_arr
            else:
                out[d:] += g * sig_arr[: N - d]
        rms_sig = float(np.sqrt(np.mean(sig_arr**2)))
        std_out = float(np.std(out))
    if not (np.all(np.isfinite(out)) and math.isfinite(rms_sig) and math.isfinite(std_out)):
        raise InvalidParameterError("multipath combination overflowed to non-finite values")
    out *= rms_sig / (std_out + 1e-12)
    return awgn(out, snr_val, rng)

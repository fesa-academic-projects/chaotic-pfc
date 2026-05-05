"""
channel.py
==========
Transmission-channel models sitting between the chaotic transmitter and
the receiver.

Two channels are provided:

* :func:`ideal_channel` — noiseless pass-through used to establish a
  best-case baseline for chaos synchronisation.
* :func:`fir_channel` — a symmetric FIR low-pass filter built on top of
  :func:`scipy.signal.firwin`, representing a band-limited physical
  link.

Both accept the transmitted signal ``s`` as a 1-D array and return the
received signal ``r`` of the same length. The FIR channel additionally
returns its filter coefficients so they can be overlaid on the PSD
plots produced by :mod:`chaotic_pfc.plotting`.
"""

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

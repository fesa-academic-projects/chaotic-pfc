"""
signals.py
==========
Generators for the information-bearing messages used throughout the
chaotic communication pipeline.

Three waveforms are provided:

* :func:`binary_message` — a square-wave BPSK-style message taking values
  in ``{-1, +1}`` with a fixed bit period.
* :func:`text_message` — an ASCII text encoded as an NRZ BPSK bit stream,
  so the transmitted message looks like real digital data instead of a
  periodic square wave.
* :func:`sinusoidal_message` — a pure cosine/sine probe useful for
  spectral-response measurements.

All functions return NumPy arrays of length ``N`` so they can be fed
directly to :func:`chaotic_pfc.comms.transmitter.transmit`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def binary_message(N: int, period: int = 20) -> NDArray:
    """Generate a periodic square-wave binary message.

    The output takes values in ``{+1, -1}``, with the first half of each
    period at ``+1`` and the second half at ``-1``. This is the standard
    BPSK-style message used by :func:`chaotic_pfc.comms.transmitter.transmit`.

    Parameters
    ----------
    N
        Total number of samples to produce.
    period
        Length of one full ``+1`` / ``-1`` cycle. Must be a positive
        even integer; each half-cycle holds ``period // 2`` samples.

    Returns
    -------
    ndarray, shape (N,)
        The message samples, each ``+1.0`` or ``-1.0``.

    Raises
    ------
    ValueError
        If ``period`` is not a positive even integer.

    Examples
    --------
    >>> binary_message(8, period=4)
    array([ 1.,  1., -1., -1.,  1.,  1., -1., -1.])
    """
    if period <= 0 or period % 2 != 0:
        raise ValueError(f"period must be a positive even integer, got {period}")
    half = period // 2
    block = np.concatenate([np.ones(half), -np.ones(half)])
    num_blocks = int(np.ceil(N / period))
    return np.tile(block, num_blocks)[:N]


def text_message(text: str, N: int, *, bit_period: int = 20) -> NDArray:
    """Encode *text* as an NRZ BPSK message of length *N*.

    Each character is converted to its 8-bit ASCII representation, the
    resulting bit stream is mapped to ``{+1, -1}`` (bit ``1`` → ``+1``,
    bit ``0`` → ``-1``) and every bit is held for *bit_period* samples
    (non-return-to-zero). The encoded block is tiled to fill exactly
    ``N`` samples, mirroring :func:`binary_message`.

    Unlike :func:`binary_message` (a periodic square wave), this produces
    an irregular run-length pattern that resembles a real digital
    transmission, which is what the ``comm-*`` figures use as their
    information-bearing message.

    Parameters
    ----------
    text
        Message text. Must contain only ASCII characters so it can be
        encoded to a single byte per character.
    N
        Total number of samples to produce.
    bit_period
        Number of samples each bit is held. Must be a positive integer.

    Returns
    -------
    ndarray, shape (N,)
        The BPSK message samples, each ``+1.0`` or ``-1.0``.

    Raises
    ------
    ValueError
        If ``bit_period`` is not a positive integer or ``text`` is empty.
    UnicodeEncodeError
        If ``text`` contains characters outside the ASCII range.

    Examples
    --------
    ``"A"`` encodes to ``0b01000001``; with one sample per bit this is

    >>> text_message("A", 8, bit_period=1)
    array([-1.,  1., -1., -1., -1., -1., -1.,  1.])
    """
    if bit_period <= 0:
        raise ValueError(f"bit_period must be a positive integer, got {bit_period}")
    if not text:
        raise ValueError("text must be a non-empty string")

    byte_values = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    bits = np.unpackbits(byte_values)
    symbols = np.where(bits == 1, 1.0, -1.0)
    block = np.repeat(symbols, bit_period)
    num_blocks = int(np.ceil(N / len(block)))
    return np.tile(block, num_blocks)[:N]


def sinusoidal_message(N: int, normalised_freq: float = 0.1) -> NDArray:
    """Generate a single-tone sinusoidal probe signal.

    Useful for frequency-response characterisation: feeding the output of
    this function through the transmitter/channel/receiver chain shows
    the system's gain and phase at one specific frequency.

    Parameters
    ----------
    N
        Number of samples to produce.
    normalised_freq
        Frequency in cycles per sample. Must satisfy ``0 < f < 0.5`` to
        avoid aliasing (Nyquist at ``f = 0.5``). The default of ``0.1``
        gives ten samples per cycle.

    Returns
    -------
    ndarray, shape (N,)
        The samples ``sin(2π · normalised_freq · n)`` for
        ``n = 0, 1, …, N − 1``.

    Raises
    ------
    ValueError
        If *normalised_freq* is not strictly between 0 and 0.5.
    """
    if not (0 < normalised_freq < 0.5):
        raise ValueError(f"normalised_freq must be in (0, 0.5), got {normalised_freq}")
    n = np.arange(N)
    return np.sin(2.0 * np.pi * normalised_freq * n)

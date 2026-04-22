"""
signals.py
==========
Generators for the information-bearing messages used throughout the
chaotic communication pipeline.

Two waveforms are provided:

* :func:`binary_message` — a square-wave BPSK-style message taking values
  in ``{-1, +1}`` with a fixed bit period.
* :func:`sinusoidal_message` — a pure cosine/sine probe useful for
  spectral-response measurements.

Both functions return NumPy arrays of length ``N`` so they can be fed
directly to :func:`chaotic_pfc.transmitter.transmit`.
"""

import numpy as np
from numpy.typing import NDArray


def binary_message(N: int, period: int = 20) -> NDArray:
    """Generate a periodic square-wave binary message.

    The output takes values in ``{+1, -1}``, with the first half of each
    period at ``+1`` and the second half at ``-1``. This is the standard
    BPSK-style message used by :func:`chaotic_pfc.transmitter.transmit`.

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
    """
    n = np.arange(N)
    return np.sin(2.0 * np.pi * normalised_freq * n)

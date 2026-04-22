"""
channel.py
==========
Channel models: ideal (noiseless) and FIR low-pass.
"""

from numpy.typing import NDArray
from scipy.signal import firwin, lfilter


def ideal_channel(s: NDArray) -> NDArray:
    return s.copy()


def fir_channel(
    s: NDArray,
    cutoff: float = 0.99,
    num_taps: int = 201,
    window: str = "hamming",
) -> tuple[NDArray, NDArray]:
    """FIR low-pass channel. Returns (r, h)."""
    h = firwin(numtaps=num_taps, cutoff=cutoff, window=window, pass_zero=True, fs=2.0)
    r = lfilter(h, [1.0], s)
    return r, h

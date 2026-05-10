"""Shared test helpers to reduce duplication across test modules."""

from scipy.signal import firwin


def make_fir_coeffs(Nc: int = 5):
    """Return a set of FIR coefficients usable in transmit/receive order-N tests."""
    return firwin(numtaps=Nc, cutoff=0.5, window="hamming")

"""Shared test helpers to reduce duplication across test modules."""

from scipy.signal import firwin


def make_fir_coeffs(Nc: int = 5):
    """Return a set of FIR coefficients usable in transmit/receive order-N tests."""
    return firwin(numtaps=Nc, cutoff=0.5, window="hamming")


def assert_seed_determinism(test_case, fn, seed=42):
    """Assert that *fn*(seed=seed) returns identical results on two calls."""
    import numpy as np

    r1 = fn(seed=seed)
    r2 = fn(seed=seed)
    if isinstance(r1, tuple):
        for a, b in zip(r1, r2, strict=False):
            np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_array_equal(r1, r2)

"""
dcsk.py
=======
Differential Chaos Shift Keying (DCSK) over a FIR-filtered Henon map.

References
----------
.. [Kolumban96] G. Kolumbán, B. Vizvári, W. Schwarz, A. Abel.
   "Differential chaos shift keying: A robust coding for chaotic
   communication." Proc. NDES, 1996.

.. [Kaddoum13] G. Kaddoum, E. Soujeri, Y. Nijsure.
   "Design of a Short Reference Noncoherent Chaos-Based
   Communication System." IEEE Trans. Commun., vol. 61,
   no. 12, pp. 4854–4863, 2013.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..dynamics.maps import henon_fir_sequence
from .channel import _wifi_interferer, awgn, channel_impulsive, channel_multipath

DCSK_DEFAULT_WC: float = 0.9091


def _chaos_sequence(
    n_samples: int, transient: int, *, n_taps: int, wc: float, window: str
) -> NDArray:
    """Generate and normalise a chaotic FIR-Henon sequence."""
    chaos = henon_fir_sequence(transient + n_samples, n_taps=n_taps, wc=wc, window=window)
    chaos = chaos[transient:]
    chaos /= float(np.std(chaos)) + 1e-12
    return chaos


def dcsk_transmit(
    bits: NDArray,
    beta: int = 64,
    n_taps: int = 5,
    wc: float = DCSK_DEFAULT_WC,
    window: str = "hamming",
    transient: int = 500,
) -> NDArray:
    """Modulate a bit sequence using DCSK over FIR-filtered Henon.

    Each bit is encoded as two consecutive ``beta``-sample slots:
    a *reference* slot followed by a *data* slot. For bit 0 the
    data slot is a copy of the reference; for bit 1 it is inverted.

    Parameters
    ----------
    bits
        Binary array (0/1) of message bits.
    beta
        Spreading factor — samples per slot.
    n_taps, wc, window
        Passed to :func:`henon_fir_sequence`.
    transient
        Burn-in samples discarded before modulation.

    Returns
    -------
    ndarray, shape (2 * beta * len(bits),)
        DCSK-modulated signal samples.
    """
    N = len(bits)
    chaos = _chaos_sequence(N * beta, transient, n_taps=n_taps, wc=wc, window=window)
    sig = np.empty(N * 2 * beta)
    for i in range(N):
        ref = chaos[i * beta : (i + 1) * beta]
        data = ref if bits[i] == 0 else -ref
        sig[i * 2 * beta : (i + 1) * 2 * beta] = np.concatenate([ref, data])
    return sig


def dcsk_receive(
    rx: NDArray,
    beta: int = 64,
) -> NDArray:
    """Demodulate a DCSK signal by correlating reference and data slots.

    Parameters
    ----------
    rx
        Received signal (must be aligned to symbol boundaries).
    beta
        Spreading factor used by the transmitter.

    Returns
    -------
    ndarray, dtype int
        Decoded bits (0 or 1).
    """
    n_bits = len(rx) // (2 * beta)
    out = np.empty(n_bits, dtype=np.int64)
    for i in range(n_bits):
        s = rx[i * 2 * beta : (i + 1) * 2 * beta]
        out[i] = 0 if float(np.dot(s[:beta], s[beta:])) > 0 else 1
    return out


# ── EF-DCSK (Efficient DCSK) ─────────────────────────────────────────────────
#
# Reference:
#   Kaddoum, G., Soujeri, E., Nijsure, Y. "Design of a Short Reference
#   Noncoherent Chaos-Based Communication System." IEEE TCOM, 2013.
#
# Each symbol uses a single β-sample slot where the reference and its
# time-reversed copy are superposed. This doubles the data rate relative
# to classical DCSK with minimal loss in BER performance.


def efdcsk_transmit(
    bits: NDArray,
    beta: int = 64,
    n_taps: int = 5,
    wc: float = DCSK_DEFAULT_WC,
    window: str = "hamming",
    transient: int = 500,
) -> NDArray:
    """Modulate bits using Efficient DCSK (EF-DCSK).

    Each bit occupies a single ``beta``-sample slot:
    ``s = ref + b * ref_reversed``, where ``b = +1`` for 0, ``-1`` for 1.

    Parameters
    ----------
    bits, beta, n_taps, wc, window, transient
        Same semantics as :func:`dcsk_transmit`.

    Returns
    -------
    ndarray, shape (beta * len(bits),)
        EF-DCSK signal, half the length of classical DCSK.
    """
    N = len(bits)
    chaos = _chaos_sequence(N * beta, transient, n_taps=n_taps, wc=wc, window=window)
    sig = np.empty(N * beta)
    for i in range(N):
        ref = chaos[i * beta : (i + 1) * beta]
        ref_rev = ref[::-1]
        b = 1 if bits[i] == 0 else -1
        sig[i * beta : (i + 1) * beta] = ref + b * ref_rev
    return sig


def efdcsk_receive(
    rx: NDArray,
    beta: int = 64,
) -> NDArray:
    """Demodulate EF-DCSK by correlating with the time-reversed signal.

    The decision variable is ``dot(received, received_reversed)``.
    """
    n_bits = len(rx) // beta
    out = np.empty(n_bits, dtype=np.int64)
    for i in range(n_bits):
        s = rx[i * beta : (i + 1) * beta]
        out[i] = 0 if float(np.dot(s, s[::-1])) > 0 else 1
    return out


def ber(tx: NDArray, rx: NDArray) -> float:
    """Bit error rate between transmitted and received bit arrays."""
    return float(np.mean(tx != rx))


# ── Channel models ───────────────────────────────────────────────────────────


def channel_interferers(
    sig: NDArray,
    snr_db: float,
    sir_dcsk_db: float = 10.0,
    sir_wifi_db: float = 15.0,
    n_taps_int: int = 9,
    wc_int: float = 0.5556,
    beta: int = 64,
    n_bits: int = 600,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """AWGN + DCSK interferer + narrow-band WiFi-like interferer.

    Parameters
    ----------
    sir_dcsk_db
        Signal-to-interference ratio for the DCSK interferer.
    sir_wifi_db
        Signal-to-interference ratio for the WiFi-like interferer.
    n_taps_int, wc_int
        Filter parameters for the interfering DCSK transmitter.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(sig)
    p_sig = float(np.mean(sig**2))
    rx = awgn(sig, snr_db, rng)

    # DCSK interferer
    bits_int = rng.integers(0, 2, n_bits)
    sig_int = dcsk_transmit(bits_int, beta=beta, n_taps=n_taps_int, wc=wc_int)
    sig_int = sig_int[:N] if len(sig_int) >= N else np.pad(sig_int, (0, N - len(sig_int)))
    p_int = float(np.mean(sig_int**2))
    esc_dcsk = np.sqrt(p_sig / (p_int * 10 ** (sir_dcsk_db / 10)))
    rx += esc_dcsk * sig_int

    # WiFi-like interferer
    sig_wifi = _wifi_interferer(N, rng=rng)
    p_wifi = float(np.mean(sig_wifi**2))
    esc_wifi = np.sqrt(p_sig / (p_wifi * 10 ** (sir_wifi_db / 10)))
    rx += esc_wifi * sig_wifi

    return rx


def channel_urban(
    sig: NDArray,
    snr_db: float,
    prob_impulso: float = 0.01,
    amp_fator: float = 10.0,
    delays: list[int] | None = None,
    gains: list[float] | None = None,
    sir_dcsk_db: float = 15.0,
    sir_wifi_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Combined urban channel: impulsive noise + multipath + interferers.

    Applies all three impairments simultaneously for realistic testing.
    """
    if delays is None:
        delays = [0, 3, 7, 15]
    if gains is None:
        gains = [1.0, 0.6, 0.4, 0.2]
    if rng is None:
        rng = np.random.default_rng()

    rx = channel_impulsive(sig, snr_db, prob_impulso, amp_fator, rng)
    rx = channel_multipath(rx, snr_db, delays, gains, rng)
    return channel_interferers(rx, snr_db, sir_dcsk_db, sir_wifi_db, rng=rng)

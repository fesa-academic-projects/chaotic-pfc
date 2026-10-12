"""Error metrics for the chaotic communication experiments.

The band-limited channel is a linear-phase FIR filter, so it delays the
received signal by its group delay ``(N_taps - 1) / 2`` samples. The
recovered message is therefore compared against the original message
after removing that known delay, otherwise the misalignment dominates
the error metrics (and, for a ±1 message, the sign error sits near 0.5
regardless of the true quality of the recovery).

Nothing here changes the transmitter, the channel or the receiver: this
module only aligns the two signals before measuring.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class DelayEstimate(TypedDict):
    """Cross-correlation delay estimate between ``m`` and ``m_hat``."""

    lag: int
    corr_at_lag: float
    corr_at_zero: float


class CommMetrics(TypedDict):
    """Error metrics for one communication experiment."""

    delay: int
    n_samples: int
    mse: float
    per_sample_error: float
    ber: float
    correlation: float


def estimate_delay(
    m: NDArray,
    m_hat: NDArray,
    *,
    transient: int,
    max_lag: int = 500,
) -> DelayEstimate:
    """Estimate the ``m`` to ``m_hat`` delay by cross-correlation.

    Lags from 0 to *max_lag* are scanned; the returned ``lag`` maximises
    ``|corr|``. The sign of ``corr_at_lag`` is reported so that a polarity
    inversion is not mistaken for a delay.

    Note that when the recovery fails (``corr_at_lag`` is near zero, as in
    a band-limited channel with no feedback filter), the argmax lag fits
    noise and drifts with the record length; the lag is only meaningful
    once there is a signal to correlate against. Callers should treat a
    small ``corr_at_lag`` as "no delay estimate", not as a measurement.

    Parameters
    ----------
    m, m_hat
        Original and recovered message, shape ``(N,)``.
    transient
        Number of leading samples discarded before correlating (the
        synchronisation transient).
    max_lag
        Largest delay scanned, in samples (default 500).

    Returns
    -------
    DelayEstimate
        ``lag`` (argmax of ``|corr|``), ``corr_at_lag`` and
        ``corr_at_zero``.
    """
    m = np.asarray(m, dtype=float)
    m_hat = np.asarray(m_hat, dtype=float)
    length = min(m.size, m_hat.size) - transient - max_lag
    if length <= 0:
        raise ValueError("signals are too short for the requested lag scan")
    ref = m[transient : transient + length]
    ref = ref - ref.mean()
    ref_norm = float(np.linalg.norm(ref))

    best_lag = 0
    best_corr = 0.0
    for lag in range(max_lag + 1):
        seg = m_hat[transient + lag : transient + lag + length]
        seg = seg - seg.mean()
        denom = ref_norm * float(np.linalg.norm(seg))
        corr = float(np.dot(ref, seg) / denom) if denom else 0.0
        if abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, lag

    seg0 = m_hat[transient : transient + length]
    corr0 = float(np.corrcoef(ref, seg0)[0, 1])
    return {"lag": best_lag, "corr_at_lag": best_corr, "corr_at_zero": corr0}


def comm_metrics(
    m: NDArray,
    m_hat: NDArray,
    *,
    transient: int,
    bit_period: int,
    delay: int,
) -> CommMetrics:
    """MSE, per-sample error, per-bit error rate and correlation.

    ``m_hat[n]`` is assumed to reproduce ``m[n - delay]``, so the samples
    compared are ``m[n]`` against ``m_hat[n + delay]`` for
    ``n >= transient``. The *transient* value itself is not modified; the
    alignment only shifts the comparison window.

    Parameters
    ----------
    m, m_hat
        Original and recovered message, shape ``(N,)``.
    transient
        Number of leading samples discarded (synchronisation transient).
    bit_period
        Samples per transmitted bit (the bit hold time).
    delay
        Delay of ``m_hat`` with respect to ``m``, in samples.

    Returns
    -------
    CommMetrics
        ``delay``, ``n_samples``, ``mse``, ``per_sample_error``, ``ber``
        and ``correlation``.
    """
    m = np.asarray(m, dtype=float)
    m_hat = np.asarray(m_hat, dtype=float)
    d = int(delay)
    stop = m.size - d
    ref = m[transient:stop]
    got = m_hat[transient + d : stop + d]
    if ref.size == 0:
        raise ValueError("no samples left after applying the delay")

    mse = float(np.mean((ref - got) ** 2))
    per_sample = float(np.mean(np.sign(got) != np.sign(ref)))
    corr = float(np.corrcoef(ref, got)[0, 1]) if ref.size > 1 else float("nan")

    n_bit = ref.size // bit_period
    bits_ref = ref[: n_bit * bit_period].reshape(n_bit, bit_period).mean(axis=1)
    bits_got = got[: n_bit * bit_period].reshape(n_bit, bit_period).mean(axis=1)
    ber = float(np.mean(np.sign(bits_got) != np.sign(bits_ref)))

    return {
        "delay": d,
        "n_samples": int(ref.size),
        "mse": mse,
        "per_sample_error": per_sample,
        "ber": ber,
        "correlation": corr,
    }

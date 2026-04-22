"""
receiver.py
===========
Chaos-synchronisation demodulators that extract the original message
from a transmitted (and possibly channel-distorted) carrier.

Both receivers rely on the Pecora-Carroll synchronisation principle:
running a second copy of the Hénon oscillator driven by the received
signal causes it to track the transmitter's state after a short
transient. The recovered message is then extracted as the difference
between the driving signal and the local state estimate.

Two receivers are provided, mirroring :mod:`chaotic_pfc.transmitter`:

* :func:`receive` — 2-D Hénon demodulator; the message is

  .. math::

      \\hat{m}[n] = \\frac{r[n] - y_1[n]}{\\mu}

  where ``y_1`` is the local 2-D state driven by ``r``.

* :func:`receive_order_n` — higher-order demodulator using the same
  FIR filter the transmitter used. The carrier is taken from the
  filtered state ``y[2][n]`` and the message follows the same formula
  with ``y_1`` replaced by ``v = y[2]``.
"""

import numpy as np
from numpy.typing import NDArray

from .maps import _henon_n4_step


def receive(
    r: NDArray,
    mu: float = 0.01,
    a: float = 1.4,
    b: float = 0.3,
    y0: float = 0.0,
    z0: float = 0.0,
) -> NDArray:
    """Demodulate the carrier ``r`` via 2-D Hénon synchronisation.

    The local oscillator is driven by ``r`` itself:

    .. math::

        y_1[n+1] &= a - r[n]^2 + b \\, y_2[n] \\\\
        y_2[n+1] &= y_1[n]

    This is the Pecora-Carroll trick: once ``y`` locks onto the
    transmitter's state, ``r[n] - y_1[n]`` equals the instantaneous
    modulation and dividing by ``μ`` recovers the message.

    Parameters
    ----------
    r
        Received carrier, shape ``(N,)``. Usually the output of
        :func:`chaotic_pfc.channel.ideal_channel` or
        :func:`chaotic_pfc.channel.fir_channel`.
    mu
        Modulation depth — must match the value used at the
        transmitter, otherwise the recovered amplitude is rescaled.
    a, b
        Hénon parameters — must match the transmitter's.
    y0, z0
        Initial conditions of the local oscillator. A random pair is a
        safe default; synchronisation converges regardless after a
        short transient.

    Returns
    -------
    ndarray, shape (N,)
        Recovered message estimate ``m̂[n] = (r[n] - y_1[n]) / μ``.
        Transient samples (first few hundred) may differ from the true
        message while the local oscillator locks in; see
        :data:`chaotic_pfc.config.CommConfig.transient` for the default
        rejection window.
    """
    N = len(r)
    y1 = np.empty(N + 1)
    y2 = np.empty(N + 1)
    y1[0], y2[0] = y0, z0
    m_hat = np.empty(N)

    for k in range(N):
        y1[k + 1] = a - r[k] ** 2 + b * y2[k]
        y2[k + 1] = y1[k]
        m_hat[k] = (r[k] - y1[k]) / mu
    return m_hat


def receive_order_n(
    r: NDArray,
    fir_coeffs: NDArray,
    mu: float = 0.01,
    a: float = 1.4,
    b: float = 0.3,
    y0: NDArray | None = None,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Demodulate the carrier ``r`` via N-th order Hénon synchronisation.

    Runs a local copy of the N-th order filtered Hénon map driven by
    ``r``, and recovers the message from the difference between the
    received sample and the local filtered state ``v = y[2]``:

    .. math::

        \\hat{m}[n] = \\frac{r[n] - v[n]}{\\mu}

    Parameters
    ----------
    r
        Received carrier, shape ``(N,)``.
    fir_coeffs
        FIR feedback coefficients, shape ``(Nc,)``. Must match the
        ones used at the transmitter — otherwise synchronisation
        fails.
    mu, a, b
        Must match the transmitter's parameters.
    y0
        Optional explicit initial state, shape ``(Nc,)``. If omitted, a
        random state is drawn from ``Uniform(0, 0.5)``.
    seed
        Seed for the RNG used when ``y0=None``. Has no effect when
        ``y0`` is provided.

    Returns
    -------
    m_hat : ndarray, shape (N,)
        Recovered message estimate.
    state : ndarray, shape (Nc, N + 1)
        Full state trajectory of the local oscillator. Column 0 holds
        ``y0``; each subsequent column is the next iterate.
    """
    N = len(r)
    Nc = len(fir_coeffs)
    rng = np.random.default_rng(seed)
    c = np.asarray(fir_coeffs, dtype=float)

    state = np.zeros((Nc, N + 1))
    state[:, 0] = y0 if y0 is not None else 0.5 * rng.random(Nc)
    m_hat = np.empty(N)

    for i in range(N):
        v = state[2, i]
        m_hat[i] = (r[i] - v) / mu
        state[:, i + 1] = _henon_n4_step(state[:, i], r[i], a, b, c)
    return m_hat, state

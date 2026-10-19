r"""Chaos-synchronisation demodulator implementing the receiver protocol.

The receiver is a callable that extracts the original message from a
transmitted (and possibly channel-distorted) carrier.

Both receivers rely on the Pecora-Carroll synchronisation principle:
running a second copy of the Hénon oscillator driven by the received
signal causes it to track the transmitter's state after a short
transient. The recovered message is then extracted as the difference
between the driving signal and the local state estimate.

Two receivers are provided, mirroring
:mod:`chaotic_pfc.comms.transmitter`:

* :func:`receive` — 2-D Hénon demodulator. The message is recovered via

  .. math::

      \\hat{m}[n] = \\frac{r[n] - y_1[n]}{\\mu}

  where ``y_1`` is the local 2-D state driven by ``r``.

* :func:`receive_order_n` — higher-order demodulator. Uses the same
  FIR filter the transmitter used. The carrier is taken from the
  filtered state ``y[2][n]`` and the message follows the same formula
  with ``y_1`` replaced by ``v = y[2]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc._validate import (
    DivergencePolicy,
    handle_divergence,
    require_1d_nonempty,
    require_divergence_policy,
    require_finite,
    require_seed,
)
from chaotic_pfc.errors import InvalidParameterError

from ..dynamics.maps import _henon_n4_step_inplace


def receive(
    r: NDArray[np.float64],
    *,
    mu: float = 0.01,
    alpha: float = 1.4,
    beta: float = 0.3,
    y0: float = 0.0,
    z0: float = 0.0,
    on_divergence: DivergencePolicy = "nan",
) -> NDArray[np.float64]:
    r"""Demodulate the carrier ``r`` via 2-D Hénon synchronisation.

    The local oscillator is driven by ``r`` itself:

    .. math::

        y_1[n+1] &= \\alpha - r[n]^2 + \\beta \\, y_2[n] \\\\
        y_2[n+1] &= y_1[n]

    This is the Pecora-Carroll trick: once ``y`` locks onto the
    transmitter's state, ``r[n] - y_1[n]`` equals the instantaneous
    modulation and dividing by ``μ`` recovers the message.

    Parameters
    ----------
    r
        Received carrier, shape ``(N,)``. Usually the output of
        :func:`chaotic_pfc.comms.channel.ideal_channel` or
        :func:`chaotic_pfc.comms.channel.fir_channel`.
    mu
        Modulation depth — must match the value used at the
        transmitter, otherwise the recovered amplitude is rescaled.
    alpha, beta
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

    Implements: :class:`~chaotic_pfc.comms.protocols.Receiver`.

    Examples
    --------
    With the transmitter and the local oscillator started from the same state
    the Pecora-Carroll receiver locks immediately and recovers a clean message:

    >>> m = binary_message(4, period=4)
    >>> m_hat = receive(transmit(m, x0=0.1, y0=0.1), y0=0.1, z0=0.1)
    >>> np.allclose(m_hat, m)
    True

    Raises
    ------
    InvalidParameterError
        If ``r`` is not a non-empty 1-D finite array, ``mu == 0``, or
        any scalar parameter is non-finite.
    """
    r_arr = require_1d_nonempty("r", r)
    mu = require_finite("mu", mu)
    if mu == 0:
        raise InvalidParameterError(
            "mu must be non-zero; received mu=0 which would cause "
            "division by zero in the demodulation step"
        )
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    y0 = require_finite("y0", y0)
    z0 = require_finite("z0", z0)
    policy = require_divergence_policy(on_divergence)
    N = len(r_arr)
    y1 = np.empty(N + 1)
    y2 = np.empty(N + 1)
    y1[0], y2[0] = y0, z0
    m_hat = np.empty(N)

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(N):
            y1[k + 1] = alpha - r_arr[k] ** 2 + beta * y2[k]
            y2[k + 1] = y1[k]
            m_hat[k] = (r_arr[k] - y1[k]) / mu
    if policy != "nan" and not np.all(np.isfinite(m_hat)):
        bad_idx = int(np.flatnonzero(~np.isfinite(m_hat))[0])
        handle_divergence(
            policy,
            f"receive(mu={mu}, alpha={alpha}, beta={beta}) diverged at sample {bad_idx}",
        )
    return m_hat


def receive_order_n(
    r: NDArray[np.float64],
    fir_coeffs: NDArray[np.float64],
    *,
    mu: float = 0.01,
    alpha: float = 1.4,
    beta: float = 0.3,
    y0: NDArray[np.float64] | None = None,
    seed: int | None = None,
    on_divergence: DivergencePolicy = "nan",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Demodulate the carrier ``r`` via N-th order Hénon synchronisation.

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
    mu, alpha, beta
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

    Note: this function does NOT implement
    :class:`~chaotic_pfc.comms.protocols.Receiver` because it returns a
    ``(recovered, state)`` tuple instead of a single ndarray. Use
    :func:`receive` (the 2-D variant) for code that types against the
    Receiver protocol.

    Raises
    ------
    InvalidParameterError
        If ``r`` or ``fir_coeffs`` is not a non-empty 1-D finite array,
        ``len(fir_coeffs) < 3``, ``mu == 0``, ``y0`` has incompatible
        shape, or any scalar parameter is non-finite.
    """
    r_arr = require_1d_nonempty("r", r)
    c = require_1d_nonempty("fir_coeffs", fir_coeffs, min_length=3)
    Nc = len(c)
    mu = require_finite("mu", mu)
    if mu == 0:
        raise InvalidParameterError(
            "mu must be non-zero; received mu=0 which would cause "
            "division by zero in the demodulation step"
        )
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    seed_val = require_seed("seed", seed)
    policy = require_divergence_policy(on_divergence)
    if y0 is not None:
        y0_arr = require_1d_nonempty("y0", y0)
        if len(y0_arr) != Nc:
            raise InvalidParameterError(
                f"y0 must have length {Nc} matching fir_coeffs, got {len(y0_arr)}"
            )
    else:
        y0_arr = None
    N = len(r_arr)
    rng = np.random.default_rng(seed_val)

    state = np.zeros((Nc, N + 1))
    state[:, 0] = y0_arr if y0_arr is not None else 0.5 * rng.random(Nc)
    m_hat = np.empty(N)

    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(N):
            v = state[2, i]
            m_hat[i] = (r_arr[i] - v) / mu
            _henon_n4_step_inplace(state[:, i + 1], state[:, i], r_arr[i], alpha, beta, c)
    if policy != "nan" and not (np.all(np.isfinite(m_hat)) and np.all(np.isfinite(state))):
        bad_cols = np.flatnonzero(~np.all(np.isfinite(state), axis=0))
        first_step = int(bad_cols[0]) if bad_cols.size else N
        handle_divergence(
            policy,
            f"receive_order_n(Nc={Nc}, mu={mu}, alpha={alpha}, beta={beta}) diverged at step {first_step}",
        )
    return m_hat, state

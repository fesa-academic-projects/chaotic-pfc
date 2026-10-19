r"""Chaos-based modulator implementing the transmitter protocol.

The transmitter is a callable that embeds a message in the state of a
Hénon oscillator.

Two transmitters are provided, mirroring the two receiver modes:

* :func:`transmit` — 2-D Hénon carrier. The message ``m[n]`` is added
  to the state ``x₁[n]`` before it feeds back into the next iteration,
  so the emitted carrier is

  .. math::

      s[n] = x_1[n] + \\mu \\, m[n]

  and the updated state becomes
  ``x_1[n+1] = \\alpha - s[n]^2 + \\beta \\, x_2[n]``.

* :func:`transmit_order_n` — higher-order Hénon with an internal FIR
  filter whose taps form the coefficient vector ``c``. The carrier is
  taken from the filtered state component ``x[2][n]``.

Both functions are deterministic given their inputs. Where a random
initial condition is needed (``transmit_order_n`` with ``x0=None``), a
``seed`` parameter exposes the RNG so callers can reproduce results.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc._validate import require_1d_nonempty, require_finite, require_seed
from chaotic_pfc.errors import InvalidParameterError

from ..dynamics.maps import _henon_n4_step_inplace


def transmit(
    message: NDArray[np.float64],
    *,
    mu: float = 0.01,
    alpha: float = 1.4,
    beta: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
) -> NDArray[np.float64]:
    r"""Modulate a message onto a 2-D Hénon carrier.

    Parameters
    ----------
    message
        Message samples ``m[n]``, shape ``(N,)``. Typically produced by
        :func:`chaotic_pfc.dynamics.signals.binary_message`.
    mu
        Modulation depth. Small values (≪ 1) keep the Hénon dynamics
        near their autonomous regime, which is important for reliable
        chaos synchronisation at the receiver. The default ``0.01``
        matches the convention used throughout the project.
    alpha, beta
        Hénon parameters. The canonical chaotic regime is ``(1.4, 0.3)``.
    x0, y0
        Initial conditions of the transmitter's internal state.

    Returns
    -------
    ndarray, shape (N,)
        The transmitted carrier ``s[n] = x_1[n] + μ · m[n]``.

    Notes
    -----
    The recurrence implemented is

    .. math::

        s[n]       &= x_1[n] + \\mu \\, m[n] \\\\
        x_1[n+1]   &= \\alpha - s[n]^2 + \\beta \\, x_2[n] \\\\
        x_2[n+1]   &= x_1[n]

    The output ``s`` has the same chaotic broadband spectrum as the
    autonomous Hénon map for ``μ ≪ 1``, with the message imprinted as a
    small spectral perturbation.

    Implements: :class:`~chaotic_pfc.comms.protocols.Transmitter`.

    Examples
    --------
    >>> s = transmit(binary_message(4, period=4), x0=0.1, y0=0.1)
    >>> [round(float(v), 4) for v in s]
    [0.11, 1.4279, -0.6189, 1.4323]

    Raises
    ------
    InvalidParameterError
        If ``message`` is not a non-empty 1-D finite array or any
        scalar parameter is non-finite.
    """
    msg = require_1d_nonempty("message", message)
    mu = require_finite("mu", mu)
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    x0 = require_finite("x0", x0)
    y0 = require_finite("y0", y0)
    N = len(msg)
    x1 = np.empty(N + 1)
    x2 = np.empty(N + 1)
    x1[0], x2[0] = x0, y0
    s = np.empty(N)

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(N):
            s[k] = x1[k] + mu * msg[k]
            x1[k + 1] = alpha - s[k] ** 2 + beta * x2[k]
            x2[k + 1] = x1[k]
    return s


def transmit_order_n(
    message: NDArray[np.float64],
    fir_coeffs: NDArray[np.float64],
    *,
    mu: float = 0.01,
    alpha: float = 1.4,
    beta: float = 0.3,
    x0: NDArray[np.float64] | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Modulate a message onto an N-th order filtered Hénon carrier.

    The carrier is taken from the filtered state component ``x[2][n]``,
    producing a signal with a richer spectrum than the 2-D variant. The
    FIR coefficients in ``fir_coeffs`` shape the feedback path and
    directly control the Lyapunov spectrum of the system.

    Parameters
    ----------
    message
        Message samples, shape ``(N,)``.
    fir_coeffs
        FIR feedback coefficients ``c``, shape ``(Nc,)``. ``Nc`` must be
        at least 3. Usually produced by :func:`scipy.signal.firwin`.
    mu
        Modulation depth. Same semantics as in :func:`transmit`.
    alpha, beta
        Hénon parameters.
    x0
        Optional explicit initial state, shape ``(Nc,)``. If omitted, a
        random state is drawn from ``Uniform(0, 0.5)`` using ``seed``.
    seed
        Seed for the RNG that produces ``x0`` when ``x0=None``. Has no
        effect if ``x0`` is provided.

    Returns
    -------
    s : ndarray, shape (N,)
        Transmitted carrier ``s[n] = v[n] + μ · m[n]`` where
        ``v[n] = x[2][n]``.
    state : ndarray, shape (Nc, N + 1)
        Full state trajectory of the N-th order system. Column ``n``
        holds the state at step ``n``, with column 0 equal to ``x0``.

    Note: this function does NOT implement
    :class:`~chaotic_pfc.comms.protocols.Transmitter` because it returns a
    ``(signal, state)`` tuple instead of a single ndarray. Use
    :func:`transmit` (the 2-D variant) for code that types against the
    Transmitter protocol.

    Raises
    ------
    InvalidParameterError
        If ``message`` or ``fir_coeffs`` is not a non-empty 1-D finite
        array, ``len(fir_coeffs) < 3``, ``x0`` has incompatible shape,
        or any scalar parameter is non-finite.
    """
    msg = require_1d_nonempty("message", message)
    c = require_1d_nonempty("fir_coeffs", fir_coeffs, min_length=3)
    Nc = len(c)
    mu = require_finite("mu", mu)
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    seed_val = require_seed("seed", seed)
    if x0 is not None:
        x0_arr = require_1d_nonempty("x0", x0)
        if len(x0_arr) != Nc:
            raise InvalidParameterError(
                f"x0 must have length {Nc} matching fir_coeffs, got {len(x0_arr)}"
            )
    else:
        x0_arr = None
    rng = np.random.default_rng(seed_val)

    N = len(msg)
    state = np.zeros((Nc, N + 1))
    state[:, 0] = x0_arr if x0_arr is not None else 0.5 * rng.random(Nc)
    s = np.empty(N)

    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(N):
            v = state[2, i]
            s[i] = v + mu * msg[i]
            _henon_n4_step_inplace(state[:, i + 1], state[:, i], s[i], alpha, beta, c)
    return s, state

"""
transmitter.py
==============
Chaos-based modulators that embed a message in the state of a Hénon
oscillator.

Two transmitters are provided, mirroring the two receiver modes:

* :func:`transmit` — 2-D Hénon carrier. The message ``m[n]`` is added
  to the state ``x₁[n]`` before it feeds back into the next iteration,
  so the emitted carrier is

  .. math::

      s[n] = x_1[n] + \\mu \\, m[n]

  and the updated state becomes
  ``x_1[n+1] = a - s[n]^2 + b \\, x_2[n]``.

* :func:`transmit_order_n` — higher-order Hénon with an internal FIR
  filter whose taps form the coefficient vector ``c``. The carrier is
  taken from the filtered state component ``x[2][n]``.

Both functions are deterministic given their inputs. Where a random
initial condition is needed (``transmit_order_n`` with ``x0=None``), a
``seed`` parameter exposes the RNG so callers can reproduce results.
"""

import numpy as np
from numpy.typing import NDArray

from .maps import _henon_n4_step


def transmit(
    message: NDArray,
    mu: float = 0.01,
    a: float = 1.4,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
) -> NDArray:
    """Modulate a message onto a 2-D Hénon carrier.

    Parameters
    ----------
    message
        Message samples ``m[n]``, shape ``(N,)``. Typically produced by
        :func:`chaotic_pfc.signals.binary_message`.
    mu
        Modulation depth. Small values (≪ 1) keep the Hénon dynamics
        near their autonomous regime, which is important for reliable
        chaos synchronisation at the receiver. The default ``0.01``
        matches the convention used throughout the TCC.
    a, b
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
        x_1[n+1]   &= a - s[n]^2 + b \\, x_2[n] \\\\
        x_2[n+1]   &= x_1[n]

    The output ``s`` has the same chaotic broadband spectrum as the
    autonomous Hénon map for ``μ ≪ 1``, with the message imprinted as a
    small spectral perturbation.
    """
    N = len(message)
    x1 = np.empty(N + 1)
    x2 = np.empty(N + 1)
    x1[0], x2[0] = x0, y0
    s = np.empty(N)

    for k in range(N):
        s[k] = x1[k] + mu * message[k]
        x1[k + 1] = a - s[k] ** 2 + b * x2[k]
        x2[k + 1] = x1[k]
    return s


def transmit_order_n(
    message: NDArray,
    fir_coeffs: NDArray,
    mu: float = 0.01,
    a: float = 1.4,
    b: float = 0.3,
    x0: NDArray | None = None,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
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
    a, b
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
    """
    Nc = len(fir_coeffs)
    rng = np.random.default_rng(seed)
    c = np.asarray(fir_coeffs, dtype=float)

    N = len(message)
    state = np.zeros((Nc, N + 1))
    state[:, 0] = x0 if x0 is not None else 0.5 * rng.random(Nc)
    s = np.empty(N)

    for i in range(N):
        v = state[2, i]
        s[i] = v + mu * message[i]
        state[:, i + 1] = _henon_n4_step(state[:, i], s[i], a, b, c)
    return s, state

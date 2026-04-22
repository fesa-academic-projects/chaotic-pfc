"""
maps.py
=======
Hénon map variants used as chaotic oscillators throughout the
communication system.

Three 2-D maps and one generalised N-th order map are provided:

* :func:`henon_standard` — the original Hénon (1976) map,

  .. math::

      x[n+1] &= 1 - a \\, x[n]^2 + y[n] \\\\
      y[n+1] &= b \\, x[n]

  chaotic at ``(a, b) = (1.4, 0.3)``.

* :func:`henon_generalised` — the parametric form preferred in
  communications literature,

  .. math::

      x_1[n+1] &= \\alpha - x_1[n]^2 + \\beta \\, x_2[n] \\\\
      x_2[n+1] &= x_1[n]

  equivalent to :func:`henon_standard` with rescaled variables.

* :func:`henon_filtered` — generalised Hénon whose nonlinear input is
  a weighted combination of two consecutive state samples:

  .. math::

      x_1[n+1] &= \\alpha - (c_0 \\, x_1[n] + c_1 \\, x_2[n])^2 + \\beta \\, x_2[n] \\\\
      x_2[n+1] &= x_1[n]

  This is the 2-tap case of the general filtered map.

* :func:`henon_order_n` — general case with an FIR filter of arbitrary
  order acting on the system's state vector. Used by the higher-order
  transmitter/receiver pair and by the Lyapunov sweep.
"""

import numpy as np
from numpy.typing import NDArray

# ── 2-D maps ────────────────────────────────────────────────────────────────


def henon_standard(
    steps: int,
    x0: float = 0.0,
    y0: float = 0.0,
    a: float = 1.4,
    b: float = 0.3,
) -> tuple[NDArray, NDArray]:
    """Iterate the standard 2-D Hénon map.

    Parameters
    ----------
    steps
        Number of iterations. Each output array has length
        ``steps + 1`` (initial condition included).
    x0, y0
        Initial conditions of the two state variables.
    a, b
        Hénon parameters. The canonical chaotic regime is
        ``(a, b) = (1.4, 0.3)``; smaller ``a`` yields periodic orbits,
        larger ``a`` yields unbounded trajectories for most ICs.

    Returns
    -------
    X, Y : ndarray, shape (steps + 1,)
        Trajectories of ``x`` and ``y`` from ``n = 0`` (the initial
        condition) to ``n = steps``.
    """
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    for i in range(steps):
        X[i + 1] = 1.0 - a * X[i] ** 2 + Y[i]
        Y[i + 1] = b * X[i]
    return X, Y


def henon_generalised(
    steps: int,
    x0: float = 0.0,
    y0: float = 0.0,
    alpha: float = 1.4,
    beta: float = 0.3,
) -> tuple[NDArray, NDArray]:
    """Iterate the generalised 2-D Hénon map.

    Mathematically equivalent to :func:`henon_standard` under a change
    of variables, but parameterised using ``α`` and ``β`` because that
    is the form used in the TCC theoretical derivations.

    Parameters
    ----------
    steps
        Number of iterations.
    x0, y0
        Initial conditions of the state variables.
    alpha, beta
        Hénon parameters. Chaotic regime at ``(α, β) = (1.4, 0.3)``.

    Returns
    -------
    X, Y : ndarray, shape (steps + 1,)
        Trajectories of ``x_1`` and ``x_2``.
    """
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    for i in range(steps):
        X[i + 1] = alpha - X[i] ** 2 + beta * Y[i]
        Y[i + 1] = X[i]
    return X, Y


def henon_filtered(
    steps: int,
    x0: float = 0.0,
    y0: float = 0.0,
    alpha: float = 1.4,
    beta: float = 0.3,
    c0: float = 0.7,
    c1: float = 0.3,
) -> tuple[NDArray, NDArray]:
    """Iterate the 2-tap filtered Hénon map.

    The quadratic nonlinearity acts on a weighted combination of the
    two most recent state samples, ``z = c_0 · x_1 + c_1 · x_2``,
    rather than on a single sample. This is the smallest non-trivial
    case of :func:`henon_order_n`.

    Parameters
    ----------
    steps
        Number of iterations.
    x0, y0
        Initial conditions of the state variables.
    alpha, beta
        Hénon parameters.
    c0, c1
        FIR filter coefficients. With ``(c0, c1) = (1, 0)`` the map
        reduces exactly to :func:`henon_generalised` — this equivalence
        is exercised by ``TestHenonFiltered`` in the test suite.

    Returns
    -------
    X, Y : ndarray, shape (steps + 1,)
        Trajectories of ``x_1`` and ``x_2``.
    """
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    for i in range(steps):
        z = c0 * X[i] + c1 * Y[i]
        X[i + 1] = alpha - z**2 + beta * Y[i]
        Y[i + 1] = X[i]
    return X, Y


# ── N-th order map (vectorised state) ──────────────────────────────────────


def _henon_n4_step(
    x: NDArray,
    s: float,
    a: float,
    b: float,
    c: NDArray,
) -> NDArray:
    """Single step of the N-th order filtered Hénon map.

    Implements one iteration of the ``N_c``-dimensional system whose
    state vector has length ``N_c = len(c)``. The carrier output of
    the oscillator is the filtered component ``x[2]``, driven by the
    FIR coefficients ``c``.

    Parameters
    ----------
    x
        Current state, shape ``(Nc,)``.
    s
        Current driving value. When called autonomously, this is
        ``x[2]`` itself; when called from the transmitter/receiver, it
        is the carrier value seen by the channel.
    a, b
        Hénon parameters.
    c
        FIR coefficients, shape ``(Nc,)``. ``Nc`` must be at least 3.

    Returns
    -------
    ndarray, shape (Nc,)
        The updated state ``x[n+1]``.

    Notes
    -----
    This is a private helper used by :func:`henon_order_n`,
    :func:`chaotic_pfc.transmitter.transmit_order_n`, and
    :func:`chaotic_pfc.receiver.receive_order_n`. Not part of the
    public API.
    """
    Nc = len(c)
    out = np.empty(Nc)

    x1_new = a - s**2 + b * x[1]
    x2_new = x[0]

    v = c[0] * x1_new + c[1] * x[0] + c[2] * x[1]
    if Nc > 3:
        v += float(np.dot(c[3:], x[3:]))

    out[0] = x1_new
    out[1] = x2_new
    out[2] = v
    if Nc > 3:
        out[3] = x[1]
    if Nc > 4:
        out[4:] = x[3 : Nc - 1]
    return out


def henon_order_n(
    steps: int,
    fir_coeffs: NDArray,
    x0: NDArray | None = None,
    a: float = 1.4,
    b: float = 0.3,
    driving: NDArray | None = None,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Iterate the N-th order Hénon map with an internal FIR filter.

    The system dimension ``N_c`` is inferred from ``len(fir_coeffs)``.
    At each step, the carrier output is the filtered state component
    ``x[2]``, and the next iterate is computed by
    :func:`_henon_n4_step`.

    The ``driving`` parameter lets callers override the nonlinear
    input: when ``driving=None`` the map runs autonomously (``s = x[2]``);
    when ``driving`` is provided the map is forced by that external
    signal. This is used by the transmitter (self-driven) vs. the
    receiver (driven by the received signal).

    Parameters
    ----------
    steps
        Number of iterations. The output state has shape
        ``(Nc, steps + 1)``.
    fir_coeffs
        FIR filter coefficients, shape ``(Nc,)``. Must have ``Nc >= 3``.
    x0
        Optional explicit initial state, shape ``(Nc,)``. If ``None``,
        a random state is drawn from ``Uniform(0, 0.5)`` per component.
    a, b
        Hénon parameters.
    driving
        Optional external driving signal, shape ``(steps,)``. When
        provided, replaces the self-driving term at each iteration.
    seed
        RNG seed used when ``x0=None`` to pick the random initial
        condition. Has no effect if ``x0`` is provided.

    Returns
    -------
    state : ndarray, shape (Nc, steps + 1)
        Full state trajectory. Column 0 is the initial condition.
    output : ndarray, shape (steps,)
        Carrier signal ``x[2][n]`` for ``n = 0, …, steps − 1``.
    """
    Nc = len(fir_coeffs)
    rng = np.random.default_rng(seed)
    c = np.asarray(fir_coeffs, dtype=float)

    state = np.zeros((Nc, steps + 1))
    state[:, 0] = x0 if x0 is not None else 0.5 * rng.random(Nc)
    output = np.empty(steps)

    for i in range(steps):
        s_i = float(driving[i]) if driving is not None else state[2, i]
        output[i] = state[2, i]
        state[:, i + 1] = _henon_n4_step(state[:, i], s_i, a, b, c)

    return state, output

r"""Hénon map variants used as chaotic oscillators in the communication system.

Four Henon map variants and one chaotic-sequence generator are provided:

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

* :func:`henon_fir_sequence` — generates a chaotic sequence from the
  FIR-filtered Hénon map.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin

from chaotic_pfc._validate import (
    DivergencePolicy,
    handle_divergence,
    require_1d_nonempty,
    require_divergence_policy,
    require_finite,
    require_in_range,
    require_positive_int,
    require_seed,
)
from chaotic_pfc.errors import InvalidParameterError

# ── 2-D maps ────────────────────────────────────────────────────────────────


def henon_standard(
    steps: int,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    alpha: float = 1.4,
    beta: float = 0.3,
    on_divergence: DivergencePolicy = "nan",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iterate the standard 2-D Hénon map.

    Parameters
    ----------
    steps
        Number of iterations. Each output array has length
        ``steps + 1`` (initial condition included).
    x0, y0
        Initial conditions of the two state variables.
    alpha, beta
        Hénon parameters. The canonical chaotic regime is
        ``(alpha, beta) = (1.4, 0.3)``; smaller ``alpha`` yields periodic
        orbits, larger ``alpha`` yields unbounded trajectories for most ICs.

    Examples
    --------
    >>> X, _ = henon_standard(4, x0=0.1, y0=0.1)
    >>> [round(float(v), 4) for v in X]
    [0.1, 1.086, -0.6212, 0.7856, -0.0505]

    Returns
    -------
    X, Y : ndarray, shape (steps + 1,)
        Trajectories of ``x`` and ``y`` from ``n = 0`` (the initial
        condition) to ``n = steps``.

    Raises
    ------
    InvalidParameterError
        If ``steps`` is not a positive integer or any scalar parameter
        is non-finite.
    DivergenceError
        If the trajectory diverges and ``on_divergence='raise'``.
    """
    steps = require_positive_int("steps", steps, min_value=1)
    x0 = require_finite("x0", x0)
    y0 = require_finite("y0", y0)
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    policy = require_divergence_policy(on_divergence)
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(steps):
            xn = 1.0 - alpha * X[i] ** 2 + Y[i]
            yn = beta * X[i]
            if not (np.isfinite(xn) and np.isfinite(yn)):
                X[i + 1 :] = np.nan
                Y[i + 1 :] = np.nan
                handle_divergence(
                    policy,
                    f"henon_standard(alpha={alpha}, beta={beta}) diverged at step {i + 1}",
                )
                break
            X[i + 1] = xn
            Y[i + 1] = yn
    return X, Y


def henon_generalised(
    steps: int,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    alpha: float = 1.4,
    beta: float = 0.3,
    on_divergence: DivergencePolicy = "nan",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iterate the generalised 2-D Hénon map.

    Mathematically equivalent to :func:`henon_standard` under a change
    of variables, but parameterised using ``α`` and ``β`` because that
    is the form used in the theoretical derivations of the project.

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

    Raises
    ------
    InvalidParameterError
        If ``steps`` is not a positive integer or any scalar parameter
        is non-finite.
    DivergenceError
        If the trajectory diverges and ``on_divergence='raise'``.
    """
    steps = require_positive_int("steps", steps, min_value=1)
    x0 = require_finite("x0", x0)
    y0 = require_finite("y0", y0)
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    policy = require_divergence_policy(on_divergence)
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(steps):
            xn = alpha - X[i] ** 2 + beta * Y[i]
            yn = X[i]
            if not (np.isfinite(xn) and np.isfinite(yn)):
                X[i + 1 :] = np.nan
                Y[i + 1 :] = np.nan
                handle_divergence(
                    policy,
                    f"henon_generalised(alpha={alpha}, beta={beta}) diverged at step {i + 1}",
                )
                break
            X[i + 1] = xn
            Y[i + 1] = yn
    return X, Y


def henon_filtered(
    steps: int,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    alpha: float = 1.4,
    beta: float = 0.3,
    c0: float = 0.7,
    c1: float = 0.3,
    on_divergence: DivergencePolicy = "nan",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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

    Raises
    ------
    InvalidParameterError
        If ``steps`` is not a positive integer or any scalar parameter
        is non-finite.
    DivergenceError
        If the trajectory diverges and ``on_divergence='raise'``.
    """
    steps = require_positive_int("steps", steps, min_value=1)
    x0 = require_finite("x0", x0)
    y0 = require_finite("y0", y0)
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    c0 = require_finite("c0", c0)
    c1 = require_finite("c1", c1)
    policy = require_divergence_policy(on_divergence)
    X = np.empty(steps + 1)
    Y = np.empty(steps + 1)
    X[0], Y[0] = x0, y0
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(steps):
            z = c0 * X[i] + c1 * Y[i]
            xn = alpha - z**2 + beta * Y[i]
            yn = X[i]
            if not (np.isfinite(xn) and np.isfinite(yn)):
                X[i + 1 :] = np.nan
                Y[i + 1 :] = np.nan
                handle_divergence(
                    policy,
                    f"henon_filtered(alpha={alpha}, beta={beta}, c0={c0}, c1={c1}) "
                    f"diverged at step {i + 1}",
                )
                break
            X[i + 1] = xn
            Y[i + 1] = yn
    return X, Y


# ── N-th order map (vectorised state) ──────────────────────────────────────


def _henon_n4_step_inplace(
    out: NDArray[np.float64],
    x: NDArray[np.float64],
    s: float,
    alpha: float,
    beta: float,
    c: NDArray[np.float64],
) -> None:
    """Single step of the N-th order filtered Hénon map, writing into *out*.

    Parameters
    ----------
    out
        Pre-allocated output buffer, shape ``(Nc,)``.
    x
        Current state, shape ``(Nc,)``.
    s
        Current driving value.
    alpha, beta
        Hénon parameters.
    c
        FIR coefficients, shape ``(Nc,)``.
    """
    Nc = len(c)

    x1_new = alpha - s**2 + beta * x[1]
    x2_new = x[0]

    v = c[0] * x1_new + c[1] * x[0] + c[2] * x[1]
    if Nc > 3:
        c_tail = c[3:]
        v += float(np.dot(c_tail, x[3:]))

    out[0] = x1_new
    out[1] = x2_new
    out[2] = v
    if Nc > 3:
        out[3] = x[1]
    if Nc > 4:
        out[4:] = x[3 : Nc - 1]


def henon_order_n(
    steps: int,
    *,
    fir_coeffs: NDArray[np.float64],
    x0: NDArray[np.float64] | None = None,
    alpha: float = 1.4,
    beta: float = 0.3,
    driving: NDArray[np.float64] | None = None,
    seed: int | None = None,
    on_divergence: DivergencePolicy = "nan",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iterate the N-th order Hénon map with an internal FIR filter.

    The system dimension ``N_c`` is inferred from ``len(fir_coeffs)``.
    At each step, the carrier output is the filtered state component
    ``x[2]``, and the next iterate is computed by
    ``_henon_n4_step_inplace``.

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
    alpha, beta
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

    Raises
    ------
    InvalidParameterError
        If ``steps < 1``, ``len(fir_coeffs) < 3``, ``x0`` or ``driving``
        has incompatible shape or non-finite values, or ``alpha``/``beta``
        is non-finite.
    """
    steps = require_positive_int("steps", steps, min_value=1)
    c = require_1d_nonempty("fir_coeffs", fir_coeffs, min_length=3)
    Nc = len(c)
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    seed = require_seed("seed", seed)
    policy = require_divergence_policy(on_divergence)
    if x0 is not None:
        x0_arr = require_1d_nonempty("x0", x0)
        if len(x0_arr) != Nc:
            raise InvalidParameterError(
                f"x0 must have length {Nc} matching fir_coeffs, got {len(x0_arr)}"
            )
    else:
        x0_arr = None
    if driving is not None:
        drv_arr = require_1d_nonempty("driving", driving)
        if len(drv_arr) < steps:
            raise InvalidParameterError(
                f"driving must have at least {steps} samples, got {len(drv_arr)}"
            )
    else:
        drv_arr = None

    rng = np.random.default_rng(seed)

    state = np.zeros((Nc, steps + 1))
    state[:, 0] = x0_arr if x0_arr is not None else 0.5 * rng.random(Nc)
    output = np.empty(steps)

    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(steps):
            s_i = float(drv_arr[i]) if drv_arr is not None else state[2, i]
            output[i] = state[2, i]
            _henon_n4_step_inplace(state[:, i + 1], state[:, i], s_i, alpha, beta, c)
    if policy != "nan" and not (np.all(np.isfinite(state)) and np.all(np.isfinite(output))):
        bad_cols = np.flatnonzero(~np.all(np.isfinite(state), axis=0))
        first_step = int(bad_cols[0]) if bad_cols.size else steps
        handle_divergence(
            policy,
            f"henon_order_n(Nc={Nc}, alpha={alpha}, beta={beta}) diverged at step {first_step}",
        )
    return state, output


# ── FIR-filtered Hénon sequence generator ──────────────────────────────────


def henon_fir_sequence(
    N: int,
    *,
    alpha: float = 1.4,
    beta: float = 0.3,
    num_taps: int = 5,
    cutoff: float = 0.9091,
    window: str = "hamming",
    on_divergence: DivergencePolicy = "nan",
) -> NDArray[np.float64]:
    r"""Generate a chaotic sequence from the FIR-filtered Hénon map.

    Iterates the generalised Hénon form ``x[n+1] = a - xf[n]^2 + b * y[n]``,
    ``y[n+1] = x[n]``, where ``xf`` is the current output of an FIR filter
    applied to the state history.

    Parameters
    ----------
    N
        Number of samples to produce.
    alpha, beta
        Hénon map parameters (default: 1.4, 0.3).
    num_taps
        FIR filter order (number of coefficients).
    cutoff
        Normalised cutoff frequency in (0, 1).
    window
        SciPy window name (e.g. ``"hamming"``, ``"kaiser"``).

    Returns
    -------
    ndarray, shape (N,)
        Chaotic samples.

    Raises
    ------
    InvalidParameterError
        If ``N`` or ``num_taps`` is not a positive integer, ``cutoff`` is
        outside ``(0, 1)``, ``window`` is invalid, or ``alpha``/``beta``
        is non-finite.
    ValueError
        If the trajectory diverges (\\|x\\| > 100) or produces NaN/Inf.

    Notes
    -----
    Deprecated since 0.8.0: no longer used internally.  For filtered
    chaotic sequences, prefer :func:`henon_order_n` with pre-computed
    FIR coefficients.
    """
    N = require_positive_int("N", N, min_value=1)
    alpha = require_finite("alpha", alpha)
    beta = require_finite("beta", beta)
    num_taps = require_positive_int("num_taps", num_taps, min_value=1)
    cutoff = require_in_range("cutoff", cutoff, 0.0, 1.0, closed="neither")
    policy = require_divergence_policy(on_divergence)
    if not isinstance(window, str):
        raise InvalidParameterError(f"window must be a string, got {window!r}")
    warnings.warn(
        "henon_fir_sequence is deprecated and will be removed in a "
        "future version.  Use henon_order_n with pre-computed FIR "
        "coefficients instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        h = firwin(num_taps, cutoff, window=window)
    except Exception as exc:
        raise InvalidParameterError(f"invalid FIR filter design parameters: {exc}") from exc
    if not np.all(np.isfinite(h)):
        raise InvalidParameterError(
            f"firwin produced non-finite coefficients for num_taps={num_taps}, window={window!r}"
        )
    buf = np.full(num_taps, 0.1)
    x_val, y_val = 0.1, 0.1
    seq = np.empty(N)

    write_idx = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(N):
            xf = 0.0
            for k in range(num_taps):
                xf += h[k] * buf[(write_idx - 1 - k) % num_taps]
            xn = alpha - xf * xf + beta * y_val
            yn = x_val
            if not np.isfinite(xn) or abs(xn) > 100:
                seq[n:] = np.nan
                handle_divergence(
                    policy,
                    f"henon_fir_sequence(alpha={alpha}, beta={beta}) diverged at n={n}",
                )
                break
            buf[write_idx] = xn
            write_idx = (write_idx + 1) % num_taps
            x_val, y_val = xn, yn
            seq[n] = xn

    return seq

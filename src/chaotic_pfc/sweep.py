"""
sweep.py
========
Parameter-sweep computation of Lyapunov exponents for the Hénon map with
an internal FIR filter.

This module sweeps the 2-D grid
    (filter order  N_s ∈ {2, …, 41}) × (normalised cutoff  ω_c ∈ (0, 1))
and, for each grid point, estimates the largest Lyapunov exponent
λ_max of the corresponding filtered Hénon system.

All hot paths are JIT-compiled with Numba. The outer 2-D loop is
flattened into a single ``prange`` so that every grid point is a task
available to the thread pool, which keeps the cores busy even when the
cost of different orders is uneven.

The public API is small:

- :class:`SweepResult` — container for the output of one sweep.
- :func:`precompute_fir_bank` — builds the FIR coefficient tensor.
- :func:`run_sweep` — full end-to-end computation for a (window, filter)
  pair.
- :func:`save_sweep` / :func:`load_sweep` — round-trip to ``.npz``.
- :data:`WINDOWS`, :data:`FILTER_TYPES`, :data:`WINDOW_DISPLAY_NAMES`
  — the catalogue of supported configurations.

Notes
-----
The inner kernels below (``_henon_n*``, ``_dhenon_n*``, ``_lyapunov_*``)
are intentionally duplicated per state dimension rather than written as
a single generic routine. Numba's type inference and loop-unrolling
behave much better on small fixed-size routines, and the 4× duplication
is the price we pay to keep the hot loop branch-free.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.signal import firwin

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Catalogue of supported FIR windows and filter types
# ═══════════════════════════════════════════════════════════════════════════

WINDOWS: tuple[str, ...] = (
    "hamming",
    "hann",
    "blackman",
    "kaiser",
    "blackmanharris",
    "boxcar",
    "bartlett",
)

FILTER_TYPES: tuple[str, ...] = ("lowpass", "highpass")

WINDOW_DISPLAY_NAMES: dict[str, str] = {
    "hamming": "Hamming",
    "hann": "Hann",
    "blackman": "Blackman",
    "kaiser": "Kaiser",
    "blackmanharris": "Blackman-Harris",
    "boxcar": "Rectangular",
    "bartlett": "Bartlett",
}

# Kaiser window needs a β parameter (attenuation trade-off).
# β = 5.0 gives ~50 dB stop-band attenuation, a common default.
_KAISER_BETA: float = 5.0


# ═══════════════════════════════════════════════════════════════════════════
# Hénon map variants — compiled with @njit for speed
# ═══════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _henon_n1(x: NDArray, alpha: float, beta: float, c: NDArray) -> NDArray:
    """1-tap filtered Hénon map iterate."""
    return np.array(
        [
            alpha - (c[0] * x[0]) ** 2 + beta * x[1],
            x[0],
        ]
    )


@njit(cache=True)
def _henon_n2(x: NDArray, alpha: float, beta: float, c: NDArray) -> NDArray:
    """2-tap filtered Hénon map iterate."""
    return np.array(
        [
            alpha - (c[0] * x[0] + c[1] * x[1]) ** 2 + beta * x[1],
            x[0],
        ]
    )


@njit(cache=True)
def _henon_n3(x: NDArray, alpha: float, beta: float, c: NDArray) -> NDArray:
    """3-tap filtered Hénon map iterate."""
    x1_new = alpha - x[2] ** 2 + beta * x[1]
    return np.array(
        [
            x1_new,
            x[0],
            c[0] * x1_new + c[1] * x[0] + c[2] * x[1],
        ]
    )


@njit(cache=True)
def _henon_n4(x: NDArray, alpha: float, beta: float, c: NDArray) -> NDArray:
    """General N-tap (N ≥ 4) filtered Hénon map iterate."""
    Ns = len(c)
    x_new = np.empty(Ns)
    x_new[0] = alpha - x[2] ** 2 + beta * x[1]
    x_new[1] = x[0]
    head = c[0] * x_new[0] + c[1] * x[0] + c[2] * x[1]
    tail = 0.0
    for k in range(3, Ns):
        tail += c[k] * x[k]
    x_new[2] = head + tail
    x_new[3] = x[1]
    for k in range(4, Ns):
        x_new[k] = x[k - 1]
    return x_new


# ═══════════════════════════════════════════════════════════════════════════
# Jacobians — compiled with @njit
# ═══════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _dhenon_n1(x: NDArray, beta: float, c: NDArray) -> NDArray:
    return np.array(
        [
            [-2.0 * c[0] ** 2 * x[0], beta],
            [1.0, 0.0],
        ]
    )


@njit(cache=True)
def _dhenon_n2(x: NDArray, beta: float, c: NDArray) -> NDArray:
    return np.array(
        [
            [
                -2.0 * c[0] ** 2 * x[0] - 2.0 * c[0] * c[1] * x[1],
                -2.0 * c[0] * c[1] * x[0] - 2.0 * c[1] ** 2 * x[1] + beta,
            ],
            [1.0, 0.0],
        ]
    )


@njit(cache=True)
def _dhenon_n3(x: NDArray, beta: float, c: NDArray) -> NDArray:
    return np.array(
        [
            [0.0, beta, -2.0 * x[2]],
            [1.0, 0.0, 0.0],
            [c[1], c[0] * beta + c[2], -2.0 * c[0] * x[2]],
        ]
    )


@njit(cache=True)
def _dhenon_n4(x: NDArray, beta: float, d: int, c: NDArray) -> NDArray:
    dx = np.zeros((d, d))
    dx[0, 1] = beta
    dx[0, 2] = -2.0 * x[2]
    dx[1, 0] = 1.0
    dx[2, 0] = c[1]
    dx[2, 1] = c[0] * beta + c[2]
    dx[2, 2] = -2.0 * c[0] * x[2]
    dx[3, 1] = 1.0
    for k in range(3, d):
        dx[2, k] = c[k]
    for k in range(4, d):
        dx[k, k - 1] = 1.0
    return dx


# ═══════════════════════════════════════════════════════════════════════════
# Lyapunov kernels with online accumulation
# ═══════════════════════════════════════════════════════════════════════════
#
# Modified Gram-Schmidt with integrated normalisation, in-place inside the
# task. Storing only the running sum of log-norms avoids allocating an
# (Ns × Nitera) matrix per task.


@njit(cache=True)
def _lyap_online_n1(x, Nitera, Ns, c, alpha, beta):
    w = np.eye(Ns)
    lyap_sum = np.zeros(Ns)
    for _ in range(Nitera):
        z = _dhenon_n1(x, beta, c) @ w
        for k in range(Ns):
            for p in range(k):
                dot = 0.0
                for row in range(Ns):
                    dot += z[row, p] * z[row, k]
                for row in range(Ns):
                    z[row, k] -= dot * z[row, p]
            nrm = 0.0
            for row in range(Ns):
                nrm += z[row, k] * z[row, k]
            nrm = nrm**0.5
            if nrm > 0.0:
                lyap_sum[k] += np.log(nrm)
                inv = 1.0 / nrm
                for row in range(Ns):
                    z[row, k] *= inv
        w = z
        x = _henon_n1(x, alpha, beta, c)
    best = -1e30
    for k in range(Ns):
        val = lyap_sum[k] / Nitera
        if val > best:
            best = val
    return best


@njit(cache=True)
def _lyap_online_n2(x, Nitera, Ns, c, alpha, beta):
    w = np.eye(Ns)
    lyap_sum = np.zeros(Ns)
    for _ in range(Nitera):
        z = _dhenon_n2(x, beta, c) @ w
        for k in range(Ns):
            for p in range(k):
                dot = 0.0
                for row in range(Ns):
                    dot += z[row, p] * z[row, k]
                for row in range(Ns):
                    z[row, k] -= dot * z[row, p]
            nrm = 0.0
            for row in range(Ns):
                nrm += z[row, k] * z[row, k]
            nrm = nrm**0.5
            if nrm > 0.0:
                lyap_sum[k] += np.log(nrm)
                inv = 1.0 / nrm
                for row in range(Ns):
                    z[row, k] *= inv
        w = z
        x = _henon_n2(x, alpha, beta, c)
    best = -1e30
    for k in range(Ns):
        val = lyap_sum[k] / Nitera
        if val > best:
            best = val
    return best


@njit(cache=True)
def _lyap_online_n3(x, Nitera, Ns, c, alpha, beta):
    w = np.eye(Ns)
    lyap_sum = np.zeros(Ns)
    for _ in range(Nitera):
        z = _dhenon_n3(x, beta, c) @ w
        for k in range(Ns):
            for p in range(k):
                dot = 0.0
                for row in range(Ns):
                    dot += z[row, p] * z[row, k]
                for row in range(Ns):
                    z[row, k] -= dot * z[row, p]
            nrm = 0.0
            for row in range(Ns):
                nrm += z[row, k] * z[row, k]
            nrm = nrm**0.5
            if nrm > 0.0:
                lyap_sum[k] += np.log(nrm)
                inv = 1.0 / nrm
                for row in range(Ns):
                    z[row, k] *= inv
        w = z
        x = _henon_n3(x, alpha, beta, c)
    best = -1e30
    for k in range(Ns):
        val = lyap_sum[k] / Nitera
        if val > best:
            best = val
    return best


@njit(cache=True)
def _lyap_online_n4(x, Nitera, Ns, c, alpha, beta):
    w = np.eye(Ns)
    lyap_sum = np.zeros(Ns)
    for _ in range(Nitera):
        z = _dhenon_n4(x, beta, Ns, c) @ w
        for k in range(Ns):
            for p in range(k):
                dot = 0.0
                for row in range(Ns):
                    dot += z[row, p] * z[row, k]
                for row in range(Ns):
                    z[row, k] -= dot * z[row, p]
            nrm = 0.0
            for row in range(Ns):
                nrm += z[row, k] * z[row, k]
            nrm = nrm**0.5
            if nrm > 0.0:
                lyap_sum[k] += np.log(nrm)
                inv = 1.0 / nrm
                for row in range(Ns):
                    z[row, k] *= inv
        w = z
        x = _henon_n4(x, alpha, beta, c)
    best = -1e30
    for k in range(Ns):
        val = lyap_sum[k] / Nitera
        if val > best:
            best = val
    return best


# ═══════════════════════════════════════════════════════════════════════════
# Flattened 2-D sweep kernel — one prange covers every (order, cutoff)
# ═══════════════════════════════════════════════════════════════════════════


@njit(parallel=True, cache=True)
def _sweep_kernel(
    orders: NDArray,
    cutoffs: NDArray,
    fir_bank: NDArray,
    gains: NDArray,
    Nitera: int,
    Nmap: int,
    n_initial: int,
    alpha: float,
    beta: float,
) -> tuple[NDArray, NDArray]:
    """Parallel inner sweep. Returns (h, h_std), both (Ncoef, Ncut)."""
    Ncoef = len(orders)
    Ncut = len(cutoffs)
    Ntot = Ncoef * Ncut

    h = np.full((Ncoef, Ncut), np.nan)
    h_std = np.full((Ncoef, Ncut), np.nan)

    for flat in prange(Ntot):
        i = flat // Ncut
        j = flat % Ncut

        Ns = int(orders[i])
        c = fir_bank[i, j, :Ns]
        gain = gains[i, j]

        # Analytic fixed point (used to seed random ICs around it)
        p1 = 0.0
        p2 = 0.0
        p3 = 0.0
        if Ns >= 2 and gain != 0.0:
            disc = (1.0 - beta) ** 2 + 4.0 * alpha * (gain**2)
            p1 = (-(1.0 - beta) + disc**0.5) / (2.0 * gain**2)
            p2 = p1
            p3 = gain * p1

        h_samples = np.empty(n_initial)
        diverged = False

        for ci in range(n_initial):
            h_samples[ci] = np.nan

            if Ns == 1:
                x = np.random.rand(Ns) * 0.1
                for _ in range(Nitera):
                    x = _henon_n1(x, alpha, beta, c)
                if np.isnan(x[0]) or np.isinf(x[0]):
                    diverged = True
                    break
                h_samples[ci] = _lyap_online_n1(x, Nmap, Ns, c, alpha, beta)

            elif Ns == 2:
                x = 0.1 * np.random.rand(Ns) + np.array([p1, p2])
                for _ in range(Nitera):
                    x = _henon_n2(x, alpha, beta, c)
                if np.isnan(x[0]) or np.isinf(x[0]):
                    diverged = True
                    break
                h_samples[ci] = _lyap_online_n2(x, Nmap, Ns, c, alpha, beta)

            elif Ns == 3:
                x = 0.1 * np.random.rand(Ns) + np.array([p1, p2, p3])
                for _ in range(Nitera):
                    x = _henon_n3(x, alpha, beta, c)
                if np.isnan(x[0]) or np.isinf(x[0]):
                    diverged = True
                    break
                h_samples[ci] = _lyap_online_n3(x, Nmap, Ns, c, alpha, beta)

            else:
                fp = np.empty(Ns)
                fp[0] = p1
                fp[1] = p2
                fp[2] = p3
                for k in range(3, Ns):
                    fp[k] = p1
                x = 0.1 * np.random.rand(Ns) + fp
                for _ in range(Nitera):
                    x = _henon_n4(x, alpha, beta, c)
                if np.isnan(x[0]) or np.isinf(x[0]):
                    diverged = True
                    break
                h_samples[ci] = _lyap_online_n4(x, Nmap, Ns, c, alpha, beta)

        if not diverged:
            total = 0.0
            count = 0
            for v in h_samples:
                if not np.isnan(v):
                    total += v
                    count += 1
            if count > 0:
                mean = total / count
                h[i, j] = mean
                var = 0.0
                for v in h_samples:
                    if not np.isnan(v):
                        var += (v - mean) ** 2
                h_std[i, j] = (var / count) ** 0.5

    return h, h_std


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SweepResult:
    """Result of a single (window, filter) sweep.

    Attributes
    ----------
    h : ndarray, shape (Ncoef, Ncut)
        Mean of λ_max across ``n_initial`` random ICs, per grid point.
        NaN where the trajectory diverged for all ICs.
    h_std : ndarray, shape (Ncoef, Ncut)
        Standard deviation of λ_max samples at each grid point.
    orders : ndarray, shape (Ncoef,)
        Filter orders N_s actually swept.
    cutoffs : ndarray, shape (Ncut,)
        Normalised cutoff frequencies ω_c ∈ (0, 1) swept.
    window : str
        FIR window used (lower-case, e.g. ``"hamming"``).
    filter_type : {"lowpass", "highpass"}
        Filter pass-zero configuration.
    metadata : dict
        Free-form metadata (simulation parameters, timing, etc.).
    """

    h: NDArray
    h_std: NDArray
    orders: NDArray
    cutoffs: NDArray
    window: str
    filter_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Human-readable name used for output directories."""
        pretty = WINDOW_DISPLAY_NAMES.get(self.window, self.window.capitalize())
        return f"{pretty} ({self.filter_type})"


# ───────────────────────────────────────────────────────────────────────────


def precompute_fir_bank(
    orders: Sequence[int] | NDArray,
    cutoffs: NDArray,
    filter_type: str,
    window: str,
) -> tuple[NDArray, NDArray]:
    """Build the FIR coefficient tensor and gain matrix for a sweep.

    Parameters
    ----------
    orders
        Filter orders to sweep. Values will be cast to ``int``.
    cutoffs
        Normalised cutoff frequencies in (0, 1).
    filter_type
        ``"lowpass"`` or ``"highpass"``.
    window
        Window name in :data:`WINDOWS`.

    Returns
    -------
    fir_bank : ndarray, shape (Ncoef, Ncut, max_order + 1)
        ``fir_bank[i, j, :len(c_ij)]`` holds the FIR coefficients for the
        ``(order_i, cutoff_j)`` pair, zero-padded on the right.
    gains : ndarray, shape (Ncoef, Ncut)
        DC gain (sum of coefficients) of each filter.

    Notes
    -----
    For highpass filters we force an odd tap count (``Nss | 1``). This is
    required by :func:`scipy.signal.firwin`: a highpass filter with
    ``pass_zero=False`` must have an odd number of taps, otherwise the
    function raises a ``ValueError``.

    ``scipy.signal.firwin`` returns *exactly* ``numtaps`` coefficients,
    which differs from MATLAB's ``fir1`` by one — callers converting
    MATLAB code should pass ``numtaps = Nss`` (not ``Nss - 1``).
    """
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"filter_type must be one of {FILTER_TYPES}, got {filter_type!r}")
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")

    orders_arr = np.asarray(orders, dtype=np.int64)
    Ncoef = len(orders_arr)
    Ncut = len(cutoffs)
    max_taps = int(orders_arr.max()) + 1  # +1 to accommodate highpass padding

    fir_bank = np.zeros((Ncoef, Ncut, max_taps))
    gains = np.zeros((Ncoef, Ncut))

    win_arg = ("kaiser", _KAISER_BETA) if window == "kaiser" else window

    for i, Nss in enumerate(orders_arr):
        for j, wc in enumerate(cutoffs):
            numtaps = (int(Nss) | 1) if filter_type == "highpass" else int(Nss)
            c = firwin(numtaps, wc, pass_zero=filter_type, window=win_arg)
            fir_bank[i, j, : len(c)] = c.astype(np.float64)
            gains[i, j] = float(np.sum(c))

    return fir_bank, gains


# ───────────────────────────────────────────────────────────────────────────


def run_sweep(
    window: str = "hamming",
    filter_type: str = "lowpass",
    *,
    orders: Sequence[int] | NDArray | None = None,
    cutoffs: NDArray | None = None,
    Nitera: int = 500,
    Nmap: int = 3000,
    n_initial: int = 25,
    alpha: float = 1.4,
    beta: float = 0.3,
    seed: int | None = 42,
    warmup: bool = True,
) -> SweepResult:
    """Run a full Lyapunov sweep for one (window, filter) combination.

    Parameters
    ----------
    window, filter_type
        FIR configuration (see :data:`WINDOWS`, :data:`FILTER_TYPES`).
    orders
        Filter orders to sweep. Defaults to ``range(2, 42)``.
    cutoffs
        Cutoff frequencies. Defaults to 100 points linearly spaced in
        ``(0, 1)``, with the endpoints nudged away from 0 and 1 to avoid
        ``firwin`` edge cases.
    Nitera
        Burn-in iterations applied to the map before starting the
        Lyapunov accumulation.
    Nmap
        Number of iterations used to estimate each Lyapunov exponent.
    n_initial
        Number of random initial conditions averaged per grid point.
    alpha, beta
        Hénon parameters (α, β).
    seed
        Seed for the RNG used to shuffle initial conditions. ``None``
        leaves Numba's RNG state untouched.
    warmup
        Run a tiny 2×3 sweep first to trigger Numba compilation. This is
        recommended before timing a real sweep because the first call
        includes several seconds of JIT overhead.

    Returns
    -------
    SweepResult
        Full result object (see :class:`SweepResult`).
    """
    if orders is None:
        orders = np.arange(2, 42)
    if cutoffs is None:
        cutoffs = np.linspace(1e-5, 1.0 - 1e-5, 100)

    orders_arr = np.asarray(orders, dtype=np.int64)
    cutoffs_arr = np.asarray(cutoffs, dtype=np.float64)

    if seed is not None:
        np.random.seed(seed)

    fir_bank, gains = precompute_fir_bank(
        orders_arr,
        cutoffs_arr,
        filter_type,
        window,
    )

    if warmup:
        # Trigger Numba compilation on a tiny slice. Output discarded.
        _sweep_kernel(
            orders_arr[:2].astype(np.float64),
            cutoffs_arr[:3],
            fir_bank[:2, :3, :],
            gains[:2, :3],
            10,
            50,
            2,
            alpha,
            beta,
        )

    h, h_std = _sweep_kernel(
        orders_arr.astype(np.float64),
        cutoffs_arr,
        fir_bank,
        gains,
        Nitera,
        Nmap,
        n_initial,
        alpha,
        beta,
    )

    return SweepResult(
        h=h,
        h_std=h_std,
        orders=orders_arr,
        cutoffs=cutoffs_arr,
        window=window,
        filter_type=filter_type,
        metadata={
            "Nitera": Nitera,
            "Nmap": Nmap,
            "n_initial": n_initial,
            "alpha": alpha,
            "beta": beta,
            "seed": seed,
        },
    )


# ───────────────────────────────────────────────────────────────────────────


def save_sweep(result: SweepResult, path: str | Path) -> Path:
    """Save a :class:`SweepResult` to a compressed ``.npz``.

    The on-disk schema matches the legacy ``variables_lyapunov.npz``
    format (keys ``h``, ``h_desvio``, ``wcorte``, ``coef``) so older
    plotting scripts keep working, and adds ``window``, ``filter_type``
    and ``metadata`` for round-tripping into :class:`SweepResult`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        h=result.h,
        h_desvio=result.h_std,
        wcorte=result.cutoffs,
        coef=result.orders,
        window=result.window,
        filter_type=result.filter_type,
        metadata=np.array(list(result.metadata.items()), dtype=object),
    )
    return path


def load_sweep(path: str | Path) -> SweepResult:
    """Load a sweep from ``.npz`` produced by :func:`save_sweep`.

    Also accepts the legacy format (only ``h``, ``h_desvio``, ``wcorte``,
    ``coef``), in which case ``window`` and ``filter_type`` are inferred
    from the parent directory name, e.g.
    ``data/sweeps/Hamming (lowpass)/variables_lyapunov.npz``.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    if "window" in data.files and "filter_type" in data.files:
        window = str(data["window"])
        filter_type = str(data["filter_type"])
    else:
        window, filter_type = _infer_config_from_path(path)

    metadata: dict = {}
    if "metadata" in data.files:
        try:
            metadata = dict(data["metadata"].tolist())
        except (TypeError, ValueError):
            metadata = {}

    return SweepResult(
        h=data["h"],
        h_std=data["h_desvio"],
        orders=data["coef"],
        cutoffs=data["wcorte"],
        window=window,
        filter_type=filter_type,
        metadata=metadata,
    )


def _infer_config_from_path(path: Path) -> tuple[str, str]:
    """Infer (window, filter_type) from a directory name like
    ``Hamming (lowpass)``. Falls back to ("unknown", "lowpass") if the
    pattern does not match.
    """
    name = path.parent.name
    for key, pretty in WINDOW_DISPLAY_NAMES.items():
        for ft in FILTER_TYPES:
            if name == f"{pretty} ({ft})":
                return key, ft
    return "unknown", "lowpass"

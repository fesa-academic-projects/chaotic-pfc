"""Sweep orchestration: FIR precomputation, run_sweep, and helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin

from ._kernel import _build_task_order, _sweep_kernel
from ._types import _KAISER_BETA, FILTER_TYPES, WINDOWS, SweepResult


def _precompute_perturbations(
    orders: NDArray,
    n_cutoffs: int,
    n_initial: int,
    seed: int | None,
) -> NDArray:
    """Build the deterministic perturbation tensor consumed by the kernel.

    Returns an array of shape ``(Ncoef, n_cutoffs, n_initial, max_Ns)``
    filled with uniform ``[0, 1)`` samples, where ``max_Ns = orders.max()``.
    """
    Ncoef = len(orders)
    max_Ns = int(orders.max()) if Ncoef > 0 else 0

    if seed is not None:
        np.random.seed(seed)

    return np.random.rand(Ncoef, n_cutoffs, n_initial, max_Ns)


def precompute_fir_bank(
    orders: Sequence[int] | NDArray,
    cutoffs: NDArray,
    filter_type: str,
    window: str,
    *,
    kaiser_beta: float = _KAISER_BETA,
    bandwidth: float = 0.2,
) -> tuple[NDArray, NDArray]:
    """Build the FIR coefficient tensor and gain matrix for a sweep.

    Parameters
    ----------
    orders
        Filter orders to sweep. Values will be cast to ``int``.
    cutoffs
        Normalised cutoff frequencies in (0, 1).
    filter_type
        ``"lowpass"``, ``"highpass"``, ``"bandpass"``, or ``"bandstop"``.
    window
        Window name in :data:`WINDOWS`.
    kaiser_beta
        Shape parameter of the Kaiser window. Ignored unless
        ``window == "kaiser"``. Must be ``>= 0``.
    bandwidth
        Width of the pass/stop band when *filter_type* is
        ``"bandpass"`` or ``"bandstop"``. Ignored for lowpass and
        highpass. Clamped so edges stay inside ``(0, 1)``.

    Returns
    -------
    fir_bank : ndarray, shape (Ncoef, Ncut, max_order)
        ``fir_bank[i, j, :order_i]`` holds the FIR coefficients for the
        ``(order_i, cutoff_j)`` pair, zero-padded on the right.
    gains : ndarray, shape (Ncoef, Ncut)
        DC gain (sum of coefficients) of each filter.
    """
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"filter_type must be one of {FILTER_TYPES}, got {filter_type!r}")
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    if window == "kaiser" and kaiser_beta < 0:
        raise ValueError(f"kaiser_beta must be >= 0, got {kaiser_beta!r}")

    orders_arr = np.asarray(orders, dtype=np.int64)
    if filter_type in ("highpass", "bandstop"):
        even = orders_arr[orders_arr % 2 == 0]
        if even.size:
            raise ValueError(
                f"{filter_type} requires odd orders; got even values "
                f"{even.tolist()}. Use np.arange(start|1, stop, 2) or "
                "filter your orders to odd integers."
            )
    Ncoef = len(orders_arr)
    Ncut = len(cutoffs)
    max_taps = int(orders_arr.max())

    fir_bank = np.zeros((Ncoef, Ncut, max_taps))
    gains = np.zeros((Ncoef, Ncut))

    win_arg = ("kaiser", float(kaiser_beta)) if window == "kaiser" else window

    use_band = filter_type in ("bandpass", "bandstop")
    bw_half = bandwidth / 2.0 if use_band else 0.0

    for i, Nss in enumerate(orders_arr):
        numtaps = int(Nss)
        for j, wc in enumerate(cutoffs):
            if use_band:
                low = max(1e-5, float(wc) - bw_half)
                high = min(1.0 - 1e-5, float(wc) + bw_half)
                c = firwin(numtaps, [low, high], pass_zero=filter_type, window=win_arg)
            else:
                c = firwin(numtaps, wc, pass_zero=filter_type, window=win_arg)
            fir_bank[i, j, :numtaps] = c.astype(np.float64)
            gains[i, j] = float(np.sum(c))

    return fir_bank, gains


def quick_sweep_params() -> tuple[NDArray, NDArray, NDArray, dict[str, float | int]]:
    """Return the (orders_lp, orders_hp, cutoffs, params) for quick-sweep mode.

    These are small grids suitable for testing and CI smoke runs.
    """
    return (
        np.arange(2, 8),
        np.arange(3, 9, 2),
        np.linspace(0.1, 0.9, 10),
        dict(Nitera=50, Nmap=200, n_initial=3, bandwidth=0.15),
    )


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
    kaiser_beta: float = _KAISER_BETA,
    bandwidth: float = 0.2,
    adaptive: bool = False,
    Nmap_min: int = 500,
    tol: float = 1e-3,
) -> SweepResult:
    """Run a full Lyapunov sweep for one (window, filter) combination.

    Parameters
    ----------
    window, filter_type
        FIR configuration (see :data:`WINDOWS`, :data:`FILTER_TYPES`).
    orders
        Filter orders to sweep. Defaults to ``np.arange(2, 42)`` for
        lowpass and ``np.arange(3, 43, 2)`` for highpass (which requires
        odd tap counts).
    cutoffs
        Cutoff frequencies. Defaults to 100 points linearly spaced in
        ``(0, 1)``.
    Nitera
        Burn-in iterations applied to the map before starting the
        Lyapunov accumulation.
    Nmap
        Maximum number of iterations used to estimate each Lyapunov
        exponent. Also the exact iteration count when ``adaptive=False``.
    bandwidth
        Width of the pass/stop band for bandpass/bandstop filters.
    n_initial
        Number of random initial conditions averaged per grid point.
    alpha, beta
        Henon parameters (alpha, beta).
    seed
        Seed for the RNG used to shuffle initial conditions. ``None``
        leaves the RNG state untouched.
    warmup
        Run a tiny sweep first to trigger Numba JIT compilation.
    kaiser_beta
        Shape parameter of the Kaiser window.
    adaptive
        When ``True``, enable adaptive early-stop in the Lyapunov kernels.
    Nmap_min
        Minimum iterations before the adaptive criterion may fire.
    tol
        Convergence tolerance for the adaptive criterion.

    Returns
    -------
    SweepResult
        Full result object.
    """
    if adaptive:
        if Nmap_min < 1:
            raise ValueError(f"Nmap_min must be >= 1, got {Nmap_min}")
        if Nmap_min > Nmap:
            raise ValueError(f"Nmap_min ({Nmap_min}) must be <= Nmap ({Nmap})")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if Nmap_min == Nmap:
            raise ValueError(
                "adaptive=True with Nmap_min == Nmap is a no-op; "
                "set Nmap_min < Nmap or pass adaptive=False."
            )
        kernel_Nmap_min = Nmap_min
        kernel_tol = float(tol)
    else:
        kernel_Nmap_min = Nmap
        kernel_tol = 0.0

    if orders is None:
        orders = (
            np.arange(3, 43, 2) if filter_type in ("highpass", "bandstop") else np.arange(2, 42)
        )
    if cutoffs is None:
        cutoffs = np.linspace(1e-5, 1.0 - 1e-5, 100)

    orders_arr = np.asarray(orders, dtype=np.int64)
    cutoffs_arr = np.asarray(cutoffs, dtype=np.float64)

    fir_bank, gains = precompute_fir_bank(
        orders_arr,
        cutoffs_arr,
        filter_type,
        window,
        kaiser_beta=kaiser_beta,
        bandwidth=bandwidth,
    )

    perturbations = _precompute_perturbations(
        orders_arr,
        len(cutoffs_arr),
        n_initial,
        seed,
    )

    task_order = _build_task_order(orders_arr, len(cutoffs_arr))

    if warmup:
        _warmup_perturbations = _precompute_perturbations(
            orders_arr[:2],
            3,
            2,
            seed=None,
        )
        _warmup_order = _build_task_order(orders_arr[:2], 3)
        _sweep_kernel(
            orders_arr[:2].astype(np.float64),
            cutoffs_arr[:3],
            fir_bank[:2, :3, :],
            gains[:2, :3],
            _warmup_perturbations,
            _warmup_order,
            10,
            50,
            50,
            0.0,
            2,
            alpha,
            beta,
        )

    h, h_std, n_iters_used = _sweep_kernel(
        orders_arr.astype(np.float64),
        cutoffs_arr,
        fir_bank,
        gains,
        perturbations,
        task_order,
        Nitera,
        Nmap,
        kernel_Nmap_min,
        kernel_tol,
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
        n_iters_used=n_iters_used,
        metadata={
            "Nitera": Nitera,
            "Nmap": Nmap,
            "n_initial": n_initial,
            "alpha": alpha,
            "beta": beta,
            "seed": seed,
            "kaiser_beta": kaiser_beta,
            "adaptive": adaptive,
            "Nmap_min": Nmap_min if adaptive else None,
            "tol": tol if adaptive else None,
        },
    )

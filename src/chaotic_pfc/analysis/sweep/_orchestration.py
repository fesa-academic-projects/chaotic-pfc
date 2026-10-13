"""Sweep orchestration: FIR precomputation, run_sweep, and helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin

from chaotic_pfc.config import DEFAULT_CONFIG

from ._kernel import _build_task_order, _sweep_kernel
from ._types import _KAISER_BETA, FILTER_TYPES, WINDOWS, SweepResult


def _precompute_perturbations(
    num_taps_grid: NDArray[Any],
    n_cutoffs: int,
    n_initial: int,
    seed: int | None,
) -> NDArray[Any]:
    """Build the deterministic perturbation tensor consumed by the kernel.

    Returns an array of shape ``(Ncoef, n_cutoffs, n_initial, max_Ns)``
    filled with uniform ``[0, 1)`` samples, where ``max_Ns = num_taps_grid.max()``.
    """
    Ncoef = len(num_taps_grid)
    max_Ns = int(num_taps_grid.max()) if Ncoef > 0 else 0

    rng = np.random.default_rng(seed)
    return rng.random((Ncoef, n_cutoffs, n_initial, max_Ns), dtype=np.float64)


def precompute_fir_bank(
    num_taps_grid: Sequence[int] | NDArray[Any],
    cutoff_grid: NDArray[Any],
    filter_type: str,
    window: str,
    *,
    kaiser_beta: float = _KAISER_BETA,
    bandwidth: float = 0.05,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Build the FIR coefficient tensor and gain matrix for a sweep.

    Parameters
    ----------
    num_taps_grid
        Number of taps to sweep (``N_s``); the filter order is
        ``N_z = N_s - 1``. Values will be cast to ``int``.
    cutoff_grid
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

    orders_arr = np.asarray(num_taps_grid, dtype=np.int64)
    if filter_type in ("highpass", "bandstop"):
        even = orders_arr[orders_arr % 2 == 0]
        if even.size:
            raise ValueError(
                f"{filter_type} requires an odd number of taps; got even values "
                f"{even.tolist()}. Use np.arange(start|1, stop, 2) or "
                "filter your num_taps_grid to odd integers."
            )
    Ncoef = len(orders_arr)
    Ncut = len(cutoff_grid)
    max_taps = int(orders_arr.max())

    fir_bank = np.zeros((Ncoef, Ncut, max_taps))
    gains = np.zeros((Ncoef, Ncut))

    win_arg = ("kaiser", float(kaiser_beta)) if window == "kaiser" else window

    use_band = filter_type in ("bandpass", "bandstop")
    bw_half = bandwidth / 2.0 if use_band else 0.0

    for i, Nss in enumerate(orders_arr):
        numtaps = int(Nss)
        for j, wc in enumerate(cutoff_grid):
            if use_band:
                low = max(1e-5, float(wc) - bw_half)
                high = min(1.0 - 1e-5, float(wc) + bw_half)
                c = firwin(numtaps, [low, high], pass_zero=filter_type, window=win_arg)
            else:
                c = firwin(numtaps, wc, pass_zero=filter_type, window=win_arg)
            fir_bank[i, j, :numtaps] = c.astype(np.float64)
            gains[i, j] = float(np.sum(c))

    return fir_bank, gains


class QuickSweepConfig(TypedDict):
    """Parameters passed to :func:`run_sweep` in quick mode."""

    Nitera: int
    Nmap: int
    n_initial: int
    bandwidth: float


class QuickSweepParams(NamedTuple):
    """Return value of :func:`quick_sweep_params`.

    Fields replace the previous positional four-tuple.
    """

    orders_lp: NDArray[Any]
    orders_hp: NDArray[Any]
    cutoff_grid: NDArray[Any]
    params: QuickSweepConfig


def quick_sweep_params() -> QuickSweepParams:
    """Return the quick-sweep grids and parameters.

    Returns
    -------
    QuickSweepParams
        ``orders_lp``, ``orders_hp``, ``cutoff_grid`` and ``params``.
    """
    return QuickSweepParams(
        np.arange(2, 8),
        np.arange(3, 9, 2),
        np.linspace(0.1, 0.9, 10),
        QuickSweepConfig(Nitera=50, Nmap=200, n_initial=3, bandwidth=0.0375),
    )


def _resolve_fir_bank(
    num_taps_grid: NDArray[Any],
    cutoff_grid: NDArray[Any],
    filter_type: str,
    window: str,
    *,
    kaiser_beta: float,
    bandwidth: float,
    fir_bank: NDArray[Any] | None,
    fir_design: Callable[[int, float], NDArray[Any]] | None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the FIR coefficient tensor, from the catalogue or a user design.

    Implements the ``run_sweep`` extension point: the sweep can use
    the built-in window catalogue, a pre-computed ``fir_bank`` or a
    ``fir_design`` callable. FIR only.
    """
    if fir_bank is not None and fir_design is not None:
        raise ValueError("pass either fir_bank or fir_design, not both")

    orders_arr = np.asarray(num_taps_grid, dtype=np.int64)
    cutoffs_arr = np.asarray(cutoff_grid, dtype=np.float64)
    n_taps_max = int(orders_arr.max())

    if fir_bank is not None:
        bank = np.asarray(fir_bank, dtype=np.float64)
        expected = (len(orders_arr), len(cutoffs_arr), n_taps_max)
        if bank.shape != expected:
            raise ValueError(
                f"fir_bank must have shape {expected} "
                f"(num_taps_grid x cutoff_grid x max num_taps), got {bank.shape}"
            )
        return bank, bank.sum(axis=2)

    if fir_design is not None:
        bank = np.zeros((len(orders_arr), len(cutoffs_arr), n_taps_max))
        for i, num_taps in enumerate(orders_arr):
            for j, cutoff in enumerate(cutoffs_arr):
                coeffs = np.asarray(fir_design(int(num_taps), float(cutoff)), dtype=np.float64)
                if coeffs.shape != (int(num_taps),):
                    raise ValueError(
                        f"fir_design({int(num_taps)}, {float(cutoff)}) returned shape "
                        f"{coeffs.shape}, expected ({int(num_taps)},)"
                    )
                bank[i, j, : int(num_taps)] = coeffs
        return bank, bank.sum(axis=2)

    return precompute_fir_bank(
        orders_arr,
        cutoffs_arr,
        filter_type,
        window,
        kaiser_beta=kaiser_beta,
        bandwidth=bandwidth,
    )


def _resolve_adaptive_params(
    adaptive: bool, Nmap_min: int, Nmap: int, tol: float
) -> tuple[int, float]:
    """Validate the adaptive options and return ``(kernel_Nmap_min, kernel_tol)``."""
    if not adaptive:
        return Nmap, 0.0
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
    return Nmap_min, float(tol)


def _default_sweep_grid(
    num_taps_grid: Sequence[int] | NDArray[Any] | None,
    cutoff_grid: NDArray[Any] | None,
    filter_type: str,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Fill the default tap/cutoff grids and return them as int64/float64 arrays."""
    if num_taps_grid is None:
        order_hi = DEFAULT_CONFIG.sweep.n_z_max + 2
        num_taps_grid = (
            np.arange(3, order_hi + 1, 2)
            if filter_type in ("highpass", "bandstop")
            else np.arange(2, order_hi)
        )
    if cutoff_grid is None:
        cutoff_grid = np.linspace(1e-5, 1.0 - 1e-5, 100)
    return np.asarray(num_taps_grid, dtype=np.int64), np.asarray(cutoff_grid, dtype=np.float64)


def _warmup_kernel(
    orders_arr: NDArray[Any],
    cutoffs_arr: NDArray[Any],
    fir_bank: NDArray[Any],
    gains: NDArray[Any],
    alpha: float,
    beta: float,
    fallback_mode: int,
) -> None:
    """Run a tiny sweep to trigger the Numba JIT compilation."""
    warmup_orders = orders_arr[:2]
    warmup_perturbations = _precompute_perturbations(warmup_orders, 3, 2, seed=None)
    warmup_order = _build_task_order(warmup_orders, 3)
    _sweep_kernel(
        warmup_orders.astype(np.float64),
        cutoffs_arr[:3],
        fir_bank[:2, :3, :],
        gains[:2, :3],
        warmup_perturbations,
        warmup_order,
        10,
        50,
        50,
        0.0,
        2,
        alpha,
        beta,
        fallback_mode,
    )


def _sweep_metadata(
    *,
    Nitera: int,
    Nmap: int,
    n_initial: int,
    alpha: float,
    beta: float,
    seed: int | None,
    kaiser_beta: float,
    adaptive: bool,
    Nmap_min: int,
    tol: float,
    fallback_mode: int,
    n_fallback: NDArray[Any],
) -> dict[str, Any]:
    """Build the ``SweepResult.metadata`` mapping."""
    return {
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
        "fallback_mode": fallback_mode,
        "n_fallback": int(np.sum(n_fallback)),
    }


def run_sweep(
    window: str = "hamming",
    filter_type: str = "lowpass",
    *,
    num_taps_grid: Sequence[int] | NDArray[Any] | None = None,
    cutoff_grid: NDArray[Any] | None = None,
    Nitera: int = 500,
    Nmap: int = 3000,
    n_initial: int = 25,
    alpha: float = 1.4,
    beta: float = 0.3,
    seed: int | None = 42,
    warmup: bool = True,
    kaiser_beta: float = _KAISER_BETA,
    bandwidth: float = 0.05,
    adaptive: bool = False,
    Nmap_min: int = 500,
    tol: float = 1e-3,
    fallback_mode: int = 1,
    fir_bank: NDArray[Any] | None = None,
    fir_design: Callable[[int, float], NDArray[Any]] | None = None,
) -> SweepResult:
    """Run a full Lyapunov sweep for one (window, filter) combination.

    Parameters
    ----------
    window, filter_type
        FIR configuration (see :data:`WINDOWS`, :data:`FILTER_TYPES`).
    num_taps_grid
        Number of FIR filter coefficients (taps) :math:`N_s` to sweep; the
        filter order is :math:`N_z = N_s - 1`. Defaults to ``np.arange(2, 42)``
        for lowpass and bandpass, and to ``np.arange(3, 43, 2)`` for highpass
        and bandstop (which require an odd number of taps).
    cutoff_grid
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
    fallback_mode
        Policy for recomputing marginal cells (``|λ| < _VONLY_MARGIN``):
        ``0`` disables recomputation, ``1`` (default) recomputes with the
        full spectrum and returns its maximum, ``2`` recomputes with the
        full spectrum but returns and adapts on column 0 only.
    fir_bank
        Optional pre-computed FIR coefficient bank with shape
        ``(len(num_taps_grid), len(cutoff_grid), max(num_taps_grid))``, zero-padded on the tap
        axis. Use it to sweep a filter that is not produced by the built-in
        window catalogue. Mutually exclusive with *fir_design*.
    fir_design
        Optional callable ``fir_design(num_taps, cutoff) -> coefficients`` used
        to build the bank one cell at a time (also allowed instead of the
        catalogue). Only FIR designs are supported. Mutually exclusive with
        *fir_bank*.

    Returns
    -------
    SweepResult
        Full result object.

    See Also
    --------
    :doc:`/source/algorithms/sweep` : Parallel architecture and divergence handling.
    """
    if fallback_mode not in (0, 1, 2):
        raise ValueError(f"fallback_mode must be 0, 1 or 2, got {fallback_mode!r}")

    kernel_Nmap_min, kernel_tol = _resolve_adaptive_params(adaptive, Nmap_min, Nmap, tol)
    orders_arr, cutoffs_arr = _default_sweep_grid(num_taps_grid, cutoff_grid, filter_type)

    fir_bank, gains = _resolve_fir_bank(
        orders_arr,
        cutoffs_arr,
        filter_type,
        window,
        kaiser_beta=kaiser_beta,
        bandwidth=bandwidth,
        fir_bank=fir_bank,
        fir_design=fir_design,
    )

    perturbations = _precompute_perturbations(
        orders_arr,
        len(cutoffs_arr),
        n_initial,
        seed,
    )

    task_order = _build_task_order(orders_arr, len(cutoffs_arr))

    if warmup:
        _warmup_kernel(orders_arr, cutoffs_arr, fir_bank, gains, alpha, beta, fallback_mode)

    h, h_std, n_iters_used, n_fallback = _sweep_kernel(
        orders_arr,
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
        fallback_mode,
    )

    return SweepResult(
        h=h,
        h_std=h_std,
        num_taps_grid=orders_arr,
        cutoff_grid=cutoffs_arr,
        window=window,
        filter_type=filter_type,
        n_iters_used=n_iters_used,
        metadata=_sweep_metadata(
            Nitera=Nitera,
            Nmap=Nmap,
            n_initial=n_initial,
            alpha=alpha,
            beta=beta,
            seed=seed,
            kaiser_beta=kaiser_beta,
            adaptive=adaptive,
            Nmap_min=Nmap_min,
            tol=tol,
            fallback_mode=fallback_mode,
            n_fallback=n_fallback,
        ),
    )

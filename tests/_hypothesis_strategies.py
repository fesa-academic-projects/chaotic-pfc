"""Reusable Hypothesis strategies for the chaotic-pfc domain."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from numpy.typing import NDArray


def safe_henon_params(
    a_range: tuple[float, float] = (0.1, 2.0),
    b_range: tuple[float, float] = (0.01, 0.9),
):
    """Hénon (a, b) in a regime that typically stays bounded."""
    return st.tuples(
        st.floats(*a_range, allow_nan=False, allow_infinity=False),
        st.floats(*b_range, allow_nan=False, allow_infinity=False),
    )


def finite_initial_conditions(dim: int = 2):
    """Initial condition vectors that are small enough to avoid overflow."""
    return st.lists(
        st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim,
    ).map(lambda xs: np.array(xs, dtype=float))


def lowpass_fir_params(
    min_taps: int = 3,
    max_taps: int = 30,
):
    """Strategy for (N_filter, wc) suitable for lowpass FIR via firwin."""
    n_taps = st.integers(min_taps, max_taps)
    wc = st.floats(0.1, 0.9, allow_nan=False, allow_infinity=False)
    return st.tuples(n_taps, wc)


def finite_ndarrays(
    shape: tuple[int, ...] | None = None,
    min_val: float = -10.0,
    max_val: float = 10.0,
):
    """Generate finite ndarrays with optional shape constraint."""
    if shape is not None:
        elements = st.lists(
            st.floats(min_val, max_val, allow_nan=False, allow_infinity=False),
            min_size=int(np.prod(shape)),
            max_size=int(np.prod(shape)),
        )
        return elements.map(lambda xs: np.array(xs, dtype=float).reshape(shape))
    return st.lists(
        st.floats(min_val, max_val, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    ).map(lambda xs: np.array(xs, dtype=float))


@st.composite
def arrays_with_nan(
    draw,
    min_size: int = 2,
    max_size: int = 100,
    nan_ratio: float = 0.5,
):
    """Arrays that may contain NaN mixed with finite values."""
    size = draw(st.integers(min_size, max_size))
    vals = draw(
        st.lists(
            st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    return _build_mixed_array(size, vals, nan_ratio)


def _build_mixed_array(size: int, vals: list[float], nan_ratio: float) -> NDArray:
    rng = np.random.default_rng()
    arr = np.array(vals, dtype=float)
    mask = rng.random(size) < nan_ratio
    arr[mask] = np.nan
    return arr


def small_sweep_results():
    """Synthetic SweepResult-like dicts with classification data."""
    shape = st.tuples(st.integers(2, 8), st.integers(2, 8))
    return st.builds(
        _build_sweep_h,
        shape,
        st.lists(
            st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False), min_size=4, max_size=64
        ),
    )


def _build_sweep_h(
    shape: tuple[int, int],
    vals: list[float],
) -> dict:
    rng = np.random.default_rng()
    total = shape[0] * shape[1]
    if len(vals) < total:
        vals = list(vals) + [0.0] * (total - len(vals))
    arr = np.array(vals[:total], dtype=float).reshape(shape)
    # Sprinkle some NaN
    if total > 4:
        nan_idx = rng.choice(total, size=max(1, total // 5), replace=False)
        arr.flat[nan_idx] = np.nan
    return {"h": arr, "shape": shape}

"""Reusable Hypothesis strategies for the chaotic-pfc domain.

Every strategy is a pure function of the values drawn from Hypothesis: there
are **no unseeded RNG calls** here, so under ``derandomize`` (the ``ci``
profile) the generated data is reproducible byte for byte, and locally a
failure can be reproduced from the seed Hypothesis prints.
"""

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
    """Arrays that may contain NaN mixed with finite values.

    The NaN mask is drawn from Hypothesis, so the array is a pure function of
    the drawn data (reproducible under ``derandomize``).
    """
    size = draw(st.integers(min_size, max_size))
    vals = draw(
        st.lists(
            st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    probs = draw(
        st.lists(
            st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    return _build_mixed_array(size, vals, probs, nan_ratio)


def _build_mixed_array(
    size: int,
    vals: list[float],
    probs: list[float],
    nan_ratio: float,
) -> NDArray:
    """Array whose ``i``-th entry is NaN iff ``probs[i] < nan_ratio``."""
    arr = np.array(vals[:size], dtype=float)
    arr[np.asarray(probs[:size], dtype=float) < nan_ratio] = np.nan
    return arr


@st.composite
def small_sweep_results(draw):
    """Synthetic SweepResult-like dicts with classification data.

    Both the values and the NaN mask are drawn from Hypothesis (no unseeded
    RNG), so the strategy is reproducible under ``derandomize``.
    """
    shape = draw(st.tuples(st.integers(2, 8), st.integers(2, 8)))
    total = shape[0] * shape[1]
    vals = draw(
        st.lists(
            st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=total,
            max_size=total,
        )
    )
    is_nan = draw(st.lists(st.booleans(), min_size=total, max_size=total))
    if total > 4 and not any(is_nan):
        is_nan[0] = True  # keep at least one divergent cell, as before
    return _build_sweep_h(shape, vals, is_nan)


def _build_sweep_h(
    shape: tuple[int, int],
    vals: list[float],
    is_nan: list[bool],
) -> dict:
    total = shape[0] * shape[1]
    arr = np.array(vals[:total], dtype=float).reshape(shape)
    mask = np.asarray(is_nan[:total], dtype=bool).reshape(shape)
    arr[mask] = np.nan
    return {"h": arr, "shape": shape}

"""Boundary parameter validation helpers for public APIs."""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from chaotic_pfc.errors import DivergenceError, DivergenceWarning, InvalidParameterError

DivergencePolicy = Literal["nan", "warn", "raise"]


def require_divergence_policy(policy: Any) -> DivergencePolicy:
    """Validate that *policy* is one of ``'nan'``, ``'warn'``, or ``'raise'``."""
    if policy == "nan":
        return "nan"
    if policy == "warn":
        return "warn"
    if policy == "raise":
        return "raise"
    raise InvalidParameterError(f"on_divergence must be 'nan', 'warn', or 'raise', got {policy!r}")


def handle_divergence(policy: DivergencePolicy, message: str) -> None:
    """Apply the configured divergence policy (``'nan'``, ``'warn'``, or ``'raise'``)."""
    if policy == "nan":
        return
    if policy == "warn":
        warnings.warn(message, DivergenceWarning, stacklevel=3)
        return
    raise DivergenceError(message)


def require_finite(name: str, value: Any, *, max_abs: float = 1e150) -> float:
    """Validate that *value* is a finite real scalar within ``[-max_abs, max_abs]``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or isinstance(value, (complex, np.complexfloating))
    ):
        raise InvalidParameterError(f"{name} must be a real number, got {value!r}")
    try:
        fval = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise InvalidParameterError(f"{name} must be a finite real number, got {value!r}") from exc
    if not math.isfinite(fval):
        raise InvalidParameterError(f"{name} must be finite, got {value!r}")
    if abs(fval) > max_abs:
        raise InvalidParameterError(
            f"{name} magnitude exceeds safe numerical bound ({max_abs:g}), got {value!r}"
        )
    return fval


def require_positive_int(name: str, value: Any, *, min_value: int = 1) -> int:
    """Validate that *value* is an integer ``>= min_value`` (rejecting bool, NaN, and non-integer float)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise InvalidParameterError(f"{name} must be an integer >= {min_value}, got {value!r}")
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if not math.isfinite(fval) or not fval.is_integer():
            raise InvalidParameterError(
                f"{name} must be an integer >= {min_value}, got non-integer {value!r}"
            )
        ival = int(fval)
    else:
        ival = int(value)
    if ival < min_value:
        raise InvalidParameterError(f"{name} must be >= {min_value}, got {value!r}")
    return ival


def require_in_range(
    name: str,
    value: Any,
    lo: float,
    hi: float,
    *,
    closed: Literal["both", "left", "right", "neither"] = "both",
) -> float:
    """Validate that *value* is a finite real scalar inside the interval ``(lo, hi)``."""
    fval = require_finite(name, value, max_abs=max(abs(lo), abs(hi), 1e150))
    if closed == "both":
        ok = lo <= fval <= hi
        bracket = f"[{lo}, {hi}]"
    elif closed == "left":
        ok = lo <= fval < hi
        bracket = f"[{lo}, {hi})"
    elif closed == "right":
        ok = lo < fval <= hi
        bracket = f"({lo}, {hi}]"
    elif closed == "neither":
        ok = lo < fval < hi
        bracket = f"({lo}, {hi})"
    else:
        raise InvalidParameterError(f"invalid closed mode {closed!r}")
    if not ok:
        raise InvalidParameterError(f"{name} must be in {bracket}, got {value!r}")
    return fval


def require_probability(name: str, value: Any) -> float:
    """Validate that *value* is a probability in ``[0, 1]``."""
    return require_in_range(name, value, 0.0, 1.0, closed="both")


def require_1d_nonempty(
    name: str,
    arr: Any,
    *,
    allow_nan: bool = False,
    min_length: int = 1,
) -> NDArray[np.float64]:
    """Validate that *arr* is a non-empty 1-D real numeric array."""
    if isinstance(arr, (str, bytes, dict, set)):
        raise InvalidParameterError(f"{name} must be a 1-D numeric array, got {type(arr).__name__}")
    try:
        raw = np.asarray(arr)
    except Exception as exc:
        raise InvalidParameterError(f"{name} could not be converted to an array: {exc}") from exc
    if raw.dtype == bool or np.issubdtype(raw.dtype, np.complexfloating) or raw.dtype == object:
        raise InvalidParameterError(f"{name} must be a real numeric array, got dtype {raw.dtype}")
    try:
        out = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(f"{name} must contain real numbers: {exc}") from exc
    if out.ndim != 1 or out.size == 0:
        if out.ndim == 1 and min_length > 1:
            raise InvalidParameterError(
                f"{name} must have at least {min_length} entries, got {out.size}"
            )
        raise InvalidParameterError(f"{name} must be a non-empty 1-D array, got shape {out.shape}")
    if out.size < min_length:
        raise InvalidParameterError(
            f"{name} must have at least {min_length} entries, got {out.size}"
        )
    if not allow_nan and not np.all(np.isfinite(out)):
        raise InvalidParameterError(f"{name} must contain only finite values")
    return out


def require_int_grid(name: str, grid: Any, *, min_value: int = 1) -> NDArray[np.int64]:
    """Validate that *grid* is a non-empty 1-D sequence of integers ``>= min_value``."""
    if isinstance(grid, (str, bytes, dict, set)):
        raise InvalidParameterError(
            f"{name} must be a 1-D integer sequence, got {type(grid).__name__}"
        )
    try:
        raw = np.asarray(grid)
    except Exception as exc:
        raise InvalidParameterError(f"{name} could not be converted to an array: {exc}") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise InvalidParameterError(
            f"{name} must be a non-empty 1-D sequence, got shape {raw.shape}"
        )
    validated = [
        require_positive_int(f"{name}[{i}]", v, min_value=min_value)
        for i, v in enumerate(raw.tolist())
    ]
    return np.asarray(validated, dtype=np.int64)


def require_seed(name: str, seed: Any) -> int | None:
    """Validate an optional RNG seed (``None`` or non-negative integer)."""
    if seed is None:
        return None
    return require_positive_int(name, seed, min_value=0)

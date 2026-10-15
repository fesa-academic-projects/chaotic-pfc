"""Sweep I/O: save, load, and path-inference helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ._types import FILTER_TYPES, WINDOW_DISPLAY_NAMES, SweepResult


def save_sweep(result: SweepResult, path: str | Path) -> Path:
    """Save a :class:`~chaotic_pfc.analysis.sweep.SweepResult` to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "h": result.h,
        "h_desvio": result.h_std,
        "cutoff_grid": result.cutoff_grid,
        "num_taps_grid": result.num_taps_grid,
        "window": result.window,
        "filter_type": result.filter_type,
        "metadata": np.array(list(result.metadata.items()), dtype=object),
    }
    if result.n_iters_used is not None:
        payload["n_iters_used"] = result.n_iters_used
    np.savez(path, **payload)  # type: ignore[arg-type]  # kwargs are arrays/scalars by construction
    return path


def load_sweep(path: str | Path) -> SweepResult:
    """Load a sweep from ``.npz`` produced by :func:`save_sweep`.

    Examples
    --------
    A save/load round-trip preserves the ``lambda_max`` grid exactly:

    >>> import os, tempfile
    >>> res = run_sweep(
    ...     "hamming", "lowpass",
    ...     num_taps_grid=np.array([3]),
    ...     cutoff_grid=np.array([0.25]),
    ...     Nmap=300, n_initial=2, warmup=False,
    ... )
    >>> with tempfile.TemporaryDirectory() as d:
    ...     again = load_sweep(save_sweep(res, os.path.join(d, "demo.npz")))
    ...     np.array_equal(again.h, res.h)
    True
    """
    path = Path(path)
    try:
        data = np.load(path, allow_pickle=True)
        files = data.files
    except Exception as exc:  # unreadable or not an .npz archive
        raise ValueError(
            f"invalid or corrupted checkpoint: {path} "
            f"(regenerate it with `chaotic-pfc sweeps regenerate`): {exc}"
        ) from exc

    if "window" in files and "filter_type" in files:
        window = str(data["window"])
        filter_type = str(data["filter_type"])
    else:
        try:
            window, filter_type = _infer_config_from_path(path)
        except ValueError:
            raise ValueError(
                f"cannot determine window/filter_type for {path}; "
                f"the .npz file has no 'window'/'filter_type' entries "
                f"and the parent directory name does not match any known "
                f"<(filter_type)> pattern"
            ) from None

    metadata: dict[str, Any] = {}
    if "metadata" in files:
        try:
            metadata = dict(data["metadata"].tolist())
        except (TypeError, ValueError):
            warnings.warn(
                f"corrupted metadata in {path} — using empty dict",
                RuntimeWarning,
                stacklevel=2,
            )

    n_iters_used = data["n_iters_used"] if "n_iters_used" in files else None

    # New key names, with a fallback to the pre-0.9 names ("coef"/"wcorte") so
    # that checkpoint files written by older versions (including the frozen
    # SIMAC state f517e0a) still load.
    num_taps_grid = data["num_taps_grid"] if "num_taps_grid" in files else data["coef"]
    cutoff_grid = data["cutoff_grid"] if "cutoff_grid" in files else data["wcorte"]

    return SweepResult(
        h=data["h"],
        h_std=data["h_desvio"],
        num_taps_grid=num_taps_grid,
        cutoff_grid=cutoff_grid,
        window=window,
        filter_type=filter_type,
        n_iters_used=n_iters_used,
        metadata=metadata,
    )


def _infer_config_from_path(path: Path) -> tuple[str, str]:
    """Infer (window, filter_type) from a directory name like ``Hamming (lowpass)``.

    Raises
    ------
    ValueError
        If the parent directory name does not match any known
        ``<display_name> (<filter_type>)`` pattern.
    """
    name = path.parent.name
    for key, pretty in WINDOW_DISPLAY_NAMES.items():
        for ft in FILTER_TYPES:
            if name == f"{pretty} ({ft})":
                return key, ft
    raise ValueError(
        f"cannot infer (window, filter_type) from directory name {name!r}; "
        f"expected pattern '<display_name> (<filter_type>)'"
    )

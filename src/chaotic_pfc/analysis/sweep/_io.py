"""Sweep I/O: save, load, and path-inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._types import FILTER_TYPES, WINDOW_DISPLAY_NAMES, SweepResult


def save_sweep(result: SweepResult, path: str | Path) -> Path:
    """Save a :class:`SweepResult` to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "h": result.h,
        "h_desvio": result.h_std,
        "wcorte": result.cutoffs,
        "coef": result.orders,
        "window": result.window,
        "filter_type": result.filter_type,
        "metadata": np.array(list(result.metadata.items()), dtype=object),
    }
    if result.n_iters_used is not None:
        payload["n_iters_used"] = result.n_iters_used
    np.savez(path, **payload)  # type: ignore[arg-type]
    return path


def load_sweep(path: str | Path) -> SweepResult:
    """Load a sweep from ``.npz`` produced by :func:`save_sweep`."""
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

    n_iters_used = data["n_iters_used"] if "n_iters_used" in data.files else None

    return SweepResult(
        h=data["h"],
        h_std=data["h_desvio"],
        orders=data["coef"],
        cutoffs=data["wcorte"],
        window=window,
        filter_type=filter_type,
        n_iters_used=n_iters_used,
        metadata=metadata,
    )


def _infer_config_from_path(path: Path) -> tuple[str, str]:
    """Infer (window, filter_type) from a directory name like ``Hamming (lowpass)``."""
    name = path.parent.name
    for key, pretty in WINDOW_DISPLAY_NAMES.items():
        for ft in FILTER_TYPES:
            if name == f"{pretty} ({ft})":
                return key, ft
    return "unknown", "lowpass"

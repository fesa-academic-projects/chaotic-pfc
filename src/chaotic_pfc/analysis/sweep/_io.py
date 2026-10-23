"""Sweep I/O: save, load, and path-inference helpers.

Security design ("verified integrity before trust"):

* New checkpoints written by :func:`save_sweep` serialise ``metadata`` as a JSON
  string array (``dtype='<U'``) and are loaded with ``allow_pickle=False``.
* Legacy checkpoints that stored ``metadata`` as a pickled ``dtype=object``
  array are deserialised only when their SHA-256 digest is listed in
  :data:`chaotic_pfc.analysis.sweep._trusted_checkpoints.TRUSTED_CHECKPOINT_SHA256`,
  a set compiled into the package. No ``SHA256SUMS`` file is read at run time:
  a manifest delivered next to the data lives in the same trust domain as the
  data and could be forged. The caller can still opt out explicitly with
  ``trust_legacy_pickle=True``.
* The file is read once into memory, the digest is computed over those bytes,
  and every ``np.load`` call reads a ``BytesIO`` view of the same bytes. This
  removes the time-of-check/time-of-use gap between hashing and loading.
* Any other pickled ``.npz`` raises
  :class:`~chaotic_pfc.errors.UntrustedDataError`.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from chaotic_pfc.errors import InvalidParameterError, UntrustedDataError

from ._trusted_checkpoints import TRUSTED_CHECKPOINT_SHA256
from ._types import FILTER_TYPES, WINDOW_DISPLAY_NAMES, SweepResult


def _load_legacy_pickled_metadata(blob: bytes, path: Path) -> dict[str, Any]:
    """Read legacy object-array metadata from already verified bytes."""
    with np.load(io.BytesIO(blob), allow_pickle=True) as data:
        raw = data["metadata"]
    try:
        return dict(raw.tolist())
    except (TypeError, ValueError):
        warnings.warn(
            f"corrupted metadata in {path}, using empty dict",
            RuntimeWarning,
            stacklevel=3,
        )
        return {}


def save_sweep(result: SweepResult, path: str | Path) -> Path:
    """Save a :class:`~chaotic_pfc.analysis.sweep.SweepResult` to a compressed ``.npz``.

    ``metadata`` is stored as a JSON string (``dtype='<U'``) so the resulting
    archive can be loaded with ``allow_pickle=False``.

    Raises
    ------
    InvalidParameterError
        If *result* is not a :class:`SweepResult` or *path* is empty.
    """
    if not isinstance(result, SweepResult):
        raise InvalidParameterError(f"result must be a SweepResult, got {type(result).__name__}")
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise InvalidParameterError("path must be a non-empty path")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "h": result.h,
        "h_desvio": result.h_std,
        "cutoff_grid": result.cutoff_grid,
        "num_taps_grid": result.num_taps_grid,
        "window": result.window,
        "filter_type": result.filter_type,
        "metadata": np.array(json.dumps(result.metadata), dtype=str),
    }
    if result.n_iters_used is not None:
        payload["n_iters_used"] = result.n_iters_used
    if result.v_mean is not None:
        payload["v_mean"] = result.v_mean
    np.savez(path, **payload)  # type: ignore[arg-type]  # kwargs are arrays/scalars by construction
    return path


def load_sweep(path: str | Path, *, trust_legacy_pickle: bool = False) -> SweepResult:
    """Load a sweep from ``.npz`` produced by :func:`save_sweep`.

    Follows the "verified integrity before trust" rule: the file is read once
    into memory, the digest is computed over those bytes, and every archive
    read uses a ``BytesIO`` view of the same bytes. If ``metadata`` is a legacy
    pickled object array, it is deserialised only when the digest is listed in
    the compiled trust anchor
    :data:`~chaotic_pfc.analysis.sweep._trusted_checkpoints.TRUSTED_CHECKPOINT_SHA256`
    or when ``trust_legacy_pickle=True`` is explicitly passed.

    Parameters
    ----------
    path
        Path to the ``.npz`` checkpoint.
    trust_legacy_pickle
        When ``True``, allow deserialising legacy pickled ``metadata`` from a
        file whose digest is not in the compiled trust anchor (emits a
        :class:`UserWarning`).

    Raises
    ------
    InvalidParameterError
        If *path* or *trust_legacy_pickle* has an invalid type.
    UntrustedDataError
        If the archive contains pickled object arrays and is neither in the
        compiled trust anchor nor loaded with ``trust_legacy_pickle=True``.
    ValueError
        If the archive is unreadable or missing required fields.

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
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise InvalidParameterError("path must be a non-empty path")
    if not isinstance(trust_legacy_pickle, bool):
        raise InvalidParameterError(
            f"trust_legacy_pickle must be a bool, got {trust_legacy_pickle!r}"
        )
    path = Path(path)
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"invalid or corrupted checkpoint: {path} "
            f"(regenerate it with `chaotic-pfc sweeps regenerate`): {exc}"
        ) from exc
    digest = hashlib.sha256(blob).hexdigest().lower()

    try:
        data = np.load(io.BytesIO(blob), allow_pickle=False)
        files = data.files
    except Exception as exc:  # unreadable or not an .npz archive
        raise ValueError(
            f"invalid or corrupted checkpoint: {path} "
            f"(regenerate it with `chaotic-pfc sweeps regenerate`): {exc}"
        ) from exc

    with data:
        try:
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

            n_iters_used = data["n_iters_used"] if "n_iters_used" in files else None
            v_mean = data["v_mean"] if "v_mean" in files else None
            num_taps_grid = data["num_taps_grid"] if "num_taps_grid" in files else data["coef"]
            cutoff_grid = data["cutoff_grid"] if "cutoff_grid" in files else data["wcorte"]
            h = data["h"]
            h_std = data["h_desvio"]
        except ValueError as exc:
            if "allow_pickle=False" in str(exc):
                raise UntrustedDataError(
                    f"Refusing to deserialise pickled object array in {path}: "
                    "non-metadata fields must never use pickle."
                ) from exc
            raise

        metadata: dict[str, Any] = {}
        need_legacy_pickle = False
        if "metadata" in files:
            try:
                raw_meta = data["metadata"]
            except ValueError as exc:
                if "allow_pickle=False" in str(exc):
                    need_legacy_pickle = True
                else:
                    raise
            else:
                try:
                    if raw_meta.ndim == 0:
                        parsed = json.loads(str(raw_meta.item()))
                        if not isinstance(parsed, dict):
                            raise ValueError("JSON metadata is not an object")
                        metadata = parsed
                    else:
                        metadata = dict(raw_meta.tolist())
                except (TypeError, ValueError, json.JSONDecodeError):
                    warnings.warn(
                        f"corrupted metadata in {path}, using empty dict",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    if need_legacy_pickle:
        if digest in TRUSTED_CHECKPOINT_SHA256:
            metadata = _load_legacy_pickled_metadata(blob, path)
        elif trust_legacy_pickle:
            warnings.warn(
                f"Loading untrusted pickled metadata from {path} with trust_legacy_pickle=True",
                UserWarning,
                stacklevel=2,
            )
            metadata = _load_legacy_pickled_metadata(blob, path)
        else:
            raise UntrustedDataError(
                f"Refusing to deserialise pickled object array in {path}: "
                "file SHA-256 is not in the compiled trust anchor. "
                "Re-save with save_sweep() or pass trust_legacy_pickle=True if you trust the source."
            )

    return SweepResult(
        h=h,
        h_std=h_std,
        num_taps_grid=num_taps_grid,
        cutoff_grid=cutoff_grid,
        window=window,
        filter_type=filter_type,
        n_iters_used=n_iters_used,
        v_mean=v_mean,
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

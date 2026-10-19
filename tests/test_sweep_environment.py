"""Environment metadata and SciPy ``firwin`` anchoring for the versioned sweeps.

Two independent guarantees are encoded here:

* ``data/sweeps/ENVIRONMENT.json`` records the Python/NumPy/SciPy/Numba stack
  that regenerated the 124 checkpoints, and this stack must stay aligned with
  ``requirements-lock.txt`` so the lock reproduces the generation environment.

* The FIR coefficients are not stored in the ``.npz`` files — the ``num_taps_grid``
  field holds the number of taps :math:`N_s` of each swept order. This module
  recomputes the taps with ``scipy.signal.firwin`` and checks them against the
  ``num_taps_grid`` field, and pins a fingerprint of the full coefficient bank so that a
  change in ``firwin`` fails with a message that names SciPy.

Set ``CHAOTIC_PFC_SWEEPS_DIR`` to point the checks at a copy of the tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.signal import firwin

from chaotic_pfc import load_sweep
from chaotic_pfc.analysis.sweep._orchestration import precompute_fir_bank
from chaotic_pfc.analysis.sweep._types import FILTER_TYPES, WINDOWS

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEPS_DIR = Path(os.environ.get("CHAOTIC_PFC_SWEEPS_DIR", REPO_ROOT / "data" / "sweeps")).resolve()
ENVIRONMENT = SWEEPS_DIR / "ENVIRONMENT.json"
LOCK = REPO_ROOT / "requirements-lock.txt"

# Pinned stack recorded in ENVIRONMENT.json.
_LOCKED_PACKAGES = ("numpy", "scipy", "numba", "llvmlite", "matplotlib")

# Canonical grid for the coefficient fingerprint (odd taps work for every
# filter type). Regenerate by running
# ``tests/test_sweep_environment.py::_compute_fingerprint`` if SciPy's
# ``firwin`` changes on purpose.
_GOLDEN_ORDERS = [3, 11, 21, 41]
_GOLDEN_CUTOFFS = [0.1, 0.3, 0.5, 0.7]
_GOLDEN_KAISER_BETA = 5.0
GOLDEN_FIRWIN_SHA256 = "6465003ea465bc90dffa356d439b2ec1c0ac5bf91d46e7a7d710ba997c49adc3"


def _environment() -> dict:
    return json.loads(ENVIRONMENT.read_text(encoding="utf-8"))


def _lock_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\\\s]+)", line)
        if match:
            versions[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2).strip()
    return versions


def _checkpoint_paths() -> list[Path]:
    return sorted(SWEEPS_DIR.rglob("*.npz"))


def _compute_fingerprint() -> str:
    """sha256 of the coefficient banks over the canonical grid."""
    digest = hashlib.sha256()
    for filter_type in FILTER_TYPES:
        for window in WINDOWS:
            bank, _ = precompute_fir_bank(
                _GOLDEN_ORDERS,
                _GOLDEN_CUTOFFS,
                filter_type,
                window,
                kaiser_beta=_GOLDEN_KAISER_BETA,
            )
            digest.update(np.ascontiguousarray(bank, dtype=np.float64).tobytes())
    return digest.hexdigest()


def test_environment_matches_lock() -> None:
    """ENVIRONMENT.json pins the same stack as requirements-lock.txt."""
    env = _environment()
    versions = _lock_versions()
    mismatched = {
        package: (env[package], versions.get(package))
        for package in _LOCKED_PACKAGES
        if env.get(package) != versions.get(package)
    }
    assert not mismatched, (
        "data/sweeps/ENVIRONMENT.json and requirements-lock.txt disagree on "
        f"{mismatched}; realign both to the environment that regenerated the sweeps."
    )


def test_environment_records_the_key_fields() -> None:
    """ENVIRONMENT.json carries the fields the release notes promise."""
    env = _environment()
    for key in (
        *_LOCKED_PACKAGES,
        "python",
        "platform",
        "package_version",
        "regeneration_date",
        "checkpoints",
    ):
        assert key in env, f"ENVIRONMENT.json is missing {key!r}"
    assert env["checkpoints"] == len(_checkpoint_paths())


def test_firwin_reproduces_stored_taps() -> None:
    """``firwin`` returns exactly the number of taps stored in the ``num_taps_grid`` field.

    The checkpoints do not store the coefficient values, so the reproducible
    link to SciPy is the number of taps of each swept order; a SciPy release
    that changes the ``firwin`` tap convention is caught here with a message
    that names SciPy and the recorded version.
    """
    paths = _checkpoint_paths()
    assert paths, f"no checkpoints under {SWEEPS_DIR}"
    expected = _environment().get("scipy", "the recorded SciPy")
    mismatches: list[tuple[str, int, int]] = []
    for path in paths:
        result = load_sweep(path)
        num_taps_grid = np.asarray(result.num_taps_grid, dtype=int)
        beta = float(result.metadata.get("kaiser_beta", _GOLDEN_KAISER_BETA))
        win_arg = ("kaiser", beta) if result.window == "kaiser" else result.window
        if result.filter_type in ("bandpass", "bandstop"):
            cutoff: float | list[float] = [0.3 - 0.025, 0.3 + 0.025]
        else:
            cutoff = 0.3
        for n_taps in (int(num_taps_grid.min()), int(num_taps_grid.max())):
            # Hann/Bartlett with two taps divide by a zero gain on purpose (the
            # cell is classified as divergent); silence only that arithmetic.
            with np.errstate(invalid="ignore", divide="ignore"):
                coeffs = firwin(n_taps, cutoff, pass_zero=result.filter_type, window=win_arg)
            if coeffs.size != n_taps:
                mismatches.append((path.parent.name, n_taps, int(coeffs.size)))
    assert not mismatches, (
        "scipy.signal.firwin returned a different number of taps than the 'num_taps_grid' "
        f"field recorded in the checkpoints (config, stored, recomputed) = "
        f"{mismatches[:3]}; the checkpoints were generated with SciPy {expected}. "
        "Check the SciPy version (see data/sweeps/ENVIRONMENT.json)."
    )


def test_dc_gain_follows_filter_type() -> None:
    """The recomputed DC gain obeys the pass-zero convention of each filter type.

    ``firwin`` with ``pass_zero=filter_type`` must give a unit DC gain for the
    lowpass and bandstop filters and a near-zero gain for highpass and bandpass
    (away from the low-order cutoff-0 corner). A SciPy change to the window or
    the ``pass_zero`` handling fails here with a SciPy message.
    """
    cutoff_grid = np.array([0.3, 0.5, 0.7])
    expected = _environment().get("scipy", "the recorded SciPy")
    for filter_type, target, tol in (
        ("lowpass", 1.0, 0.2),
        ("bandstop", 1.0, 0.2),
        ("highpass", 0.0, 0.2),
        ("bandpass", 0.0, 0.2),
    ):
        _, gains = precompute_fir_bank([21], cutoff_grid, filter_type, "hamming")
        if not np.allclose(gains[0], target, atol=tol):
            raise AssertionError(
                f"DC gain of the {filter_type} filter is {gains[0].tolist()}, "
                f"expected ~{target}; scipy.signal.firwin changed "
                f"(checkpoints generated with SciPy {expected})."
            )


def test_firwin_coefficient_fingerprint() -> None:
    """The full FIR coefficient bank is byte-stable for the recorded SciPy."""
    got = _compute_fingerprint()
    assert got == GOLDEN_FIRWIN_SHA256, (
        "the FIR coefficient bank no longer matches the pinned fingerprint "
        f"({got} != {GOLDEN_FIRWIN_SHA256}); scipy.signal.firwin changed. The "
        "checkpoints were generated with SciPy "
        f"{_environment().get('scipy', '?')}; check the SciPy version or update "
        "ENVIRONMENT.json, requirements-lock.txt and the golden hash together."
    )


def test_stored_orders_are_the_canonical_grid() -> None:
    """The ``num_taps_grid`` field of every checkpoint is the documented order grid."""
    bad: list[tuple[str, list[int]]] = []
    for path in _checkpoint_paths():
        result = load_sweep(path)
        num_taps_grid = np.asarray(result.num_taps_grid, dtype=int)
        if result.filter_type in ("highpass", "bandstop"):
            expected = np.arange(3, 43, 2)
        else:
            expected = np.arange(2, 42)
        if not np.array_equal(num_taps_grid, expected):
            bad.append((path.parent.name, num_taps_grid.tolist()))
    assert not bad, f"checkpoint num_taps_grid are not the canonical grid: {bad[:3]}"

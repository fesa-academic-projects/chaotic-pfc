"""Build a small but structurally complete sweep tree for the paper-tooling tests.

The manuscript tooling (``chaotic_pfc.paper``) assumes the canonical layout of
the versioned sweeps: six fixed windows plus the 25 Kaiser ``beta_K`` values for
each of the four filter types (124 configurations), all sharing one cutoff
grid. The checkpoints here are synthetic (seeded), with a reduced grid so the
tests run in seconds, but the *structure* is exact, which is what exercises the
section code paths of ``paper.numbers`` and the figure generators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chaotic_pfc.analysis.sweep import save_sweep
from chaotic_pfc.analysis.sweep._types import SweepResult

FILTER_TYPES = ("lowpass", "highpass", "bandpass", "bandstop")

FIXED_WINDOWS = (
    ("boxcar", "Rectangular"),
    ("bartlett", "Bartlett"),
    ("hann", "Hann"),
    ("hamming", "Hamming"),
    ("blackman", "Blackman"),
    ("blackmanharris", "Blackman--Harris"),
)
KAISER_BETAS = tuple(round(2.0 + 0.5 * i, 1) for i in range(25))
CUTOFFS = np.round(np.linspace(0.01, 0.99, 10), 6)
# Orders spanning the Nz ranges the sections rely on (Nz <= 10 and Nz > 20, and
# Nz = 4 for the highpass map). Highpass and bandstop require odd tap counts.
TAPS_ODD = (3, 5, 7, 9, 11, 21, 23, 31, 33, 41)
TAPS_EVENISH = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 31, 32, 41)


def _taps(filter_type: str) -> tuple[int, ...]:
    return TAPS_ODD if filter_type in ("highpass", "bandstop") else TAPS_EVENISH


def _make_result(
    window: str,
    filter_type: str,
    taps: tuple[int, ...],
    kaiser_beta: float | None,
    rng: np.random.Generator,
) -> SweepResult:
    shape = (len(taps), len(CUTOFFS))
    draw = rng.random(shape)
    h = np.where(
        draw < 0.45,
        np.nan,
        np.where(draw < 0.85, -0.02 + 0.01 * rng.random(shape), 0.01 + 0.08 * rng.random(shape)),
    )
    h_std = np.where(np.isnan(h), np.nan, 0.005 + 0.005 * rng.random(shape))
    n_iters_used = np.where(np.isnan(h), np.nan, 700.0 + 25.0 * rng.random(shape))
    metadata: dict = {
        "n_fallback": int(rng.integers(0, 20)),
        "adaptive": True,
        "alpha": 1.4,
        "beta": 0.3,
    }
    if kaiser_beta is not None:
        metadata["kaiser_beta"] = kaiser_beta
    return SweepResult(
        h=h,
        h_std=h_std,
        orders=np.asarray(taps, dtype=np.int64),
        cutoffs=CUTOFFS.copy(),
        window=window,
        filter_type=filter_type,
        n_iters_used=n_iters_used,
        metadata=metadata,
    )


def build_synthetic_sweeps(root: Path) -> Path:
    """Write the 124 synthetic checkpoints under ``root`` and return ``root``."""
    rng = np.random.default_rng(20261012)
    for filter_type in FILTER_TYPES:
        taps = _taps(filter_type)
        for window, display in FIXED_WINDOWS:
            save_sweep(
                _make_result(window, filter_type, taps, None, rng),
                root / f"{display} ({filter_type})" / "variables_lyapunov.npz",
            )
        for beta in KAISER_BETAS:
            save_sweep(
                _make_result("kaiser", filter_type, taps, beta, rng),
                root / "kaiser" / filter_type / f"beta_{beta:.2f}" / "variables_lyapunov.npz",
            )
    return root

"""Regression tests for the chaos boundary and the recentred fallback band."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chaotic_pfc.analysis.stats import area_summary
from chaotic_pfc.analysis.sweep import (
    LAMBDA_CHAOS_THRESHOLD,
    SweepResult,
    load_sweep,
    run_sweep,
    save_sweep,
)
from chaotic_pfc.analysis.sweep_plotting import OrbitClass, classify


def test_threshold_value_is_frozen() -> None:
    """The manuscript criterion (10) uses theta = 5e-3."""
    assert LAMBDA_CHAOS_THRESHOLD == 5e-3


def test_classify_uses_the_threshold_not_zero() -> None:
    h = np.array([np.nan, 0.0, 0.004999, 0.005, 0.005001, 0.2])
    classes = classify(h)
    assert classes.tolist() == [
        int(OrbitClass.DIVERGENT),
        int(OrbitClass.PERIODIC),
        int(OrbitClass.PERIODIC),
        int(OrbitClass.PERIODIC),
        int(OrbitClass.CHAOTIC),
        int(OrbitClass.CHAOTIC),
    ]


def test_area_summary_uses_the_threshold() -> None:
    h = np.array([[0.004, 0.006], [np.nan, 0.005]])
    result = SweepResult(
        h=h,
        h_std=np.full_like(h, np.nan),
        num_taps_grid=np.array([2, 3], dtype=np.int64),
        cutoff_grid=np.array([0.25, 0.5]),
        window="hamming",
        filter_type="lowpass",
    )
    summary = area_summary(result)
    assert summary["n_chaotic"] == 1
    assert summary["n_periodic"] == 2
    assert summary["n_divergent"] == 1


def test_run_sweep_records_threshold_and_single_vector_mean(tmp_path: Path) -> None:
    result = run_sweep(
        "hamming",
        "lowpass",
        num_taps_grid=np.array([3]),
        cutoff_grid=np.array([0.5]),
        Nitera=50,
        Nmap=300,
        n_initial=4,
        warmup=False,
    )
    assert result.metadata["chaos_threshold"] == LAMBDA_CHAOS_THRESHOLD
    assert result.metadata["vonly_margin"] == 5e-3
    assert result.v_mean is not None
    np.testing.assert_array_equal(result.v_mean, result.h)

    path = save_sweep(result, tmp_path / "threshold.npz")
    loaded = load_sweep(path)
    assert loaded.metadata["chaos_threshold"] == LAMBDA_CHAOS_THRESHOLD
    assert loaded.metadata["vonly_margin"] == 5e-3
    assert loaded.v_mean is not None
    np.testing.assert_array_equal(loaded.v_mean, result.v_mean)


@pytest.mark.parametrize("v", [-0.006, -0.001, 0.0])
def test_points_below_or_at_threshold_do_not_trigger_fallback(v: float) -> None:
    """The fallback is centred at +5e-3, not at zero."""
    assert abs(v - LAMBDA_CHAOS_THRESHOLD) >= 5e-3


def test_new_fallback_band_contains_only_positive_means() -> None:
    band = np.linspace(-0.02, 0.02, 4001)
    trigger = np.abs(band - LAMBDA_CHAOS_THRESHOLD) < 5e-3
    assert np.all(band[trigger] > 0.0)
    assert np.all(band[trigger] < 1e-2)

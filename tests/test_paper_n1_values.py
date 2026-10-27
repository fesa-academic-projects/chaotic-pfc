"""Regression values for the result labels added in the N1 pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from chaotic_pfc.paper.numbers import compute

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def numbers() -> dict[str, str]:
    """Compute the manuscript labels from the versioned sweeps once."""
    return compute(Path("data/sweeps"), extras=False)


def test_threshold_sensitivity(numbers: dict[str, str]) -> None:
    assert numbers["ROB.threshold_1e-2.lowpass_drop_pct"] == "2.9"
    assert numbers["ROB.threshold_1e-2.highpass_drop_pct"] == "15.7"
    assert numbers["ROB.threshold_1e-2.bandpass_drop_pct"] == "2.3"
    assert numbers["ROB.threshold_1e-2.bandstop_drop_pct"] == "2.5"
    assert numbers["ROB.threshold_1e-3.lowpass_gain_pct"] == "17"


def test_disagreement_and_attenuation(numbers: dict[str, str]) -> None:
    assert numbers["CON.lowpass.disagreement_pct_Nz_le10"] == "38"
    assert numbers["ATT.bandstop.beta0"] == "20.1"
    assert numbers["ATT.bandstop.Nz10_min"] == "0.70"
    assert numbers["ATT.bandstop.Nz10_max"] == "2.04"
    assert numbers["ATT.bandstop.others_max_db"] == "6.1"
    assert numbers["ATT.bandstop.others_bound_db"] == "6.2"


def test_filter_extremes(numbers: dict[str, str]) -> None:
    assert numbers["T2.bandstop.other_Nc_min"] == "42"
    assert numbers["T2.bandstop.other_Nc_max"] == "55"
    assert numbers["KAI.highpass.beta_ge17_min"] == "3.6"
    assert numbers["KAI.highpass.beta_ge17_max"] == "4.0"
    assert numbers["WIDTH.bandpass.Nz10_min"] == "0.18"
    assert numbers["WIDTH.bandpass.Nz10_max"] == "0.38"


def test_locality_and_cost(numbers: dict[str, str]) -> None:
    assert numbers["LOCALITY.n_points"] == "30"
    assert numbers["LOCALITY.kept_positive"] == "18"
    assert numbers["LOCALITY.changed_sign"] == "12"
    assert numbers["LOCALITY.median_slope"] == "-0.95"
    assert numbers["COST.sweeps_184_minutes"] == "10"


def test_grid_and_review_constants(numbers: dict[str, str]) -> None:
    assert numbers["GRID.n_sweeps"] == "184"
    assert numbers["GRID.odd_sweeps"] == "92"
    assert numbers["GRID.fixed_configs"] == "24"
    assert numbers["GRID.kaiser_configs"] == "160"
    assert numbers["GRID.kaiser_betas"] == "40"
    assert numbers["GRID.total_points"] == "552000"
    assert numbers["REFERENCE.sphere_radius"] == "0.01"
    assert numbers["REVIEW.n_per_type"] == "400"
    assert numbers["REVIEW.n_points"] == "200"
    assert numbers["REVIEW.range"] == "1.5"


def test_positive_estimates_and_bins(numbers: dict[str, str]) -> None:
    assert numbers["POS.positive"] == "54470"
    assert numbers["POS.marginal_le_1e-3"] == "13068"
    assert numbers["POS.bin_1_2"] == "4732"
    assert numbers["POS.bin_2_3"] == "1870"
    assert numbers["POS.bin_3_4"] == "683"
    assert numbers["POS.bin_4_5"] == "498"
    assert numbers["POS.bin_5_20_min"] == "400"
    assert numbers["POS.bin_5_20_max"] == "516"

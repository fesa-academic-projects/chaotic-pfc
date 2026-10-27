"""Labels for the DC-gain convergence shares of the JCIS manuscript."""

from __future__ import annotations

from chaotic_pfc.paper.numbers import section_review_metrics


def test_high_gain_convergence_labels() -> None:
    """The shares of non-chaotic points with |G| >= 0.1 are labelled."""
    out: dict[str, str] = {}
    section_review_metrics(out)

    expected = {
        "lowpass": "21",
        "highpass": "34",
        "bandpass": "45",
        "bandstop": "0",
    }
    for filter_type, value in expected.items():
        assert out[f"GAIN.{filter_type}.converged_pct_absG_ge_0.1"] == value
        assert out[f"GAIN.{filter_type}.n_absG_ge_0.1"]
        assert out[f"GAIN.{filter_type}.converged_absG_ge_0.1"]

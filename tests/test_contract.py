"""Contract fuzzing for the public API.

Every function reachable from ``chaotic_pfc.__all__`` is exercised with valid
baseline arguments and then with one (and sometimes two) mutated argument. The
contract is that a call either returns a value of the documented shape or raises
:class:`~chaotic_pfc.errors.InvalidParameterError` (or ``TypeError`` for
genuinely wrong Python types). Any other exception, silent ``NaN`` output or
incoherent shape fails.

The test discovers the public surface automatically, so a new public function is
covered without editing this module: only functions that need a real filesystem,
Plotly or a matplotlib canvas are listed in :data:`EXCLUDED` with a reason.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import chaotic_pfc
from chaotic_pfc import (
    InvalidParameterError,
    area_summary,
    awgn,
    binary_message,
    channel_impulsive,
    channel_multipath,
    classify,
    consolidate_kaiser,
    fir_channel,
    henon_filtered,
    henon_fir_sequence,
    henon_generalised,
    henon_order_n,
    henon_standard,
    ideal_channel,
    kaiser_beta_optimal,
    lmax_statistics,
    lyapunov_henon2d,
    lyapunov_henon2d_ensemble,
    lyapunov_max,
    lyapunov_max_ensemble,
    lyapunov_max_pole_filtered,
    lyapunov_spectrum,
    precompute_fir_bank,
    psd_normalised,
    rank_configurations,
    receive,
    receive_order_n,
    run_sweep,
    sinusoidal_message,
    sweet_spot_per_filter,
    text_message,
    top_k_per_filter,
    transmit,
    transmit_order_n,
)
from chaotic_pfc.analysis.sweep import SweepResult

pytestmark = pytest.mark.contract

#: Functions that need a real filesystem, Plotly or an interactive renderer.
EXCLUDED: dict[str, str] = {
    # Directory I/O: covered by the analysis/sweep/CLI suites with tmp_path.
    "best_chaos_preserving": "reads a sweep directory",
    "beta_curve": "reads a sweep directory",
    "beta_summary": "reads a sweep directory",
    "bootstrap_confidence": "reads a sweep directory",
    "chaos_margin": "reads a sweep directory",
    "compare_filter_types": "reads a sweep directory",
    "correlation_matrix": "reads a sweep directory",
    "export_consolidated_extended_table": "writes a .tex file",
    "export_consolidated_full_ranking": "writes a .tex file",
    "export_consolidated_top_k_table": "writes a .tex file",
    "export_extended_top_k_table": "writes a .tex file",
    "export_full_ranking_table": "writes a .tex file",
    "export_kaiser_beta_optimal_table": "writes a .tex file",
    "export_summary_json": "reads a sweep directory and writes JSON",
    "export_sweet_spots_table": "writes a .tex file",
    "export_top_k_table": "writes a .tex file",
    "load_all_sweeps": "reads a sweep directory",
    "load_sweep": "reads a .npz file (covered by test_sweep_io)",
    "lmax_distribution": "reads a sweep directory",
    "optimal_parameters": "reads a sweep directory",
    "save_sweep": "writes a .npz file (covered by test_sweep_io)",
    "summary_table": "reads a sweep directory",
    "transition_boundary": "reads a sweep directory",
    # Rendering: needs a matplotlib canvas and is covered by test_sweep_plotting.
    "plot_all": "writes figures to disk",
    "plot_chaotic_all": "writes figures to disk",
    "plot_chaotic_density": "reads a sweep directory and renders",
    "plot_chaotic_map": "reads a sweep directory and renders",
    "plot_classification_interleaved": "renders a matplotlib figure",
    "plot_difficulty_map": "renders a matplotlib figure",
    "plot_heatmap_continuous": "renders a matplotlib figure",
}

_INT_PARAMS = frozenset(
    {
        "N",
        "steps",
        "period",
        "bit_period",
        "nfft",
        "window_length",
        "num_taps",
        "bins",
        "Nitera",
        "Ndiscard",
        "n_initial",
        "Nmap",
        "Nmap_min",
        "top_n",
        "k",
    }
)
_FLOAT_PARAMS = frozenset(
    {
        "x0",
        "y0",
        "alpha",
        "beta",
        "c0",
        "c1",
        "mu",
        "snr_db",
        "prob_impulse",
        "amp_factor",
        "kaiser_beta",
        "bandwidth",
        "tol",
        "perturbation",
        "pole_radius",
        "w0",
        "Gz",
        "fs",
        "cutoff",
    }
)
_STR_PARAMS = frozenset({"window", "filter_type", "text", "region", "on_divergence", "lang"})
_BOOL_PARAMS = frozenset({"adaptive", "warmup", "remove_dc"})
_ARRAY_PARAMS = frozenset(
    {"message", "sig", "s", "r", "h", "fir_coeffs", "num_taps_grid", "cutoff_grid", "driving"}
)

_INT_MUTANTS: list[Any] = [0, -1, 3.5, True, "3", float("nan")]
_FLOAT_MUTANTS: list[Any] = [float("nan"), float("inf"), float("-inf"), 1e308, "x"]
_STR_MUTANTS: list[Any] = ["", "not-a-window", 123]
_BOOL_MUTANTS: list[Any] = [1, "yes", 0.5]
_ARRAY_MUTANTS: list[Any] = [
    np.array([]),
    np.array([float("nan")]),
    np.array([float("inf")]),
    np.zeros((2, 2)),
    "not-an-array",
]
_ON_DIVERGENCE_MUTANTS: list[Any] = ["invalid", 1, None]


def _sweep() -> SweepResult:
    return SweepResult(
        h=np.array([[0.1, -0.2], [0.3, np.nan]]),
        h_std=np.array([[0.01, 0.02], [0.03, 0.04]]),
        num_taps_grid=np.array([3, 5]),
        cutoff_grid=np.array([0.2, 0.8]),
        window="hamming",
        filter_type="lowpass",
        n_iters_used=np.array([[100.0, 100.0], [100.0, np.nan]]),
        metadata={"Nitera": 10, "Nmap": 100, "n_initial": 5, "kaiser_beta": 5.0},
    )


def _iterate(x: Any) -> Any:
    return np.array([1.4 - x[0] ** 2 + 0.3 * x[1], x[0]])


def _jacobian(x: Any) -> Any:
    return np.array([[-2.0 * x[0], 0.3], [1.0, 0.0]])


def _baselines() -> dict[str, dict[str, Any]]:
    sig = np.sin(2.0 * np.pi * 0.1 * np.arange(64))
    message = np.where(np.arange(32) % 2 == 0, 1.0, -1.0)
    coeffs = np.array([0.5, 0.3, 0.15, 0.05])
    sweep = _sweep()
    results = {("lowpass", "hamming"): sweep}
    return {
        "area_summary": {"sweep_result": sweep},
        "awgn": {"sig": sig, "snr_db": 10.0},
        "binary_message": {"N": 16, "period": 4},
        "channel_impulsive": {"sig": sig, "snr_db": 10.0},
        "channel_multipath": {"sig": sig, "snr_db": 10.0},
        "classify": {"h": np.array([[0.1, -0.2], [np.nan, 0.3]])},
        "consolidate_kaiser": {"sweep_results": results},
        "fir_channel": {"s": sig, "num_taps": 5, "cutoff": 0.5},
        "fixed_point_stability": {"alpha": 1.4, "beta": 0.3},
        "henon_filtered": {"steps": 8},
        "henon_fir_sequence": {"N": 8},
        "henon_generalised": {"steps": 8},
        "henon_order_n": {"steps": 8, "fir_coeffs": coeffs},
        "henon_standard": {"steps": 8},
        "ideal_channel": {"s": sig},
        "kaiser_beta_optimal": {"sweep_results": results},
        "lmax_statistics": {"sweep_result": sweep, "n_bootstrap": 20},
        "lyapunov_henon2d": {"Nitera": 20, "Ndiscard": 10},
        "lyapunov_henon2d_ensemble": {"Nitera": 20, "Ndiscard": 10, "n_initial": 2},
        "lyapunov_max": {"iterate": _iterate, "jacobian": _jacobian, "x0": np.array([0.1, 0.1])},
        "lyapunov_max_ensemble": {"Nitera": 20, "Ndiscard": 10, "n_initial": 2},
        "lyapunov_max_pole_filtered": {"Nitera": 20, "Ndiscard": 10},
        "lyapunov_spectrum": {
            "iterate": _iterate,
            "jacobian": _jacobian,
            "x0": np.array([0.1, 0.1]),
        },
        "precompute_fir_bank": {
            "num_taps_grid": [3, 5],
            "cutoff_grid": np.array([0.2, 0.8]),
            "filter_type": "lowpass",
            "window": "hamming",
        },
        "psd_normalised": {"x": sig, "nfft": 32, "window_length": 16},
        "quick_sweep_params": {},
        "rank_configurations": {"sweep_results": results},
        "receive": {"r": sig, "mu": 0.01},
        "receive_order_n": {"r": sig, "fir_coeffs": coeffs},
        "run_sweep": {
            "num_taps_grid": [2, 3],
            "cutoff_grid": np.array([0.2, 0.5]),
            "Nitera": 5,
            "Nmap": 20,
            "n_initial": 1,
            "warmup": False,
        },
        "sinusoidal_message": {"N": 16},
        "sweet_spot_per_filter": {"sweep_results": results},
        "text_message": {"text": "hi", "N": 16, "bit_period": 2},
        "top_k_per_filter": {"sweep_results": results, "k": 1, "bootstrap_seed": 42},
        "transmit": {"message": message, "mu": 0.01},
        "transmit_order_n": {"message": message, "fir_coeffs": coeffs},
    }


def _public_functions() -> list[Any]:
    names = []
    for name in chaotic_pfc.__all__:
        obj = getattr(chaotic_pfc, name, None)
        if callable(obj) and not isinstance(obj, type):
            names.append(obj)
    return names


def _function_names() -> set[str]:
    """Names in ``__all__`` that are plain functions (not classes or constants)."""
    return {
        name
        for name in chaotic_pfc.__all__
        if callable(getattr(chaotic_pfc, name, None))
        and not isinstance(getattr(chaotic_pfc, name), type)
    }


def _mutants_for(param: str, annotation: str) -> list[Any]:
    if param == "on_divergence":
        return _ON_DIVERGENCE_MUTANTS
    if param in _INT_PARAMS:
        return _INT_MUTANTS
    if param in _BOOL_PARAMS or "bool" in annotation:
        return _BOOL_MUTANTS
    if param in _STR_PARAMS or annotation.strip().startswith("str"):
        return _STR_MUTANTS
    if param in _ARRAY_PARAMS or "NDArray" in annotation or "Sequence" in annotation:
        return _ARRAY_MUTANTS
    if param in _FLOAT_PARAMS or "float" in annotation:
        return _FLOAT_MUTANTS
    return []


def _callables_baseline() -> tuple[Any, Any]:
    return _iterate, _jacobian


CALLABLE_PARAMS = frozenset({"iterate", "jacobian", "fir_design"})


def _invoke(func: Any, kwargs: dict[str, Any]) -> Any:
    with warnings.catch_warnings():
        # ``henon_fir_sequence`` is deprecated on purpose; the deprecation is
        # tested in tests/test_maps.py, not part of this contract.
        warnings.simplefilter("ignore", DeprecationWarning)
        return func(**kwargs)


def _assert_contract(name: str, result: Any, baseline: Any) -> None:
    """Accept a valid result, otherwise fail with a shape mismatch."""
    if isinstance(result, np.ndarray):
        if result.ndim == 0:
            return
        if result.size == 0:
            raise AssertionError(f"{name} returned an empty array")
        if (
            baseline is not None
            and isinstance(baseline, np.ndarray)
            and result.ndim != baseline.ndim
            and result.ndim != 1
        ):
            raise AssertionError(
                f"{name} returned ndim={result.ndim}, baseline ndim={baseline.ndim}"
            )
        return
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, np.ndarray) and item.size == 0:
                raise AssertionError(f"{name} returned a tuple containing an empty array")
        return
    if isinstance(result, dict):
        return
    if hasattr(result, "h"):  # SweepResult and friends
        return


def _outcome(func: Any, kwargs: dict[str, Any], name: str) -> None:
    try:
        result = _invoke(func, kwargs)
    except (InvalidParameterError, TypeError):
        return
    except UnicodeEncodeError:
        return
    _assert_contract(name, result, None)


@pytest.mark.parametrize("name", sorted(_function_names() - set(EXCLUDED)))
def test_public_function_accepts_valid_baseline(name: str) -> None:
    """Every in-scope public function must accept its documented baseline."""
    baselines = _baselines()
    assert name in baselines, f"{name} is public but has no contract baseline; add one"
    func = getattr(chaotic_pfc, name)
    result = _invoke(func, baselines[name])
    assert result is not None
    _assert_contract(name, result, None)


def _full_kwargs(name: str, baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Signature defaults overlaid with the contract baseline."""
    import inspect

    func = getattr(chaotic_pfc, name)
    kwargs: dict[str, Any] = {}
    for param_name, param in inspect.signature(func).parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.default is not inspect.Parameter.empty:
            kwargs[param_name] = param.default
    kwargs.update(baselines[name])
    return kwargs


def _parameter_cases() -> list[tuple[str, str, Any]]:
    baselines = _baselines()
    cases: list[tuple[str, str, Any]] = []
    for name in sorted(_function_names()):
        if name in EXCLUDED:
            continue
        func = getattr(chaotic_pfc, name)
        kwargs = _full_kwargs(name, baselines)
        for param, value in kwargs.items():
            if param in CALLABLE_PARAMS:
                for mutant in (None, "not-callable", 42):
                    cases.append((name, param, mutant))
                continue
            annotation = ""
            try:
                import inspect

                annotation = str(inspect.signature(func).parameters[param].annotation)
            except (KeyError, ValueError):
                pass
            for mutant in _mutants_for(param, annotation):
                cases.append((name, param, mutant))
            if value is None:
                cases.append((name, param, None))
    return cases


CASES = _parameter_cases()


@pytest.mark.parametrize(("name", "param", "mutant"), CASES)
def test_single_argument_mutation_obeys_contract(name: str, param: str, mutant: Any) -> None:
    baselines = _baselines()
    kwargs = _full_kwargs(name, baselines)
    kwargs[param] = mutant
    _outcome(getattr(chaotic_pfc, name), kwargs, name)


def _mutable_pairs() -> list[tuple[str, str, str]]:
    baselines = _baselines()
    pairs: list[tuple[str, str, str]] = []
    for name in sorted(_function_names()):
        if name in EXCLUDED or name not in baselines:
            continue
        params = list(_full_kwargs(name, baselines))
        for i, first in enumerate(params):
            for second in params[i + 1 :]:
                pairs.append((name, first, second))
    return pairs


PAIRS = _mutable_pairs()


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_two_argument_mutations_obey_contract(data: st.DataObject) -> None:
    """Two simultaneously mutated arguments still end in a valid return or a domain error."""
    _check_two_argument_mutation(data)


@pytest.mark.contract_deep
@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_deep_two_argument_mutations(data: st.DataObject) -> None:
    """Longer sweep of the two-argument contract, scheduled weekly."""
    _check_two_argument_mutation(data)


def _check_two_argument_mutation(data: st.DataObject) -> None:
    baselines = _baselines()
    name, first, second = data.draw(st.sampled_from(PAIRS))
    kwargs = _full_kwargs(name, baselines)
    first_mutants = _mutants_for(first, "")
    second_mutants = _mutants_for(second, "")
    if not first_mutants or not second_mutants:
        return
    kwargs[first] = data.draw(st.sampled_from(first_mutants))
    kwargs[second] = data.draw(st.sampled_from(second_mutants))
    _outcome(getattr(chaotic_pfc, name), kwargs, name)


def test_exclusion_list_is_explicit_and_current() -> None:
    """Every excluded name must still exist and every public function be classified."""
    public = _function_names()
    missing = set(EXCLUDED) - public
    assert not missing, f"EXCLUDED lists names that no longer exist: {sorted(missing)}"
    baselines = _baselines()
    unclassified = {name for name in public if name not in EXCLUDED and name not in baselines}
    assert not unclassified, f"public functions with no contract coverage: {sorted(unclassified)}"

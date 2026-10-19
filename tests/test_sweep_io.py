"""Sweep I/O, path inference and helpers."""

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from chaotic_pfc.analysis.sweep import (
    FILTER_TYPES,
    WINDOW_DISPLAY_NAMES,
    WINDOWS,
    SweepResult,
    _build_task_order,
    _infer_config_from_path,
    load_sweep,
    precompute_fir_bank,
    quick_sweep_params,
    run_sweep,
    save_sweep,
)

# ═══════════════════════════════════════════════════════════════════════════
# Catalogue sanity checks


class TestIO(unittest.TestCase):
    def test_round_trip_preserves_arrays(self):
        result = run_sweep(
            window="hann",
            filter_type="highpass",
            num_taps_grid=np.array([3, 5]),  # highpass requires odd num_taps_grid
            cutoff_grid=np.linspace(0.2, 0.8, 3),
            Nitera=20,
            Nmap=100,
            n_initial=2,
            seed=0,
            warmup=False,
        )
        with TemporaryDirectory() as td:
            path = Path(td) / "sweep.npz"
            save_sweep(result, path)
            self.assertTrue(path.exists())

            loaded = load_sweep(path)
            np.testing.assert_array_equal(loaded.num_taps_grid, result.num_taps_grid)
            np.testing.assert_array_equal(loaded.cutoff_grid, result.cutoff_grid)
            self.assertEqual(loaded.window, "hann")
            self.assertEqual(loaded.filter_type, "highpass")
            self.assertEqual(loaded.display_name, "Hann (highpass)")

            # NaN-aware array equality
            both_nan = np.isnan(loaded.h) & np.isnan(result.h)
            equal = both_nan | (loaded.h == result.h)
            self.assertTrue(np.all(equal))

    def test_legacy_format_inferred_from_path(self):
        """An .npz without explicit window/filter_type fields should still
        load correctly when the parent directory follows the naming
        convention 'Window (filter)/'."""
        with TemporaryDirectory() as td:
            subdir = Path(td) / "Blackman (lowpass)"
            subdir.mkdir()
            path = subdir / "variables_lyapunov.npz"

            np.savez(
                path,
                h=np.array([[0.1, 0.2]]),
                h_desvio=np.array([[0.01, 0.02]]),
                wcorte=np.array([0.3, 0.7]),
                coef=np.array([5]),
            )

            loaded = load_sweep(path)
            self.assertEqual(loaded.window, "blackman")
            self.assertEqual(loaded.filter_type, "lowpass")
            self.assertIsNone(loaded.n_iters_used)

    def test_n_iters_used_round_trip(self):
        """``n_iters_used`` must survive a save/load round-trip."""
        result = run_sweep(
            window="hamming",
            filter_type="lowpass",
            num_taps_grid=np.arange(2, 5),
            cutoff_grid=np.linspace(0.2, 0.8, 4),
            Nitera=30,
            Nmap=200,
            n_initial=3,
            seed=42,
            warmup=False,
            adaptive=True,
            Nmap_min=50,
            tol=1e-2,
        )
        self.assertIsNotNone(result.n_iters_used)
        with TemporaryDirectory() as td:
            path = Path(td) / "adaptive.npz"
            save_sweep(result, path)
            loaded = load_sweep(path)
            self.assertIsNotNone(loaded.n_iters_used)
            np.testing.assert_array_equal(
                np.isnan(loaded.n_iters_used),
                np.isnan(result.n_iters_used),
            )
            finite = ~np.isnan(loaded.n_iters_used)
            np.testing.assert_array_equal(
                loaded.n_iters_used[finite],
                result.n_iters_used[finite],
            )


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Lyapunov early-stop
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers(unittest.TestCase):
    def test_quick_sweep_params_reduces_work(self):
        num_taps_grid_lp, _num_taps_grid_hp, cutoff_grid, params = quick_sweep_params()
        self.assertLess(len(num_taps_grid_lp), 10)
        self.assertLess(len(cutoff_grid), 20)
        self.assertIsInstance(params, dict)
        self.assertIn("Nitera", params)
        self.assertLess(params["Nitera"], 500)

    def test_build_task_order_covers_all_combos(self):
        num_taps_grid = np.arange(2, 5)
        tasks = _build_task_order(num_taps_grid, Ncut=4)
        self.assertEqual(len(tasks), len(num_taps_grid) * 4)
        self.assertEqual(tasks.ndim, 1)

    def test_build_task_order_empty(self):
        tasks = _build_task_order(np.array([]), Ncut=0)
        self.assertEqual(len(tasks), 0)

    def test_infer_config_from_path_lowpass(self):
        path = Path("data/sweeps/Hamming (lowpass)/variables_lyapunov.npz")
        filt, cutoff = _infer_config_from_path(path)
        self.assertEqual(filt, "hamming")
        self.assertEqual(cutoff, "lowpass")

    def test_infer_config_from_path_highpass(self):
        path = Path("data/sweeps/Hamming (highpass)/variables_lyapunov.npz")
        filt, cutoff = _infer_config_from_path(path)
        self.assertEqual(filt, "hamming")
        self.assertEqual(cutoff, "highpass")

    def test_infer_config_from_path_unexpected_format(self):
        path = Path("kaiser/bandpass/beta_5.00/variables_lyapunov.npz")
        with self.assertRaises(ValueError):
            _infer_config_from_path(path)

    def test_display_name_all_filter_types(self):
        for ft in FILTER_TYPES:
            result = SweepResult(
                h=np.zeros((2, 2)),
                h_std=np.zeros((2, 2)),
                num_taps_grid=np.array([3, 4]),
                cutoff_grid=np.array([0.2, 0.8]),
                window="hamming",
                filter_type=ft,
            )
            name = result.display_name
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_load_sweep_no_niters(self):
        rng = np.random.default_rng(0)
        result = SweepResult(
            h=rng.uniform(-0.5, 0.5, size=(2, 2)),
            h_std=np.zeros((2, 2)),
            num_taps_grid=np.array([3, 4]),
            cutoff_grid=np.array([0.2, 0.8]),
            window="hamming",
            filter_type="lowpass",
        )
        with TemporaryDirectory() as td:
            path = Path(td) / "variables_lyapunov.npz"
            save_sweep(result, path)
            loaded = load_sweep(path)
            np.testing.assert_array_equal(loaded.h, result.h)

    def test_bandstop_odd_filter_raises(self):
        with self.assertRaises(ValueError):
            precompute_fir_bank(
                window="hann",
                filter_type="bandstop",
                num_taps_grid=np.arange(2, 5),
                cutoff_grid=np.array([0.2, 0.8]),
            )

    def test_legacy_keys_are_accepted(self):
        """A checkpoint written with the pre-0.9 keys ('coef'/'wcorte') loads.

        ``load_sweep`` must keep reading files produced by older versions,
        including the frozen SIMAC state ``f517e0a``.
        """
        from chaotic_pfc.errors import UntrustedDataError

        with TemporaryDirectory() as td:
            path = Path(td) / "legacy.npz"
            np.savez(
                path,
                h=np.zeros((2, 2)),
                h_desvio=np.zeros((2, 2)),
                coef=np.array([3, 5]),
                wcorte=np.array([0.25, 0.5]),
                window=np.array("hamming"),
                filter_type=np.array("lowpass"),
                metadata=np.array([("Nmap", 300)], dtype=object),
            )
            with self.assertRaises(UntrustedDataError):
                load_sweep(path)
            with self.assertWarns(UserWarning):
                loaded = load_sweep(path, trust_legacy_pickle=True)
            np.testing.assert_array_equal(loaded.num_taps_grid, [3, 5])
            np.testing.assert_array_equal(loaded.cutoff_grid, [0.25, 0.5])
            self.assertEqual(loaded.window, "hamming")
            self.assertEqual(loaded.filter_type, "lowpass")
            self.assertEqual(loaded.metadata["Nmap"], 300)

    def test_new_save_uses_json_string_without_pickle(self):
        result = SweepResult(
            h=np.array([[0.1, -0.2]]),
            h_std=np.array([[0.01, 0.02]]),
            num_taps_grid=np.array([3]),
            cutoff_grid=np.array([0.2, 0.8]),
            window="hamming",
            filter_type="lowpass",
            metadata={"Nitera": 500, "Nmap": 3000, "kaiser_beta": 5.0},
        )
        with TemporaryDirectory() as td:
            path = Path(td) / "safe.npz"
            save_sweep(result, path)
            with np.load(path, allow_pickle=False) as raw:
                self.assertEqual(raw["metadata"].dtype.kind, "U")
            loaded = load_sweep(path)
            self.assertEqual(loaded.metadata, result.metadata)

    def test_malicious_pickled_npz_is_refused_without_executing_payload(self):
        from chaotic_pfc.errors import UntrustedDataError

        class _Exploit:
            def __init__(self, target: Path) -> None:
                self.target = target

            def __reduce__(self):
                return (Path.write_text, (self.target, "pwned"))

        with TemporaryDirectory() as td:
            sentinel = Path(td) / "sentinel.txt"
            evil_npz = Path(td) / "Hamming (lowpass)" / "variables_lyapunov.npz"
            evil_npz.parent.mkdir(parents=True)
            np.savez(
                evil_npz,
                h=np.zeros((1, 1)),
                h_desvio=np.zeros((1, 1)),
                num_taps_grid=np.array([3]),
                cutoff_grid=np.array([0.5]),
                window=np.array("hamming"),
                filter_type=np.array("lowpass"),
                metadata=np.array([_Exploit(sentinel)], dtype=object),
            )
            with self.assertRaises(UntrustedDataError):
                load_sweep(evil_npz)
            self.assertFalse(sentinel.exists(), "Pickled payload was executed!")

            # Also verify non-metadata pickled arrays are rejected even before metadata
            evil_h = Path(td) / "evil_h.npz"
            np.savez(
                evil_h,
                h=np.array([_Exploit(sentinel)], dtype=object),
                h_desvio=np.zeros((1, 1)),
                num_taps_grid=np.array([3]),
                cutoff_grid=np.array([0.5]),
                window=np.array("hamming"),
                filter_type=np.array("lowpass"),
            )
            with self.assertRaises(UntrustedDataError):
                load_sweep(evil_h, trust_legacy_pickle=True)
            self.assertFalse(sentinel.exists(), "Pickled h payload was executed!")

    def test_all_124_versioned_legacy_checkpoints_load(self):
        sweeps_dir = Path(__file__).resolve().parents[1] / "data" / "sweeps"
        paths = sorted(sweeps_dir.rglob("variables_lyapunov.npz"))
        self.assertEqual(len(paths), 124)
        for p in paths:
            sr = load_sweep(p)
            self.assertIn("Nmap", sr.metadata)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel function tests (Numba-JIT helpers)
# ═══════════════════════════════════════════════════════════════════════════


def _manifest_npz_digests() -> set[str]:
    repo = Path(__file__).resolve().parents[1]
    digests: set[str] = set()
    for manifest in (
        repo / "data" / "sweeps" / "SHA256SUMS",
        repo / "articles" / "simac" / "SHA256SUMS",
    ):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[1].endswith(".npz"):
                digests.add(parts[0].lower())
    return digests


def test_trusted_digest_set_matches_manifests() -> None:
    """The compiled trust anchor must equal the versioned .npz digests."""
    from chaotic_pfc.analysis.sweep._trusted_checkpoints import (
        TRUSTED_CHECKPOINT_SHA256,
    )

    assert set(TRUSTED_CHECKPOINT_SHA256) == _manifest_npz_digests()


class _SentinelPayload:
    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self):
        return (Path.write_text, (self.target, "pwned"))


def test_manifest_in_cwd_does_not_enable_pickle(tmp_path: Path, monkeypatch) -> None:
    """A forged SHA256SUMS next to the data must not authorise the payload.

    Regression test for the trust-bypass defect: the manifest used to be read
    from the current working directory, so an attacker could ship the
    checkpoint and a manifest listing its own hash.
    """
    from chaotic_pfc.errors import UntrustedDataError

    sentinel = tmp_path / "sentinel.txt"
    evil = tmp_path / "Hamming (lowpass)" / "variables_lyapunov.npz"
    evil.parent.mkdir(parents=True)
    np.savez(
        evil,
        h=np.zeros((1, 1)),
        h_desvio=np.zeros((1, 1)),
        num_taps_grid=np.array([3]),
        cutoff_grid=np.array([0.5]),
        window=np.array("hamming"),
        filter_type=np.array("lowpass"),
        metadata=np.array([_SentinelPayload(sentinel)], dtype=object),
    )

    forged = tmp_path / "data" / "sweeps" / "SHA256SUMS"
    forged.parent.mkdir(parents=True)
    digest = hashlib.sha256(evil.read_bytes()).hexdigest()
    forged.write_text(f"{digest}  Hamming (lowpass)/variables_lyapunov.npz\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(UntrustedDataError):
        load_sweep(evil)
    assert not sentinel.exists(), "pickled payload was executed via a forged manifest"

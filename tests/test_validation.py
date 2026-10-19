"""Tests for chaotic_pfc.errors and chaotic_pfc._validate boundary checks."""

from __future__ import annotations

import unittest

import numpy as np

from chaotic_pfc import (
    ChaoticPFCError,
    DivergenceError,
    DivergenceWarning,
    InvalidParameterError,
    UntrustedDataError,
    awgn,
    binary_message,
    channel_impulsive,
    channel_multipath,
    classify,
    fir_channel,
    fixed_point_stability,
    henon_filtered,
    henon_generalised,
    henon_order_n,
    henon_standard,
    ideal_channel,
    lyapunov_henon2d,
    lyapunov_henon2d_ensemble,
    lyapunov_max,
    lyapunov_max_ensemble,
    lyapunov_max_pole_filtered,
    lyapunov_spectrum,
    psd_normalised,
    receive,
    receive_order_n,
    run_sweep,
    sinusoidal_message,
    text_message,
    transmit,
    transmit_order_n,
)
from chaotic_pfc._validate import (
    require_1d_nonempty,
    require_finite,
    require_in_range,
    require_positive_int,
    require_probability,
)


class TestErrorHierarchy(unittest.TestCase):
    def test_inheritance(self):
        self.assertTrue(issubclass(InvalidParameterError, ChaoticPFCError))
        self.assertTrue(issubclass(InvalidParameterError, ValueError))
        self.assertTrue(issubclass(UntrustedDataError, ChaoticPFCError))
        self.assertTrue(issubclass(UntrustedDataError, ValueError))
        self.assertTrue(issubclass(DivergenceError, ChaoticPFCError))
        self.assertTrue(issubclass(DivergenceWarning, UserWarning))


class TestValidateHelpers(unittest.TestCase):
    def test_require_finite(self):
        self.assertEqual(require_finite("x", 1.5), 1.5)
        self.assertEqual(require_finite("x", np.float64(2.0)), 2.0)
        for bad in (
            float("nan"),
            float("inf"),
            float("-inf"),
            1e308,
            -1e308,
            True,
            False,
            "1.0",
            1j,
        ):
            with self.assertRaises(InvalidParameterError):
                require_finite("x", bad)

    def test_require_positive_int(self):
        self.assertEqual(require_positive_int("n", 5), 5)
        self.assertEqual(require_positive_int("n", np.int64(7)), 7)
        for bad in (0, -1, 3.7, float("nan"), float("inf"), True, False, "5"):
            with self.assertRaises(InvalidParameterError):
                require_positive_int("n", bad)

    def test_require_in_range_and_probability(self):
        self.assertEqual(require_in_range("x", 0.5, 0.0, 1.0, closed="neither"), 0.5)
        with self.assertRaises(InvalidParameterError):
            require_in_range("x", 0.0, 0.0, 1.0, closed="neither")
        with self.assertRaises(InvalidParameterError):
            require_in_range("x", 1.0, 0.0, 1.0, closed="left")
        self.assertEqual(require_probability("p", 0.0), 0.0)
        self.assertEqual(require_probability("p", 1.0), 1.0)
        for bad in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(InvalidParameterError):
                require_probability("p", bad)

    def test_require_1d_nonempty(self):
        arr = require_1d_nonempty("a", [1.0, 2.0])
        self.assertEqual(arr.shape, (2,))
        for bad in (
            [],
            [[1.0, 2.0]],
            [1.0, float("nan")],
            [1.0, float("inf")],
            "abc",
            [True, False],
        ):
            with self.assertRaises(InvalidParameterError):
                require_1d_nonempty("a", bad)


class TestBoundaryValidation(unittest.TestCase):
    def test_channels_reject_invalid_parameters(self):
        sig = np.ones(50)
        for bad_snr in (float("nan"), float("inf"), float("-inf"), 1e308, -1e308):
            with self.assertRaises(InvalidParameterError):
                awgn(sig, bad_snr)
            with self.assertRaises(InvalidParameterError):
                channel_impulsive(sig, bad_snr)
            with self.assertRaises(InvalidParameterError):
                channel_multipath(sig, bad_snr)
        for bad_prob in (-0.1, 1.5, float("nan"), float("inf")):
            with self.assertRaises(InvalidParameterError):
                channel_impulsive(sig, 10.0, prob_impulse=bad_prob)
        for bad_amp in (-1.0, float("nan"), float("inf"), 1e308):
            with self.assertRaises(InvalidParameterError):
                channel_impulsive(sig, 10.0, amp_factor=bad_amp)
        for bad_cutoff in (0.0, 1.0, -0.5, float("nan")):
            with self.assertRaises(InvalidParameterError):
                fir_channel(sig, cutoff=bad_cutoff)

    def test_lyapunov_rejects_overflow_and_invalid_parameters(self):
        for fn in (
            fixed_point_stability,
            lyapunov_henon2d,
            lyapunov_henon2d_ensemble,
            lyapunov_max_pole_filtered,
            lyapunov_max_ensemble,
        ):
            for bad in (float("nan"), float("inf"), 1e308, -1e308):
                with self.assertRaises(InvalidParameterError):
                    fn(alpha=bad)
                with self.assertRaises(InvalidParameterError):
                    fn(beta=bad)
        for fn4d in (fixed_point_stability, lyapunov_max_pole_filtered, lyapunov_max_ensemble):
            for bad_r in (-0.1, 1.0, 1.5, float("nan"), float("inf")):
                with self.assertRaises(InvalidParameterError):
                    fn4d(pole_radius=bad_r)
        for fn_ens in (lyapunov_henon2d_ensemble, lyapunov_max_ensemble):
            for bad_p in (0.0, -0.1, float("nan"), float("inf"), 1e308):
                with self.assertRaises(InvalidParameterError):
                    fn_ens(perturbation=bad_p)

        # Generic lyapunov_spectrum / lyapunov_max jacobian shape check
        with self.assertRaises(InvalidParameterError):
            lyapunov_spectrum(
                iterate=lambda x: x,
                jacobian=lambda x: np.ones((3, 3)),
                x0=np.array([0.1, 0.2]),
            )
        with self.assertRaises(InvalidParameterError):
            lyapunov_max(
                iterate=lambda x: x,
                jacobian=lambda x: np.ones((2, 3)),
                x0=np.array([0.1, 0.2]),
            )

    def test_spectral_and_signals_reject_invalid_parameters(self):
        sig = np.sin(2 * np.pi * 0.1 * np.arange(1024))
        for bad_fs in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(InvalidParameterError):
                psd_normalised(sig, fs=bad_fs)
        for bad_n in (0, -5, 3.5, True):
            with self.assertRaises(InvalidParameterError):
                binary_message(bad_n)
            with self.assertRaises(InvalidParameterError):
                sinusoidal_message(bad_n)
            with self.assertRaises(InvalidParameterError):
                text_message("hi", bad_n)
        with self.assertRaises(InvalidParameterError):
            text_message("hi", 20, bit_period=2.5)

    def test_run_sweep_rejects_invalid_grids_and_counts(self):
        with self.assertRaises(InvalidParameterError):
            run_sweep(num_taps_grid=[], cutoff_grid=np.array([0.3]), warmup=False)
        with self.assertRaises(InvalidParameterError):
            run_sweep(num_taps_grid=[3], cutoff_grid=np.array([]), warmup=False)
        with self.assertRaises(InvalidParameterError):
            run_sweep(num_taps_grid=[3], cutoff_grid=np.array([0.3, float("nan")]), warmup=False)
        with self.assertRaises(InvalidParameterError):
            run_sweep(num_taps_grid=[3], cutoff_grid=np.array([0.3]), n_initial=0, warmup=False)
        with self.assertRaises(InvalidParameterError):
            run_sweep(num_taps_grid=[3.7], cutoff_grid=np.array([0.3]), warmup=False)
        with self.assertRaises(InvalidParameterError):
            run_sweep(num_taps_grid=[3], cutoff_grid=np.array([0.3]), Nitera=-1, warmup=False)

    def test_maps_and_comms_reject_invalid_inputs(self):
        for map_fn in (henon_standard, henon_generalised, henon_filtered):
            with self.assertRaises(InvalidParameterError):
                map_fn(0)
            with self.assertRaises(InvalidParameterError):
                map_fn(10, alpha=float("nan"))
        with self.assertRaises(InvalidParameterError):
            henon_order_n(0, fir_coeffs=np.array([0.5, 0.3, 0.2]))
        with self.assertRaises(InvalidParameterError):
            transmit(np.array([]))
        with self.assertRaises(InvalidParameterError):
            transmit(np.ones((2, 2)))
        with self.assertRaises(InvalidParameterError):
            transmit(np.array([1.0, float("nan")]))
        with self.assertRaises(InvalidParameterError):
            receive(np.array([1.0, float("inf")]))
        with self.assertRaises(InvalidParameterError):
            transmit_order_n(np.ones(10), np.array([0.5, 0.3, 0.2]), mu=float("nan"))
        with self.assertRaises(InvalidParameterError):
            receive_order_n(np.ones(10), np.array([0.5, 0.3, 0.2]), alpha=float("inf"))
        with self.assertRaises(InvalidParameterError):
            classify(np.array([0.1, float("inf")]))
        with self.assertRaises(InvalidParameterError):
            ideal_channel(np.array([1.0, float("nan")]))


class TestOnDivergencePolicy(unittest.TestCase):
    def test_invalid_policy_raises(self):
        with self.assertRaises(InvalidParameterError):
            henon_standard(10, on_divergence="invalid")  # type: ignore[arg-type]

    def test_maps_on_divergence(self):
        for fn in (henon_standard, henon_generalised, henon_filtered):
            X, _ = fn(50, alpha=10.0, on_divergence="nan")
            self.assertTrue(np.any(np.isnan(X)))
            with self.assertWarns(DivergenceWarning):
                fn(50, alpha=10.0, on_divergence="warn")
            with self.assertRaises(DivergenceError):
                fn(50, alpha=10.0, on_divergence="raise")

        c = np.array([5.0, 5.0, 5.0])
        state, _ = henon_order_n(50, fir_coeffs=c, alpha=10.0, seed=0, on_divergence="nan")
        self.assertTrue(np.any(~np.isfinite(state)))
        with self.assertWarns(DivergenceWarning):
            henon_order_n(50, fir_coeffs=c, alpha=10.0, seed=0, on_divergence="warn")
        with self.assertRaises(DivergenceError):
            henon_order_n(50, fir_coeffs=c, alpha=10.0, seed=0, on_divergence="raise")

    def test_lyapunov_on_divergence(self):
        res = lyapunov_henon2d(alpha=10.0, Nitera=50, Ndiscard=20, on_divergence="nan")
        self.assertTrue(np.isnan(res.lyapunov_max))
        with self.assertWarns(DivergenceWarning):
            lyapunov_henon2d(alpha=10.0, Nitera=50, Ndiscard=20, on_divergence="warn")
        with self.assertRaises(DivergenceError):
            lyapunov_henon2d(alpha=10.0, Nitera=50, Ndiscard=20, on_divergence="raise")

        with self.assertRaises(DivergenceError):
            lyapunov_max_pole_filtered(alpha=10.0, Nitera=50, Ndiscard=20, on_divergence="raise")
        with self.assertRaises(DivergenceError):
            lyapunov_henon2d_ensemble(
                alpha=10.0, Nitera=50, Ndiscard=20, n_initial=2, on_divergence="raise"
            )
        with self.assertRaises(DivergenceError):
            lyapunov_max_ensemble(
                alpha=10.0, Nitera=50, Ndiscard=20, n_initial=2, on_divergence="raise"
            )
        with self.assertRaises(DivergenceError):
            lyapunov_max(
                iterate=lambda x: np.array([10.0 - x[0] ** 2, x[0]]),
                jacobian=lambda x: np.array([[-2.0 * x[0], 0.0], [1.0, 0.0]]),
                x0=np.array([10.0, 10.0]),
                Nitera=20,
                Ndiscard=10,
                on_divergence="raise",
            )

    def test_comms_on_divergence(self):
        msg = np.ones(100)
        with self.assertWarns(DivergenceWarning):
            transmit(msg, mu=10.0, on_divergence="warn")
        with self.assertRaises(DivergenceError):
            transmit(msg, mu=10.0, on_divergence="raise")

        c = np.array([5.0, 5.0, 5.0])
        with self.assertRaises(DivergenceError):
            transmit_order_n(msg, c, mu=10.0, on_divergence="raise")
        with self.assertRaises(DivergenceError):
            receive(np.full(50, 1e150), mu=0.01, beta=1e100, on_divergence="raise")
        with self.assertRaises(DivergenceError):
            receive_order_n(np.full(50, 1e150), c, mu=0.01, beta=1e100, on_divergence="raise")

    def test_run_sweep_on_divergence_only_when_all_cells_diverge(self):
        with self.assertWarns(DivergenceWarning):
            run_sweep(
                num_taps_grid=[3],
                cutoff_grid=np.array([0.5]),
                alpha=10.0,
                Nitera=20,
                Nmap=50,
                n_initial=2,
                warmup=False,
                on_divergence="warn",
            )
        with self.assertRaises(DivergenceError):
            run_sweep(
                num_taps_grid=[3],
                cutoff_grid=np.array([0.5]),
                alpha=10.0,
                Nitera=20,
                Nmap=50,
                n_initial=2,
                warmup=False,
                on_divergence="raise",
            )


if __name__ == "__main__":
    unittest.main()


class TestValidationHelperEdges(unittest.TestCase):
    """Boundary and message details of the validators (mutation-test targets)."""

    def test_require_finite_accepts_explicit_magnitude_bound(self):
        self.assertEqual(require_finite("x", 5.0, max_abs=10.0), 5.0)
        with self.assertRaisesRegex(InvalidParameterError, "safe numerical bound"):
            require_finite("x", 10.5, max_abs=10.0)
        with self.assertRaisesRegex(InvalidParameterError, "must be a real number"):
            require_finite("x", 1 + 2j)

    def test_require_positive_int_accepts_integral_float_and_rejects_text(self):
        self.assertEqual(require_positive_int("n", 5.0), 5)
        self.assertEqual(require_positive_int("n", np.float64(6.0)), 6)
        with self.assertRaises(InvalidParameterError):
            require_positive_int("n", "5")

    def test_require_in_range_closed_modes(self):
        self.assertEqual(require_in_range("x", 0.0, 0.0, 1.0, closed="left"), 0.0)
        self.assertEqual(require_in_range("x", 1.0, 0.0, 1.0, closed="right"), 1.0)
        with self.assertRaises(InvalidParameterError):
            require_in_range("x", 1.0, 0.0, 1.0, closed="left")
        with self.assertRaises(InvalidParameterError):
            require_in_range("x", 0.0, 0.0, 1.0, closed="right")
        with self.assertRaisesRegex(InvalidParameterError, "invalid closed mode"):
            require_in_range("x", 0.5, 0.0, 1.0, closed="sideways")  # type: ignore[arg-type]

    def test_require_1d_nonempty_options(self):
        arr = require_1d_nonempty("a", [np.nan, 1.0], allow_nan=True)
        self.assertTrue(np.isnan(arr[0]))
        with self.assertRaisesRegex(InvalidParameterError, "at least 3 entries"):
            require_1d_nonempty("a", [1.0, 2.0], min_length=3)
        with self.assertRaises(InvalidParameterError):
            require_1d_nonempty("a", np.array([1 + 1j]))
        with self.assertRaises(InvalidParameterError):
            require_1d_nonempty("a", np.array([object()], dtype=object))

    def test_require_int_grid_and_seed(self):
        from chaotic_pfc._validate import require_int_grid, require_seed

        np.testing.assert_array_equal(require_int_grid("g", [3, 5]), np.array([3, 5]))
        with self.assertRaises(InvalidParameterError):
            require_int_grid("g", [])
        with self.assertRaises(InvalidParameterError):
            require_int_grid("g", [3.5])
        with self.assertRaises(InvalidParameterError):
            require_int_grid("g", [0])
        self.assertIsNone(require_seed("seed", None))
        self.assertEqual(require_seed("seed", 0), 0)
        with self.assertRaises(InvalidParameterError):
            require_seed("seed", -1)

    def test_divergence_policy_helpers(self):
        from chaotic_pfc._validate import handle_divergence, require_divergence_policy

        self.assertEqual(require_divergence_policy("nan"), "nan")
        self.assertEqual(require_divergence_policy("warn"), "warn")
        self.assertEqual(require_divergence_policy("raise"), "raise")
        with self.assertRaisesRegex(InvalidParameterError, "on_divergence must be"):
            require_divergence_policy("ignore")
        handle_divergence("nan", "no effect")
        with self.assertWarns(DivergenceWarning):
            handle_divergence("warn", "message")
        with self.assertRaisesRegex(DivergenceError, "message"):
            handle_divergence("raise", "message")

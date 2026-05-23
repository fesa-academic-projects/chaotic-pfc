"""tests/test_scientific_validation.py — Lyapunov exponents validated against published literature.

Reference values
----------------
- Wolf et al. 1985, Physica D 16, 285-317: λ₁ ≈ 0.418, λ₂ ≈ −1.622
  Used 10 000 iterations with Gram-Schmidt reorthonormalisation.
- Sprott 2003, "Chaos and Time-Series Analysis", Oxford Univ. Press,
  Section 5.2: λ₁ ≈ 0.419, λ₂ ≈ −1.623.
- Analytical identity: for the Hénon map det(J) = −b (constant), so
  λ₁ + λ₂ = ln|b| = ln(0.3) ≈ −1.203 972 804…

Each test uses a fixed seed and a high iteration count (20 000) to
reach the asymptotic regime that published values assume.
"""

import unittest

import numpy as np

from chaotic_pfc.dynamics.lyapunov import lyapunov_henon2d


class TestLyapunovAgainstLiterature(unittest.TestCase):
    """Compare lyapunov_henon2d against published reference values."""

    # ── Wolf et al. 1985 ───────────────────────────────────────────────

    def test_henon_wolf_1985_lambda1(self):
        """λ₁ for Hénon (a=1.4, b=0.3) matches Wolf+1985 within 1.5%."""
        r = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=20_000, Ndiscard=1_000, seed=42)
        wolf_lambda1 = 0.418
        np.testing.assert_allclose(r.all_exponents[0], wolf_lambda1, rtol=1.5e-2)

    def test_henon_wolf_1985_lambda2(self):
        """λ₂ for Hénon (a=1.4, b=0.3) matches Wolf+1985 within 1.5%."""
        r = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=20_000, Ndiscard=1_000, seed=42)
        wolf_lambda2 = -1.622
        np.testing.assert_allclose(r.all_exponents[1], wolf_lambda2, rtol=1.5e-2)

    def test_henon_wolf_1985_lyapunov_max(self):
        """Convenience property lyapunov_max matches Wolf λ₁."""
        r = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=20_000, Ndiscard=1_000, seed=42)
        wolf_lambda1 = 0.418
        np.testing.assert_allclose(r.lyapunov_max, wolf_lambda1, rtol=1.5e-2)

    # ── Sprott 2003 (textbook) ─────────────────────────────────────────

    def test_henon_sprott_2003_lambda1(self):
        """λ₁ matches Sprott+2003 textbook value within 2%."""
        r = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=20_000, Ndiscard=1_000, seed=42)
        sprott_lambda1 = 0.419
        np.testing.assert_allclose(r.all_exponents[0], sprott_lambda1, rtol=2e-2)

    def test_henon_sprott_2003_lambda2(self):
        """λ₂ matches Sprott+2003 textbook value within 2%."""
        r = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=20_000, Ndiscard=1_000, seed=42)
        sprott_lambda2 = -1.623
        np.testing.assert_allclose(r.all_exponents[1], sprott_lambda2, rtol=2e-2)

    # ── Analytical identity ────────────────────────────────────────────

    def test_lambda_sum_equals_log_determinant_standard(self):
        """λ₁ + λ₂ = ln|b| for standard Hénon (a=1.4, b=0.3).

        Mathematical identity: sum of Lyapunov exponents equals
        ln|det J|. For the Hénon map, det J = −b (constant), so
        λ₁ + λ₂ = ln(b).  This test is independent of any published
        reference — it is a mathematical proof.
        """
        beta = 0.3
        r = lyapunov_henon2d(alpha=1.4, beta=beta, Nitera=10_000, Ndiscard=1_000, seed=42)
        computed_sum = float(np.sum(r.all_exponents))
        expected_sum = float(np.log(beta))
        np.testing.assert_allclose(computed_sum, expected_sum, rtol=1e-6)

    def test_lambda_sum_equals_log_determinant_alt_params(self):
        """λ₁ + λ₂ = ln|b| holds for non-standard but bounded parameters (b=0.2)."""
        beta = 0.2
        r = lyapunov_henon2d(
            alpha=1.4, beta=beta, Nitera=10_000, Ndiscard=1_000, seed=42
        )
        computed_sum = float(np.sum(r.all_exponents))
        expected_sum = float(np.log(beta))
        np.testing.assert_allclose(computed_sum, expected_sum, rtol=1e-6)

    def test_lambda_sum_log_det_different_alpha(self):
        """λ₁ + λ₂ = ln|b| holds independent of α (here α=1.0, b=0.5)."""
        r = lyapunov_henon2d(
            alpha=1.0, beta=0.5, Nitera=10_000, Ndiscard=1_000, seed=42
        )
        computed_sum = float(np.sum(r.all_exponents))
        expected_sum = float(np.log(0.5))
        np.testing.assert_allclose(computed_sum, expected_sum, rtol=1e-6)

    # ── Deterministic repeatability ────────────────────────────────────

    def test_fixed_seed_reproducibility(self):
        """Same seed + same parameters → identical λ_max."""
        r1 = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=1_000, Ndiscard=100, seed=42)
        r2 = lyapunov_henon2d(alpha=1.4, beta=0.3, Nitera=1_000, Ndiscard=100, seed=42)
        self.assertEqual(r1.lyapunov_max, r2.lyapunov_max)


if __name__ == "__main__":
    unittest.main()

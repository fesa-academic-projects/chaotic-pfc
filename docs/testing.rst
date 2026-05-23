.. _testing:

Property-based testing
======================

In addition to traditional example-based unit tests, ``chaotic-pfc``
uses `Hypothesis <https://hypothesis.readthedocs.io>`__ to verify
**mathematical invariants** — properties that must hold across the
entire input domain, not just for a few hand-picked examples.

What is property-based testing?
--------------------------------

Instead of writing:

.. code-block:: python

   def test_henon_a14_b03():
       X, Y = henon_standard(100, a=1.4, b=0.3)
       assert X.shape == (101,)

you write:

.. code-block:: python

   @given(a=st.floats(0.1, 2.0), b=st.floats(0.01, 0.9), n=st.integers(5, 50))
   def test_henon_output_length(self, a, b, n):
       X, Y = henon_standard(n, a=a, b=b)
       assert len(X) == n + 1

Hypothesis generates hundreds of random ``(a, b, n)`` tuples and
checks the invariant for every one. When a counterexample is found,
Hypothesis **shrinks** it to the simplest failing case and prints a
minimal reproduction.

Properties currently tested
----------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Property
   * - ``henon_standard``
     - Output is either all-finite or diverges monotonically (once NaN appears, it never recovers).
   * - ``henon_standard``
     - Output length equals ``n + 1``.
   * - ``henon_filtered``
     - With stable lowpass FIR, output has correct shape.
   * - ``lyapunov_henon2d``
     - For (a,b) very near (1.4, 0.3), if orbit stays bounded, λ_max > 0.
   * - ``lyapunov_henon2d``
     - λ₁ + λ₂ = ln(b) for any bounded orbit (analytical identity).
   * - ``binary_message``
     - Output values in {−1, +1}, shape (N,).
   * - ``sinusoidal_message``
     - Output in [−1, 1], all finite.
   * - ``area_summary``
     - n_chaotic + n_periodic + n_divergent = n_total (counting invariant).
   * - ``lmax_statistics``
     - When n_used ≥ 3, the 95% CI contains the mean.
   * - ``consolidate_kaiser``
     - At most 1 Kaiser entry per filter type after consolidation.
   * - ``consolidate_kaiser``
     - Non-Kaiser windows pass through unchanged.

Running locally vs. CI
-----------------------

Two Hypothesis profiles control the trade-off between speed and
thoroughness:

.. list-table::
   :header-rows: 1

   * - Profile
     - ``max_examples``
     - Use case
   * - ``dev``
     - 50
     - Local development (default)
   * - ``ci``
     - 500
     - CI pipeline

.. code-block:: bash

   # Local dev (fast, default)
   pytest tests/test_properties.py

   # CI thoroughness
   pytest tests/test_properties.py --hypothesis-profile=ci

   # Run with full test suite
   pytest --hypothesis-profile=ci

The CI workflow (``.github/workflows/ci.yml``) uses ``--hypothesis-profile=ci``
automatically.

Custom strategies
-----------------

Reusable input generators live in ``tests/_hypothesis_strategies.py``:

* ``safe_henon_params()`` — (a, b) in bounded-chaos regime
* ``finite_initial_conditions(dim)`` — IC vectors avoiding overflow
* ``lowpass_fir_params()`` — (N_filter, wc) for stable lowpass FIR
* ``finite_ndarrays(shape)`` — arbitrary finite arrays
* ``arrays_with_nan()`` — arrays with NaN sprinkled in
* ``small_sweep_results()`` — synthetic SweepResult-like data

Adding a new property test
---------------------------

1. Decide the invariant — what MUST be true for ALL valid inputs?
2. Write a strategy (or reuse an existing one) that generates valid inputs.
3. Decorate the test method with ``@given(...)``.
4. Use ``@settings(max_examples=N, deadline=Ms)`` to bound runtime.
5. Run Hypothesis; if it finds a counterexample, investigate whether
   it is a bug in the code or a flaw in the property formulation.

.. note::

   Property-based tests **complement** example-based tests — they do
   not replace them.  Example tests document expected behaviour for
   specific, meaningful cases; property tests guard against
   regressions in edge cases that humans would never write manually.

Performance benchmarks
----------------------

In addition to correctness tests, ``chaotic-pfc`` includes a
`pytest-benchmark <https://pytest-benchmark.readthedocs.io>`__ suite
measuring hot-path performance and comparing against a committed baseline.

.. list-table::
   :header-rows: 1

   * - Benchmark
     - Operation
     - Typical time (ms)
   * - ``test_henon_standard_1000_iters``
     - Hénon map, 1000 iterations
     - ~2.6
   * - ``test_henon_generalised_1000_iters``
     - Generalised Hénon, 1000 iterations
     - ~2.7
   * - ``test_henon_filtered_1000_iters``
     - FIR-filtered Hénon (c0=1, c1=0), 1000 iterations
     - ~3.1
   * - ``test_henon_order_n_1000_iters``
     - Order-N Hénon (Nc=4), 1000 iterations
     - ~4.3
   * - ``test_henon_standard_10000_iters``
     - Hénon map, 10 000 iterations
     - ~27
   * - ``test_lyapunov_henon2d_2000_iters``
     - Single Lyapunov, 2000 QR steps
     - ~53
   * - ``test_lyapunov_max_4d_2000_iters``
     - 4-D pole-filtered Lyapunov, 2000 QR steps
     - ~120
   * - ``test_lyapunov_ensemble_25_ics``
     - 25-IC ensemble, 500 QR steps each
     - ~330
   * - ``test_mini_sweep_30_points``
     - Quick-mode sweep, 30 grid points
     - ~2330

.. code-block:: bash

   # Run all benchmarks
   pytest benchmarks/ --benchmark-only

   # Compare against baseline (fails if any mean regresses >25%)
   pytest benchmarks/ --benchmark-only \\
     --benchmark-compare=benchmarks/baseline/baseline_v0_7_0.json \\
     --benchmark-compare-fail=mean:25%

The CI ``benchmark`` job runs only on pull requests.  The 25% threshold
accounts for hardware variance in shared GitHub runners.


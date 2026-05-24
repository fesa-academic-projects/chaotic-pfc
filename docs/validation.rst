.. _validation:

Scientific validation
=====================

The Lyapunov exponent computation in ``chaotic_pfc`` has been
numerically validated against published reference values from the
scientific literature and against an analytical mathematical identity.

Reference values
----------------

The following table summarises the Lyapunov spectrum of the generalised
Hénon map (:math:`\alpha = 1.4`, :math:`\beta = 0.3`) — equivalent to
the standard Hénon (:math:`a = 1.4`, :math:`b = 0.3`) under a linear
change of variables —, as reported by
independent sources, alongside the value computed by
:func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d` with
:math:`N_{\text{itera}} = 20\,000` and seed 42.

.. list-table::
   :header-rows: 1

   * - Source
     - :math:`\lambda_1`
     - :math:`\lambda_2`
     - :math:`\lambda_1 + \lambda_2`
   * - Wolf et al. (1985) [1]_
     - :math:`+0.418`
     - :math:`-1.622`
     - :math:`-1.204`
   * - Sprott (2003) [2]_
     - :math:`+0.419`
     - :math:`-1.623`
     - :math:`-1.204`
   * - | Analytical identity
       | :math:`\ln|\beta| = \ln(0.3)`
     - —
     - —
     - :math:`-1.203\,973`
   * - **This work** (Nitera = 20 000)
     - :math:`+0.417\,567`
     - :math:`-1.621\,539`
     - :math:`-1.203\,973`

.. list-table:: Relative error of ``chaotic-pfc`` vs. each reference
   :header-rows: 1

   * - Comparison
     - Relative error
   * - :math:`\lambda_1` vs. Wolf 1985
     - 0.10 %
   * - :math:`\lambda_1` vs. Sprott 2003
     - 0.34 %
   * - :math:`\lambda_2` vs. Wolf 1985
     - 0.03 %
   * - :math:`\lambda_2` vs. Sprott 2003
     - 0.09 %
   * - Sum vs. :math:`\ln(0.3)`
     - :math:`2.2 \times 10^{-16}` (machine precision)

Analytical identity
-------------------

For the Hénon map, the Jacobian determinant is constant:
:math:`\det J = -\beta`.  Consequently, the sum of the Lyapunov exponents
must satisfy

.. math::

   \lambda_1 + \lambda_2 = \ln|\det J| = \ln \beta.

This identity holds for **any** dissipative parameter pair
:math:`(\alpha, \beta)` for which the orbit remains bounded.  It does **not**
depend on any external reference — it is a mathematical theorem.

The test suite verifies this identity at three parameter sets:

* :math:`(\alpha = 1.4, \; \beta = 0.3)` — standard chaotic regime
* :math:`(\alpha = 1.4, \; \beta = 0.2)` — strongly dissipative
* :math:`(\alpha = 1.0, \; \beta = 0.5)` — non-chaotic (periodic)

In every case the error is below :math:`10^{-6}`.

How to run the validation tests
--------------------------------

.. code-block:: bash

   pytest tests/test_scientific_validation.py -v

All nine tests pass (≈ 4 s on a modern CPU).  The Wolf and Sprott
comparisons use :math:`N_{\text{itera}} = 20\,000`; the analytical
identity tests use :math:`10\,000`.

Known limitations
-----------------

* **Finite-time convergence.**  Published Lyapunov values correspond to
  the asymptotic limit :math:`N_{\text{itera}} \to \infty`.  Our code
  with :math:`N_{\text{itera}} = 2\,000` (the default for sweeps)
  produces :math:`\lambda_1 \approx 0.408` — a 2.4 % deviation from
  Wolf.  The error decreases to 0.10 % at :math:`20\,000` iterations.
  The default iteration count is a conscious trade-off between sweep
  throughput and per-point accuracy; the analytical identity confirms
  the implementation is correct at *any* iteration count.

* **Orbit divergence.**  Not every :math:`(a, b)` pair yields a bounded
   attractor.  The test suite uses only parameter sets known to produce
   bounded orbits in the long-time limit.

References
----------

.. [1] A. Wolf, J. B. Swift, H. L. Swinney, and J. A. Vastano,
   "Determining Lyapunov exponents from a time series,"
   *Physica D: Nonlinear Phenomena*, vol. 16, no. 3, pp. 285–317, 1985.
   DOI: `10.1016/0167-2789(85)90011-9 <https://doi.org/10.1016/0167-2789(85)90011-9>`_

.. [2] J. C. Sprott, *Chaos and Time-Series Analysis*,
   Oxford University Press, 2003, Section 5.2.
   ISBN: 978-0-19-850840-3

.. _algorithms-lyapunov:

Lyapunov exponent computation
==============================

Motivation
----------

In chaotic dynamics, the **largest Lyapunov exponent** :math:`\lambda_{\max}`
quantifies how quickly two initially nearby trajectories diverge from each
other.  If :math:`\lambda_{\max} > 0`, the system is chaotic — a tiny
perturbation in the initial condition grows exponentially and the long-term
behaviour becomes unpredictable.  If :math:`\lambda_{\max} \le 0`, the
system is periodic or stable.

For chaotic communication, :math:`\lambda_{\max}` is the single most
important diagnostic: it tells us whether the Hénon map with a given
FIR filter still preserves chaos, or whether the filter *killed* the
chaos (making the scheme useless for physical-layer security).

Mathematical definition
-----------------------

For a discrete-time dynamical system :math:`\mathbf{x}_{n+1} = f(\mathbf{x}_n)`,
the largest Lyapunov exponent is defined as the time-averaged exponential
divergence rate:

.. math::

   \lambda_{\max}
   = \lim_{N \to \infty} \frac{1}{N}
     \sum_{k=1}^{N} \ln \frac{\| \delta \mathbf{x}_k \|}
                               {\| \delta \mathbf{x}_{k-1} \|},

where :math:`\delta \mathbf{x}_k` is the evolved tangent vector at step
:math:`k`, obtained by applying the Jacobian :math:`J(\mathbf{x}_k)` and
re-normalising.

In the code, this is the **"Gram-Schmidt"** (more precisely, **Modified
Gram-Schmidt — MGS**) approach to Lyapunov spectra.  The MGS variant is
more stable numerically than classical Gram-Schmidt, especially when
the tangent vectors become nearly parallel after many iterations.

Wolf et al. method
------------------

Alan Wolf, Jack Swift, Harry Swinney, and John Vastano published the
canonical algorithm in 1985 [Wolf1985]_.  For a known map (as opposed
to a time series), the procedure is:

1. Choose an initial condition :math:`\mathbf{x}_0` on the attractor
   (in our case, perturbed slightly around the fixed point).
2. Initialise an orthonormal tangent basis :math:`W` (identity matrix).
3. For :math:`k = 1` to :math:`N`:

   (a) Evolve the system: :math:`\mathbf{x}_k = f(\mathbf{x}_{k-1})`.
   (b) Evolve the tangent basis: :math:`Z = J(\mathbf{x}_k) \cdot W`.
   (c) Apply Modified Gram-Schmidt to :math:`Z`, obtain orthonormal
       :math:`Q` and the column norms :math:`n_i`.
   (d) Accumulate: :math:`\lambda_i \mathrel{+}= \ln n_i`.
   (e) Replace :math:`W \leftarrow Q`.

4. The Lyapunov spectrum is :math:`\lambda_i / N`.

Pseudocode
----------

.. code-block:: none

   # Input:
   #   x     : initial condition (dimension D)
   #   f     : map iteration
   #   J     : Jacobian of f
   #   Nitera: number of QR steps
   # Output: Lyapunov spectrum (D-vector)

   W = identity(D, D)          # orthonormal basis
   λ = zeros(D)                # accumulator

   for k = 1 to Nitera:
       x = f(x)                # evolve the orbit
       Z = J(x) @ W            # evolve the basis
       for i = 1 to D:         # Modified Gram-Schmidt
           v_i = Z[:, i]
           for j = 1 to i-1:
               v_i = v_i - dot(Q[:, j], v_i) * Q[:, j]
           n = norm(v_i)
           λ[i] += log(max(n, 1e-300))
           Q[:, i] = v_i / max(n, 1e-300)
       W = Q

   return λ / Nitera

Adaptations in ``chaotic-pfc``
------------------------------

**Modified Gram-Schmidt, not classical.**  Classical Gram-Schmidt projects
each vector against *all* preceding vectors at once; MGS projects against
one at a time and is more resistant to round-off error.  This is critical
when the orbit visits regions where the Jacobian is ill-conditioned.

**Underflow guard.**  The ``log(max(norm, 1e-300))`` clamp prevents
``log(0)`` (which would produce NaN) while being indistinguishable from
zero for any physically meaningful norm.  The constant :math:`10^{-300}`
was chosen well below machine underflow for ``float64``
(:math:`\approx 10^{-308}` in subnormal range).

**Ensemble protocol.**  A single initial condition samples one trajectory.
To assess the **statistical robustness** of the Lyapunov estimate,
:func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d_ensemble` draws
:math:`N_{\text{CI}}` independent initial conditions from a uniform box
around the positive fixed point and computes the spectrum for each.  The
output (:class:`~chaotic_pfc.dynamics.lyapunov.EnsembleResult`) reports
the mean, maximum, and per-IC values, plus a count of how many ICs yield
positive :math:`\lambda_{\max}` (chaotic) versus non-positive (periodic).

This is especially important when the attractor has multiple basins or
when the filter parameters push the system close to the chaos boundary.

Validation
----------

The implementation has been validated against published reference values
from Wolf et al. (1985).  For the standard Hénon map
:math:`(a = 1.4, b = 0.3)`, our code with :math:`N_{\text{itera}} = 20\,000`
yields :math:`\lambda_1 = 0.417567` (Wolf reports :math:`0.418`, error
:math:`0.10\%`).  See :doc:`/validation` for the full validation report.

A mathematical identity — :math:`\lambda_1 + \lambda_2 = \ln|b|` — is also
verified routinely in the test suite (:ref:`validation`), confirming
correctness at machine precision independently of any published reference.

Known limitations
-----------------

- **Finite-time convergence.**  Published values correspond to
  :math:`N \to \infty`.  With the sweep default of :math:`N = 500`, the
  estimate can deviate by ~2–3% from the asymptotic value.  The sum
  identity :math:`\lambda_1 + \lambda_2 = \ln|b|` remains exact at any
  :math:`N`, confirming that the bias is a convergence-rate issue, not a
  bug.

- **Orbit divergence.**  Some parameter combinations cause the orbit to
  diverge to infinity.  In those cases the QR loop produces NaN (instead
  of the historical sentinel :math:`-1 \times 10^{30}`).  NaN is the
  correct mathematical answer — the exponent is undefined when the
  attractor does not exist.

- **QR cost.**  The :math:`O(D^3)` QR step is the bottleneck.  For the
  4‑D filtered system, each iteration is ~10× more expensive than for
  the 2‑D standard map.

References
----------

.. [Wolf1985] A. Wolf, J. B. Swift, H. L. Swinney, and J. A. Vastano,
   "Determining Lyapunov exponents from a time series,"
   *Physica D: Nonlinear Phenomena*, vol. 16, no. 3, pp. 285–317, 1985.
   DOI: `10.1016/0167-2789(85)90011-9 <https://doi.org/10.1016/0167-2789(85)90011-9>`_

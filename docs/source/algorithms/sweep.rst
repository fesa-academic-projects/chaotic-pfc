.. _algorithms-sweep:

Parameter sweep architecture
============================

Motivation
----------

The Hénon map has two parameters :math:`(a, b)`; the FIR-filtered version
adds the filter order :math:`N_z` and the normalised cutoff frequency
:math:`\omega_c`.  The central question of the project is: **for which
:math:`(N_z, \omega_c)` combinations does the map remain chaotic?**

A single Lyapunov computation answers this for one grid point.  The
**parameter sweep** answers it for the entire 2‑D grid, producing the
classification maps that form the empirical backbone of the PFC.

Grid structure
--------------

The sweep runs over:

* **Orders** :math:`N_z = N_s - 1`, where :math:`N_s` is the number of FIR
  filter coefficients (taps).  Lowpass and bandpass filters sweep
  :math:`N_s \in \{2, \dots, 41\}`, i.e. :math:`N_z \in \{1, \dots, 40\}`.
  Highpass and bandstop filters require an odd number of taps, so they sweep
  :math:`N_z \in \{2, 4, \dots, 40\}` only.
* **Cutoffs** :math:`\omega_c \in (0, 1)` — normalised cutoff frequency.

For each grid point :math:`(N_z, \omega_c)`:

1. A **FIR coefficient bank** is precomputed once per window type.
2. A **Lyapunov ensemble** of :math:`N_{\text{CI}}` initial conditions
   estimates :math:`\lambda_{\max}`.
3. The mean :math:`\lambda_{\max}` (and its standard deviation) is
   stored in the output :class:`~chaotic_pfc.analysis.sweep.SweepResult`.

The sweep uses a **hybrid estimator**: the default single-vector path
(described below) runs at :math:`O(N_s)` per iteration; cells whose
:math:`\lambda_{\max}` is close to the classification threshold
:math:`\theta = 5 \times 10^{-3}` (:math:`|\lambda - \theta| < 5 \times 10^{-3}`)
are transparently recomputed with the full spectrum at :math:`O(N_s^2)` per
iteration.  This eliminates classification flips near the chaotic/periodic
transition while keeping the common-case cost low.

The total cost therefore depends on the fraction :math:`f` of marginal cells:

.. math::

   \text{cost} \approx N_{\text{orders}} \times N_{\text{cutoffs}}
   \times N_{\text{CI}} \times N_{\text{itera}}
   \times \left[ O(N_s) + f \cdot O(N_s^2) \right]

Parallel architecture
---------------------

The sweep is a data-parallel problem parallelised with Numba:

**Numba prange with load-balanced task ordering.**

The entire sweep runs inside a single ``numba.prange`` loop
(``_sweep_kernel``).
Each thread receives a contiguous block of the iteration space
following Numba's static schedule.  To prevent load imbalance (grid
points with larger-order filters are proportionally more expensive),
the tasks are ordered by estimated cost before being interleaved
across threads by ``_build_task_order``.
This keeps every thread busy for approximately the same wall time
without any inter-process communication overhead.

**Why pre‑generated perturbations?**

Numba's ``prange`` cannot use NumPy's global random state (it would be
a data race).  Instead, the Python side pre‑generates all initial‑condition
perturbations with :func:`numpy.random.default_rng` seeded by the
user‑supplied seed.  The deterministic perturbation array is passed to the
JIT kernel as a read‑only input.

Adaptive early-stop
-------------------

Computing :math:`\lambda_{\max}` with :math:`N_{\text{itera}} = 3000`
iterations gives a precise estimate; doing that for every grid point
is slow.  **Adaptive early‑stop** monitors the running estimate of
:math:`\lambda_{\max}` at checkpoints (every :math:`K = 100` iterations)
and exits early when the estimate stabilises.

The convergence criterion:

.. math::

   \left| \bar{\lambda}_k - \bar{\lambda}_{k-K} \right|
   < \varepsilon \quad \text{for } M \text{ consecutive checkpoints}

where :math:`\bar{\lambda}_k` is the running mean at checkpoint :math:`k`.
Default values: :math:`K = 100`, :math:`M = 2`, :math:`\varepsilon = 10^{-3}`.

Across the 124 versioned sweeps the effective accumulation horizon lands in a
narrow band: the median number of iterations per configuration is between about
**700 and 1200**, with a floor of 700 (the minimum before the criterion may
fire), and the cap of 3000 is reached at only 2 of the 372000 grid points.

In practice this provides a 3–4× speedup on typical sweeps with negligible
accuracy loss (the exponent estimate is already converged long before
:math:`N_{\text{itera}} = 3000` for stable orbits).

Benettin block reorthonormalisation
------------------------------------

The Modified Gram–Schmidt (MGS) reorthonormalisation at every iteration
accounts for ~82% of the kernel's CPU time.  Following Benettin et al.
(Meccanica **15**:9–20, 1980), the tangent vectors are propagated without
MGS for :math:`K = 10` consecutive map iterations and reorthonormalised
once per block.  The QR factorisation of a :math:`K`-step product is
mathematically equivalent to the product of the per-iteration factors
for the Lyapunov spectrum (the log of the determinant adds correctly);
individual exponents differ by at most :math:`\sim 10^{-2}` in
high-variance cells and :math:`\lesssim 10^{-3}` in converged cells —
well within the sampling noise of the ensemble estimate.

Safety margins for :math:`K = 10`:

* Norm growth per block: :math:`\sim e^{\lambda_1 K} \approx 55` — no
  overflow risk in IEEE float64.
* Inter-vector collapse: :math:`\sim e^{(\lambda_1 - \lambda_2) K} \approx
  5 \times 10^{8}` for a typical gap of 2 — the MGS resolves this without
  loss of orthogonality.

The block period divides the adaptive checkpoint interval
(:math:`K \mid K_{\text{checkpoint}} = 100`), so every convergence
checkpoint falls on an orthonormalisation tick.  A final partial-block
MGS runs when :math:`N_{\text{itera}}` is not a multiple of :math:`K`
(the warmup divergence early-exit bypasses the partial block because the
Lyapunov sum is already NaN).

Combined with the deferred tangent-vector scan from the previous
section, the block scheme reduces the kernel runtime by 3–6× depending on
the fraction of divergent grid points (which short-circuit the Lyapunov
loop early).  On a full bandstop sweep the measured speedup is 5.9×.

Single-vector estimator
-----------------------

The first column of the tangent matrix is **independent** of all other
columns in the MGS factorisation.  Under exact arithmetic, column 0 can
be evolved and normalised in isolation, and its growth rate is exactly
the largest Lyapunov exponent.  The per-iteration cost drops from
:math:`O(N_s^2)` (full-spectrum J@W + MGS) to :math:`O(N_s)`.

The :math:`_v` kernel variants (``_lyap_online_core_v``,
``_propagate_v_n12``, ``_propagate_v_nN``) implement exactly this
— they evolve only the first tangent vector and accumulate
:math:`\log \|\mathbf{v}\|` once per Benettin block.  On a full bandstop
sweep the measured speedup vs. the full-spectrum kernel is **3.6×**.

Hybrid fallback for marginal cells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finite-block MGS (:math:`K=10`) can cause adjacent exponents to cross
in 2.6–19.4 % of finite cells (measured across 124 sweeps):
:math:`\text{lyap\_sum}[k] > \text{lyap\_sum}[0]` for some :math:`k > 0`.
When this happens, the single-vector estimator (which always picks column
0) differs from the true :math:`\lambda_{\max}` by at most
:math:`3.5 \times 10^{-3}` — never enough to flip the sign of an
individual IC.

To eliminate the tiny probability of a classification flip near the
transition boundary, the sweep kernel applies a **hybrid strategy**:

1. Run the fast :math:`_v` path for all ICs.
2. If the cell's mean :math:`|\lambda - \theta| < 5 \times 10^{-3}` (the
   ``_VONLY_MARGIN`` half-width around ``LAMBDA_CHAOS_THRESHOLD``,
   :math:`\theta = 5 \times 10^{-3}`),
   re-run *all* ICs for that cell with the full-spectrum kernel.
3. The decision is atomic per cell: no cell mixes estimates from
   different estimators.

The margin (:math:`5 \times 10^{-3}`) is conservative: the largest
measured :math:`|\Delta \lambda|` across 124 sweeps is
:math:`3.5 \times 10^{-3}`, so any cell accepted by the :math:`_v` path
(:math:`|\lambda| \geq 5 \times 10^{-3}`) cannot change classification
sign after the correction.  The fallback fraction depends strongly on
the filter type: highpass, bandpass, and bandstop sweeps typically see
:math:`\sim 0`–:math:`12\%` of finite cells falling back (most cells
have well-separated Lyapunov exponents); lowpass sweeps, where a large
fraction of periodic cells have :math:`|\lambda|` below the margin,
see :math:`20\%`–:math:`73\%`.

The wall-clock cost of the recomputation was measured on the four baseline
configurations (Kaiser, the best :math:`\beta_K` per filter type, the full
:math:`40 \times 100` grid, 25 initial conditions, adaptive early-stop) by
timing ``fallback_mode=1`` against ``fallback_mode=0``: the full-spectrum
recomputation takes about **23 % of the total sweep time** — 23 % (lowpass),
20 % (highpass), 27 % (bandpass) and ~0 % (bandstop). The fallback is a
material but not dominant cost; it is kept because it removes the
classification-flip risk, and any future optimization must preserve the
bit-exact reproduction described above. A dedicated optimization attempt
(measured on the Kaiser lowpass β_K=13.5 sweep: reusing the post-burn-in state
bounds the saving at the burn-in share, ~5.6 % of the total) did not reach the
10 % adoption threshold and was dropped; see the project's issue tracker.

The fallback changes very few classes. On the **full grid** of the four
Table II configurations, comparing ``fallback_mode`` 0, 1 and 2 changes the class
of **2 cells** for the highpass configuration and **2** for the bandpass one (all
periodic to chaotic); on the reduced grid used by
``tests/test_fallback_class_changes.py`` no cell changes class. The direction is
the expected one: the full-spectrum recomputation can move a marginal cell
across the threshold, never by more than the measured finite-horizon gap.

Versioned checkpoint policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``.npz`` checkpoints committed to the repository were regenerated
with the current kernel (the cancellation-free positive fixed point and
the Benettin-K estimator) and are canonical: the regression test
``tests/test_sweep_regressions.py::test_mode1_bitexact_baseline``
reproduces them **bit for bit** with ``fallback_mode=1``.  A change that
breaks this equality is a regression, and the checkpoints must then be
regenerated deliberately with ``chaotic-pfc sweeps regenerate``.

Checkpoint format
^^^^^^^^^^^^^^^^^

Each ``.npz`` stores ``h`` (the :math:`\lambda_{\max}` grid), ``h_desvio`` (its
standard deviation), ``num_taps_grid`` and ``cutoff_grid`` (the swept axes),
``window``, ``filter_type``, ``metadata`` (including ``bandwidth`` for newly
generated sweeps) and, for adaptive sweeps, ``n_iters_used``.
:func:`~chaotic_pfc.analysis.sweep.load_sweep` also accepts the pre-0.9 key
names ``coef`` and ``wcorte`` (still used by the frozen SIMAC state
``f517e0a``), so those files keep loading.

.. _backend-dependence:

Backend dependence (Numba versus pure Python)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bit-for-bit reproduction above requires Numba.  Without it
(``NUMBA_DISABLE_JIT=1``, or the ``chaotic_pfc._compat`` fallback) the
same sweep gives statistically equivalent, but not identical, exponents,
because the compiled tangent recurrence rounds slightly differently from the
interpreted one.  On a representative sweep (Kaiser lowpass,
:math:`\beta_K = 13.5`, :math:`N_s \in \{2,\dots,11\}`, the 100 cutoffs,
25 initial conditions, seed 42 — 1000 cells):

* the number of chaotic points is the same in both backends,
  :math:`N_c = 563`;
* **no cell changes class** (chaotic / periodic / divergent);
* over the cells that are chaotic in both, :math:`|\Delta\lambda_1|` is
  :math:`1.5 \times 10^{-4}` on average and at most
  :math:`3.9 \times 10^{-3}` — below the :math:`5 \times 10^{-3}`
  marginal-cell margin;
* the raw ``h`` value is bit-identical in 38 % of the cells and differs in
  the remaining 62 %, always at the level of the small
  :math:`\Delta\lambda_1` above.

The discrepancy is not in the map: ``_henon_n12_inplace`` /
``_henon_nN_inplace``, the fixed point, ``log``, ``sqrt`` and scalar dot
products are bit-identical between the two backends.  It appears in the
tangent-propagation kernels
(``_propagate_v_nN`` and
``_propagate_nN``), whose sum-of-products recurrence
(``row2 = c1*w0 + cb02*w1 + m22*w2`` plus the ``tail`` accumulation) is
contracted/reassociated at the last bit by LLVM; the map's sensitive
dependence then amplifies that ~1 ULP seed.  Install Numba (the ``[fast]``
extra) whenever the checkpoints must be reproduced exactly.  The coefficient
bank itself comes from :func:`scipy.signal.firwin`, so the bit-for-bit
reproduction also depends on SciPy: ``data/sweeps/ENVIRONMENT.json`` records
the version used, ``tests/test_sweep_environment.py`` pins a fingerprint of the
coefficient banks, and a SciPy change that altered ``firwin`` would shift the
checkpoints.

Divergence handling (Opção B)
-----------------------------

An orbit is taken as unbounded when its state becomes non-finite (NaN or
infinity); from that iteration on every term of the exponent is
non-finite, so the simulation can be stopped.  The divergence rule has
three steps, as in the article:

1. **Estimator.** An initial condition whose state becomes non-finite is
   reported as non-finite.
2. **Burn-in.** A grid point is **divergent** when at least one of its
   ``n_initial`` initial conditions becomes non-finite during the
   ``Nitera`` burn-in iterations.
3. **After the burn-in.** An initial condition that becomes non-finite
   only inside the ``Nmap`` estimation window is **discarded**, and the
   value of the grid point is the mean of :math:`\lambda_1` over the
   remaining initial conditions.  A point whose filter coefficients are
   non-finite is also divergent.

The original implementation used a sentinel value
:math:`-1 \times 10^{30}` to mark those points; **Opção B** (applied in
v0.7.0) replaces it with **NaN**, which propagates correctly through all
downstream functions:
:func:`~chaotic_pfc.analysis.stats.area_summary` excludes it from the
chaotic/periodic counts (third category: *divergent*), and
:func:`~chaotic_pfc.analysis.stats.lmax_statistics` ignores it when
computing means and confidence intervals.

Reproducibility
---------------

Every sweep is **deterministic** given the same seed:

1. The user seed (``seed`` kwarg) initialises a single
   :func:`numpy.random.default_rng` generator; there are no per‑worker
   sub‑seeds.
2. Initial‑condition perturbations are drawn on the Python side
   (before entering Numba).
3. The Numba kernel consumes those pre‑drawn perturbations and holds no
   random state of its own, only deterministic arithmetic.

Running the same sweep twice with the same seed produces byte‑identical
``.npz`` files.  This is critical for scientific reproducibility.  The CI runs a
dedicated ``baseline`` job that reproduces the 124 versioned checkpoints **bit
for bit** (``CHAOTIC_PFC_RUN_BASELINE=1 pytest tests/test_sweep_regressions.py``)
and a ``pipeline`` job that smoke-tests the orchestration with
``chaotic-pfc run all --quick-sweep``.

Known limitations
-----------------

* **Sweep duration.**  A full sweep (40 orders × 100 cutoffs × 25 ICs ×
  3000 iterations) takes hours on consumer hardware.  The quick‑mode
  parameters (~6 orders × 10 cutoffs × 3 ICs × 50 iterations) run in
  seconds and are used for CI smoke testing.

* **Memory.**  The FIR coefficient bank for 40 orders × 100 cutoffs
  requires ~320 KB — negligible.  The bottleneck is CPU, not memory.

* **Warmup compilation.**  The first call to ``_sweep_kernel`` triggers
   Numba's JIT compilation, which adds a few seconds of startup time.
   The orchestration layer runs a tiny warmup sweep (:code:`warmup=True`)
   so that the actual sweep benefits from the cached native code.

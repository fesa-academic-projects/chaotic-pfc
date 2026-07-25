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

* **Orders** :math:`N_z \in \{2, 3, \dots, 41\}` — FIR filter length.
* **Cutoffs** :math:`\omega_c \in (0, 1)` — normalised cutoff frequency.

For each grid point :math:`(N_z, \omega_c)`:

1. A **FIR coefficient bank** is precomputed once per window type.
2. A **Lyapunov ensemble** of :math:`N_{\text{CI}}` initial conditions
   estimates :math:`\lambda_{\max}`.
3. The mean :math:`\lambda_{\max}` (and its standard deviation) is
   stored in the output :class:`~chaotic_pfc.analysis.sweep.SweepResult`.

The sweep uses a **hybrid estimator**: the default single-vector path
(described below) runs at :math:`O(N_s)` per iteration; cells whose
:math:`\lambda_{\max}` is close to zero (:math:`|\lambda| < 5 \times 10^{-3}`)
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

The entire sweep runs inside a single :func:`numba.prange` loop
(:func:`~chaotic_pfc.analysis.sweep._kernel._sweep_kernel`).
Each thread receives a contiguous block of the iteration space
following Numba's static schedule.  To prevent load imbalance (grid
points with larger-order filters are proportionally more expensive),
the tasks are ordered by estimated cost before being interleaved
across threads by :func:`~chaotic_pfc.analysis.sweep._kernel._build_task_order`.
This keeps every thread busy for approximately the same wall time
without any inter-process communication overhead.

**Why pre‑generated perturbations?**

Numba's ``prange`` cannot use NumPy's global random state (it would be
a data race).  Instead, the Python side pre‑generates all initial‑condition
perturbations using a :class:`~numpy.random.SeedSequence` rooted on the
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

The :math:`_v` kernel variants (:func:`_lyap_online_core_v`,
:func:`_propagate_v_n12`, :func:`_propagate_v_nN`) implement exactly this
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
2. If the cell's mean :math:`|\lambda| < 5 \times 10^{-3}` (the
   :data:`~chaotic_pfc.analysis.sweep._types._VONLY_MARGIN` threshold),
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

Versioned checkpoint policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``.npz`` checkpoints committed to the repository were computed with
the per-iteration estimator and remain canonical — they are
**never regenerated**.  The Benettin-K kernel applies only to new
computations.  Its deviation from the per-iteration result is bounded
by the early-stop tolerance :math:`\varepsilon` except in marginal cells
where :math:`|\lambda_{\max}| < 3 \times 10^{-3}`; in that regime the
binary verdict (chaotic vs. periodic) is inherently unstable under
both estimators.

Divergence handling (Opção B)
-----------------------------

The original implementation used a sentinel value :math:`-1 \times 10^{30}`
to mark grid points where the orbit diverged.  This was problematic: the
sentinel leaked into downstream statistics, producing spurious
classifications and biasing aggregate numbers.

**Opção B** (applied in v0.7.0) replaces the sentinel with **NaN**.

* The Lyapunov kernel detects overflow/divergence and sets the output to
  ``NaN`` at that grid point.
* ``NaN`` propagates correctly through all downstream functions:
  :func:`~chaotic_pfc.analysis.stats.area_summary` excludes it from
  the chaotic/periodic counts (third category: *divergent*);
  :func:`~chaotic_pfc.analysis.stats.lmax_statistics` ignores it when
  computing means and confidence intervals.

This is both mathematically correct (the exponent is undefined when the
attractor does not exist) and numerically safer than a magic sentinel.

Reproducibility
---------------

Every sweep is **deterministic** given the same seed:

1. The user seed (``seed`` kwarg) is expanded via
   :class:`numpy.random.SeedSequence` into per‑worker sub‑seeds.
2. Initial‑condition perturbations are drawn on the Python side
   (before entering Numba).
3. The Numba kernel uses no random state , only deterministic arithmetic.

Running the same sweep twice with the same seed produces byte‑identical
``.npz`` files.  This is critical for scientific reproducibility and
for the CI pipeline (the smoke test checks exact file size equality).

Known limitations
-----------------

* **Sweep duration.**  A full sweep (40 orders × 100 cutoffs × 25 ICs ×
  3000 iterations) takes hours on consumer hardware.  The quick‑mode
  parameters (~6 orders × 10 cutoffs × 3 ICs × 50 iterations) run in
  seconds and are used for CI smoke testing.

* **Memory.**  The FIR coefficient bank for 40 orders × 100 cutoffs
  requires ~320 KB — negligible.  The bottleneck is CPU, not memory.

* **Warmup compilation.**  The first call to :func:`_sweep_kernel` triggers
   Numba's JIT compilation, which adds a few seconds of startup time.
   The orchestration layer runs a tiny warmup sweep (:code:`warmup=True`)
   so that the actual sweep benefits from the cached native code.

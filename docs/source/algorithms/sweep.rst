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

The total cost is:

.. math::

   \text{cost} \approx N_{\text{orders}} \times N_{\text{cutoffs}}
   \times N_{\text{CI}} \times N_{\text{itera}} \times O(D^3)

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



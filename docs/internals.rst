.. _internals:

Internals
=========

This page documents the internal design, algorithms, and numerical
methods used inside ``chaotic-pfc``.

Numba kernel architecture
-------------------------

The performance-critical kernels of the sweep pipeline are written in
Python with Numba JIT decorators, compiled to machine code at import
time. This section describes the key internal functions.

Hénon map iterators
~~~~~~~~~~~~~~~~~~~

Two private in-place iterator functions live in :mod:`chaotic_pfc.analysis.sweep._kernel`:

``_henon_n12_inplace(x, x_new, alpha, beta, c)``
  Optimised for 1–2 filter taps. Uses a fixed-size unrolled loop over
  the filter dimension, avoiding dynamic array indexing overhead.

``_henon_nN_inplace(x, x_new, alpha, beta, c)``
  General-purpose for :math:`N \ge 3` taps. Uses a parametric loop
  over the filter coefficients.

Both operate **in-place** on pre-allocated state buffers to avoid
garbage collection pressure during the hot loop. The state is
double-buffered (ping-pong pattern) so that each iteration writes to an
alternating buffer, eliminating copy overhead.

Modified Gram-Schmidt accumulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_mgs_accumulate(z, Ns, lyap_sum)`` performs the Modified Gram-Schmidt
(MGS) re-orthonormalisation of the perturbation vectors at each
iteration. Unlike classical Gram-Schmidt (CGS), MGS computes the
projection coefficients sequentially *after each projection
subtraction*, which significantly improves numerical stability for
near-degenerate subspaces:

1. For each vector :math:`v_i`:
   a. Project :math:`v_i` onto all remaining :math:`v_j` (:math:`j > i`).
   b. Subtract the projection *before* computing the next projection.
2. Accumulate :math:`\ln \|v_i\|` into ``lyap_sum``.

The factor-2 serialisation overhead of MGS over CGS is negligible
compared to the :math:`\mathcal{O}(N_s^3)` total cost of the inner
loop.

Adaptive early-stop mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_adaptive_checkpoint(n, Ns, lyap_sum, last_lyap, stable_count, tol,
Nmap_min, Nmap)`` is called every ``_ADAPTIVE_CHECKPOINT_EVERY = 100``
iterations. It computes the current :math:`\lambda_{\max}` estimate and
checks whether it has stabilised:

.. math::

    \left| \lambda_{\text{current}} - \lambda_{\text{previous}} \right| < \text{tol}

If this condition holds for ``_ADAPTIVE_STREAK = 2`` consecutive
checkpoints *and* ``Nmap_min`` iterations have elapsed, the kernel
signals early termination. The actual number of iterations used is
recorded in the ``SweepResult`` attribute ``n_iters_used``, available
for the difficulty-map visualisation.

Lyapunov estimator kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

``_lyap_online_n12`` and ``_lyap_online_nN`` are the inner Lyapunov
estimator kernels. They:

1. Run ``Nitera`` burn-in iterations (trajectory convergence to the
   attractor).
2. Run ``Nmap`` estimation iterations (or fewer, if adaptive
   early-stop triggers).
3. At each estimation iteration, evolve the perturbation vectors
   through the Jacobian (tangent map) and apply MGS.
4. Return the mean :math:`\lambda_{\max}` over ``n_initial``
   independent initial conditions tested in sequence.

The "online" naming reflects that the estimator updates continuously
during the trajectory, rather than collecting the full state history
and computing Lyapunov exponents offline.

Parallel sweep kernel
~~~~~~~~~~~~~~~~~~~~~

``_sweep_kernel`` is the single ``prange`` (or ``range``, without
Numba) parallel loop over all (order, cutoff) grid points. It receives
the precomputed FIR coefficient bank, perturbation tensor, and task
order permutation as inputs.

The ``_build_task_order`` function creates a permuted list of
``(order_idx, cutoff_idx)`` pairs that interleaves tasks from different
orders, ensuring balanced thread load. Without this permutation, a
naive loop over orders would assign threads to vastly unequal workloads
(the MGS inner loop scales as :math:`\mathcal{O}(N_s^3)` with filter
order :math:`N_s`).

Fixed-point stability analysis
------------------------------

:func:`~chaotic_pfc.dynamics.lyapunov.fixed_point_stability` provides a
quick eigenvalue-based sanity check before the full Lyapunov
simulation:

1. Compute the coordinates of the fixed point(s) of the filtered
   Hénon system.
2. Assemble the Jacobian matrix :math:`\mathbf{J}` of the expanded
   :math:`K'`-dimensional system at the fixed point.
3. Compute the eigenvalues of :math:`\mathbf{J}` via
   :func:`scipy.linalg.eig`.
4. Classify: if :math:`\max |\lambda_i| < 1`, the fixed point is
   linearly stable (periodic); if :math:`\max |\lambda_i| > 1`, the
   fixed point is unstable, and the full Lyapunov estimator is needed
   to determine whether the resulting orbit is chaotic or divergent.

This two-stage approach saves significant computation: points with
stable fixed points can often be classified without running the full
MGS estimator. However, points with unstable fixed points require the
full estimator because the Lyapunov exponent can be negative (periodic
orbit) or positive (chaotic) despite the instability.

FIR bank precomputation
-----------------------

The FIR coefficient bank is the largest precomputed data structure in
the sweep pipeline. For ``N_orders = 40`` orders and
``N_cutoffs = 100`` cutoffs, it contains:

* Shape ``(N_coef, N_cutoffs, max_taps)`` — each ``(order, cutoff)``
  pair produces a filter of length ``order``, zero-padded to
  ``max_taps``.
* Shape ``(N_coef, N_cutoffs)`` — the gain matrix maps each filter's
  total gain.

The bank is built by :func:`~chaotic_pfc.analysis.sweep.precompute_fir_bank`,
which calls :func:`scipy.signal.firwin` for each combination. The
coefficients are stored in **float64** (double precision) to match the
data type used throughout the Numba kernels.

Signal generators
-----------------

Two message waveform generators in :mod:`chaotic_pfc.dynamics.signals`
feed the communication pipeline:

:func:`~chaotic_pfc.dynamics.signals.binary_message`
  Produces a BPSK-style square wave of :math:`\{+1, -1\}` with a
  configurable bit period (in samples). At the default period of 20
  samples, each bit spans 20 iterations of the chaotic map.

:func:`~chaotic_pfc.dynamics.signals.sinusoidal_message`
  Produces a single-tone cosine probe for spectral-response
  measurements. Useful for characterising the channel transfer
  function independently of the chaotic carrier.

PSD estimation
--------------

:func:`~chaotic_pfc.dynamics.spectral.psd_normalised` estimates the
power spectral density (PSD) via Welch's averaged periodogram method.
Key features:

* Returns **peak-normalised** PSD (maximum value set to 1) with
  frequency axis in :math:`\omega / \pi \in [0, 1]`.
* Supports seven window types: Hamming, Hann (Hanning), Blackman,
  Kaiser (:math:`\beta=5`), Blackman-Harris, Boxcar (rectangular),
  and Bartlett (triangular).
* The ``remove_dc`` option subtracts the mean before computing the
  PSD, suppressing the DC component.
* Wraps :func:`scipy.signal.welch` with internal parameter
  normalisation (``fs=2`` maps to :math:`\omega / \pi`).

The normalised frequency convention is essential for the sweep
pipeline, where the cutoff frequency :math:`\omega_c / \pi` directly
maps to the ``cutoff`` parameter of :func:`scipy.signal.firwin`.

Channel models
--------------

Beyond the basic :func:`~chaotic_pfc.comms.channel.ideal_channel` and
:func:`~chaotic_pfc.comms.channel.fir_channel`, the DCSK module
provides four composite channel models:

:func:`~chaotic_pfc.comms.dcsk.awgn`
  Additive white Gaussian noise with configurable SNR (dB).

:func:`~chaotic_pfc.comms.dcsk.channel_impulsive`
  Middleton Class-A impulsive noise model, producing bursts of
  high-amplitude interference superimposed on AWGN.

:func:`~chaotic_pfc.comms.dcsk.channel_multipath`
  Configurable multipath propagation with user-specified tap delays
  (samples) and gain factors (linear). Models frequency-selective
  fading.

:func:`~chaotic_pfc.comms.dcsk.channel_interferers`
  Composite channel combining AWGN, a DCSK interferer (simulating
  co-channel chaotic transmissions), and a WiFi-like narrowband
  interferer.

:func:`~chaotic_pfc.comms.dcsk.channel_urban`
  All-in-one urban channel model combining AWGN, multipath,
  impulsive noise, and interferers — representing a worst-case
  composite propagation environment.

Translation layer
-----------------

:mod:`chaotic_pfc._i18n` provides bilingual figure labels (Portuguese
and English) without requiring ``gettext`` at runtime. Features:

* Simple dictionary-based lookup via :func:`chaotic_pfc._i18n.t`.
* Default language controllable via ``CHAOTIC_PFC_LANG`` environment
  variable or ``--lang`` CLI flag.
* Covers attractor titles, sensitivity axis labels, communication
  grid subplot titles, and DCSK plotting strings.
* Falls back to the raw key if a translation is missing.

Configuration model
-------------------

The :mod:`chaotic_pfc.config` module uses a shallow hierarchy of
dataclasses:

.. code-block:: text

    ExperimentConfig
    ├── comm: CommConfig
    │   └── henon: HenonConfig          # a, b
    ├── channel: ChannelConfig          # cutoff, num_taps
    ├── internal_fir: InternalFIRConfig # cutoff, num_taps, fir_coeffs()
    ├── spectral: SpectralConfig        # nfft, window_length, window, ...
    ├── lyapunov: LyapunovConfig        # Nitera, Ndiscard, perturbation, ...
    ├── plot: PlotConfig                # time_window, dpi, figures_dir, fmt
    ├── sweep: SweepConfig              # Nmap, n_initial, order range, ...
    └── seed: int = 42

Each sub-config is a ``@dataclass`` with only primitive fields (int,
float, str) or nested dataclasses with primitives. This design enables:

* ``dataclasses.replace(cfg, seed=7)`` for test isolation.
* ``dataclasses.asdict(cfg)`` for serialisation.
* Type-safe access through attribute paths (``cfg.comm.henon.a``).
* Clear ownership: every configuration field belongs to exactly one
  subsystem.

.. _architecture:

Architecture
============

This page describes how the subpackages of ``chaotic-pfc`` connect and
how data flows through the system: from chaotic signal generation to
statistical analysis of parameter sweeps.

Package layout
--------------

.. code-block:: text

    chaotic_pfc                          # top-level namespace (~90 public symbols)
    ├── _version.py                      # single source of truth: __version__
    ├── _compat.py                       # Numba compatibility layer (no-op fallback)
    ├── _i18n.py                         # bilingual figure labels (pt / en)
    ├── config.py                        # centralised dataclass configuration
    ├── dynamics/                        # core mathematical machinery
    │   ├── maps.py                      # Hénon map variants (standard → order-N)
    │   ├── lyapunov.py                  # Lyapunov exponents (single-IC + ensemble)
    │   ├── spectral.py                  # PSD estimation (Welch method)
    │   └── signals.py                   # message waveform generators
    ├── comms/                           # chaotic communication pipeline
    │   ├── protocols.py                 # structural contracts (Transmitter, Channel, Receiver)
    │   ├── transmitter.py               # chaos-based modulator (Pecora-Carroll)
    │   ├── receiver.py                  # chaos-synchronisation demodulator
    │   ├── channel.py                   # transmission models (ideal, FIR)
    │   └── dcsk.py                      # DCSK / EF-DCSK + channel impairments
    ├── analysis/                        # parameter sweeps and statistical post-processing
    │   ├── sweep/                       # Lyapunov sweep framework
    │   │   ├── _types.py                # SweepResult dataclass, filter/window catalogues
    │   │   ├── _io.py                   # .npz serialisation (save_sweep / load_sweep)
    │   │   ├── _kernel.py               # Numba-JIT kernels (MGS, prange sweep)
    │   │   └── _orchestration.py        # high-level orchestration (FIR bank, run_sweep)
    │   ├── stats.py                     # statistical analysis suite
    │   ├── sweep_plotting.py            # 2-D classification maps
    │   └── sweep_plotting_3d.py         # 3-D Plotly beta-volume (optional)
    ├── plotting/                        # publication-quality figures
    │   └── figures.py                   # attractors, SDIC overlay, 4×2 comm grids
    └── cli/                             # unified command-line interface
        ├── __init__.py                  # build_parser() + main()
        ├── _common.py                   # shared helpers (PSD, save/show, backend)
        ├── run_all.py                   # orchestrate full pipeline
        ├── attractors.py                # phase-space portraits
        ├── sensitivity.py               # SDIC visualisation
        ├── comm_ideal.py                # noiseless channel
        ├── comm_fir.py                  # FIR low-pass channel
        ├── comm_order_n.py              # order-N Hénon + FIR channel
        ├── lyapunov.py                  # Lyapunov spectra
        ├── dcsk.py                      # BER-vs-SNR comparison
        ├── sweep/                       # sweep compute / plot / beta-sweep / plot-3d
        └── analysis.py                  # statistical report

Data flow
---------

Communication pipeline
~~~~~~~~~~~~~~~~~~~~~~

The full end-to-end communication chain:

.. code-block:: text

                            +------------------+
                            |  binary_message()|  <- dynamics.signals
                            |  (BPSK: +/-1)   |
                            +--------+---------+
                                     |  m[n]
                                     v
    +-------------------+   +------------------+
    |  henon_standard() |-->|    transmit()    |
    |  (carrier)        |   |  s[n]=x1[n]      |
    +-------------------+   |       + mu*m[n]  |
                            +--------+---------+
                                     |  s[n]
                                     v
                            +------------------+
                            |    channel()     |  <- comms.channel
                            |  ideal / FIR /   |
                            |  AWGN / multipath|
                            +--------+---------+
                                     |  r[n]
                                     v
                            +------------------+
                            |    receive()     |  <- comms.receiver
                            |  recover via     |
                            |  synchronisation |
                            +--------+---------+
                                     |  m_hat[n]
                                     v
                            +------------------+
                            |       BER        |  <- comms.dcsk
                            |    (if DCSK)     |
                            +------------------+

The transmitter embeds a binary message into the chaotic carrier via
Pecora-Carroll modulation: :math:`s[n] = x_1[n] + \mu \cdot m[n]`.
The channel applies distortion (ideal pass-through, FIR band-limiting,
AWGN, multipath, or composite urban model). The receiver synchronises
to the carrier and recovers the message estimate
:math:`\hat{m}[n] \approx m[n]`.

Parameter sweep pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

The sweep framework is the computational core of the project. Its
design balances raw throughput, deterministic reproducibility, and
optional Numba dependency:

.. code-block:: text

    precompute_fir_bank()
    ├── firwin() × Norders × Ncutoffs    →  fir_bank  (Ncoef, Ncut, max_taps)
    └── gains                            →  gains     (Ncoef, Ncut)

    _build_task_order()                   →  permuted task list (load-balanced)

    _precompute_perturbations()           →  noise tensor (default_rng(seed))

    _sweep_kernel(prange)                 ← Numba JIT (or pure-Python fallback)
    ├── burn-in (Nitera iters)            →  state buffer
    ├── Lyapunov (Nmap iters, MGS)        →  λ_max per IC
    └── aggregate (n_initial ICs)         →  mean, std

    SweepResult                          →  .npz checkpoint (data/sweeps/)
    ├── stats.py                          →  comparison, ranking, correlation
    └── sweep_plotting.py                 →  classification figures (SVG + PNG)

Every hot loop is Numba-JIT compiled. The outer loop is a single
``prange`` where tasks are **interleaved** (not chunked) via the
permuted task order, balancing thread load across the heterogeneous
(high-cutoff = many-tap = slow, low-cutoff = few-tap = fast) grid
points.

Key design decisions
--------------------

**Numba as an optional accelerator.** The package imports and runs
correctly without Numba installed. All JIT-compiled kernels fall back
to pure Python via :mod:`chaotic_pfc._compat`: a thin compatibility
layer that no-ops ``njit``, aliases ``prange`` to ``range``, and
returns ``1`` for ``get_num_threads()``. Install the ``[fast]`` extra
for 20–50× speedup on the sweep.

**Deterministic reproducibility.** Perturbation vectors are
pre-generated on the Python side using ``default_rng(seed)`` and passed as
NumPy arrays into the Numba kernel. This is necessary because Numba's
per-thread RNG does not honour the global seed. Every run with the same
seed produces bit-exact identical results.

**Lazy optional imports.** 3-D visualisation requires ``plotly``
installed via the ``[viz3d]`` extra. The import is deferred until
:mod:`chaotic_pfc.analysis.sweep_plotting_3d` is actually used, so
``import chaotic_pfc`` always succeeds. The same pattern applies to
:samp:`scipy.signal.firwin` imports within dataclass methods.

**Structural protocols.** :mod:`chaotic_pfc.comms.protocols` defines
``typing.Protocol`` classes: ``Transmitter``, ``Channel``,
``Receiver``: as structural contracts. Concrete implementations
(``transmit``, ``ideal_channel``, ``receive``, etc.) satisfy these
protocols by duck typing. This provides mypy-level documentation of the
communication pipeline's interface without runtime overhead.

**Shallow config hierarchy.** All defaults reside in
:mod:`chaotic_pfc.config` as dataclasses. The hierarchy is
intentionally shallow: every sub-config has only primitive fields
or nested dataclasses of primitives: enabling trivial copying
(``dataclasses.replace``), test isolation, and serialisation
(``dataclasses.asdict``).

**Single ``prange`` with load balancing.** The sweep kernel uses a
single parallel region (not nested parallelism) with a permuted task
order. Tasks with different filter orders have vastly different
per-iteration costs (the MGS inner loop scales as
:math:`\mathcal{O}(N_s^3)`), so simple block partitioning would leave
threads idle. Interleaving ensures that each thread sees a balanced
mix of light and heavy tasks.

**Adaptive early-stop.** The Lyapunov estimator implements an optional
early-stop criterion: if :math:`\lambda_{\max}` stabilises within a
specified tolerance for a streak of consecutive checkpoints, the
remaining iterations are skipped. This provides 3–4× speedup over
fixed-iteration computation with negligible accuracy loss (< 0.1% in
:math:`\lambda_{\max}`).

Configuration
-------------

All default values live in :mod:`chaotic_pfc.config` as
dataclasses:

.. code-block:: python

    from chaotic_pfc.config import DEFAULT_CONFIG as cfg

    # Hénon map parameters
    alpha = cfg.comm.henon.a              # 1.4
    beta  = cfg.comm.henon.b              # 0.3

    # Communication pipeline
    N_samples = cfg.comm.N                # 1_000_000
    modulation_depth = cfg.comm.mu        # 0.01

    # Channel model
    cutoff_freq = cfg.channel.cutoff      # 0.99 (×π)
    num_taps = cfg.channel.num_taps       # 201

    # Sweep parameters
    Nmap = cfg.sweep.Nmap                 # 3000
    N_initial_conditions = cfg.sweep.n_initial  # 25

Experiment scripts import ``DEFAULT_CONFIG`` and override only the
fields exposed as CLI flags. The :meth:`~chaotic_pfc.config.ExperimentConfig.to_namespace`
method generates an ``argparse.Namespace`` for ``run all``, forwarding
a single namespace to all sub-experiments.

Subpackages and their responsibilities
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Subpackage
     - Responsibility
   * - ``dynamics``
     - Hénon map variants, Lyapunov exponents (2-D and 4-D), PSD
       estimation (Welch), binary/sinusoidal message generators
   * - ``comms``
     - Chaotic modulation (transmitter), synchronised demodulation
       (receiver), channel models (ideal, FIR low-pass), DCSK/EF-DCSK
       with AWGN and multipath
   * - ``analysis``
     - Parameter sweep orchestration, Numba JIT kernels, statistical
       post-processing (ranking, correlation, bootstrap CI),
       2-D/3-D plotting
   * - ``plotting``
     - Publication-quality SVG figures with STIX fonts: attractor
       portraits, SDIC overlays, 4×2 communication grids
   * - ``cli``
     - Unified argparse-based CLI with hierarchical subcommands,
       shared helpers (PSD computation, save/show logic)
   * - ``config``
     - Centralised dataclass configuration, default singleton

chaotic-pfc documentation
=========================

A scientific computing library implementing a complete chaos-based
digital communication system using the Hénon map — providing
physical-layer security through chaotic carriers.

The project models an end-to-end pipeline: message generation, chaotic
modulation at the transmitter, propagation through ideal or FIR
band-limited channels, and demodulation through chaos synchronisation at
the receiver — backed by a rigorous Lyapunov-exponent analysis of the
underlying nonlinear dynamics. The library also includes a full
numerical study that classifies regions of the (filter order, cutoff
frequency) parameter space as chaotic, periodic, or divergent.

.. admonition:: Status
   :class: note

   Undergraduate final project (PFC) in Engineering at FESA (Faculdade
   Engenheiro Salvador Arena), Brazil. Developed as part of the scientific
   article *"Sistemas de comunicação baseados em mapas de Hénon utilizando
   filtro FIR"* (2026).

Features
--------

* **Hénon map variants** — 2-D standard Hénon, generalised Hénon, filtered Hénon,
  and N-th order filtered Hénon maps as chaotic oscillators
  (:mod:`chaotic_pfc.dynamics.maps`).
* **Chaos-based communication** — transmitter/receiver pair via Pecora-Carroll
  synchronisation (:mod:`chaotic_pfc.comms.transmitter`,
  :mod:`chaotic_pfc.comms.receiver`).
* **Channel models** — ideal pass-through, FIR band-limited channels, AWGN,
  multipath propagation, impulsive noise, and urban composite channels
  (:mod:`chaotic_pfc.comms.channel`, :mod:`chaotic_pfc.comms.dcsk`).
* **Lyapunov exponent computation** — single-IC and N-IC ensemble protocol
  using tangent-map method with Modified Gram-Schmidt re-orthonormalisation
  (:mod:`chaotic_pfc.dynamics.lyapunov`).
* **High-performance parameter sweep** — deterministic parallel Lyapunov sweep
  across the (filter order, cutoff frequency) grid via Numba JIT compilation
  (:mod:`chaotic_pfc.analysis.sweep`).
* **Modulation schemes** — DCSK, EF-DCSK, and Pecora-Carroll modulation with
  BER-vs-SNR comparison (:mod:`chaotic_pfc.comms.dcsk`).
* **Statistical analysis suite** — summary tables, filter-type comparison,
  transition boundaries, Spearman correlation, bootstrap confidence intervals,
  and optimal-parameter identification (:mod:`chaotic_pfc.analysis.stats`).
* **Publication-quality figures** — SVG outputs with STIX fonts, attractor
  portraits, SDIC visualisation, 4×2 communication grids, and 3-D Plotly
  volumes (:mod:`chaotic_pfc.plotting.figures`).
* **Unified CLI** — ``chaotic-pfc run <experiment>`` backed by importable
  submodules under :mod:`chaotic_pfc.cli`.

Installation
------------

Requires **Python 3.11+**.

.. code-block:: bash

   pip install -e ".[dev]"        # development tools (pytest, ruff, mypy)
   pip install -e ".[dev,fast]"    # development + Numba JIT acceleration
   pip install -e ".[viz3d]"       # 3-D visualisation with Plotly

The package imports and runs *without* Numba installed — kernels fall back
to pure Python via :mod:`chaotic_pfc._compat`. Install the ``[fast]`` extra
for 20–50× speedup on the parameter sweep.

Quick start
-----------

.. code-block:: bash

   chaotic-pfc run all --no-display --quick-sweep

The ``--quick-sweep`` flag runs the full pipeline with a reduced
Lyapunov grid — seconds of compute time rather than hours — sufficient
to smoke-test every component. Remove it to execute the full-resolution sweep.

Key experiments
---------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - **Attractors**
     - Phase-space portraits of three Hénon variants: ``chaotic-pfc run attractors``
   * - **Sensitivity (SDIC)**
     - Exponential divergence visualisation: ``chaotic-pfc run sensitivity``
   * - **Communication**
     - Full pipeline over ideal/FIR channels: ``chaotic-pfc run comm-ideal``
   * - **Lyapunov Spectra**
     - Single-IC and ensemble protocols: ``chaotic-pfc run lyapunov``
   * - **Parameter Sweep**
     - High-resolution (order × cutoff) grid: ``chaotic-pfc run sweep compute``
   * - **BER Comparison**
     - DCSK, EF-DCSK, Pecora-Carroll: ``chaotic-pfc run dcsk``

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   background
   architecture
   internals
   usage
   development
   contributing
   api/index

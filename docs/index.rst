chaotic-pfc documentation
=========================

A scientific computing library implementing a complete chaos-based
digital communication system using the Hénon map.

The project models an end-to-end pipeline — message generation, chaotic
modulation at the transmitter, propagation through ideal or FIR
channels, and demodulation through chaos synchronisation at the
receiver — backed by a rigorous Lyapunov-exponent analysis of the
underlying nonlinear dynamics.

.. admonition:: Status
   :class: note

   Undergraduate final project (TCC) in Engineering at FESA, Brazil.

Features
--------

* 2-D Hénon, generalised Hénon, filtered Hénon, and N-th order filtered
  Hénon maps (:mod:`chaotic_pfc.maps`).
* Chaos-based transmitter / receiver pair
  (:mod:`chaotic_pfc.transmitter`, :mod:`chaotic_pfc.receiver`).
* Ideal and FIR band-limited channels (:mod:`chaotic_pfc.channel`).
* Lyapunov exponent computation — single IC and N-IC ensemble
  protocol (:mod:`chaotic_pfc.lyapunov`).
* Deterministic parallel Lyapunov sweep across the (filter order,
  cutoff) grid via Numba (:mod:`chaotic_pfc.sweep`).
* Unified CLI (``chaotic-pfc run <experiment>``) backed by importable
  submodules under :mod:`chaotic_pfc.cli`.

Quick start
-----------

.. code-block:: bash

   pip install -e ".[dev]"
   chaotic-pfc run all --no-display --quick-sweep

The ``--quick-sweep`` flag runs the full pipeline with a reduced
Lyapunov grid — seconds of compute time rather than hours — which is
enough to smoke-test everything. Remove it to run the full sweep.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   api/index

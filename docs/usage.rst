.. _usage:

Usage guide
===========

This page walks through every CLI workflow with example commands and
their expected outputs.

Full pipeline (quick mode)
--------------------------

.. code-block:: bash

   chaotic-pfc run all --no-display --quick-sweep

Runs every experiment in sequence using a reduced Lyapunov grid
(seconds of compute time rather than hours). Produces attractor
portraits, SDIC visualisation, communication figures, Lyapunov CSV
tables, sweep ``.npz`` checkpoints, and classification maps.

.. tip::

   Use ``--quick-sweep`` for smoke testing. Remove it to execute the
   full-resolution sweep (~40 orders × 100 cutoffs × 25 ICs × 3000
   iterations each).

Single experiments
------------------

Attractors
~~~~~~~~~~

.. code-block:: bash

   chaotic-pfc run attractors

Generates three phase-space portraits (standard, generalised, and
filtered Hénon maps). The :math:`x`-axis shows the first state
variable; the :math:`y`-axis shows the second.

.. code-block:: bash

   chaotic-pfc run attractors --save --lang en

Saves SVG figures with English titles to the ``figures/`` directory.

Sensitivity (SDIC)
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   chaotic-pfc run sensitivity

Overlays two Hénon trajectories with initial conditions separated by
:math:`10^{-4}` to visualise exponential divergence — the hallmark
of chaotic dynamics.

Communication pipeline
~~~~~~~~~~~~~~~~~~~~~~

Three channel types are available for the full transmit-channel-receive
chain:

.. code-block:: bash

   chaotic-pfc run comm-ideal       # noiseless pass-through channel
   chaotic-pfc run comm-fir         # FIR low-pass band-limited channel
   chaotic-pfc run comm-order-n     # order-N Hénon + FIR channel

Each produces a :math:`4 \times 2` grid of time-domain and PSD plots
for the message, carrier, received signal, and recovered message.

Add ``--save`` to write figures to ``figures/``. Use ``--lang pt``
for Portuguese figure labels.

Lyapunov spectra
~~~~~~~~~~~~~~~~

.. code-block:: bash

   chaotic-pfc run lyapunov

Computes Lyapunov exponents for 2-D and 4-D systems in four parts:

* **(A)** 2-D Hénon — single initial condition.
* **(B)** 4-D pole-filtered Hénon — single IC.
* **(C)** 2-D Hénon — ensemble protocol with :math:`N_{\text{CI}}` ICs.
* **(D)** 4-D pole-filtered Hénon — ensemble protocol with :math:`N_{\text{CI}}` ICs.

.. code-block:: bash

   chaotic-pfc run lyapunov --save --n-ci 50

Saves per-IC CSV tables with 50 initial conditions to
``data/lyapunov/``.

Sweep compute
~~~~~~~~~~~~~

Runs the 2-D (filter order, cutoff frequency) Lyapunov sweep:

.. code-block:: bash

   chaotic-pfc run sweep compute --window hamming --filter lowpass

Results are saved as ``data/sweeps/<display-name>/variables_lyapunov.npz``.

.. code-block:: bash

   # Quick mode (~seconds, reduced grid) for smoke testing
   chaotic-pfc run sweep compute --window hamming --filter lowpass --quick

   # Run all window×filter combinations
   chaotic-pfc run sweep compute --all

   # Kaiser window with custom beta and bandpass bandwidth
   chaotic-pfc run sweep compute --window kaiser --filter bandpass \
       --kaiser-beta 8.0 --bandwidth 0.3

.. code-block:: bash

   # Adaptive early-stop: 3-4× speedup with negligible accuracy loss
   chaotic-pfc run sweep compute --window hamming --filter lowpass \
       --adaptive --Nmap-min 500 --tol 1e-3

Sweep plot
~~~~~~~~~~

Generates classification figures from saved ``.npz`` checkpoints:

.. code-block:: bash

   # Plot all window×filter combinations in data/sweeps/
   chaotic-pfc run sweep plot --all

   # Plot only one combination
   chaotic-pfc run sweep plot --window hamming --filter lowpass

   # Specify output directory
   chaotic-pfc run sweep plot --all --figures-dir figures/custom/

.. code-block:: bash

   # Save figures without displaying
   chaotic-pfc run sweep plot --all --save --no-display

Produces (per combination):

* **Heatmap** — continuous :math:`\lambda_{\max}` over
  :math:`(N_z, \omega_c / \pi)`.
* **Classification interleaved** — discrete map: periodic (blue),
  chaotic (red), unbounded (grey).
* **Difficulty map** — adaptive iteration count per grid point
  (only for adaptive sweeps).
* **Beta curves** — :math:`\lambda_{\max}` evolution across Kaiser
  :math:`\beta` values (only for Kaiser sweeps).

Beta sweep
~~~~~~~~~~

Runs the Lyapunov sweep for a range of Kaiser :math:`\beta` values:

.. code-block:: bash

   chaotic-pfc run sweep beta-sweep --beta-min 2.0 --beta-max 10.0 --beta-step 0.5

Runs sweeps for every :math:`\beta` in the interval for each filter
type under the Kaiser window. Results go to ``data/sweeps/kaiser/``.

.. code-block:: bash

   # Adaptive mode saves significant time on beta-sweeps
   chaotic-pfc run sweep beta-sweep --beta-min 2.0 --beta-max 10.0 \
       --beta-step 0.5 --adaptive --tol 1e-3

3-D visualisation (requires plotly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[viz3d]"
   chaotic-pfc run sweep plot-3d --all

Opens an interactive 3-D Plotly volume stacking all :math:`\beta`
surfaces, with configurable camera angle and colour scale.

DCSK comparison
~~~~~~~~~~~~~~~

.. code-block:: bash

   chaotic-pfc run dcsk

Generates a BER-vs-SNR plot comparing three modulation schemes over an
FIR-filtered Hénon map with AWGN:

* Pecora-Carroll synchronisation (coherent)
* Classical DCSK (non-coherent, half bit rate)
* EF-DCSK (non-coherent, full bit rate)

Options:

.. code-block:: bash

   chaotic-pfc run dcsk --snr-min -10 --snr-max 20 --snr-step 2
   chaotic-pfc run dcsk --save --no-display
   chaotic-pfc run dcsk --lang pt  # Portuguese labels

Statistical analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   chaotic-pfc run analysis

Prints a comprehensive 10-section report summarising all sweep results
found under ``data/sweeps/``:

#. **Summary table** — one row per sweep with chaotic/periodic/divergent percentages.
#. **Filter-type comparison** — aggregates per filter type (lowpass, highpass, bandpass, bandstop).
#. **Best chaos-preserving filters** — ranked by chaotic coverage.
#. **Lambda-max distribution** — histogram with skewness statistics.
#. **Transition boundaries** — first chaotic cutoff per filter order.
#. **Spectral robustness (chaos margin)** — width of the chaotic region.
#. **Spearman correlation** — (order, cutoff) vs. :math:`\lambda_{\max}`.
#. **Bootstrap 95% CI** — confidence intervals for chaotic proportion.
#. **Optimal parameters** — (order, cutoff) pairs yielding the highest :math:`\lambda_{\max}`.
#. **Kaiser beta evolution** — :math:`\lambda_{\max}` as function of :math:`\beta`.

.. code-block:: bash

   # Export the summary table as JSON
   chaotic-pfc run analysis --json data/analysis_summary.json

   # Analyse a specific sweep directory
   chaotic-pfc run analysis --data-dir data/sweeps

Language support
----------------

The CLI supports bilingual figure labels via the ``--lang`` flag or the
``CHAOTIC_PFC_LANG`` environment variable:

.. code-block:: bash

   # English labels (default)
   chaotic-pfc run attractors --lang en

   # Portuguese (Brazil) labels
   chaotic-pfc run attractors --lang pt

   # Set default language via environment variable
   export CHAOTIC_PFC_LANG=pt
   chaotic-pfc run attractors

The flag is supported by ``attractors``, ``sensitivity``,
``comm-ideal``, ``comm-fir``, ``comm-order-n``, ``dcsk``, and
``run all``.

Further reading
---------------

* :doc:`architecture` — how the modules fit together.
* :doc:`internals` — Numba kernels, MGS, adaptive early-stop.
* :doc:`background` — complete theoretical foundations.
* :doc:`development` — development environment and tooling.

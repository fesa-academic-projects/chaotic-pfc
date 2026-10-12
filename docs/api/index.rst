API Reference
=============


.. note::

   The API reference below is auto-generated from the source code docstrings
   and is maintained in English only. This is consistent with the practice
   adopted by major scientific Python projects (NumPy, SciPy, scikit-learn)
   and reflects the technical audience of the reference material.
   Narrative documentation (tutorial, background, usage, algorithms) is
   available in Portuguese.

This page documents every public module in the ``chaotic_pfc``
package. Each link below opens the module's reference, listing its
functions, classes, and dataclasses with full signatures and
NumPy-style docstrings extracted from the source.

.. autosummary::
   :toctree: generated
   :recursive:

   chaotic_pfc.dynamics.maps
   chaotic_pfc.dynamics.signals
   chaotic_pfc.dynamics.lyapunov
   chaotic_pfc.dynamics.spectral
   chaotic_pfc.comms.transmitter
   chaotic_pfc.comms.channel
   chaotic_pfc.comms.receiver
   chaotic_pfc.comms.protocols
   chaotic_pfc.comms.metrics
   chaotic_pfc.analysis.sweep
   chaotic_pfc.analysis.stats
   chaotic_pfc.analysis.latex_export
   chaotic_pfc.analysis.sweep_plotting
   chaotic_pfc.analysis.sweep_plotting_3d
   chaotic_pfc.plotting.figures
   chaotic_pfc.config
   chaotic_pfc.cli

Public API
----------

The stable public API is the top-level ``chaotic_pfc`` namespace (its
``__all__`` list: 75 symbols).  Each symbol is documented once, in its
defining submodule above; import it directly from the package root::

    from chaotic_pfc import run_sweep, henon_standard, fir_channel

Sweep catalogue
---------------

The sweep package publishes three configuration constants:

.. list-table::
   :header-rows: 1

   * - Constant
     - Meaning
     - Values
   * - ``WINDOWS``
     - Supported FIR window keys (lower-case)
     - ``hamming``, ``hann``, ``blackman``, ``kaiser``, ``blackmanharris``, ``boxcar``, ``bartlett``
   * - ``FILTER_TYPES``
     - Supported pass-zero configurations
     - ``lowpass``, ``highpass``, ``bandpass``, ``bandstop``
   * - ``WINDOW_DISPLAY_NAMES``
     - Human-readable display name for each window key
     - ``hamming`` → ``Hamming``, …

Their canonical values, kept in sync with the code:

.. autodata:: chaotic_pfc.analysis.sweep._types.WINDOWS
   :no-index:

.. autodata:: chaotic_pfc.analysis.sweep._types.FILTER_TYPES
   :no-index:

.. autodata:: chaotic_pfc.analysis.sweep._types.WINDOW_DISPLAY_NAMES
   :no-index:

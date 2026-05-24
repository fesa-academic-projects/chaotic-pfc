.. _algorithms-fir:

FIR filter design
=================

Motivation
----------

In the chaotic communication scheme, a finite impulse response (FIR)
filter is inserted **inside the Hénon feedback loop**.  This turns the
standard 2‑D map into a higher‑dimensional system whose dynamics are
shaped by the filter coefficients.

The motivation is two‑fold:

1. **Band‑limiting.**  Chaotic signals produced by the unfiltered Hénon
   map have a broad, flat spectrum.  Real communication channels have
   limited bandwidth ; the FIR filter constrains the chaotic signal to
   occupy only the desired frequency band.

2. **Controllable chaos.**  By varying the filter order and cutoff, we
   can study which filter parameters *preserve* chaos and which
   *suppress* it : the core empirical question of the PFC.

``firwin`` vs. MATLAB ``fir1``
------------------------------

The project uses :func:`scipy.signal.firwin` rather than implementing
its own FIR design.  There is a documented off‑by‑one difference
relative to MATLAB's ``fir1``:

* MATLAB: ``fir1(N, wc)`` produces ``N+1`` coefficients.
* SciPy: ``firwin(N, wc)`` produces ``N`` coefficients.

To obtain the same filter as MATLAB's ``fir1(N, wc)``, call
``firwin(N+1, wc)``.  The ``chaotic-pfc`` codebase consistently
uses ``firwin(Nss, wc)`` where ``Nss`` is the number of filter
taps (coefficients), which matches the convention used in the
academic article.

The ``fs=2`` default in ``firwin`` means ``wc`` is in units of
:math:`\pi` rad/sample — consistent with the normalisation
:math:`\omega_c / \pi` used throughout the sweep literature.

Filter types
------------

Four pass‑zero configurations are supported:

.. list-table::
   :header-rows: 1

   * - Type
     - Pass‑band
     - Stop‑band
   * - ``lowpass``
     - :math:`[0, \omega_c]`
     - :math:`(\omega_c, \pi]`
   * - ``highpass``
     - :math:`(\omega_c, \pi]`
     - :math:`[0, \omega_c]`
   * - ``bandpass``
     - :math:`[\omega_c - \Delta/2,\; \omega_c + \Delta/2]`
     - remainder
   * - ``bandstop``
     - remainder
     - :math:`[\omega_c - \Delta/2,\; \omega_c + \Delta/2]`

For bandpass and bandstop, the bandwidth :math:`\Delta` defaults to
:math:`0.0375` — a narrow band typical of the project's experimental
setup.

Lowpass is the default because it is the most common configuration in
communication engineering and serves as the baseline for all comparisons.

Windows
-------

The FIR design uses one of seven window functions to shape the filter's
frequency response:

.. list-table::
   :header-rows: 1

   * - Window
     - Key
     - Sidelobe attenuation
     - Ripple
   * - Hamming
     - ``hamming``
     - ~53 dB
     - Moderate
   * - Hann
     - ``hann``
     - ~44 dB
     - Moderate
   * - Blackman
     - ``blackman``
     - ~74 dB
     - Low
   * - Blackman-Harris
     - ``blackmanharris``
     - ~92 dB
     - Very low
   * - Rectangular
     - ``boxcar``
     - ~13 dB
     - High
   * - Bartlett
     - ``bartlett``
     - ~25 dB
     - Moderate
   * - Kaiser
     - ``kaiser``
     - Adjustable via :math:`\beta`
     - Adjustable

**Kaiser window** deserves special attention.  The shape parameter
:math:`\beta` controls the trade‑off between main‑lobe width and
sidelobe level.  The project includes a dedicated :math:`\beta`-sweep
(``chaotic-pfc run sweep beta-sweep``) that evaluates Lyapunov exponents
across :math:`\beta \in [2, 14]` in steps of :math:`0.5` for each filter
type.

Stability and the fixed point
-----------------------------

When an FIR filter is inserted into the Hénon feedback loop, the
Jacobian matrix grows from :math:`2 \times 2` to :math:`(2 + N_c)
\times (2 + N_c)`, where :math:`N_c` is the number of filter
coefficients that feed back into the dynamics.

The **fixed point** of the combined map‑plus‑filter system is computed
by solving a linear system involving the filter coefficients.  For the
system to have a bounded attractor, the fixed point must be finite
and the filter must be stable (all poles inside the unit circle).

The function :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_max` handles
the 4‑D pole‑filtered variant, which introduces an additional stability
parameter (pole radius :math:`r`) that must satisfy :math:`0 \le r < 1`.

Known limitations
-----------------

* **Cutoff extremes.**  :math:`\omega_c \to 0` produces a filter that
  passes almost nothing (all‑stop); :math:`\omega_c \to 1` passes
  almost everything (all‑pass).  Both extremes degenerate the filter
  and produce behaviour indistinguishable from the unfiltered map —
  the sweep grid excludes :math:`\omega_c = 0` and :math:`\omega_c = 1`.

* **Order limits.**  For :math:`N_z \ge 40`, the filter design becomes
  numerically ill‑conditioned (the window function has extreme values
  at the filter edges).  The sweep currently tops at :math:`N_z = 41`.

* **DC gain.**  FIR filters designed with ``firwin`` have a DC gain
  of approximately 1.0 for lowpass , but not exactly, due to window
  tapering.  The chaotic map is sensitive to the exact coefficient
  magnitudes, which is why the FIR bank is precomputed once rather
  than reconstructed per grid point.

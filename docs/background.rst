.. _background:

Background
==========

This page presents the complete theoretical foundations of
``chaotic-pfc``, covering the physics, mathematics, and engineering
principles behind chaos-based digital communication with FIR-filtered
Hénon maps.

Physical-layer security and chaos-based communication
-----------------------------------------------------

The protection of critical communication infrastructures has become a
central element of digital sovereignty and information security in the
contemporary landscape, especially in a context shaped by "information
warfare" and the constant threat of interception and denial-of-service
attacks [OliveiraFilgueiras22]_ [Sapinski23]_.

While conventional cryptographic methods operate predominantly at the
upper layers of the OSI model (Transport, Application), a strategic
advantage of **Chaos-Based Communication Systems (CBCS)** is providing
security directly at the *Physical Layer* [Baptista21]_.

This approach exploits the intrinsic properties of the chaotic carrier
signal:

* **Broad bandwidth** — the signal occupies a wide portion of the
  spectrum, resembling white noise.
* **Noise-like appearance** — the waveform shows no discernible pattern
  to an observer lacking the system parameters.
* **Sensitivity to initial conditions (SDIC)** — infinitesimal parameter
  mismatches cause exponential divergence, preventing unauthorised
  synchronisation.

Together these properties enable *concealed transmission*, where the
message is not merely encrypted but physically embedded into a chaotic
orbit. The scientific foundation for CBCS was established in 1990 when
Pecora and Carroll demonstrated that two independent chaotic systems
could be synchronised [PecoraCarroll90]_. This discovery proved
that the apparent unpredictability inherent to chaos could be exploited
in a controlled and reproducible manner.

The strategic relevance of CBCS is evidenced by its application in
frontier areas such as 5G network standards, ultra-wideband (UWB)
communications, and Internet of Things (IoT) systems.

The Hénon map
-------------

The Hénon map is a two-dimensional discrete dynamical system introduced
by Michel Hénon in 1976 as a simplified model of a Poincaré section of
continuous systems. Its widespread use as a paradigm for chaotic signal
generation stems from the combination of mathematical simplicity with
rich dynamical behaviour [Henon76]_.

Standard form
~~~~~~~~~~~~~

The system is described by the recurrence equations:

.. math::

    x_1[n+1] &= 1 - a \, x_1[n]^2 + b \, x_2[n] \\
    x_2[n+1] &= x_1[n]

where :math:`x_1[n]` and :math:`x_2[n]` are the state variables at
discrete time :math:`n`, and the parameters :math:`(a, b)` control the
dynamical behaviour.

Canonical chaotic regime
~~~~~~~~~~~~~~~~~~~~~~~~

For the parameter pair

.. math::

    (a, b) = (1.4, 0.3)

the map exhibits a **strange attractor** with:

* Fractal dimension :math:`\approx 1.26`
* Largest Lyapunov exponent :math:`\lambda_1 \approx 0.42` (positive,
  confirming chaos)
* SDIC: trajectories with initial separation :math:`10^{-4}` diverge
  beyond recognition within tens of iterations
* Phase-space structure resembling a "boomerang" or "seagull wing" shape

Parameter dependence
^^^^^^^^^^^^^^^^^^^^

The qualitative behaviour of the Hénon map depends critically on
:math:`(a, b)`:

* For :math:`a < 1.06`: the system is periodic (the fixed point is
  stable).
* For :math:`1.06 \lesssim a \lesssim 1.43`: the system is chaotic,
  with the canonical chaotic regime at :math:`(1.4, 0.3)`.
* For :math:`a > 1.43`: most initial conditions diverge to infinity,
  rendering the system inoperable for communication.

The parameter :math:`b` controls dissipation: :math:`|b| < 1`
makes the map area-contracting (dissipative), while :math:`|b| > 1`
would be area-expanding.

Variants implemented in the package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Generalised Hénon** (:func:`~chaotic_pfc.dynamics.maps.henon_generalised`)
  Uses :math:`\alpha` and :math:`\beta` as parameters, expressed in
  a coupled-oscillator form:

  .. math::

     x_1[n+1] = \alpha - x_1[n]^2 + \beta \, x_2[n]

  This form is convenient for sweeping the parameter space when the
  dissipative coupling :math:`\beta` is the object of study.

**Filtered Hénon** (:func:`~chaotic_pfc.dynamics.maps.henon_filtered`)
  Passes the state through a 2-tap FIR filter before feeding back:

  .. math::

     x_1[n+1] = \alpha - (c_0 x_1[n] + c_1 x_1[n-1])^2 + \beta \, x_2[n]

  With :math:`c_0 = 1, c_1 = 0` this reduces to the generalised map.
  With nontrivial :math:`(c_0, c_1)` the filtered term introduces an
  additional degree of freedom.

**Order-N Hénon** (:func:`~chaotic_pfc.dynamics.maps.henon_order_n`)
  Generalises to :math:`N_c` filter taps, yielding an
  :math:`N_c`-dimensional state vector. This is the main workhorse
  of the parameter-sweep pipeline. For :math:`N_c = 2` with
  :math:`c_0 = 1, c_1 = 0` it reproduces the standard map.

FIR filtering and band-limited chaos
-------------------------------------

The bandwidth problem
~~~~~~~~~~~~~~~~~~~~~

Chaotic signals are **inherently broadband**: their continuous power
spectrum extends over a wide frequency range. However, physical
transmission channels are invariably **band-limited** — every real-world
communication medium (copper wire, optical fibre, radio-frequency
spectrum) imposes a finite bandwidth constraint.

Transmitting a broadband chaotic signal through a band-limited channel
without adaptation results in severe distortion, compromising both
synchronisation quality and message recovery.

The Fontes-Eisencraft solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reconcile bandwidth constraints with chaotic signal generation,
Fontes and Eisencraft (2016) proposed inserting **finite impulse
response (FIR) filters** directly into the feedback loop of the chaotic
generator [FontesEisencraft16]_. The filter design relies on classical
signal-processing techniques — windowing methods (Hamming, Blackman,
Kaiser) — as established by Oppenheim and Schafer
[OppenheimSchafer09]_.

In this architecture, the variable :math:`x_1[n]` in the nonlinearity
:math:`x_1[n]^2` is replaced by the *filtered* version
:math:`x_3[n] = \sum_{k=0}^{N_z-1} c_k \, x_1[n-k]`, where
:math:`c_k` are the FIR coefficients and :math:`N_z` is the filter
order (number of taps).

The result is a **band-limited chaotic signal** whose spectrum is shaped
by the FIR filter, making it compatible with practical channel bandwidth
constraints without sacrificing chaotic behaviour — *provided* the
filter parameters are chosen within regions that preserve the chaotic
regime.

Dimensionality expansion
~~~~~~~~~~~~~~~~~~~~~~~~

Inserting an FIR filter into the feedback loop expands the system
dimensionality. For a map of order :math:`K` and a filter with
:math:`N_s` coefficients, the resulting filtered system has dimension:

.. math::

    K' = K + N_s - 1

For the 2-D Hénon (:math:`K = 2`) with an :math:`N_s`-tap filter, this
yields a :math:`(N_s + 1)`-dimensional system. The stability analysis
then requires the Jacobian matrix of the expanded :math:`K'`-dimensional
system.

Shrimps and bifurcation cascades
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Investigations by Borges and Eisencraft (2022) revealed that FIR
filtering introduces unexpected dynamical complexity
[BorgesEisencraft22]_. The linear filter interacts non-trivially
with the map's nonlinearity, potentially inducing:

* **Bifurcation cascades** — sequences of period-doubling transitions
  as filter parameters vary.
* **"Shrimps"** — islands of periodicity immersed within chaotic
  regions of the parameter space. Named for their characteristic
  shape in parameter-space diagrams.

The presence of shrimps is critical for communication security: if the
system enters a periodic regime due to quantisation errors or hardware
variations, the signal becomes predictable and vulnerable to spectral
analysis.

This interaction motivates the **central research question** of the
project:

   *Under what conditions does FIR filtering preserve the chaotic regime
   necessary for physical-layer security under bandwidth constraints?*

Lyapunov exponents
------------------

Definition
~~~~~~~~~~

The **largest Lyapunov exponent** :math:`\lambda_{\max}` quantifies
the average rate of exponential divergence (or convergence) of
infinitesimally close trajectories in phase space:

.. math::

    \lambda_{\max} = \lim_{n \to \infty} \frac{1}{n}
    \sum_{k=1}^{n} \ln \frac{\| \delta \mathbf{x}_k \|}
                              {\| \delta \mathbf{x}_0 \|}

where :math:`\delta \mathbf{x}_k` is the perturbation vector at
iteration :math:`k`, evolved through the tangent map of the system.

Classification
~~~~~~~~~~~~~~

.. list-table:: Orbit classification by :math:`\lambda_{\max}`
   :header-rows: 1
   :widths: 25 50 25

   * - Condition
     - Meaning
     - Regime
   * - :math:`\lambda_{\max} > 0`
     - Exponential divergence of nearby trajectories; SDIC present
     - **Chaotic**
   * - :math:`\lambda_{\max} \leq 0`
     - Trajectories converge or remain bounded without exponential growth
     - **Periodic / quasiperiodic**
   * - :math:`\lambda_{\max} = \text{NaN}`
     - Trajectory diverged to infinity within the simulation window
     - **Divergent (unbounded)**

.. important::

   The condition :math:`\lambda_{\max} > 0` is the **mathematical
   imperative** that guarantees SDIC and, consequently, the security of
   concealed transmission. This is the metric used throughout the sweep
   pipeline to classify each (order, cutoff) grid point.

Combinatorial validation protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sweep pipeline estimates :math:`\lambda_{\max}` at **every point**
of a 2-D grid of (filter order :math:`N_z`, cutoff frequency
:math:`\omega_c`) pairs, building a classification map of the entire
parameter space. For each grid point:

1. **FIR bank precomputation** — all filter coefficient sets are
   computed analytically (for zero-placement experiments) or via
   :func:`scipy.signal.firwin` (for windowed designs).
2. **Initial condition generation** — :math:`N_{\text{IC}}` initial
   conditions are uniformly distributed in a sphere of radius
   :math:`0.01` centered at the stable fixed point :math:`p^+` of the
   filtered system.
3. **Transient discard** — the first :math:`N_{\text{discard}}`
   iterations are discarded to allow convergence to the attractor.
4. **Lyapunov estimation** — :math:`N_{\text{itera}}` iterations are
   accumulated using the tangent-map method with Modified Gram-Schmidt
   (MGS) re-orthonormalisation at every step.
5. **Ensemble aggregation** — :math:`\lambda_{\max}` is reported as
   the mean over all :math:`N_{\text{IC}}` initial conditions.

Tangent-map method and Modified Gram-Schmidt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tangent-map method propagates both the main orbit and a set of
linearised perturbation vectors simultaneously. However, the
exponential growth of these vectors would quickly cause numerical
overflow or collapse (all vectors aligning with the dominant
eigen-direction).

The **Modified Gram-Schmidt (MGS)** procedure is applied at every
iteration to re-orthonormalise the perturbation vectors, with the
logarithmic scaling factors accumulated for the final estimate of
:math:`\lambda_{\max}`. The implementation uses QR factorisation
compiled via Numba JIT for performance.

Fixed-point stability
~~~~~~~~~~~~~~~~~~~~~

For the filtered Hénon map, the stability analysis also requires the
Jacobian matrix of the expanded :math:`K'`-dimensional system.
Convergent results from Borges, Silva, and Eisencraft (2024) demonstrate
that:

* The **fixed-point locations** depend primarily on the total filter
  gain :math:`G`.
* The **stability** of these points is highly sensitive to the
  individual coefficient distribution and the zero locations of the
  filter in the complex plane [BorgesSilvaEisencraft24]_.

This dependence motivates the need for a systematic (rather than
pointwise) characterisation of the filters employed.

Numba JIT acceleration
~~~~~~~~~~~~~~~~~~~~~~

The inner loop of the Lyapunov computation is
:math:`\mathcal{O}(N_s^3)` due to the MGS re-orthonormalisation of an
:math:`N_s`-dimensional perturbation ensemble. Numba JIT compilation
reduces this to practical execution times, enabling sweeps with up to
:math:`N_s = 41` filter taps, :math:`N_{\text{IC}} = 25` initial
conditions per grid point, and :math:`N_{\text{itera}} = 3\,000`
iterations per IC — totalling :math:`40 \times 100 = 4\,000` grid
points, each with :math:`25 \times 3\,000 = 75\,000` orbit
evaluations. Without Numba, this would require tens of hours; with
JIT, it completes in minutes on a modern multicore processor.

Pecora-Carroll synchronisation
-------------------------------

Principle
~~~~~~~~~

The receiver runs a copy of the transmitter's chaotic oscillator driven
by the received signal. This is formalised as a **conditional response
subsystem**: given a master chaotic system with state vector
:math:`\mathbf{x}(t)`, a slave subsystem is synchronised when:

.. math::

    \|\mathbf{x}(t) - \hat{\mathbf{x}}(t)\| \to 0
    \quad \text{as} \quad t \to \infty

where :math:`\hat{\mathbf{x}}(t)` is the receiver state.

The **necessary condition** for convergence is that all **conditional
Lyapunov exponents** of the slave subsystem are negative, ensuring that
perturbations in the state difference are attenuated over iterations
[PecoraCarroll90]_ [Williams01]_.

Receiver equations for the 2-D Hénon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the standard 2-D Hénon, the receiver state evolves as:

.. math::

    y_1[n+1] &= 1 - a \, r[n]^2 + b \, y_2[n] \\
    y_2[n+1] &= y_1[n]

where :math:`r[n]` is the received (possibly channel-distorted) carrier.
The receiver does **not** use its own :math:`y_1[n]` in the
nonlinearity — instead, it is driven by :math:`r[n]`, which carries the
transmitter state plus the embedded message.

After a short transient (typically 200–500 iterations), the receiver
state converges: :math:`y_1[n] \approx x_1[n]`. The message is then
recovered by subtracting the local estimate:

.. math::

    \hat{m}[n] \approx m[n] = \frac{r[n] - y_1[n]}{\mu}

where :math:`\mu` is the modulation depth (a small positive constant,
typically 0.01).

Modulation schemes
------------------

The package implements three modulation techniques:

**CSK — Chaos Shift Keying.** Sistematised by Williams (2001)
[Williams01]_. Each binary symbol corresponds to a distinct set of
parameters or initial conditions of the chaotic generator: the message
is encoded in the selection between two different chaotic attractors.
The receiver performs detection by correlation or energy comparison,
identifying which attractor the received signal belongs to. CSK requires
a synchronisation period before each symbol, reducing spectral
efficiency.

**DCSK — Differential Chaos Shift Keying.** Introduced by Kolumban et
al. (1996) [Kolumban96]_. Each bit period is split into two equal
halves: a reference slot transmitting a chaotic reference sequence,
followed by a data slot transmitting the same reference (bit 0) or its
negated version (bit 1). The receiver correlates the two slots to
recover the bit. DCSK does **not** require chaos synchronisation,
making it inherently robust to channel variations. The cost is a 50%
reduction in bit rate (one information bit per two transmitted chaotic
sequences).

**EF-DCSK — Efficient DCSK.** Proposed by Kaddoum et al. (2013)
[Kaddoum13]_. Improves the data rate by using a single slot per bit:
the reference and its time-reversed copy are superposed in the same
slot. The decoder correlates the received signal with its own
time-reverse to recover the bit. This doubles the throughput of
classical DCSK with minimal BER degradation.

Performance metrics
-------------------

Quantitative evaluation of digital communication performance relies on
two standard metrics [Haykin01]_ [LathiDing09]_:

**BER — Bit Error Rate.** The ratio of incorrectly received bits to the
total number of transmitted bits. In chaotic systems, BER integrates
the combined effects of channel AWGN (additive white Gaussian noise),
synchronisation imperfections, quantisation errors, and transient
settling before receiver convergence.

**SNR — Signal-to-Noise Ratio.** The ratio of average signal power to
noise power, expressed in dB:

.. math::

    \text{SNR}_{\text{dB}} = 10 \log_{10} \frac{P_s}{P_n}

The **BER-vs-SNR curve** is the primary comparison tool between
modulation schemes, identifying the SNR threshold required for
acceptable operation (commonly :math:`\text{BER} \leq 10^{-3}`).

The ``chaotic-pfc run dcsk`` command produces exactly this curve for
Pecora-Carroll synchronisation, classical DCSK, and EF-DCSK over an
FIR-filtered Hénon map with AWGN.

References
----------

.. [Baptista21] M. S. Baptista.
   "Chaos-based communication systems: Current trends and challenges."
   Springer, 2021.

.. [BorgesEisencraft22] V. S. Borges, M. Eisencraft.
   "A filtered Hénon map."
   Chaos, Solitons and Fractals, vol. 165, 2022.

.. [BorgesSilvaEisencraft24] V. S. Borges, R. Silva, M. Eisencraft.
   "Stability analysis of the filtered Hénon map."
   (preprint), 2024.

.. [FontesEisencraft16] R. Fontes, M. Eisencraft.
   "A digital bandlimited chaotic communication system."
   Commun. Nonlinear Sci. Numer. Simul., vol. 37, pp. 374--385, 2016.

.. [Haykin01] S. Haykin.
   "Communication Systems." 4th ed., Wiley, 2001.

.. [Henon76] M. Hénon.
   "A two-dimensional mapping with a strange attractor."
   Commun. Math. Phys., vol. 50, pp. 69--77, 1976.



.. [LathiDing09] B. P. Lathi, Z. Ding.
   "Modern Digital and Analog Communication Systems."
   4th ed., Oxford University Press, 2009.

.. [OliveiraFilgueiras22] E. V. Oliveira, R. Filgueiras.
   "A importância da segurança da informação para as organizações."
   Revista Alomorfia, v. 6, n. 1, pp. 438--447, 2022.

.. [OppenheimSchafer09] A. V. Oppenheim, R. W. Schafer.
   "Discrete-Time Signal Processing." 3rd ed., Pearson, 2009.

.. [PecoraCarroll90] L. M. Pecora, T. L. Carroll.
   "Synchronization in chaotic systems."
   Physical Review Letters, v. 64, n. 8, p. 821, 1990.

.. [Sapinski23] A. Sapinski.
   "The Importance and Challenges of Information Security in the
   Digital Age." Scientific Journal of Bielsko-Biala School of
   Finance and Law, v. 1, pp. 52--55, 2023.

.. [Williams01] C. Williams.
   "Chaotic communications over radio channels."
   IEEE Trans. Circuits Syst. I, v. 48, n. 12, pp. 1394--1404, 2001.

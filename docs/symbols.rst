.. _symbols:

Symbols glossary
================

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Symbol
     - Name
     - Appears in

   * - :math:`\alpha`
     - Hénon constant term (generalised form)
     - :func:`~chaotic_pfc.dynamics.maps.henon_generalised`,
       :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`,
       ``_sweep_kernel``

   * - :math:`\beta`
     - Hénon coupling / dissipation parameter
     - :func:`~chaotic_pfc.dynamics.maps.henon_generalised`,
       :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`,
       ``_sweep_kernel``

   * - :math:`a`
     - Hénon nonlinear parameter (classic form)
     - :func:`~chaotic_pfc.dynamics.maps.henon_standard`

   * - :math:`b`
     - Hénon coupling parameter (classic form)
     - :func:`~chaotic_pfc.dynamics.maps.henon_standard`

   * - :math:`\mu`
     - Modulation depth
     - :func:`~chaotic_pfc.comms.transmitter.transmit`,
       :func:`~chaotic_pfc.comms.receiver.receive`

   * - :math:`N_s`
     - Number of FIR filter coefficients (taps)
     - ``_sweep_kernel``,
       :func:`~chaotic_pfc.analysis.sweep.precompute_fir_bank`

   * - :math:`N_z = N_s - 1`
     - FIR filter order (number of zeros)
     - ``_sweep_kernel``,
       :func:`~chaotic_pfc.analysis.sweep.precompute_fir_bank`

   * - :math:`N`
     - Number of samples in a transmitted sequence
     - :func:`~chaotic_pfc.config.CommConfig`,
       :func:`~chaotic_pfc.comms.transmitter.transmit`

   * - :math:`\omega_c`, :math:`f_c`
     - Normalised cutoff frequency, :math:`\omega_c / \pi \in (0, 1)`
     - :func:`~chaotic_pfc.analysis.sweep.precompute_fir_bank`,
       :func:`~chaotic_pfc.config.InternalFIRConfig`

   * - :math:`c_k`, :math:`b_k`
     - FIR filter coefficients
     - :func:`~chaotic_pfc.dynamics.maps.henon_order_n`,
       ``_henon_nN_inplace``

   * - :math:`G`
     - Filter DC gain, :math:`G = \sum_k c_k`
     - ``_sweep_kernel``,
       ``_fixed_point``

   * - :math:`u_n`
     - FIR filter output, :math:`u_n = \sum_k c_k x_{n-k}`
     - :func:`~chaotic_pfc.dynamics.maps.henon_order_n`,
       ``_henon_nN_inplace``

   * - :math:`x_n, y_n`
     - Hénon map state variables
     - :func:`~chaotic_pfc.dynamics.maps.henon_standard`,
       :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`

   * - :math:`\lambda_1`, :math:`\lambda_{\max}`
     - Largest Lyapunov exponent
     - :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`,
       ``_sweep_kernel``

   * - :math:`s[n]`
     - Transmitted (modulated) signal
     - :func:`~chaotic_pfc.comms.transmitter.transmit`

   * - :math:`r[n]`
     - Received signal (after channel)
     - :func:`~chaotic_pfc.comms.receiver.receive`,
       :func:`~chaotic_pfc.comms.channel.awgn`

   * - :math:`m[n]`
     - Original binary message
     - :func:`~chaotic_pfc.dynamics.signals.binary_message`

   * - :math:`\hat{m}[n]`
     - Recovered message estimate
     - :func:`~chaotic_pfc.comms.receiver.receive`

   * - :math:`\beta_K`
     - Kaiser window shape parameter
     - :func:`~chaotic_pfc.analysis.sweep.precompute_fir_bank`,
       ``run_beta_sweep``

   * - :math:`N_{\text{discard}}`
     - Burn-in (transient discard) iterations
     - :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`,
       :func:`~chaotic_pfc.config.LyapunovConfig`

   * - :math:`N_{\text{IC}}`
     - Number of initial conditions in ensemble
     - :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d_ensemble`,
       ``_sweep_kernel``

   * - :math:`\text{SNR}`, :math:`E_b/N_0`
     - Signal-to-noise ratio (dB)
     - :func:`~chaotic_pfc.comms.channel.awgn`

   * - :math:`\text{BER}`
     - Bit error rate
     - :func:`~chaotic_pfc.comms.receiver.receive` (via MSE proxy)

Terminology
-----------

The English documentation and the Brazilian-Portuguese translation use one
canonical form per concept; the table gives both, with the forms to avoid.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Concept
     - English
     - Portuguese (pt-BR)
   * - orbit class
     - **divergent** (the class); *unbounded* (the orbit)
     - **divergente**; *órbita não limitada* (avoid "ilimitado")
   * - cutoff
     - **cutoff frequency**
     - **frequência de corte** (avoid a bare "corte")
   * - order / coefficients
     - **order** :math:`N_z`; **number of coefficients** :math:`N_s`
     - **ordem**; **número de coeficientes** (avoid "taps")
   * - chaotic coverage
     - **chaotic coverage**
     - **cobertura caótica**
   * - design window
     - **design window**
     - **janela de projeto**
   * - fixed point
     - **fixed point**
     - **ponto fixo**
   * - checkpoint
     - **checkpoint** (``.npz``)
     - **checkpoint**
   * - filter types
     - lowpass, highpass, bandpass, bandstop
     - passa-baixas, passa-altas, passa-faixa, rejeita-faixa

The classification figures label the three orbit classes with **periodic
orbits**, **chaotic orbits** and **unbounded orbits**. This follows the
terminology of the advisor's article (reference [14] of the manuscript,
``BorgesCNSNS2024``): V. S. Borges, M. T. M. Silva, M. Eisencraft, "Chaotic
properties of an FIR filtered Hénon map", *Communications in Nonlinear Science
and Numerical Simulation*, vol. 131, p. 107845, 2024, which uses "unbounded
orbits" for the orbits that diverge to infinity (its classification lists
periodic orbits in blue, chaotic orbits in red and divergence to infinity in
gray). The manuscript's figure captions describe the same class as *divergent*.

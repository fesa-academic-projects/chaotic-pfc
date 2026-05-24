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
       :func:`~chaotic_pfc.analysis.sweep._kernel._sweep_kernel`

   * - :math:`\beta`
     - Hénon coupling / dissipation parameter
     - :func:`~chaotic_pfc.dynamics.maps.henon_generalised`,
       :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`,
       :func:`~chaotic_pfc.analysis.sweep._kernel._sweep_kernel`

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

   * - :math:`N_z`, :math:`N_s`, :math:`N`
     - FIR filter order (number of taps)
     - :func:`~chaotic_pfc.analysis.sweep._kernel._sweep_kernel`,
       :func:`~chaotic_pfc.analysis.sweep._orchestration.precompute_fir_bank`

   * - :math:`\omega_c`, :math:`f_c`
     - Normalised cutoff frequency, :math:`\omega_c / \pi \in (0, 1)`
     - :func:`~chaotic_pfc.analysis.sweep._orchestration.precompute_fir_bank`,
       :func:`~chaotic_pfc.config.InternalFIRConfig`

   * - :math:`c_k`, :math:`b_k`
     - FIR filter coefficients
     - :func:`~chaotic_pfc.dynamics.maps.henon_order_n`,
       :func:`~chaotic_pfc.analysis.sweep._kernel._henon_nN_inplace`

   * - :math:`G`
     - Filter DC gain, :math:`G = \sum_k c_k`
     - :func:`~chaotic_pfc.analysis.sweep._kernel._sweep_kernel`,
       :func:`~chaotic_pfc.dynamics.lyapunov._fixed_point`

   * - :math:`u_n`
     - FIR filter output, :math:`u_n = \sum_k c_k x_{n-k}`
     - :func:`~chaotic_pfc.dynamics.maps.henon_order_n`,
       :func:`~chaotic_pfc.analysis.sweep._kernel._henon_nN_inplace`

   * - :math:`x_n, y_n`
     - Hénon map state variables
     - :func:`~chaotic_pfc.dynamics.maps.henon_standard`,
       :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`

   * - :math:`\lambda_1`, :math:`\lambda_{\max}`
     - Largest Lyapunov exponent
     - :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`,
       :func:`~chaotic_pfc.analysis.sweep._kernel._sweep_kernel`

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
     - :func:`~chaotic_pfc.analysis.sweep._orchestration.precompute_fir_bank`,
       :func:`~chaotic_pfc.cli.sweep._beta.run_beta_sweep`

   * - :math:`N_{\text{discard}}`
     - Burn-in (transient discard) iterations
     - :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d`,
       :func:`~chaotic_pfc.config.LyapunovConfig`

   * - :math:`N_{\text{IC}}`
     - Number of initial conditions in ensemble
     - :func:`~chaotic_pfc.dynamics.lyapunov.lyapunov_henon2d_ensemble`,
       :func:`~chaotic_pfc.analysis.sweep._kernel._sweep_kernel`

   * - :math:`\text{SNR}`, :math:`E_b/N_0`
     - Signal-to-noise ratio (dB)
     - :func:`~chaotic_pfc.comms.channel.awgn`

   * - :math:`\text{BER}`
     - Bit error rate
     - :func:`~chaotic_pfc.comms.receiver.receive` (via MSE proxy)

.. _tutorial:

Tutorial
========

This tutorial runs the whole ``chaotic-pfc`` pipeline once, from the Hénon map
to FIR-filtered chaotic communication, in about a minute of compute time. It
uses the 0.9 API (see :doc:`/usage` for the rename table).

1. Install
----------

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,fast]"     # dev tools + Numba JIT

2. A Hénon orbit, in Python
---------------------------

.. code-block:: python

   import numpy as np
   from chaotic_pfc import henon_standard, lyapunov_henon2d

   X, Y = henon_standard(200, x0=0.1, y0=0.1)          # alpha=1.4, beta=0.3 by default
   result = lyapunov_henon2d(Nitera=2000, Ndiscard=1000)
   print(result.lyapunov_max)                          # ~0.419

The generic estimator takes any map and its Jacobian:

.. code-block:: python

   from chaotic_pfc import lyapunov_spectrum

   spec = lyapunov_spectrum(
       iterate=lambda x: np.array([1.4 - x[0] ** 2 + 0.3 * x[1], x[0]]),
       jacobian=lambda x: np.array([[-2.0 * x[0], 0.3], [1.0, 0.0]]),
       x0=np.array([0.1, 0.1]),
   )

3. The full pipeline, from the CLI
----------------------------------

.. code-block:: bash

   chaotic-pfc run all --no-display --quick-sweep

This runs every experiment with a reduced Lyapunov grid (seconds instead of
hours): attractor portraits, the SDIC visualisation, the communication figures,
the Lyapunov tables, the sweep checkpoints and the classification maps. Drop
``--quick-sweep`` for the full-resolution sweep.

4. The notebook
---------------

For a guided version with the figures inline:

.. code-block:: bash

   jupyter nbconvert --to notebook --execute examples/tour.ipynb   # headless
   jupyter notebook examples/tour.ipynb                            # interactive

The notebook is also run in CI on every push, so it stays up to date with the
API.

Next steps
----------

* :doc:`/usage` — every CLI workflow, flag by flag.
* :doc:`/source/algorithms/lyapunov` — how the exponents are estimated.
* :doc:`/source/algorithms/sweep` — the parameter sweep and reproducibility.

.. _tutorial:

Tutorial
========

A self-contained Jupyter notebook walks through the full ``chaotic-pfc``
pipeline — from the Hénon map to FIR-filtered chaotic communication —
in about one minute of compute time.

.. code-block:: bash

   # Execute without opening a browser
   jupyter nbconvert --to notebook --execute examples/tour.ipynb

   # Or open interactively
   jupyter notebook examples/tour.ipynb

The notebook is also run in CI on every push to ensure it stays
up-to-date with the API.

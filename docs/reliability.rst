.. _reliability:

Reliability and failure analysis
================================

This page records the failure modes that matter for ``chaotic-pfc`` and the
layers that detect them. It follows the FMEA structure: severity (S),
occurrence (O) and detection (D) are scored from 1 (best) to 10 (worst), and
the risk priority number is ``RPN = S x O x D`` before the mitigation.

FMEA
----

.. list-table::
   :header-rows: 1
   :widths: 17 19 4 4 4 5 25 22

   * - Failure mode
     - Effect
     - S
     - O
     - D
     - RPN
     - Mitigation
     - Detection
   * - Out-of-bounds access in a Numba kernel
     - Wrong grid cell, possible crash
     - 9
     - 3
     - 8
     - 216
     - Bounds derived from ``len(cutoff_grid)``; warmup uses the same dtype
       and slice
     - ``test_contract.py`` and ``test_kernel_paths.py`` with
       ``NUMBA_DISABLE_JIT=1``; ``test_sweep_orchestration`` warmup spy
   * - Silent type conversion (``3.7`` to ``3``)
     - A different filter is computed than requested
     - 7
     - 6
     - 7
     - 294
     - ``require_positive_int`` rejects non-integral floats and bools
     - ``test_validation.py``; contract fuzzing
   * - Silent ``NaN`` from invalid parameters
     - Corrupted statistics published as results
     - 9
     - 6
     - 8
     - 432
     - Domain validation plus ``on_divergence``; ``np.errstate`` at the source
     - ``-W error::RuntimeWarning`` job; ``test_validation.py``
   * - ``N_s`` stored as ``N_z``
     - Wrong order column in tables and figures
     - 6
     - 6
     - 8
     - 288
     - ``N_z = N_s - 1`` in the rankings; shared vocabulary
     - ``test_analysis.py``; Semgrep ``ns-as-nz``
   * - Constants diverging between modules
     - Different correlation or percentage in two reports
     - 6
     - 5
     - 8
     - 240
     - Single ``WINDOW_SIDELOBE_DB`` and ``pct`` helpers
     - ``test_spearman_matches_between_report_and_numbers``; golden tables
   * - Rounding half-way cases differently
     - Table values disagree with the manuscript
     - 5
     - 6
     - 8
     - 240
     - ``Decimal`` ``ROUND_HALF_UP`` from integer counts
     - ``test_round_half_up_percentages``; Semgrep ``numpy-round``
   * - Pickle deserialisation of a hostile ``.npz``
     - Arbitrary code execution on the user's machine
     - 10
     - 4
     - 9
     - 360
     - JSON metadata; digest checked against a compiled trust anchor; the same
       bytes are hashed and loaded (no TOCTOU)
     - ``test_sweep_io.py`` forged-manifest and malicious-payload tests; Semgrep
       ``allow-pickle`` and ``read-sha256sums``
   * - Partial LaTeX installation
     - Figure generation fails or falls back silently
     - 5
     - 6
     - 7
     - 210
     - Cached render probe with the project preamble
     - ``test_latex_probe_failure_falls_back_to_mathtext``
   * - Incompatible dependency version
     - Import or numerical failure on a supported Python
     - 8
     - 5
     - 7
     - 280
     - Declared floors, lock file, minimum-version CI job
     - ``minimums`` job; ``pip-audit``; ``test_sweep_environment.py``
   * - Non-determinism across thread counts
     - Published result cannot be reproduced
     - 7
     - 4
     - 8
     - 224
     - Pre-drawn perturbations; per-cell independent kernels
     - ``TestThreadDeterminism``
   * - Invalid generated artefact
     - Broken manuscript tables shipped to a reviewer
     - 7
     - 5
     - 8
     - 280
     - Golden files plus LaTeX compilation in the pinned image
     - ``test_golden_tables.py``; *Manuscript tables (LaTeX)* job
   * - Regression from a rename
     - Silent ``TypeError`` in examples or benchmarks
     - 6
     - 6
     - 6
     - 216
     - Keyword aliases with warnings; contract fuzzing over ``__all__``
     - ``test_contract.py``; ``test_maps``/``test_sweep_orchestration`` alias
       tests

Lessons from Therac-25 and Ariane 501
-------------------------------------

Therac-25 (Leveson and Turner, *IEEE Computer*, 1993) and Ariane 501 (Lions
et al., *Inquiry Board Report*, 1996) are the reference incidents for this
project.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Lesson
     - Incident
     - Layer in this repository
   * - Reused code out of its validated domain
     - Ariane 501 converted a 64-bit horizontal bias with a 16-bit routine
       designed for a different range and then reused the value.
     - Domain checks (:mod:`chaotic_pfc._validate`), explicit
       ``on_divergence``, and the metamorphic tests that pin each estimator's
       assumptions.
   * - No independent check of a computed value
     - Ariane trusted a value without a second check. Therac trusted a flag
       without verifying the machine state.
     - Metamorphic identities (spectrum sum equals ``mean(log|det J|)``), the
       analytical ``ln|beta|`` identity, and ``mpmath`` at 50 digits.
   * - Silent recovery hid a fault
     - Therac displayed "Malfunction 54" and continued; Ariane's exception
       handler was not exercised in flight.
     - ``InvalidParameterError``, ``DivergenceError`` and
       ``UntrustedDataError`` are explicit; ``-W error::RuntimeWarning`` makes
       silent NaNs fail the build.
   * - Inadequate testing of boundary states
     - Both systems were tested for nominal inputs, not for out-of-range or
       concurrent states.
     - Hypothesis contract fuzzing (single and paired mutations) and the
       ``NUMBA_DISABLE_JIT=1`` job.
   * - Configuration and documentation drift
     - Therac's hardware and software changed without a matching safety
       analysis.
     - Golden files, the manuscript audit, Semgrep rules per incident, and
       literal ``CHANGELOG`` entries.
   * - No traceability from requirement to code
     - Reviewers could not map a requirement to the line that implemented it.
     - Each FMEA row points to the test or job that detects it; each Semgrep
       rule names its incident.
   * - Single point of failure in the process
     - One team, one toolchain, one assumption.
     - Independent layers: unit, contract, metamorphic, mutation, static
       analysis, artefact compilation and supply-chain audit.

How to use this page
--------------------

When adding a new numerical module, add a row to the FMEA table with at least
one detection mechanism, and a contract baseline in
``tests/test_contract.py``. When fixing a defect, add a Semgrep rule that
remembers the failure mode.

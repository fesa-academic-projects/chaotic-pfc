.. _security:

Security model
==============

This page describes the threat model of the ``chaotic-pfc`` library. It covers
what the library protects, where untrusted input can enter, and which controls
exist. It also states what the library does **not** protect.

Assets
------

1. **Integrity of the scientific results.** The 124 versioned sweep
   checkpoints, the exported tables and the manuscript numbers are canonical.
   A silently corrupted input must never be published as a result.
2. **Integrity of the user's machine.** Opening a checkpoint must not execute
   code, and importing or calling the library must not run external programs
   with attacker-controlled arguments.
3. **Reproducibility.** A run must be bit-identical for the same seed and
   thread count, so that a third party can verify a published number.

Trust boundaries and attack surfaces
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Surface
     - Untrusted input
     - Control
   * - Public API parameters
     - Scalars, arrays and strings passed by callers
     - Boundary validation (:mod:`chaotic_pfc._validate`) raising
       :class:`~chaotic_pfc.errors.InvalidParameterError`
   * - ``.npz`` checkpoints
     - A file produced by a third party
     - ``allow_pickle=False`` plus SHA-256 manifest verification
       (:class:`~chaotic_pfc.errors.UntrustedDataError`)
   * - Manuscript and table files
     - ``.tex``, ``.bib`` and SVG sources
     - Deterministic audit, golden files and LaTeX compilation in CI
   * - Environment
     - ``CHAOTIC_PFC_*`` variables, ``DISPLAY``, ``NO_COLOR``
     - Documented variables only; no command execution from them
   * - Dependencies
     - NumPy, SciPy, Matplotlib, Numba, optional Plotly
     - Version floors, lock file, ``pip-audit``, Dependabot
   * - CI and releases
     - Workflows, tags, artefacts
     - SHA-pinned actions, least privilege, ``zizmor``, ``gitleaks``,
       build provenance and SBOM

Threats and mitigations
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Threat
     - Mitigation
   * - Malicious pickle in a checkpoint (code execution)
     - New saves store metadata as JSON. Legacy pickled files are opened only
       when their SHA-256 digest is listed in a repository ``SHA256SUMS``
       manifest, or when the caller explicitly passes
       ``trust_legacy_pickle=True``.
   * - Invalid parameter producing ``NaN`` or a cryptic exception
     - Every public function validates its arguments and raises
       ``InvalidParameterError`` (a ``ValueError`` subclass) with the parameter
       name and the violated domain.
   * - Silent divergence published as a result
     - ``on_divergence`` selects ``"nan"`` (default), ``"warn"``
       (``DivergenceWarning``) or ``"raise"`` (``DivergenceError``).
   * - Silent floating-point warnings
     - CI runs the suite with ``-W error::RuntimeWarning``; expected warnings
       are handled with a local ``np.errstate`` at the source.
   * - Unsafe deserialisation added later
     - A Semgrep rule fails the build on ``np.load(..., allow_pickle=True)``
       outside the verified legacy loader.
   * - Out-of-bounds access in a Numba kernel
     - The contract suite and the kernel tests also run with
       ``NUMBA_DISABLE_JIT=1``, where an out-of-bounds index raises
       ``IndexError``.
   * - Database drift between modules (sidelobes, percentages, orders)
     - Single shared constants and helpers (``WINDOW_SIDELOBE_DB``,
       ``pct``/``round_half_up``) with unit and golden-file tests.
   * - Invalid generated artefact
     - Every exported table is regenerated from the checkpoints and compiled
       in the pinned TeX Live image.
   * - Tampering with the canonical data
     - Pre-commit hooks reject staged edits under ``data/sweeps/``,
       ``articles/simac/`` and ``articles/jcis/Figures/`` unless
       ``CHAOTIC_PFC_ALLOW_FROZEN=1``, and verify both ``SHA256SUMS``.
   * - Supply-chain compromise
     - Actions pinned by SHA, least-privilege permissions, hashed lock file,
       ``pip-audit``, ``zizmor``, ``gitleaks``, CycloneDX SBOM and build
       provenance attestations.
   * - Renamed API used with old keywords
     - Deprecated aliases warn and reject ambiguous combinations. They are
       removed at 1.0.

Security layers, in order of defence
------------------------------------

1. **Validate at the boundary** (:mod:`chaotic_pfc.errors` and
   :mod:`chaotic_pfc._validate`).
2. **State the divergence policy** explicitly (``on_divergence``).
3. **Verify integrity before trust** (JSON metadata, SHA-256 manifests).
4. **Fuzz the contract** (``tests/test_contract.py``, short and deep profiles).
5. **Pin numerical references** (metamorphic identities, ``mpmath``).
6. **Prove determinism** across Numba thread counts.
7. **Fail on silent warnings** (``RuntimeWarning`` gate).
8. **Compile the artefacts** inside the pinned TeX Live image.
9. **Remember the incidents** (Semgrep rules and Ruff rule families).
10. **Audit the supply chain** (hashes, ``pip-audit``, ``zizmor``,
    ``gitleaks``, SBOM, provenance).
11. **Guard the repository** (pre-commit hooks and importable rulesets).

Non-goals
---------

The Pecora-Carroll chaos-based scheme implemented here is **not
cryptographically secure**. Its security relies on synchronous chaotic
dynamics, and several attacks against such schemes are known in the
literature; see G. Alvarez and S. Li, "Some basic cryptographic requirements
for chaos-based cryptosystems", *International Journal of Bifurcation and
Chaos*, vol. 16, no. 8, pp. 2129-2151, 2006, doi:10.1142/S0218127406015970.

In particular:

* the library does not provide confidentiality, authentication or key
  management;
* the chaotic carrier must not be used as a replacement for a standard
  authenticated cipher;
* any deployment that needs confidentiality must put standard cryptography
  (for example TLS or a modern AEAD construction) in the layers above this
  library.

``chaotic-pfc`` is a research tool for the numerical study of the filtered
Hénon map and of chaos-based communication. It is not a security product.

Reporting a vulnerability
-------------------------

See the repository ``SECURITY.md`` for the private reporting channel and the
supported versions.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- The ``.npz`` sweep checkpoints now use the API vocabulary: the ``coef`` key
  (number of coefficients) is ``num_taps_grid`` and ``wcorte`` is
  ``cutoff_grid``. ``load_sweep`` still reads the old keys, so checkpoints from
  earlier versions (including the frozen SIMAC state ``f517e0a``) load
  unchanged.
- `requires-python` is now `>=3.13` (SPEC 0): Python 3.12 is no longer
  supported. The dependency floors are NumPy 2.2, SciPy 1.15 and Matplotlib
  3.11, and a CI job installs exactly those minimums and runs the suite.
- The standalone scripts in `tools/` are replaced by CLI subcommands:
  `tools/paper_numbers.py` and `tools/audit_paper.py` by
  `chaotic-pfc paper numbers|audit`; `tools/regenerate_sweeps.py`,
  `compare_sweeps.py` and `report_sweep_stats.py` by
  `chaotic-pfc sweeps regenerate|compare|stats`; and `tools/jcis_figures.py`
  by `chaotic-pfc paper figures`. Only `tools/check_pt_br_translations.sh`
  (documentation tooling) remains in `tools/`.
- The JCIS manuscript moved from `jcis/` to `articles/jcis/`; the frozen
  VIII SIMAC record now lives in `articles/simac/`. The `make` targets keep
  their names (`paper-pdf`, `paper-diff`, `check-paper`, `jcis-figures`).
- `chaotic-pfc run sweep beta-sweep` writes to `data/sweeps/kaiser/` (the
  layout `run all` and the analysis tools read) instead of `data/sweeps/beta/`.
- The 4-D pole-filtered Lyapunov estimator `lyapunov_max` is renamed
  `lyapunov_max_pole_filtered`; the name `lyapunov_max` now denotes the generic
  estimator for an arbitrary map.
- Old-to-new names (per function/parameter):

  | Function | Old | New |
  |---|---|---|
  | `henon_standard` | `a`, `b` | `alpha`, `beta` |
  | `henon_fir_sequence` | `a`, `b`, `n_taps`, `wc` | `alpha`, `beta`, `num_taps`, `cutoff` |
  | `henon_order_n` | `a`, `b` | `alpha`, `beta` |
  | `transmit`, `transmit_order_n` | `a`, `b` | `alpha`, `beta` |
  | `receive`, `receive_order_n` | `a`, `b` | `alpha`, `beta` |
  | `HenonConfig` | `a`, `b` | `alpha`, `beta` |
  | `run_sweep`, `precompute_fir_bank` | `orders`, `cutoffs` | `num_taps_grid`, `cutoff_grid` |
  | `plot_classification_interleaved`, `plot_heatmap_continuous` | `orders`, `cutoffs` | `num_taps_grid`, `cutoff_grid` |
  | `SweepResult` (attributes) | `.orders`, `.cutoffs` | `.num_taps_grid`, `.cutoff_grid` (the `.npz` keys `coef`/`wcorte` are unchanged in 0.9) |
  | `beta_summary` | `dict[str, dict[float, dict]]` | `dict[str, dict[float, BetaPoint]]` (`BetaPoint` TypedDict) |
  | `quick_sweep_params` | positional 4-tuple | `QuickSweepParams` NamedTuple (`num_taps_grid_lp`, `num_taps_grid_hp`, `cutoff_grid`, `params`) with `params` a `QuickSweepConfig` TypedDict |
  | `QuickSweepParams` (fields) | `.orders_lp`, `.orders_hp` | `.num_taps_grid_lp`, `.num_taps_grid_hp` |
- Public functions take their non-leading arguments by keyword only,
  e.g. `henon_standard(steps, *, x0, y0, alpha, beta)`,
  `transmit(message, *, mu, alpha, beta, x0, y0)`,
  `fir_channel(s, *, num_taps, cutoff, window)`; the filter argument order is
  `num_taps` then `cutoff`.
- The remaining public functions that return a mapping already carry a named
  element type and keep the mapping: `bootstrap_confidence`,
  `compare_filter_types`, `kaiser_beta_optimal`, `lmax_distribution`,
  `sweet_spot_per_filter`, `top_k_per_filter` (`dict[str, <TypedDict>]`) and
  `consolidate_kaiser`, `load_all_sweeps` (`dict[tuple[str, str], SweepResult]`).
  Their key set is data-dependent (a filter type or a window may be absent), so a
  fixed-key `TypedDict` would over-constrain them.

### Added
- A `minimums` CI job installs the SPEC 0 floors (Python 3.13, NumPy 2.3,
  SciPy 1.15, Matplotlib 3.11) and runs the whole suite; a `release-pypi` job
  publishes to PyPI with OIDC Trusted Publishing on `v*` tags, after
  `release-testpypi`.
- Security and compliance: `SECURITY.md`, `CODEOWNERS`, CodeQL (Python) and
  OpenSSF Scorecard workflows, REUSE (`LICENSES/`, `REUSE.toml`, `reuse lint`
  in CI), `codemeta.json` and `.zenodo.json`.
- `make paper-pdf` now compiles the manuscript inside the pinned image
  `texlive/texlive@sha256:acd65bd2…` (TeX Live 2026; needs Docker or Podman, and
  `make setup` warns when neither is installed), so `articles/jcis/article.pdf`
  is byte-reproducible; `make paper-pdf-local` builds a system-TeX-Live preview
  into `articles/jcis/build/` (gitignored). The CI job *Manuscript PDF
  reproducibility* rebuilds the PDF in the same image and requires it to be
  byte-identical to the versioned file (other TeX Live distributions are not
  byte-identical; the caveat is documented in the README and RELEASING).
- `articles/simac/SHA256SUMS` pins the two frozen SIMAC PDFs and the two
  reconstructed `.docx` files, with `tests/test_simac_manifest.py` validating
  the manifest on every test run.
- Audit check **C8** (`chaotic-pfc paper audit --sweeps data/sweeps`) verifies
  that every number cited in the Results and the Final Remarks matches a
  `paper numbers` label (compared as half-up roundings), with a documented list
  of intentional ignores (map/harness constants, figure-layout lengths,
  publication years, values derived inline). `make check-paper` passes
  `--sweeps data/sweeps`, so the check runs in the manuscript-audit CI job.
- `chaotic-pfc paper numbers|audit|figures` and
  `chaotic-pfc sweeps regenerate|compare|stats`: the manuscript and sweep
  tooling now lives in the package (`chaotic_pfc.paper`, `chaotic_pfc.sweeps`)
  and is driven from the CLI; the DOI whitelist ships as package data and is
  read with `importlib.resources`.
- `data/sweeps/SHA256SUMS`, the sha256 of the 124 checkpoints, rewritten by
  `chaotic-pfc sweeps regenerate` and verified by an always-on test.
- `articles/`: the JCIS manuscript under `articles/jcis/` and the frozen
  VIII SIMAC record (its PDFs, the reconstructed `.docx` files, and an errata)
  under `articles/simac/`.
- CI: a `paper-check` job (`make check-paper` on push and PR), a weekly
  `linkcheck` workflow, and a `build` job whose wheel/sdist is reused by every
  other job and by the release.
- `make setup` (venv + extras + pre-commit hooks + optional-tool checks) and
  `make install` aligned with the README.
- CLI (clig.dev): global `--debug` (or `$CHAOTIC_PFC_DEBUG=1`, full traceback),
  `--quiet`/`--verbose`, `--color {auto,always,never}` (an explicit value wins;
  `auto` honours a non-empty `$NO_COLOR`), `--no-progress`, and a help string on
  every argument. Long operations (`sweeps regenerate`, `run all`,
  `sweep compute`, `run sweep beta-sweep`, `paper numbers`) print a
  `index/total label` progress counter to stderr only when stderr is a terminal
  and neither `--quiet` nor `--no-progress` was passed (never on stdout, even
  with `--json`). `sweeps regenerate --dry-run` lists what would be regenerated
  without writing anything.
- `.editorconfig` and a pull-request template.
- Regression and validation tests for the sweep kernel: the partial-divergence
  rule, the non-finite Hann/Bartlett filters at `N_s = 2`, the adaptive
  early-stop streak, and the frequency response of the designed filters.
- `fallback_mode` policy for the hybrid Lyapunov estimator (`0`, `1`, `2`),
  with `1` (full-spectrum recomputation of marginal cells) as the default.
- `tests/test_cli_paper_sweeps.py` covering the new CLI groups.
- `data/sweeps/ENVIRONMENT.json` recording the stack that regenerated the 124
  checkpoints, with `tests/test_sweep_environment.py` pinning the lock file, the
  `coef` field of the checkpoints and a byte fingerprint of the
  `scipy.signal.firwin` coefficient banks.
- Tests for the manuscript tooling (`tests/test_paper_tools.py`) on a synthetic
  sweep tree: `paper numbers` (every section), the `paper.figures` generators
  and `run paper-figures`.
- A `no-numba` CI job and tests covering the pure-Python fallback: the
  `_compat` shims (Numba absent) and the sweep-kernel paths.
- Kernel-anchor tests: the standard Hénon map (~0.419, Sprott) and the analytic
  `alpha=2, beta=0` map (conjugate to the logistic map at `r=4`, `ln 2`).
- Frequency-response tests for every window of the catalogue times the four
  filter types, and a fallback-mode class-stability test on the baseline
  configurations.
- `lyapunov_spectrum(iterate, jacobian, x0)` and `lyapunov_max(...)`: a public,
  map-agnostic Lyapunov estimator with doctests and tests against the standard
  Hénon map and the analytic `alpha=2, beta=0` map (`ln 2`).
- `OrbitClass` (`IntEnum`): the `classify` codes `-1`/`0`/`2` now have names
  (`PERIODIC`/`CHAOTIC`/`DIVERGENT`); the values are unchanged.
- `run_sweep(fir_bank=...)` and `run_sweep(fir_design=...)`: sweep a
  user-supplied FIR coefficient bank or design function, validated and
  documented (FIR only).
- `chaotic_pfc._formulas`: the single definition of the cancellation-free
  positive fixed point and the 2-D Hénon step used by the kernel and the
  Python code.

### Changed
- Raised the `[fast]` extra floor from `numba>=0.59` to `numba>=0.61.2` (the
  first Numba release with both Python 3.13 wheels and `numpy<2.3,>=1.24`
  support) and pinned `numba==0.61.2` in the `minimums` CI job.
- Regenerated the reference LaTeX analysis tables under
  `data/analysis_output/tables/{pt,en}/` with `chaotic-pfc run analysis
  export-tables`, adding the previously missing `en/` directory and reflecting
  the `N_z = N_s - 1`, `longtable` and `ROUND_HALF_UP` fixes.
- Dependencies updated to the latest stable compatible with the SPEC 0 floors:
  NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.0, llvmlite 0.50.0, Matplotlib 3.11.2
  (plus the dev/docs stacks); `requirements-lock.txt` was regenerated from the
  new environment. The 124 sweep checkpoints were regenerated with the new
  kernel: **no number changed** (`sweeps compare`: 0 class changes,
  `max |Δλ| = 0`; `paper numbers`: 0 diffs; the JCIS figures and the exported
  tables are byte-identical), so the manuscript, the figures and the bit-exact
  baseline are untouched.
- The Brazilian-Portuguese filter names in the exported LaTeX tables are now
  plural (`Passa-baixas`, `Passa-altas`); the eight tables under
  `data/analysis_output/tables/pt/` were regenerated (all numbers unchanged,
  and the JCIS/SIMAC figures are untouched).
- All GitHub Actions are pinned by commit SHA (with a version comment), and
  every workflow declares `permissions: contents: read` at the top, granting
  extra scopes (`id-token: write`, `security-events: write`) only to the jobs
  that need them.
- `pyproject.toml` uses a PEP 639 licence expression and `license-files`, the
  `Typing :: Typed` classifier and a minimal `build-system`
  (`setuptools>=77`); the CI lists and checks the sdist/wheel contents. REUSE
  declares BSD-3-Clause for the whole tree.
- Dependabot groups NumPy, SciPy and Numba into a single `scientific-core`
  pull request, separate from the other dependencies.
- `chaotic_pfc.analysis.sweep_plotting` and `chaotic_pfc.analysis.stats` were
  split into private subpackages (`sweep_plotting/`, `stats/`) behind unchanged
  facades; `tests/test_sweep.py` was split by subject.
- The `release` job now runs only after `lint`, `typecheck`, `test` and
  `baseline` pass on the tag (those jobs no longer skip tags), and it publishes
  the artefact built once by `build`.
- `chaotic-pfc paper audit` now fails with exit code 1 and
  `error: missing input file …` when the `.tex` or `.bib` does not exist
  (it previously printed a warning and exited 0).
- `chaotic-pfc.paper.audit` and `chaotic-pfc.paper.numbers` import each other
  normally (no more `tools/` sibling loading); the JCIS classification panels
  are defined once and shared with `chaotic-pfc run paper-figures`.
- `make paper-pdf`'s `SOURCE_DATE_EPOCH` is stable across the
  `jcis/` → `articles/jcis/` move, so the committed PDF is unchanged.
- The Lyapunov sweep kernel now inlines the cancellation-free positive fixed
  point (`2α / ((1−β) + √((1−β)² + 4αG²))`), stable as `G → 0`.
- The 124 versioned sweeps in `data/sweeps/` were regenerated with the current
  kernel; `tests/test_sweep_regressions.py` reproduces them bit for bit with
  Numba and `fallback_mode=1`.
- `requirements-lock.txt` is aligned to the environment that regenerated the
  124 checkpoints (`data/sweeps/ENVIRONMENT.json`), and now includes
  `hypothesis` so the test suite (and the bit-exact baseline) runs from the
  lock.
- The shared figure-writing helper moved to `chaotic_pfc._figure_io`, so the
  `analysis` package no longer imports a private name from `plotting`
  (`tests/test_layering.py` keeps the edge removed).
- The expected overflow of the diverging map in the pure-Python loops
  (`transmit`/`receive` and the `henon_*` iterators) is silenced with
  `np.errstate`; the Numba kernels never emitted the warning.
- Every CLI command now routes expected failures through the `guard` helper
  (one-line `error: …` on stderr, exit code 1); an unreadable `.npz` reports
  `invalid or corrupted checkpoint: <path>`, and `paper numbers` filters the
  expected SciPy `firwin` `RuntimeWarning` and explains it on stderr.
- `mypy --strict` is enforced across `src/chaotic_pfc` (`strict = true`, plus
  `--strict` in the `typecheck` Makefile target and the CI job). The only
  overrides are the third-party `numba`/`scipy`/`matplotlib` stubs.
- `classify` now returns an `NDArray[np.int64]` array carrying the `OrbitClass`
  codes (it returned `float64` before); the classification figures are
  unchanged.
- The public-API array annotations carry the concrete element type:
  `NDArray[np.float64]` for states, signals, coefficients, cutoffs and exponents
  and `NDArray[np.int64]` for the tap counts and the `classify`/`OrbitClass`
  codes (`classify` now returns an `int64` array). Only the `eigenvalues` fields
  keep `NDArray[Any]`, because they are real or complex depending on the
  Jacobian.

### Fixed
- `_lyapunov_spectrum` now runs inside `np.errstate(over="ignore",
  invalid="ignore")`, returns `np.full(dim, np.nan)` immediately when the state
  or `J @ W` becomes non-finite, and documents the QR column ordering in
  `lyapunov_spectrum` and `lyapunov_max`.
- Replaced deprecated `Colormap.set_bad` and `Colormap.set_under` calls in
  `chaotic_pfc.analysis.sweep_plotting` with `cmap.with_extremes(bad=...,
  under=...)`.
- `chaotic_pfc.plotting.figures.latex_available` and `setup_rc` now run a lazy,
  cached render probe with `text.usetex=True` and the project `mathptmx`
  preamble instead of only checking `shutil.which("latex")`, falling back to
  STIX mathtext when the TeX installation is incomplete.
- `_warmup_kernel` now passes `int64` tap counts (matching `run_sweep` so Numba
  compiles `_sweep_kernel` once) and bounds the warmup cutoff count with
  `min(3, len(cutoffs_arr))` so grids with fewer than 3 cutoffs do not index
  out of bounds.
- Updated `benchmarks/` (`standalone.py`, `test_henon_bench.py`,
  `test_lyapunov_bench.py`, `test_sweep_bench.py`) to the 0.9 public API names
  (`alpha`/`beta`, `lyapunov_max_pole_filtered`, `num_taps_grid`/`cutoff_grid`)
  and added a `pytest benchmarks --benchmark-disable -q` smoke step to the CI
  `test` job.
- `chaotic_pfc.analysis.stats` now uses `Decimal` with `ROUND_HALF_UP` from
  exact integer counts (shared with `paper.numbers`) instead of NumPy
  round-half-to-even, so half-way percentages (such as `434/4000 = 10.9%` and
  `149/2000 = 7.5%`) match the manuscript tables.
- `chaotic_pfc.sweeps.report` and `chaotic_pfc.paper.numbers` now share a single
  `WINDOW_SIDELOBE_DB` table (13/27/32/43/58/92 dB, Harris sidelobe levels),
  fixing the inverted Hann/Hamming rank order in `chaotic-pfc sweeps stats`.
- `export_full_ranking_table` and `export_consolidated_full_ranking` now emit
  valid LaTeX `longtable` syntax with the column specification `{col_spec}` and
  place `\caption` and `\label` followed by `\\` inside `\begin{longtable}`.
- `sweet_spot_per_filter` and `optimal_parameters` now record the filter order
  `N_z = N_s - 1` instead of the tap count `N_s` (`num_taps_grid[i]`).
- The pure-Python Lyapunov wrappers (`chaotic_pfc.dynamics.lyapunov`) were
  reverted to their own NumPy Hénon step and fixed point. The single-formula refactor
  had swapped them for the Numba-compiled `_formulas` versions, which differ by
  ~1 ULP (first divergence at iteration 6750, amplified by the chaotic map) and
  moved the single-seed validation value from `0.41756661` to `0.42280037`. The
  sweep kernel keeps the shared compiled formula, so the versioned `.npz`, the
  `paper numbers` and the bit-exact baseline are unchanged; a new
  `tests/test_lyapunov_regression.py` pins the published values.
- Every bibliography entry now carries an identifier: `Sprott2003` and
  `Oppenheim2009` (books with no publisher DOI) keep their ISBN and gained a
  publisher URL, and `articles/jcis/article.pdf` was rebuilt.
- `make check-paper` now scans the article figure tree
  (`--figures articles/jcis/Figures --all-figures`), verifying the 9 SVGs of
  the manuscript; it previously pointed at the unrelated `figures/` tree and
  skipped the scan.
- The CI coverage gate (`--cov-fail-under=75`) is met again (70.33 % before
  the coverage work): the manuscript tooling moved into the package is now covered by
  `tests/test_paper_tools.py`.
- The JCIS manuscript (`articles/jcis/article.tex`) no longer ends the
  Subsection IV-B paragraph with the leftover fragment "The lowpass filter
  with", the remainder of a removed sentence about Experiment V of the earlier
  study; `articles/jcis/article.pdf` was rebuilt.
- `make paper-pdf` pins `SOURCE_DATE_EPOCH` to the last commit that changed the
  manuscript sources, so the committed `articles/jcis/article.pdf` is
  reproducible across unrelated commits.
- The Hypothesis property tests are reproducible again. The strategies in
  `tests/_hypothesis_strategies.py` no longer call the unseeded
  `np.random.default_rng()` (the NaN masks are drawn from Hypothesis), the `ci`
  profile sets `derandomize=True`, and
  `test_standard_henon_yields_positive_lyapunov_max` — whose premise (λ_max > 0
  for every bounded orbit in the box around (1.4, 0.3)) is false inside the
  periodic windows — is replaced by a deterministic check of the canonical
  chaotic point (λ_max ≈ 0.41922) plus the invariants λ1 + λ2 = ln|b| and
  λ1 ≥ λ2 for bounded orbits. This removes an intermittent failure of the test
  suite.

### Documentation
- The pt-BR catalogue check now fails on an entry whose `msgstr` still equals
  its `msgid` (unless it is listed in the new
  `docs/locale/pt_BR/translation-exceptions.txt`) and on forbidden glossary
  terms (`taps`, `esfera`, singular `passa-baixa`, and the pt-PT forms
  `partilh*`, `secç*`, `registad*`, `ficheir*`, `utilizador*`, `ecrã`).
- The narrative documentation follows the Diátaxis split (Tutorial,
  How-to guides, Reference, Explanation); the tutorial is self-contained and
  the main public functions (`henon_standard`, `classify`,
  `precompute_fir_bank`, `run_sweep`, `load_sweep`, `transmit`, `receive`,
  `lyapunov_spectrum`, …) carry executable doctests run by `make docs-strict`
  (25 tests).
- A terminology glossary (English + pt-BR, with the forms to avoid) was added to
  `docs/symbols.rst`.
- Corrections in the algorithm pages: the estimator narrative now describes the
  production path (single tangent vector renormalised every ten iterations plus
  the full-spectrum fallback for marginal cells), the reproducibility section
  also names SciPy's `firwin`, the adaptive early-stop documents its measured
  horizon, and the fallback section records the full-grid Table II measurement
  and the `C8` cost; the stale "CI only smoke-tests" claim was fixed.
- The README documents the project scope, the global CLI flags and
  the 0.9 migration table (linked to this changelog), corrects the
  `paper numbers` example, and documents the `CHAOTIC_PFC_RUN_BASELINE` baseline,
  `python3`, the LaTeX/Inkscape toolchain and the badges; `docs/usage.rst` gains
  the same CLI/API sections.
- `docs/conf.py`: `nitpick_ignore_regex` was narrowed to the external libraries
  and NumPy-style type tokens; the references to the project's own symbols are
  now qualified in the source, and the `n_initial` special case was removed by
  writing the shape as a literal.

## [0.8.0] - 2026-09-13

### Added
- Property-based testing with Hypothesis: 11 invariants for Hénon maps, Lyapunov exponents,
  signals, area summaries, and Kaiser consolidation (`tests/test_properties.py`).
- Custom Hypothesis strategies in `tests/_hypothesis_strategies.py` and profiles
  (`dev` / `ci`) in `tests/conftest.py`.
- Scientific validation tests against Wolf et al. (1985) and Sprott (2003) with
  reference tables in `docs/validation.rst`.
- Tutorial notebook (`examples/tour.ipynb`, ~1 min) covering Hénon maps, Lyapunov
  exponents, filtered chaos, Pecora-Carroll communication, and parameter sweeps.
- CI job (`notebook`) that executes `examples/tour.ipynb` on every push.
- Algorithm documentation pages: Lyapunov computation (`source/algorithms/lyapunov.rst`),
  parameter sweep (`source/algorithms/sweep.rst`), FIR filter design
  (`source/algorithms/fir_design.rst`).
- `docs/testing.rst` and `docs/tutorial.rst` documentation pages.
- Mathematical symbol glossary (`docs/symbols.rst`) mapping every symbol used in the
  library to its definition and source modules.
- References sections citing Hénon (1976), Pecora-Carroll (1990), Wolf (1985),
  Sprott (2003), Fontes-Eisencraft (2016), and Borges-Eisencraft (2022) in key
  public docstrings.
- Complete Brazilian Portuguese translations for all 13 narrative documentation
  pages via `docs/locale/pt_BR/LC_MESSAGES/`.
- API reference translation policy: auto-generated docstrings kept in English only
  with an explanatory note in `api/index.rst`.
- Performance benchmarks with pytest-benchmark: 9 tests covering Hénon maps,
  Lyapunov exponents, and mini-sweeps with v0.7.0 baseline.
- `area_summary()`, `lmax_statistics()`, `rank_configurations()`, `top_k_per_filter()`,
  `sweet_spot_per_filter()`, `consolidate_kaiser()`, `kaiser_beta_optimal()` in `stats.py`.
- `AreaSummary`, `LmaxStats`, `ConfigRank`, `SweetSpot`, `KaiserBetaOptimal` TypedDicts.
- Bilingual LaTeX table export (`latex_export.py`): 8 exporters covering top-k,
  extended top-k, full ranking, sweet spots, beta-optimal, and consolidated Kaiser.
- CLI subcommand `chaotic-pfc run analysis export-tables` with `--lang pt|en|all`.
- Chaotic union and density cross-sweep figures (`plot_chaotic_map`, `plot_chaotic_density`).
- LaTeX auto-detection in `setup_rc()` with mathtext fallback for systems without LaTeX.
- Interleaved layout in `plot_heatmap_continuous` and `plot_difficulty_map`.
- `validate-pyproject` pre-commit hook.
- Python 3.14 support in CI and classifiers.
- ASCII text-message generator (`text_message`) encoding a string as an NRZ BPSK
  bit stream, and a `--text` flag on `comm-ideal`, `comm-fir`, and `comm-order-n`
  (default `"hello"` via `CommConfig.message_text`).
- `chaotic-pfc run paper-figures` subcommand rendering the SIMAC and JCIS
  camera-ready figures at their final printed size; it replaces the standalone
  `tools/paper_figures.py` script.
- `CITATION.cff` (CFF 1.2.0) listing the five authors with ORCIDs, plus a
  `cffconvert` pre-commit hook validating it.

### Changed
- `CHANGELOG.md` fully conforms to Keep a Changelog 1.1.0 with reference links.
- `pyproject.toml`: description, keywords, and classifiers tightened to reflect the
  Pecora-Carroll scope; `hypothesis` in `[dev]` extra; `nbsphinx` + `jupyter` in
  `[docs]` extra.
- `docs/conf.py`: `nbsphinx` extension added; `nbsphinx_execute` set to `"always"`
  so the tour notebook renders with outputs on Read the Docs.
- Default figure format changed from PNG+SVG to SVG-only (`SweepConfig.fig_fmts`,
  `plot_all` fmt, CLI `--fmt` default).  PNG available via `--fmt png`.
- `tour.ipynb`: code cells use English-only strings; LaTeX equations use proper
  display/inline modes; Pecora-Carroll communication demo added; bilingual
  structure cleaned up (H1 EN + H2 PT-BR, `---` separators, no `/` mixed labels).
- `henon_fir_sequence` deprecated (no callers; superseded by `henon_order_n`).
- TCC renamed to PFC across all layers (English source, Python help strings,
  Portuguese translations).
- Prose em-dashes replaced with commas, colons, or parentheses in both English
  `.rst` sources and Portuguese `.po` translations.
- 374 tracked PNG figures removed from version control; `optimize-png` pre-commit
  hook removed (optimize-svg kept, manual stage only).
- Generated Plotly 3-D HTML figures removed from version control (4 files, ~15 K lines).
- Orphan `algorithms/` i18n folder removed (stale leftovers from a previous refactor).
- `_save_both` renamed to `_save_svg`; hardcoded PNG paths removed from plotting code.
- `nbsphinx_execute` changed from `"never"` to `"always"`.
- `analysis_output/` output files moved to `data/analysis_output/tables/{pt,en}/`.
- `--lang` flag aligned with other subcommands (`pt`/`en` instead of `pt_BR`).
- LaTeX tabular tables use `\resizebox{\textwidth}{!}`, `\footnotesize`, and
  `\setlength{\tabcolsep}{4pt}` for A4 fit.
- CI test job uses `--hypothesis-profile=ci`.
- Default bandwidth narrowed from 0.2 to 0.05 across the sweep pipeline.
- README (both languages): updated description, scope, validation wording, and
  BibTeX citing section.
- `np.random.seed` replaced with `np.random.default_rng` in benchmarks.
- Pre-commit: ruff bumped to v0.15.14; `validate-pyproject` hook added; mypy
  configured as `language: system`.
- `comm-ideal`, `comm-fir`, and `comm-order-n` transmit the ASCII text message
  instead of a periodic square wave; `to_namespace()` exposes the text so
  `run all` keeps working.
- Minimum supported Python raised to 3.12 (`requires-python`, classifiers, CI).
- Publication typography restored in generated figures: parallel workers
  re-apply `setup_rc()`, and the `usetex` branch loads `mathptmx` so the LaTeX
  and mathtext paths both target Times.
- `sweep plot` meshes rasterized and saved at dpi=600 from `fig.bbox_inches`,
  shrinking SVG size and save time; `sweep plot --all` renders in parallel.
- Sweep kernel sped up: single-vector Lyapunov estimator with hybrid fallback,
  Benettin block reorthonormalisation (K=10), and divergence checks hoisted out
  of the hot loop.
- Remaining PT-BR prose translated (`internals`, `architecture`) and
  double-escaped math in the translations fixed.
- Version, README, and AUTHORS aligned to 0.8.0 and `CITATION.cff`; AUTHORS now
  identifies every contributor by ORCID.

### Removed
- DCSK and EF-DCSK from project scope (module, CLI, i18n keys, SVG figures,
  documentation pages, README sections, source docstring references).
- Joblib/loky references from documentation — the codebase never used joblib;
  parallelism is via Numba ``prange`` only.
- Dead WiFi interferer channel code (`channel_urban`).
- BER mention from `CommConfig` docstring.
- Sentinel value `-1e30` from sweep figures and Lyapunov ensemble data
  (replaced with proper NaN propagation).

### Fixed
- `henon_fir_sequence` uses the generalised Hénon form (a - xf² + b·y, y = x)
  for consistency with the rest of the codebase.
- `receive_order_n` validates `len(fir_coeffs) >= 3`, matching the check in
  `henon_order_n`.
- `henon_standard`, `henon_order_n`, and JIT kernels guard against divergence
  with early NaN propagation.
- `psd_normalised` raises `ValueError` on identically zero input, honouring its
  documented contract.
- `_fixed_point` guards against degenerate filter cases (pole on unit circle,
  near-zero DC gain).
- Lyapunov kernel guards against divergence leaking the `-1e30` sentinel into
  sweep averages; Ns=1 fixed-point offset handled correctly.
- Zero-norm handling unified between kernel (`1e-300` clamp) and public
  API (`1e-300` fallback in `_gram_schmidt`).
- All mypy errors resolved across the codebase (array covariance, TypedDict
  returns, `**kwargs` unpacking, optional imports).
- `_compat.prange` fallback accepts `(start, stop)` signature matching Numba's API.
- Parameter validation added to public APIs: `awgn` SNR, `channel_impulsive`
  probability, `lyapunov` perturbation and iteration counts, `sweep` adaptive
  parameters, `spectral` window name and `kaiser_beta`.
- Corrupted sweep metadata and unknown config paths now surface clear
  `ValueError` instead of cryptic attribute errors.
- `LmaxDistribution` return type aligned with actual implementation.
- `plot_comm_grid` uses explicit sentinel defaults for `y_lim` parameters.
- `receive` and `receive_order_n` guard against `mu=0` division by zero.
- `run_all` sets attractor steps to 50k for dense phase portraits.
- `load_all_sweeps()` Kaiser detection: switched from `kaiser_beta is not None`
  to `result.window == "kaiser"`.
- `top_k_per_filter()` re-ranks entries per filter group instead of preserving
  global rank numbers.
- Sphinx docstring warnings in `lyapunov.py` resolved (`lyapunov_max_ensemble`,
  `lyapunov_henon2d_ensemble`).
- Duplicate Sphinx citation warnings resolved by consolidating `[Citations]` in
  `background.rst`.
- `:option:` replaced with `:code:` in sweep.rst.
- Fixed-point approximation in sweep kernel documented with rationale.
- Welch segment overlap default (`noverlap=None` = 50%) documented in `psd_normalised`.
- `rng=None` non-reproducible behaviour documented in `awgn`.
- Various cross-reference fixes in `transmitter`, `signals`, `channel`,
  `lyapunov`, `config`, and `sweep_plotting` docstrings.
- DCSK references cleaned from all docstrings, READMEs, and documentation pages.
- Numba bindings typed under the new `prange`-as-class stubs.
- Matplotlib stub gaps in the panel-label measurement path silenced for mypy.

## [0.7.0] - 2026-05-10

### Fixed
- FIR circular buffer off-by-one in `henon_fir_sequence`: `h[0]` was multiplying the
  oldest buffer value instead of the newest, scrambling coefficient order for all DCSK.
- Triple AWGN in `channel_urban`: each sub-call applied independent AWGN at the requested
  SNR, producing 3x the intended noise power. SNR is now compensated per sub-call.
- Wrong expected Lyapunov exponent sum: diagnostic printed `2*ln|beta|` instead of the
  correct `ln|beta| + 2*ln|r|` for the 4-D pole-filtered system.
- `summary_table` chaotic count silently undercounted when `h` contained NaN; now uses
  the same `~isnan` mask as periodic and divergent counts (symmetric with `beta_summary`).
- Four broken Sphinx cross-references: `chaotic_pfc.channel.*` routed to
  `chaotic_pfc.comms.channel.*`; `chaotic_pfc.spectral.*` to `chaotic_pfc.dynamics.spectral.*`.
- Missing `[Kolumban96]` and `[Kaddoum13]` bibliography entries in `background.rst`.
- `to_namespace()` missing `Nc` and `internal_cutoff` fields, breaking `run all` at
  step [05] with `AttributeError`.

### Added
- `DCSK_DEFAULT_WC` exposed at package top level and in `__all__`.
- i18n support for sweep classification legends via `_build_legend_handles(lang)` and
  new `_i18n` keys (`sweep.legend.periodic`, `.chaotic`, `.unbounded`).
- `FixedPointInfo` TypedDict for `fixed_point_stability` return type.
- Type hints (`NDArray`, `SpectralConfig`) on `compute_psds` in `cli/_common.py`.
- Regression test `test_to_namespace_has_all_required_fields` guarding against future
  `to_namespace` field omissions.
- Documentation of Pecora-Carroll vs DCSK BER comparison asymmetry in `cli/dcsk.py`.

### Changed
- `maps.py` module header updated from "Three 2-D maps" to "Four Henon map variants and
  one chaotic-sequence generator".
- `test_cli.EXPERIMENTS` extended with `dcsk` and `analysis` subcommands.
- Phantom `--save` and `--no-display` flags removed from `sweep/_beta.py`,
  `sweep/_plot.py`, and `sweep/_plot_3d.py` (they were no-ops).
- `transmit_order_n` and `receive_order_n` docstrings clarified: both explicitly state
  they do NOT implement the `Transmitter`/`Receiver` protocols.
- `to_namespace()` expanded with `seed` field; intentional omissions of `PlotConfig`,
  `SpectralConfig`, and `SweepConfig` documented.
- `run_all.py` references `DCSK_DEFAULT_WC` and `cfg.comm.mu` instead of hardcoded
  magic numbers; duplicate local `DEFAULT_CONFIG` import removed.
- `plotting/` restored to coverage measurement (was excluded despite `test_plotting.py`).
- `DEFAULT_CONFIG` singleton documented as read-only with `dataclasses.replace()` guidance.
- CI `--cov-fail-under` raised from 65 to 75.
- CI pipeline smoke test upgraded from Python 3.12 to 3.14.
- `RELEASING.md` updated to include `docs/conf.py` in version bump checklist.
- Makefile `.PHONY` completed with all targets (`format-check`, `docs-pt`, etc.).
- Test class inheritance standardized to `(_IsolatedCwdMixin, unittest.TestCase)` across
  all smoke test classes.
- `test_transmit_diverges_with_large_mu` moved from `TestTransmitOrderN` to
  `TestTransmitStandard`.
- Sphinx `.po` files regenerated via `sphinx-intl update`; 92 fuzzy entries resolved.
- Stale translations fixed: symbol count (62), `default_rng(seed)`, channel module paths.
- README `__all__` count corrected from `~65` to `62`.
- `docs/contributing.rst` symbol count corrected from `61` to `62`.

### Removed
- All em-dashes (travessões) from documentation: replaced with colons, commas, or
  semicolons. Applies to all `.rst`, `.md`, and source docstrings.

## [0.6.2] - 2026-05-10

### Fixed
- TestPyPI publish rejected: filename previously used and deleted. Patch bump only.

*Note: version 0.6.1 was a failed publish to TestPyPI (rejected filename) and was never released.*

## [0.6.0] - 2026-05-10

### Added
- `docs/background.rst` expanded with full theoretical foundations from the PFC article:
  physical-layer security, FIR filtering, Lyapunov exponents, Pecora-Carroll synchronisation,
  DCSK/EF-DCSK/CSK modulation, BER/SNR metrics, parameter dependence, and 14 references.
- `docs/internals.rst`: Numba kernel architecture, MGS, adaptive early-stop, FIR bank
  precomputation, fixed-point stability, signal generators, PSD estimation, channel models.
- `docs/development.rst`: environment setup, tooling, testing, CI/CD, documentation build
  workflow (including i18n gettext/update-po), and code style conventions.
- `docs/contributing.rst`: philosophy, contribution workflow, commit format, docstring
  conventions, public API rules, and pull request checklist.
- `docs/architecture.rst` expanded with full package tree, communication pipeline diagram,
  parameter sweep pipeline, 7 design decisions, and subpackage responsibility table.
- `docs/usage.rst` expanded with index, all CLI subcommands with examples, adaptive sweep
  options, and language support section.
- Bilingual translations (pt_BR) for all hand-written documentation pages: index,
  background, architecture, usage, internals, development, and contributing.
- `chaotic_pfc._i18n` module: bilingual figure labels (pt / en) with dictionary-based
  lookup, controllable via `CHAOTIC_PFC_LANG` env var or `--lang` CLI flag.
- `--lang pt|en` CLI flag for attractors, sensitivity, comm-*, dcsk, and run-all.
- `docs/_redirect.html`: RTD language landing page.

### Changed
- `docs/conf.py` refactored: full i18n/l10n, clean Furo sidebar, intersphinx, copybutton,
  xelatex for Unicode PDF support.
- `docs/Makefile` rewritten with `html`, `html-pt`, `html-all`, `pdf`, `pdf-pt`, `epub`,
  `epub-pt`, `gettext`, and `update-po` targets.
- Root `Makefile` updated with `docs-all`, `docs-pdf`, `docs-pdf-pt`, `docs-epub`,
  `docs-epub-pt` targets.
- `.readthedocs.yaml` simplified to default Sphinx builder with `htmlzip` format.
- `.gitignore`: added `_build/`, `_readthedocs/`, and `*.mo` patterns.
- `pyproject.toml` docs extra: removed unused `sphinxcontrib-bibtex`.
- `README.md` and `README_pt-BR.md` synchronised.
- API reference `.po` files removed: auto-generated docstrings stay in English.

### Fixed
- `run_all` not passing `lang` attribute to experiment subcommands (`AttributeError`).
- RST substitution warnings in docstrings (`|x|` → `\\|x\\|` in sweep_plotting, dcsk,
  run_all; title underline in spectral).
- LaTeX PDF build failures: switched to xelatex for Unicode characters in docstrings,
  replaced Unicode box-drawing in architecture diagram with plain ASCII.
- Placeholder `FIRST AUTHOR <EMAIL@ADDRESS>` in `.po` headers.
- Broken `spectral.py` entry in `api/index.po`.

## [0.5.0] - 2026-05-07

### Added
- Numba made optional via `_compat.py` fallback layer (`[fast]` extra, `pip install chaotic-pfc[fast]`).
- `Transmitter`, `Channel`, `Receiver` Protocols in `comms/protocols.py`.
- `PlotGridOptions` dataclass for `plot_comm_grid`.
- `.codecov.yml` with 5% threshold (warn -2%, fail -5%).
- `.github/dependabot.yml` with grouped weekly updates for pip and GitHub Actions.
- `.github/ISSUE_TEMPLATE/` (bug report and feature request).
- `RELEASING.md` with step-by-step release process.
- TestPyPI CD: `release` job in `ci.yml` using OIDC trusted publishing (triggered on `git tag v*`).
- Dual-language README: `README.md` (EN) + `README_pt-BR.md` with language switcher.
- Hero figure in README (Lyapunov classification map).
- `py.typed` marker (PEP 561).
- `chaotic_pfc.__version__` attribute.
- `CHANGELOG.md` following Keep a Changelog.
- `CONTRIBUTING.md` with development workflow and project conventions.
- `Makefile` with 13 targets.
- `strict_markers = true` in pytest config.
- Python 3.14 in CI test matrix.

### Changed
- Minimum Python bumped from 3.10 to 3.11.
- CI: sequential gate (`lint ∥ typecheck` → `test` → `pipeline` → `docs`). PRs gate at test only.
- CI: `--cov-fail-under=55` quality gate.
- `henon_order_n`: `fir_coeffs` is now keyword-only.
- DCSK transmit functions share `_chaos_sequence` helper.
- Adaptive Lyapunov early-stop extracted into `_adaptive_checkpoint`.
- CLI `comm_*` modules share `compute_psds` and `save_or_show`.
- `plotly` import is now lazy (`_get_go()`): package imports without plotly installed.
- `aggregate_beta_sweeps` / `plot_3d_beta_volume` removed from top-level `__init__`;
  import directly from `chaotic_pfc.analysis.sweep_plotting_3d`.
- `pyproject.toml`: classifiers and Documentation URL added.
- `analysis_summary.json` default path moved to `data/`.

### Fixed
- `PlotGridOptions.time_window` uses `default_factory` for Python 3.11 compatibility.
- Duplicate `if:` conditions in CI pipeline and docs jobs.
- CodeQL double-import warning in `test_cli_smoke.py`.

## [0.4.0] - 2026-05-04

### Added
- Comprehensive statistical sweep analysis suite: filter-type comparison, lambda_max
  distributions, transition boundaries, spectral robustness, Spearman correlation,
  bootstrap confidence intervals, parameter ranking, beta-evolution curves, and interpretation.
- DCSK and EF-DCSK chaotic communication modules with 4 channel models (AWGN, impulsive,
  multipath, urban interferers).
- Bandpass and bandstop filter types with bandwidth parameter; `FILTER_TYPES` expanded to 4.
- `chaotic-pfc run analysis` CLI subcommand.
- `chaotic-pfc run dcsk` CLI subcommand for BER-vs-SNR comparison.
- `chaotic-pfc run sweep beta-sweep` for Kaiser beta sweeps.
- `chaotic-pfc run sweep plot-3d` for Plotly 3-D visualisation.
- `chaotic-pfc run sweep compute --bandwidth` flag.
- `chaotic-pfc run all --adaptive` with `--Nmap-min` and `--tol`.
- `py.typed` marker (PEP 561).
- `chaotic_pfc.__version__`.
- `CHANGELOG.md` following Keep a Changelog.
- `CONTRIBUTING.md`.
- `Makefile` with 13 targets.
- `strict_markers = true` in pytest config.
- `scripts/benchmark.py`: performance benchmarks for Henon maps, FIR bank, and Lyapunov exponents.
- `PlotGridOptions` dataclass as a typed alternative to `plot_comm_grid` keyword arguments.
- `ExperimentConfig.to_namespace()`: generates `argparse.Namespace` from config defaults,
  eliminates `_fill_config_defaults`.
- TypedDict definitions in `stats.py`: `SummaryRow`, `FilterTypeAggregate`, `OptimalParams`,
  `LmaxDistribution`, `CorrelationMatrix`, `BootstrapConfidence`.
- `tests/_test_helpers.py` with shared `make_fir_coeffs` and `assert_seed_determinism`.
- 18 new tests.

### Changed
- Sources reorganised into 4 subpackages: `dynamics/`, `comms/`, `analysis/`, `plotting/`.
- `analysis.py` renamed to `stats.py`; `plotting.py` renamed to `figures.py`.
- `sweep.py` (1139 lines) split into `_types`, `_kernel`, `_orchestration`, `_io` submodules.
- `test_maps.py` split into `test_maps.py`, `test_signals.py`, `test_lyapunov.py`.
- CLI output and `print()` statements translated to English (figure titles kept in
  Portuguese for the academic article).
- `henon_order_n`: `fir_coeffs` is now keyword-only (`*` marker in signature).
- DCSK `dcsk_transmit`/`efdcsk_transmit` share a `_chaos_sequence` helper.
- Adaptive Lyapunov early-stop block extracted into `_adaptive_checkpoint`.
- CLI `comm_ideal`, `comm_fir`, `comm_order_n` share `compute_psds` and `save_or_show`.
- `dcsk.py` CLI uses `add_save_display_flags` for consistency.
- `_save()` in `plotting/figures.py` now creates parent directories.
- `_coeffs()` removed from `test_transmitter` and `test_receiver`; replaced by shared `make_fir_coeffs`.
- `plotly` import in `sweep_plotting_3d.py` is now lazy (`_get_go()`).
- `aggregate_beta_sweeps` and `plot_3d_beta_volume` removed from top-level `__init__.py`.
- `coverage.run.omit` updated from dead `*/plotting.py` to `*/plotting/*`.
- CI test job now uses a Python version matrix `["3.10", "3.12"]`.
- Pre-commit config expanded.
- README updated with new package structure and a "Public API" section.
- `analysis_summary.json` output path moved to `data/` and added to `.gitignore`.
- `run_all.py` step 08 now plots all window×filter combinations.

### Fixed
- Duplicate test method names in `test_sweep_plotting.py`.
- `FILTER_TYPES` hardcoded 5 times in `cli/analysis.py`: now uses the constant from `analysis.sweep`.
- Wrong expected value in `TestHenonStandard.test_first_iteration`.
- Wrong fixed-point assertion in `TestLyapunov.test_henon2d_fixed_points`.
- Flaky `TestLyapunovEnsemble.test_chaotic_average`.
- Outdated `:mod:` references in `sweep_plotting.py`, `config.py`, `channel.py`, `receiver.py`.
- Outdated `Originally scripts/...` comments in 7 CLI modules.
- Outdated module paths in `docs/api/index.rst`.
- Build artifacts (`.egg-info/`, `__pycache__/`) removed from version control.
- Leftover `run_all.py` in project root removed.

## [0.3.0] - 2026-05-03

### Added
- Kaiser beta-sweep in the Lyapunov exponent pipeline with interactive 3-D Plotly plots.
- Adaptive early-stop for Lyapunov kernels: convergence check every 100 iterations.
- Lyapunov ensemble protocol: `N_ci` ICs sampled uniformly in +/-perturbation around the
  fixed point, with per-IC CSV export.
- Sweep plotting: `lambda_max == 0` classified as periodic, difficulty map for adaptive sweeps.
- Smoke tests for all CLI subcommands.
- Sphinx documentation with Furo theme, auto-generated API reference from NumPy docstrings.
- `.readthedocs.yaml` for Read the Docs hosting.
- `requirements-lock.txt` for byte-exact CI reproducibility.
- Codecov coverage upload in CI with matching badge.
- BSD 3-Clause LICENSE and AUTHORS file.

### Changed
- CLI unified: standalone scripts (01–08 + run_all.py) replaced by `chaotic-pfc run <subcommand>`.
- Codebase reformatted and linted with Ruff; type-checked with mypy.
- Pre-commit hooks added.
- NumPy-style docstrings adopted across the entire library.
- Sweep kernel refactored: in-place buffers, merged n1–n4 into two regimes (n12, nN),
  prange load balancing via round-robin task ordering.
- Sweep kernel made deterministic under `np.random.seed`.
- CI split into parallel jobs: lint, typecheck, test, pipeline smoke test, docs build.
- Sweep plots now generate both PNG and SVG by default.
- `henon_processar.py` extracted into `sweep.py` + `sweep_plotting.py` modules.
- Coverage rose from 56% to 62%; 8 modules at 100%.

### Fixed
- Build artifacts (`.egg-info/`, `__pycache__/`) removed from version control.
- Leftover `run_all.py` in project root removed.

## [0.2.0] - 2026-04-22

### Added
- Initial release: Henon map variants, FIR channel models, Pecora-Carroll synchronisation.
- Lyapunov exponent computation (single IC) and parameter sweep over `(order, cutoff)` grid.
- Basic CLI scripts (01–08) for each experiment step.

[Unreleased]: https://github.com/fesa-academic-projects/chaotic-pfc/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.8.0
[0.7.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.7.0
[0.6.2]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.6.2
[0.6.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.6.0
[0.5.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.5.0
[0.4.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.4.0
[0.3.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.3.0
[0.2.0]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/v0.2.0

# Contributing to chaotic-pfc

This document describes the development workflow, project conventions, and tooling used in this repository. It applies to every contributor: human or automated: and exists to keep changes disciplined, reviewable, and traceable.

The conventions here bias toward caution over speed. For trivial tasks (typo fixes, comment edits, one-line tweaks), use judgment.

## 1. Development workflow

### Think before coding

Before implementing anything non-trivial:

- State assumptions explicitly. If something is uncertain, ask before building on it.
- If a request admits multiple interpretations, surface them: don't pick silently.
- If a simpler approach exists than what was asked for, mention it. Push back when warranted.
- If something in the existing code is unclear, stop and name what's confusing rather than guessing.

### Simplicity first

Write the minimum code that solves the problem at hand. Specifically:

- No features beyond what was asked for.
- No abstractions for single-use code.
- No "flexibility" or configurability that wasn't requested.
- No error handling for scenarios that cannot occur.

If you find yourself writing 200 lines for what should be 50, stop and rewrite. The question to ask is whether a senior engineer reviewing the diff would call it overcomplicated. If yes, simplify.

### Surgical changes

Touch only what the task requires. When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even when you'd write it differently personally.
- If you notice unrelated dead code, mention it: don't delete it as part of an unrelated change.

When your edits create orphans (unused imports, dangling helpers, etc.), clean up only the orphans your changes produced. Leave pre-existing dead code alone unless removing it is the explicit goal of the commit.

The test for whether a diff is surgical: every changed line should trace directly to the stated goal.

### Goal-driven execution

Transform tasks into verifiable goals before starting:

- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure the existing tests pass before and after, with no behavioural change."

For multi-step tasks, write a brief plan with explicit checks:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let work proceed independently. Weak criteria ("make it work") require constant clarification and lead to wasted effort.

## 2. Tooling and code generation

Parts of this codebase were developed with LLM assistance, primarily for porting numerical routines from MATLAB and Jupyter notebooks into a packaged Python library, and for accelerating prototyping during structural refactors. All generated code was reviewed, tested, and integrated by the human authors listed in `AUTHORS`.

The conventions in section 1 exist in part because that workflow benefits from explicit discipline: assumptions get stated, changes stay surgical, and verification happens before merge. They are not LLM-specific: they are the same conventions a careful human contributor would follow: but they were written down rather than left implicit because part of the team is a model that does not infer unwritten norms.

The standard toolchain for this project is described in section 3. Whatever tools you use to write code, the output must pass that toolchain before being committed.

## 3. Project conventions

### Language

Code, identifiers, docstrings, and the public API are in English. Figure titles and a few CLI messages are kept in Portuguese where they feed directly into the academic article that accompanies this project. Commit messages are in English.

### Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/). The commit history is part of the deliverable and is expected to read as a coherent narrative of the project's evolution.

Common prefixes used here: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`. Use the imperative mood ("add X", not "added X") and keep the subject line under ~70 characters. Use the body for context and rationale when the change isn't self-explanatory.

### Code style and static checks

Formatting and linting are enforced by [Ruff](https://docs.astral.sh/ruff/). Static type checking is enforced by [mypy](https://mypy-lang.org/). Both are configured in `pyproject.toml` and run as pre-commit hooks and in CI.

To install the hooks locally:

```bash
pre-commit install
```

To run all checks manually:

```bash
pre-commit run --all-files
```

The same checks run in CI on every push and pull request. Local pre-commit is a convenience for catching problems before pushing, not a replacement for the CI gate.

### Docstrings

Public functions, classes, and modules use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). They are consumed by Sphinx with the `numpydoc` extension to build the documentation hosted on Read the Docs.

### Tests

Every new module under `src/chaotic_pfc/` ships with a corresponding test file under `tests/`. Numba-compiled kernels are tested directly (not only through their orchestrators): see `tests/test_sweep.py` for the pattern.

Shared test utilities live in `tests/_test_helpers.py`. When a helper (fixture, coefficient builder, assertion) is needed by more than one test module, extract it there rather than duplicating it.

Run the suite with:

```bash
pytest
```

Slow tests are marked with `@pytest.mark.slow` and can be skipped with `pytest -m "not slow"`. CI runs the full suite plus an end-to-end pipeline smoke test (`chaotic-pfc run all --quick-sweep --no-display`).

### Benchmarking

A lightweight benchmark script lives at `scripts/benchmark.py`. It measures wall-clock time for core operations (Henon maps, FIR bank, Lyapunov exponents) and serves as a sanity check that performance hasn't regressed after refactors. Run it before tagging a release:

```bash
python scripts/benchmark.py
```

### Public API

The canonical public API is what `from chaotic_pfc import ...` exposes via `chaotic_pfc/__init__.py`. Anything reached via a leading-underscore submodule (e.g. `chaotic_pfc.analysis.sweep._kernel`) is implementation detail and may change without notice.

When adding a function intended for public use, expose it through the appropriate subpackage `__init__.py` and include it in the top-level reexport. When adding implementation helpers, prefix them with an underscore and do not reexport them.

### Versioning and changelog

The project follows [Semantic Versioning](https://semver.org/). The version is set in `pyproject.toml` and exposed at `chaotic_pfc.__version__`.

User-visible changes are recorded in `CHANGELOG.md` under an `[Unreleased]` section while in development. On release, that section is renamed to the version and date being tagged, and a fresh empty `[Unreleased]` is opened.

### Optional dependencies

Heavy or specialised dependencies (currently Plotly for 3-D visualisation) are declared as optional extras in `pyproject.toml` and must not be imported at package-import time. Code that depends on an optional extra is reachable only through a deeper import path; importing the top-level `chaotic_pfc` package must succeed without any extras installed.

---

These conventions are working if: diffs contain only changes that trace to the task, regressions are caught by tests rather than by users, and clarifying questions arrive before implementation rather than after mistakes.

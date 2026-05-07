.. _contributing:

Contributing
============

Thank you for your interest in contributing to ``chaotic-pfc``. This
page outlines the contribution workflow and project conventions.

For environment setup and tooling details, see :doc:`development`.

Philosophy
----------

**Think before coding.** Before implementing, state your assumptions,
surface multiple interpretations of the task, and clarify any ambiguity.
If a simpler alternative exists, mention it before reaching for
complexity.

**Simplicity first.** Write the minimum code necessary:

* No features that weren't requested.
* No single-use abstractions ("in case we need it later").
* No spurious configurability for hypothetical scenarios.
* No error handling for states proven impossible by the call chain.

If a 200-line implementation can become 50 lines with the same
correctness, rewrite it. The test: would a senior engineer call the
result overcomplicated?

**Surgical changes.** Touch only what the task requires:

* Do not improve adjacent code you were not asked to change.
* Do not refactor unbroken code.
* Match the existing style even if you would write it differently.
* Dead code unrelated to the task: mention it, do not delete it.
* Clean up only orphans your own changes created.

**Goal-driven execution.** Transform tasks into verifiable goals. For
multi-step tasks, write a plan with explicit verification checks
before writing code.

Workflow
--------

.. code-block:: bash

    # 1. Fork the repository on GitHub

    # 2. Create a feature branch
    git checkout -b feat/my-feature

    # 3. Make changes, following the code style conventions

    # 4. Run the quality suite
    make check-all

    # 5. Commit with Conventional Commits format
    git commit -m "feat: add support for complex filter coefficients"

    # 6. Push and open a pull request
    git push -u origin feat/my-feature

Commit message format
---------------------

This project follows **Conventional Commits**:

.. code-block:: text

    <type>: <short description>

    <optional body with longer explanation>

Types:

* ``feat`` — new feature.
* ``fix`` — bug fix.
* ``refactor`` — code restructuring without behaviour change.
* ``test`` — adding or updating tests.
* ``docs`` — documentation changes.
* ``chore`` — maintenance (dependency updates, tooling config).
* ``ci`` — CI/CD configuration.

Guidelines:

* Use imperative mood ("add" not "added" or "adds").
* Subject line under ~70 characters.
* Separate subject from body with a blank line.
* Language: English.

Code standards
--------------

Before submitting a pull request, ensure:

.. code-block:: bash

    # Formatting passes
    make format-check

    # Linter is clean
    make lint

    # Type checking passes
    make typecheck

    # All tests pass (including slow tests)
    make test

    # Smoke test passes
    chaotic-pfc run all --no-display --quick-sweep

All checks are enforced by CI. The pull request will not be merged
until the CI pipeline is green.

Docstrings
----------

Write all docstrings in **NumPy style**:

.. code-block:: python

    def function_name(param1: int, param2: str) -> float:
        """One-line summary.

        Extended description (optional but encouraged for public API).

        Parameters
        ----------
        param1 : int
            Description of param1.
        param2 : str
            Description of param2.

        Returns
        -------
        float
            Description of return value.

        Raises
        ------
        ValueError
            If param1 is negative.

        Notes
        -----
        Any implementation notes, references, or equations.

        Examples
        --------
        >>> function_name(1, "test")
        0.5
        """

Classes and dataclasses use the ``Attributes`` section:

.. code-block:: python

    class MyClass:
        """One-line summary.

        Extended description.

        Attributes
        ----------
        field1 : int
            Description.
        """

Guiding principles:

* Public functions MUST have docstrings.
* Internal/private functions SHOULD have docstrings.
* Keep the one-line summary under 80 characters.
* Use backticks for code references.
* Use ``:class:``, ``:func:``, ``:mod:`` for Sphinx cross-references.
* Avoid vague language like "process the data".

Public API
----------

The canonic public API is defined in
:mod:`chaotic_pfc.__init__` via the ``__all__`` list (~70 symbols).
All names in ``__all__`` are re-exported at the top level:

.. code-block:: python

    from chaotic_pfc import henon_standard, transmit, receive, run_sweep

Modules whose names begin with ``_`` (e.g. ``_compat.py``,
``_kernel.py``) contain implementation details and may change without
notice. To add a new public function:

1. Define it in the appropriate subpackage module.
2. Import it in the subpackage's ``__init__.py``.
3. Import it in ``chaotic_pfc/__init__.py``.
4. Add it to ``__all__``.

Optional dependencies
---------------------

The project has four optional extras:

* ``[fast]`` — **numba** for JIT acceleration.
* ``[dev]`` — **pytest**, **ruff**, **mypy**, **pre-commit**.
* ``[docs]`` — **sphinx**, **furo**, **sphinx-copybutton**, **myst-parser**, **sphinx-intl**.
* ``[viz3d]`` — **plotly** for 3-D visualisation.

Optional modules must be importable without their extras. Use lazy
imports:

.. code-block:: python

    # Inside a function, not at module level:
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "3-D visualisation requires plotly. "
            "Install with: pip install chaotic-pfc[viz3d]"
        ) from None

Versioning
----------

This project follows **Semantic Versioning** (SemVer):

* **MAJOR** — incompatible API changes.
* **MINOR** — backwards-compatible new functionality.
* **PATCH** — backwards-compatible bug fixes.

The version is stored in two places:

* ``pyproject.toml`` — ``project.version`` field.
* ``src/chaotic_pfc/_version.py`` — ``__version__`` string.

The ``[Unreleased]`` section of ``CHANGELOG.md`` is renamed to the
version number and date on release. See ``RELEASING.md`` in the
repository root for the full release process.

Issue reporting
---------------

Use GitHub Issues for:

* Bug reports — include reproduction steps, expected behaviour, and
  environment details (Python version, OS, Numba version).
* Feature requests — describe the use case and proposed solution.
* Documentation issues — point to the specific page and section.

Before opening a new issue, search existing issues to avoid
duplicates.

Pull request checklist
----------------------

* All tests pass (``make test``).
* Fast tests pass (``make test-fast``).
* Linter is clean (``make lint``).
* Formatting is correct (``make format-check``).
* Type checking passes (``make typecheck``).
* New functions have NumPy-style docstrings.
* New public functions are added to ``__all__``.
* CHANGELOG.md is updated under ``[Unreleased]``.
* Commit messages follow Conventional Commits.

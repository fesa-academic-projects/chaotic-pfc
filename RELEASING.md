# Releasing a new version

## Prerequisites

- Push access to the repository and a signing key configured for `git tag -s`.
- Trusted publishers configured on **[TestPyPI](https://test.pypi.org) and
  [PyPI](https://pypi.org)** for this repository (one-time, manual; see below).
- Maintainer access to the Zenodo GitHub integration (for the data DOI).
- Write access to the Read the Docs project (to activate the `stable` version).

### One-time PyPI Trusted Publishing (manual)

Both the TestPyPI and the PyPI projects need a **Trusted Publisher** pointing at
this repository (workflow `ci.yml`, environment left empty):

1. On [PyPI](https://pypi.org) create the `chaotic-pfc` project and add a
   pending publisher: owner `fesa-academic-projects`, repository `chaotic-pfc`,
   workflow `ci.yml`, environment *(empty)*.
2. Repeat on [TestPyPI](https://test.pypi.org).
3. No API token is stored: the `release-testpypi` and `release-pypi` jobs
   authenticate through the OIDC `id-token`.

## Steps

### 1. Update the changelog

In `CHANGELOG.md`, rename the `## [Unreleased]` section to the new version and date
(using ` - ` separator, per Keep a Changelog):

```markdown
## [X.Y.Z] - 2026-06-01
```

Open a fresh empty `## [Unreleased]` section above it. Add reference links at the
bottom matching the existing pattern:

```markdown
[X.Y.Z]: https://github.com/fesa-academic-projects/chaotic-pfc/releases/tag/vX.Y.Z
```

### 2. Bump the version

Update the version in two places: `pyproject.toml`

```toml
version = "X.Y.Z"
```

and `src/chaotic_pfc/_version.py`:

```python
__version__ = "X.Y.Z"
```

`docs/conf.py` reads the version from the package (`version = release = _pkg_version`),
so it must **not** be edited by hand. Then update `CITATION.cff`:

```yaml
version: X.Y.Z
date-released: "2026-06-01"
```

Commit all three with a message like `release: bump version to X.Y.Z`.

Before tagging, confirm the manuscript and data checks are green with the CLI
behind the unchanged `make` targets: `make check-paper` (runs
`chaotic-pfc paper audit`), `make jcis-figures` (runs `chaotic-pfc paper
figures`) leaving `git status` clean, and the sweep manifest with
`chaotic-pfc sweeps regenerate --manifest-only` plus the test suite.

### 3. Tag the release

Create an **annotated and signed** tag (`-s`, not `-a`) so it can be verified
later with `git tag -v vX.Y.Z`:

```bash
git tag -s vX.Y.Z -m "vX.Y.Z: <short description>"
git push origin vX.Y.Z
```

Pushing the tag triggers the release jobs in CI. They start only after the
`lint`, `typecheck`, `test` (the 3.13–3.14 matrix) and `baseline` jobs have
passed on the tag, so a tag can no longer publish untested code. Then:

1. `release-testpypi` builds nothing new — it reuses the `dist` artefact from
   the `build` job — and publishes the wheel and sdist to
   https://test.pypi.org.
2. `release-pypi` publishes the **same** artefacts to the real PyPI with OIDC
   Trusted Publishing (no token).

A **baseline failure** in a Dependabot `scientific-core` PR (numpy/scipy/numba)
is expected when the numerical bits change; regenerate the sweeps, compare them
and update the numbers, figures and `benchmarks/baseline/` in that same PR
before tagging (see CONTRIBUTING).

### 4. Verify on TestPyPI

Open https://test.pypi.org/project/chaotic-pfc and confirm the new version
is listed. Install it from TestPyPI as a smoke check:

```bash
pip install --index-url https://test.pypi.org/simple/ chaotic-pfc
```

### 5. Create a GitHub Release

1. Go to **Releases → Draft a new release**.
2. Choose the tag just pushed (`vX.Y.Z`).
3. Title: `vX.Y.Z`.
4. Body: paste the relevant section from `CHANGELOG.md`.
5. Mark as **pre-release** (until 1.0.0).
6. Publish.

### 6. Archive on Zenodo

The Zenodo GitHub integration mints a DOI for the release. The concept DOI
(`10.5281/zenodo.22740317`) always resolves to the latest version; each release
gets its own version DOI. Record the new version DOI.

### 7. Record the DOIs

- `CITATION.cff`: add the new version DOI to `identifiers` (keep the concept DOI)
  and keep `version` / `date-released` in sync.
- `docs/citing.rst`: add the new version DOI next to the concept DOI.
- `articles/jcis/references.bib`: set the version DOI of the software entry
  (the `chaotic-pfc` `@misc` record, key `chaoticPFC2026`) to the DOI of the
  released version.

### 8. Rebuild the manuscript PDF

`articles/jcis/article.pdf` is versioned and rebuilt by `make paper-pdf`, which
compiles it inside the pinned image `texlive/texlive@sha256:acd65bd2…`
(TeX Live 2026; Docker or Podman is required, and `make setup` checks for it).
Use `make paper-pdf-local` only for a quick preview with the system TeX Live (it
writes to `articles/jcis/build/`, never versioned). The CI job *Manuscript PDF
reproducibility* rebuilds the PDF in the same pinned image and requires it to be
**byte-identical** to the committed file. Other TeX Live distributions are not
byte-identical (font-package and `ToUnicode` differences), so always regenerate
with `make paper-pdf` before tagging so the versioned PDF matches the tagged
text.

### 9. Read the Docs

In the Read the Docs project, activate the new tag as the **`stable`** version
(and keep `latest` on `main`). Confirm the build for the tag is green and that
https://chaotic-pfc.readthedocs.io/en/stable/ serves the released documentation.


## Initial configuration (one-time)

Repository settings and external accounts configured once, before the first
release from this branch. Nothing here is automated.

### Repository protection (GitHub)

- Branch protection on `main`: require a pull request and at least one approval,
  require the status checks `Build distribution`, `Lint (Ruff)`,
  `Type check (mypy)`, `Unit tests (pytest …)`, `Minimum versions (SPEC 0)`,
  `Numba baseline regression`, `Pure-Python fallback (no Numba)`,
  `Contract fuzzing (pure Python)`, `RuntimeWarning gate`,
  `Static analysis (Semgrep)`, `Manuscript audit (check-paper)`,
  `Manuscript tables (LaTeX)`, `REUSE compliance` and `CodeQL`, and
  disallow force pushes.
- Import the versioned rulesets instead of configuring the rules by hand:
  `.github/rulesets/main.json` protects the default branch (signed commits,
  linear history, no deletion or force push, required checks) and
  `.github/rulesets/tags.json` protects the `v*` release tags. Apply them with
  the GitHub CLI:

  ```bash
  gh api --method POST /repos/fesa-academic-projects/chaotic-pfc/rulesets \
    --input .github/rulesets/main.json
  gh api --method POST /repos/fesa-academic-projects/chaotic-pfc/rulesets \
    --input .github/rulesets/tags.json
  ```

  The import requires repository-admin rights, so it is a manual step after
  merging. Re-read the ruleset list with
  `gh api /repos/fesa-academic-projects/chaotic-pfc/rulesets` to confirm both
  are active.
- The pre-commit hooks `frozen-artifacts` and `artefact-manifests` reject
  staged edits under `data/sweeps/`, `articles/simac/` and
  `articles/jcis/Figures/` and verify the `SHA256SUMS` manifests. A deliberate
  regeneration must be committed with `CHAOTIC_PFC_ALLOW_FROZEN=1`.
- Confirm the `CODEOWNERS` team (`@fesa-academic-projects/maintainers`) exists
  and matches `.github/CODEOWNERS`.
- Enable Dependabot alerts and security updates.

### PyPI and TestPyPI (Trusted Publishing)

Register a pending publisher on both indexes (no API token is stored): project
`chaotic-pfc`, owner `fesa-academic-projects`, repository `chaotic-pfc`,
workflow `ci.yml`, environment empty. The `release-testpypi` and `release-pypi`
jobs use OIDC `id-token: write` and publish on `v*` tags, in that order.

### Zenodo and DOI

Enable the GitHub–Zenodo integration for the repository and keep `main` in the
release branch list. After the GitHub release, record the version DOI in
`CITATION.cff`, `codemeta.json`, `docs/citing.rst` and the software entry of
`articles/jcis/references.bib`.

### Read the Docs

Activate the new tag as the `stable` version and keep `latest` on `main`;
confirm the built documentation is green.

### OpenSSF Best Practices badge

Register the project and complete the questionnaire (aim for the "passing"
badge).

### Re-sign the commits and push

The commits in this branch are unsigned by design. Before publishing, re-sign
them and push:

```bash
git fetch origin
git rebase --exec 'git commit --amend --no-edit -S' origin/main
git push origin main
```

Watch the first full CI run and fix anything that only appears in CI before
tagging.

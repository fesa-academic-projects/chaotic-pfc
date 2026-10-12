# Releasing a new version

## Prerequisites

- Push access to the repository and a signing key configured for `git tag -s`.
- Trusted publisher configured on [TestPyPI](https://test.pypi.org) for this repo.
- Maintainer access to the Zenodo GitHub integration (for the data DOI).
- Write access to the Read the Docs project (to activate the `stable` version).

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

Pushing the tag triggers the `release` job in CI. It starts only after the
`lint`, `typecheck`, `test` (the full 3.12–3.14 matrix) and `baseline` jobs have
passed on the tag, so a tag can no longer publish untested code. The `release`
job then:

1. Builds a wheel (`.whl`) and source distribution (`.tar.gz`).
2. Publishes both to https://test.pypi.org.

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
  (the `chaotic-pfc` `@software` record) to the DOI of the released version.

### 8. Read the Docs

In the Read the Docs project, activate the new tag as the **`stable`** version
(and keep `latest` on `main`). Confirm the build for the tag is green and that
https://chaotic-pfc.readthedocs.io/en/stable/ serves the released documentation.

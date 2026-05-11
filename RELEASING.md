# Releasing a new version

## Prerequisites

- Push access to the repository.
- Trusted publisher configured on [TestPyPI](https://test.pypi.org) for this repo.

## Steps

### 1. Update the changelog

In `CHANGELOG.md`, rename the `## [Unreleased]` section to the new version and date:

```markdown
## [X.Y.Z]: 2026-06-01
```

Open a fresh empty `## [Unreleased]` above it.

### 2. Bump the version

In `pyproject.toml`, update the `version` field:

```toml
version = "X.Y.Z"
```

Update `src/chaotic_pfc/_version.py`:

```python
__version__ = "X.Y.Z"
```

And `docs/conf.py`:

```python
version = "X.Y.Z"
release = "X.Y.Z"
```

Commit all three with a message like `release: bump version to X.Y.Z`.

### 3. Tag the release

```bash
git tag -a vX.Y.Z -m "vX.Y.Z: <short description>"
git push origin vX.Y.Z
```

Pushing the tag triggers the `release` job in CI, which:
1. Runs lint + typecheck + tests.
2. Builds a wheel (`.whl`) and source distribution (`.tar.gz`).
3. Publishes both to https://test.pypi.org.

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

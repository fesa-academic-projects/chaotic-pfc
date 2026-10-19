## Summary

<!-- What changed, and why. Link the issue if there is one. -->

## Checklist

- [ ] `make check-all` passes (lint, format, typecheck, fast tests)
- [ ] Tests cover the change (or the reason they cannot is explained)
- [ ] Documentation updated when behaviour changed (docstrings, `docs/`, README)
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` for user-visible changes
- [ ] pt-BR catalogues refreshed for changed narrative pages
      (`cd docs && make gettext && make update-po`)
- [ ] No versioned data changed unintentionally
      (`data/sweeps/SHA256SUMS` refreshed by `chaotic-pfc sweeps regenerate` if so)

## Notes for the reviewer

<!-- Anything that deserves a closer look, or trade-offs you considered. -->

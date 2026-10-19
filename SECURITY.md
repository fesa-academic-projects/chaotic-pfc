# Security policy

## Supported versions

Only the latest minor release receives security fixes.

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| 0.8.x   | :x:                |
| < 0.8   | :x:                |

## Threat model

The library's assets, trust boundaries, threats, mitigations and explicit
non-goals are documented in
[`docs/security.rst`](docs/security.rst). The main points for a user of the
API:

- public parameters are validated at the boundary and raise
  `InvalidParameterError` (a `ValueError` subclass);
- divergence is explicit: `on_divergence="nan" | "warn" | "raise"`;
- checkpoints are opened with `allow_pickle=False`; legacy pickled metadata is
  only read after SHA-256 verification against a repository manifest, or with
  `trust_legacy_pickle=True`;
- the Pecora-Carroll chaotic scheme is **not** a replacement for standard
  cryptography.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Report it
privately using GitHub's
[security advisories](https://github.com/fesa-academic-projects/chaotic-pfc/security/advisories/new)
("Report a vulnerability"), or by e-mail to a maintainer listed in
[AUTHORS](AUTHORS).

Include, as far as possible:

- a description of the issue and its impact;
- the affected version(s) and platform;
- a minimal reproduction;
- any suggested fix or mitigation.

## Response times

| Stage | Target |
| ----- | ------ |
| Acknowledge the report | 3 business days |
| Initial assessment and severity | 10 business days |
| Fix or documented mitigation for critical and high issues | 30 days |
| Coordinated advisory | with the fix, crediting the reporter unless they prefer otherwise |

Critical issues are handled in a private branch, released as a patch on the
supported minor line, and published together with a GitHub security advisory.

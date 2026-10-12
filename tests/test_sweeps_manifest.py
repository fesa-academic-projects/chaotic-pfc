"""Checksum manifest for the versioned sweep checkpoints.

``data/sweeps/SHA256SUMS`` pins the sha256 of every ``variables_lyapunov.npz``,
so a corrupted or silently regenerated checkpoint is caught by the normal test
run (and therefore by CI, which executes the whole suite). The manifest is
rewritten by ``chaotic-pfc sweeps regenerate``.

These tests always run — unlike the opt-in bit-exact baseline in
``test_sweep_regressions.py`` (``CHAOTIC_PFC_RUN_BASELINE=1``). Set
``CHAOTIC_PFC_SWEEPS_DIR`` to point the check at a copy of the tree.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEPS_DIR = Path(os.environ.get("CHAOTIC_PFC_SWEEPS_DIR", REPO_ROOT / "data" / "sweeps"))
SWEEPS_DIR = SWEEPS_DIR.resolve()
MANIFEST = SWEEPS_DIR / "SHA256SUMS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest() -> dict[str, str]:
    if not MANIFEST.is_file():
        pytest.fail(f"missing checksum manifest: {MANIFEST}")
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator:
            pytest.fail(f"malformed manifest line: {line!r}")
        entries[relative] = digest
    return entries


def _checkpoints_on_disk() -> set[str]:
    return {path.relative_to(SWEEPS_DIR).as_posix() for path in SWEEPS_DIR.rglob("*.npz")}


def test_manifest_lists_every_checkpoint() -> None:
    """The manifest and the tree describe exactly the same set of checkpoints."""
    entries = set(_read_manifest())
    on_disk = _checkpoints_on_disk()
    assert entries == on_disk, (
        f"missing from SHA256SUMS: {sorted(on_disk - entries)}; "
        f"listed but absent: {sorted(entries - on_disk)}"
    )


def test_manifest_hashes_match() -> None:
    """Every checkpoint still hashes to the value recorded in the manifest."""
    mismatched = [
        relative
        for relative, digest in _read_manifest().items()
        if _sha256(SWEEPS_DIR / relative) != digest
    ]
    assert not mismatched, f"checkpoint(s) changed since SHA256SUMS: {mismatched}"

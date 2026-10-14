"""Checksum manifest for the frozen VIII SIMAC artefacts.

``articles/simac/SHA256SUMS`` pins the sha256 of the two PDFs (the official,
frozen record) and the two reconstructed ``.docx`` files, so a corrupted or
silently replaced artefact is caught by the normal test run (and therefore by
CI). The SIMAC record is frozen: these files must never change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMAC_DIR = REPO_ROOT / "articles" / "simac"
MANIFEST = SIMAC_DIR / "SHA256SUMS"
SUFFIXES = (".pdf", ".docx")


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
        digest, separator, name = line.partition("  ")
        if not separator:
            pytest.fail(f"malformed manifest line: {line!r}")
        entries[name] = digest
    return entries


def _artefacts_on_disk() -> set[str]:
    return {path.name for path in SIMAC_DIR.iterdir() if path.suffix.lower() in SUFFIXES}


def test_manifest_lists_every_artefact() -> None:
    """The manifest and the tree describe exactly the same set of artefacts."""
    entries = set(_read_manifest())
    on_disk = _artefacts_on_disk()
    assert entries == on_disk, (
        f"missing from SHA256SUMS: {sorted(on_disk - entries)}; "
        f"listed but absent: {sorted(entries - on_disk)}"
    )


def test_manifest_hashes_match() -> None:
    """Every SIMAC artefact still hashes to the recorded value (frozen)."""
    mismatched = [
        name for name, digest in _read_manifest().items() if _sha256(SIMAC_DIR / name) != digest
    ]
    assert not mismatched, f"SIMAC artefact(s) changed since SHA256SUMS: {mismatched}"

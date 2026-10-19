"""Tests for the repository guard rails used by pre-commit and CI."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "tools" / "check_frozen_artifacts.sh"
MANIFEST = REPO / "tools" / "check_sweeps_manifest.sh"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


def test_manifest_check_passes_on_the_versioned_tree() -> None:
    proc = subprocess.run([str(MANIFEST)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "artefact manifests OK" in proc.stdout


def test_frozen_hook_rejects_and_override_allows(tmp_path: Path) -> None:
    _git("init", "-q", cwd=tmp_path)
    frozen = tmp_path / "data" / "sweeps" / "probe"
    frozen.mkdir(parents=True)
    (frozen / "x.npz").write_bytes(b"probe")
    _git("add", "data/sweeps/probe/x.npz", cwd=tmp_path)

    blocked = subprocess.run(
        [str(FROZEN)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CHAOTIC_PFC_ALLOW_FROZEN": "0"},
    )
    assert blocked.returncode == 1
    assert "frozen artefact trees" in blocked.stderr
    assert "data/sweeps/probe/x.npz" in blocked.stderr

    allowed = subprocess.run(
        [str(FROZEN)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CHAOTIC_PFC_ALLOW_FROZEN": "1"},
    )
    assert allowed.returncode == 0


def test_ruleset_files_are_valid_json_with_required_rules() -> None:
    import json

    main = json.loads((REPO / ".github" / "rulesets" / "main.json").read_text(encoding="utf-8"))
    types = {rule["type"] for rule in main["rules"]}
    for required in (
        "deletion",
        "non_fast_forward",
        "required_linear_history",
        "required_signatures",
        "pull_request",
        "required_status_checks",
    ):
        assert required in types, f"main ruleset is missing {required}"
    contexts = {
        check["context"]
        for check in next(
            rule for rule in main["rules"] if rule["type"] == "required_status_checks"
        )["parameters"]["required_status_checks"]
    }
    for job in ("Lint (Ruff)", "Type check (mypy)", "Static analysis (Semgrep)"):
        assert job in contexts

    tags = json.loads((REPO / ".github" / "rulesets" / "tags.json").read_text(encoding="utf-8"))
    assert tags["target"] == "tag"
    assert any(ref.startswith("refs/tags/v") for ref in tags["conditions"]["ref_name"]["include"])

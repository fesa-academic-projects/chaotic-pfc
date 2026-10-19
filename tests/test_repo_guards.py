"""Tests for the repository guard rails used by pre-commit and CI."""

from __future__ import annotations

import os
import re
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


def _pull_request_without_paths(workflow: dict) -> bool:
    triggers = workflow.get("on", workflow.get(True))
    if triggers is None:
        return False
    if isinstance(triggers, str):
        return triggers == "pull_request"
    if isinstance(triggers, list):
        return "pull_request" in triggers
    if isinstance(triggers, dict):
        if "pull_request" not in triggers:
            return False
        config = triggers["pull_request"]
        return not (isinstance(config, dict) and ({"paths", "paths-ignore"} & set(config)))
    return False


def _expanded_job_names(workflow: dict) -> set[str]:
    import itertools

    names: set[str] = set()
    for job_id, job in (workflow.get("jobs") or {}).items():
        raw = job.get("name", job_id)
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        slots = re.findall(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}", raw)
        if not slots:
            names.add(raw)
            continue
        values = [matrix.get(slot) for slot in slots]
        if any(not isinstance(value, list) for value in values):
            names.add(raw)
            continue
        for combo in itertools.product(*values):
            name = raw
            for slot, value in zip(slots, combo, strict=True):
                name = name.replace(f"${{{{ matrix.{slot} }}}}", str(value))
            names.add(name)
    return names


def test_every_required_check_matches_a_pull_request_job() -> None:
    import json

    import yaml

    ruleset = json.loads((REPO / ".github" / "rulesets" / "main.json").read_text(encoding="utf-8"))
    required = {
        check["context"]
        for check in next(
            rule for rule in ruleset["rules"] if rule["type"] == "required_status_checks"
        )["parameters"]["required_status_checks"]
    }
    available: set[str] = set()
    for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        if _pull_request_without_paths(workflow):
            available |= _expanded_job_names(workflow)
    missing = sorted(required - available)
    assert not missing, (
        f"required checks without a matching job on pull_request without paths filters: {missing}"
    )


def test_supply_chain_tools_are_pinned_outside_the_lock() -> None:
    """Semgrep runs from an isolated exact version and gitleaks is checksum-pinned."""
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert not any(spec.startswith("semgrep") for spec in dev), (
        "semgrep must not be a development dependency"
    )
    lock = (REPO / "requirements-lock.txt").read_text(encoding="utf-8")
    assert "semgrep==" not in lock
    assert "mcp==" not in lock

    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uvx semgrep==1.181.0" in ci
    assert "sha256sum -c -" in ci
    assert "gitleaks git --redact --exit-code 1" in ci
    assert "gitleaks-action" not in ci

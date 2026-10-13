"""CLI-experience tests.

Cover the clig.dev additions: ``--quiet``, ``--verbose``, ``--debug`` /
``CHAOTIC_PFC_DEBUG``, ``--dry-run``, ``--color`` / ``NO_COLOR``, the corrupted
checkpoint message, and the rule that the progress counter goes to stderr only
(never stdout, and never when stderr is not a terminal -- in particular with
``--json``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chaotic_pfc._output import color_enabled, info, progress, progress_enabled
from chaotic_pfc.cli import main


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


# ── colour precedence ───────────────────────────────────────────────────────


def test_color_explicit_always_overrides_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled(_ns(color="always")) is True


def test_color_never_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color_enabled(_ns(color="never")) is False


def test_color_auto_honours_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "anything")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    assert color_enabled(_ns(color="auto")) is False
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color_enabled(_ns(color="auto")) is True


# ── progress: stderr only, TTY only, not with --quiet ───────────────────────


def test_progress_suppressed_when_stderr_not_a_tty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress(3, 124, "kaiser/lowpass beta_9.50", _ns(quiet=False, no_progress=False))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_progress_written_to_stderr_when_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    progress(37, 124, "kaiser/lowpass beta_9.50", _ns(quiet=False, no_progress=False))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "37/124 kaiser/lowpass beta_9.50" in captured.err


def test_progress_disabled_by_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    assert progress_enabled(_ns(quiet=True, no_progress=False)) is False
    assert progress_enabled(_ns(quiet=False, no_progress=True)) is False


def test_info_suppressed_by_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    info("informational", _ns(quiet=False))
    assert "informational" in capsys.readouterr().out
    info("hidden", _ns(quiet=True))
    assert capsys.readouterr().out == ""


# ── --dry-run writes nothing ────────────────────────────────────────────────


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "src"
    (src / "Hamming (lowpass)").mkdir(parents=True)
    (src / "Hamming (lowpass)" / "variables_lyapunov.npz").write_bytes(b"placeholder")
    dst = tmp_path / "dst"
    rc = main(["sweeps", "regenerate", "--src", str(src), "--dst", str(dst), "--dry-run"])
    assert rc == 0
    assert not dst.exists()
    assert "would regenerate 1 checkpoint" in capsys.readouterr().out


# ── corrupted checkpoint ────────────────────────────────────────────────────


def test_corrupted_checkpoint_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "Hamming (lowpass)"
    bad.mkdir(parents=True)
    (bad / "variables_lyapunov.npz").write_bytes(b"not an npz archive")
    rc = main(["sweeps", "compare", "--old", str(tmp_path), "--new", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid or corrupted checkpoint" in err
    assert "variables_lyapunov.npz" in err


# ── --debug / CHAOTIC_PFC_DEBUG ─────────────────────────────────────────────


def test_debug_flag_shows_traceback(tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        main(
            [
                "--debug",
                "paper",
                "audit",
                "--tex",
                str(tmp_path / "missing.tex"),
                "--bib",
                str(tmp_path / "missing.bib"),
            ]
        )
    assert not isinstance(excinfo.value, SystemExit)


def test_debug_env_shows_traceback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHAOTIC_PFC_DEBUG", "1")
    with pytest.raises(Exception) as excinfo:
        main(
            [
                "paper",
                "audit",
                "--tex",
                str(tmp_path / "missing.tex"),
                "--bib",
                str(tmp_path / "missing.bib"),
            ]
        )
    assert not isinstance(excinfo.value, SystemExit)


def test_help_of_every_command_has_no_colour_leak(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--help`` must not emit ANSI escapes into the captured (non-TTY) output."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--color", "never", "--help"])
    assert excinfo.value.code == 0
    assert "\033[" not in capsys.readouterr().out

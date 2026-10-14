#!/usr/bin/env python3
"""Fail when a pt-BR narrative catalogue has an untranslated entry or a term
the glossary forbids.

Companion to ``tools/check_pt_br_translations.sh``.  It reads every narrative
``pt_BR`` catalogue (everything under ``locale/pt_BR/LC_MESSAGES`` except the
auto-generated ``api/generated``) and reports:

* an entry whose ``msgstr`` still equals its ``msgid`` and is **not** listed in
  ``docs/locale/pt_BR/translation-exceptions.txt`` (a genuinely identical entry:
  code, maths, tool names, bibliographic entries, bilingual glossary cells);
* a forbidden term in a translated string (the glossary lists the forms to
  avoid); the glossary page itself is exempt because it quotes them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MESSAGES = REPO_ROOT / "docs" / "locale" / "pt_BR" / "LC_MESSAGES"
EXCEPTIONS = MESSAGES.parent / "translation-exceptions.txt"
GENERATED = "api/generated"

# Terms the glossary forbids in the translation (see docs/symbols.rst).
FORBIDDEN = (
    r"\btaps\b",  # use "coeficientes" (the multipath singular "tap" is fine)
    r"\besferas?\b",  # the sampling region is a cube
    r"\bpassa-baixa\b",  # singular; the plural "passa-baixas" is correct
    r"\bpartilh",  # pt-PT: use "compartilh"
    r"\bsecç",  # pt-PT: use "seç"
    r"\bregistad",  # pt-PT: use "registr"
    r"\bficheir",  # pt-PT: use "arquiv"
    r"\butilizador",  # pt-PT: use "usuári"
    r"\becrã",  # pt-PT: use "tela"
)
# The glossary page quotes the forbidden forms on purpose.
FORBIDDEN_EXEMPT = {"symbols.po"}


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text)


def _norm(text: str) -> str:
    return " ".join(_unescape(text).split())


def _entries(path: Path):
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        m = re.search(
            r'(?m)^msgid\s+((?:"(?:[^"\\]|\\.)*"\s*)+)\nmsgstr\s+((?:"(?:[^"\\]|\\.)*"\s*)+)',
            block,
        )
        if not m:
            continue
        unq = lambda s: "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', s))  # noqa: E731
        yield unq(m.group(1)), unq(m.group(2))


def main() -> int:
    exceptions = set()
    for line in EXCEPTIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            exceptions.add(" ".join(line.split()))

    failures = 0
    for path in sorted(MESSAGES.rglob("*.po")):
        rel = path.relative_to(MESSAGES).as_posix()
        if rel.startswith(GENERATED):
            continue
        for msgid, msgstr in _entries(path):
            if not msgid:
                continue
            if msgstr == msgid and _norm(msgid) not in exceptions:
                print(f"untranslated: {rel}: {_norm(msgid)[:80]}", file=sys.stderr)
                failures += 1
            if rel not in FORBIDDEN_EXEMPT:
                for pattern in FORBIDDEN:
                    if re.search(pattern, msgstr, flags=re.IGNORECASE):
                        print(
                            f"forbidden term {pattern!r}: {rel}: {msgstr[:80]}",
                            file=sys.stderr,
                        )
                        failures += 1
    if failures:
        print(
            f"\n{failures} pt_BR problem(s): translate the entry or, if it is a "
            "genuine identity, add its normalised msgid to "
            "docs/locale/pt_BR/translation-exceptions.txt.",
            file=sys.stderr,
        )
        return 1
    print("pt_BR entries are translated and free of forbidden terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

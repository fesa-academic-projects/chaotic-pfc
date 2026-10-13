"""Shared figure-writing helper.

Lives in the top-level package so that ``analysis`` and ``plotting`` can both
use it without ``analysis`` importing a private name from ``plotting``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_figure(fig: Any, path: str | Path | None, **savefig_kwargs: Any) -> None:
    """Write ``fig`` to ``path`` if not ``None``, creating parent directories.

    Extra keyword arguments are forwarded to ``fig.savefig()``.
    """
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **savefig_kwargs)

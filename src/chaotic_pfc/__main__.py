"""
__main__.py
===========
Thin entry point used by ``python -m chaotic_pfc`` and by the
``chaotic-pfc`` console script declared in ``pyproject.toml``.

All CLI logic lives in :mod:`chaotic_pfc.cli`; this module exists only
so that both entry points delegate to the same place. Keeping it
minimal avoids any temptation to grow a second, divergent command
hierarchy here.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

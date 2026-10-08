"""User-side directory layout (DESIGN.md 7): ``$VENTRI_HOME`` or ``~/.ventri``."""
from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    h = os.environ.get("VENTRI_HOME")
    return (Path(h) if h else Path("~/.ventri")).expanduser()


def expand(p: str | os.PathLike[str]) -> Path:
    """``~/.ventri/x`` follows ``$VENTRI_HOME``; other paths expand ``~`` normally;
    relative paths are relative to the Ventri home."""
    s = os.fspath(p)
    if s == "~/.ventri" or s.startswith("~/.ventri/"):
        return home() / s.removeprefix("~/.ventri").lstrip("/")
    path = Path(s).expanduser()
    return path if path.is_absolute() else home() / path

"""``fs.read / fs.list / fs.search`` (read) and ``fs.write / fs.edit`` (write-local).

Every path is resolved (``~`` expanded, symlinks followed) and must lie inside
one of the configured ``roots``; anything else is refused by the tool itself,
so "read inside roots" is what the default ``allow`` for read tools means.
Writes default to ``ask`` (``write: ask | allow | deny``)."""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import ventri

from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry

MAX_FILE = 5_000_000


class Roots:
    def __init__(self, roots: list[str]) -> None:
        self.roots = [Path(os.path.realpath(Path(r).expanduser())) for r in roots]

    def resolve(self, p: str, *, must_exist: bool = False) -> Path:
        if not self.roots:
            raise ToolError("no roots configured for this tool")
        raw = Path(p).expanduser()
        if not raw.is_absolute():
            raw = self.roots[0] / raw
        real = Path(os.path.realpath(raw))
        if not any(real == r or r in real.parents for r in self.roots):
            raise ToolError(f"{p}: outside the allowed roots ({', '.join(map(str, self.roots))})")
        if must_exist and not real.exists():
            raise ToolError(f"{p}: no such file or directory")
        return real


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ToolError(f"{path}: not a file")
    if path.stat().st_size > MAX_FILE:
        raise ToolError(f"{path}: larger than {MAX_FILE} bytes")
    data = path.read_bytes()
    if b"\0" in data[:4096]:
        raise ToolError(f"{path}: binary file")
    return data.decode("utf-8", "replace")


class ReadArgs(BaseModel):
    path: str = Field(description="File path (absolute, ~/..., or relative to the first root)")
    offset: int = Field(0, description="Character offset to start at")
    limit: int = Field(20_000, description="Maximum characters to return")


class ListArgs(BaseModel):
    path: str = Field(".", description="Directory")
    pattern: str = Field("*", description="Glob for names, e.g. *.md")
    recursive: bool = Field(False, description="Recurse into subdirectories")


class SearchArgs(BaseModel):
    query: str = Field(description="Text (or regular expression if regex=true) to find")
    path: str = Field(".", description="Directory to search")
    glob: str = Field("*", description="File-name glob, e.g. *.md")
    regex: bool = False


class WriteArgs(BaseModel):
    path: str
    content: str
    mode: Literal["create", "overwrite", "append"] = Field(
        "create", description="create fails if the file exists")


class EditArgs(BaseModel):
    path: str
    old: str = Field(description="Exact text to replace (must occur exactly `count` times)")
    new: str
    count: int = 1


def walk(base: Path, pattern: str, recursive: bool, limit: int = 2000) -> list[Path]:
    out: list[Path] = []
    it = base.rglob("*") if recursive else base.iterdir()
    for p in sorted(it):
        if any(part.startswith(".") for part in p.relative_to(base).parts):
            continue
        if fnmatch.fnmatch(p.name, pattern):
            out.append(p)
            if len(out) >= limit:
                break
    return out


def make_tools(roots: Roots, *, write: str = "ask", prefix: str = "fs") -> list[Tool]:
    def read(a: ReadArgs, tc: ToolContext) -> str:
        p = roots.resolve(a.path, must_exist=True)
        text = _read_text(p)
        chunk = text[a.offset:a.offset + a.limit]
        more = len(text) - a.offset - len(chunk)
        head = f"{p} ({len(text)} chars)"
        return head + "\n" + chunk + (f"\n[... {more} more chars; continue with offset={a.offset + len(chunk)}]"
                                      if more > 0 else "")

    def ls(a: ListArgs, tc: ToolContext) -> str:
        base = roots.resolve(a.path, must_exist=True)
        if not base.is_dir():
            raise ToolError(f"{a.path}: not a directory")
        rows = []
        for p in walk(base, a.pattern, a.recursive, 500):
            rel = p.relative_to(base)
            rows.append(f"{rel}/" if p.is_dir() else f"{rel}  ({p.stat().st_size} B)")
        return f"{base}:\n" + ("\n".join(rows) or "(empty)")

    def search(a: SearchArgs, tc: ToolContext) -> str:
        base = roots.resolve(a.path, must_exist=True)
        try:
            rx = re.compile(a.query if a.regex else re.escape(a.query), re.IGNORECASE)
        except re.error as e:
            raise ToolError(f"bad regex: {e}") from e
        hits: list[str] = []
        for p in walk(base, a.glob, True):
            if not p.is_file() or p.stat().st_size > MAX_FILE:
                continue
            try:
                text = _read_text(p)
            except ToolError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(base)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= 200:
                        return "\n".join(hits) + "\n[... more matches truncated]"
        return "\n".join(hits) or "no matches"

    def write_(a: WriteArgs, tc: ToolContext) -> str:
        p = roots.resolve(a.path)
        if a.mode == "create" and p.exists():
            raise ToolError(f"{a.path} exists (use mode=overwrite or append)")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if a.mode == "append" else "w", encoding="utf-8") as f:
            f.write(a.content)
        return f"wrote {len(a.content)} chars to {p} ({a.mode})"

    def edit(a: EditArgs, tc: ToolContext) -> str:
        p = roots.resolve(a.path, must_exist=True)
        text = _read_text(p)
        n = text.count(a.old)
        if n != a.count:
            raise ToolError(f"expected {a.count} occurrence(s) of the old text, found {n}")
        p.write_text(text.replace(a.old, a.new), encoding="utf-8")
        return f"edited {p}: {n} replacement(s)"

    def subj(a: Any) -> dict[str, str]:
        try:
            return {"path": str(roots.resolve(a.path))}
        except ToolError:
            return {"path": a.path}

    w: Any = write if write in ("ask", "allow", "deny") else "ask"
    return [
        Tool(f"{prefix}.read", "Read a text file (paged by characters).", read, ReadArgs,
             parallel_safe=True, subject=subj, untrusted=True),
        Tool(f"{prefix}.list", "List a directory.", ls, ListArgs, parallel_safe=True, subject=subj),
        Tool(f"{prefix}.search", "Search text in files under a directory.", search, SearchArgs,
             parallel_safe=True, subject=subj, untrusted=True),
        Tool(f"{prefix}.write", "Create, overwrite or append to a text file.", write_, WriteArgs,
             risk=Risk.WRITE_LOCAL, idempotent=False, default_action=w, subject=subj),
        Tool(f"{prefix}.edit", "Replace exact text in a file.", edit, EditArgs,
             risk=Risk.WRITE_LOCAL, idempotent=False, default_action=w, subject=subj),
    ]


class FsConfig(BaseModel):
    roots: list[str] = Field(default_factory=lambda: ["~/.ventri/workspace"])
    write: Literal["ask", "allow", "deny"] = "ask"


@ventri.plugin(name="tool:fs", config=FsConfig)
def fs(ctx: Any, cfg: FsConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.fs`` (config ``roots``, ``write``)."""
    from ..paths import expand

    for r in cfg.roots:
        if r.startswith("~/.ventri"):
            expand(r).mkdir(parents=True, exist_ok=True)
    roots = Roots([str(expand(r)) for r in cfg.roots])
    for t in make_tools(roots, write=cfg.write):
        registry.register(ctx, t)


plugin = fs

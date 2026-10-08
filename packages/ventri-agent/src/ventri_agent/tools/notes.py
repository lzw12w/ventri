"""``notes.*`` -- a Markdown folder (Obsidian-vault compatible).

``notes.list / read / search`` are reads; ``notes.write`` (create / overwrite /
append) is ``write-local`` and asked by default. Note names are paths inside
the vault; ``.md`` is added when missing.

The file work is shared with ``fs.*`` (Hermes-derived core in ``fs.py`` /
``_hermes_fs``): ``notes.read`` pages by lines with ``N|`` line numbers,
``notes.search`` uses ripgrep (Python fallback), ``notes.write`` writes
atomically, preserves CRLF/BOM/undecodable bytes and -- like ``fs.write`` --
refuses to overwrite a note this session has not read in full (or that changed
since). Read-before-write state is the same session-scoped ``FileState``, so a
note read with ``notes.read`` may be edited with ``fs.edit`` and vice versa.
All of it runs in a worker thread."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import ventri

from ..paths import expand
from ._hermes_fs import search as hsearch
from ._hermes_fs.common import MAX_LINES
from .fs import Roots, file_state, in_thread, read_impl, walk, write_impl
from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry


class NotesConfig(BaseModel):
    vault: str = "~/notes"
    write: Literal["ask", "allow", "deny"] = "ask"


class NameArgs(BaseModel):
    name: str = Field(description="Note path inside the vault, e.g. 'weekly/2026-W41' (.md optional)")
    offset: int = Field(1, description="Line number to start reading from (1-indexed, default: 1)")
    limit: int = Field(MAX_LINES, description=f"Maximum number of lines to read (default and max: {MAX_LINES})")


class ListArgs(BaseModel):
    folder: str = Field("", description="Folder inside the vault ('' = root)")


class SearchArgs(BaseModel):
    query: str = Field(description="Text to find (case-insensitive substring) in note names and contents")
    limit: int = Field(50, description="Maximum number of matching lines to return (default: 50)")
    offset: int = Field(0, description="Skip the first N matching lines (pagination)")


class WriteArgs(BaseModel):
    name: str = Field(description="Note path inside the vault (.md optional)")
    content: str = Field(description="Markdown text (the whole note for create/overwrite)")
    mode: Literal["create", "overwrite", "append"] = Field("create", description=(
        "create: new note, fails if it exists; overwrite: replace the whole note (read it in full first); "
        "append: add to the end"))


READ_DESC = ("Read a note, with 'LINE_NUM|CONTENT' line numbers (the prefix is not part of the note). Long "
             "notes are paged: continue with offset=. Read a note before overwriting it.")
WRITE_DESC = ("Create, overwrite or append to a note. overwrite replaces the whole note and is refused unless "
              "this session has read the current note in full; append adds a newline first when the note does "
              "not end with one.")


def make_tools(vault: Roots, write: str = "ask") -> list[Tool]:
    root = vault.roots[0]

    def path(name: str) -> Path:
        n = name if name.endswith(".md") else name + ".md"
        return vault.resolve(n)

    def show(p: Path) -> str:
        return vault.display(p)

    async def ls(a: ListArgs, tc: ToolContext) -> str:
        def work() -> str:
            base = vault.resolve(a.folder or ".", must_exist=True)
            names = [str(p.relative_to(root).with_suffix("")) for p in walk(base, "*.md", True, 1000)]
            return "\n".join(names) or "(no notes)"
        return await in_thread(work, cancellable=True)

    async def read(a: NameArgs, tc: ToolContext) -> str:
        st = file_state(tc)

        def work() -> str:
            p = path(a.name)
            if not p.exists() and not p.is_symlink():
                similar = [str(q.relative_to(root).with_suffix("")) for q in walk(root, "*.md", True, 1000)
                           if Path(a.name).stem.lower() in q.stem.lower()][:5]
                raise ToolError(f"no note {a.name!r}" + (f" (similar: {', '.join(similar)})" if similar else ""))
            return read_impl(vault, st, a.name, p, a.offset, a.limit, show)
        return await in_thread(work, cancellable=True)

    async def search(a: SearchArgs, tc: ToolContext) -> str:
        def work() -> str:
            q = a.query.lower()
            hits = [f"{n}: (title match)" for n in
                    (str(p.relative_to(root).with_suffix("")) for p in walk(root, "*.md", True))
                    if q and q in n.lower()]
            spec = hsearch.SearchSpec(pattern=a.query, base=str(root), file_glob="*.md", limit=a.limit,
                                      offset=a.offset, ignore_case=True, fixed_strings=True)
            res = hsearch.search(spec, lambda p: show(Path(p)))
            if res.error:
                raise ToolError(res.error)
            for m in res.matches:
                rel = str(Path(m.path).relative_to(root).with_suffix(""))
                hits.append(f"{rel}:{m.line_number}: {m.content.strip()[:200]}")
            out = "\n".join(hits) or "no matches"
            if res.truncated:
                out += f"\n[more matches: continue with offset={spec.offset + spec.limit}]"
            return out
        return await in_thread(work, cancellable=True)

    async def write_(a: WriteArgs, tc: ToolContext) -> str:
        st = file_state(tc)

        def work() -> str:
            p = path(a.name)
            return write_impl(st, a.name, p, a.content, a.mode, show)
        return await in_thread(work, cancellable=False)

    w: Any = write
    return [
        Tool("notes.list", "List notes in the Markdown vault.", ls, ListArgs, parallel_safe=True),
        Tool("notes.read", READ_DESC, read, NameArgs, parallel_safe=True, untrusted=True,
             subject=lambda a: {"path": a.name}),
        Tool("notes.search", "Full-text search in notes (case-insensitive substring).", search, SearchArgs,
             parallel_safe=True, untrusted=True),
        Tool("notes.write", WRITE_DESC, write_, WriteArgs, risk=Risk.WRITE_LOCAL,
             idempotent=False, default_action=w, subject=lambda a: {"path": a.name}),
    ]


@ventri.plugin(name="tool:notes", config=NotesConfig)
def notes(ctx: Any, cfg: NotesConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.notes`` (config ``vault``, ``write``)."""
    v = expand(cfg.vault)
    v.mkdir(parents=True, exist_ok=True)
    for t in make_tools(Roots([str(v)]), cfg.write):
        registry.register(ctx, t)


plugin = notes

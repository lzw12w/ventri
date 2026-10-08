"""``notes.*`` -- a Markdown folder (Obsidian-vault compatible).

``notes.list / read / search`` are reads; ``notes.write`` (create / overwrite /
append) is ``write-local`` and asked by default. Note names are paths inside
the vault; ``.md`` is added when missing."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

import ventri

from ..paths import expand
from .fs import Roots, _read_text, walk
from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry


class NotesConfig(BaseModel):
    vault: str = "~/notes"
    write: Literal["ask", "allow", "deny"] = "ask"


class NameArgs(BaseModel):
    name: str = Field(description="Note path inside the vault, e.g. 'weekly/2026-W41' (.md optional)")


class ListArgs(BaseModel):
    folder: str = Field("", description="Folder inside the vault ('' = root)")


class SearchArgs(BaseModel):
    query: str


class WriteArgs(BaseModel):
    name: str
    content: str
    mode: Literal["create", "overwrite", "append"] = "create"


def make_tools(vault: Roots, write: str = "ask") -> list[Tool]:
    root = vault.roots[0]

    def path(name: str) -> Any:
        n = name if name.endswith(".md") else name + ".md"
        return vault.resolve(n)

    def ls(a: ListArgs, tc: ToolContext) -> str:
        base = vault.resolve(a.folder or ".", must_exist=True)
        names = [str(p.relative_to(root).with_suffix("")) for p in walk(base, "*.md", True, 1000)]
        return "\n".join(names) or "(no notes)"

    def read(a: NameArgs, tc: ToolContext) -> str:
        p = path(a.name)
        if not p.exists():
            raise ToolError(f"no note {a.name!r}")
        return _read_text(p)

    def search(a: SearchArgs, tc: ToolContext) -> str:
        q = a.query.lower()
        hits = []
        for p in walk(root, "*.md", True):
            try:
                text = _read_text(p)
            except ToolError:
                continue
            rel = str(p.relative_to(root).with_suffix(""))
            if q in rel.lower():
                hits.append(f"{rel}: (title match)")
            for n, line in enumerate(text.splitlines(), 1):
                if q in line.lower():
                    hits.append(f"{rel}:{n}: {line.strip()[:200]}")
            if len(hits) > 200:
                break
        return "\n".join(hits[:200]) or "no matches"

    def write_(a: WriteArgs, tc: ToolContext) -> str:
        p = path(a.name)
        if a.mode == "create" and p.exists():
            raise ToolError(f"note {a.name!r} exists (use overwrite or append)")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if a.mode == "append" else "w", encoding="utf-8") as f:
            f.write(a.content)
        return f"saved note {p.relative_to(root)} ({a.mode})"

    w: Any = write
    return [
        Tool("notes.list", "List notes in the Markdown vault.", ls, ListArgs, parallel_safe=True),
        Tool("notes.read", "Read a note.", read, NameArgs, parallel_safe=True, untrusted=True,
             subject=lambda a: {"path": a.name}),
        Tool("notes.search", "Full-text search in notes (case-insensitive substring).", search, SearchArgs,
             parallel_safe=True, untrusted=True),
        Tool("notes.write", "Create, overwrite or append to a note.", write_, WriteArgs, risk=Risk.WRITE_LOCAL,
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

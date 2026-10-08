"""Dependency diagnostics (DESIGN.md 4.9): explain every PENDING fiber.

After each settle the kernel builds, for PENDING fibers, the graph "fiber ->
fibers that could provide one of its missing required keys" and runs Tarjan's
SCC algorithm on it. Fibers on a cycle get ``pending_reason = "cycle: a#1 → b#2 →
a#1"`` (and one ``dep.cycle`` trace event per new cycle); others get
``missing: K (no plugin provides it)`` or ``waiting: K (provider x#3 is ...)``.

A fiber "can provide" K if it declared K in its plugin's ``provides`` metadata or
bound K during an earlier activation, in the same realm and transaction world.
Non-guarantee: a cycle between plugins that never declared ``provides`` and never
ran cannot be told apart from a missing provider.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .fiber import State
from .plugin import keyname

if TYPE_CHECKING:  # pragma: no cover
    from .fiber import Fiber
    from .kernel import Kernel


def _missing(k: Kernel, f: Fiber) -> list[Any]:
    tx = f.tx
    return [key for key in f.inject if k._lookup(k._realm_of(f, key), key, tx) is None]


def _providers(k: Kernel, f: Fiber, key: Any, removed: dict) -> list[Fiber]:
    tx, realm = f.tx, k._realm_of(f, key)
    out = []
    for p in k._providers.get(key, ()):
        if p.state is State.DISPOSED or (p.tx is not None and p.tx is not tx):
            continue
        if tx is not None and p.tx is None:
            gone = removed.get(tx)
            if gone is None:
                gone = removed[tx] = tx._removed()
            if p in gone:
                continue
        if k._realm_of(p, key) is realm:
            out.append(p)
    return out


def _sccs(nodes: list[Fiber], edges: dict[Fiber, list[Fiber]]) -> list[list[Fiber]]:
    """Tarjan's algorithm (iterative)."""
    index: dict[Fiber, int] = {}
    low: dict[Fiber, int] = {}
    on_stack: set[Fiber] = set()
    stack: list[Fiber] = []
    out: list[list[Fiber]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack.add(v)
            succ = edges.get(v, [])
            if i < len(succ):
                work.append((v, i + 1))
                w = succ[i]
                if w not in index:
                    work.append((w, 0))
                elif w in on_stack:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w is v:
                        break
                out.append(comp)
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return out


def _cycle_path(start: Fiber, members: set[Fiber], edges: dict[Fiber, list[Fiber]]) -> list[Fiber]:
    """Shortest path start -> ... -> start inside one SCC (BFS)."""
    prev: dict[Fiber, Fiber] = {}
    frontier = [start]
    while frontier:
        nxt = []
        for v in frontier:
            for w in edges.get(v, []):
                if w not in members:
                    continue
                if w is start:
                    path = [v]
                    while path[-1] is not start:
                        path.append(prev[path[-1]])
                    return [start, *reversed(path[:-1]), start] if v is not start else [start, start]
                if w not in prev:
                    prev[w] = v
                    nxt.append(w)
        frontier = nxt
    return [start, start]


def diagnose(k: Kernel) -> None:
    pending = [f for f in k._pending if f.state is State.PENDING and not f._dispose_requested]
    if not pending:
        k._cycles = set()
        return
    removed: dict = {}
    edges: dict[Fiber, list[Fiber]] = {}
    notes: dict[Fiber, list[str]] = {}
    for f in pending:
        if f._parked:
            f.pending_reason = "parked: stop-first replacement in progress"
            continue
        missing = _missing(k, f)
        parts: list[str] = []
        out: list[Fiber] = []
        for key in missing:
            provs = _providers(k, f, key, removed)
            if not provs:
                parts.append(f"missing: {keyname(key)} (no plugin provides it)")
                continue
            waiting = []
            for p in provs:
                if p.state is State.PENDING:
                    out.append(p)
                    waiting.append(f"{p.label} is pending")
                elif p.state is State.FAILED:
                    waiting.append(f"{p.label} failed: {p.error!r}")
                else:
                    waiting.append(f"{p.label} is {p.state.value}")
            parts.append(f"waiting: {keyname(key)} (provider {', '.join(waiting)})")
        if not missing:
            parent = f.parent
            if parent is not None and parent.state not in (State.LOADING, State.ACTIVE):
                parts.append(f"waiting: parent {parent.label} is {parent.state.value}")
        edges[f] = out
        notes[f] = parts
    cycles: set[frozenset] = set()
    for comp in _sccs([f for f in pending if f in edges], edges):
        members = set(comp)
        if len(comp) == 1 and comp[0] not in edges.get(comp[0], []):
            continue
        key = frozenset(members)
        cycles.add(key)
        for f in comp:
            path = _cycle_path(f, members, edges)
            notes[f] = ["cycle: " + " → ".join(p.label for p in path)]
        if key not in k._cycles:
            first = min(comp, key=lambda x: x.id)
            path = _cycle_path(first, members, edges)
            k._trace("dep.cycle", first, cycle=[p.label for p in path])
    k._cycles = cycles
    for f, parts in notes.items():
        f.pending_reason = "; ".join(parts) or None

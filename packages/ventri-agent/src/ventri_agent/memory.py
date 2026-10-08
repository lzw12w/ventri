"""Long-term memory v1 (DESIGN.md 5.3): SQLite + FTS5, plus session working memory.

* ``LongTermMemory`` (root realm): items of kind ``fact`` / ``preference`` /
  ``episode`` / ``procedure`` with source (session id, message id), time,
  confidence and a sensitive tag. Search = FTS5 with the ``trigram`` tokenizer
  (works for Chinese without a segmenter) plus a ``LIKE`` fallback for terms
  shorter than three characters.
* Writes are de-duplicated (normalised text, or character-trigram Jaccard >=
  0.8 within a kind: merged, confidence = max). Sensitive items (health,
  finance, credential-like) are stored ``pending`` until the user confirms
  (``/memory`` or ``va memory confirm``).
* ``WorkingMemory`` (session realm): the current task's notes/plan, readable and
  writable by the model through ``work.read`` / ``work.write``; persisted in
  the session log.

Non-guarantees: no vector search in v1 (the ``[vec]`` extra is reserved);
SQLite calls are synchronous (the database is local and small).
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

import ventri

from .messages import now_ts
from .paths import expand

Kind = Literal["fact", "preference", "episode", "procedure"]
KINDS: tuple[str, ...] = ("preference", "fact", "procedure", "episode")  # snapshot priority
_CREDENTIAL = re.compile(r"(sk-[A-Za-z0-9]{16,}|password|密码|api[_ -]?key|token)", re.IGNORECASE)


@dataclass
class MemoryItem:
    id: int
    kind: str
    text: str
    source_session: str | None
    source_message: str | None
    created: float
    updated: float
    confidence: float
    sensitive: bool
    status: str  # active | pending

    def line(self) -> str:
        flag = " (pending)" if self.status == "pending" else ""
        return f"#{self.id} [{self.kind}] {self.text}{flag}"


def _norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def _trigrams(text: str) -> set[str]:
    t = _norm(text)
    return {t[i:i + 3] for i in range(max(1, len(t) - 2))}


def similar(a: str, b: str) -> float:
    x, y = _trigrams(a), _trigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


class LongTermMemory:
    """Service. ``path=None`` keeps the database in memory (tests)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path) if self.path else ":memory:", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS memories(
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,
                    source_session TEXT, source_message TEXT, created REAL, updated REAL,
                    confidence REAL DEFAULT 0.7, sensitive INTEGER DEFAULT 0, status TEXT DEFAULT 'active');
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text, content='memories', content_rowid='id', tokenize='trigram');
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
                    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;
            """)

    def close(self) -> None:
        self._db.close()

    def _rows(self, sql: str, args: tuple[Any, ...] = ()) -> list[MemoryItem]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [MemoryItem(r["id"], r["kind"], r["text"], r["source_session"], r["source_message"],
                           r["created"], r["updated"], r["confidence"], bool(r["sensitive"]), r["status"])
                for r in rows]

    # ------------------------------------------------------------- writes
    def add(self, text: str, kind: str = "fact", *, source_session: str | None = None,
            source_message: str | None = None, confidence: float = 0.7,
            sensitive: bool | None = None, confirmed: bool = False) -> tuple[MemoryItem, bool]:
        """Add (or merge into a near-duplicate). Returns ``(item, created)``.
        Sensitive items are stored as ``pending`` unless ``confirmed``."""
        text = text.strip()
        if not text:
            raise ValueError("empty memory")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if sensitive is None:
            sensitive = bool(_CREDENTIAL.search(text))
        for item in self.list(kind=kind, status=None):
            if _norm(item.text) == _norm(text) or similar(item.text, text) >= 0.8:
                longer = text if len(text) > len(item.text) else item.text
                with self._lock, self._db:
                    self._db.execute("UPDATE memories SET text=?, confidence=?, updated=? WHERE id=?",
                                     (longer, max(confidence, item.confidence), now_ts(), item.id))
                return self.get(item.id) or item, False
        status = "pending" if sensitive and not confirmed else "active"
        ts = now_ts()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO memories(kind, text, source_session, source_message, created, updated, "
                "confidence, sensitive, status) VALUES (?,?,?,?,?,?,?,?,?)",
                (kind, text, source_session, source_message, ts, ts, confidence, int(sensitive), status))
            rid = cur.lastrowid
        item = self.get(int(rid or 0))
        assert item is not None
        return item, True

    def update(self, item_id: int, text: str) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE memories SET text=?, updated=? WHERE id=?", (text.strip(), now_ts(), item_id))

    def confirm(self, item_id: int) -> bool:
        with self._lock, self._db:
            n = self._db.execute("UPDATE memories SET status='active', updated=? WHERE id=? AND status='pending'",
                                 (now_ts(), item_id)).rowcount
        return n > 0

    def forget(self, item_id: int) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM memories WHERE id=?", (item_id,)).rowcount > 0

    # -------------------------------------------------------------- reads
    def get(self, item_id: int) -> MemoryItem | None:
        rows = self._rows("SELECT * FROM memories WHERE id=?", (item_id,))
        return rows[0] if rows else None

    def list(self, *, kind: str | None = None, status: str | None = "active",
             since: float | None = None) -> list[MemoryItem]:
        sql, args = "SELECT * FROM memories WHERE 1=1", []
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if status:
            sql += " AND status=?"
            args.append(status)
        if since is not None:
            sql += " AND created>=?"
            args.append(since)
        return self._rows(sql + " ORDER BY id", tuple(args))

    def search(self, query: str, k: int = 8, *, kinds: list[str] | None = None) -> list[MemoryItem]:
        terms = [t for t in re.split(r"\s+", query.strip()) if t]
        if not terms:
            return []
        found: dict[int, MemoryItem] = {}
        long_terms = [t for t in terms if len(t) >= 3]
        if long_terms:
            q = " OR ".join('"' + t.replace('"', '""') + '"' for t in long_terms)
            for it in self._rows("SELECT m.* FROM memories_fts f JOIN memories m ON m.id=f.rowid "
                                 "WHERE memories_fts MATCH ? AND m.status='active' ORDER BY bm25(memories_fts), "
                                 "m.confidence DESC LIMIT ?", (q, k * 2)):
                found.setdefault(it.id, it)
        for t in terms:
            if len(t) < 3:
                for it in self._rows("SELECT * FROM memories WHERE status='active' AND text LIKE ? "
                                     "ORDER BY confidence DESC LIMIT ?", (f"%{t}%", k)):
                    found.setdefault(it.id, it)
        out = [it for it in found.values() if not kinds or it.kind in kinds]
        return out[:k]

    def top(self, k: int = 12) -> list[MemoryItem]:
        """Snapshot for the stable prefix: by kind priority, confidence, recency."""
        items = self.list()
        items.sort(key=lambda m: (KINDS.index(m.kind), -m.confidence, -m.updated))
        return items[:k]

    # ------------------------------------------------------------ export
    def export_markdown(self) -> str:
        lines = ["# Ventri memory", ""]
        for kind in KINDS:
            items = self.list(kind=kind, status=None)
            if not items:
                continue
            lines += [f"## {kind}", ""]
            for it in items:
                tags = [f"id={it.id}", f"confidence={it.confidence:.2f}"]
                if it.sensitive:
                    tags.append("sensitive")
                if it.status != "active":
                    tags.append(it.status)
                lines.append(f"- {it.text}  <!-- {' '.join(tags)} -->")
            lines.append("")
        return "\n".join(lines)

    def import_markdown(self, text: str) -> int:
        """Import ``- item`` lines under ``## <kind>`` headings (edited exports).
        Returns how many new items were created (duplicates merge)."""
        kind, n = "fact", 0
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("## "):
                k = line[3:].strip()
                kind = k if k in KINDS else "fact"
            elif line.startswith("- "):
                body = re.sub(r"\s*<!--.*?-->\s*$", "", line[2:]).strip()
                if body:
                    n += self.add(body, kind, confirmed=True, sensitive=False)[1]
        return n


class WorkingMemory:
    """Session-realm scratchpad (task notes / plan)."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def write(self, text: str, mode: str = "replace") -> str:
        self.text = (self.text + ("\n" if self.text else "") + text) if mode == "append" else text
        return self.text


class MemoryConfig(BaseModel):
    path: str = "~/.ventri/memory.db"


@ventri.plugin(name="memory-sqlite", config=MemoryConfig, provides={"memory": LongTermMemory})
def memory_sqlite(ctx: Any, cfg: MemoryConfig) -> None:
    """``use: ventri_agent.memory`` (or ``ventri_agent.memory.sqlite``) -- LongTermMemory."""
    mem = LongTermMemory(expand(cfg.path) if cfg.path != ":memory:" else None)
    ctx.provide(LongTermMemory, mem)
    ctx.on_dispose(mem.close)


sqlite = memory_sqlite
plugin = memory_sqlite

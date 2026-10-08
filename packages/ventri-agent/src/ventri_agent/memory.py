"""Long-term memory v1 (DESIGN.md 5.3): SQLite + FTS5, plus session working memory.

* ``LongTermMemory`` (root realm): items of kind ``fact`` / ``preference`` /
  ``episode`` / ``procedure`` with source (session id, message id), time,
  confidence and a sensitive tag. Search = FTS5 with the ``trigram`` tokenizer
  (works for Chinese without a segmenter) plus a ``LIKE`` fallback for terms
  shorter than three characters.
* Writes are de-duplicated only when the normalised text is identical (within
  a kind; confidence = max). A *near* duplicate (character-trigram Jaccard >=
  0.8, e.g. "uses PostgreSQL 16" -> "uses PostgreSQL 15") is a correction or
  refinement: the new item is stored and the old one is marked ``superseded``
  with a link both ways (``supersedes`` / ``superseded_by``) -- newer wins and
  nothing is silently dropped. ``update`` can restore a superseded item.
  Sensitive items (health, finance, credential-like) are stored ``pending``
  until the user confirms (``/memory`` or ``va memory confirm``); a pending
  item's supersede takes effect on confirmation.
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
    status: str  # active | pending | superseded
    supersedes: int | None = None
    superseded_by: int | None = None

    def line(self) -> str:
        flag = {"pending": " (pending)", "superseded": f" (superseded by #{self.superseded_by})"}.get(self.status, "")
        return f"#{self.id} [{self.kind}] {self.text}{flag}"


@dataclass
class Remembered:
    """Outcome of :meth:`LongTermMemory.remember`."""

    item: MemoryItem
    action: Literal["created", "duplicate", "superseded"]
    previous: MemoryItem | None = None   # the duplicate, or the item ``item`` supersedes


def sensitive_text(text: str) -> bool:
    return bool(_CREDENTIAL.search(text))


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
            cols = {r["name"] for r in self._db.execute("PRAGMA table_info(memories)")}
            for col in ("supersedes", "superseded_by"):   # v0.2 databases: add the link columns
                if col not in cols:
                    self._db.execute(f"ALTER TABLE memories ADD COLUMN {col} INTEGER")

    def close(self) -> None:
        self._db.close()

    def _rows(self, sql: str, args: tuple[Any, ...] = ()) -> list[MemoryItem]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [MemoryItem(r["id"], r["kind"], r["text"], r["source_session"], r["source_message"],
                           r["created"], r["updated"], r["confidence"], bool(r["sensitive"]), r["status"],
                           r["supersedes"], r["superseded_by"])
                for r in rows]

    # ------------------------------------------------------------- writes
    def add(self, text: str, kind: str = "fact", *, source_session: str | None = None,
            source_message: str | None = None, confidence: float = 0.7,
            sensitive: bool | None = None, confirmed: bool = False) -> tuple[MemoryItem, bool]:
        """:meth:`remember`, returning ``(item, created)`` (``created`` is False
        only for an exact duplicate)."""
        r = self.remember(text, kind, source_session=source_session, source_message=source_message,
                          confidence=confidence, sensitive=sensitive, confirmed=confirmed)
        return r.item, r.action != "duplicate"

    def remember(self, text: str, kind: str = "fact", *, source_session: str | None = None,
                 source_message: str | None = None, confidence: float = 0.7,
                 sensitive: bool | None = None, confirmed: bool = False) -> Remembered:
        """Store ``text``. Identical (normalised) text in the same kind is a
        duplicate; a near duplicate (similarity >= 0.8) is superseded by the new
        item. Sensitive items are stored as ``pending`` unless ``confirmed``."""
        text = text.strip()
        if not text:
            raise ValueError("empty memory")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if sensitive is None:
            sensitive = sensitive_text(text)
        live = [it for it in self.list(kind=kind, status=None) if it.status in ("active", "pending")]
        norm = _norm(text)
        for item in live:
            if _norm(item.text) == norm:
                with self._lock, self._db:
                    # only case / spacing / punctuation differ: keep the newer spelling
                    self._db.execute("UPDATE memories SET text=?, confidence=?, updated=? WHERE id=?",
                                     (text, max(confidence, item.confidence), now_ts(), item.id))
                return Remembered(self.get(item.id) or item, "duplicate", item)
        best: MemoryItem | None = None
        best_score = 0.0
        for item in live:
            score = similar(item.text, text)
            if score >= 0.8 and score > best_score:
                best, best_score = item, score
        status = "pending" if sensitive and not confirmed else "active"
        ts = now_ts()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO memories(kind, text, source_session, source_message, created, updated, "
                "confidence, sensitive, status, supersedes) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (kind, text, source_session, source_message, ts, ts, confidence, int(sensitive), status,
                 best.id if best else None))
            rid = int(cur.lastrowid or 0)
            if best is not None and status == "active":
                self._supersede(best.id, rid)
        item = self.get(rid)
        assert item is not None
        if best is None:
            return Remembered(item, "created")
        return Remembered(item, "superseded", self.get(best.id) or best)

    def _supersede(self, old: int, new: int) -> None:
        """Caller holds the lock and transaction."""
        self._db.execute("UPDATE memories SET status='superseded', superseded_by=?, updated=? "
                         "WHERE id=? AND status IN ('active', 'pending')", (new, now_ts(), old))

    def update(self, item_id: int, text: str, *, sensitive: bool | None = None,
               confirmed: bool = False) -> MemoryItem | None:
        """Replace an item's text (a correction). A superseded item becomes
        active again (its link is cleared; the item that superseded it stays).
        Text that looks sensitive puts the item back to ``pending`` unless
        ``confirmed``. Returns the updated item, or None if it does not exist."""
        text = text.strip()
        if not text:
            raise ValueError("empty memory")
        item = self.get(item_id)
        if item is None:
            return None
        if sensitive is None:
            sensitive = sensitive_text(text)
        status = "pending" if sensitive and not confirmed else "active"
        with self._lock, self._db:
            self._db.execute("UPDATE memories SET text=?, sensitive=?, status=?, superseded_by=NULL, updated=? "
                             "WHERE id=?", (text, int(sensitive), status, now_ts(), item_id))
        return self.get(item_id)

    def confirm(self, item_id: int) -> bool:
        with self._lock, self._db:
            row = self._db.execute("SELECT supersedes FROM memories WHERE id=? AND status='pending'",
                                   (item_id,)).fetchone()
            if row is None:
                return False
            self._db.execute("UPDATE memories SET status='active', updated=? WHERE id=?", (now_ts(), item_id))
            if row["supersedes"] is not None:
                self._supersede(int(row["supersedes"]), item_id)
        return True

    def forget(self, item_id: int) -> bool:
        with self._lock, self._db:
            n = self._db.execute("DELETE FROM memories WHERE id=?", (item_id,)).rowcount
            if n:
                self._db.execute("UPDATE memories SET supersedes=NULL WHERE supersedes=?", (item_id,))
                self._db.execute("UPDATE memories SET superseded_by=NULL WHERE superseded_by=?", (item_id,))
            return n > 0

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
                if it.status == "superseded":
                    tags.append(f"superseded_by={it.superseded_by}")
                elif it.status != "active":
                    tags.append(it.status)
                lines.append(f"- {it.text}  <!-- {' '.join(tags)} -->")
            lines.append("")
        return "\n".join(lines)

    def import_markdown(self, text: str) -> int:
        """Import ``- item`` lines under ``## <kind>`` headings (edited exports).
        Returns how many new items were created (exact duplicates are skipped,
        near duplicates supersede earlier lines; ``superseded`` lines are history
        and are not imported)."""
        kind, n = "fact", 0
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("## "):
                k = line[3:].strip()
                kind = k if k in KINDS else "fact"
            elif line.startswith("- ") and "superseded_by=" not in line:
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

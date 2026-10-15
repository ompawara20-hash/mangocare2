"""SQLite persistence (metadata only; uploaded images are never stored)."""
import json
import sqlite3
from contextlib import contextmanager
from .config import settings

SCHEMA = """CREATE TABLE IF NOT EXISTS analyses(
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
 class_id TEXT, class_name TEXT, confidence REAL, alternatives TEXT,
 validation_score REAL, model_name TEXT, model_version TEXT)"""


@contextmanager
def conn():
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(settings.database_path)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.execute(SCHEMA)


def insert(rec: dict) -> None:
    with conn() as c:
        c.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?,?,?)", (
            rec["id"], rec["created_at"], rec["status"], rec["class_id"], rec["class_name"],
            rec["confidence"], json.dumps(rec["alternatives"]), rec["validation_score"],
            rec["model_name"], rec["model_version"]))


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["alternatives"] = json.loads(d["alternatives"] or "[]")
    return d


def get(aid: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    return _row(r) if r else None


def delete(aid: str) -> bool:
    with conn() as c:
        return c.execute("DELETE FROM analyses WHERE id=?", (aid,)).rowcount > 0


def page(page: int, size: int, status: str | None) -> tuple[list[dict], int]:
    where, args = ("WHERE status=?", [status]) if status else ("", [])
    with conn() as c:
        total = c.execute(f"SELECT COUNT(*) FROM analyses {where}", args).fetchone()[0]
        rows = c.execute(f"SELECT * FROM analyses {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                         args + [size, (page - 1) * size]).fetchall()
    return [_row(r) for r in rows], total


def stats() -> dict:
    with conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM analyses GROUP BY status").fetchall()
    out = {r["status"]: r["n"] for r in rows}
    out["total"] = sum(out.values())
    return out

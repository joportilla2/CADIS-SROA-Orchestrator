from __future__ import annotations

"""
Author: Omar Portilla Jaimes
Email: jorge.portilla2@unipamplona.edu.co

Protocol / persistence utilities.

v5 design goal:
- Default: no per-event persistence (fast, small outputs).
- Reproducibility is guaranteed via config + code fingerprints + checksums.
- Optional qualitative audit: store a small subset of per-case records (JSONL).
- Optional full tracing: per-case JSONL; SQLite only if explicitly requested.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from pathlib import Path
import datetime
import hashlib
import json
import sqlite3
import uuid

def utc_now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class BaseStore:
    """Abstract persistence interface.

    This is intentionally minimal: the simulation may optionally emit *case-level*
    records for audit or full tracing, but per-event storage is disabled by default.
    """
    def record_case(self, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return

class NullStore(BaseStore):
    def record_case(self, record: Dict[str, Any]) -> None:
        return

class JsonlCaseStore(BaseStore):
    """Append-only JSONL store for case-level records."""
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def record_case(self, record: Dict[str, Any]) -> None:
        self._fh.write(stable_json(record) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

class SqliteStore(BaseStore):
    """Optional per-event store with tamper-evident hash chain (legacy).

    Disabled by default. Only enable if:
      --trace-level full --use-sqlite
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._con = sqlite3.connect(db_path)
        self._con.execute(
            """CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                consultation_id TEXT,
                logical_clock INTEGER,
                payload_json TEXT,
                prev_hash TEXT,
                this_hash TEXT,
                timestamp_utc TEXT
            )"""
        )
        self._con.commit()
        self._last_hash_by_cons: Dict[str, str] = {}

    def append_event(self, consultation_id: str, logical_clock: int, payload: Dict[str, Any]) -> None:
        eid = str(uuid.uuid4())
        payload_json = stable_json(payload)
        prev = self._last_hash_by_cons.get(consultation_id, "")
        this_hash = sha256_hex(prev + payload_json)
        self._con.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (eid, consultation_id, logical_clock, payload_json, prev, this_hash, utc_now_iso()),
        )
        self._last_hash_by_cons[consultation_id] = this_hash

    def record_case(self, record: Dict[str, Any]) -> None:
        # For sqlite mode, we still allow storing a summary per case if desired.
        # Kept as no-op here to avoid schema changes.
        return

    def close(self) -> None:
        try:
            self._con.commit()
            self._con.close()
        except Exception:
            pass

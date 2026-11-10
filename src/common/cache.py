"""A tiny persistent cache in SQLite (data/cache/cache.sqlite by default).

Used for embeddings and LLM responses, so re-running an experiment with the
same inputs costs nothing and returns identical results.

    cache = Cache(path)
    cache.get("llm", request_dict)          # -> stored value or None
    cache.put("llm", request_dict, value)

The key is a SHA-256 of the request serialized with sorted keys, so two dicts
with the same content always hit the same entry. Any change to the request
(model, prompt text, settings) is a different key: stale results can't leak in.
Delete the file to clear everything.
"""

import hashlib
import json
import sqlite3
from pathlib import Path


class Cache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
            " PRIMARY KEY (namespace, key))"
        )

    @staticmethod
    def key(payload) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, namespace: str, payload):
        row = self.db.execute(
            "SELECT value FROM cache WHERE namespace = ? AND key = ?", (namespace, self.key(payload))
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, namespace: str, payload, value) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO cache (namespace, key, value) VALUES (?, ?, ?)",
            (namespace, self.key(payload), json.dumps(value, ensure_ascii=False)),
        )
        self.db.commit()

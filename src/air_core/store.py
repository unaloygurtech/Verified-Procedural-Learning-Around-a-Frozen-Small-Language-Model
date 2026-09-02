from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL,
    prompt_tokens INTEGER,
    generated_tokens INTEGER,
    passed INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('candidate', 'active', 'rejected')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class ExperimentStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(SCHEMA)
            yield connection
            connection.commit()
        finally:
            connection.close()

    def record_run(
        self,
        *,
        kind: str,
        prompt: str,
        response: str,
        elapsed_seconds: float,
        prompt_tokens: int | None,
        generated_tokens: int | None,
        passed: bool | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    kind, prompt, response, elapsed_seconds, prompt_tokens,
                    generated_tokens, passed, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    prompt,
                    response,
                    elapsed_seconds,
                    prompt_tokens,
                    generated_tokens,
                    None if passed is None else int(passed),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def upsert_skill(self, *, name: str, body: str, state: str = "candidate") -> int:
        if state not in {"candidate", "active", "rejected"}:
            raise ValueError(f"invalid skill state: {state}")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO skills (name, body, state)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET body=excluded.body, state=excluded.state
                """,
                (name, body, state),
            )
            row = connection.execute(
                "SELECT id FROM skills WHERE name = ?", (name,)
            ).fetchone()
            return int(row["id"])

    def set_skill_state(self, *, name: str, state: str) -> None:
        if state not in {"candidate", "active", "rejected"}:
            raise ValueError(f"invalid skill state: {state}")
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE skills SET state = ? WHERE name = ?", (state, name)
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown skill: {name}")

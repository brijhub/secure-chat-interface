"""SQLite storage for per-user conversation history."""

import json
import sqlite3
from contextlib import closing

from config import HISTORY_DB_PATH


def init_history():
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(HISTORY_DB_PATH)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS messages_conversation
            ON messages (email, session_id, id)
        """)


def load_history(email, session_id):
    with closing(sqlite3.connect(HISTORY_DB_PATH)) as connection:
        rows = connection.execute(
            "SELECT message_json FROM messages WHERE email = ? AND session_id = ? ORDER BY id",
            (email, session_id),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def append_messages(email, session_id, messages):
    # One transaction saves the question and response together.
    with closing(sqlite3.connect(HISTORY_DB_PATH)) as connection, connection:
        connection.executemany(
            "INSERT INTO messages (email, session_id, message_json) VALUES (?, ?, ?)",
            [(email, session_id, json.dumps(message, ensure_ascii=False)) for message in messages],
        )


def clear_history(email, session_id):
    with closing(sqlite3.connect(HISTORY_DB_PATH)) as connection, connection:
        connection.execute(
            "DELETE FROM messages WHERE email = ? AND session_id = ?",
            (email, session_id),
        )

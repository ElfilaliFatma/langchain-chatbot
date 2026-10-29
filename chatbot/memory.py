"""Conversation state: sessions, persistence and history trimming.

LangChain's ``RunnableWithMessageHistory`` only needs a *function* that maps a
session id to a ``BaseChatMessageHistory``. That function is the seam where the
application decides where memory lives; here it can be RAM or SQLite, and both
implement the same interface.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class ChatSession:
    """Metadata about one conversation, independent of the message store."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New conversation"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0

    def touch(self, messages: int = 2) -> None:
        self.updated_at = time.time()
        self.message_count += messages

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }


class SessionStore:
    """In-memory store, mainly for tests and ephemeral demos."""

    def __init__(self) -> None:
        self._histories: dict[str, InMemoryChatMessageHistory] = {}
        self._sessions: dict[str, ChatSession] = {}

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories[session_id]

    def save_session(self, session: ChatSession) -> None:
        self._sessions[session.session_id] = session

    def load_session(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[ChatSession]:
        return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def delete_session(self, session_id: str) -> None:
        self._histories.pop(session_id, None)
        self._sessions.pop(session_id, None)


class SQLiteSessionStore(SessionStore):
    """Durable store: two tables, one row per message.

    A single connection with ``check_same_thread=False`` is enough for a desktop
    app. Streamlit reruns the script on every interaction, so the store is
    cached with ``st.cache_resource`` in the UI.
    """

    def __init__(self, path: str | Path = "chatbot.db") -> None:
        super().__init__()
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
            )

    # ------------------------------------------------------------------ history
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        return _SQLiteHistory(self._connection, session_id)

    # ------------------------------------------------------------------ sessions
    def save_session(self, session: ChatSession) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title=excluded.title,
                    updated_at=excluded.updated_at,
                    message_count=excluded.message_count
                """,
                (
                    session.session_id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    session.message_count,
                ),
            )
        self._sessions[session.session_id] = session

    def load_session(self, session_id: str) -> ChatSession | None:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return ChatSession(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
        )

    def list_sessions(self) -> list[ChatSession]:
        rows = self._connection.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [
            ChatSession(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self._sessions.pop(session_id, None)

    def close(self) -> None:
        self._connection.close()


class _SQLiteHistory(BaseChatMessageHistory):
    """``BaseChatMessageHistory`` backed by the ``messages`` table."""

    def __init__(self, connection: sqlite3.Connection, session_id: str) -> None:
        self.connection = connection
        self.session_id = session_id
        self._ensure_session()

    def _ensure_session(self) -> None:
        exists = self.connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (self.session_id,)
        ).fetchone()
        if not exists:
            now = time.time()
            with self.connection:
                self.connection.execute(
                    "INSERT INTO sessions (session_id, title, created_at, updated_at, "
                    "message_count) VALUES (?, 'New conversation', ?, ?, 0)",
                    (self.session_id, now, now),
                )

    @property
    def messages(self) -> list[BaseMessage]:
        rows = self.connection.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (self.session_id,),
        ).fetchall()
        return [_to_message(row["role"], row["content"]) for row in rows]

    def add_message(self, message: BaseMessage) -> None:
        role = "human" if isinstance(message, HumanMessage) else "ai"
        with self.connection:
            self.connection.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (self.session_id, role, message.content, time.time()),
            )
            self.connection.execute(
                "UPDATE sessions SET updated_at = ?, message_count = message_count + 1 "
                "WHERE session_id = ?",
                (time.time(), self.session_id),
            )

    def clear(self) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (self.session_id,)
            )
            self.connection.execute(
                "UPDATE sessions SET message_count = 0 WHERE session_id = ?",
                (self.session_id,),
            )


def _to_message(role: str, content: str) -> BaseMessage:
    if role == "human":
        return HumanMessage(content=content)
    return AIMessage(content=content)


def window_messages(
    messages: Iterable[BaseMessage], max_turns: int
) -> list[BaseMessage]:
    """Keep the last ``max_turns`` user/assistant pairs.

    A naive ``messages[-N:]`` can start on an assistant message, which confuses
    some models; this helper always starts the window on a human message.
    """

    if max_turns <= 0:
        return []
    items = list(messages)
    if not items:
        return []
    limit = max_turns * 2
    if len(items) <= limit:
        return items
    trimmed = items[-limit:]
    while trimmed and not isinstance(trimmed[0], HumanMessage):
        trimmed = trimmed[1:]
    return trimmed


class TokenWindow:
    """Cheap character-based budget, used to bound the prompt size.

    Real tokenisers differ per model; for a demo a 4-chars-per-token heuristic
    is sufficient and keeps the dependency list small. The class is isolated so
    it can be replaced by ``tiktoken`` or the model's own counter.
    """

    def __init__(self, max_characters: int = 8000, chars_per_token: float = 4.0):
        self.max_characters = max_characters
        self.chars_per_token = chars_per_token

    def estimate(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)

    def fit(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        budget = self.max_characters
        kept: list[BaseMessage] = []
        for message in reversed(messages):
            size = len(str(message.content)) + 8
            if size > budget and kept:
                break
            budget -= size
            kept.append(message)
        kept.reverse()
        while kept and not isinstance(kept[0], HumanMessage):
            kept.pop(0)
        return kept


def export_session(store: SessionStore, session_id: str, path: str | Path) -> Path:
    """Write a session to JSON (useful for reports and reproducibility)."""

    history = store.get_history(session_id)
    session = store.load_session(session_id)
    payload = {
        "session": session.to_dict() if session else {"session_id": session_id},
        "messages": [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in history.messages
        ],
    }
    target = Path(path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

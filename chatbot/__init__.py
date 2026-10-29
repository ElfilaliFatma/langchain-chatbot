"""Conversational chatbot with LangChain.

Core pieces:

* :class:`ChatSession`      -- one conversation (memory + metadata)
* :class:`SessionStore`     -- persistence (in-memory or SQLite)
* :class:`ChatBot`          -- the LCEL chain + history-aware wrapper
"""

from .bot import ChatBot, build_default_bot
from .config import Settings
from .memory import ChatSession, SessionStore, SQLiteSessionStore, TokenWindow

__all__ = [
    "ChatBot",
    "ChatSession",
    "SessionStore",
    "SQLiteSessionStore",
    "Settings",
    "TokenWindow",
    "build_default_bot",
]

__version__ = "1.0.0"

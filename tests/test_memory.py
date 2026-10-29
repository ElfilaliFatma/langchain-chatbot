from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chatbot.memory import (
    ChatSession,
    SessionStore,
    SQLiteSessionStore,
    TokenWindow,
    export_session,
    window_messages,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield SessionStore()
    else:
        instance = SQLiteSessionStore(tmp_path / "test.db")
        yield instance
        instance.close()


def test_history_appends_and_reads(store):
    history = store.get_history("abc")
    history.add_message(HumanMessage("hi"))
    history.add_message(AIMessage("hello"))
    assert [m.content for m in history.messages] == ["hi", "hello"]


def test_histories_are_isolated(store):
    store.get_history("a").add_message(HumanMessage("first"))
    store.get_history("b").add_message(HumanMessage("second"))
    assert [m.content for m in store.get_history("a").messages] == ["first"]
    assert [m.content for m in store.get_history("b").messages] == ["second"]


def test_clear_empties_history(store):
    history = store.get_history("abc")
    history.add_message(HumanMessage("hi"))
    history.clear()
    assert history.messages == []


def test_sessions_round_trip(store):
    session = ChatSession(session_id="s1", title="Deploys")
    store.save_session(session)
    loaded = store.load_session("s1")
    assert loaded is not None and loaded.title == "Deploys"
    assert store.list_sessions()[0].session_id == "s1"
    store.delete_session("s1")
    assert store.load_session("s1") is None


def test_export_writes_json(store, tmp_path):
    store.get_history("s2").add_message(HumanMessage("question"))
    store.save_session(ChatSession(session_id="s2", title="T"))
    path = export_session(store, "s2", tmp_path / "out.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["messages"][0] == {"role": "human", "content": "question"}


def test_window_messages_keeps_last_turns_only():
    messages = []
    for index in range(6):
        messages.append(HumanMessage(f"q{index}"))
        messages.append(AIMessage(f"a{index}"))
    trimmed = window_messages(messages, 2)
    assert len(trimmed) == 4
    assert trimmed[0].content == "q4"
    assert isinstance(trimmed[0], HumanMessage)


def test_window_starts_on_human_even_with_odd_history():
    messages = [AIMessage("stray"), HumanMessage("q"), AIMessage("a")]
    assert window_messages(messages, 10) == messages
    trimmed = window_messages(messages, 1)
    assert [m.content for m in trimmed] == ["q", "a"]


def test_window_zero_returns_empty():
    assert window_messages([HumanMessage("q")], 0) == []


def test_token_window_drops_oldest_messages():
    window = TokenWindow(max_characters=20, chars_per_token=1)
    messages = [HumanMessage("x" * 10), AIMessage("y" * 10), HumanMessage("z" * 10)]
    kept = window.fit(messages)
    assert [m.content for m in kept] == ["z" * 10]


def test_token_window_estimate():
    assert TokenWindow(chars_per_token=4).estimate("abcdefgh") == 2

"""The chatbot: history-aware chain, streaming and session bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from .config import GUARDRAILS, Settings
from .memory import (
    ChatSession,
    SessionStore,
    SQLiteSessionStore,
    TokenWindow,
    window_messages,
)
from .prompts import TITLE_PROMPT, chat_prompt, persona_text


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
        )
    if settings.provider == "openai":
        if not settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
        )
    raise ValueError(f"Unsupported provider '{settings.provider}'.")


@dataclass
class Turn:
    session_id: str
    question: str
    answer: str


class ChatBot:
    """Conversation engine.

    The chain is ``prompt | llm | parser`` wrapped by
    ``RunnableWithMessageHistory``. The wrapper injects the stored history into
    the ``history`` placeholder and appends the new human/AI messages
    automatically, so the bot code never manipulates the message list by hand.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        store: SessionStore | None = None,
        settings: Settings | None = None,
        persona: str = "assistant",
    ):
        self.llm = llm
        self.settings = settings or Settings()
        self.store = store or SessionStore()
        self.persona = persona
        self.window = TokenWindow()

        base_chain = chat_prompt | llm | StrOutputParser()
        self.chain = RunnableWithMessageHistory(
            base_chain,
            get_session_history=self.store.get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        self.title_chain = TITLE_PROMPT | llm | StrOutputParser()

    # ------------------------------------------------------------------ sessions
    def new_session(self, title: str = "New conversation") -> ChatSession:
        session = ChatSession(title=title)
        self.store.save_session(session)
        return session

    def set_persona(self, persona: str) -> None:
        self.persona = persona

    def history(self, session_id: str):
        return self.store.get_history(session_id)

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"session_id": session_id}}

    def _inputs(self, text: str) -> dict:
        return {
            "input": text,
            "persona": persona_text(self.persona),
            "guardrails": GUARDRAILS,
            "today": date.today().isoformat(),
        }

    # ---------------------------------------------------------------------- chat
    def ask(self, session_id: str, text: str) -> Turn:
        text = (text or "").strip()
        if not text:
            raise ValueError("Message is empty.")
        self._prepare_history(session_id)
        answer = self.chain.invoke(self._inputs(text), config=self._config(session_id)).strip()
        self._bookkeeping(session_id, text, answer)
        return Turn(session_id, text, answer)

    def stream(self, session_id: str, text: str) -> Iterator[str]:
        """Yield answer chunks; the history is written when the stream ends."""

        text = (text or "").strip()
        if not text:
            raise ValueError("Message is empty.")
        self._prepare_history(session_id)
        chunks: list[str] = []
        for chunk in self.chain.stream(self._inputs(text), config=self._config(session_id)):
            if chunk:
                chunks.append(chunk)
                yield chunk
        answer = "".join(chunks).strip()
        self._normalize_last_answer(session_id, answer)
        self._bookkeeping(session_id, text, answer)

    # ------------------------------------------------------------------- internals
    def _prepare_history(self, session_id: str) -> None:
        """Trim the stored history in place before the run.

        ``RunnableWithMessageHistory`` reads the whole history, so the window
        has to be applied to the store itself. Trimming at *read* time would be
        better for auditability but requires a custom history wrapper; this
        trade-off is documented in the report.
        """

        history = self.store.get_history(session_id)
        messages = list(history.messages)
        trimmed = window_messages(messages, self.settings.history_turns)
        trimmed = self.window.fit(trimmed)
        if len(trimmed) != len(messages):
            history.clear()
            for message in trimmed:
                history.add_message(message)

    def _bookkeeping(self, session_id: str, question: str, answer: str) -> None:
        session = self.store.load_session(session_id)
        if session is None:
            session = ChatSession(session_id=session_id)
        elif session.title == "New conversation":
            session.title = self.suggest_title(question, answer)
        session.touch(0)
        self.store.save_session(session)

    def _normalize_last_answer(self, session_id: str, answer: str) -> None:
        """Trim whitespace that the streaming aggregator leaves in the store.

        Token-level streaming concatenates chunks verbatim, so answers routinely
        end with a trailing space. ``BaseChatMessageHistory`` exposes no update
        method, so the history is rewritten only when the stored text differs.
        """

        history = self.store.get_history(session_id)
        messages = list(history.messages)
        if not messages or messages[-1].type != "ai":
            return
        if messages[-1].content == answer:
            return
        messages[-1] = type(messages[-1])(content=answer)
        history.clear()
        for message in messages:
            history.add_message(message)

    def suggest_title(self, question: str, answer: str) -> str:
        """Cheap deterministic title, with a model fallback for long prompts."""

        text = (question or answer).strip().replace("\n", " ")
        if not text:
            return "New conversation"
        words = text.split()
        title = " ".join(words[:5])
        return (title[:48] + "…") if len(title) > 48 else title

    def suggest_title_with_model(self, question: str, answer: str) -> str:
        """Optional LLM title generation (one extra call)."""

        try:
            title = self.title_chain.invoke({"question": question, "answer": answer})
            return title.strip().strip('"').strip(".")[:60] or "New conversation"
        except Exception:
            return self.suggest_title(question, answer)


def build_default_bot(
    settings: Settings | None = None, persistent: bool = True, persona: str = "assistant"
) -> ChatBot:
    settings = settings or Settings.from_env()
    store: SessionStore
    if persistent:
        store = SQLiteSessionStore(settings.database_url)
    else:
        store = SessionStore()
    return ChatBot(build_llm(settings), store, settings, persona)

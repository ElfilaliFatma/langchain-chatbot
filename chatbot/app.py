"""Streamlit chat interface with session management and streaming."""

from __future__ import annotations

import streamlit as st

from chatbot.bot import ChatBot, build_default_bot
from chatbot.config import PERSONAS, Settings
from chatbot.memory import export_session

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bot(provider: str, model: str, temperature: float, persona: str) -> ChatBot:
    settings = Settings.from_env()
    settings = Settings(
        provider=provider,
        model=model,
        base_url=settings.base_url,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        history_turns=settings.history_turns,
        database_url=settings.database_url,
        api_key=settings.api_key,
    )
    return build_default_bot(settings, persistent=True, persona=persona)


def render_sidebar(bot: ChatBot) -> None:
    with st.sidebar:
        st.header("Conversations")
        if st.button("➕ New conversation", use_container_width=True):
            session = bot.new_session()
            st.session_state.session_id = session.session_id
            st.rerun()

        sessions = bot.store.list_sessions()
        for session in sessions:
            active = session.session_id == st.session_state.get("session_id")
            label = f"{'🟢' if active else '💬'} {session.title}"
            if st.button(label, key=f"s-{session.session_id}", use_container_width=True):
                st.session_state.session_id = session.session_id
                st.rerun()

        st.divider()
        st.caption(f"Persona: {bot.persona}")
        st.caption(f"History window: {bot.settings.history_turns} turns")
        if st.session_state.get("session_id"):
            if st.button("Export this conversation", use_container_width=True):
                path = export_session(
                    bot.store, st.session_state.session_id, "conversation.json"
                )
                st.download_button(
                    "Download JSON",
                    path.read_bytes(),
                    file_name="conversation.json",
                    mime="application/json",
                )


def main() -> None:
    st.title("🤖 LangChain Chatbot")
    st.caption("Multi-turn memory · personas · streaming · SQLite persistence")

    with st.sidebar:
        st.subheader("Model")
        provider = st.selectbox("Provider", ["ollama", "openai"])
        model = st.text_input("Model", "mistral")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        persona = st.selectbox("Persona", sorted(PERSONAS))

    bot = get_bot(provider, model, temperature, persona)
    if "session_id" not in st.session_state:
        st.session_state.session_id = bot.new_session().session_id
    session_id = st.session_state.session_id
    render_sidebar(bot)

    session = bot.store.load_session(session_id)
    st.subheader(session.title if session else "Conversation")

    history = bot.history(session_id).messages
    for message in history:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    question = st.chat_input("Ask anything…")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            collected = ""
            try:
                for chunk in bot.stream(session_id, question):
                    collected += chunk
                    placeholder.markdown(collected + "▌")
                placeholder.markdown(collected)
            except Exception as exc:
                placeholder.error(f"Generation failed: {exc}")
        st.rerun()


if __name__ == "__main__":
    main()

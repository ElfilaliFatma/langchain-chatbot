"""Prompt assembly for the chatbot."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import GUARDRAILS, PERSONAS

SYSTEM_TEMPLATE = "{persona}\n\n{guardrails}\n\nCurrent date: {today}."


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            ("human", "<user>\n{input}\n</user>"),
        ]
    )


chat_prompt = build_prompt()

TITLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Give a title of at most 5 words summarising the conversation. "
            "Reply with the title only, no quotes, no period.",
        ),
        ("human", "User: {question}\nAssistant: {answer}"),
    ]
)


def persona_text(name: str) -> str:
    return PERSONAS.get(name, PERSONAS["assistant"])


__all__ = ["chat_prompt", "TITLE_PROMPT", "persona_text", "GUARDRAILS", "PERSONAS"]

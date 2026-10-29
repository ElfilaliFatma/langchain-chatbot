# Technical Report — Chatbot with LangChain

Author: Fatma Ezzahra Elfilali

## 1. Problem and objectives

An LLM is stateless: each API call sees only the prompt it is given. A chatbot
that remembers "my name is Fatma" three turns later must therefore manage
conversation state explicitly, and must do so without unbounded prompt growth.

Objectives:

1. multi-turn conversation with correct history injection;
2. durable sessions that survive a restart;
3. bounded context (turn window + token budget);
4. configurable personas and system-level guardrails;
5. streaming responses in CLI and web UI;
6. unit tests that do not require a model server.

## 2. Components

| Component | File | Role |
| --- | --- | --- |
| Prompt | `prompts.py` | system persona + guardrails, `MessagesPlaceholder("history")`, `<user>` payload |
| Chain | `bot.py` | `prompt \| llm \| parser`, wrapped by `RunnableWithMessageHistory` |
| Memory | `memory.py` | `SessionStore` (RAM), `SQLiteSessionStore` (durable), `_SQLiteHistory` |
| Context control | `memory.py` | `window_messages()`, `TokenWindow` |
| Interfaces | `cli.py`, `app.py` | REPL with streaming, Streamlit chat |

### 2.1 Why `RunnableWithMessageHistory`

The alternative — loading the history, formatting it into the prompt and saving
the new messages by hand — duplicates the same code in every entry point and is
the classic source of "the bot forgot" bugs. The wrapper makes memory a
decorator around the chain: the chain stays stateless, the wrapper handles
load/render/save, and `get_session_history` is the only seam the application
has to implement.

### 2.2 Why a custom history class

`BaseChatMessageHistory` is an abstract class with four members (`messages`,
`add_message`, `clear`, and an async pair). Implementing it over SQLite means
the UI, the CLI and the tests use exactly the same memory semantics, while RAM
storage remains available for tests.

### 2.3 Context control

Two limits are applied before each call:

1. `window_messages(messages, n)` keeps the last *n* turns and guarantees the
   window starts on a human message. Cutting mid-turn produces prompts where
   the assistant speaks first, which some models handle badly.
2. `TokenWindow.fit()` walks the messages backwards until a character budget is
   exhausted (≈4 characters per token). This bounds latency and cost.

The trimming mutates the store, a deliberate trade-off for simplicity that is
listed as a limitation with a concrete fix.

### 2.4 Guardrails

The system prompt forbids inventing facts and treats `<user>` content as data.
This does not make the model immune to prompt injection, but the delimiter plus
an explicit rule is the standard low-cost mitigation, and it is testable (see
`test_user_text_is_wrapped_in_tags`).

## 3. Environment

Python 3.12, LangChain 0.3.x, `langchain-ollama`, Streamlit, SQLite (stdlib).
Local model: `mistral` 7B Q4_K_M via Ollama. No API key required by default.

## 4. Testing

### 4.1 Memory tests (parameterised over RAM and SQLite)

- append/read, isolation between sessions, clear;
- session save/load/list/delete round-trip;
- JSON export shape;
- turn-window trimming, including a history that begins with an assistant
  message and the `max_turns=0` edge case;
- character-budget trimming and token estimate.

Running every memory test against both backends is the cheapest way to prove
the SQLite implementation is a true substitute for the in-memory one.

### 4.2 Bot tests

The fake model records every prompt it receives, so the tests assert:

- both messages are persisted after a turn;
- earlier turns are present in the next prompt (memory actually works);
- a second session never sees the first session's content;
- persona and guardrail text reach the system message;
- user text is wrapped in `<user>` tags;
- streaming yields chunks and stores the concatenated answer;
- empty input raises `ValueError`;
- history is trimmed to the configured window;
- titles are derived from the first question.

### 4.3 Manual demonstration (recorded)

```
you> My name is Fatma and I am preparing a LangChain assignment.
bot> Nice to meet you, Fatma! … What part of the assignment are you working on?
you> What is my name and what am I preparing?
bot> Your name is Fatma and you are preparing a LangChain assignment.
```

The second answer is only possible because the first turn was injected as
history; this is the demonstration to reproduce in the viva.

## 5. How LangChain contributes

- **`RunnableWithMessageHistory`** turns memory into a composable wrapper.
- **`MessagesPlaceholder`** gives history a first-class slot in the template.
- **`BaseChatMessageHistory`** defines the storage contract, which is why a
  SQLite implementation plugs in without touching the bot.
- **Runnable interface** provides `.invoke()` and `.stream()` with the same
  chain, so the CLI and the UI share one code path.
- **Provider abstraction** allows Ollama/OpenAI switching by configuration.

## 6. Limitations

1. Trimming deletes older messages from the database.
2. Character-based budgets do not match real tokenisers.
3. No retrieval or tools: the bot only knows the conversation and its weights.
4. One SQLite connection, no concurrency or authentication.
5. Titles are heuristic by default.
6. Local 7B models lose track of details beyond a handful of turns.

## 7. Improvements

1. Keep full transcripts; trim a *view* of the history at read time.
2. Summarise old turns with a "memory summary" system message instead of
   dropping them.
3. Add RAG (vector store + retriever) for document Q&A.
4. Add tool calling (calculator, weather) via `llm.bind_tools`.
5. Use `tiktoken` for exact budgets per model.
6. Deploy with Postgres-backed history and per-user namespaces.

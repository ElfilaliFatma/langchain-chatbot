# Chatbot with LangChain

A conversational assistant with real multi-turn memory, selectable personas,
token streaming and durable SQLite sessions.

![architecture](docs/architecture.md)

## What is implemented

| Feature | Implementation |
| --- | --- |
| Multi-turn memory | `RunnableWithMessageHistory` + `MessagesPlaceholder("history")` |
| Persistence | `SQLiteSessionStore` implementing `BaseChatMessageHistory` |
| Context control | `window_messages()` (whole turns) + `TokenWindow` (character budget) |
| Personas | tutor / career coach / translator / general assistant |
| Guardrails | system-level rules + `<user>` delimiters against prompt injection |
| Streaming | `chain.stream()` consumed by the CLI and the Streamlit UI |
| Session titles | deterministic 5-word title, optional LLM titles |
| Export | `export_session()` writes the transcript to JSON |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
ollama pull mistral
copy .env.example .env
```

### CLI

```powershell
python -m chatbot.cli                                  # interactive
python -m chatbot.cli --persona tutor -q "What is a closure?"
python -m chatbot.cli --list
python -m chatbot.cli --resume 4f2a9c1d8e7b
python -m chatbot.cli --resume 4f2a9c1d8e7b --export chat.json
```

### Web UI

```powershell
streamlit run chatbot/app.py
```

### Tests

```powershell
pytest
```

Tests use a scripted, stream-capable fake model and a temporary SQLite file, so
they run offline and leave no state behind.

## Layout

```
chatbot/
  config.py   settings + personas + guardrails
  prompts.py  prompt assembly (system / history / user)
  memory.py   sessions, SQLite history, trimming, export
  bot.py      ChatBot, RunnableWithMessageHistory, streaming
  cli.py      REPL and one-shot mode
  app.py      Streamlit UI
tests/
```

## Why LangChain

`RunnableWithMessageHistory` removes the main source of bugs in chat apps —
manually splicing history into the prompt and forgetting to save messages. The
`MessagesPlaceholder` gives a typed slot for history, the runnable interface
gives `.invoke()` and `.stream()` for free, and `BaseChatMessageHistory` is the
only contract a custom store must satisfy (here: SQLite).

## Limitations

- Trimming mutates stored history, so the database keeps only the recent window.
- The character-based token estimate is approximate.
- No retrieval: the bot cannot answer from private documents.
- Single-user desktop scale (one SQLite connection, no auth).
- No tool calling or web access.

## Improvements

- Keep full history in SQLite and trim only at read time (wrapper around the
  history object), so transcripts stay complete for audit.
- Add RAG with a vector store for document questions.
- Summarise older turns into a "memory summary" message instead of dropping them.
- Add tools (weather, calculator) with `bind_tools`.
- Add `tiktoken`-based budgeting and a per-model context limit.
- Add authentication and per-user session namespaces for deployment.

Full write-up: [`docs/REPORT.md`](docs/REPORT.md).

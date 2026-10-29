# Architecture

```
User text
   │
   ├─► <user>…</user> wrapper            (injection hardening)
   │
   ▼
ChatPromptTemplate
   ├─ system : persona + guardrails + date
   ├─ history: MessagesPlaceholder  ◄──── RunnableWithMessageHistory
   └─ human  : {input}                       │
                                              │ get_session_history(session_id)
                                              ▼
                                     SessionStore / SQLiteSessionStore
                                              │
                                              ▼
                                    messages table (role, content, ts)
   ▼
ChatModel (Ollama / OpenAI)  ──► StrOutputParser ──► answer
   │                                   │
   └──────── stream() ────────────────►┘
```

## Request lifecycle

1. `_prepare_history()` trims the stored history to `history_turns` whole turns
   and to the `TokenWindow` character budget.
2. `RunnableWithMessageHistory` loads the trimmed history, renders the prompt
   and calls the model.
3. The wrapper appends the human message and the AI answer to the store.
4. `_bookkeeping()` updates the session timestamp and, on the first turn,
   derives a title.

## Storage

```sql
sessions(session_id PK, title, created_at, updated_at, message_count)
messages(id PK, session_id FK, role, content, created_at)
```

`_SQLiteHistory` translates `BaseChatMessageHistory` calls into SQL, which is
the only integration point between LangChain and the database.

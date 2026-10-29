"""Interactive REPL and one-shot mode.

    python -m chatbot.cli                 # interactive, persistent sessions
    python -m chatbot.cli --list          # list saved sessions
    python -m chatbot.cli -q "hello"      # one-shot
    python -m chatbot.cli --resume <id>   # continue a stored conversation
"""

from __future__ import annotations

import argparse
import sys
import warnings

from .bot import build_default_bot
from .memory import export_session
from .prompts import PERSONAS

# LangChain warns that RunnableWithMessageHistory is slated for replacement by
# LangGraph persistence. The project deliberately targets the LangChain API
# (the migration path is documented in docs/REPORT.md), so the warning is
# suppressed here to keep demo output readable.
warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory is deprecated.*")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="chatbot", description="LangChain chatbot.")
    parser.add_argument("-q", "--question", help="Ask one question and exit.")
    parser.add_argument("--persona", default="assistant", choices=sorted(PERSONAS))
    parser.add_argument("--resume", help="Resume an existing session id.")
    parser.add_argument("--list", action="store_true", help="List stored sessions.")
    parser.add_argument("--export", help="Export the session to JSON and exit.")
    parser.add_argument("--ephemeral", action="store_true", help="Do not persist to SQLite.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; model answers in Arabic or with emoji
    # would raise UnicodeEncodeError on print without this.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    bot = build_default_bot(persistent=not args.ephemeral, persona=args.persona)

    if args.list:
        sessions = bot.store.list_sessions()
        if not sessions:
            print("No stored sessions.")
        for session in sessions:
            print(f"{session.session_id}  {session.message_count:>3} msg  {session.title}")
        return 0

    session = (
        bot.store.load_session(args.resume) or bot.new_session()
        if args.resume
        else bot.new_session()
    )
    if args.resume and bot.store.load_session(args.resume) is None:
        print(f"Unknown session '{args.resume}', starting a new one.", file=sys.stderr)

    if args.export:
        path = export_session(bot.store, session.session_id, args.export)
        print(f"Exported to {path}")
        return 0

    if args.question:
        print(bot.ask(session.session_id, args.question).answer)
        return 0

    print(f"Session {session.session_id} · persona={bot.persona}")
    print("Type 'exit' to quit, 'reset' to clear the history.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        if text.lower() == "reset":
            bot.history(session.session_id).clear()
            print("History cleared.")
            continue
        print("bot> ", end="", flush=True)
        try:
            for chunk in bot.stream(session.session_id, text):
                print(chunk, end="", flush=True)
            print()
        except Exception as exc:
            print(f"\n[error] {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

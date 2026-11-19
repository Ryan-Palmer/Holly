"""Milestone 1: talk to Holly in text, at the command line.

    python -m app.chat [--model MODEL] [--no-timing]

Commands: /reset clears the conversation, /quit leaves.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .lmstudio import LmStudio, ReplyTiming
from .persona import Conversation
from .speech_text import speech_problems

DIM, RESET, CYAN, YELLOW = "\033[2m", "\033[0m", "\033[36m", "\033[33m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Text chat with Holly")
    parser.add_argument("--model", help="LM Studio model key (default: from holly.toml)")
    parser.add_argument("--no-timing", action="store_true", help="hide latency figures")
    args = parser.parse_args()
    # The Windows console defaults to cp1252, which can't print emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config()
    model = args.model or cfg.llm.model
    lm = LmStudio(cfg.llm)
    convo = Conversation(cfg.persona)

    print(f"{DIM}Loading {model}...{RESET}", end="", flush=True)
    lm.ensure_loaded(model)
    print(f"\r{DIM}Holly is on {model}. /reset to start over, /quit to leave.{RESET}\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/quit", "/exit"):
            return
        if user == "/reset":
            convo.reset()
            print(f"{DIM}Conversation cleared.{RESET}\n")
            continue

        timing = ReplyTiming()
        parts: list[str] = []
        print(f"{CYAN}Holly:{RESET} ", end="", flush=True)
        try:
            for delta in lm.stream_reply(convo.messages_for(user), timing, model=model):
                parts.append(delta)
                print(delta, end="", flush=True)
        except KeyboardInterrupt:
            print(f" {DIM}[interrupted]{RESET}")
        print()

        reply = "".join(parts).strip()
        convo.record(user, reply)

        problems = speech_problems(reply)
        if problems:
            print(f"{YELLOW}  unspeakable: {', '.join(problems)}{RESET}")
        if not args.no_timing:
            first = f"{timing.first_token_ms:.0f}" if timing.first_token_ms is not None else "-"
            thinking = f", thought {timing.reasoning_chars} chars" if timing.reasoning_chars else ""
            print(f"{DIM}  first token {first} ms, total {timing.total_ms:.0f} ms{thinking}{RESET}")
        print()


if __name__ == "__main__":
    sys.exit(main())

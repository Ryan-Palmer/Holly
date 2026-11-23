"""The console's colours, and the time at the start of a line of talk."""

from __future__ import annotations

import datetime

DIM, RESET, CYAN, YELLOW = "\033[2m", "\033[0m", "\033[36m", "\033[33m"


def stamp() -> str:
    """The time now, dimmed, to start a line of talk in the console."""
    return f"{DIM}{datetime.datetime.now():%H:%M:%S}{RESET} "


def note(text: str, colour: str = DIM) -> None:
    """A line about what Holly's doing, dimmed (or in `colour`), not part of the talk."""
    print(f"{colour}  ({text}){RESET}", flush=True)

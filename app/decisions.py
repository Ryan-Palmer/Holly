"""Holly's side checks, in one place: each small question the model is asked about a turn
before he answers it. Is a status report wanted? Have they taken him up on his offer, and
with which tool? Is a timer come due a job for a tool? Do they want something from the
past? Which memory, if any, fits what's just been said?

Each is a Decision: what it's asked (its prompt, beside the code it serves, and the line
made from its inputs), the answers it may give (its first word: YES or NO, a tool's name, a
number), what's made of them, what if LM Studio can't answer, and its labelled cases
(app/labelled.py), which the evals run it on (`--only side`, `--only memory`). Their
answers are short and free text, not JSON: they're asked before he answers, where every
token is waited for, and the evals read how sure each answer was from its log
probabilities, which LM Studio's structured output leaves out.

Holly asks them through Holly._decide, by name: the event log keeps each answer under it,
and a replay gives the answer it gave then (app/replay.py)."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class Decision:
    name: str  # its key in the event log and the registry
    prompt: str  # what it's told
    asked: Callable[..., str]  # what it's asked, from its inputs
    answers: str  # what it may answer, for reading: "YES / NO", "a tool's name", "a number or NONE"
    read: Callable[..., Any]  # read(text, *inputs): what's made of its answer
    fallback: Any = False  # if LM Studio can't answer, and it's asked quietly
    max_tokens: int = 4

    def ask(self, lm, model: str | None, *inputs, quiet: bool = True):
        """Its answer for these inputs. If LM Studio can't answer: the fallback, quietly
        (as Holly asks: he answers as he would have), or with `quiet` off (the evals, and the
        checks whose callers handle it), the error."""
        try:
            text = lm.ask(self.prompt, self.asked(*inputs), model, max_tokens=self.max_tokens, temperature=0)
        except Exception:
            if not quiet:
                raise
            return self.fallback
        return self.read(text, *inputs)


REGISTRY: dict[str, Decision] = {}


def register(d: Decision) -> Decision:
    assert d.name not in REGISTRY, d.name
    REGISTRY[d.name] = d
    return d


# Readers ----------------------------------------------------------------------------


def starts(word: str) -> Callable[..., bool]:
    """Does the answer start with `word` (YES, PAST, DO)?"""
    return lambda text, *inputs: text.strip().upper().startswith(word)


def tool_named(text: str, tools) -> str | None:
    """The tool its answer names, if it's one of `tools`."""
    text = text.strip().lower()
    word = text.split()[0].strip(".,:;!\"'") if text else ""
    return word if word in tools else None


def numbered(text: str) -> int:
    """The number its answer starts with; 0 for none."""
    m = re.match(r"\W*(\d+)", text)
    return int(m.group(1)) if m else 0


def all_decisions() -> dict[str, Decision]:
    """Every decision, registered: the modules that hold them, imported."""
    from . import archive, offers, status, timers  # noqa: F401

    return REGISTRY

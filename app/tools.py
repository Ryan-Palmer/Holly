"""Holly's tools: what each is (its schema, as the model sees it), his rules for it (the
.md beside its module, in his system prompt), and running it.

Each tool's class lives in its own module (ClockTool in app/clock.py, RecallTool in
app/archive.py...). A Toolbox holds the ones he has, in a fixed order: every request
sends the same tools, at the start of the prompt, so changing them would break LM
Studio's cache.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

from .console import DIM, RESET, YELLOW, note


def rules(module: str) -> str:
    """A tool's rules for Holly, from the .md beside its module: rules("search") reads
    app/search.md."""
    return Path(__file__).with_name(f"{module}.md").read_text(encoding="utf-8")


@dataclasses.dataclass
class Turn:
    """What a tool may need of the reply it's part of."""

    user: str  # the crew's words
    number: int  # replies so far this session, this one included
    times: object  # the reply's TurnTimes (app/speak.py): lookups add their time to search_ms
    filler: Callable[[str], None] = lambda kind: None  # play a stock line of this kind now (on the event loop)
    answering_choice: bool = False  # the crew are answering his search-or-report question
    crew_lines: Callable[[], str] = lambda: ""  # what they've said in his history (Conversation.crew_lines)


class Tool:
    name = ""
    schema: dict = {}
    rules: str | None = None  # the module whose .md has his rules for it, if it has its own
    local = False  # instant (no stock line), and not counted among the lookups a reply may make
    filler = "search"  # the stock line, by kind, that covers a slow one's wait

    async def run(self, args: dict, turn: Turn) -> str:
        """Do it; returns what goes back to the model."""
        raise NotImplementedError


class Lookup(Tool):
    """A slow tool that finds something out (search, weather, recall), in a worker thread."""

    async def look_up(self, label: str, job: Callable, turn: Turn):
        """Run `job` (returning app.search.Lookup); or if it breaks, what he's told instead."""
        print(f"{DIM}  ({label}){RESET}", flush=True)
        try:
            found = await asyncio.to_thread(job)
        except Exception as e:  # LM Studio, a provider or the weather service broke: he says he couldn't
            note(f"{self.name} failed: {e!r}", YELLOW)
            return (f"The {self.name} tool broke ({type(e).__name__}). Tell the crew you couldn't find "
                    "out, in your own words, and give no time, number, date or name for it.")
        turn.times.search_ms += found.search_ms + found.summary_ms
        again = f"; searched again: {'; '.join(found.queries[1:])}" if len(found.queries) > 1 else ""
        note(f"{found.results} results{again}, {found.search_ms:.0f} + {found.summary_ms:.0f} ms: {found.answer}")
        return found


class Toolbox:
    def __init__(self, tools: list[Tool], max_lookups: int):
        self.tools = tools
        self.by_name = {t.name: t for t in tools}
        self.max_lookups = max_lookups

    def __contains__(self, name: str) -> bool:
        return name in self.by_name

    def schemas(self) -> list[dict] | None:
        return [t.schema for t in self.tools] or None

    def rules(self) -> str:
        """His rules for all of them, each .md once, in order."""
        return "\n\n".join(rules(m) for m in dict.fromkeys(t.rules for t in self.tools if t.rules))

    def local(self, name: str) -> bool:
        return name in self.by_name and self.by_name[name].local

    def filler(self, names: list[str]) -> str | None:
        """The stock line for a step calling these tools: none if they're all instant; the
        recall line if they're all recalls; else the search line."""
        slow = [self.by_name[n] for n in names if n in self.by_name and not self.by_name[n].local]
        if not slow:
            return None
        return slow[0].filler if all(t.filler == slow[0].filler for t in slow) else "search"

    async def run(self, call: dict, turn: Turn, lookups: int) -> str:
        """Run one tool call ({"name", "arguments"}); `lookups`: slow ones so far this reply,
        this one included. Returns what goes back to the model."""
        print()
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        tool = self.by_name.get(call["name"])
        if tool is None:
            return f"There is no tool called {call['name']}."
        if not tool.local and lookups > self.max_lookups:
            return "That's enough searching. Answer with what you have."
        return await tool.run(args, turn)

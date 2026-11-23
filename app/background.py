"""What Holly does while he sleeps: his notes, reminders from the ship's records, and
research, one at a time and in that order. Anyone speaking stops it.

Listening (app/listen.py), he dozes once he's been quiet a while; typed to (app/speak.py),
whenever he isn't replying. Either way it's `doze()` to start whatever's due and `wake()`
when someone speaks: notes being written are dropped (written again at the next sleep),
and so is the research step in hand (each step is saved only when it's done).
"""

from __future__ import annotations

import asyncio
import time

from . import memory
from .console import DIM, RESET, note
from .records import reminders as pick_reminders, source_of

NOTES_RETRY_S = 60.0  # after his notes failed to be written, wait this long before trying again
RESEARCH_RETRY_S = 60.0  # after a research step failed (LM Studio, the network), wait this long


def context_report(cfg, convo, used: int | None) -> str:
    """How full his context is, and how far his history is from being folded into notes."""
    size = cfg.llm.context_length
    soft = cfg.persona.history_soft_tokens
    history = convo.tokens()
    # `used` is LM Studio's count; the history's is characters / 4, which runs ~10% high.
    report = f"context {used:,} of {size:,} tokens" if used else f"context ? of {size:,} tokens"
    if history > soft:
        return report + "; older history goes into his notes when he next sleeps"
    return report + f"; ~{soft - history:,} more before older history goes into his notes"


class Background:
    def __init__(self, cfg, convo, lm, model: str, tools: list[dict] | None, archive=None, records=None,
                 research=None):
        """`tools`: the schemas his replies send, for having LM Studio reread his prompt."""
        self.cfg, self.convo, self.lm, self.model, self.tools = cfg, convo, lm, model, tools
        self.archive, self.records, self.research = archive, records, research
        self.reminded = False  # the reminders fit the talk so far
        self.writing_notes = False  # the part of compacting that's safe to cancel
        self.notes: asyncio.Task | None = None  # older history into his notes, and reminders
        self.notes_retry_at = 0.0  # after a failed try, not before this
        self.researching: asyncio.Task | None = None  # research steps
        self.research_retry_at = 0.0

    # Starting and stopping --------------------------------------------------

    def doze(self) -> None:
        """He's asleep: start whatever's due, if it isn't under way. Notes first;
        research waits for them."""
        self.start_notes()
        self.start_research()

    def wake(self) -> None:
        """Someone's speaking, or he's about to: drop notes being written, and the
        research step in hand. Both are done again at his next sleep."""
        if self.notes and self.writing_notes:
            # Not finished: the history is left as it was. The reread after the notes
            # can't be stopped, and costs nothing extra: the reply would reread anyway.
            self.notes.cancel()
        if self.researching:
            self.researching.cancel()

    def stop(self) -> None:
        """He's shutting down."""
        for task in (self.notes, self.researching):
            if task:
                task.cancel()

    # His notes and reminders ------------------------------------------------

    def notes_due(self) -> bool:
        """His notes (or reminders from the records) want writing, and may be tried now."""
        return bool(self.convo.needs_compacting() or (self.records and not self.reminded)) and (
            time.perf_counter() >= self.notes_retry_at)

    def start_notes(self) -> None:
        if self.notes or self.researching or not self.notes_due():
            return
        self.notes = asyncio.create_task(self._sleep_work())
        self.notes.add_done_callback(self._noted)

    async def _sleep_work(self) -> bool:
        """False if the notes couldn't be written."""
        ok = True
        if self.convo.needs_compacting():
            ok = await self.compact()
        if self.records and not self.reminded:
            try:
                await self.refresh_reminders()
            except Exception as e:  # the notes may be fine: not their failure, nor their retry
                self.reminded = True  # try again after the next talk, not straight away
                note(f"couldn't pick reminders from the ship's records: {e!r}")
        return ok

    def _noted(self, task: asyncio.Task) -> None:
        self.notes = None
        if task.cancelled():
            return  # someone spoke: tried again later, no wait
        if (e := task.exception()) or not task.result():
            # LM Studio refused, or the notes came back empty: not again for a while,
            # rather than once per mic block.
            self.notes_retry_at = time.perf_counter() + NOTES_RETRY_S
            note(f"couldn't write his notes{f': {e!r}' if e else ''}; trying again in {NOTES_RETRY_S:.0f} s")

    async def compact(self, end: int | None = None, label: str = "asleep") -> bool:
        """Fold the older history, history[:end], into his notes (app/memory.py), then
        have LM Studio read the new, shorter prompt, so his next reply doesn't wait for
        it. Also at startup, for the last session. Returns True if it was compacted."""
        convo = self.convo
        if not (end := end or convo.compaction_split()):
            return False
        dropped = convo.history[:end]
        exchanges = sum(m["role"] == "user" for m in dropped)
        before = convo.tokens()
        t0 = time.perf_counter()
        self.writing_notes = True
        try:
            notes = await memory.write_notes(self.lm, self.model, convo.memory, convo.transcript(end),
                                             self.cfg.persona.memory_tokens)
        finally:
            self.writing_notes = False
        if not notes or not convo.apply_compaction(end, dropped, notes):
            return False
        self.reminded = False  # the talk has moved on: reminders to fit it
        t1 = time.perf_counter()
        prompt = await self.lm.prewarm(convo.messages_for(), self.model, self.tools)
        if self.archive:  # the archived exchanges' vectors, now rather than at his next recall
            try:
                await asyncio.to_thread(self.archive.embed_missing)
            except Exception as e:
                note(f"couldn't embed the archive: {e!r}")
        print(f"{DIM}  ({label}: folded {exchanges} exchanges into his notes, ~{len(notes) // 4} tokens; "
              f"history ~{before:,} -> ~{convo.tokens():,} tokens; {t1 - t0:.1f} s to write, "
              f"{time.perf_counter() - t1:.1f} s to reread)\n  {context_report(self.cfg, convo, prompt)}{RESET}",
              flush=True)
        return True

    async def refresh_reminders(self, prewarm: bool = True) -> None:
        """A few entries from the ship's records for his system prompt, close to what's
        been said lately (app/records.py); then, if they changed, LM Studio rereads the
        prompt, as after his notes."""
        picks = await asyncio.to_thread(pick_reminders, self.records, self.convo.recent_topics(),
                                        self.cfg.recall.reminders)
        self.reminded = True
        if picks == self.convo.reminders:
            return
        self.convo.reminders = picks
        note(f"from the ship's records, on his mind: {'; '.join(source_of(p) for p in picks)}")
        if prewarm:
            await self.lm.prewarm(self.convo.messages_for(), self.model, self.tools)

    # Research ---------------------------------------------------------------

    def start_research(self) -> None:
        """The next steps of any research jobs (app/research.py), one at a time. Not while
        his notes are due or being written: they come first."""
        if (not self.research or self.researching or self.notes or self.notes_due()
                or time.perf_counter() < self.research_retry_at or not self.research.has_work()):
            return
        self.researching = asyncio.create_task(self._research())
        self.researching.add_done_callback(self._researched)

    async def _research(self) -> None:
        while self.research.has_work() and not self.notes_due():
            note(f"research {await self.research.step()}")
        if not self.research.has_work():
            # All done: his own prompt back into LM Studio's cache, for his next reply.
            await self.lm.prewarm(self.convo.messages_for(), self.model, self.tools)

    def _researched(self, task: asyncio.Task) -> None:
        self.researching = None
        if task.cancelled():
            return  # someone spoke: carried on at the next sleep
        if e := task.exception():
            self.research_retry_at = time.perf_counter() + RESEARCH_RETRY_S
            note(f"a research step failed: {e!r}; trying again in {RESEARCH_RETRY_S:.0f} s")

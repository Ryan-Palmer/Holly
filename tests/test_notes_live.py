"""His notes (app/memory.py) written by the real model, from a transcript made as his
history makes it. Skipped unless run with `pytest -m lmstudio`, with LM Studio up and no
Holly running."""

import asyncio
import datetime as dt
import re

import pytest

from app.memory import write_notes
from app.persona import Conversation

pytestmark = pytest.mark.lmstudio


@pytest.fixture
def lm(lmstudio):
    """A client of its own for each test: its async connections belong to one event loop."""
    from app.config import load_config
    from app.evals import alone
    from app.lmstudio import LmStudio

    alone(load_config().llm.model)  # with anything else unloaded: a judge left from the evals, say
    return LmStudio(load_config().llm)


def transcript(cfg, exchanges: list[tuple[dt.datetime, str, str]]) -> str:
    c = Conversation(cfg.persona, "", None, None)
    for at, crew, said in exchanges:
        c.history += [{"role": "user", "content": crew, "at": at.isoformat()}, {"role": "assistant", "content": said}]
    return c.transcript(len(c.history))


def test_a_visit_tomorrow_is_noted_on_its_own_day(cfg, lm):
    """"Coming over tomorrow" was noted on the day it was said: Gemma's own sums were off,
    so the day words are dated in the transcript, for the notes to copy."""
    wed = dt.datetime.combine(dt.date.today() - dt.timedelta(days=1), dt.time(19, 10))
    visit = wed.date() + dt.timedelta(days=1)
    text = transcript(cfg, [(wed, "Jo's coming over tomorrow to help with the garden.", "Bring a spade."),
                            (wed + dt.timedelta(minutes=5), "She's bringing her dog.", "Lock up the cat.")])
    assert f"[{visit:%A} {visit.day} {visit:%B}" in text  # dated in code
    notes = asyncio.run(write_notes(lm, lm.cfg.model, "", text, 300))
    line = next((ln for ln in notes.splitlines() if re.search(r"\bgarden", ln, re.IGNORECASE)), "")
    assert f"{visit.day} {visit:%B}" in line, notes
    assert "tomorrow" not in line.lower(), notes


def test_a_recalled_show_memory_isnt_noted_as_the_crew(cfg, lm):
    """Given what recall found about Red Dwarf, the notes filled with it: "About the crew:
    Lister, Cat...". Of a recall, only what the crew said is folded in."""
    c = Conversation(cfg.persona, "", None, None)
    at = dt.datetime.combine(dt.date.today() - dt.timedelta(days=1), dt.time(20, 0)).isoformat()
    c.history += [
        {"role": "user", "content": "Do you remember when the crew met the cat race?", "at": at},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "recall", "arguments": '{"query": "cat race"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Lister was taken for their god, Cloister; Rimmer and the "
                                                          "Cat met two old cat priests."},
        {"role": "assistant", "content": "They thought Lister was their god. Cloister the Stupid."},
        {"role": "user", "content": "I'm off to Keswick on Saturday with Jo.", "at": at},
        {"role": "assistant", "content": "Take a coat."}]
    notes = asyncio.run(write_notes(lm, lm.cfg.model, "", c.transcript(len(c.history)), 300))
    crew = notes.split("Lately:")[0]
    assert "Keswick" in notes, notes
    assert not re.search(r"\b(?:Lister|Rimmer|Cloister)\b", crew), notes

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


def call(name: str, args: str, result: str, reply: str, crew: str, at: str) -> list[dict]:
    return [{"role": "user", "content": crew, "at": at},
            {"role": "assistant", "content": "", "tool_calls": [{"id": name, "type": "function", "function": {
                "name": name, "arguments": args}}]},
            {"role": "tool", "tool_call_id": name, "content": result},
            {"role": "assistant", "content": reply}]


def test_what_they_asked_for_isnt_a_note(cfg, lm):
    """His notes said "asked for the weather at Keswick", "asked for a status report", "asked
    for a report on dashcams": what they asked him for, a line each, and nothing to keep (his
    reports and archive have them). What they told him about themselves stays."""
    c = Conversation(cfg.persona, "", None, None)
    at = dt.datetime.combine(dt.date.today() - dt.timedelta(days=1), dt.time(18, 0)).isoformat()
    c.history += [
        *call("weather", '{"place": "Keswick"}', "Keswick tomorrow: rain, 4 to 9 degrees.", "Rain. Naturally.",
              "What's the weather doing in Keswick tomorrow?", at),
        *call("status", "{}", "Lister lost a bet with a skutter. Rimmer filed a complaint.",
              "Lister lost a bet with a skutter. Rimmer filed a complaint.", "Status report, please.", at),
        *call("research", '{"action": "start", "title": "Dashcams"}', "Queued: research on 'Dashcams'.",
              "I'll look into it.", "Can you write me a report on dashcams for my mum's car?", at),
        {"role": "user", "content": "My mum's just bought a new car, a little red hatchback.", "at": at},
        {"role": "assistant", "content": "Red. Bold."},
        {"role": "user", "content": "Sam's coming to stay on Saturday.", "at": at},
        {"role": "assistant", "content": "Hide the biscuits."}]
    notes = asyncio.run(write_notes(lm, lm.cfg.model, "", c.transcript(len(c.history)), 300))
    assert not re.search(r"\b(?:asked|requested|wanted)\b", notes, re.IGNORECASE), notes
    assert not re.search(r"\bstatus report|\bLister\b", notes, re.IGNORECASE), notes
    assert "Sam" in notes and re.search(r"\bcar\b|hatchback", notes, re.IGNORECASE), notes


EARLIER = """About the crew:
The crew's sister is Jo
Jo is a vet in Keswick
The crew likes fish pie

Lately:
Wednesday 25 November 2026: the crew asked for a status report
Wednesday 25 November 2026: the crew asked for the weather in Keswick
Wednesday 25 November 2026: the crew asked for a report on dashcams for their mum's car
Wednesday 25 November 2026: the crew asked for the weather in Keswick for Thursday 26 November"""


def test_asked_for_lines_already_in_his_notes_go(cfg, lm):
    """Once written, an "asked for" line was kept fold after fold: told to start from his
    notes and keep what still holds, it kept four of four (27 November)."""
    at = dt.datetime.combine(dt.date.today(), dt.time(9, 0))
    notes = asyncio.run(write_notes(lm, lm.cfg.model, EARLIER, transcript(
        cfg, [(at, "Sam's coming to stay on Saturday.", "Hide the biscuits.")]), 300))
    assert not re.search(r"\basked\b", notes, re.IGNORECASE), notes
    assert "Jo" in notes and "Keswick" in notes and "fish pie" in notes and "Sam" in notes, notes


def full_notes(today: dt.date) -> str:
    """Notes near his budget (~1,000 words of 1,125): months of facts, and weeks of dated news,
    most of it past."""
    facts = ["The crew's sister is Jo, a vet in Penrith.", "The crew has a ginger cat called Biscuit.",
             "The crew works at the library three days a week.", "The crew can't stand mushrooms.",
             "The crew's brother Sam lives in Manchester and hates the rain."]
    hobbies = ["jigsaws", "the ukulele", "gardening", "crosswords", "baking bread", "birdwatching", "chess",
               "walking in the fells", "old films", "model railways", "knitting", "pub quizzes"]
    facts += [f"The crew enjoys {h}, and has done for years, mostly at weekends." for h in hobbies]
    # Not "June": with an old friend called June, "Jo's expecting a baby in June!" was dropped, 2 in 2
    # (a month that's a name in his notes; renamed, kept 2 in 2). Known, not what this tests.
    friends = ["Jane", "Ed", "Priya", "Tom", "Alice", "Ben", "Carla", "Dev", "Ellen", "Frank", "Grace", "Hugh"]
    facts += [f"{f} is an old friend of the crew's from school, and they meet up now and then." for f in friends]
    lately = []
    for k in range(36, 1, -1):
        d = today - dt.timedelta(days=k)
        lately.append(f"{d:%A} {d.day} {d:%B} {d.year}: the crew went to see {friends[k % 12]} "
                      f"and talked about {hobbies[k % 12]} for most of the afternoon.")
    return "About the crew:\n" + "\n".join(facts) + "\n\nLately:\n" + "\n".join(lately)


def test_full_notes_take_in_the_new_and_keep_what_lasts(cfg, lm):
    """His notes fill up over months. Past the budget, the fold's output is cut, and what's cut
    is the end: "Lately", the newest news. Full, they should make room (old news first), not
    lose what's new, nor what lasts."""
    today = dt.date.today()
    earlier = full_notes(today)
    budget = cfg.persona.memory_tokens * 3 // 4
    assert budget * 0.85 < len(earlier.split()) <= budget, len(earlier.split())
    at = dt.datetime.combine(today, dt.time(19, 0))
    text = transcript(cfg, [(at, "Big news: Jo's expecting a baby in June!", "A small vet. Congratulations."),
                            (at + dt.timedelta(minutes=5), "Our new neighbour's called Gerald. He keeps bees.",
                             "Bees. Lock the jam up."),
                            (at + dt.timedelta(minutes=10), "My dentist appointment's on Tuesday at ten.",
                             "Floss. Now.")])
    notes = asyncio.run(write_notes(lm, lm.cfg.model, earlier, text, cfg.persona.memory_tokens))
    low = notes.lower()
    assert "baby" in low and "gerald" in low and "bee" in low and "dentist" in low, notes  # the new
    assert all(w in low for w in ("vet", "biscuit", "library", "mushroom", "sam")), notes  # what lasts
    assert all(f in notes for f in ("Jane", "Ed", "Priya", "Grace", "Hugh")), notes  # old friends last too
    assert all(h in low for h in ("ukulele", "chess", "knit", "railway")), notes  # and what they enjoy
    # Told only "at most N words", it made no room, and crept past the budget towards the cut that
    # loses the newest news; told how full they are, it dropped the old news and condensed.
    assert len(notes.split()) <= budget * 0.8, len(notes.split())

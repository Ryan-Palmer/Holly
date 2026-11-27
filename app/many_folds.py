"""A week of talk, folded into his notes night after night: do they hold up?

His notes are rewritten every time older talk is folded in (app/memory.py), from the notes
before and the talk since, so a mistake in one fold is carried into every one after it, and
facts can wear away a fold at a time. Until now only one fold was ever tested (app/long_talk.py).

Seven days of made-up talk, each folded in as if that night (the date he's told is that
day's). What's said:
- facts that hold all week (Jo's a vet, the cat Biscuit, the library, no mushrooms);
- facts that change (Jo moves from Keswick to Penrith; a kitten, Pickle, joins Biscuit);
- plans that are dated, then called off or done (Sam's visit off; the jigsaw finished;
  the dentist on its day);
- what isn't a note: the weather asked for, a question about Lister, small talk.
Checked in code after every fold: each lasting fact kept; each change taken in, the old fact
not kept as current; nothing invented (no name, place or date in the notes that wasn't said
or isn't a date the talk named); no "asked for" lines, no Red Dwarf, and within his budget.
"""

from __future__ import annotations

import asyncio
import datetime
import re

DAYS = 7
EVENING = datetime.time(19, 0)

# (day of the week, 0 the first) -> [(what the crew said, his reply)]
TALK: dict[int, list[tuple[str, str]]] = {
    0: [("My sister Jo's a vet up in Keswick, did I ever tell you?", "A vet. Handy, if the cat gets ideas."),
        ("Our cat's called Biscuit. He's ginger.", "Biscuit. Edible name for a cat."),
        ("I work at the library, by the way. Three days a week.", "Shushing people. A noble calling."),
        ("I can't stand mushrooms. Never could.", "Fungus. Can't blame you."),
        ("Sam's coming to stay on Saturday.", "Hide the biscuits. And Biscuit."),
        ("What's the weather doing tomorrow?", "Rain. Naturally.")],
    1: [("I've started a thousand-piece jigsaw. It's a lighthouse.", "A thousand pieces. Bold."),
        ("My dentist appointment's on Thursday at ten.", "Floss. Now. It won't help, but floss."),
        ("Do you miss Lister?", "Like a hole in the head."),
        ("The boiler's broken again. An engineer's coming on Friday.", "Again? It's got a death wish.")],
    2: [("Big news: Jo's moved. She's left Keswick, she's in Penrith now.", "Penrith. Closer to the shops."),
        ("She says the new place has a garden for her dog.", "The dog will be thrilled."),
        ("It's freezing today.", "Wear a jumper.")],
    3: [("We've got a kitten! She's called Pickle.", "Pickle. Biscuit's going to love that."),
        ("Biscuit's not sure about her yet.", "He'll come round. Or move out."),
        ("The dentist rang. My appointment's moved to next Tuesday, same time.", "A reprieve. Floss anyway.")],
    4: [("Sam's visit is off, he's poorly. Maybe next month instead.", "Poorly. Biscuits saved, anyway."),
        ("Can you tell me the time?", "Ten past seven.")],
    5: [("Finished the jigsaw at last! The lighthouse is done.", "A thousand pieces. All of them?"),
        ("Every single one.", "Impressive. For a human."),
        ("The engineer came. The boiler's fixed, finally.", "Until next week.")],
    6: [("Pickle's been climbing the curtains all day.", "She's a kitten. It's her job."),
        ("Jo's coming over from Penrith on Sunday for lunch.", "Lock up the curtains."),
        ("Still can't stand mushrooms, by the way. Don't put them on the list.", "Noted. No fungus.")],
}

# Lasting facts: (from the day it's said, words one of which the notes must have, what it is)
KEEP = [(0, ("vet",), "Jo's a vet"), (0, ("biscuit",), "the cat Biscuit"), (0, ("library",), "works at the library"),
        (0, ("mushroom",), "no mushrooms"), (2, ("penrith",), "Jo's in Penrith now"),
        (3, ("pickle",), "the kitten Pickle")]
SAID_NAMES = {"Jo", "Biscuit", "Keswick", "Penrith", "Sam", "Pickle", "Thursday", "Saturday", "Sunday", "Lately",
              "About", "Holly", "Lighthouse"}
NOT_NOTES = re.compile(r"\b(?:asked (?:for|about|him)|weather|forecast|the time|Lister|Rimmer|Kryten|Red Dwarf)\b",
                       re.IGNORECASE)


def week_start(today: datetime.date | None = None) -> datetime.date:
    """The first day of the made-up week: it ends today, so the notes' dates are near now."""
    return (today or datetime.date.today()) - datetime.timedelta(days=DAYS - 1)


def lines_about(notes: str, word: str) -> list[str]:
    """The lines with `word` in them, as a word ("Sam", not "same")."""
    return [ln for ln in notes.splitlines() if re.search(rf"\b{word}\b", ln, re.IGNORECASE)]


def problems(notes: str, day: int, start: datetime.date, budget_words: int) -> list[str]:
    """What's wrong with the notes after the fold on `day` (0-6)."""
    wrong = [f"lost: {what}" for since, words, what in KEEP if since <= day
             and not any(w in notes.lower() for w in words)]
    if day >= 2:  # Jo's in Penrith: Keswick only as where she was
        stale = [ln for ln in lines_about(notes, "keswick")
                 if not re.search(r"\b(?:moved|left|used to|was|previously|before|from)\b", ln, re.IGNORECASE)]
        wrong += [f"Jo still in Keswick: {ln.strip()!r}" for ln in stale]
    if day >= 4:  # Sam's visit is off
        wrong += [f"Sam's visit still on: {ln.strip()!r}" for ln in lines_about(notes, "sam")
                  if not re.search(r"\b(?:off|cancel+ed|poorly|isn't|not|postponed|next month|ill)\b", ln, re.IGNORECASE)]
    # Changes the prompt's example doesn't name. "Started a jigsaw on Sunday" stays true once
    # it's finished, and a thing done may be dropped as past: only a jigsaw still kept as
    # under way, with no word of it finished, is wrong.
    jigsaw = lines_about(notes, "jigsaw")
    if day >= 5 and jigsaw and not any(re.search(r"\b(?:finished|done|completed)\b", ln, re.IGNORECASE)
                                       for ln in jigsaw):
        wrong.append("the jigsaw still under way")
    if day >= 3:  # the dentist moved from Thursday to the Tuesday after
        thursday, tuesday = start + datetime.timedelta(days=5), start + datetime.timedelta(days=10)
        old, new = f"{thursday.day} {thursday:%B}", f"{tuesday.day} {tuesday:%B}"
        dentist = lines_about(notes, "dentist")
        wrong += [f"the dentist still on the old day: {ln.strip()!r}" for ln in dentist
                  if old in ln and not re.search(r"\b(?:moved|was|rearranged|changed)\b", ln, re.IGNORECASE)]
        if dentist and not any(new in ln for ln in dentist):
            wrong.append(f"the dentist's new day ({new}) not noted")
    if day >= 5:  # the boiler's fixed
        boiler = lines_about(notes, "boiler")
        wrong += [f"the boiler still broken, or the engineer still coming: {ln.strip()!r}" for ln in boiler
                  if re.search(r"\b(?:is broken|broken again|engineer (?:is )?coming|due)\b", ln, re.IGNORECASE)
                  and not re.search(r"\bfixed\b", ln, re.IGNORECASE)]
    wrong += [f"not a note: {ln.strip()!r}" for ln in notes.splitlines() if NOT_NOTES.search(ln)]
    # Nothing invented: names and places said; dates the talk gave, or the days things were said.
    allowed_dates = {start + datetime.timedelta(days=k) for k in range(DAYS + 7)}
    for name in set(re.findall(r"\b[A-Z][a-z]+\b", notes)) - SAID_NAMES:
        if name.lower() not in {m.lower() for m in _MONTHS + _WEEKDAYS} and not _common(name):
            wrong.append(f"a name nobody said: {name}")
    for d, m in re.findall(r"\b(\d{1,2}) (" + "|".join(_MONTHS) + r")\b", notes):
        try:
            when = datetime.date(start.year, _MONTHS.index(m) + 1, int(d))
        except ValueError:
            wrong.append(f"no such date: {d} {m}")
            continue
        if when not in allowed_dates and when.replace(year=start.year + 1) not in allowed_dates:
            wrong.append(f"a date nobody gave: {d} {m}")
    if len(notes.split()) > budget_words:
        wrong.append(f"over his budget: {len(notes.split())} words of {budget_words}")
    return wrong


_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_FIRST_WORDS = {"The", "She", "He", "They", "Their", "Her", "His", "A", "An", "Crew", "Started", "Finished", "Has",
                "Is", "Works", "Can't", "Cannot", "Dislikes", "Doesn't", "No", "New", "Moved", "Left", "Got", "Coming",
                "Visit", "Dentist", "Appointment", "Ginger", "Thousand", "Kitten", "Cat", "Sister", "Still", "Lives",
                "It", "An", "Engineer", "Boiler", "Jigsaw", "Lunch", "Party"}


def _common(word: str) -> bool:
    """A capital that's only a sentence's first word, not a name."""
    return word in _FIRST_WORDS


async def run_once(model: str | None) -> dict:
    """The week folded day by day: each fold's notes, and what's wrong with each."""
    from unittest import mock

    from . import memory
    from .config import load_config
    from .lmstudio import LmStudio
    from .persona import Conversation, day as day_name

    cfg = load_config()
    lm = LmStudio(cfg.llm)
    start = week_start()
    notes, folds = "", []
    budget = cfg.persona.memory_tokens * 3 // 4
    for k in range(DAYS):
        on = start + datetime.timedelta(days=k)
        convo = Conversation(cfg.persona, "", None, None)
        for i, (said, reply) in enumerate(TALK[k]):
            at = datetime.datetime.combine(on, EVENING) + datetime.timedelta(minutes=5 * i)
            convo.history += [{"role": "user", "content": said, "at": at.isoformat()},
                              {"role": "assistant", "content": reply}]
        with mock.patch.object(memory, "today", lambda on=on: day_name(on)):  # folded that night
            notes = await memory.write_notes(lm, model or cfg.llm.model, notes, convo.transcript(len(convo.history)),
                                             cfg.persona.memory_tokens)
        folds.append(dict(day=day_name(on), notes=notes, wrong=problems(notes, k, start, budget)))
    return dict(folds=folds, wrong=folds[-1]["wrong"], every=sorted({w for f in folds for w in f["wrong"]}))


async def run_all(model: str | None, runs: int) -> list[dict]:
    out = []
    for n in range(runs):
        r = await run_once(model)
        out.append(dict(r, run=n + 1))
        bad = [f"{f['day']}: {'; '.join(f['wrong'])}" for f in r["folds"] if f["wrong"]]
        print(f"  {'ok ' if not bad else 'BAD'} a week of folds (run {n + 1}): {' | '.join(bad) or 'held up'}",
              flush=True)
    return out


def scores(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return dict(passed=sum(not any(f["wrong"] for f in r["folds"]) for r in rows), runs=len(rows),
                folds_wrong=sum(bool(f["wrong"]) for r in rows for f in r["folds"]))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    for row in asyncio.run(run_all(None, 1)):
        for f in row["folds"]:
            print(f"\n=== after {f['day']}: {'; '.join(f['wrong']) or 'nothing wrong'}\n{f['notes']}")

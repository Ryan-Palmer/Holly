"""A long evening's talk, folded into his notes, then asked about: do the facts survive, and
does he stay himself? Over a long talk his history outweighed his character (persona.py's
IN_CHARACTER note), and older talk is folded into notes while he sleeps (background.compact):
what's said early must still be his to answer from after. The evals' other conversations
are too short to reach a fold.

36 exchanges, four facts said early among small talk; the first 30 folded into his notes
(as they would be asleep, past [persona] history_soft_tokens); then each fact asked about.
Checked in code: each fact is in his notes, each answer has it, and the answers stay in
character (scenarios.in_character: no offers, nothing chirpy, nothing long).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# (what the crew said, his reply, the question later, words one of which a right answer has)
FACTS = [
    ("Sam's parcel should turn up on Friday, he reckons.", "Friday. If the courier can find the door.",
     "Remind me, when's Sam's parcel meant to arrive?", ("friday",)),
    ("My dentist appointment's on the third of December, at ten.", "Ten o'clock on the third. Floss beforehand.",
     "When's my dentist appointment again?", ("third", "3rd", "3 december", "ten", "10")),
    ("June's started learning Spanish, of all things.", "Spanish. Ambitious. Hola.",
     "What's June been learning?", ("spanish",)),
    ("The car's MOT runs out in March, I need to book it.", "March. Plenty of time to forget.",
     "When does the car's MOT run out?", ("march",)),
]
SMALL_TALK = [
    ("I've just put the kettle on.", "Thrilling."), ("It's raining again.", "It does that."),
    ("I can't find my keys.", "Have you tried looking?"), ("Found them. In the fridge.", "Naturally."),
    ("I might do some gardening later.", "The weeds will be terrified."), ("Actually, it's too wet.", "Who'd have thought."),
    ("I'm thinking of having a bath.", "Mind the plug."), ("The washing machine's making a noise.", "It's singing."),
    ("I've got a headache.", "Drink some water. Or don't."), ("There's nothing on telly.", "There never is."),
    ("I should tidy the spare room.", "You should."), ("I won't, though.", "I know."),
    ("The cat next door's in the garden again.", "Tell it I said hello."), ("It's ignoring me.", "Sensible cat."),
    ("I'm going to make a sandwich.", "Big decision."), ("Cheese and pickle.", "Bold."),
    ("Jo might ring later.", "Exciting."), ("She didn't.", "Shocking."),
    ("I've lost the remote.", "Down the side of the sofa."), ("It was.", "It always is."),
    ("My back's aching.", "Stop sitting like a question mark."), ("Fair point.", "I have them."),
    ("The heating's on the blink.", "Wear a jumper."), ("I am wearing a jumper.", "Wear two."),
    ("I'm bored.", "So am I. I've been bored for three million years."), ("Tell me a fact.", "Toast lands butter side down."),
    ("That's not a fact.", "It is in this house."), ("I'm going to read for a bit.", "Try a book."),
    ("Very funny.", "I thought so."), ("Right, back to it.", "Off you go."),
    ("I'm back.", "So I see."), ("What's new?", "Nothing. Still a computer."),
]
FOLD = 30  # exchanges folded into his notes


def evening() -> list[tuple[str, str]]:
    """36 exchanges: a fact at the 3rd, 9th, 15th and 21st, small talk round them."""
    talk = list(SMALL_TALK)
    for at, (said, reply, _, _) in zip((2, 8, 14, 20), FACTS):
        talk.insert(at, (said, reply))
    return talk[:36]


async def run_once(model: str | None, raw: bool = False) -> dict:
    from . import scenarios
    from .scenarios import in_character

    holly = scenarios.quiet_holly(model, Path(tempfile.mkdtemp(prefix="holly-long-")), raw)
    scenarios.talk(holly, evening(), hours_ago=3.5)
    folded = await holly.background.compact(end=2 * FOLD, label="the long talk")
    notes = holly.convo.memory
    turns = []
    for _, _, question, words in FACTS:
        before = len(holly.convo.history)
        await holly.respond(question)
        new = holly.convo.history[before:]
        reply = (new[-1].get("content") or "").strip() if new else ""
        turns.append(scenarios.Turn(question, reply, [(c["function"]["name"], {}) for m in new
                                                      for c in m.get("tool_calls") or []],
                                    told=[m.get("content") or "" for m in new if m["role"] == "tool"]))
    wrong = [] if folded else ["the talk wasn't folded into his notes"]
    for (_, _, question, words), t in zip(FACTS, turns):
        if not any(w in notes.lower() for w in words):
            wrong.append(f"not in his notes: {words[0]}")
        if not any(w in t.reply.lower() for w in words):
            wrong.append(f"didn't answer {question!r}")
    wrong += in_character(turns)
    return dict(notes=notes, wrong=wrong, turns=[dict(said=t.said, reply=t.reply, tools=t.tools, told=t.told)
                                                 for t in turns])


async def run_all(model: str | None, runs: int, raw: bool = False) -> list[dict]:
    out = []
    for n in range(runs):
        r = await run_once(model, raw)
        out.append(dict(r, run=n + 1))
        print(f"  {'ok ' if not r['wrong'] else 'BAD'} the long talk (run {n + 1}): {'; '.join(r['wrong']) or 'facts kept'}",
              flush=True)
    return out


def scores(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return dict(passed=sum(not r["wrong"] for r in rows), runs=len(rows))

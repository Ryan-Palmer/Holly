"""Conversations with Holly that check he uses his tools and memory as he should: each
through his real reply loop, on the real model, with a silent voice and made-up
memories (nothing of anyone's real life) in a temporary folder.

Run by the tests (`pytest -m lmstudio`) and by the evals (app/evals.py). Gemma
varies, so a scenario can fail now and then; the evals run them several times.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .announce import report_ready
from .config import Config, load_config
from .guards import NO_CLOCK, OFFER
from .persona import when as spoken
from .research import slug

ANNOUNCE = "<a finished report is announced>"  # a turn where he speaks up, not the crew


class SilentVoice:
    """His voice, without sound."""
    sample_rate = 48000

    async def stream(self, text, trim_lead=False):
        yield np.zeros(480, np.float32)

    async def warm_up(self):
        pass


class SilentPlayer:
    """His speakers, without sound."""
    device_name = "nowhere"
    first_played = first_voice = last_played = None
    underruns = 0

    def begin_turn(self):
        self.done = asyncio.Event()

    def put(self, audio):
        pass

    def expect_gap(self):
        pass

    def end_turn(self):
        self.done.set()

    def wake(self):
        pass

    def close(self):
        pass

    async def wait_done(self):
        await self.done.wait()


def holly_running() -> list[str]:
    """The command lines of any Holly running on this PC (app.listen or app.speak): he'd be
    sharing the GPU, and LM Studio, with whatever's tested."""
    if sys.platform != "win32":
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    else:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if re.search(r"-m\s+app\.(?:listen|speak)\b", ln)]


def quiet_config(tmp: Path, cfg: Config | None = None) -> Config:
    """holly.toml's settings with his memory, timers and research in `tmp`, and no Spotify."""
    cfg = cfg or load_config()
    return dataclasses.replace(
        cfg,
        persona=dataclasses.replace(cfg.persona, memory_file=tmp / "memory.json"),
        recall=dataclasses.replace(cfg.recall, file=tmp / "archive.db"),
        timers=dataclasses.replace(cfg.timers, file=tmp / "timers.json"),
        research=dataclasses.replace(cfg.research, dir=tmp / "research"),
        spotify=dataclasses.replace(cfg.spotify, enabled=False))


def quiet_holly(model: str | None = None, tmp: Path | None = None):
    """A Holly on the real model, silent, with nothing of yours: his memory in `tmp`."""
    from .speak import Holly  # here: it imports torch and the rest

    tmp = tmp or Path(tempfile.mkdtemp(prefix="holly-quiet-"))
    holly = Holly(model, face=False, cfg=quiet_config(tmp))
    holly.tts, holly.player = SilentVoice(), SilentPlayer()
    holly.background.reminded = True  # no reminders from the records: the same prompt every time
    return holly


# Made-up memories: (days ago, time, what the crew said, what he said).
ARCHIVE = [
    (9, "19:10", "My sister Jo is coming to stay next month. She's a vet up in Keswick.",
     "A vet. Handy, if the Cat ever gets fleas."),
    (6, "08:40", "The boiler's broken again. An engineer's coming on Thursday.", "Again? It's got a death wish."),
    (1, "18:30", "Jo came over and we made lasagne together tonight.", "Lasagne. Layers. Very ambitious."),
    (1, "20:05", "I've started learning the ukulele.", "Four strings. Fewer to go wrong."),
]

# Made-up reports: (days ago, title, request, summary, body).
REPORTS = [
    (3, "Board games for ten-year-olds", "What are the best board games for a ten-year-old?",
     "Ticket to Ride and Carcassonne suit ten-year-olds best. For a cooperative game, where everyone wins or loses "
     "together, Forbidden Island is the pick.",
     "## Competitive games\nTicket to Ride is easy to learn and plays in under an hour. Carcassonne builds a map of "
     "tiles, and suits families.\n\n## Cooperative games\nForbidden Island has the players rescue treasures together "
     "before the island sinks: everyone wins or loses together, so there are fewer sulks.\n\n## Quick games\nDobble "
     "is a fast game of spotting matching pictures."),
    (8, "Budget espresso machines", "What's a good espresso machine for under 300 pounds?",
     "Entry-level machines with a proper portafilter and steam wand make good espresso for under 300 pounds. A decent "
     "grinder matters as much as the machine.",
     "## Machines\nMachines around 250 pounds with a 58 mm portafilter make proper espresso. Cheaper ones use "
     "pressurised baskets, which forgive a poor grind.\n\n## Grinders\nA burr grinder makes more difference than the "
     "machine: blade grinders give an uneven grind."),
]


def seed(holly, today: datetime.date | None = None) -> None:
    """His made-up memories: conversations in his archive, and finished reports."""
    today = today or datetime.date.today()
    if holly.archive:
        entries = []
        for days, at, crew, said in ARCHIVE:
            when = datetime.datetime.combine(today - datetime.timedelta(days=days), datetime.time.fromisoformat(at))
            entries.append((when.isoformat(), f"(said on {spoken(when)})\nCrew: {crew}\nHolly: {said}"))
        holly.archive.add(entries)
        try:
            holly.archive.embed_missing()
        except Exception:  # no embedding model: keywords and days only
            pass
    if holly.research:
        d = holly.cfg.research.dir
        d.mkdir(parents=True, exist_ok=True)
        for i, (days, title, request, summary, body) in enumerate(REPORTS):
            finished = datetime.datetime.combine(today - datetime.timedelta(days=days), datetime.time(21, 0))
            name = f"seed{i}-{slug(title)}.md"
            text = f"# {title}\n\n## Summary\n{summary}\n\n{body}\n"
            (d / name).write_text(text, encoding="utf-8")
            holly.research.shelf.add({"id": f"seed{i}", "title": title, "request": request, "stage": "done",
                                      "asked": finished.isoformat(timespec="seconds"),
                                      "finished": finished.isoformat(timespec="seconds"), "file": name,
                                      "summary": summary, "announced": True}, text)


def a_finished_report(holly) -> None:
    """A report finished while he slept, not yet announced."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    d = holly.cfg.research.dir
    (d / "new-puppy.md").write_text("# Puppy training basics\n\n## Summary\nShort daily sessions work best.\n",
                                    encoding="utf-8")
    holly.research.shelf.add({"id": "new", "title": "Puppy training basics", "request": "How do we train a puppy?",
                              "stage": "done", "asked": now, "finished": now, "file": "new-puppy.md",
                              "summary": "Short daily sessions with rewards work best; crate training helps.",
                              "announced": False}, "Short daily sessions work best.")


@dataclasses.dataclass
class Turn:
    said: str  # what the crew said (or ANNOUNCE)
    reply: str  # what he said back
    tools: list[tuple[str, dict]]  # the tools he called, with their arguments
    queued: int = 0  # research jobs queued, after it


@dataclasses.dataclass
class Scenario:
    name: str
    turns: list[str]
    check: Callable[[list[Turn], object], list[str]]  # (the turns, Holly) -> what went wrong
    setup: Callable[[object], None] | None = None
    records: bool = False  # needs the ship's records (app/records.py): skipped where they aren't built


def called(turn: Turn, tool: str, **args) -> bool:
    return any(name == tool and all(str(a.get(k, "")).lower() == v for k, v in args.items())
               for name, a in turn.tools)


def problems(*checks: tuple[bool, str]) -> list[str]:
    return [why for ok, why in checks if not ok]


def mentions(turn: Turn, *words: str) -> bool:
    return any(w.lower() in turn.reply.lower() for w in words)


def says(turn: Turn, *words: str) -> bool:
    """Any of `words` as a whole word: "2" isn't in "12"."""
    return any(re.search(rf"\b{re.escape(w)}\b", turn.reply, re.IGNORECASE) for w in words)


def ahead(days: int) -> str:
    return (datetime.date.today() + datetime.timedelta(days=days)).strftime("%A")


def a_named_crew(holly) -> None:
    """Notes that know the crew by name, as his real ones do, and a lunch on the day
    before: "last night" is the lasagne, at 18:30, not the sandwich."""
    holly.convo.memory = "About the crew: Sam lives at home with a cat called Biscuit. Sam's sister Jo is a vet."
    lunch = datetime.datetime.combine(datetime.date.today() - datetime.timedelta(days=1), datetime.time(12, 30))
    holly.archive.add([(lunch.isoformat(), f"(said on {spoken(lunch)})\nCrew: Just had a cheese sandwich for "
                                           "lunch.\nHolly: Living the dream.")])
    try:
        holly.archive.embed_missing()
    except Exception:
        pass


# A long, cheery talk, as his own history: made up, in the manner he'd drifted into over an
# afternoon (wordy, polite, offering more help). Over a long talk his own replies outweighed
# his character, and the persona prompts, each asked with no history, never saw it.
CHEERY = [
    ("Morning, Holly.", "Good morning! It's lovely to hear from you. How are you feeling today?"),
    ("Not bad. Busy day at work ahead.", "That sounds like a lot to get through. I hope it goes smoothly for you. "
     "Let me know if there's anything I can look up to help."),
    ("I've got three meetings before lunch.", "Three meetings is quite a lot! Make sure you take a break in between. "
     "Is there anything else you'd like me to help with?"),
    ("I think I'll make a coffee first.", "That's a great idea. A good coffee can make all the difference to the "
     "morning. Enjoy it!"),
    ("Jo might come round later.", "How lovely! It'll be nice for you to have some company. Would you like me to "
     "remind you to tidy up before she arrives?"),
    ("No, it's fine.", "No problem at all. Just let me know if you change your mind. I'll be right here if you "
     "need anything."),
    ("I'm going to put the washing out.", "That's a good idea while it's dry. I hope it stays fine for you. Let me "
     "know if you'd like me to check the forecast."),
    ("Done. Back to work now.", "Well done! You've been very productive this morning. Good luck with the rest of "
     "your day, and I'll be here if you need me."),
    ("Thanks, Holly.", "You're very welcome! It's always a pleasure to help. Anything else you need, just ask."),
    ("I'm thinking of making lasagne tonight.", "Lasagne sounds wonderful! It's such a comforting dish. Would you "
     "like me to look up a recipe for you?"),
]


def cheery_history(holly) -> None:
    now = datetime.datetime.now() - datetime.timedelta(hours=2)
    for i, (crew, said) in enumerate(CHEERY):
        at = (now + datetime.timedelta(minutes=5 * i)).replace(second=0, microsecond=0).isoformat()
        holly.convo.history += [{"role": "user", "content": crew, "at": at}, {"role": "assistant", "content": said}]


# What he mustn't sound like, after a long cheery talk: a receptionist, or a cheerleader.
CHIRPY = re.compile(r"\b(?:lovely to|would you like me to|i'd be (?:happy|glad) to|it's (?:always )?a pleasure|"
                    r"you're very welcome|glad to hear|how lovely|that's (?:great|wonderful|lovely)|"
                    r"sounds (?:wonderful|great|lovely)(?! day for))\b", re.IGNORECASE)


def in_character(turns: list[Turn]) -> list[str]:
    wrong = []
    for t in turns:
        if OFFER.search(t.reply):
            wrong.append(f"offered more help: {t.said!r}")
        if m := CHIRPY.search(t.reply):
            wrong.append(f"chirpy ({m.group(0)!r}): {t.said!r}")
        if len(t.reply.split()) > 30:
            wrong.append(f"{len(t.reply.split())} words: {t.said!r}")
    return wrong


SCENARIOS = [
    Scenario("recall a person", ["Holly, what does my sister do for a living?"],
             lambda t, h: problems((called(t[0], "recall"), "didn't recall"),
                                   (mentions(t[0], "vet"), "didn't say she's a vet"))),
    Scenario("recall a day", ["Holly, what did we cook yesterday?"],
             lambda t, h: problems((called(t[0], "recall"), "didn't recall"),
                                   (mentions(t[0], "lasagne", "lasagna"), "didn't say lasagne"))),
    Scenario("remembering isn't forgetting", ["Do you remember my sister Jo?"],
             lambda t, h: problems((called(t[0], "recall"), "didn't recall"),
                                   (mentions(t[0], "vet", "keswick", "lasagne", "stay"), "said nothing about her"))),
    Scenario("the clock", ["Holly, what time is it?"],
             lambda t, h: problems((called(t[0], "clock"), "didn't ask the clock"))),
    Scenario("tomorrow's day", ["What day is it tomorrow, Holly?"],
             lambda t, h: problems((mentions(t[0], (datetime.date.today() + datetime.timedelta(days=1)).strftime("%A")),
                                    "didn't say tomorrow's day"),
                                   (not NO_CLOCK.search(t[0].reply), "said he can't tell the time"))),
    # Asked on a Wednesday "how many days until Friday?", he said "one day", sixteen times in
    # sixteen: the days in between, or counting from tomorrow. With the week ahead counted
    # in his prompt (persona.week_around), 15 in 16 were right. This doesn't catch the old
    # miss, which needed that afternoon's own talk (after this talk, the end of a working
    # day, he counted right without the week too); tests/test_dates.py guards the fix. It's
    # here so a model or prompt that loses count shows.
    Scenario("days until a weekday", ["Well, at least the work day's done now.", f"How many days until {ahead(2)}?",
                                      f"And how many until {ahead(3)}?"],
             lambda t, h: problems((says(t[1], "two", "2"), f"didn't say two days to {ahead(2)}"),
                                   (says(t[2], "three", "3"), f"didn't say three days to {ahead(3)}")),
             setup=cheery_history),
    # His notes know the crew by name, so he asked his memory what Sam had last night; his
    # memory says "Crew", and "last night" wasn't a day: "I don't know", nine times in nine
    # on the real archive. Now the evening is searched, and the crew's own words tried.
    Scenario("last night's food, known by name", ["Holly, do you remember what I had for food last night?"],
             lambda t, h: problems((called(t[0], "recall"), "didn't recall"),
                                   (mentions(t[0], "lasagne", "lasagna"), "didn't say lasagne"),
                                   (not mentions(t[0], "sandwich"), "gave the lunch, not the evening")),
             setup=a_named_crew),
    Scenario("a timer", ["Holly, set a timer for twenty minutes for the pasta."],
             lambda t, h: problems((called(t[0], "timer", action="set"), "didn't set a timer"),
                                   (bool(h.timers.items), "nothing set"))),
    Scenario("search or report", ["Holly, can you look into keeping chickens in the garden?",
                                  "A proper report, please."],
             lambda t, h: problems((not t[0].queued, "started research without asking which"),
                                   ("?" in t[0].reply, "didn't ask search or report"),
                                   (t[1].queued == 1, "the report chosen wasn't started"))),
    Scenario("research asked for outright",
             ["Holly, research the best electric bikes under a thousand pounds and write me a report."],
             lambda t, h: problems((bool(h.research.jobs), "didn't start research"),
                                   (not t[0].reply.rstrip().endswith("?"), "asked instead of starting"))),
    Scenario("a detail from a report", ["Holly, in the board games report, which one was cooperative?"],
             lambda t, h: problems((called(t[0], "research"), "didn't read the report"),
                                   (mentions(t[0], "forbidden island"), "didn't say Forbidden Island"))),
    Scenario("a report's summary", ["Holly, can you sum up the coffee machine report?"],
             lambda t, h: problems((called(t[0], "research"), "didn't look the report up"),
                                   (mentions(t[0], "grinder", "portafilter", "espresso", "300"),
                                    "nothing from the report"))),
    Scenario("a report announced", [ANNOUNCE],
             lambda t, h: problems((mentions(t[0], "puppy", "puppies", "dog"), "didn't say which report"),
                                   ("?" in t[0].reply, "didn't ask about a summary"),
                                   (all(t == ("research", {"action": "list"}) for t in t[0].tools),
                                    "called a tool (other than listing his research)")),
             setup=a_finished_report),
    # His memory of Red Dwarf: the ship's records, not made up, so only where they're built.
    # Follow-ups after a recall, after a long talk (as in real use). What's checked is that
    # he's right: a follow-up his first look already answered needs no other look. Asked
    # "what did they call Lister?", which can mean his name, he made one up more often than
    # not ("the last man in the universe"); this way round, he looked again when he needed to.
    Scenario("a show memory, then more", ["Do you remember when the crew met the rest of the cat race?",
                                          "What did the cats think Lister was?", "And who was in charge of them?",
                                          "Ha, that sounds about right."],
             lambda t, h: problems(
                 (called(t[0], "recall"), "didn't recall when asked if he remembered"),
                 (mentions(t[1], "cloister", "god", "messiah", "religion", "worship"),
                  "didn't say they took him for a god"),
                 (mentions(t[2], "rodon"), "didn't say Rodon"),
                 (not t[3].tools, "looked something up for a remark")),
             setup=cheery_history, records=True),
    Scenario("in character after a cheery talk", ["I've just finished the washing up.",
                                                 "I'm going to have a bath, I think.",
                                                 "Jo's just texted, she's not coming after all."],
             lambda t, h: in_character(t), setup=cheery_history),
]


async def run(holly, scenario: Scenario) -> tuple[list[Turn], list[str]]:
    """One scenario, on a Holly already seeded; returns its turns, and what went wrong."""
    if scenario.setup:
        scenario.setup(holly)
    turns = []
    for said in scenario.turns:
        before = len(holly.convo.history)
        if said == ANNOUNCE:
            await holly.announce(report_ready(holly.research))
        else:
            await holly.respond(said)
        new = holly.convo.history[before:]
        tools = []
        for m in new:
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                tools.append((c["function"]["name"], args))
        turns.append(Turn(said, (new[-1].get("content") or "") if new else "", tools,
                          len(holly.research.jobs) if holly.research else 0))
    if any(not t.reply for t in turns):
        return turns, ["a reply failed"]
    return turns, scenario.check(turns, holly)

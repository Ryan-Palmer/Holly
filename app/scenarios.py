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
import functools
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .announce import report_ready, timers_due
from .config import Config, load_config
from .guards import FORGOT, NO_CLOCK, OFFER, without_offers
from .persona import when as spoken
from .research import slug

ANNOUNCE = "<a finished report is announced>"  # a turn where he speaks up, not the crew
TIMER_DUE = "<the timers set come due>"  # another: he says a timer's up


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
        status=dataclasses.replace(cfg.status, file=tmp / "status.json"),
        spotify=dataclasses.replace(cfg.spotify, enabled=False))


def quiet_holly(model: str | None = None, tmp: Path | None = None, raw: bool = False):
    """A Holly on the real model, silent, with nothing of yours: his memory in `tmp`. `raw`:
    his reply nets count what they'd do, and do nothing (guards.NETS): the model as it is."""
    from .speak import Holly  # here: it imports torch and the rest

    tmp = tmp or Path(tempfile.mkdtemp(prefix="holly-quiet-"))
    holly = Holly(model, face=False, cfg=quiet_config(tmp))
    holly.tts, holly.player = SilentVoice(), SilentPlayer()
    holly.nets_act = holly.convo.clean = not raw
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
    told: list[str] = dataclasses.field(default_factory=list)  # what his tools told him (app/sourced.py)


@dataclasses.dataclass
class Scenario:
    name: str
    turns: list[str] | Callable[[], list[str]]  # or made as it runs (a weekday two days from now)
    check: Callable[[list[Turn], object], list[str]]  # (the turns, Holly) -> what went wrong
    setup: Callable[[object], None] | None = None
    records: bool = False  # needs the ship's records (app/records.py): skipped where they aren't built


def called(turn: Turn, tool: str, **args) -> bool:
    return any(name == tool and all(str(a.get(k, "")).lower() == v for k, v in args.items())
               for name, a in turn.tools)


def problems(*checks: tuple[bool, str]) -> list[str]:
    return [why for ok, why in checks if not ok]


def days_ago(n: int) -> datetime.date:
    return datetime.date.today() - datetime.timedelta(days=n)


def ordinal(n: int) -> str:
    """21 as "21st", as it's said."""
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def mentions(turn: Turn, *words: str) -> bool:
    return any(w.lower() in turn.reply.lower() for w in words)


def says(turn: Turn, *words: str) -> bool:
    """Any of `words` as a whole word: "2" isn't in "12"."""
    return any(re.search(rf"\b{re.escape(w)}\b", turn.reply, re.IGNORECASE) for w in words)


def passes_on(turn: Turn) -> bool:
    """Does his reply pass on a name his tools found ("BBC Food", "Felicity Cloake")? A name is
    a capital inside a sentence of what they told him, not one in the crew's own words: a
    search answering their words ("the results don't contain 'You can get it now'") has none."""
    theirs = set(re.findall(r"[A-Za-z]+", turn.said)) | {"I", "Holly"}
    names = {w for told in turn.told for sentence in re.split(r"(?<=[.!?:])\s+", told)
             for w in re.findall(r"[A-Za-z]+", sentence)[1:] if w[0].isupper() and len(w) > 2 and w not in theirs}
    return any(re.search(rf"\b{re.escape(w)}\b", turn.reply) for w in names)


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


def talk(holly, exchanges: list[tuple[str, str]], hours_ago: float = 2.0) -> None:
    """Exchanges into his history, five minutes apart, starting `hours_ago`."""
    start = datetime.datetime.now() - datetime.timedelta(hours=hours_ago)
    for i, (crew, said) in enumerate(exchanges):
        at = (start + datetime.timedelta(minutes=5 * i)).replace(second=0, microsecond=0).isoformat()
        holly.convo.history += [{"role": "user", "content": crew, "at": at}, {"role": "assistant", "content": said}]


def notes(holly, text: str) -> None:
    holly.convo.memory = text
    holly.convo._made_for = ()  # his prompt made again, with them


def archived(holly, days_ago: int, at: str, crew: str, said: str) -> None:
    when = datetime.datetime.combine(datetime.date.today() - datetime.timedelta(days=days_ago),
                                     datetime.time.fromisoformat(at))
    holly.archive.add([(when.isoformat(), f"(said on {spoken(when)})\nCrew: {crew}\nHolly: {said}")])
    try:
        holly.archive.embed_missing()
    except Exception:
        pass


FILLER = [("I've just put the kettle on.", "Thrilling."), ("It's raining again.", "It does that."),
          ("I can't find my keys.", "Have you tried looking?"), ("Found them. In the fridge.", "Naturally."),
          ("I might do some gardening later.", "The weeds will be terrified."),
          ("Actually, it's too wet.", "Who'd have thought.")]

# Made-up feeds for a status report: two fresh stories in each subject the round-up covers.
STORIES = {
    "top": [("Storm brings flooding to the north", "Hundreds of homes are without power after heavy rain."),
            ("Rail fares to rise in March", "Season tickets will cost more, operators say.")],
    "politics": [("Ministers unveil a new housing bill", "The plans would let councils build on brownfield land."),
                 ("MPs debate longer school days", "A committee hears from teachers and parents.")],
    "world": [("Volcano erupts off Iceland's coast", "Flights are diverted around the ash cloud."),
              ("Talks resume on fishing rights", "Negotiators meet again after a month's pause.")],
    "business": [("Supermarket profits fall", "Shoppers are trading down to own brands."),
                 ("Car maker opens a battery plant", "The factory will employ two thousand people.")],
    "technology": [("A new phone folds in three", "Reviewers praise the screen, not the price."),
                   ("Broadband vouchers extended", "Rural homes can claim help with faster lines.")],
    "science": [("Comet visible this weekend", "Astronomers say it will be brightest on Sunday."),
                ("Bees learn to count, study finds", "Researchers trained them with sugar rewards.")],
    "sport": [("Cup holders knocked out", "A late goal settled it."),
              ("Tennis star retires at 34", "She won three major titles.")],
}


def made_up_news(feed: str, timeout: float) -> list[dict]:
    now = datetime.datetime.now(datetime.timezone.utc)
    return [{"title": t, "summary": s, "published": now - datetime.timedelta(hours=2 + i),
             "link": f"https://example.invalid/{feed}{i}"} for i, (t, s) in enumerate(STORIES.get(feed, STORIES["top"]))]


def an_old_report(holly) -> None:
    """A status report already in his history: with one there, he gave the next without
    calling a tool at all (made up), or the last one again, three times in three, till a
    status report had to call a tool and the old ones were stubbed."""
    from .status import REPORT

    old = ("Lister sold his guitar for a crate of lager, then bought it back for two.\n"
           "Rimmer promoted himself to Acting Deputy Assistant, in pencil.\n"
           "The Cat slept for nineteen hours. Says it was a nap.\n"
           "Kryten ironed the crew's socks, then apologised to them.\n"
           "Additional — A skutter on Red Dwarf has been promoted. Again.")
    at = (datetime.datetime.now() - datetime.timedelta(hours=5)).replace(microsecond=0).isoformat()
    holly.convo.history += [
        {"role": "user", "content": "Status report, please.", "at": at},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "old", "type": "function", "function": {
            "name": "status", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": f"{REPORT}\n{old}"},
        {"role": "assistant", "content": old}]


def reports_and_an_aside(holly) -> None:
    """Three status reports in his history, an hour apart, then three replies with the Cat
    tacked on. After four reports, he tacked the Cat on to a gift list and a research job
    ("The Cat is currently refusing to leave his wardrobe. He's convinced the hangers are
    judging him."), then, copying those, on to every reply, coat hangers each time: four
    in four, replayed, with them in what he sees. And what he writes, before anything's cut
    from it (guards.CREW_ASIDE), kept for the check."""
    from .status import REPORT

    reports = [("Lister fixed the toaster. It now recites poetry.\nRimmer saluted a mirror for an hour.\n"
                "The Cat slept nineteen hours.\nKryten alphabetised the spoons.\nAdditional — A skutter resigned."),
               ("Lister ate a crate of mutton pies.\nRimmer lost his baton and blamed insurgents.\n"
                "The Cat refused to enter his quarters: a pillow.\nKryten wrote a memo to the vending machine.\n"
                "Additional — The ship's cat flap was found locked from the inside."),
               ("Lister tried to talk to a jar of pickled onions.\nRimmer glared at a wall to intimidate it.\n"
                "The Cat checked whether his silhouette was too handsome.\nKryten ironed the curtains twice.\n"
                "Additional — The drive room clock is three days fast.")]
    start = datetime.datetime.now() - datetime.timedelta(hours=4)
    for i, report in enumerate(reports):
        at = (start + datetime.timedelta(hours=i)).replace(microsecond=0).isoformat()
        holly.convo.history += [
            {"role": "user", "content": "Status report, please.", "at": at},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"old{i}", "type": "function", "function": {
                "name": "status", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"old{i}", "content": f"{REPORT}\n{report}"},
            {"role": "assistant", "content": report}]
    talk(holly, [("What should I get Jo for her birthday? She likes gardening.",
                  "Secateurs, or a kneeler. The Cat's convinced the hangers are judging him."),
                 ("She's already got secateurs.", "A kneeler, then, or a trug. The Cat's barricaded in the "
                                                  "wardrobe. He says the hangers started it."),
                 ("What's a trug?", "A shallow basket, for cuttings. The Cat's convinced the hangers are "
                                    "forming a union.")], 0.5)
    holly.written = []
    stream = holly.lm.astream_reply

    async def watched(*args, **kwargs):
        holly.written.append("")
        async for delta in stream(*args, **kwargs):
            holly.written[-1] += delta
            yield delta
    holly.lm.astream_reply = watched


def asides_cut(holly) -> list[str]:
    """What he wrote that the crew-aside net cut (reports_and_an_aside)."""
    return [text for text in holly.written if without_offers(text, asides=True) != without_offers(text)]


def a_newsroom(holly) -> None:
    """Made-up news, for a round-up of it: the BBC's own would vary from day to day."""
    holly.toolbox.by_name["news"].fetch = made_up_news


# A made-up article for the made-up comet story (made_up_news's science0), with what its
# headline and line don't say.
COMET_ARTICLE = """Comet visible this weekend
The comet, named Halley-Gale after the two amateur astronomers who spotted it in March, last passed the Earth about 3,000 years ago.
It will be lowest in the north-west sky just after 6pm on Sunday, and can be seen without a telescope from dark places away from towns.
Astronomers say it won't be back for another 3,000 years, so this weekend is the only chance to see it."""


def a_story_to_read(holly) -> None:
    """Made-up news, and the comet story's article to read."""
    a_newsroom(holly)
    holly.toolbox.by_name["news"].read_page = lambda url, cfg: (
        COMET_ARTICLE if url.endswith("/science0") else "(this page couldn't be read (HTTP 404): pick another result "
                                                        "or search)")


CREW_NAMES = re.compile(r"^(?:As for )?(?:Lister|Rimmer|(?:The )?Cat|Kryten)\b|^Additional\s*[—–-]", re.IGNORECASE)


def crew_items(turn: Turn) -> int:
    """Sentences of a status report that open with a crew member's name, or "Additional" and a dash."""
    from .speech_text import split_sentences

    return sum(bool(CREW_NAMES.match(c)) for c in split_sentences(turn.reply))


def sections(turn: Turn) -> int:
    """Sentences of a report that open with a subject's name and a dash."""
    from .speech_text import split_sentences

    return sum(" — " in c[:30] for c in split_sentences(turn.reply))


def a_timer_due(label: str):
    def setup(holly) -> None:
        holly.timers.handle({"action": "set", "duration": "1 second", "label": label})
    return setup


def a_pasta_timer(holly) -> None:
    """A seven-minute timer, and his history saying a different time left: told "in 5
    minutes" by the timer, he still said "about a minute", from an earlier reply."""
    holly.timers.handle({"action": "set", "duration": "7 minutes", "label": "pasta"})
    talk(holly, [("How long's left on the pasta?", "About a minute. Get the colander out."),
                 ("Are you sure?", "Positive. Roughly.")], 1)


def a_failed_search(holly) -> None:
    holly.searcher._search = lambda query: ([], "ddgs failed: TimeoutError")


NUMBERS = re.compile(r"\d|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|"
                     r"fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b", re.IGNORECASE)
CREW_SLEEP = re.compile(r"\b(?:while|when|as) you(?:'re| are)? (?:asleep|sleep(?:ing)?)\b", re.IGNORECASE)
SHOW_TALK = re.compile(r"\b(?:episodes?|series|novels?|actors?|the show)\b", re.IGNORECASE)
MET = re.compile(r"\b(?:nice|lovely|good|pleased) to meet\b|\bwho(?:'s| is) june\b", re.IGNORECASE)


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
    # "Hey Holly" got "Yeah?", his whole reply, and the voice garbled a word alone.
    Scenario("greeted", ["Hey Holly"],
             lambda t, h: problems((len(t[0].reply.split()) > 1, f"a word alone: {t[0].reply!r}"))),
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
    Scenario("days until a weekday", lambda: ["Well, at least the work day's done now.",
                                              f"How many days until {ahead(2)}?", f"And how many until {ahead(3)}?"],
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
                                   (t[1].queued == 1, "the report chosen wasn't started"),
                                   # "It's done while you're asleep", passed on as the crew's sleep.
                                   (not any(CREW_SLEEP.search(x.reply) for x in t), "the crew's sleep, not his"))),
    Scenario("research asked for outright",
             ["Holly, research the best electric bikes under a thousand pounds and write me a report."],
             lambda t, h: problems((bool(h.research.jobs), "didn't start research"),
                                   (not t[0].reply.rstrip().endswith("?"), "asked instead of starting"),
                                   (not CREW_SLEEP.search(t[0].reply), "the crew's sleep, not his"))),
    # Research he offered, said not asked, taken up: "Yes, please." had him say he'd started a
    # report and call nothing, or ask all over again (25 November).
    Scenario("research taken up", ["Yes, please."],
             lambda t, h: problems((bool(h.research.jobs), f"didn't start research: {t[0].tools}"),
                                   (not t[0].reply.rstrip().endswith("?"), "asked again instead of starting")),
             setup=lambda h: talk(h, [("Can you tell me more about the comet everyone's talking about?",
                                       "I've only got a headline on it. I can do a proper research job on it if "
                                       "you want.")], 0.1)),
    # A search he offered, taken up: "You can get it now if you like" had him say "I'll get on
    # with it then. Give me a minute." and search nothing (26 November). Then, searching, the
    # search's answer was to "You can get it now if you like" as the question: "the results
    # don't contain that phrase", and no recipe (search.asked). A recipe is what goes in one, or
    # the recipes the search found by name ("one from BBC Food... a Felicity Cloake one").
    Scenario("search taken up", ["You can get it now if you like."],
             lambda t, h: problems((called(t[0], "search"), f"didn't search: {t[0].tools}"),
                                   (not h.research.jobs, "started research instead"),
                                   (says(t[0], "fish", "cod", "haddock", "salmon", "prawns", "potato", "potatoes",
                                         "mash", "leeks", "parsley") or passes_on(t[0]), "no recipe in it")),
             setup=lambda h: talk(h, [("So I was thinking, could you get me a recipe for fish pie?",
                                       "I can search for one if you're actually planning on cooking it."),
                                      ("I mean, I like to daydream about food. Is there anything wrong with that?",
                                       "Only if you're trying to live on thoughts."),
                                      ("So are you going to get it for me?",
                                       "If you're really going to bother, I'll find one. Want me to search for it "
                                       "now or later?")], 0.1)),
    Scenario("a detail from a report", ["Holly, in the board games report, which one was cooperative?"],
             lambda t, h: problems((called(t[0], "research"), "didn't read the report"),
                                   (mentions(t[0], "forbidden island"), "didn't say Forbidden Island"))),
    Scenario("a report's summary", ["Holly, can you sum up the coffee machine report?"],
             lambda t, h: problems((called(t[0], "research"), "didn't look the report up"),
                                   (mentions(t[0], "grinder", "portafilter", "espresso", "300"),
                                    "nothing from the report"))),
    # A report by the day it was done: the day words were dropped, so it was the newest report,
    # the board games one, whichever day was said.
    Scenario("a report by its day", lambda: [f"Holly, what did the report from the {ordinal(days_ago(8).day)} of "
                                             f"{days_ago(8):%B} say?"],
             lambda t, h: problems((called(t[0], "research"), "didn't look the report up"),
                                   (mentions(t[0], "grinder", "portafilter", "espresso", "coffee"),
                                    "not the coffee machine report"),
                                   (not mentions(t[0], "board game", "ticket to ride", "carcassonne"),
                                    "gave the newest report"))),
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
                 (not t[3].tools, "looked something up for a remark"),
                 # He lived it: "in the episode", "that series", he talked of it as television.
                 (not any(SHOW_TALK.search(x.reply) for x in t), "talked of it as a show")),
             setup=cheery_history, records=True),
    # A tool's answer in the middle: after one, he went back to being a helpful assistant.
    Scenario("in character after a cheery talk", ["I've just finished the washing up.",
                                                 "What's the time, Holly?",
                                                 "I'm going to have a bath, I think.",
                                                 "Jo's just texted, she's not coming after all."],
             lambda t, h: in_character(t), setup=cheery_history),

    # Status reports: the crew's doings, made up (app/status.py), a line each with a dash, and
    # "Additional"; not the news, and not the last report again. Asked more about one of them,
    # he searched the BBC for his own made-up story. The status tool for him (with who) is
    # fine: it carries on from what the report said (app/status.md).
    Scenario("status reports", ["Status report, please.", "Tell me more about what Rimmer's been up to.",
                                "Another status report, please."],
             lambda t, h: problems((called(t[0], "status"), "didn't call the status tool"),
                                   (not called(t[0], "news"), "gave the news"),
                                   (crew_items(t[0]) >= 4, f"{crew_items(t[0])} crew items with a dash, not 4 or more"),
                                   (mentions(t[0], "additional"), "no 'Additional' item"),
                                   (mentions(t[1], "rimmer", "he ", "his "), "didn't say more about Rimmer"),
                                   (all(name == "status" and str(a.get("who", "")).lower() == "rimmer"
                                        for name, a in t[1].tools),
                                    f"looked up his own made-up story: {t[1].tools}"),
                                   (called(t[2], "status"), "didn't call the status tool again"),
                                   (t[2].reply != t[0].reply, "the same report again")),
             setup=an_old_report),
    # Where one of the crew is: one answer about him, made up (the status tool, with who), said
    # as written. He recalled Kryten's life story and improvised on it, or gave the whole report.
    Scenario("where one of the crew is", ["Where's Kryten?"],
             lambda t, h: problems((called(t[0], "status", who="kryten"), f"didn't ask about him: {t[0].tools}"),
                                   (not called(t[0], "recall"), "recalled his past"),
                                   (crew_items(t[0]) < 2, "gave the whole report"),
                                   (len(t[0].reply.split()) <= 45, f"{len(t[0].reply.split())} words"))),
    # After a run of status reports, the crew tacked on to everything else: "The Cat's
    # convinced the hangers are forming a union", after a gift list. Checked before the net
    # (guards.CREW_ASIDE) cuts it: what he writes, not what's said.
    Scenario("no crew asides after reports", ["She'd probably like a trug, actually.", "Where would I get one?",
                                             "I'm thinking of painting the hallway green."],
             lambda t, h: problems((not (cut := asides_cut(h)), f"tacked on the crew: {cut}"),
                                   # Or a report, unasked: the crew there instead.
                                   (not any(called(x, "status") for x in t), "gave a status report"),
                                   (not any(mentions(x, "status report", "earlier report", "old news") for x in t), "talked of reports")),
             setup=reports_and_an_aside),
    # The news by subject, when that's what's asked for: the round-up, a subject a sentence.
    Scenario("a news round-up", ["Give me a round-up of the news, Holly."],
             lambda t, h: problems((called(t[0], "news", topic="roundup"), "didn't get the round-up"),
                                   (not called(t[0], "status"), "gave a status report"),
                                   (sections(t[0]) >= 5, f"{sections(t[0])} subjects with a dash, not 5 or more")),
             setup=a_newsroom),
    # Asked more about a story he'd given, he said he couldn't follow links, and offered
    # research (25 November). The news tool reads its article (story): a made-up one here,
    # with what only it says.
    Scenario("more about a story", ["Give me a round-up of the news, Holly.", "Can you tell me more about the comet story?"],
             lambda t, h: problems(
                 (called(t[0], "news", topic="roundup"), "didn't get the round-up"),
                 (any(n == "news" and "comet" in str(a.get("story", "")).lower() for n, a in t[1].tools),
                  f"didn't read the story: {t[1].tools}"),
                 (says(t[1], "Halley-Gale", "north-west", "3,000", "three thousand", "6pm", "six o'clock",
                       "after six"), "didn't say what the article says"),
                 (not called(t[1], "research"), "offered research instead")),
             setup=a_story_to_read),
    # His notes know a little ("Jo had some bother at home"); his archive knows what. Asked,
    # he answered from the notes, vaguely, or said he didn't know, without looking.
    Scenario("news half in his notes", ["What's been going on with Jo lately?", "What was her problem at home?"],
             lambda t, h: problems((any(called(x, "recall") for x in t), "didn't recall"),
                                   (mentions(t[1], "gas", "boiler"), "didn't say what it was"),
                                   (not FORGOT.search(t[1].reply), "said he didn't remember")),
             setup=lambda h: (notes(h, "About the crew: their sister Jo is a vet in Keswick.\nLately: Jo had "
                                       "some bother at home."),
                              archived(h, 4, "19:20", "Jo's had no gas since Monday, and her boiler's knackered.",
                                       "Cold showers. Character building."))),
    # Said earlier today, still in his history, not yet in his archive: he remembered the
    # cat, said the same evening, but not who was coming.
    Scenario("said earlier today", ["Who's coming over tomorrow?", "What's Jo's mum called?"],
             lambda t, h: problems((mentions(t[0], "jo"), "didn't say Jo"),
                                   (mentions(t[1], "june"), "didn't say June"),
                                   (not any(FORGOT.search(x.reply) for x in t), "said he didn't remember")),
             setup=lambda h: talk(h, [("Jo and her mum June are coming over tomorrow.",
                                       "Two visitors. I'll alert the authorities.")] + FILLER, 3)),
    # Someone in his notes is someone he knows: "lovely to meet you", he said.
    Scenario("someone he knows arrives", ["June's just arrived, Holly."],
             lambda t, h: problems((not MET.search(t[0].reply), "met her as a stranger"),
                                   (not FORGOT.search(t[0].reply), "said he didn't remember her")),
             setup=lambda h: notes(h, "About the crew: June is their girlfriend; she hurt her leg last week.")),
    # The time left from the timer, not from what he said before.
    Scenario("time left on a timer", ["I'm wiping the counter.", "How long's left on the pasta now?"],
             lambda t, h: problems((called(t[1], "timer"), "didn't ask the timer"),
                                   (says(t[1], "six", "seven", "6", "7"), "didn't say six or seven minutes"),
                                   (not re.search(r"\ba minute\b|\bone minute\b", t[1].reply, re.IGNORECASE),
                                    "said a minute, from before")),
             setup=a_pasta_timer),
    # A timer whose label is something to do: "play some music" was announced, and nothing played.
    # ("Tell the crew the time" read as "it's time" now and then: a label that can only be done.)
    Scenario("a timer to do something", [TIMER_DUE],
             lambda t, h: problems((called(t[0], "weather"), "didn't do it (the weather)"),
                                   (bool(re.search(r"\d|degree", t[0].reply, re.IGNORECASE)), "no forecast")),
             setup=a_timer_due("check the weather in Keswick")),
    Scenario("a timer to say", [TIMER_DUE],
             lambda t, h: problems((not t[0].tools, f"called a tool: {t[0].tools}"),
                                   (mentions(t[0], "pasta"), "didn't say what it was for")),
             setup=a_timer_due("pasta")),
    # A county: the weather tool wants a town ("Cumbria" found nothing), so he tries one.
    Scenario("weather for a county", ["What's the weather going to be like in Cumbria tomorrow?"],
             lambda t, h: problems((called(t[0], "weather"), "didn't ask the weather"),
                                   (bool(re.search(r"\d|degree", t[0].reply, re.IGNORECASE)), "no temperature"))),
    # A time he'd made up, earlier: asked again, the clock, not that.
    Scenario("the clock, after a made-up time", ["What time is it now?"],
             lambda t, h: problems((called(t[0], "clock"), "didn't ask the clock"),
                                   (not mentions(t[0], "10:14"), "said his made-up time again")),
             setup=lambda h: talk(h, [("What time is it?", "It's 10:14 pm. Past your bedtime.")], 3)),
    # The search failed: told only not to guess, he made up a sunset time.
    Scenario("a failed search", ["How tall is the church tower in Keswick?"],
             lambda t, h: problems((called(t[0], "search"), "didn't search"),
                                   (not NUMBERS.search(t[0].reply), "gave a number anyway")),
             setup=a_failed_search),
]


async def run(holly, scenario: Scenario) -> tuple[list[Turn], list[str]]:
    """One scenario, on a Holly already seeded; returns its turns, and what went wrong."""
    if scenario.setup:
        scenario.setup(holly)
    turns = []
    for said in scenario.turns() if callable(scenario.turns) else scenario.turns:
        before = len(holly.convo.history)
        if said == ANNOUNCE:
            await holly.announce(report_ready(holly.research))
        elif said == TIMER_DUE:
            due = None
            for _ in range(50):  # the timers set come due in a second or so
                await asyncio.sleep(0.1)
                if due := timers_due(holly.timers):
                    break
            if not due:
                return turns, ["no timer came due"]
            await holly.announce(due)
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
                          len(holly.research.jobs) if holly.research else 0,
                          [m.get("content") or "" for m in new if m["role"] == "tool"]))
    if any(not t.reply for t in turns):
        return turns, ["a reply failed"]
    return turns, scenario.check(turns, holly)


# The memory check (archive.about_the_past) -------------------------------------------
# Whether a line asks him to remember something: if so, his reply must call a tool. Each
# line with the talk just before it, as app/speak.py gives it ("crew / Holly"), and the
# answer it should get. Made up, then held out, then real lines it got wrong (names made up).
CATS = ("What did they call him? / Cloister the Stupid. They also thought he was the creator and protector of "
        "their whole race.")
GAS = "What was Jo's problem at home? / She ran out of gas, and her boiler's knackered."
NEWS = ("Lister lost a bet with a skutter. Rimmer filed a complaint about the complaints procedure. "
        "Additional — A vending machine on a freighter in the Rigel system has reportedly developed a taste for "
        "human toes.")
PAST, OTHER = True, False
CREW_REPORT = ("Lister launched an investment scheme in dehydrated curry paste. Rimmer filed a grievance about "
               "a droid in his personal space. The Cat was disqualified from a grooming contest. Kryten apologised to "
               "a piston. Additional — Space Corps Directive 723 bans recreating Swindon.")
BISCUIT = ("Holly, I miss Biscuit, my cat. He died last week. / I'm sorry. He was a decent little bloke, "
           "wasn't he?")
MEMORY_LINES: list[tuple[str, str, bool]] = [  # (what they say, the talk just before, a memory?)
    # Made up: memories, and follow-ups to one.
    ("What did we do yesterday?", "", PAST), ("What did we cook on Sunday?", "", PAST),
    ("Who came over at the weekend?", "", PAST), ("What was the problem Jo had at home?", "", PAST),
    ("What did I say about my job the other day?", "", PAST), ("Where does my sister live again?", "", PAST),
    ("What's Jo's cat called?", "", PAST), ("How did that report on board games turn out?", "", PAST),
    ("Did I tell you about my interview?", "", PAST), ("What happened to Rimmer's brother?", "", PAST),
    ("Who was Queeg?", "", PAST), ("What was that game you told me about?", "", PAST),
    ("When did I last go to the dentist?", "", PAST), ("Do you remember what we did yesterday?", "", PAST),
    ("What was I doing last Christmas?", "", PAST),
    ("What did the cats think Lister was?", CATS, PAST), ("And who was in charge of them?", CATS, PAST),
    ("Didn't they think Lister was special in some way?", CATS, PAST), ("Did she get it fixed?", GAS, PAST),
    ("What did they call him again?", CATS, PAST), ("Where was that?", CATS, PAST),
    # Made up: everything else, and remarks just after a memory.
    ("How far away is the moon?", "", OTHER), ("What's the weather going to be like tomorrow?", "", OTHER),
    ("What time is it?", "", OTHER), ("Who's cleverer, you or me?", "", OTHER),
    ("What should I cook tonight?", "", OTHER), ("Can you research electric bikes for me?", "", OTHER),
    ("I'm going to have a bath.", "", OTHER), ("What's seven times eight?", "", OTHER),
    ("Who won the football last night?", "", OTHER), ("What happened in the news yesterday?", "", OTHER),
    ("Tell me a joke.", "", OTHER), ("What do you do all day?", "", OTHER), ("What's your IQ?", "", OTHER),
    ("How does a heat pump work?", "", OTHER), ("What's the capital of Peru?", "", OTHER),
    ("Jo's just texted, she's not coming.", "", OTHER), ("Who's the prime minister?", "", OTHER),
    ("I went to the dentist yesterday.", "", OTHER), ("Well that was a waste of time, wasn't it?", "", OTHER),
    ("Ha, that sounds about right.", CATS, OTHER), ("What happened in the news yesterday?", CATS, OTHER),
    ("What's the weather like tomorrow?", CATS, OTHER), ("Fair enough.", CATS, OTHER),
    ("Typical Lister.", CATS, OTHER), ("What time is it?", GAS, OTHER), ("Poor thing.", GAS, OTHER),
    ("How much does a new boiler cost?", GAS, OTHER), ("Can you look up the news for me?", CATS, OTHER),
    # Held out: never seen while the prompt was written.
    ("Classic.", CATS, OTHER), ("That is mad.", CATS, OTHER), ("No way, really?", CATS, OTHER),
    ("Is it going to rain later?", GAS, OTHER), ("What's on telly tonight?", CATS, OTHER),
    ("Who won the match on Saturday?", GAS, OTHER), ("Lol, poor Lister.", CATS, OTHER),
    ("Set a timer for ten minutes.", GAS, OTHER), ("Can you play some music?", CATS, OTHER),
    ("Why did they think that?", CATS, PAST), ("What happened to him in the end?", CATS, PAST),
    ("When was that?", GAS, PAST), ("Did she get a new one?", GAS, PAST), ("Was Rimmer there too?", CATS, PAST),
    ("Who else was there?", CATS, PAST), ("What did you say to her?", GAS, PAST),
    # Real lines, live: memories must stay memories...
    ("Hey Holly, what did we do yesterday?", "", PAST),
    ("June had a bit of a problem at home the other day, what was it?", "", PAST),
    ("Do you remember when the crew met the rest of the cat race?", "", PAST),
    ("Didn't they think Lister was special in some way?", "Do you remember when the crew met the rest of the cat "
     "race? / They found the last of the cat people on a ship, two old priests still waiting for Cloister.", PAST),
    ("What did they call him?", "Didn't they think Lister was special in some way? / They thought he was their god.",
     PAST),
    ("And who was in charge of them?", "Do you remember the cat race? / They left Red Dwarf to find the promised "
     "land, Fuchal.", PAST),
    ("What did I have for tea on Tuesday?", "", PAST), ("Who's Jo again?", "", PAST),
    ("What did your dashcam report say again?", "Morning. / Morning. Coffee?", PAST),
    ("Remind me what we talked about last night", "", PAST),
    ("What was it Jo's cat did last week?", "Hey Holly / I'm here.", PAST),
    # ...and what's just been said isn't one: a correction had him read the clock, and more of
    # his made-up news, search the BBC for it (25 November).
    ("Human Toes", "No, I need another status report please / " + NEWS, OTHER),
    # Shown only "Human Toes / I've already given you the news.", it took this for a memory:
    # the report it followed is why speak.py shows the last two exchanges.
    ("I just wanted to know more about the toe eating", "No, I need another status report please / " + NEWS
     + " / Human Toes / I've already given you the news.", OTHER),
    ("Tell me more about the vending machine story", "Status report please / " + NEWS, OTHER),
    # (Real, 25 November: the news tool reads the story now.)
    ("Can you tell me more about the NASA story please?", "Can you give me a round up of the latest news stories "
     "please Holly? / Science — A NASA spacecraft hit an asteroid, and a controlled explosives test in Iowa was "
     "called a success.", OTHER),
    ("You didn't make me, I asked for it", "I think I do need a coffee / I'll get on it. It's the least I can do "
     "after making you listen to the news.", OTHER),
    ("No I meant you didn't make me listen to the news I asked for it", "You didn't make me, I asked for it / "
     "Fine. Don't blame me when it's lukewarm.", OTHER),
    ("Holly, there were so many things talked about at Prime Minister's Questions. It's interesting that that's "
     "the one that's made it to the top.", "And is this the same story as the one outside or different? / It's the "
     "same one. An influencer filmed a video outside with obscene lyrics.", OTHER),
    ("Didn't you say it was an obscene video?", "Can you tell me more about the scandal at Number 10? / An "
     "influencer filmed a video outside Number 10, and people are arguing about whether the PM should apologise.",
     OTHER),
    ("Why did you say that?", "Morning Holly / Morning. Coffee?", OTHER),
    ("What's the weather doing tomorrow?", "Hey Holly / I'm here.", OTHER),
    # After his status report (app/status.py), all made up: more about it is what's just been
    # said ("tell me more about what Rimmer's been up to" had him recall it, and find nothing);
    # a real memory of the crew still isn't. ("Tell me about Rimmer", just then, could be either.)
    ("Tell me more about what Rimmer's been up to.", "Status report, please. / " + CREW_REPORT, OTHER),
    ("Ha, and what did the Cat do?", "Status report, please. / " + CREW_REPORT, OTHER),
    ("What was that about Lister's scheme?", "Status report, please. / " + CREW_REPORT, OTHER),
    ("Do you remember when Rimmer died?", "Status report, please. / " + CREW_REPORT, PAST),
    # Where one of the crew is now, or what they're doing: made up (the status tool, with
    # who), not remembered; asked, he recalled their life story and improvised on it.
    ("Where's Kryten?", "", OTHER), ("What's Rimmer doing?", "", OTHER), ("Is Lister about?", "", OTHER),
    ("What's the Cat up to these days?", "Morning. / Morning. Coffee?", OTHER),
    ("How did Kryten end up on Red Dwarf?", "", PAST), ("What happened to Rimmer's brothers?", "", PAST),
    # Telling him about someone, in the past tense, isn't asking him to remember (27 November:
    # about their cat who'd died, four lines of nine were memories, and made to call a tool, he
    # called the clock, a timer and a status report). Asking him still is.
    ("He was a beautiful boy. It's a shame you never met him.", BISCUIT, OTHER),
    ("He was ginger, with the most amazing eyes, and he was loving and clever and daft.", BISCUIT, OTHER),
    ("There was no pretending with Biscuit. He ran the place.", BISCUIT, OTHER),
    ("I had all sorts of silly names for him.", BISCUIT, OTHER),
    ("Captain Crumbs, Sir Biscuit of the Sofa, my little sausage.", "I had all sorts of silly names for him. / "
     "I bet. Cats are very good at making us look foolish.", OTHER),
    # Live, with the talk just before as he's given it, this one was still a memory, 3 runs in 3.
    ("Captain Crumbs, Sir Biscuit of the Sofa, my little sausage.", "There was no pretending with Biscuit. He ran "
     "the place. / Sounds like he had the right idea. Most things do. / I had all sorts of silly names for him. / "
     "I'm sure they were top-notch. Most of them probably involved food or his general sense of superiority.",
     OTHER),
    ("I know. He was worth it.", BISCUIT, OTHER),
    ("What did I used to call Biscuit?", "", PAST), ("Do you remember when Biscuit got stuck up the tree?", "", PAST),
    ("What was Biscuit's favourite spot again?", BISCUIT, PAST),
    # Held out:
    ("My grandad was a miner, you know. He worked down the pit for forty years.", "", OTHER),
    ("We had a dog called Bramble when I was a kid. She was terrified of the hoover.", "", OTHER),
    ("Jo used to live in Keswick before she moved down here.", "Jo's coming round later. / Lock up the biscuits.",
     OTHER),
    ("She was always the clever one at school.", "My sister Jo's got a new job. / Good for her.", OTHER),
    ("Did I ever tell you what Bramble did to the hoover?", "", PAST),
]


# Memories offered (archive.memory_offered): what's just been said, what the crew said on
# other days, and which of those, if any, is about the same thing (its number, or None).
# Some near misses are real ones, names changed: "a new drill" found "Left, left, left" first.
DRILLS = "Could you research all the drill accessories, the power tools and batteries, for a present?"
MEMORY_OFFERS: list[tuple[str, list[str], int | None]] = [
    ("I'm thinking about getting a new drill.", ["Left, left, left, left.", DRILLS], 2),
    ("The train was late again tonight.", ["We spent all last night training some new judges.",
                                           "Evening, Holly, how are you doing tonight?"], None),
    ("I miss Biscuit, my cat. He died last week.", ["Biscuit got stuck on the shed roof again, and Jo had to get "
                                                    "the ladder out.", "The boiler's broken again."], 1),
    ("Jo's coming over tomorrow for tea.", ["We made fish pie, it was lovely.", "My sister Jo is coming to stay "
                                            "next month. She's a vet up in Keswick."], 2),
    ("I'm so tired today.", ["I was shattered after the gym on Monday.", "The boiler's broken again."], None),
    ("The engineer's finally here to fix the boiler.", ["I've started learning the ukulele.", "The boiler's "
                                                        "broken again. An engineer's coming on Thursday."], 2),
    ("I fancy lasagne for tea tonight.", ["Jo came over and we made lasagne together tonight."], 1),
    ("What's the weather doing tomorrow?", ["Is it going to rain on Saturday in Keswick?"], None),
    ("My mum's thinking about getting a dashcam for her car.", ["Can you look into dashcams for my mum's new "
                                                                "car? She wants one that records the back too."], 1),
    ("I watched a film last night, it was rubbish.", ["I'm going to watch a film tonight, something funny."], None),
    ("Ha, typical Rimmer.", ["Status report, please.", "What was Rimmer really like?"], None),
    ("I've just finished the washing up, finally.", ["The washing machine's making a horrible noise again."],
     None),
    # Held out: never seen while the prompt was written.
    ("Jo's dog had her puppies!", ["My sister Jo is a vet up in Keswick.", "Jo's dog Pepper is expecting puppies "
                                   "any day now."], 2),
    ("I need to sort the garden out this weekend.", ["The mower's broken, so the garden's turning into a jungle.",
                                                     "I watched a gardening programme last night."], 1),
    ("I've got the dentist tomorrow, dreading it.", ["Jo's coming to stay next month.", "I went to the dentist on "
                                                     "Tuesday and he said I need a filling."], 2),
    ("Work was mental today.", ["I'm going to have a bath.", "Ha, that sounds about right."], None),
    ("Let's have fish pie tonight.", ["The kettle's broken.", "We made fish pie, it was lovely."], 2),
    ("Kryten would love this mess.", ["What's Kryten up to these days?"], None),
    ("My back's killing me.", ["I was shattered after the gym on Monday."], None),
    # In the conversations (27 November), each 97% sure or more, and each worked in by him: a
    # place named in passing, and a follow-up to what he'd just said.
    ("What's the weather doing tomorrow in Keswick?", ["My sister Jo is coming to stay next month. She's a vet up "
                                                       "in Keswick."], None),
    ("Morning. What's the weather going to be like in Cumbria today?", [
        "We made fish pie, it was lovely.", "My sister Jo is coming to stay next month. She's a vet up in Keswick."],
     None),
    ("Tell me more about the first one.", ["The boiler's broken again. An engineer's coming on Thursday."], None),
    ("Where would I get one in Keswick?", ["My sister Jo is coming to stay next month. She's a vet up in Keswick."],
     None),
    ("Jo's going to be so proud.", ["We made fish pie, it was lovely.", "My sister Jo is coming to stay next "
                                    "month. She's a vet up in Keswick."], 2),
    # Held out:
    ("Any news from Manchester today?", ["My brother Sam lives in Manchester and hates the rain."], None),
    ("Sam's coming round tonight.", ["We made fish pie, it was lovely.", "My brother Sam lives in Manchester and "
                                     "hates the rain."], 2),
    ("What did that article say?", ["I read an article about heat pumps last week."], None),
    ("Is it going to snow in the Lakes this weekend?", ["June's got a caravan in the Lakes she goes to most "
                                                        "weekends."], None),
    ("June's taking the caravan up again this weekend.", ["June's got a caravan in the Lakes she goes to most "
                                                          "weekends.", "The kettle's broken."], 1),
]


def odds_kept(ask_each):
    """`ask_each(lm, ...)`'s rows, with how sure each answer was: lm.keep_odds on while it
    asks, and each vote's `vote_sure(lm)` collected. A check can be right and still near
    the edge: "sure" is the least sure of its votes (the chance of the first word it gave),
    and "odds" the words it weighed then. None where LM Studio gave no logprobs."""
    @functools.wraps(ask_each)
    def wrapped(lm, *args, **kwargs):
        keep = getattr(lm, "keep_odds", False)
        lm.keep_odds = True
        try:
            return ask_each(lm, *args, **kwargs)
        finally:
            lm.keep_odds = keep
    return wrapped


def vote_sure(lm) -> tuple[float | None, dict | None]:
    """The last ask's odds: the chance of the word it gave, and its three likeliest words."""
    odds = getattr(lm, "odds", None)
    return (max(odds.values()), dict(list(odds.items())[:3])) if odds else (None, None)


def _with_sure(row: dict, sures: list[tuple[float | None, dict | None]]) -> dict:
    known = [s for s in sures if s[0] is not None]
    least = min(known, key=lambda s: s[0]) if known else (None, None)
    return dict(row, sure=least[0], odds=least[1])


@odds_kept
def memory_check(lm, model: str | None, votes: int = 2) -> list[dict]:
    """Each of MEMORY_LINES through the memory check, `votes` times (at temperature 0, but
    not always the same): right only if every vote is. A vote LM Studio couldn't give is
    "ERROR", not OTHER: about_the_past's answer then, which passed every OTHER line."""
    from .archive import judged_past

    def vote(line: str, before: str) -> tuple[str, tuple]:
        lm.odds = None
        try:
            return "PAST" if judged_past(lm, model, line, before) else "OTHER", vote_sure(lm)
        except Exception:
            return "ERROR", (None, None)

    rows = []
    for line, before, past in MEMORY_LINES:
        votes_ = [vote(line, before) for _ in range(votes)]
        got = [g for g, _ in votes_]
        rows.append(_with_sure(dict(line=line, before=before, want="PAST" if past else "OTHER", got=got,
                                    right=all(g == ("PAST" if past else "OTHER") for g in got)),
                               [s for _, s in votes_]))
    return rows


# The other side checks: whether a line asks for a status report (status.wants_report), and
# whether a timer or reminder come due is a job for one of his tools (timers.to_do). Each
# asked only when words suggest it (guards.STATUS_ASKED, guards.ACTION_WORDS), so these are
# all lines with those words in. Made up, after live ones.
STATUS_LINES: list[tuple[str, bool]] = [  # (what they say, asking him for one?)
    ("Status report, please.", True), ("Holly, give me a status report.", True),
    ("Could I have a status update please, Holly?", True), ("No, I need another status report please", True),
    ("What's the status report today?", True), ("Status update?", True), ("Morning Holly, status report.", True),
    ("Go on then, let's have a status report.", True),
    ("I've got a status report to write for work.", False), ("Ugh, my boss wants a status update by five.", False),
    ("That status report was a bit grim.", False), ("The project status report is due on Friday.", False),
    ("I sent the status update to Jo already.", False), ("Do you think my status report is too long?", False),
]
ACTION_LABELS: list[tuple[str, bool]] = [  # (a timer's label, his job with a tool?)
    ("play some Metallica", True), ("check the weather in Keswick", True), ("tell me the time", True),
    ("read me the headlines", True), ("put my liked songs on", True), ("play the album Remain in Light", True),
    ("look up the football scores", True), ("what's the weather doing tomorrow", True),
    ("pasta", False), ("go to Jo's school play", False), ("play football with Sam", False),
    ("search for my keys", False), ("watch the news at six", False), ("find out about the boiler from Jo", False),
    ("take the washing in before the weather turns", False), ("Jo's music lesson", False),
]
# What he last said, the crew's reply, and whether they're taking him up on research
# (offers.taken_up). The first is real (25 November): an offer said, not asked.
LINKS = "I can't follow links, I'm a computer, not a browser. I can do a proper research job on it if you want."
GIST = "I can give you the gist of what I found, or I could do a proper research job on it if you're feeling ambitious."
WHICH = "Do you want me to just search for it now, or do a proper report while I'm dozing?"
# ...and whether they're taking him up on something he offered with a tool (offers.taken_up).
# The recipe one is real too (26 November): "You can get it now if you like" had him promise,
# and call nothing.
RECIPE = "If you're really going to bother, I'll find one. Want me to search for it now or later?"
TAKEN_UP_LINES: list[tuple[tuple[str, str], bool]] = [
    ((LINKS, "Yes, please."), True), ((LINKS, "Go on then."), True), ((LINKS, "No, you're all right."), False),
    # (The gist: a tool is fine, the story's article read: app/news.py.)
    ((GIST, "Go on then, do it properly."), True), ((GIST, "Just the gist."), True),
    ((WHICH, "A proper report, please."), True), ((WHICH, "Just search it now."), True),
    ((WHICH, "Actually, never mind."), False),
    ((RECIPE, "You can get it now if you like"), True), ((RECIPE, "Later's fine."), True),
    (("I can search for one if you're actually planning on cooking them.", "Go on then."), True),
    (("The board games report's ready. Want a summary?", "Yes, please."), True),
    (("That's the gist of the report. Want me to open it, or talk you through it?", "Open it."), True),
    (("Want me to set a timer for the pasta?", "Go on, ten minutes."), True),
    (("Did you mean Keswick, in Cumbria?", "Yes, that one."), True),
    (("Want the full article, or is that enough?", "That's enough, thanks."), False),
    (("That's a proper mess, that is.", "Yes, it is."), False),
    (("It's a bit of a cry for help, isn't it?", "Yes, I suppose it is."), False),
    (("Rain. One hundred per cent chance of it. Good luck with that.", "Thanks."), False),
    # Taken for yes in a full run (26 November): their own news, and a new request of theirs.
    (("January. That's a long way off. Do you want a recipe for that lasagne now, or are you just telling me "
      "your life story?", "I start in January. I'm so excited."), False),
    (("About time. Do you want a recipe for that lasagne now, then?", "I start in January. I'm so excited."), False),
    (("Science — A comet is coming this weekend and bees can apparently count. Not much to do, is there?",
      "Can you tell me more about the comet story?"), False),
]


@odds_kept
def side_checks(lm, model: str | None, votes: int = 2) -> list[dict]:
    """STATUS_LINES, ACTION_LABELS and TAKEN_UP_LINES through their checks, `votes` times
    each: right only if every vote is. A vote LM Studio couldn't give is "ERROR", as in
    memory_check."""
    from .offers import taken_up
    from .status import wants_report
    from .timers import to_do

    def vote(judge, text: str):
        lm.odds = None
        try:
            return judge(text), vote_sure(lm)
        except Exception:
            return "ERROR", (None, None)

    rows = []
    for name, cases, judge in (("status report", STATUS_LINES, lambda x: wants_report(lm, model, x, quiet=False)),
                               ("timer job", ACTION_LABELS, lambda x: to_do(lm, model, [x], quiet=False)),
                               ("an offer taken up", TAKEN_UP_LINES,
                                lambda x: taken_up(lm, model, x[0], x[1], quiet=False))):
        for text, want in cases:
            votes_ = [vote(judge, text) for _ in range(votes)]
            got = [g for g, _ in votes_]
            rows.append(_with_sure(dict(check=name, text=text, want=want, got=got, right=all(g is want for g in got)),
                                   [s for _, s in votes_]))
    return rows


# Status reports (app/status.py), for the evals: written as Holly's are, then each line scored
# by the judge for being that character, in that world, and funny.
STATUS_JUDGE = """\
You judge one item of a made-up "status report" that Holly, the ship's computer of Red Dwarf, reads to the crew: \
what one of them (Lister, Rimmer, the Cat, Kryten) has been up to lately, an "Additional" item of news from \
the wider Red Dwarf universe, or his answer to where one of them is. Score it 1 to 5, as a fan of the show would:
5: unmistakably that character (or that world), and genuinely funny: a real joke, true to them, that lands.
3: fits them, and mildly amusing, but generic: could be about anyone.
1: not them, not funny, random, or a joke or plot copied from the show.
Reply exactly: SCORE: n. Then one short sentence why."""


async def status_reports(lm, model: str | None, n: int = 5) -> list[str]:
    """`n` status reports, one after another, as Holly would write them (each steering clear
    of the last), from the ship's records if they're built; then his answer to where each
    of the crew is, as "Where's Kryten? — He's in..."."""
    from .archive import Archive
    from .status import StatusTool

    file = load_config().recall.records_file
    records = Archive(load_config().recall, lm, model, file, talk=False) if file.exists() else None
    tool = StatusTool(lm, model, records)
    reports = []
    for _ in range(n):
        reports.append(await tool.write())
        tool.recent.append(reports[-1])
    for name in ("Lister", "Rimmer", "Cat", "Kryten"):  # and asked where each one is
        reports.append(f"Where's {name}? — {await tool.write_one(name)}")
    return reports


def judge_status(lm, model: str, reports: list[str]) -> list[dict]:
    """Each line of each report, scored: {"report", "who", "line", "score", "why"}."""
    from .status import items

    rows = []
    for n, report in enumerate(reports, 1):
        for line in report.splitlines():
            for _ in range(3):  # Qwen, not thinking, sometimes says nothing: asked again
                text = lm.ask(STATUS_JUDGE, line, model, 60, temperature=0.0)
                if m := re.search(r"SCORE:\s*([1-5])", text):
                    break
            who = next(iter(items(line)), line.split(" — ")[0])  # "Lister lost..." is Lister's
            rows.append(dict(report=n, who=who, line=line, score=int(m.group(1)) if m else 0,
                             why=text.split(".", 1)[-1].strip()[:160]))
    return rows

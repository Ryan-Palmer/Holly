"""Holly's status report: what the crew of Red Dwarf have been up to lately, all made up,
and an "Additional" item from the wider Red Dwarf universe.

Asked for a status report, Holly gave the BBC's news by subject. Now the news is the news
tool's (app/news.py), asked for as such, and a status report is the ship's: a line each
for Lister, Rimmer, the Cat and Kryten, then "Additional". It's written out of his sight,
as search and recall judge out of it: a request of its own, with a few entries from each
crew member's page of the ship's records (who they are, what they're like, what's
happened to them) and a page about the wider universe for "Additional", so the jokes come
from who they are, not from nowhere. Each gets a kind of thing to have been up to, drawn
at random, and the last few reports to steer clear of. Only the finished report goes back
to Holly, to say in his own manner; the records he never sees, so his history stays lean.

Never their quotes: given lines from the show, he recited them (the character pass,
app/persona's examples), so the records' Quotes sections are left out.
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import re
from pathlib import Path

from .console import DIM, RESET, YELLOW, note
from .tools import Tool, Turn

# How a status report's tool result starts.
REPORT = "Your status report on the crew, written for you to give:"
# Reports of before, as they start, for cutting down on a turn with a new one: the news
# round-up (once his status report), as it started then and starts now.
REPORTS = (REPORT, "The BBC's latest, by subject, for a status report:", "The BBC's latest, by subject:")
EARLIER = "(An earlier report: old news now.)"
# After the report, in his history: it's said as written (app/speak.py), and this is for after.
AFTER = ("(You gave this as your status report, as written. It's all made up, yours: asked more about any of it, "
         "make up more, as flatly, in keeping with it. There's nothing to look up.)")
# Asked where one of them is, or what they're doing: his answer, said as written too.
ANSWER = "Your answer about one of the crew, written for you to give:"
AFTER_ANSWER = ("(You gave this as your answer, as written. It's made up, yours: asked more, make up more, in "
                "keeping with it. There's nothing to look up.)")
SAID = (REPORT, ANSWER)  # what's said as written, as a tool's answer starts


def report_of(result: str) -> str:
    """What's to be said, from the status tool's answer: a report, or an answer about one of them."""
    for start, end in ((REPORT, AFTER), (ANSWER, AFTER_ANSWER)):
        if result.startswith(start):
            return result.removeprefix(start).removesuffix(end).strip()
    return result.strip()

# Each crew member: as named in the report, and their page of the ship's records.
CREW = (("Lister", "Dave Lister"), ("Rimmer", "Arnold Rimmer"), ("Cat", "Cat"), ("Kryten", "Kryten"))
# Pages of the wider Red Dwarf universe, for "Additional": in-world, not the show's making.
WORLD = ("Skutters", "GELF", "Pleasure GELF", "Jupiter Mining Corporation", "Talkie Toaster", "Starbug",
         "Space Corps", "Space Corps Directives", "Simulant", "Simulant Battle Cruiser", "Hologram",
         "Hologram Newsreader", "Mechanoid", "Polymorph", "Vending Machine", "The Chocolate Dispenser",
         "Dispenser 34", "Despair Squid", "Medical Unit", "Drive Room", "The Tank",
         "Biologically Engineered Garbage Gobbler")
# What each has been up to, drawn at random: left to choose, every report was a malfunction.
KINDS = ("a scheme to make money", "a new hobby", "a feud with someone aboard", "something they've lost",
         "a machine that's let them down", "food", "an attempt at self-improvement", "a competition",
         "a complaint", "a rule broken", "a gadget", "their appearance", "a misunderstanding",
         "trying to impress someone", "a health scare", "a bet", "an anniversary", "a bad night's sleep",
         "a new rota", "a letter or message", "an experiment", "a phobia")
# Where one of them is, asked: drawn at random (left to choose, nearly all were in the galley).
PLACES = ("the drive room", "the galley", "his quarters", "the cargo bay", "Starbug's cockpit", "the medical bay",
          "the observation dome", "a corridor on Floor 13", "the laundry", "the science room", "the hologram suite",
          "the disused cinema", "the engine room", "the airlock", "the officers' bathroom", "the stasis room",
          "the recreation room", "a storage locker", "the ship's library", "the gym")
NAMES = ("Lister", "Rimmer", "Cat", "Kryten", "Additional")  # a report's items, in order
DRAFTS = 3  # reports written, the best of each item kept: alone, one in three was a dud
NOTES = 3  # entries from each crew member's page
WORLD_NOTES = 2
NOTE_CHARS = 500
RECENT = 3  # last reports, to write something different from

TOOL = {
    "type": "function",
    "function": {
        "name": "status",
        "description": "A status report on the ship: what the crew (Lister, Rimmer, the Cat, Kryten) have been "
                       "up to lately, and an 'Additional' item from the wider Red Dwarf universe. Call it when "
                       "asked for a status report or a status update; with `who`, when asked where one of them "
                       "is or what they're doing now. Not the news: that's the news tool.",
        "parameters": {"type": "object", "properties": {"who": {
            "type": "string", "enum": ["Lister", "Rimmer", "Cat", "Kryten"],
            "description": "Optional: one of the crew, asked where they are or what they're doing now: just them, "
                           "not the whole report."}}},
    },
}

WRITER_PROMPT = """\
You write the status report for Holly, the ship's computer of Red Dwarf: a mining ship three million years \
into deep space, with a crew of four. He reads it out like the news, deadpan: what each of them has been up to \
lately aboard, then one "Additional" item from the wider Red Dwarf universe. All of it is made up, and new: \
small, petty, domestic happenings aboard, not adventures, and never the plot of anything that's happened before.

The joke comes from who they are. For each, build on one specific detail from their notes: a person, a \
possession, a place, something from their past, a skill or the lack of it, named as someone who knows them \
would. Put it in a small new situation of the kind you're given for them, and let their nature drive it \
(Lister's idleness and slobbery, Rimmer's cowardice, pomposity and petty rules, the Cat's vanity, Kryten's \
servility and guilt). Set it up straight, then end on the punchline: the last few words give the flaw away, \
or show it winning again. It must be something that could only be about that one of them, and not the \
obvious every time: Kryten and a speck of dirt, the Cat and a mirror, Lister and a curry have all been done.

"Additional" is an official notice, recall, warning or incident somewhere in their universe, to do with what \
you're given for it, told dead straight: its absurdity is the joke.

Flat and factual, like a bulletin: no exclamation marks, no puns for their own sake, no explaining the joke, \
and no quotes or catchphrases from the show. Begin each with what they did, as a bulletin would ("Lister — \
Lodged a formal complaint..."), each with a different verb. Each is one or two short sentences, at most 25 \
words: it's read aloud. Write exactly five lines, each starting with the name and a dash, and nothing else:
Lister — ...
Rimmer — ...
Cat — ...
Kryten — ...
Additional — ..."""

PICK_PROMPT = """\
You pick the items for a made-up status report on the crew of Red Dwarf, which Holly, the ship's computer, \
reads out deadpan. For each of Lister, Rimmer, Cat, Kryten and Additional you're given options. Pick the one a \
fan of the show would find funniest: a real joke that lands, true to that character (or their world) and only \
them, not random, and not copied from the show. Where another option is as funny, prefer it to the obvious \
(Lister and a curry, the Cat and a mirror, Kryten and a speck of dirt). Reply with exactly five lines, the name \
and the option's number: "Lister: 2", and so on."""

# Asked about one of them: where they are or what they're doing, in Holly's own voice.
ONE_PROMPT = """\
Someone asks Holly, the ship's computer of Red Dwarf (a mining ship three million years into deep space, with a \
crew of four), where {name} is, or what {pronoun}'s doing. Write Holly's answer, as he'd say it, deadpan and dry: \
where {pronoun} is right now and what {pronoun}'s doing, all made up. Something small and domestic aboard, not an \
adventure, and never the plot of anything that's happened before.

The joke comes from who {pronoun} is: build on one specific detail from the notes (a person, a possession, a \
place, something from the past, a skill or the lack of it), in a small new situation of the kind you're given, \
and let {possessive} nature drive it. Set it up straight, then end on the punchline. It must be something that \
could only be about {name}, and not the obvious every time.

One or two short sentences, at most 30 words, beginning with where {pronoun} is ("In the drive room, ..." or \
"{Pronoun}'s in the ..."). No exclamation marks, no quotes or catchphrases from the show, no offers, and nothing \
but the answer."""

PICK_ONE_PROMPT = """\
You pick Holly's answer to where one of the crew of Red Dwarf is, or what they're doing: all made up, said \
deadpan. Pick the one a fan of the show would find funniest: a real joke that lands, true to that character, \
not random, not the obvious, and not copied from the show. Reply with the option's number alone."""

PRONOUNS = {"Lister": ("he", "his"), "Rimmer": ("he", "his"), "Cat": ("he", "his"), "Kryten": ("he", "his")}
_HEADER = re.compile(r"^\(Ship's records: [^)]*\)\s*")
_LINE = re.compile(r"^\W*(Lister|Rimmer|(?:The )?Cat|Kryten|Additional)\b\W*(.+)$", re.IGNORECASE)


# A crew member's sections that say who they are (most of each pick), rather than what
# happened to them (one of the rest): given plot, the writer retold it.
CHARACTER = ("Personality", "Skills", "Skills and hobbies", "Aboard Red Dwarf", "Relationships", "Nicknames",
             "Description", "One of the Boys from the Dwarf", "Childhood", "Biography")


def section(entry: str) -> str:
    """An entry's section: "(Ship's records: Dave Lister, Personality) ..." is "Personality"."""
    head = entry[:entry.find(")")] if ")" in entry else ""
    return head.split(", ", 1)[1] if ", " in head else ""


def notes(records, title: str, n: int, rng: random.Random) -> list[str]:
    """`n` entries from a page of the ship's records, at random, without its Quotes: all but
    one from the sections saying who they are (CHARACTER), if it has them. Each without its
    "(Ship's records: ...)", and cut to NOTE_CHARS."""
    if not records:
        return []
    entries = [e for e in records.page(title) if section(e) != "Quotes"]
    who = [e for e in entries if section(e) in CHARACTER]
    picked = rng.sample(who, min(n - 1, len(who))) if who else []
    rest = [e for e in entries if e not in picked]
    picked += rng.sample(rest, min(n - len(picked), len(rest)))
    return [_HEADER.sub("", e)[:NOTE_CHARS].strip() for e in picked]


def brief(records, recent: list[str], rng: random.Random | None = None) -> str:
    """What the writer is given: for each crew member, a kind of thing to have been up to
    and notes from their page; a page of the wider universe for "Additional"; the last
    reports, to steer clear of."""
    rng = rng or random.Random()
    kinds = rng.sample(KINDS, len(CREW))
    parts = []
    for (name, title), kind in zip(CREW, kinds):
        lines = "\n".join(f"- {n}" for n in notes(records, title, NOTES, rng)) or "- (no notes: what you know of them)"
        parts.append(f"{name}: something to do with {kind}. From their records:\n{lines}")
    world = rng.choice(WORLD)
    lines = "\n".join(f"- {n}" for n in notes(records, world, WORLD_NOTES, rng)) or "- (no notes)"
    parts.append(f"Additional: news from the wider Red Dwarf universe, to do with: {world}. From the records:\n{lines}")
    if recent:
        parts.append("Reports already given lately: write about different things.\n" + "\n\n".join(recent))
    return "\n\n".join(parts)


def brief_one(records, name: str, recent: list[str], said: str = "", rng: random.Random | None = None) -> str:
    """What the writer is given for one of them: a kind of thing to be doing, notes from
    their page, what's been said of them lately (to steer clear of), and what the report
    given this session said of them (to keep to)."""
    rng = rng or random.Random()
    title = dict(CREW)[name]
    lines = "\n".join(f"- {n}" for n in notes(records, title, NOTES, rng)) or "- (no notes: what you know of them)"
    if said:
        # After a report this session: what it said of them, carried on (left to a kind of thing
        # drawn at random, he'd moved on to something else, as if the report hadn't happened).
        return (f"{name}. Holly's status report said of them a little while ago: \"{said}\"\nYour answer carries on "
                f"from exactly that: where they are now, and what they're doing because of it.\n\nFrom their "
                f"records:\n{lines}")
    text = (f"{name}: in {rng.choice(PLACES)}, doing something to do with {rng.choice(KINDS)}. From their "
            f"records:\n{lines}")
    done = [t for report in recent if (t := items(report).get(name))]
    if done:  # given as "said lately", the Cat's letter to his reflection came back, three times in three
        text += "\n\nNot these again, nor anything like them:\n" + "\n".join(f"- {d}" for d in done[-4:])
    return text


def last_given(history: list[dict]) -> str:
    """The last status report in his history, if any: given before he was restarted, it's what
    an answer about one of them carries on from."""
    return next((report_of(m["content"]) for m in reversed(history)
                 if m["role"] == "tool" and (m.get("content") or "").startswith(REPORT)), "")


def who(name) -> str | None:
    """The crew member asked about, as named here: "the Cat" is "Cat", "kryten" "Kryten"."""
    name = str(name or "").strip().lower().removeprefix("the ")
    return next((n for n, _ in CREW if n.lower() == name), None)


_ITEM = re.compile(r"^(?:As for )?(?:The )?(Lister|Rimmer|Cat|Kryten|Additional)\b[\s,:—–-]*(.+)$", re.IGNORECASE)
# A first word that isn't what they did: then "As for Lister, his guitar...", not "Lister his guitar".
_NOT_DONE = {"a", "an", "the", "his", "he", "her", "she", "their", "they", "it", "its", "this", "that", "after",
             "despite", "following", "in", "on", "while", "when", "during"}


def said_as(name: str, text: str) -> str:
    """An item as said: "Lister lost a bet.", "The Cat refused..." Written "Lister — Lost a
    bet.", the voice paused at the dash, and the names came out as headings. "Additional"
    keeps it, a heading: a pause before the made-up news."""
    if name == "Additional":
        return f"Additional — {text}"
    who = "The Cat" if name == "Cat" else name
    first, sep, rest = text.partition(" ")
    if first.lower() not in _NOT_DONE:
        if first[:1].islower():
            return f"{who} {text}"
        if re.fullmatch(r"[A-Z][a-z']+[.,;:!?]*", first):  # "Lost": a verb, at the start of a sentence
            return f"{who} {first.lower()}{sep}{rest}"
    # Not what they did ("His spare heads...", "DivaDroid recalled..."): "As for Kryten, ..."
    lead = first if re.fullmatch(r"[A-Z][a-z']+[.,;:!?]*", first) is None else first.lower()
    return f"As for {'the Cat' if who == 'The Cat' else who}, {lead}{sep}{rest}"


def items(report: str) -> dict[str, str]:
    """A report's items, by name, as said or as written: {"Lister": "lost a bet.", ...}."""
    return {m.group(1).title(): m.group(2).strip() for line in report.splitlines()
            if (m := _ITEM.match(line.strip()))}


def tidy(text: str) -> str:
    """The writer's report, a line an item as "Name — ...", anything else dropped; as it
    came if it isn't in that form at all."""
    out = []
    for line in text.splitlines():
        if m := _LINE.match(line.strip()):
            name = m.group(1).title().removeprefix("The ")
            out.append(f"{name} — {m.group(2).strip()}")
    return "\n".join(out) if len(out) >= 3 else text.strip()


def without_earlier_reports(messages: list[dict], before: int) -> list[dict]:
    """`messages`, with the reports in the first `before` (his history) cut to a line: what
    he was given, and what he said of it. Sent on a turn with a new report. Asked again six
    minutes after the last, he gave the last one again word for word, not as asked. Kept in
    his history as they were: only this request leaves them out."""
    out, report = [], False
    for i, m in enumerate(messages):
        if i < before and m["role"] == "tool" and (m.get("content") or "").startswith(REPORTS):
            m, report = {**m, "content": EARLIER}, True
        elif i < before and report and m["role"] == "assistant" and not m.get("tool_calls"):
            m, report = {**m, "content": EARLIER}, False
        elif m["role"] == "user":
            report = False
        out.append(m)
    return out


# Is the crew's line asking for a status report? Asked only when it says "status report" or
# "status update" (guards.STATUS_ASKED): words alone took "my boss wants a status update
# by five" for one, and he'd have been made to give it.
ASKED_PROMPT = """\
You sort what someone says to Holly, the household computer (and the ship's computer from Red Dwarf). He gives \
a "status report" when asked: what the crew of Red Dwarf have been up to.
YES: they're asking him for one now ("status report, please", "give me a status update, Holly", "another status \
report", "what's the status report today?").
NO: anything else: a status report or update of their own or someone else's ("I've got a status report to \
write", "my boss wants a status update by five"), or a remark about one he's given.
Reply with one word: YES or NO."""


def wants_report(lm, model: str | None, line: str, quiet: bool = True) -> bool:
    """Is `line` asking him for a status report? If it can't be asked, False (he answers as
    he would have), or with `quiet` off (the evals), the error."""
    try:
        return lm.ask(ASKED_PROMPT, f"They say: {line}", model, max_tokens=4,
                      temperature=0).strip().upper().startswith("YES")
    except Exception:
        if not quiet:
            raise
        return False


class StatusTool(Tool):
    """Writes a status report (WRITER_PROMPT), out of his sight, for him to give."""

    name, schema, rules, filler = "status", TOOL, "status", "status"

    def __init__(self, lm, model: str, records=None, file: Path | None = None):
        """`file`: the last few reports, kept from one session to the next (data/status.json),
        so tomorrow's isn't today's again."""
        self.lm, self.model, self.records, self.file = lm, model, records, file
        self.recent: collections.deque[str] = collections.deque(maxlen=RECENT)
        self.given = ""  # the report given this session: what's asked of one of them after keeps to it
        try:
            if file and file.exists():
                self.recent.extend(json.loads(file.read_text(encoding="utf-8"))[-RECENT:])
        except (OSError, ValueError):
            pass  # unreadable: started afresh

    def remember(self, report: str) -> None:
        self.recent.append(report)
        if self.file:
            try:
                self.file.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.file.with_suffix(".tmp")
                tmp.write_text(json.dumps(list(self.recent), indent=1, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.file)
            except OSError as e:
                note(f"couldn't keep the status report: {e!r}")

    async def write(self) -> str:
        """DRAFTS reports, each from its own notes and kinds of thing (one at a time: LM Studio
        runs one request at a time), then the best of each item, picked (PICK_PROMPT)."""
        drafts = []
        for _ in range(DRAFTS):
            text, _ = await self.lm.aask(WRITER_PROMPT, brief(self.records, list(self.recent)), self.model,
                                         max_tokens=450, temperature=0.9)
            drafts.append(items(tidy(text)))
        options = {name: [d[name] for d in drafts if d.get(name)] for name in NAMES}
        if not all(options.values()):  # a draft not in the form asked for: the first that is, as it came
            return next(("\n".join(said_as(n, d[n]) for n in NAMES) for d in drafts if all(d.get(n) for n in NAMES)),
                        "\n".join(said_as(n, o[0]) for n, o in options.items() if o))
        listing = "\n\n".join(f"{name}:\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(opts, 1))
                              for name, opts in options.items())
        pick = await asyncio.to_thread(self.lm.ask, PICK_PROMPT, listing, self.model, 40, 0.0)
        chosen = {m.group(1).title().removeprefix("The "): int(m.group(2))
                  for m in re.finditer(r"(Lister|Rimmer|(?:The )?Cat|Kryten|Additional)\W+(\d)", pick, re.IGNORECASE)}
        return "\n".join(said_as(name, opts[min(max(chosen.get(name, 1), 1), len(opts)) - 1])
                         for name, opts in options.items())

    async def write_one(self, name: str) -> str:
        """Holly's answer about one of them (ONE_PROMPT): DRAFTS of it, the best picked."""
        he, his = PRONOUNS[name]
        prompt = ONE_PROMPT.format(name=name if name != "Cat" else "the Cat", pronoun=he, Pronoun=he.title(),
                                   possessive=his)
        said = items(self.given).get(name, "")
        follow = ""
        if said:
            # After a report this session, what it said of them carried on: told so in the brief
            # alone, the answer moved on to something new, and the pick took the funniest anyway.
            follow = (f"His status report said of {name} a little while ago: \"{said}\" The answer must carry on "
                      "from exactly that (the same business, and where it's led since), not anything new.")
            prompt += "\n\n" + follow
        drafts = []
        for _ in range(DRAFTS):
            text, _ = await self.lm.aask(prompt, brief_one(self.records, name, list(self.recent), said), self.model,
                                         max_tokens=120, temperature=0.9)
            if text := " ".join(text.split()).strip('"'):
                drafts.append(text)
        if len(drafts) < 2:
            return drafts[0] if drafts else ""
        listing = "\n".join(f"{i}. {d}" for i, d in enumerate(drafts, 1))
        pick = await asyncio.to_thread(self.lm.ask, PICK_ONE_PROMPT + (f"\n{follow}" if follow else ""),
                                       f"About {name}:\n{listing}", self.model, 8, 0.0)
        m = re.search(r"\d", pick)
        return drafts[min(max(int(m.group(0)) if m else 1, 1), len(drafts)) - 1]

    async def run(self, args: dict, turn: Turn) -> str:
        name = who(args.get("who"))
        print(f"{DIM}  (status: {'about ' + name if name else 'writing the report'}){RESET}", flush=True)
        try:
            text = await (self.write_one(name) if name else self.write())
        except Exception as e:  # LM Studio: he says the monitors are down, not that he's made it up
            note(f"status failed: {e!r}", YELLOW)
            return (f"The {'monitors on ' + name if name else 'status report'} didn't come through. Tell the crew, "
                    "in a few words, that the monitors are on the blink.")
        if not text:
            return "Nothing came through. Tell the crew, in a few words, that the monitors are on the blink."
        if name:
            self.remember(f"{name} — {text}")
            return f"{ANSWER}\n{text}\n{AFTER_ANSWER}"
        self.remember(text)
        self.given = text
        return f"{REPORT}\n{text}\n{AFTER}"

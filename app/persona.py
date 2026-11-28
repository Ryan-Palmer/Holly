"""Holly's persona: the system prompt, example exchanges and rolling history,
with his notes on older conversation, kept on disk between sessions."""

from __future__ import annotations

import datetime
import json
import os
import re
import tomllib
from pathlib import Path

from . import guards
from .config import PersonaConfig

CHARS_PER_TOKEN = 4  # roughly, for English
TOOL_RESULT_CHARS = 600  # of each tool result, in the transcript the notes are written from


# The day it is, for a replay of a turn from another day (app/replay.py): None, today.
PINNED: datetime.date | None = None


def today_date() -> datetime.date:
    return PINNED or datetime.date.today()


def today() -> str:
    return day(today_date())


def day(d: datetime.date) -> str:
    return d.strftime("%A %d %B %Y").replace(" 0", " ")


def week_around(d: datetime.date) -> str:
    """Yesterday, and the week ahead with how far off each day is, for his system prompt.
    Told only today's date, Gemma counted from the wrong day: asked on a Wednesday, "how
    many days until Friday?" got "one day" sixteen times in sixteen, and Saturday, Monday
    and Tuesday were each a day short too."""
    def short(x: datetime.date) -> str:
        return f"{x:%A} {x.day} {x:%B}"

    ahead = [f"{short(d + datetime.timedelta(days=n))} (in {n} day{'s' if n > 1 else ''}"
             f"{', tomorrow' if n == 1 else ', a week today' if n == 7 else ''})" for n in range(1, 8)]
    return f"Yesterday was {short(d - datetime.timedelta(days=1))}. The days ahead: {', '.join(ahead)}."


def when(t: datetime.datetime) -> str:
    """Like "Friday 21 November 2026 at 9:40 pm"."""
    return f"{day(t.date())} at {t.strftime('%I:%M %p').lstrip('0').lower()}"


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_WORDS = re.compile(
    r"\b(the day after tomorrow|day after tomorrow|tomorrow|tonight|today|yesterday|last night|"
    r"(?:(?:next|this|on|by|until|till|coming|last)\s+)?(?:" + "|".join(WEEKDAYS) + r"))\b",
    re.IGNORECASE)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
# A month used as a month: after a word that says so ("in June", "by March", "the 14th of June",
# "last June"), not a name ("June's coming round", "ask June"): some months are names too.
# With an old friend called June in his notes,
# "Jo's expecting a baby in June!" was noted as "June is expecting a baby in June 2027", 3 runs
# in 3; and in a busier fold, dropped. Dated, as the day words are, a month can't be a person.
_MONTH_WORDS = re.compile(
    r"\b((?:in|by|until|till|since|from|during|for|through|over|before|after|of|next|last|this|"
    r"early|late|mid)(?:\s+(?:early|late|mid|the end of|the start of))?\s+(" + "|".join(MONTHS) + r"))\b(?!'s)",
    re.IGNORECASE)
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
             "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
             "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth",
             "thirtieth", "thirty-first"]
_DAY_OF = re.compile(r"(?:\b(\d{1,2})(?:st|nd|rd|th)?|\b(" + "|".join(o.replace("-", "[- ]") for o in reversed(_ORDINALS))
                     + r"))\s+$", re.IGNORECASE)
# Whether a sentence is about the past or the future, for a bare weekday in it ("I went
# to the dentist on Monday", "Jo's coming on Saturday"). A future word wins: "I said
# I'd ring on Monday" is ahead. Regular "-ed" verbs aren't all listed: "excited about
# Saturday" and "booked for Tuesday" are ahead.
_FUTURE = re.compile(r"\b(will|shall|going to|gonna|plan(?:ning|s)?|booked|due)\b|'ll\b|'d\b", re.IGNORECASE)
_PAST = re.compile(
    r"\b(was|were|went|did|had|saw|met|got|came|left|said|told|bought|made|took|ate|gave|found|"
    r"heard|spent|paid|sent|won|lost|ran|drove|flew|slept|been|ago|visited|played|watched|"
    r"finished|started|called|cooked|walked|talked|moved|stayed|arrived|happened|celebrated)\b",
    re.IGNORECASE)


def resolve_day(words: str, said: datetime.date, past: bool = False) -> datetime.date:
    """The date that day words ("tomorrow", "next Tuesday", "on Saturday")
    mean, said on `said`. A weekday is the next one after the day it was
    said, as "next Tuesday" usually means in speech; or with `past` (or
    "last Tuesday"), the one before it."""
    w = words.lower()
    if w in ("today", "tonight"):
        return said
    if w in ("yesterday", "last night"):
        return said - datetime.timedelta(days=1)
    if w == "tomorrow":
        return said + datetime.timedelta(days=1)
    if "after tomorrow" in w:
        return said + datetime.timedelta(days=2)
    target = WEEKDAYS.index(w.split()[-1])
    first = w.split()[0]
    if first == "last" or (past and first not in ("next", "this", "coming")):
        return said - datetime.timedelta(days=(said.weekday() - target) % 7 or 7)
    return said + datetime.timedelta(days=(target - said.weekday()) % 7 or 7)


def date_days(text: str, said: datetime.date) -> str:
    """`text` with the date after each day word: "coming over tomorrow
    [Sunday 22 November 2026]". For the notes written from a transcript.
    Gemma's own sums were off: given "next Tuesday" said on Wednesday 18
    November, it wrote the 25th, even with a calendar to hand. And with the
    dates in brackets at the end of the line, it put a visit "tomorrow" on
    the day it was said, and gave the next line's note the visit's date.
    A weekday in a sentence about the past is the one before (see _PAST)."""

    def dated(m: re.Match) -> str:
        start = max(text.rfind(c, 0, m.start()) for c in ".!?") + 1
        ends = [i for i in (text.find(c, m.end()) for c in ".!?") if i >= 0]
        sentence = text[start:min(ends) if ends else len(text)]
        past = not _FUTURE.search(sentence) and bool(_PAST.search(sentence))
        return f"{m.group(1)} [{day(resolve_day(m.group(1), said, past))}]"

    def sentence_of(m: re.Match) -> str:
        start = max(text.rfind(c, 0, m.start()) for c in ".!?") + 1
        ends = [i for i in (text.find(c, m.end()) for c in ".!?") if i >= 0]
        return text[start:min(ends) if ends else len(text)]

    def month_dated(m: re.Match) -> str:
        sentence = sentence_of(m)
        past = not _FUTURE.search(sentence) and bool(_PAST.search(sentence))
        name = m.group(2).capitalize()
        before = m.group(1).split()[0]
        # "The third of December": the day too. Marked "[December 2026]", the note said only
        # "December 2026: dentist at ten", and the third was lost.
        if before.lower() == "of" and (d := _DAY_OF.search(text, 0, m.start())):
            n = int(d.group(1)) if d.group(1) else _ORDINALS.index(d.group(2).lower().replace(" ", "-")) + 1
            year = month_year(name, "", said, past)
            try:
                when = datetime.date(year, MONTHS.index(name) + 1, n)
                if not past and when < said:
                    when = when.replace(year=year + 1)
                return f"{m.group(1)} [{day(when)}]"
            except ValueError:
                pass
        return f"{m.group(1)} [{name} {month_year(name, before, said, past)}]"

    text = _DAY_WORDS.sub(dated, text)
    return _MONTH_WORDS.sub(month_dated, text)


def month_year(name: str, before: str, said: datetime.date, past: bool = False) -> int:
    """The year a month means, said on `said`: "last June" the one before, "next June" the
    one after this month, "this June" this year's; otherwise the next one coming, or with
    `past` (a sentence about what happened), the last one gone."""
    m, now, before = MONTHS.index(name) + 1, said.month, before.lower()
    if before == "last":
        return said.year if m < now else said.year - 1
    if before == "next":
        return said.year if m > now else said.year + 1
    if before == "this":
        return said.year
    if past:
        return said.year if m <= now else said.year - 1
    return said.year if m >= now else said.year + 1


def set_aside(path: Path, why: str) -> Path:
    """Rename `path` out of the way, with the time, so nothing earlier set
    aside is overwritten: memory.json -> memory.forgotten-20261121-184002.json."""
    aside = path.with_name(f"{path.stem}.{why}-{datetime.datetime.now():%Y%m%d-%H%M%S}{path.suffix}")
    path.replace(aside)
    return aside


# The keys LM Studio is sent; others (a message's time, "at") stay here.
API_KEYS = ("role", "content", "tool_calls", "tool_call_id")


def about_the_crew(user: str, tools: set[str]) -> bool:
    """Is this exchange about the crew of Red Dwarf: asked about by name (guards.CREW_ASKED),
    or a status report or a memory given? Then, and the exchange after ("what's he doing
    there?"), they're no aside (guards.CREW_ASIDE)."""
    return bool(guards.CREW_ASKED.search(user) or tools & {"status", "recall"})

class Conversation:
    def __init__(self, cfg: PersonaConfig, extra_prompt: str = "", memory_file: Path | None = None,
                 archive=None):
        """`extra_prompt` is added to the system prompt, before the examples:
        the rules for each tool he has (app/search.md and the like). With
        `memory_file`, the history and notes are loaded from it, and saved to
        it after every change. With `archive` (app/archive.py), every exchange
        that leaves the history is stored there."""
        from . import context

        # The lines added to what he's sent (persona/notes.toml), and his system prompt.
        self.notes = context.load_notes(cfg.notes)
        with open(cfg.examples, "rb") as f:
            examples = tomllib.load(f).get("example", [])
        self.system_prompt = context.system_prompt(cfg.system_prompt.read_text(encoding="utf-8"), extra_prompt,
                                                   examples, self.notes)
        # History budgets, in characters (~4 a token).
        self.soft_chars = cfg.history_soft_tokens * CHARS_PER_TOKEN
        self.hard_chars = cfg.history_hard_tokens * CHARS_PER_TOKEN
        self.keep_chars = cfg.history_keep_tokens * CHARS_PER_TOKEN
        self.resume_chars = cfg.history_resume_tokens * CHARS_PER_TOKEN
        self.memory_tokens = cfg.memory_tokens
        self.history: list[dict] = []
        self.memory = ""  # his notes on the older history, written while he sleeps
        self.memory_file = memory_file
        self.archive = archive
        self.last_spoke: datetime.datetime | None = None  # the last exchange, this session or before
        self.previous_end: datetime.datetime | None = None  # when the last session's talk ended
        self._save_failed = False
        # His replies sent without what the reply nets cut (messages_for); off with them
        # (Holly.nets_act, evals --raw), to see the model as it is.
        self.clean = True
        self._made_for: tuple = ()
        self.preamble: list[dict] = []
        self.load()

    def worn_phrases(self, replies: int = 8, at_most: int = 4) -> list[str]:
        """Phrases in more than one of his last few replies (context.worn_phrases)."""
        from .context import worn_phrases

        return worn_phrases(self.history, replies, at_most)

    def messages_for(self, user_text: str | None = None, taken_up: bool | None = None,
                     memory: str = "") -> list[dict]:
        """What he's sent for a new line (context.messages_for)."""
        from .context import messages_for

        return messages_for(self, user_text, taken_up, memory)

    def record(self, user_text: str, holly_text: str, steps: list[dict] | None = None) -> int:
        """Add an exchange: what was said, plus any `steps` between (his
        searches and their results, as sent to the model).

        The steps stay in the history. Without them, his history showed
        factual questions answered straight off, and after a few exchanges he
        stopped searching and said "I've searched the internet" instead.

        The history is normally kept in bounds by compacting it while he
        sleeps (see compaction_split). If he's kept talking past
        `history_hard_tokens`, the oldest exchanges are dropped at once, in
        one chunk, down to two thirds of that: LM Studio reuses its cache only
        while the start of the prompt is unchanged, so dropping one exchange
        per turn would make it reread the whole history every time. Only size
        counts, not the number of exchanges: it's size that fills the context.
        Returns how many exchanges were dropped."""
        self.last_spoke = datetime.datetime.now().replace(second=0, microsecond=0)
        self.history.append({"role": "user", "content": user_text, "at": self.last_spoke.isoformat()})
        self.history.extend(steps or [])
        self.history.append({"role": "assistant", "content": holly_text})
        dropped = 0
        if self._chars(0) > self.hard_chars:
            starts = self._starts()
            keep = starts[0]
            while self._chars(keep) > self.hard_chars * 2 // 3 and keep != starts[-1]:
                keep = next(i for i in starts if i > keep)
            dropped = sum(m["role"] == "user" for m in self.history[:keep])
            self._archive(keep)
            del self.history[:keep]
        self.save()
        return dropped

    # Compacting ------------------------------------------------------------

    def tokens(self) -> int:
        """The history's rough size in tokens."""
        return self._chars(0) // CHARS_PER_TOKEN

    def needs_compacting(self) -> bool:
        return self._chars(0) > self.soft_chars and len(self._starts()) > 1

    def compaction_split(self, keep_chars: int | None = None) -> int:
        """Where to split the history for compacting: everything before the
        index is summarised into his notes, the rest (the newest exchanges,
        up to `keep_chars`, `history_keep_tokens` by default, and always the
        last one) is kept word for word. 0 if there's nothing to compact."""
        keep_chars = self.keep_chars if keep_chars is None else keep_chars
        starts = self._starts()
        if len(starts) < 2:
            return 0
        keep = starts[-1]
        for i in reversed(starts[1:-1]):
            if self._chars(i) > keep_chars:
                break
            keep = i
        return keep

    def resume_split(self) -> int:
        """At startup, where to split the last session's history: all but its
        newest `history_resume_tokens` goes into his notes. 0 if it all fits."""
        if self._chars(0) <= self.resume_chars:
            return 0
        return self.compaction_split(self.resume_chars)

    def transcript(self, end: int) -> str:
        """history[:end] as plain text, for writing notes on, with the day
        and time each exchange began. Of an exchange where he recalled, only
        what the crew said: the answers are already kept (the archive, the
        ship's records), and given them, the notes filled with Red Dwarf
        ("About the crew: Lister, Cat..."). The crew's line stays: "Remember
        Jo? She's coming on Friday now" is news."""
        return "\n".join(text for _, text in self.exchanges(end, recalled=False))

    def exchanges(self, end: int, recalled: bool = True) -> list[tuple[str | None, str]]:
        """history[:end] as one plain-text entry per exchange, each with when
        it was said (ISO, or None for history from before that was kept).
        Without `recalled`, an exchange where he used recall is cut down to
        what the crew said."""
        out: list[tuple[str | None, list[str], list[str]]] = []  # (when, its lines, the crew's own)
        skip: set[int] = set()
        for m in self.history[:end]:
            text = (m.get("content") or "").strip()
            if m["role"] == "user" or not out:
                out.append((m.get("at"), [], []))
            lines, crew = out[-1][1], out[-1][2]
            if any(c["function"]["name"] == "recall" for c in m.get("tool_calls") or []):
                skip.add(len(out) - 1)
            if m["role"] == "user":
                if at := m.get("at"):
                    said = datetime.datetime.fromisoformat(at)
                    lines.append(f"(said on {when(said)})")
                    crew.append(lines[-1])
                    text = date_days(text, said.date())
                lines.append(f"Crew: {text}")
                crew.append(lines[-1])
            elif m["role"] == "assistant":
                for call in m.get("tool_calls") or []:
                    f = call["function"]
                    lines.append(f"(Holly used the {f['name']} tool: {f['arguments']})")
                if text:
                    lines.append(f"Holly: {text}")
            elif m["role"] == "tool":
                lines.append(f"(It told him: {' '.join(text.split())[:TOOL_RESULT_CHARS]})")
        return [(at, "\n".join(lines if recalled or i not in skip else crew))
                for i, (at, lines, crew) in enumerate(out) if recalled or i not in skip or crew]

    def last_about_the_crew(self) -> bool:
        """Was the last exchange about the crew of Red Dwarf (about_the_crew)?"""
        start = next((i for i in range(len(self.history) - 1, -1, -1) if self.history[i]["role"] == "user"), None)
        if start is None:
            return False
        tools = {c["function"]["name"] for m in self.history[start:] for c in m.get("tool_calls") or []}
        return about_the_crew(self.history[start].get("content") or "", tools)

    def crew_lines(self, n: int = 40, chars: int = 300) -> str:
        """What the crew have said in the history, newest last, each with when: for
        recall's judge, which otherwise saw only the archive. Asked who was coming
        over, recall found nothing (the visit hadn't left his history yet), and he
        said he didn't remember, though it was in front of him."""
        lines = []
        for m in self.history:
            if m["role"] == "user" and (text := (m.get("content") or "").strip()) and not text.startswith("["):
                said = f"{when(datetime.datetime.fromisoformat(m['at']))}: " if m.get("at") else ""
                lines.append(f"- {said}{' '.join(text.split())[:chars]}")
        return "\n".join(lines[-n:])

    def _archive(self, end: int) -> None:
        """Keep history[:end] in the archive, as it leaves the history. A
        failure is reported, not raised: losing the archive's copy is better
        than losing the turn."""
        if not self.archive or not end:
            return
        try:
            self.archive.add(self.exchanges(end))
        except Exception as e:  # a locked or broken database file
            print(f"Couldn't add to Holly's archive: {e!r}", flush=True)

    def apply_compaction(self, end: int, dropped: list[dict], notes: str) -> bool:
        """Replace history[:end] with `notes`, if the history still starts
        with the messages that were summarised (`dropped`)."""
        if len(self.history) < end or any(a is not b for a, b in zip(self.history[:end], dropped)):
            return False
        self._archive(end)
        del self.history[:end]
        self.memory = notes.strip()
        self.save()
        return True

    def _starts(self) -> list[int]:
        return [i for i, m in enumerate(self.history) if m["role"] == "user"]

    def _chars(self, start: int) -> int:
        return sum(len(m.get("content") or "") + len(str(m.get("tool_calls", "")))
                   for m in self.history[start:])

    def reset(self) -> None:
        """Forget everything, on disk too, the archive included."""
        self.history.clear()
        self.memory = ""
        if self.archive:
            self.archive.clear()
        self.last_spoke = self.previous_end = None
        self.save()

    # On disk -------------------------------------------------------------------

    def load(self) -> None:
        """Pick up where the last session left off. A file that won't read (a
        slip while editing it by hand) is set aside, and he starts afresh,
        rather than not starting at all."""
        if not self.memory_file or not self.memory_file.exists():
            return
        try:
            saved = json.loads(self.memory_file.read_text(encoding="utf-8"))
            history = saved.get("history", [])
            memory = saved.get("memory", "") or ""
            at = saved.get("last_spoke")
            last = datetime.datetime.fromisoformat(at) if at else None
            if not isinstance(memory, str) or not all(
                    isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool") for m in history):
                raise ValueError("unexpected contents")
        except (ValueError, TypeError, AttributeError, UnicodeDecodeError) as e:
            aside = set_aside(self.memory_file, "unreadable")
            print(f"Holly's memory file couldn't be read ({e}); it's set aside as {aside.name}, "
                  "and he starts afresh.", flush=True)
            return
        self.history, self.memory = history, memory
        self.last_spoke = self.previous_end = last

    def save(self) -> None:
        """Write the history and notes out, whole: to a temporary file first,
        then renamed over the old one, so a crash can't leave half a file."""
        if not self.memory_file:
            return
        state = {"memory": self.memory, "history": self.history,
                 "last_spoke": self.last_spoke.isoformat() if self.last_spoke else None}
        tmp = self.memory_file.with_suffix(".tmp")
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.memory_file)
            self._save_failed = False
        except OSError as e:
            if not self._save_failed:  # once, not every turn
                print(f"Couldn't save Holly's memory to {self.memory_file}: {e}", flush=True)
            self._save_failed = True

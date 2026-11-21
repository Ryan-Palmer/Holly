"""Holly's persona: the system prompt, example exchanges and rolling history,
with his notes on older conversation, kept on disk between sessions."""

from __future__ import annotations

import datetime
import json
import os
import re
import tomllib
from pathlib import Path

from .config import PersonaConfig

CHARS_PER_TOKEN = 4  # roughly, for English
TOOL_RESULT_CHARS = 600  # of each tool result, in the transcript the notes are written from


def today() -> str:
    return day(datetime.date.today())


def day(d: datetime.date) -> str:
    return d.strftime("%A %d %B %Y").replace(" 0", " ")


def when(t: datetime.datetime) -> str:
    """Like "Friday 21 November 2026 at 9:40 pm"."""
    return f"{day(t.date())} at {t.strftime('%I:%M %p').lstrip('0').lower()}"


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_WORDS = re.compile(
    r"\b(the day after tomorrow|day after tomorrow|tomorrow|tonight|today|yesterday|"
    r"(?:(?:next|this|on|by|until|till|coming|last)\s+)?(?:" + "|".join(WEEKDAYS) + r"))\b",
    re.IGNORECASE)
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
    if w == "yesterday":
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

    return _DAY_WORDS.sub(dated, text)


def set_aside(path: Path, why: str) -> Path:
    """Rename `path` out of the way, with the time, so nothing earlier set
    aside is overwritten: memory.json -> memory.forgotten-20261121-184002.json."""
    aside = path.with_name(f"{path.stem}.{why}-{datetime.datetime.now():%Y%m%d-%H%M%S}{path.suffix}")
    path.replace(aside)
    return aside


# The keys LM Studio is sent; others (a message's time, "at") stay here.
API_KEYS = ("role", "content", "tool_calls", "tool_call_id")


class Conversation:
    def __init__(self, cfg: PersonaConfig, extra_prompt: str = "", memory_file: Path | None = None,
                 archive=None):
        """`extra_prompt` is added to the system prompt, before the examples:
        the rules for looking things up, when Holly can search. With
        `memory_file`, the history and notes are loaded from it, and saved to
        it after every change. With `archive` (app/archive.py), every exchange
        that leaves the history is stored there."""
        self.system_prompt = cfg.system_prompt.read_text(encoding="utf-8").strip()
        if extra_prompt:
            self.system_prompt += "\n\n" + extra_prompt.strip()
        with open(cfg.examples, "rb") as f:
            examples = tomllib.load(f).get("example", [])
        # The examples go inside the system prompt, not as fake earlier turns.
        # As turns, they made Qwen 3.8 (thinking off) answer with nothing at
        # all once the real history passed about three exchanges.
        if examples:
            lines = [f'Crew: "{ex["user"]}"\nHolly: "{ex["holly"]}"' for ex in examples]
            self.system_prompt += "\n\nExamples of how you talk:\n\n" + "\n\n".join(lines)
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
        self.reminders: list[str] = []  # from the ship's records (app/records.py), refreshed as he sleeps
        self.last_spoke: datetime.datetime | None = None  # the last exchange, this session or before
        self.previous_end: datetime.datetime | None = None  # when the last session's talk ended
        self._save_failed = False
        self._made_for: tuple = ()
        self.preamble: list[dict] = []
        self.load()
        self._build_preamble()

    def _build_preamble(self) -> None:
        """Holly knows the date (without it he searched for last year's
        results), and has his notes on anything older than the history, and a
        few things from the ship's records to bring up now and then. All go at
        the end of the system prompt, which changes only when one of them does
        (the notes and reminders only while he sleeps), so LM Studio's cache
        survives until then."""
        key = (today(), self.memory, self.previous_end, tuple(self.reminders))
        if self._made_for != key:
            self._made_for = key
            content = f"{self.system_prompt}\n\nToday is {key[0]}."
            if self.previous_end:
                content += (f" You last talked with the crew on {when(self.previous_end)}, "
                            "before you were last switched off.")
            if self.memory:
                content += ("\n\nYour notes on your earlier conversations, which you no longer "
                            f"remember word for word:\n{self.memory}")
            if self.reminders:
                content += ("\n\nThings from the ship's records that have come back to you lately. Now "
                            "and then, when one fits what's being said, mention it in passing as your own "
                            "memory, in your own words. At most one in a reply, and most replies none; "
                            "never force one in.\n" + "\n".join(f"- {r}" for r in self.reminders))
            self.preamble = [{"role": "system", "content": content}]

    def messages_for(self, user_text: str | None = None) -> list[dict]:
        """The full message list to send for a new user line. The time isn't
        here: he asks the clock tool for it (app/clock.py). Without
        `user_text`, just the preamble and history."""
        self._build_preamble()
        tail = [{"role": "user", "content": user_text}] if user_text is not None else []
        history = [{k: m[k] for k in API_KEYS if k in m} for m in self.history]
        return [*self.preamble, *history, *tail]

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
        and time each exchange began."""
        return "\n".join(text for _, text in self.exchanges(end))

    def exchanges(self, end: int) -> list[tuple[str | None, str]]:
        """history[:end] as one plain-text entry per exchange, each with when
        it was said (ISO, or None for history from before that was kept)."""
        out: list[tuple[str | None, list[str]]] = []
        for m in self.history[:end]:
            text = (m.get("content") or "").strip()
            if m["role"] == "user" or not out:
                out.append((m.get("at"), []))
            lines = out[-1][1]
            if m["role"] == "user":
                if at := m.get("at"):
                    said = datetime.datetime.fromisoformat(at)
                    lines.append(f"(said on {when(said)})")
                    text = date_days(text, said.date())
                lines.append(f"Crew: {text}")
            elif m["role"] == "assistant":
                for call in m.get("tool_calls") or []:
                    f = call["function"]
                    lines.append(f"(Holly used the {f['name']} tool: {f['arguments']})")
                if text:
                    lines.append(f"Holly: {text}")
            elif m["role"] == "tool":
                lines.append(f"(It told him: {' '.join(text.split())[:TOOL_RESULT_CHARS]})")
        return [(at, "\n".join(lines)) for at, lines in out]

    def recent_topics(self, n: int = 4) -> list[str]:
        """The crew's last `n` lines, newest first, for picking reminders that fit."""
        # Not the notes he's handed when a timer comes due ("[The timer 'pasta'...]").
        said = [m["content"] for m in self.history if m["role"] == "user" and m.get("content")
                and not m["content"].startswith("[")]
        return said[::-1][:n]

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

"""Holly's timers and reminders: set by a tool, kept on disk, announced when due.

"Set a timer for 20 minutes for the pasta", "remind me at half seven to ring
Mum": Gemma calls the `timer` tool with a duration, or a time of day and
perhaps a day, and the code works out the moment (Gemma's own sums with dates
were off; see persona.resolve_day). They're saved to data/timers.json, so a
reminder survives a restart; one that came due while he was off is announced
when he's next up. When one is due, Holly speaks up unprompted, at the next
pause (app/listen.py, Holly.run).
"""

from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

from .announce import retry_wait
from .console import DIM, RESET
from .persona import day, resolve_day, _DAY_WORDS
from .tools import Tool, Turn

TOOL = {
    "type": "function",
    "function": {
        "name": "timer",
        "description": "Timers and reminders: set one (for a length of time, or for a time of day), "
                       "list what's set, or cancel one. When it's due, you'll be told, and say so.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "list", "cancel"]},
                "label": {"type": "string", "description":
                          "What it's for, in a few words: 'pasta', 'ring Mum'. To cancel: its label, or 'all'."},
                "duration": {"type": "string", "description":
                             "For a timer: how long, like '20 minutes', '1 hour 30 minutes', '90 seconds'."},
                "at": {"type": "string", "description":
                       "For a reminder: the time of day, 24-hour, like '19:30' or '07:00'."},
                "day": {"type": "string", "description":
                        "With 'at', if not the next time it comes round: 'tomorrow', 'Friday', '25 December'."},
            },
            "required": ["action"],
        },
    },
}

_UNITS = {"h": 3600, "hr": 3600, "hour": 3600, "m": 60, "min": 60, "minute": 60, "s": 1, "sec": 1, "second": 1}
_PART = re.compile(r"\b(\d+(?:\.\d+)?|an?|half an?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
                   re.IGNORECASE)
_HALF = re.compile(r"\band a half\b", re.IGNORECASE)  # "an hour and a half": half the unit before it
_H_MM = re.compile(r"\b(\d+)\s*h\s*(\d{1,2})\b(?!\s*[a-z])", re.IGNORECASE)  # "1h30"
# Words in a label that don't tell timers apart, for cancelling: "the pasta timer" is "pasta".
_LABEL_NOISE = {"the", "a", "an", "my", "timer", "timers", "reminder", "reminders", "for", "to", "about"}
_AT = re.compile(r"^\s*(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*$", re.IGNORECASE)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
_DATE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")", re.IGNORECASE)


def parse_duration(text: str) -> datetime.timedelta | None:
    """'20 minutes', '1 hour 30 minutes', '1.5 hours', 'half an hour', '90 seconds'."""
    text = _H_MM.sub(r"\1 hours \2 minutes", text or "")
    total = 0.0
    for m in _PART.finditer(text):
        amount, unit = m.groups()
        n = 0.5 if amount.lower().startswith("half") else 1.0 if amount.lower() in ("a", "an") else float(amount)
        u = unit.lower()
        seconds = _UNITS[u if u in _UNITS else u.rstrip("s")]
        total += n * seconds
        # "an hour and a half" read as an hour: the half goes with this unit.
        if _HALF.match(text[m.end():].lstrip()):
            total += seconds / 2
    if not total and (bare := re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", text or "")):
        total = float(bare.group(1)) * 60  # a bare number: minutes
    return datetime.timedelta(seconds=total) if total > 0 else None


def parse_when(at: str, day_text: str, now: datetime.datetime) -> datetime.datetime | None:
    """A time of day ('19:30', '7:30 pm', '7pm'), on a day if given ('tomorrow',
    'Friday', '25 December'), else the next time it comes round."""
    at = {"noon": "12:00", "midday": "12:00", "midnight": "00:00"}.get((at or "").strip().lower(), at)
    m = _AT.match(at or "")
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if (ampm := (m.group(3) or "").lower().replace(".", "")) == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    on = None
    day_text = (day_text or "").strip()
    if day_text:
        if dm := _DATE.search(day_text):
            try:
                on = datetime.date(now.year, MONTHS.index(dm.group(2).lower()) + 1, int(dm.group(1)))
            except ValueError:
                return None
            if on < now.date():
                on = on.replace(year=now.year + 1)
        elif wm := _DAY_WORDS.search(day_text):
            words = wm.group(1)
            # "Friday", said on a Friday, means today if the time's still to come, else next week.
            if words.lower().split()[-1] == now.strftime("%A").lower() and words.lower().split()[0] != "next":
                on = now.date()
                if datetime.datetime.combine(on, datetime.time(hour, minute)) <= now:
                    on += datetime.timedelta(days=7)
            else:
                on = resolve_day(words, now.date())
        else:
            return None
    when = datetime.datetime.combine(on or now.date(), datetime.time(hour, minute))
    if not on and when <= now:
        when += datetime.timedelta(days=1)  # "at 7": the next 7 o'clock
    return when if when > now else None  # "today at 9" said at 10: gone


def say_when(when: datetime.datetime, now: datetime.datetime) -> str:
    """'19:52 today, in 20 minutes', '07:00 tomorrow', '19:30 on Friday 27 November'."""
    days = (when.date() - now.date()).days
    on = "today" if days == 0 else "tomorrow" if days == 1 else f"on {day(when.date()).rsplit(' ', 1)[0]}"
    left = when - now
    if days == 0 or left < datetime.timedelta(hours=12):
        mins = round(left.total_seconds() / 60)
        span = (f"in {mins // 60} hour{'s' * (mins // 60 != 1)}" + (f" {mins % 60} minutes" if mins % 60 else "")
                if mins >= 60 else f"in {mins} minute{'s' * (mins != 1)}" if mins else
                f"in {round(left.total_seconds())} seconds")
        return f"{when:%H:%M} {on}, {span}"
    return f"{when:%H:%M} {on}"


class Timers:
    """The timers and reminders set, saved to `path` after every change."""

    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict] = []  # {"label", "due" (ISO, local), "kind": "timer" or "reminder"}
        if path.exists():
            try:
                self.items = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                print(f"Couldn't read Holly's timers ({e}); starting with none.", flush=True)

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self.items, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"Couldn't save Holly's timers: {e}", flush=True)

    def handle(self, args: dict, now: datetime.datetime | None = None) -> str:
        """Do what the tool call asks; returns what Holly is told."""
        now = (now or datetime.datetime.now()).replace(microsecond=0)
        action = str(args.get("action") or "set").lower()
        label = " ".join(str(args.get("label") or "").split()) or "timer"
        if action == "list":
            if not self.items:
                return "Nothing is set."
            def listed(i: dict) -> str:
                due = datetime.datetime.fromisoformat(i["due"])
                return f"{i['label']} ({i['kind']}), " + (
                    f"due {say_when(due, now)}" if due > now else
                    f"came due at {due:%H:%M} and hasn't been announced yet")

            # With the time now, and to say it as given: told "in 5 minutes", he said "about a
            # minute", counting on from his own guesses a turn before.
            return ("Set: " + "; ".join(listed(i) for i in sorted(self.items, key=lambda i: i["due"]))
                    + f". (It's {now:%H:%M} now. Say how long is left exactly as given here; never work it out "
                    "yourself.)")
        if action == "cancel":
            gone = self._matching(label)
            if not gone:
                return f"There's nothing set called {label!r}. " + self.handle({"action": "list"}, now)
            self.items = [i for i in self.items if i not in gone]
            self._save()
            return "Cancelled: " + ", ".join(i["label"] for i in gone) + "."
        if duration := parse_duration(str(args.get("duration") or "")):
            due, kind = now + duration, "timer"
        elif args.get("at"):
            due = parse_when(str(args["at"]), str(args.get("day") or ""), now)
            if not due:
                return (f"Nothing set: {args.get('at')!r} (day {args.get('day')!r}) has already passed, or "
                        "isn't a time I can read. If it's a time to come, call the timer tool again with 'at' "
                        "as a 24-hour time like '19:30', and 'day' like 'tomorrow'.")
            kind = "reminder"
        else:
            return ("A timer needs a 'duration' (like '20 minutes'), or a reminder an 'at' time (like '19:30'). "
                    "Call the timer tool again with one.")
        self.items.append({"label": label, "due": due.isoformat(), "kind": kind})
        self._save()
        return (f"Set: {label}, due {say_when(due, now)}. Tell the crew when it's due, exactly as given here, "
                "in your own words.")

    def _matching(self, label: str) -> list[dict]:
        """What a cancel for `label` means: 'all'; the one called exactly that; else
        those whose words hold all its words, or all of whose words it holds ("pasta"
        cancels "pasta timer", "the pasta timer" cancels "pasta"). Whole words: a
        substring match had "steak" cancel "tea". No words to go on ("cancel the
        timer"): the only one set, if there's just one."""
        if label.lower() == "all":
            return list(self.items)
        if exact := [i for i in self.items if i["label"].lower() == label.lower()]:
            return exact

        def words(text: str) -> set[str]:
            return set(re.findall(r"\w+", text.lower())) - _LABEL_NOISE

        asked = words(label)
        if not asked:
            return list(self.items) if len(self.items) == 1 else []
        return [i for i in self.items if (have := words(i["label"])) and (asked <= have or have <= asked)]

    def due(self, now: datetime.datetime | None = None) -> list[dict]:
        """Timers and reminders whose time has come, oldest first; taken off the list."""
        now = now or datetime.datetime.now()
        ready = [i for i in self.items if datetime.datetime.fromisoformat(i.get("retry") or i["due"]) <= now]
        if ready:
            self.items = [i for i in self.items if i not in ready]
            self._save()
        return sorted(ready, key=lambda i: i["due"])

    def put_back(self, items: list[dict], now: datetime.datetime | None = None) -> None:
        """These came due but weren't announced after all (the reply failed): keep
        them, to try again in a minute, then two, then four, up to a quarter of an
        hour apart. Taken off the list and never said, a reminder was simply lost."""
        now = now or datetime.datetime.now()
        for i in items:
            tries = i.get("tries", 0)
            self.items.append({**i, "tries": tries + 1,
                               "retry": (now + datetime.timedelta(seconds=retry_wait(tries))).isoformat()})
        self._save()

    def clear(self) -> None:
        self.items = []
        self._save()


def announcement(items: list[dict], now: datetime.datetime | None = None) -> str:
    """What Holly is told, as the crew's turn, when timers or reminders come due."""
    now = now or datetime.datetime.now()
    lines = []
    for i in items:
        due = datetime.datetime.fromisoformat(i["due"])
        late = now - due
        # Late because he was off, or (tried before) because he couldn't get the words out.
        why = ", and you couldn't say so then" if i.get("tries") else ", while you were switched off"
        when = (f"it came due at {due:%H:%M}" + (f" on {day(due.date())}" if due.date() != now.date() else "")
                + why) if late > datetime.timedelta(minutes=2) else "it's due now"
        lines.append(f"The {i['kind']} '{i['label']}': {when}.")
    # Told only to tell the crew, a timer for "play some Metallica in 10 seconds"
    # came due with "Time for Metallica" and no music: he announced it, as told.
    return ("[" + " ".join(lines) + " If it's for something you do with a tool (playing music, say), "
            "call the tool now and do it; otherwise tell the crew, in a sentence or two, in your own words. "
            "Nobody has spoken: you're speaking up by yourself.]")


# Is a label come due a job for one of his tools? Asked only of labels with a tool's word in
# them (guards.ACTION_WORDS): words alone took "go to Jo's school play" for music.
TO_DO_PROMPT = """\
A timer or reminder the crew set has come due. Holly, the household computer, has tools: music on Spotify, the \
weather, the news, the clock and web search.
DO: the label is something for Holly himself to do now with one of those ("play some Metallica", "put my liked \
songs on", "check the weather in Keswick", "read me the headlines", "tell me the time").
SAY: anything else: something for the crew to do or be reminded of ("pasta", "go to Jo's school play", "play \
football with Sam", "search for my keys", "watch the news at six").
Reply with one word: DO or SAY."""


def to_do(lm, model: str | None, labels: list[str], quiet: bool = True) -> bool:
    """Is any of these labels something he does with a tool? If it can't be asked, False
    (the note still tells him to do it, as before), or with `quiet` off (the evals), the error."""
    try:
        return any(lm.ask(TO_DO_PROMPT, f"The label: {label}", model, max_tokens=4,
                          temperature=0).strip().upper().startswith("DO") for label in labels)
    except Exception:
        if not quiet:
            raise
        return False


class TimerTool(Tool):
    name, schema, rules, local = "timer", TOOL, "timers", True

    def __init__(self, timers: Timers):
        self.timers = timers

    async def run(self, args: dict, turn: Turn) -> str:
        result = self.timers.handle(args)
        print(f"{DIM}  (timer) {result.split('. Tell the crew')[0]}{RESET}", flush=True)
        return result

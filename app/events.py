"""Everything that happens in a turn, written down as it happens: one file a day,
data/events/2026-11-28.jsonl, beside his memory. Each line is an event (JSON):

- "msg": a message, once a file, by its id (its content's hash). Requests and snapshots
  give ids, not messages: his history goes out with every request, and would fill the
  file a hundred times over.
- "state": his history and notes as they are, before a turn, when they're not what the
  file's last turn left them: a new session, a new file, notes written while he dozed,
  history dropped, /reset.
- "request": what he was sent, each time, exactly (ids), with the tool choice; then what
  he wrote, why it was asked again (a net), and the tools he called.
- "turn": the turn whole, at its end: what was said, the side checks' answers, his tools'
  answers, what was said aloud and what was kept, the nets that fired, the times.

Any turn can be rebuilt from this (state_before) and replayed on the real model with the
same notes, the same tools' answers and the same day (app/replay.py). Replaying his real
history found why he said the weather again and again; this makes it a command rather
than a script a time. A log only: it never holds up his reply."""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path


def msg_id(m: dict) -> str:
    return hashlib.sha1(json.dumps(m, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


class EventLog:
    def __init__(self, folder: Path | None):
        """In `folder`; None, nothing's written."""
        self.folder = folder
        self.session = datetime.datetime.now().isoformat(timespec="seconds")
        self.file: Path | None = None
        self.seen: set[str] = set()  # ids written to self.file
        self.left: tuple | None = None  # the state the file's last turn left (fingerprint)
        self._failed = False

    def _path(self) -> Path | None:
        if not self.folder:
            return None
        path = self.folder / f"{datetime.date.today().isoformat()}.jsonl"
        if path != self.file:  # a new day: its file stands alone
            self.file, self.seen, self.left = path, set(), None
        return path

    def _write(self, lines: list[dict]) -> None:
        if not (path := self._path()) or not lines:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
            self._failed = False
        except OSError as e:
            if not self._failed:  # once, not every turn
                print(f"Couldn't write Holly's event log to {path}: {e}", flush=True)
            self._failed = True

    def ids(self, messages: list[dict], out: list[dict]) -> list[str]:
        """The messages' ids; any not yet in the file go into `out`, to be written first."""
        self._path()
        ids = []
        for m in messages:
            i = msg_id(m)
            if i not in self.seen:
                self.seen.add(i)
                out.append(dict(kind="msg", id=i, m=m))
            ids.append(i)
        return ids

    @staticmethod
    def fingerprint(convo) -> tuple:
        h = convo.history
        return (len(h), id(h[0]) if h else None, id(h[-1]) if h else None, convo.memory,
                convo.previous_end.isoformat() if convo.previous_end else None)

    def before_turn(self, convo) -> None:
        """His history and notes, if they're not what the file's last turn left."""
        self._path()
        if self.left == self.fingerprint(convo):
            return
        out: list[dict] = []
        ids = self.ids(convo.history, out)
        out.append(dict(kind="state", session=self.session, at=now(), memory=convo.memory, history=ids,
                        previous_end=convo.previous_end.isoformat() if convo.previous_end else None))
        self._write(out)

    def after_turn(self, convo) -> None:
        self.left = self.fingerprint(convo)

    def request(self, turn: int, messages: list[dict], choice: str | None) -> dict:
        """What he's sent, written now (a crash mid-reply leaves it); returns the event, for
        what he wrote to be added to it in the turn's summary."""
        out: list[dict] = []
        ids = self.ids(messages, out)
        event = dict(kind="request", session=self.session, turn=turn, at=now(), messages=ids, choice=choice)
        self._write([*out, event])
        return event

    def turn(self, event: dict, recorded: list[dict]) -> None:
        out: list[dict] = []
        event = dict(kind="turn", session=self.session, **event, recorded=self.ids(recorded, out))
        self._write([*out, event])

    def clear(self) -> None:
        """/reset: the log goes with everything else of theirs."""
        if self.folder and self.folder.exists():
            shutil.rmtree(self.folder)
        self.file, self.seen, self.left = None, set(), None


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# Reading -------------------------------------------------------------------------


def read(path: Path) -> list[dict]:
    """A day's events, with their messages in place of ids ("messages", "history",
    "recorded"); "msg" events left out."""
    msgs: dict[str, dict] = {}
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue  # a line half-written in a crash
        if e.get("kind") == "msg":
            msgs[e["id"]] = e["m"]
            continue
        for key in ("messages", "history", "recorded"):
            if key in e:
                e[key] = [msgs[i] for i in e[key]]
        events.append(e)
    return events


def turns(events: list[dict]) -> list[dict]:
    """The day's turns, in order, each with its requests ("requests")."""
    out = []
    asked: dict[tuple, list[dict]] = {}
    for e in events:
        if e["kind"] == "request":
            asked.setdefault((e["session"], e["turn"]), []).append(e)
        elif e["kind"] == "turn":
            out.append(dict(e, requests=asked.get((e["session"], e["turn"]), [])))
    return out


def state_before(events: list[dict], turn: dict) -> dict:
    """His history and notes just before `turn` (an event from turns()): the last state
    written before it, and what each turn after that kept."""
    state: dict | None = None
    for e in events:
        if e["kind"] == "turn" and e["session"] == turn["session"] and e["turn"] == turn["turn"]:
            break
        if e["kind"] == "state":
            state = dict(memory=e["memory"], history=[dict(m) for m in e["history"]],
                         previous_end=e["previous_end"])
        elif e["kind"] == "turn" and state is not None and not e.get("failed"):
            state["history"] += [dict(m) for m in e["recorded"]]
    if state is None:
        raise ValueError("no state written before that turn")
    return state


def comparable(messages: list[dict]) -> list[dict]:
    """`messages` with their tool calls' ids numbered in order ("call1", "call2"): LM Studio
    makes up new ids each time, so a replayed request is the same one with other ids."""
    named: dict[str, str] = {}

    def name(i: str | None) -> str | None:
        return None if i is None else named.setdefault(i, f"call{len(named) + 1}")

    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [dict(c, id=name(c.get("id"))) for c in m["tool_calls"]]
        if "tool_call_id" in m:
            m["tool_call_id"] = name(m["tool_call_id"])
        out.append(m)
    return out


def day_file(folder: Path, day: str | None = None) -> Path:
    """The file for `day` (2026-11-28), or the newest."""
    if day:
        return folder / f"{day}.jsonl"
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no event log in {folder}")
    return files[-1]

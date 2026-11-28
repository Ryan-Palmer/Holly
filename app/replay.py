"""A real turn again, from his event log (app/events.py): his history and notes as they were
just before it, on the day it was, with the side checks' answers and his tools' answers it
had then, so his request is the one he was sent. Then the model answers it afresh, as many
times as asked: how often does he say the forecast again? Does a fix stop it?

    python -m app.replay                       the newest day's turns, numbered
    python -m app.replay --day 2026-11-27      another day's
    python -m app.replay 12 --show             turn 12 as it happened: his request, what he
                                               wrote, the nets, his tools, the side checks
    python -m app.replay 12 --runs 6           turn 12 again, six times
    python -m app.replay 12 --say "Fair enough" --runs 6 --match "degrees|gusts"
                                               another line in its place, counting replies
                                               that match

--say asks the side checks afresh (the recorded answers were for another line); --live asks
them, and his tools, afresh for the recorded line too. A tool asked for something not in
the log gets told so, unless --live: a replay mustn't spend searches or set real timers.
Replies go nowhere: silent, his memory in a temporary folder. Not while Holly is running."""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import re
import sys
import tempfile
from pathlib import Path

from . import events as ev
from . import persona
from .config import load_config

NOT_RECORDED = "(Replaying: no answer was recorded for this call.)"


def folder() -> Path:
    return load_config().persona.memory_file.with_name("events")


def listing(turns: list[dict]) -> str:
    def short(text: str, n: int) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= n else text[:n - 1] + "…"

    return "\n".join(f"{i:4}  {t['at'][11:16]}  {short(t['said'], 50):50}  {short(t.get('kept', '(failed)'), 60)}"
                     for i, t in enumerate(turns, 1))


def show(turn: dict) -> str:
    """A turn as it happened."""
    lines = [f"{turn['at']}  {turn['model']}{'' if turn['nets_act'] else '  (raw)'}",
             f"Said: {turn['said']}"]
    if d := turn.get("decisions"):
        lines.append("Side checks: " + "; ".join(f"{k} {v!r}" for k, v in d.items()))
    for n, (req, wrote) in enumerate(zip(turn["requests"], turn["replies"] + [{}] * len(turn["requests"])), 1):
        last = req["messages"][-1]
        lines += ["", f"Request {n} ({len(req['messages'])} messages, tool choice {req['choice']}); its last:",
                  f"  [{last['role']}] {last.get('content') or ''}"]
        if wrote:
            lines.append(f"Wrote: {wrote['raw']!r}" + (f"  calls {[c['name'] for c in wrote['calls']]}"
                                                       if wrote["calls"] else ""))
            if wrote.get("redo"):
                lines.append(f"  asked again: {wrote['redo']}")
            if wrote.get("reasoning"):
                lines.append(f"  reasoning: {wrote['reasoning']!r}")
    for t in turn.get("tools", []):
        lines += ["", f"Tool {t['name']} {t['arguments']}:", f"  {t['result'][:600]}"]
    lines += ["", f"Said aloud: {' '.join(turn.get('spoken', []))}", f"Kept: {turn.get('kept', '(failed)')}"]
    if turn.get("nets"):
        lines.append(f"Nets: {turn['nets']}")
    return "\n".join(lines)


def recorded_tools(holly, recorded: list[dict], live: bool) -> list[str]:
    """Holly's tools answer as they did, from `recorded`, in turn: a call to the same tool
    with the same arguments first, else the same tool. Returns a list each call's source is
    added to ("recorded", "other arguments", "not recorded", "live")."""
    queue = [dict(t) for t in recorded]
    real = holly.toolbox.run
    sources: list[str] = []

    async def run(call, turn, lookups):
        for same_args in (True, False):
            for i, t in enumerate(queue):
                if t["name"] == call["name"] and (not same_args or t["arguments"] == call["arguments"]):
                    sources.append(f"{call['name']}: {'recorded' if same_args else 'recorded, other arguments'}")
                    return queue.pop(i)["result"]
        if live:
            sources.append(f"{call['name']}: live")
            return await real(call, turn, lookups)
        sources.append(f"{call['name']}: not recorded")
        return NOT_RECORDED

    holly.toolbox.run = run
    return sources


def prepare(holly, events: list[dict], turn: dict, say: str | None = None, live: bool = False) -> list[str]:
    """`holly` as he was just before `turn`; returns where each tool's answer comes from."""
    state = ev.state_before(events, turn)
    persona.PINNED = datetime.date.fromisoformat(turn["at"][:10])
    holly.convo.history = copy.deepcopy(state["history"])
    holly.convo.memory = state["memory"]
    holly.convo.previous_end = (datetime.datetime.fromisoformat(state["previous_end"])
                                if state["previous_end"] else None)
    holly.convo._made_for = ()
    holly.turns = turn["turn"] - 1
    decisions = dict(turn.get("decisions") or {})
    holly.replaying = None if live else ({"held": decisions["held"]} if say and "held" in decisions else
                                         {} if say else decisions)
    return recorded_tools(holly, [] if say and not live else turn.get("tools", []), live)


async def replay(events: list[dict], turn: dict, runs: int, say: str | None = None, match: str | None = None,
                 model: str | None = None, live: bool = False) -> list[str]:
    from .scenarios import quiet_holly

    replies, hits = [], 0
    pattern = re.compile(match, re.IGNORECASE) if match else None
    for run in range(runs):
        holly = quiet_holly(model or turn["model"], Path(tempfile.mkdtemp(prefix="holly-replay-")),
                            raw=not turn["nets_act"])
        if run == 0:
            holly.lm.ensure_loaded(holly.model)
        sources = prepare(holly, events, turn, say, live)
        before = len(holly.convo.history)
        await holly.respond(say or turn["said"])
        reply = " ".join(m.get("content") or "" for m in holly.convo.history[before:] if m["role"] == "assistant")
        hit = bool(pattern and pattern.search(reply))
        hits += hit
        replies.append(reply)
        print(f"  run {run + 1}{' MATCH' if hit else ''}: {reply!r}" + (f"  [{'; '.join(sources)}]" if sources else ""),
              flush=True)
    if pattern:
        print(f"{hits} of {runs} replies match {match!r}")
    return replies


class RecordedModel:
    """The model, as it answered then: each request gets the reply written then, word by
    word, with the same tool calls. Side checks must come from the log: asking one fails."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    async def keep_loaded(self, model=None) -> bool:
        return False

    async def astream_reply(self, messages, timing, model=None, tools=None, calls=None, tool_choice=None):
        self.requests.append(messages)
        if not self.replies:
            raise LookupError("asked more often than it was then")
        wrote = self.replies.pop(0)
        for word in re.findall(r"\S+\s*|\s+", wrote["raw"]):
            yield word
        if calls is not None:
            n = len(self.requests)
            calls.extend({"id": f"r{n}_{i}", "name": c["name"], "arguments": c["arguments"]}
                         for i, c in enumerate(wrote["calls"]))

    def ask(self, *args, **kwargs):
        raise LookupError("a side check not in the log")


async def check_turn(events: list[dict], turn: dict, cfg=None) -> str | None:
    """Replays `turn` with the model's replies as they were. None if every request he's sent
    is the one he was sent then (ids aside); else what differs. On `cfg` (holly.toml's, his
    memory in a temporary folder, Spotify off, as the evals' Holly is: a turn from a Holly
    with Spotify on had other rules in his system prompt)."""
    from .scenarios import SilentPlayer, SilentVoice, quiet_config
    from .speak import Holly

    holly = Holly(turn["model"], face=False, cfg=cfg or quiet_config(Path(tempfile.mkdtemp(prefix="holly-check-"))))
    holly.tts, holly.player = SilentVoice(), SilentPlayer()
    holly.nets_act = holly.convo.clean = turn["nets_act"]
    holly.events = ev.EventLog(None)  # not into the log being checked
    model = RecordedModel(turn["replies"])
    holly.lm.keep_loaded, holly.lm.astream_reply, holly.lm.ask = model.keep_loaded, model.astream_reply, model.ask
    prepare(holly, events, turn)
    try:
        await holly.respond(turn["said"])
    except LookupError as e:
        return str(e)
    logged = [r["messages"] for r in turn["requests"]]
    if len(model.requests) != len(logged):
        return f"{len(model.requests)} requests, not {len(logged)}"
    for n, (a, b) in enumerate(zip(model.requests, logged), 1):
        a, b = ev.comparable(a), ev.comparable(b)
        if a != b:
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return f"request {n}, message {i}: {str(a[i] if i < len(a) else None)[:200]!r} " \
                   f"against {str(b[i] if i < len(b) else None)[:200]!r}"
    return None


async def check(paths: list[Path], cfg=None) -> tuple[int, list[str]]:
    """Every turn in these event logs (files, or folders searched for them), replayed with
    the model's replies as they were: for a change that mustn't change what he's sent.
    Returns how many turns were the same, and what differed in the others."""
    files = [f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*.jsonl")))]
    same, differ = 0, []
    for f in files:
        events = ev.read(f)
        for t in ev.turns(events):
            if t.get("failed"):
                continue
            if why := await check_turn(events, t, cfg):
                differ.append(f"{f} turn {t['turn']} ({t['said'][:40]!r}): {why}")
            else:
                same += 1
    return same, differ


def main() -> None:
    p = argparse.ArgumentParser(description="Replay a turn from Holly's event log")
    p.add_argument("turn", nargs="?", type=int, help="the turn's number in the day's listing")
    p.add_argument("--day", help="the day (2026-11-28); the newest if not given")
    p.add_argument("--folder", type=Path, help="the event log's folder (default: beside his memory)")
    p.add_argument("--show", action="store_true", help="show the turn as it happened, and stop")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--say", help="say this in its place")
    p.add_argument("--match", help="count the replies matching this regular expression")
    p.add_argument("--model", help="another model as Holly's brain")
    p.add_argument("--live", action="store_true", help="ask the side checks and his tools afresh")
    p.add_argument("--check", nargs="+", type=Path, metavar="LOG",
                   help="replay every turn in these logs (files or folders) with the model's replies as they "
                        "were, and say which requests aren't the ones he was sent: no model needed")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.check:
        same, differ = asyncio.run(check(args.check))
        print("\n".join(differ))
        print(f"{same} of {same + len(differ)} turns sent the same requests")
        sys.exit(1 if differ else 0)
    try:
        path = ev.day_file(args.folder or folder(), args.day)
        events = ev.read(path)
    except FileNotFoundError as e:
        sys.exit(f"Nothing to replay: {e}")
    turns = ev.turns(events)
    if args.turn is None:
        print(f"{path}:\n{listing(turns)}")
        return
    if not 1 <= args.turn <= len(turns):
        sys.exit(f"{path} has turns 1 to {len(turns)}")
    turn = turns[args.turn - 1]
    if args.show:
        print(show(turn))
        return
    from .scenarios import holly_running

    if running := holly_running():
        sys.exit(f"Holly is running ({running[0][:80]}): close him first")
    asyncio.run(replay(events, turn, args.runs, args.say, args.match, args.model, args.live))


if __name__ == "__main__":
    main()

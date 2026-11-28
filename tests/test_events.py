"""His event log (app/events.py) and replaying a turn from it (app/replay.py)."""

import asyncio
import datetime

from app import events as ev
from app import persona, replay
from conftest import FakeLm

TIMERS = [("timer", {"action": "list"})]


def two_turns(make_holly):
    lm = FakeLm(["Evening. You again.", TIMERS, "None set. Shocking restraint."])
    holly = make_holly(lm)
    asyncio.run(holly.respond("Evening, Holly."))
    asyncio.run(holly.respond("Any timers going?"))
    return holly, lm


def test_each_turn_is_logged_with_what_he_was_sent(make_holly):
    holly, lm = two_turns(make_holly)
    events = ev.read(ev.day_file(holly.events.folder))
    turns = ev.turns(events)
    assert [t["said"] for t in turns] == ["Evening, Holly.", "Any timers going?"]
    second = turns[1]
    assert [r["messages"] for r in second["requests"]] == [r["messages"] for r in lm.requests[1:]]
    assert second["tools"][0]["name"] == "timer" and second["kept"] == "None set. Shocking restraint."
    assert [w["calls"] for w in second["replies"]] == [[dict(name="timer", arguments='{"action": "list"}')], []]
    assert "held" in second["decisions"] and "memory" in second["decisions"]
    # His history and notes before it: as the first turn left them.
    assert ev.state_before(events, second)["history"] == holly.convo.history[:2]
    # Messages are written once a file, not once a request.
    lines = ev.day_file(holly.events.folder).read_text(encoding="utf-8").splitlines()
    assert sum('"kind": "msg"' in ln for ln in lines) == len({ev.msg_id(m) for r in lm.requests
                                                               for m in r["messages"]}
                                                              | {ev.msg_id(m) for m in holly.convo.history})


def test_a_replayed_turn_sends_what_he_was_sent(make_holly, monkeypatch):
    monkeypatch.setattr(persona, "PINNED", None)
    holly, lm = two_turns(make_holly)
    events = ev.read(ev.day_file(holly.events.folder))
    turn = ev.turns(events)[1]
    again = FakeLm([TIMERS, "Still none."], side=lambda s, u: (_ for _ in ()).throw(AssertionError("asked " + s[:40])))
    other = make_holly(again)
    sources = replay.prepare(other, events, turn)
    asyncio.run(other.respond(turn["said"]))
    sent = [ev.comparable(r["messages"]) for r in again.requests]
    assert sent == [ev.comparable(r["messages"]) for r in turn["requests"]] and len(sent) == 2
    assert again.requests[1]["messages"][-1]["content"] == "Nothing is set."  # the recorded answer
    assert ev.comparable([{"role": "tool", "tool_call_id": "x9", "content": ""}]) == [
        {"role": "tool", "tool_call_id": "call1", "content": ""}]
    assert sources == ["timer: recorded"] and not again.side_requests  # side checks given, not asked
    assert persona.PINNED == datetime.date.today()


def test_a_tool_not_in_the_log_isnt_run_on_a_replay(make_holly, monkeypatch):
    monkeypatch.setattr(persona, "PINNED", None)
    holly, _ = two_turns(make_holly)
    events = ev.read(ev.day_file(holly.events.folder))
    other = make_holly(FakeLm([[("clock", {})], "Late, as it happens."]))
    sources = replay.prepare(other, events, ev.turns(events)[0])
    asyncio.run(other.respond("Evening, Holly."))
    assert sources == ["clock: not recorded"]
    assert other.convo.history[-2]["content"] == replay.NOT_RECORDED


def test_his_state_is_written_again_when_it_changes_between_turns(make_holly):
    holly = make_holly(FakeLm(["Evening. You again.", "Noted, for once."]))
    asyncio.run(holly.respond("Evening, Holly."))
    holly.convo.memory = "Jo is coming on Friday."  # notes written while he dozed
    del holly.convo.history[:2]
    asyncio.run(holly.respond("Anything new?"))
    events = ev.read(ev.day_file(holly.events.folder))
    assert [e["kind"] for e in events].count("state") == 2
    state = ev.state_before(events, ev.turns(events)[1])
    assert state["memory"] == "Jo is coming on Friday." and state["history"] == []


def test_a_failed_turn_is_logged_and_kept_out_of_his_state(make_holly):
    holly = make_holly(FakeLm([ConnectionError("LM Studio went"), "Evening. You again."]))
    asyncio.run(holly.respond("Hello?"))
    asyncio.run(holly.respond("Evening, Holly."))
    events = ev.read(ev.day_file(holly.events.folder))
    failed, ok = ev.turns(events)
    assert "ConnectionError" in failed["failed"] and failed["recorded"] == []
    assert ev.state_before(events, ok)["history"] == []
    assert "(failed)" in replay.listing([failed])


def test_reset_clears_the_log(make_holly):
    holly, _ = two_turns(make_holly)
    assert holly.events.folder.exists()
    holly.events.clear()
    assert not holly.events.folder.exists()
    asyncio.run(holly.respond("Hello again."))  # and it starts again, whole
    events = ev.read(ev.day_file(holly.events.folder))
    assert [e["kind"] for e in events].count("state") == 1

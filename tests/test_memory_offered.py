"""Memories offered, unasked (context.memory_for, archive.memories and
memory_offered): what the crew said on another day about what they've just said, given him
with it, if the model judges it the same thing; at most every few replies, each once."""

import asyncio
import dataclasses
import datetime as dt

from app import archive as A
from app.persona import when
from conftest import FakeLm

SAID = dt.datetime(2026, 11, 10, 18, 30)


def entry(at: dt.datetime, crew: str, holly: str = "Noted.", tool: str = "") -> tuple[str, str]:
    middle = f"(Holly used the recall tool: {{}})\n" if tool else ""
    return at.isoformat(), f"(said on {when(at)})\nCrew: {crew}\n{middle}Holly: {holly}"


SHED = entry(SAID, "Biscuit got stuck on the shed roof again, and Jo had to get the ladder out.",
             "A cat with no sense of altitude.")
BOILER = entry(SAID + dt.timedelta(days=1), "The boiler's broken again. An engineer's coming on Thursday.",
               "Again? It's got a death wish.")
HELLO = entry(SAID + dt.timedelta(days=2), "Evening, Holly.", "Evening.")


def test_the_crews_words_and_the_day_only():
    day, said = A.crew_said(SHED[1])
    assert day == "Tuesday 10 November" and said.startswith("Biscuit got stuck") and "altitude" not in said
    assert A.crew_said("Crew: " + "word " * 200)[1].endswith(" ...")


def test_memories_by_keyword_his_replies_left_out(cfg):
    a = A.Archive(cfg.recall, FakeLm(), "m")
    a.add([SHED, BOILER, HELLO])
    found = a.memories("I miss Biscuit so much, he was a daft old thing")
    assert [s for _, _, s in found] == [A.crew_said(SHED[1])[1]]
    assert all("Evening" not in s for _, _, s in a.memories("Evening, Holly, how was your evening?"))


def test_the_model_says_which_or_none():
    found = [(7, "Tuesday 10 November", "Biscuit got stuck on the shed roof"), (9, "", "The boiler's broken")]
    lm = FakeLm(side=lambda system, user: "1")
    assert A.memory_offered(lm, "m", "I miss Biscuit", found) == 7
    assert "1. (Tuesday 10 November) Biscuit" in lm.side_requests[0]["user"]
    assert A.memory_offered(FakeLm(side=lambda s, u: "NONE"), "m", "I miss Biscuit", found) is None
    assert A.memory_offered(FakeLm(side=lambda s, u: "5"), "m", "I miss Biscuit", found) is None
    assert A.memory_offered(FakeLm(side=lambda s, u: "1"), "m", "I miss Biscuit", []) is None


def offering(system: str, user: str) -> str:
    return "1" if system == A.OFFER_PROMPT else ""


def test_offered_with_the_line_not_kept_and_not_again(make_holly):
    lm = FakeLm(["Daft, was he.", "Noted.", "Right.", "Noted.", "Noted."], side=offering)
    holly = make_holly(lm)
    holly.archive.add([SHED])
    asyncio.run(holly.respond("I miss Biscuit, he was a daft old thing."))
    sent = lm.requests[0]["messages"][-1]["content"]
    assert "From your memory" in sent and "shed roof" in sent and "the ladder out" in sent
    assert "altitude" not in sent  # his old reply isn't given him, to copy
    assert holly.memory and "shed roof" not in holly.convo.history[-2]["content"]  # in this request only
    # Not the next reply, nor the one after (offer_every), and the same memory never again.
    for line in ("Biscuit loved that shed, you know.", "Biscuit really did love that shed.",
                 "Biscuit once fell asleep in that shed.", "That shed was Biscuit's favourite place."):
        asyncio.run(holly.respond(line))
        assert "From your memory" not in lm.requests[-1]["messages"][-1]["content"], line


def test_not_for_a_short_line_or_when_turned_off(make_holly, cfg):
    lm = FakeLm(["Noted.", "Noted."], side=offering)
    holly = make_holly(lm)
    holly.archive.add([SHED])
    asyncio.run(holly.respond("Biscuit, eh?"))
    off = make_holly(FakeLm(["Noted."], side=offering),
                     cfg=dataclasses.replace(cfg, recall=dataclasses.replace(cfg.recall, offer_memories=False)))
    off.archive.add([SHED])
    asyncio.run(off.respond("I miss Biscuit, he was a daft old thing."))
    for h in (holly, off):
        assert not h.memory and not any(r["system"] == A.OFFER_PROMPT for r in h.lm.side_requests)


def test_too_slow_is_dropped(make_holly, cfg, monkeypatch):
    import time

    def slow(system, user):
        if system == A.OFFER_PROMPT:
            time.sleep(0.3)
            return "1"
        return ""

    holly = make_holly(FakeLm(["Noted."], side=slow),
                       cfg=dataclasses.replace(cfg, recall=dataclasses.replace(cfg.recall, offer_ms=50)))
    holly.archive.add([SHED])
    asyncio.run(holly.respond("I miss Biscuit, he was a daft old thing."))
    assert not holly.memory and "From your memory" not in holly.lm.requests[0]["messages"][-1]["content"]

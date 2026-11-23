"""Holly's reply loop with a scripted model: what he says, what he's kept, the tools he
runs, and the safety nets that make him redo a reply."""

import asyncio
import datetime as dt

import pytest

from app import speak
from app.speak import LOST_THREAD
from conftest import FakeLm


def respond(holly, *said):
    async def go():
        return [await holly.respond(s) for s in said]
    return asyncio.run(go())


def tools_called(holly) -> list[str]:
    return [c["function"]["name"] for m in holly.convo.history for c in m.get("tool_calls") or []]


def test_a_plain_reply_is_said_and_kept(make_holly):
    holly = make_holly(FakeLm(["Morning. You're up early."]))
    times, = respond(holly, "Morning, Holly.")
    assert holly.tts.said == ["Morning.", "You're up early."]
    assert [m["role"] for m in holly.convo.history] == ["user", "assistant"]
    assert holly.convo.history[-1]["content"] == "Morning. You're up early."
    assert not times.failed and holly.cfg.persona.memory_file.exists()


def test_every_request_sends_the_same_tools(make_holly):
    lm = FakeLm([[("clock", {})], "It's ten past ten.", "Bye."])
    holly = make_holly(lm)
    respond(holly, "What time is it?", "Bye.")
    names = [[t["function"]["name"] for t in r["tools"]] for r in lm.requests]
    assert names[0] == names[1] == names[2]  # the cache needs the same start every time
    assert {"search", "clock", "weather", "recall", "timer", "research"} <= set(names[0])


def test_a_failed_reply_is_said_but_not_kept(make_holly):
    holly = make_holly(FakeLm([ConnectionError("LM Studio went away")]))
    times, = respond(holly, "What's the capital of Peru?")
    assert LOST_THREAD in holly.tts.said and times.failed and holly.convo.history == []


def test_a_broken_tool_is_told_to_him(make_holly, monkeypatch):
    def broken(*a, **k):
        raise KeyError("current")

    monkeypatch.setattr(speak.weather, "forecast", broken)
    lm = FakeLm([[("weather", {"place": "Keswick"})], "Couldn't get the weather."])
    holly = make_holly(lm)
    respond(holly, "What's the weather in Keswick?")
    told = lm.requests[1]["messages"][-1]["content"]
    assert told.startswith("The weather tool broke (KeyError)")
    assert holly.convo.history[-1]["content"] == "Couldn't get the weather."


def test_the_clock(make_holly):
    lm = FakeLm([[("clock", {})], "It's ten past ten."])
    holly = make_holly(lm)
    respond(holly, "What time is it?")
    assert "This is the exact time" in lm.requests[1]["messages"][-1]["content"]


def test_a_time_nobody_gave_him_is_marked_a_guess(make_holly):
    holly = make_holly(FakeLm(["It's 10:14.", [("clock", {})], "It's 10:14."]))
    respond(holly, "What time is it?")
    assert holly.convo.history[-1]["content"].endswith("[That time was a guess, not from the clock: it's probably wrong.]")
    respond(holly, "And now?")  # from the clock: no mark
    assert "guess" not in holly.convo.history[-1]["content"]


def test_a_stray_thought_is_dropped_and_asked_again(make_holly):
    lm = FakeLm(["Thought. ", "Hello there."])
    holly = make_holly(lm)
    respond(holly, "Hello.")
    assert holly.tts.said == ["Hello there."] and len(lm.requests) == 2


def test_timer(make_holly):
    lm = FakeLm([[("timer", {"action": "set", "label": "pasta", "duration": "20 minutes"})], "Twenty minutes."])
    holly = make_holly(lm)
    respond(holly, "Set a timer for 20 minutes for the pasta.")
    assert [i["label"] for i in holly.timers.items] == ["pasta"]
    assert lm.requests[1]["messages"][-1]["content"].startswith("Set: pasta, due")


def with_jo(holly) -> None:
    holly.archive.add([(dt.datetime(2026, 11, 1, 9).isoformat(), "Crew: My sister Jo is a vet.\nHolly: Noted.")])


def judge_says(answer):
    return lambda system, user: answer if "Question:" in user else ""


def test_recall(make_holly):
    lm = FakeLm([[("recall", {"query": "Jo"})], "Jo. Your sister, the vet."], side=judge_says("ANSWER: A vet."))
    holly = make_holly(lm)
    with_jo(holly)
    respond(holly, "What does Jo do?")
    told = lm.requests[1]["messages"][-1]["content"]
    assert told.startswith("A vet.\n(This is your memory of it")


@pytest.mark.parametrize("claim, user", [
    ("I've checked my records and there's nothing on her.", "Who's Jo?"),
    ("Sorry, the name's gone a bit foggy.", "Do you remember Jo?"),
])
def test_saying_he_looked_without_recall_is_redone_with_recall(make_holly, claim, user):
    lm = FakeLm([claim, [("recall", {"query": "Jo"})], "Your sister. The vet."], side=judge_says("ANSWER: A vet."))
    holly = make_holly(lm)
    with_jo(holly)
    respond(holly, user)
    assert claim.split(".")[0] not in " ".join(holly.tts.said)  # never said
    assert lm.requests[1]["tool_choice"] == "required"
    assert tools_called(holly) == ["recall"] and holly.convo.history[-1]["content"] == "Your sister. The vet."


def test_forgetting_without_being_asked_about_memory_is_fine(make_holly):
    holly = make_holly(FakeLm(["I don't remember a thing before Tuesday."]))
    respond(holly, "Tell me a story.")
    assert holly.tts.said == ["I don't remember a thing before Tuesday."]


START = ("research", {"action": "start", "title": "Keeping chickens", "request": "Could we keep chickens?"})


def test_research_not_asked_for_outright_is_held_for_his_question(make_holly):
    lm = FakeLm([[START], "A quick search now, or a proper report while I doze?",
                 [START], "Right, I'll get on with it while I doze."])
    holly = make_holly(lm)
    respond(holly, "Could you look into keeping chickens?")
    assert not holly.research.jobs  # held: he asks which
    assert "Ask them which they'd like" in lm.requests[1]["messages"][-1]["content"]
    respond(holly, "Go on, a report.")
    assert lm.requests[2]["tool_choice"] == "required"  # their answer: a tool must be called
    assert [j["title"] for j in holly.research.jobs] == ["Keeping chickens"]


def test_research_asked_for_outright_starts(make_holly):
    lm = FakeLm([[START], "It's in hand."])
    holly = make_holly(lm)
    respond(holly, "Research keeping chickens for me, a proper report.")
    assert [j["title"] for j in holly.research.jobs] == ["Keeping chickens"]


def test_promising_research_without_starting_it_is_redone(make_holly):
    lm = FakeLm(["I'll put together a proper report while I doze.", [START], "It's in hand."])
    holly = make_holly(lm)
    respond(holly, "Can you research keeping chickens properly?")
    assert lm.requests[1]["tool_choice"] == "required"
    assert holly.research.jobs and "proper report" not in " ".join(holly.tts.said)


def test_asking_isnt_promising(make_holly):
    holly = make_holly(FakeLm(["Shall I have a proper look while I doze?"]))
    respond(holly, "Can you look into chickens?")
    assert holly.tts.said == ["Shall I have a proper look while I doze?"]
    assert holly._answering_choice()  # his own offer: whatever they say next answers it


def test_worn_phrases_are_noted_for_one_request(make_holly):
    lm = FakeLm(["Well, that's a bit odd.", "Well, that's a bit much.", "Fine."])
    holly = make_holly(lm)
    respond(holly, "One.", "Two.", "Three.")
    assert "Lately you've said" in lm.requests[2]["messages"][-1]["content"]
    assert all("Lately" not in (m.get("content") or "") for m in holly.convo.history)


def test_a_failed_timer_announcement_keeps_the_timer(make_holly):
    holly = make_holly(FakeLm([ConnectionError("LM Studio restarting")]))
    holly.timers.items = [{"label": "pasta", "due": "2026-11-22T00:00:00", "kind": "timer"}]
    due = holly.timers.due()
    times = asyncio.run(holly.announce(due))
    assert times.failed and [i["label"] for i in holly.timers.items] == ["pasta"]


def test_a_failed_report_announcement_is_tried_again(make_holly):
    holly = make_holly(FakeLm([ConnectionError("LM Studio restarting"), "Your report's ready. Summary?"]))
    holly.research.shelf.add({"id": "r1", "title": "Keeping chickens", "request": "chickens", "stage": "done",
                              "finished": dt.datetime.now().isoformat(timespec="seconds"), "file": "r1.md",
                              "summary": "Hens need a coop.", "announced": False})
    job = holly.research.ready()
    asyncio.run(holly.announce_report(job))
    assert holly.research.ready() is None  # not yet
    later = dt.datetime.now() + dt.timedelta(minutes=2)
    asyncio.run(holly.announce_report(holly.research.ready(later)))
    assert holly.research.ready(later + dt.timedelta(hours=1)) is None
    assert "is finished, and the report is written" in holly.convo.history[0]["content"]

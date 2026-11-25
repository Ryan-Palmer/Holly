"""Holly's reply loop with a scripted model: what he says, what he's kept, the tools he
runs, and the safety nets that make him redo a reply."""

import asyncio
import datetime as dt

import pytest

from app import announce, archive, research, weather
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
    assert holly.tts.said == ["Morning. You're up early."]  # a word alone goes with the next (speech_text.py)
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

    monkeypatch.setattr(weather, "forecast", broken)
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


def test_saying_he_cant_tell_the_time_is_redone_with_the_clock(make_holly):
    lm = FakeLm(["Tuesday, probably. I've no clock of my own.", [("clock", {})], "It's 23:37. Tuesday tomorrow."])
    holly = make_holly(lm)
    respond(holly, "What day is it tomorrow?")
    assert lm.requests[1]["tool_choice"] == "required" and tools_called(holly) == ["clock"]
    assert "no clock" not in " ".join(holly.tts.said)


def test_a_tool_call_written_as_text_is_never_said(make_holly):
    lm = FakeLm(['recall{query:<|"|>Lister<|"|>}<tool_call|>', [("recall", {"query": "Lister"})], "Not much."],
                side=lambda system, user: "NOTHING")
    holly = make_holly(lm)
    respond(holly, "Do you miss Lister?")
    assert not any("<|" in s for s in holly.tts.said) and lm.requests[1]["tool_choice"] == "required"
    assert holly.convo.history[-1]["content"] == "Not much."


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


def test_a_timer_counted_down_unasked_isnt_said_and_asked_is_looked_up(make_holly):
    lm = FakeLm([[("timer", {"action": "set", "label": "pasta", "duration": "10 minutes"})], "Ten minutes.",
                 "Wholewheat. Six minutes left.",  # unasked, and his guess: not said, nor kept
                 "Six minutes left.",  # asked, and not looked up: asked again, made to look
                 [("timer", {"action": "list"})], "Nine minutes."])
    holly = make_holly(lm)
    respond(holly, "Set a timer for 10 minutes for the pasta.", "It's wholewheat, by the way.",
            "How long's left on the pasta?")
    assert holly.tts.said.count("Six minutes left.") == 0
    replies = [m["content"] for m in holly.convo.history if m["role"] == "assistant" and m.get("content")]
    assert replies == ["Ten minutes.", "Wholewheat.", "Nine minutes."]
    assert lm.requests[2]["tool_choice"] is None and lm.requests[4]["tool_choice"] == "required"
    assert tools_called(holly) == ["timer", "timer"]


def test_a_time_of_day_he_wasnt_given_isnt_said_unasked_and_is_looked_up_asked(make_holly):
    lm = FakeLm(["Wholewheat. It's 22:15.",  # unasked, nobody gave him that: not said, nor kept
                 "It's 22:16.",  # asked, unchecked: asked again, made to look
                 [("clock", {})], "It's ten past ten."])
    holly = make_holly(lm)
    respond(holly, "It's wholewheat, by the way.", "What time is it now?")
    replies = [m["content"] for m in holly.convo.history if m["role"] == "assistant" and m.get("content")]
    assert replies == ["Wholewheat.", "It's ten past ten."]
    assert "It's 22:15." not in holly.tts.said and "It's 22:16." not in holly.tts.said
    assert lm.requests[2]["tool_choice"] == "required" and tools_called(holly) == ["clock"]


def test_unasked_countdowns_are_left_out_of_what_he_sees(cfg):
    from app.persona import Conversation
    c = Conversation(cfg.persona, "", None)
    c.record("It's wholewheat.", "Wholewheat. Six minutes left.")
    c.record("How long left?", "Five minutes left.")  # asked: his answer stays
    sent = [m["content"] for m in c.messages_for() if m["role"] == "assistant"]
    assert sent == ["Wholewheat.", "Five minutes left."]


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


def test_asked_if_he_remembers_he_must_look(make_holly):
    def side(system, user):
        if system == archive.PAST_PROMPT:
            return "PAST" if user.endswith("Do you remember what Jo does?") else "OTHER"
        return judge_says("ANSWER: A vet.")(system, user)
    lm = FakeLm([[("recall", {"query": "Jo"})], "Your sister. The vet.", "Me, obviously."], side=side)
    holly = make_holly(lm)
    with_jo(holly)
    respond(holly, "Do you remember what Jo does?", "Who's cleverer, you or me?")
    assert lm.requests[0]["tool_choice"] == "required" and tools_called(holly) == ["recall"]
    assert lm.requests[1]["tool_choice"] is None  # once he's looked, he answers
    assert lm.requests[2]["tool_choice"] is None  # not a memory question


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
    times = asyncio.run(holly.announce(announce.timers_due(holly.timers)))
    assert times.failed and [i["label"] for i in holly.timers.items] == ["pasta"]


def test_a_timer_to_do_something_must_call_a_tool(make_holly):
    """"Check the weather in Keswick" came due, and he announced it without checking, one
    time in four, though the note told him to use the tool. A "pasta" timer is just said."""
    from app import timers
    lm = FakeLm([[("weather", {"place": "Keswick"})], "Six degrees and drizzle.", "Pasta's done.", "Off you go."],
                side=lambda system, user: ("DO" if "weather" in user else "SAY") if system == timers.TO_DO_PROMPT else "")
    holly = make_holly(lm)
    due = (dt.datetime.now() - dt.timedelta(seconds=5)).isoformat(timespec="seconds")
    holly.timers.items = [{"label": "check the weather in Keswick", "due": due, "kind": "timer"}]
    asyncio.run(holly.announce(announce.timers_due(holly.timers)))
    assert lm.requests[0]["tool_choice"] == "required" and lm.requests[1]["tool_choice"] is None
    holly.timers.items = [{"label": "pasta", "due": due, "kind": "timer"}]
    asyncio.run(holly.announce(announce.timers_due(holly.timers)))
    assert lm.requests[2]["tool_choice"] is None
    # "Play" in a reminder for the crew: the words alone took it for music. The model says no.
    holly.timers.items = [{"label": "go to Jo's school play", "due": due, "kind": "reminder"}]
    asyncio.run(holly.announce(announce.timers_due(holly.timers)))
    assert lm.requests[3]["tool_choice"] is None
    asked = [r["user"] for r in lm.side_requests if r["system"] == timers.TO_DO_PROMPT]
    assert asked == ["The label: check the weather in Keswick", "The label: go to Jo's school play"]  # not pasta


def test_a_failed_report_announcement_is_tried_again(make_holly):
    holly = make_holly(FakeLm([ConnectionError("LM Studio restarting"), "Your report's ready. Summary?"]))
    holly.research.shelf.add({"id": "r1", "title": "Keeping chickens", "request": "chickens", "stage": "done",
                              "finished": dt.datetime.now().isoformat(timespec="seconds"), "file": "r1.md",
                              "summary": "Hens need a coop.", "announced": False})
    asyncio.run(holly.announce(announce.report_ready(holly.research)))
    assert holly.research.ready() is None  # not yet
    later = dt.datetime.now() + dt.timedelta(minutes=2)
    job = holly.research.ready(later)
    asyncio.run(holly.announce(announce.Announcement("again", research.announcement(job),
                                                     lambda ok: holly.research.announced(job, ok))))
    assert holly.research.ready(later + dt.timedelta(hours=1)) is None
    assert "is finished, and the report is written" in holly.convo.history[0]["content"]


def test_a_closing_offer_of_help_is_neither_said_nor_kept(make_holly):
    holly = make_holly(FakeLm(["Fifty-six. Anything else you need me to work out?", "Anything else?"]))
    respond(holly, "What's seven times eight?", "Thanks.")
    assert holly.tts.said == ["Fifty-six.", "Anything else?"]  # alone, it's all there is: said
    assert [m["content"] for m in holly.convo.history if m["role"] == "assistant"] == ["Fifty-six.", "Anything else?"]


def test_asked_about_the_past_in_other_words_he_must_look(make_holly):
    def side(system, user):
        if system == archive.PAST_PROMPT:
            return "PAST" if user.endswith("They say: What did we do yesterday?") else "OTHER"
        return "ANSWER: You made lasagne with Jo." if "Question:" in user else ""
    lm = FakeLm([[("recall", {"query": "yesterday"})], "Lasagne, with Jo. Ambitious.", "Fifty-six."], side=side)
    holly = make_holly(lm)
    respond(holly, "What did we do yesterday?", "What's seven times eight?")
    assert lm.requests[0]["tool_choice"] == "required" and tools_called(holly) == ["recall"]
    assert lm.requests[2]["tool_choice"] is None
    judged = [r["user"] for r in lm.side_requests if r["system"] == archive.PAST_PROMPT]
    assert judged[0] == "They say: What did we do yesterday?"
    assert judged[1].startswith("Just before: What did we do yesterday? / Lasagne")  # for follow-ups


def test_a_status_reports_sections_get_a_longer_pause(make_holly, monkeypatch):
    from app import news
    from app.search import Lookup
    from app.speak import SECTION_PAUSE_S
    monkeypatch.setattr(news, "roundup", lambda cfg, **kw: Lookup("roundup", "The BBC's latest, by subject...", 7, 1, 0))
    report = ("Man City's stalling. Politics' got the age of consent debate. Well — maybe so. "
              "Tech's got AI planning. Additional. The vending machine dispenses warm lemonade.")
    lm = FakeLm([[("news", {"topic": "roundup"})], report, "Science is overrated. Sport's a waste of time."])
    holly = make_holly(lm)
    played: list[int] = []
    holly.player.put = lambda audio: played.append(len(audio))
    respond(holly, "Status report, please.")
    pause = int(holly.tts.sample_rate * SECTION_PAUSE_S)
    assert played.count(pause) == 3  # before Politics, Tech and Additional: his own way of putting them
    assert "Additional — The vending machine dispenses warm lemonade." in holly.tts.said
    played.clear()
    respond(holly, "Thoughts on science?")
    assert played.count(pause) == 0  # no news this turn: sentences like these aren't sections


def test_each_reply_checks_his_model_is_loaded_his_way_first(make_holly):
    """Before anything's asked of LM Studio: had its engine crashed and loaded the model
    again by itself, every prompt longer than its default context was cut, silently."""
    lm = FakeLm(["Yes?", "Drizzle."])
    holly = make_holly(lm)
    lm.reloads = 1
    respond(holly, "Hey Holly")
    assert lm.checks == 1
    respond(holly, "What's the weather doing?")
    assert lm.checks == 2 and holly.convo.history[-1]["content"] == "Drizzle."


REPORT_LINES = ("Lister — Lost a bet.\nRimmer — Filed a complaint.\nCat — Slept.\nKryten — Ironed socks.\n"
                "Additional — A skutter was promoted.")


def side(asked: str = "YES"):
    """The side tasks of a status report: the check (asked), each draft, and the pick."""
    from app import status

    def answer(system: str, user: str) -> str:
        if system == status.ASKED_PROMPT:
            return asked(user) if callable(asked) else asked
        if system == status.WRITER_PROMPT:
            return REPORT_LINES
        if system == status.PICK_PROMPT:
            return "Lister: 1\nRimmer: 1\nCat: 1\nKryten: 1\nAdditional: 1"
        return ""
    return answer


def test_a_status_report_is_said_as_written(make_holly):
    """Told to give it as written, he retold it shorter, and the jokes came out mangled."""
    lm = FakeLm([[("status", {})]], side=side())
    holly = make_holly(lm)
    respond(holly, "Status report, please.")
    assert len(lm.requests) == 1  # the tool call; then no retelling
    assert holly.convo.history[-1]["content"] == REPORT_LINES
    assert holly.tts.said[0].startswith("Lister — Lost a bet.") and "Additional — A skutter was promoted." in \
        " ".join(holly.tts.said)


def test_where_one_of_them_is_is_said_as_written(make_holly):
    """"Where's Kryten?": one answer about him, made up, said as written, not the whole report."""
    from app import status

    def answer(system, user):
        if system.startswith(status.PICK_ONE_PROMPT):
            return "1"
        return "He's in the laundry, apologising to a sock." if system.startswith("Someone asks Holly") else ""

    lm = FakeLm([[("status", {"who": "Kryten"})]], side=answer)
    holly = make_holly(lm)
    respond(holly, "Where's Kryten?")
    assert len(lm.requests) == 1 and holly.convo.history[-1]["content"] == "He's in the laundry, apologising to a sock."
    assert holly.tts.said == ["He's in the laundry, apologising to a sock."]


def test_a_new_status_report_isnt_sent_the_last_one(make_holly):
    from app import status
    lm = FakeLm([[("status", {})], "Science is overrated.", [("status", {})]], side=side())
    holly = make_holly(lm)
    respond(holly, "Status report, please.")
    respond(holly, "Thoughts on science?")
    assert any(m.get("content") == REPORT_LINES for m in lm.requests[1]["messages"])  # other turns have it
    respond(holly, "Another status report.")
    # Not even before he calls the tool: with the last one there, he gave it again, or made
    # one up, without calling. And he must call a tool.
    asked = lm.requests[2]
    contents = [m.get("content") or "" for m in asked["messages"]]
    assert REPORT_LINES not in contents and asked["tool_choice"] == "required"
    assert contents.count(status.EARLIER) == 2  # what he was given, and what he said
    assert sum(m.get("content") == REPORT_LINES for m in holly.convo.history) == 2  # both still in his history


def test_a_status_report_mentioned_isnt_one_asked_for(make_holly):
    """"My boss wants a status update by five" had him made to give the news: the words
    are only a reason to ask the model."""
    from app import status
    lm = FakeLm([[("status", {})], "Deadlines. Lovely."],
                side=side(lambda user: "YES" if "please" in user else "NO"))
    holly = make_holly(lm)
    respond(holly, "Status report, please.")
    respond(holly, "Ugh, my boss wants a status update by five.")
    asked = lm.requests[1]
    assert asked["tool_choice"] is None and REPORT_LINES in [m.get("content") for m in asked["messages"]]
    assert [r["user"] for r in lm.side_requests if r["system"] == status.ASKED_PROMPT] == [
        "They say: Status report, please.", "They say: Ugh, my boss wants a status update by five."]


def test_an_old_news_round_up_counts_as_an_earlier_report(make_holly):
    """Reports kept in his history from when a status report was the news are stubbed too."""
    from app import status
    old = [{"role": "user", "content": "Status report"},
           {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                                                                 "function": {"name": "news", "arguments": "{}"}}]},
           {"role": "tool", "tool_call_id": "1", "content": "The BBC's latest, by subject, for a status report:\n..."},
           {"role": "assistant", "content": "Politics' got a bill."}, {"role": "user", "content": "Status report"}]
    out = status.without_earlier_reports(old, 4)
    assert [m["content"] for m in out[2:4]] == [status.EARLIER, status.EARLIER]


def test_the_memory_check_sees_the_last_two_exchanges():
    """"I just wanted to know more about the toe eating", after "Human Toes" / "I've already
    given you the news", was taken for a memory: the toe eating was his own news, a turn
    before. A long reply keeps its start and its end."""
    from types import SimpleNamespace
    from app.speak import Holly
    report = "Politics — a bill. " + "Sport — a match. " * 60 + "Additional — a vending machine eats toes."
    history = [{"role": "user", "content": "Guess what I want"}, {"role": "assistant", "content": "A nap."},
               {"role": "user", "content": "Status report"},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
               {"role": "tool", "tool_call_id": "1", "content": "The BBC's latest..."},
               {"role": "assistant", "content": report},
               {"role": "user", "content": "Human Toes"}, {"role": "assistant", "content": "I gave you the news."}]
    before = Holly._last_exchange(SimpleNamespace(convo=SimpleNamespace(history=history)))
    assert before.startswith("Status report / Politics — a bill.")
    assert before.endswith("a vending machine eats toes. / Human Toes / I gave you the news.")
    assert "A nap" not in before and "The BBC" not in before and " ... " in before

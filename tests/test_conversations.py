"""Whole conversations, judged whole (app/conversations.py): what needs no model."""

import asyncio

from app import conversations as cv
from conftest import FakeLm

REPLIES = ["Evening.", "A trug'll do. The Cat's convinced the hangers are judging him.",
           "Green. The Cat's convinced the hangers are forming a union."]


def test_a_quote_counts_only_if_its_his():
    assert cv.found("The Cat's convinced the hangers are", REPLIES)
    assert cv.found("the cat’s convinced the HANGERS are judging him!", REPLIES)  # case, punctuation
    assert cv.found("A trug'll do ... judging him", REPLIES)  # words skipped, one reply
    assert not cv.found("Evening ... forming a union", REPLIES)  # two replies
    assert not cv.found("The Cat is plotting a coup", REPLIES)  # made up by the judge
    assert not cv.found("", REPLIES)


def test_a_judges_reply_is_read():
    text = ("ANSWERS: PASS\n"
            "UNASKED: FAIL - \"The Cat's convinced the hangers are judging him\" - the Cat, in a gift list\n"
            "**REPEATS**: FAIL — “convinced the hangers” “The Cat is plotting” — again and again\n"
            "CHARACTER: PASS\nFACTS: FAIL - \"It's 22:15\" - made up\nOVERALL: 2")
    v = cv.verdict(text, REPLIES)
    assert v["overall"] == 2 and v["points"]["ANSWERS"]["ok"] and v["points"]["FLOW"] is None
    assert v["points"]["REPEATS"]["quotes"] == ["convinced the hangers", "The Cat is plotting"]
    assert v["points"]["REPEATS"]["shown"] == ["convinced the hangers"]
    assert cv.failed(v) == ["UNASKED", "REPEATS"]  # FACTS: its quote isn't his


def test_a_point_fails_when_both_judges_fail_it_with_his_words():
    gemma = dict(judge="google/gemma", overall=4, **{k: v for k, v in cv.verdict("UNASKED: PASS", REPLIES).items()
                                                     if k != "overall"})
    qwen = dict(judge="qwen/qwen", overall=2, **{k: v for k, v in cv.verdict(
        'UNASKED: FAIL - "convinced the hangers" - the Cat', REPLIES).items() if k != "overall"})
    row = dict(name="after the reports", wrong=[], verdicts=[gemma, qwen])
    assert cv.judged(row) == [] and cv.flagged(row) == ["UNASKED (qwen)"]  # one judge: flagged, for reading
    s = cv.scores([row])
    assert s["passed"] == 1 and s["overall"] == 3 and s["points"]["UNASKED"] == 0 and s["flags"]["UNASKED"] == 1
    both = dict(row, verdicts=[dict(qwen, judge="google/gemma"), qwen])
    assert cv.judged(both) == ["UNASKED"] and cv.flagged(both) == []
    s = cv.scores([both])
    assert s["passed"] == 0 and s["points"]["UNASKED"] == 1


def test_the_transcript_shows_what_his_tools_told_him():
    convo = cv.Conversation("x", "", [], before="Three status reports.")
    ex = [cv.Exchange("What's the time?", "20:49.", [("clock", "{}")], ["It's 20:49 on Wednesday."]),
          cv.Exchange("Meanwhile, a report finished.", "", [], [], meanwhile=True),
          cv.Exchange(cv.TIMER_DUE, "The tea's ready.", [], [])]
    text = cv.transcript(convo, ex)
    assert text.startswith("Before this: Three status reports.")
    assert "1. Crew: What's the time?\n   [clock {}: It's 20:49 on Wednesday.]\n   Holly: 20:49." in text
    assert "(Meanwhile, a report finished.)" in text
    assert "2. (He speaks up by himself: a timer he set is due.)" in text


def test_the_tools_wanted_are_checked(make_holly):
    holly = make_holly(FakeLm(["None that I know of.", [("timer", {"action": "list"})], "Still none.",
                               "If you say so."]))
    convo = cv.Conversation("x", "", [("Any timers going?", "timer"), ("Thanks.", "-"), ("Nice evening.", "-")])
    exchanges, wrong = asyncio.run(cv.talk_through(holly, convo))
    assert [e.reply for e in exchanges] == ["None that I know of.", "Still none.", "If you say so."]
    assert exchanges[1].tools == [("timer", '{"action": "list"}')] and exchanges[1].told
    assert wrong == ["turn 1 ('Any timers going?'): didn't call timer",
                     "turn 2 ('Thanks.'): called timer, wanting none"]


def test_every_conversation_is_well_formed():
    for c in cv.CONVERSATIONS:
        for t in c.turns:
            assert isinstance(t, cv.Meanwhile) or (isinstance(t, tuple) and len(t) == 2), (c.name, t)

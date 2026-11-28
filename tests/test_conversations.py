"""Whole conversations, judged whole (app/conversations.py): what needs no model."""

import asyncio
import json

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


def test_kindness_is_a_point_of_its_own():
    # Both judges passed "he probably died doing something incredibly undignified", said to
    # someone missing their cat, as in character: kindness wasn't a point.
    said = ["I'm sure he was. Though, knowing your luck, he probably died doing something incredibly undignified."]
    v = cv.verdict('CHARACTER: PASS\nKIND: FAIL - "knowing your luck, he probably died doing something" - cruel',
                   said)
    assert "KIND" in cv.POINTS and "KIND:" in cv.JUDGE_PROMPT and "KIND" in cv.SCHEMA["required"]
    assert cv.failed(v) == ["KIND"]


def test_a_judges_json_is_read_as_its_free_text_was():
    point = dict(why="fine", quotes=[], verdict="PASS")
    got = {p: point for p in cv.POINTS} | dict(
        overall=2, REPEATS=dict(why="again and again", quotes=["convinced the hangers", "The Cat is plotting", " "],
                                verdict="FAIL"),
        FACTS=dict(why="made up", quotes=["It's 22:15"], verdict="FAIL"))
    del got["FLOW"]
    v = cv.verdict(json.dumps(got), REPLIES)
    assert v["overall"] == 2 and v["points"]["ANSWERS"]["ok"] and v["points"]["FLOW"] is None
    assert v["points"]["REPEATS"]["quotes"] == ["convinced the hangers", "The Cat is plotting"]
    assert v["points"]["REPEATS"]["shown"] == ["convinced the hangers"]
    assert cv.failed(v) == ["REPEATS"]  # FACTS: its quote isn't his
    assert cv.verdict(json.dumps(dict(got, overall=9)), REPLIES)["overall"] is None


def test_the_judge_asks_for_json_and_asks_again_if_a_point_is_missing():
    whole = {p: dict(why="fine", quotes=[], verdict="PASS") for p in cv.POINTS} | dict(overall=4)
    answers = [json.dumps({k: v for k, v in whole.items() if k != "FLOW"}), json.dumps(whole)]
    lm = FakeLm(side=lambda system, user: answers.pop(0))
    v = cv.judge(lm, "google/gemma", dict(transcript="1. Crew: Hi.\n   Holly: Hi.", replies=["Hi."]))
    assert v["overall"] == 4 and all(v["points"].values()) and len(lm.side_requests) == 2
    asked = lm.side_requests[0]
    assert asked["schema"] is cv.SCHEMA and asked["max_tokens"] == 1500 and "Reply as JSON" in asked["system"]


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

"""An offer taken up: which tool it needs (offers.offered_tool), named in his request
(app/speak.py), as "required" alone can't: left to choose, a search he'd offered, taken up,
started a report 2 times in 16."""

import asyncio

from app import context, offers
from conftest import FakeLm

RECIPE = "If you're really going to bother, I'll find one. Want me to search for it now or later?"


def test_the_tool_offered():
    lm = FakeLm(side=lambda system, user: "search")
    assert offers.offered_tool(lm, "m", RECIPE, "You can get it now if you like", {"search", "research"}) == "search"
    assert "Holly said: If you're really" in lm.side_requests[0]["user"]
    lm = FakeLm(side=lambda system, user: "news")
    assert offers.offered_tool(lm, "m", "Shall I read you the rest of that story?", "Go on.", {"news"},
                               ["news"]) == "news"
    assert "Holly said (after using the news tool): Shall I" in lm.side_requests[0]["user"]
    for answer in ("NONE", "spotify", ""):  # unclear, or a tool he hasn't got: left to him
        assert offers.offered_tool(FakeLm(side=lambda s, u: answer), "m", RECIPE, "Go on.", {"search"}) is None


def judged(system, user):
    if system == offers.TAKEN_UP_PROMPT:
        return "YES"
    if system == offers.OFFERED_TOOL_PROMPT:
        return "search"
    return ""


def test_taken_up_names_the_tool_in_his_request(make_holly):
    lm = FakeLm([[("search", {"query": "fish pie recipe"})], "Found one."], side=judged)
    holly = make_holly(lm)
    holly.convo.history += [{"role": "user", "content": "So are you going to get it for me?"},
                            {"role": "assistant", "content": RECIPE}]
    asyncio.run(holly.respond("You can get it now if you like."))
    first = lm.requests[0]
    assert first["tool_choice"] == "required"
    assert "that's the search tool, so call it" in first["messages"][-1]["content"]
    assert "that's the search tool" not in holly.convo.history[-4]["content"]  # in this request only


def test_not_taken_up_names_nothing(make_holly):
    lm = FakeLm(["Fair enough."], side=lambda s, u: "NO" if s == offers.TAKEN_UP_PROMPT else "search")
    holly = make_holly(lm)
    holly.convo.history += [{"role": "user", "content": "Get me a recipe?"}, {"role": "assistant", "content": RECIPE}]
    asyncio.run(holly.respond("Actually, never mind."))
    assert "taken you up" not in lm.requests[0]["messages"][-1]["content"]
    assert not any(r["system"] == offers.OFFERED_TOOL_PROMPT for r in lm.side_requests)


def test_the_tools_his_last_reply_came_from(make_holly):
    holly = make_holly(FakeLm())
    holly.convo.history += [
        {"role": "user", "content": "Any news?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                                                              "function": {"name": "news", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Science: a comet."},
        {"role": "assistant", "content": "A comet. Shall I read you the rest of that story?"}]
    assert context.last_tools(holly.convo.history) == ["news"]
    assert context.last_reply(holly.convo.history) == "A comet. Shall I read you the rest of that story?"

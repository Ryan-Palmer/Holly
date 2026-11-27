"""The parts Holly's reply loop is built from: the toolbox, the safety nets' verdicts,
speaking up, and what he does while he sleeps."""

import pytest
import asyncio
import datetime as dt

from app import guards
from app.announce import pending, retry_wait, timers_due
from app.background import Background
from app.persona import Conversation
from app.timers import Timers
from app.tools import Tool, Toolbox, Turn
from conftest import FakeLm


class Echo(Tool):
    name, schema, local = "echo", {"function": {"name": "echo"}}, True

    async def run(self, args, turn):
        return f"echo {args.get('x')} for {turn.user}"


class Slow(Tool):
    name, schema, filler = "slow", {"function": {"name": "slow"}}, "recall"

    async def run(self, args, turn):
        return "slow done"


def test_toolbox():
    box = Toolbox([Echo(), Slow()], max_lookups=1)
    turn = Turn("hi", 1, object())
    run = lambda name, args, n=1: asyncio.run(box.run({"name": name, "arguments": args}, turn, n))
    assert box.schemas() == [Echo.schema, Slow.schema] and Toolbox([], 1).schemas() is None
    assert run("echo", '{"x": 3}') == "echo 3 for hi"
    assert run("echo", "not json") == "echo None for hi"
    assert run("nothing", "{}") == "There is no tool called nothing."
    assert run("slow", "{}", 2).startswith("That's enough searching")
    assert run("echo", "{}", 9).startswith("echo")  # instant ones aren't limited
    assert box.local("echo") and not box.local("slow") and "slow" in box
    assert box.filler(["echo"]) is None and box.filler(["slow"]) == "recall" and box.filler(["echo", "slow"]) == "recall"


def test_his_toolbox_has_every_rule_once(make_holly):
    holly = make_holly(FakeLm())
    names = [t["function"]["name"] for t in holly.tools]
    assert names == ["search", "clock", "weather", "news", "recall", "timer", "research", "status"]  # the order the cache relies on
    system = holly.convo.messages_for()[0]["content"]
    for md in ("Research:", "Timers", "Status reports:"):
        assert system.count(md) >= 1
    assert holly.toolbox.rules().count(holly.toolbox.rules().split("\n")[0]) == 1


def unsaid(sentence, user="Hello.", **kw):
    state = dict(stray={"thought"}, researched=False, can_research=True,
                 answering_choice=False) | kw
    return guards.unsaid(sentence, user, **state)


def test_unsaid():
    assert unsaid("Thought.") == guards.STRAY
    assert unsaid("I've checked my records.") is None  # (the net that caught it never fired: it came out)
    assert "research" in unsaid("I'll look into it while I doze.", "Research slugs.")
    assert unsaid("Shall I look into it while I doze?", "Research slugs.") is None  # a question, not a promise
    assert "research" in unsaid("I'm on it.", "Yes please.", answering_choice=True)
    assert unsaid("I'm on it.", "Yes please.") is None
    assert guards.needs_tool(unsaid("I'm on it.", "Yes please.", answering_choice=True))
    assert not guards.needs_tool(guards.STRAY)
    assert "clock" in unsaid("It's Tuesday, I think. I've no clock of my own.", can_clock=True)
    assert unsaid("I've no clock of my own.", can_clock=True, clocked=True) is None
    assert guards.needs_tool(unsaid("No idea what time it is.", can_clock=True))
    assert unsaid("Lister? Miss him? Like a hole in the head.") is None


def test_guessed_time():
    assert guards.guessed_time("It's 10:14.", ["What time is it?"])
    assert not guards.guessed_time("It's 10:14.", ["Here: 10:14 on Monday"])
    assert not guards.guessed_time("Ten minutes.", [])


def test_one_retry_policy():
    assert [retry_wait(n) for n in range(6)] == [60, 120, 240, 480, 900, 900]


def test_announcements_timers_first(tmp_path, make_holly):
    holly = make_holly(FakeLm())
    t = Timers(tmp_path / "t.json")
    t.items = [{"label": "pasta", "due": "2026-11-22T00:00:00", "kind": "timer"}]
    holly.research.shelf.add({"id": "r", "title": "Hens", "request": "hens", "stage": "done", "file": "r.md",
                              "finished": dt.datetime.now().isoformat(timespec="seconds"), "summary": "Coops.",
                              "announced": False})
    first = pending(t, holly.research)
    assert first.what == "due: pasta" and "The timer 'pasta'" in first.note and not t.items
    first.done(False)
    assert t.items and t.items[0]["tries"] == 1  # kept, to try again
    second = pending(t, holly.research)
    assert second.what == "research finished: Hens"
    second.done(True)
    assert pending(t, holly.research) is None and timers_due(None) is None


class Steps:
    """Research that takes a while each step."""

    def __init__(self, steps):
        self.left = steps
        self.started = 0

    def has_work(self):
        return self.left > 0

    async def step(self):
        self.started += 1
        await asyncio.sleep(0.05)
        self.left -= 1
        return "a step"


def background(cfg, research=None, side=None):
    convo = Conversation(cfg.persona)
    lm = FakeLm(side=side or (lambda s, u: "About the crew:\n- They talk a lot."))
    quick = lm.aask

    async def slowly(*args, **kwargs):  # notes take a moment to write, as on the real model
        await asyncio.sleep(0.05)
        return await quick(*args, **kwargs)

    lm.aask = slowly
    return Background(cfg, convo, lm, "m", None, research=research), convo


def test_research_while_asleep_stops_when_woken(cfg):
    async def go():
        research = Steps(3)
        bg, _ = background(cfg, research)
        bg.doze()
        await asyncio.sleep(0.07)  # one step done, the second under way
        bg.wake()
        await asyncio.sleep(0.01)  # its done-callback runs a tick later
        assert research.left == 2 and bg.researching is None
        bg.doze()
        while bg.researching:
            await asyncio.sleep(0.01)
        assert research.left == 0

    asyncio.run(go())


def test_notes_come_first_and_wake_drops_them(cfg):
    async def go():
        research = Steps(1)
        bg, convo = background(cfg, research)
        for i in range(30):
            convo.record("x" * 400, "y" * 400)
        convo.soft_chars, convo.keep_chars = 4000, 2000
        bg.doze()
        assert bg.notes and not bg.researching  # notes first; research waits
        await asyncio.sleep(0)
        assert bg.writing_notes
        bg.wake()  # someone speaks mid-notes: the history is left as it was
        await asyncio.sleep(0.01)
        assert convo.needs_compacting() and not convo.memory
        bg.doze()
        while bg.notes:
            await asyncio.sleep(0.01)
        assert convo.memory.startswith("About the crew") and not convo.needs_compacting()
        bg.doze()  # now research
        assert bg.researching

    asyncio.run(go())


def test_notes_wait_for_the_research_step_in_hand(cfg):
    async def go():
        research = Steps(5)
        bg, convo = background(cfg, research)
        bg.doze()
        await asyncio.sleep(0.01)  # a step under way
        assert bg.researching and research.started == 1
        for i in range(30):  # meanwhile the history passes its budget
            convo.record("x" * 400, "y" * 400)
        convo.soft_chars, convo.keep_chars = 4000, 2000
        bg.doze()
        assert bg.notes is None  # not alongside a research step: one request at a time
        while bg.researching:  # the research stops after its step, for the notes
            await asyncio.sleep(0.01)
        assert research.left == 4
        bg.doze()
        assert bg.notes and not bg.researching

    asyncio.run(go())


def test_failed_notes_wait_before_trying_again(cfg):
    def broken(system, user):
        raise ConnectionError("LM Studio restarting")

    async def go():
        bg, convo = background(cfg, side=broken)
        for i in range(30):
            convo.record("x" * 400, "y" * 400)
        convo.soft_chars, convo.keep_chars = 4000, 2000
        bg.doze()
        while bg.notes:
            await asyncio.sleep(0.01)
        assert not bg.notes_due()  # not again for a minute
        bg.doze()
        assert bg.notes is None

    asyncio.run(go())


def test_his_notes_check_his_model_is_loaded_his_way_first(cfg):
    """LM Studio's engine crashed, and it loaded his model again with its own defaults:
    longer prompts were cut to fit, with no error. Sleep-time work checks first."""
    seen = []  # how many checks had been made by each request for his notes
    bg, convo = background(cfg)
    bg.lm.side = lambda system, user: seen.append(bg.lm.checks) or "About the crew:\n- They talk a lot."
    for i in range(30):
        convo.record("x" * 400, "y" * 400)
    convo.soft_chars, convo.keep_chars = 4000, 2000
    assert asyncio.run(bg._sleep_work()) is True
    assert seen and set(seen) == {1} and bg.lm.checks == 1


def test_each_research_step_checks_his_model_is_loaded_his_way_first(cfg):
    """A crash can come in the middle of a job: each step checks, not just the first."""
    class Checked:
        def __init__(self, steps):
            self.left, self.seen = steps, []

        def has_work(self):
            return self.left > 0

        async def step(self):
            self.seen.append(bg.lm.checks)
            self.left -= 1
            return "a step"

    research = Checked(3)
    bg, _ = background(cfg, research)
    asyncio.run(bg._research())
    assert research.seen == [1, 2, 3]


@pytest.mark.parametrize("said", ["Additional. The ship's toaster has resigned.", "Additional: the ship's toaster has resigned.",
                                  "Additional, the ship's toaster has resigned.", "Additional... the ship's toaster has resigned.",
                                  "Additional — the ship's toaster has resigned."])
def test_additional_is_said_with_a_dash(said):
    from app.speech_text import split_sentences
    chunks = split_sentences(f"In sport, a draw. {said}")
    assert chunks[0] == "In sport, a draw." and chunks[1].startswith("Additional — ")  # one breath, a dash's pause


def test_a_search_answers_what_was_sought_and_what_was_said():
    """"You can get it now if you like", taking him up on a recipe search, was the search's
    whole question: "the results don't contain that phrase", 10 times in 10."""
    from app.search import SearchTool, asked

    assert asked("best fish pie recipe", "You can get it now if you like.") == (
        "best fish pie recipe (what the crew said: You can get it now if you like.)")

    class Searcher:
        def look_up(self, query, question):
            self.got = (query, question)
            return "Found."

    searcher = Searcher()
    tool = SearchTool(searcher)
    tool.look_up = lambda what, fn, turn: _run(fn)
    asyncio.run(tool.run({"query": "best fish pie recipe"}, Turn("You can get it now if you like.", 1, None)))
    assert searcher.got == ("best fish pie recipe", asked("best fish pie recipe", "You can get it now if you like."))


async def _run(fn):
    return fn()

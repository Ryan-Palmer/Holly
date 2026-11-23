"""Research: the tool's actions, and a job through every step, with a stand-in for
Gemma and for the web."""

import asyncio
import datetime as dt
import json
import threading

import pytest

from app import research as R
from app.search import Result
from conftest import FakeLm

REPORT = ("# Heat pumps for an old house\n\n## Summary\nThey move heat [1]. Old houses need insulation [2].\n\n"
          "## How they work\nThey move heat from outside air [1].")


def gemma(system: str, user: str) -> str:
    """Each research step's answer, told apart by its prompt."""
    if "plan the research" in system:
        return "1. What is a heat pump and how does it work?\n2. Do heat pumps suit old houses?"
    if "researching one question" in system:
        return "READ: 1, 2" if "(none yet)" in user and "READ:" in system else "DONE: enough"
    if "take notes" in system:
        return "- Heat pumps move heat from outside air into the house.\n- NOTHING\n- They suit low flow temperatures."
    if "important parts" in system:
        return "COMPLETE" if "grants" in user else "GAP: What grants are there for heat pumps in the UK?"
    if "Write a report" in system or "Check every statement" in system:
        return REPORT
    if "Sum up" in system:
        return "Heat pumps move heat from outside into the house. Old houses need insulating first."
    if "referring, in a few words" in system:
        return ""
    raise AssertionError(system[:80])


@pytest.fixture
def web(monkeypatch):
    """No network: made-up results and pages."""
    monkeypatch.setitem(R.PROVIDERS, "ddgs", lambda q, c: [
        Result(f"Page {i} about {q}", f"https://example.org/{abs(hash(q)) % 1000}/{i}", "snippet") for i in range(3)])
    monkeypatch.setattr(R, "wiki_search", lambda q, c: [{
        "title": "Heat pump (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Heat_pump", "snippet": "A heat pump",
        "wiki": "Heat pump"}])
    monkeypatch.setattr(R, "wiki_page", lambda t, c: "A heat pump is a device that moves heat. " * 20)
    monkeypatch.setattr(R, "fetch_page", lambda url, c, chars=6000: "Text about heat pumps and old houses. " * 20)


def research(cfg, lm=None, **limits) -> R.Research:
    rcfg = R.dataclasses.replace(cfg.research, questions=2, gaps=1, searches=2, reads=2, **limits)
    return R.Research(rcfg, cfg.search, lm or FakeLm(side=gemma), "m")


START = {"action": "start", "title": "Heat pumps for an old house",
         "request": "Would a heat pump work in our 1930s house?"}


async def finish(r: R.Research) -> list[str]:
    labels = []
    while r.has_work():
        labels.append(await r.step())
    return labels


def test_queue_list_and_cancel(cfg):
    r = research(cfg)
    assert r.handle({"action": "list"}) == "There's no research under way or finished."
    assert r.handle(START).startswith("Queued: research on 'Heat pumps for an old house'.")
    assert "already under way" in r.handle(START)
    assert r.handle({"action": "start", "title": "Llamas", "request": "How long do llamas live?"}).startswith(
        "Queued: research on 'Llamas', after 1 other job")
    assert r.handle({"action": "list"}) == "Research: 'Heat pumps for an old house': queued; 'Llamas': queued."
    assert r.handle({"action": "cancel", "title": "llamas"}) == "Cancelled: Llamas."
    assert r.handle({"action": "frobnicate"}).startswith("The research tool has no action")
    assert len(json.loads((cfg.research.dir / "jobs.json").read_text())) == 1


def test_ids_unique_within_a_second(cfg):
    r = research(cfg)
    now = dt.datetime(2026, 11, 23, 20, 0, 0)
    r.handle({"action": "start", "title": "A", "request": "a"}, now)
    r.handle({"action": "start", "title": "B", "request": "b"}, now)
    assert [j["id"] for j in r.jobs] == ["20261123-200000", "20261123-200000-2"]


def test_a_job_from_start_to_finish(cfg, web):
    r = research(cfg)
    r.handle(START)
    labels = asyncio.run(finish(r))
    assert labels[0].startswith("'Heat pumps for an old house': planned 2 questions")
    assert any("gaps: What grants" in label for label in labels)
    assert r.jobs == [] and json.loads((cfg.research.dir / "jobs.json").read_text()) == []
    report = r.shelf.find("")[0]
    md = (cfg.research.dir / report["file"]).read_text(encoding="utf-8")
    assert md.startswith("# Heat pumps") and "## Sources\n\n1. [" in md and "Researched by Holly" in md
    assert "NOTHING" not in md
    # Announced, then asked about.
    assert r.ready()["title"] == report["title"]
    assert "ask if they'd like a summary" in R.announcement(r.ready())
    r.announced(r.ready(), ok=True)
    assert r.ready() is None
    assert "Old houses need insulating" in r.handle({"action": "summary", "title": "the heat pump one"})
    read = r.handle({"action": "read", "title": "heat pumps"})
    assert "in full:" in read and "[1]" not in read and "## Sources" not in read
    assert r.handle({"action": "summary", "title": "llamas"}).startswith("There's no finished report on 'llamas'")
    assert "finished" in r.handle({"action": "list"})


def test_open(cfg, web, monkeypatch):
    opened = []
    monkeypatch.setattr(R.os, "startfile", opened.append, raising=False)
    r = research(cfg)
    r.handle(START)
    asyncio.run(finish(r))
    assert r.handle({"action": "open", "title": "Heat pumps for an old house"}).startswith("Opened the report")
    assert opened[0].name.endswith("heat-pumps-for-an-old-house.md")


def test_a_cancelled_step_leaves_the_job_as_it_was(cfg, web):
    slow = FakeLm(side=gemma)

    async def never(*args, **kwargs):
        await asyncio.sleep(10)

    slow.aask = never
    r = research(cfg, slow)
    r.handle(START)
    before = json.dumps({**r.jobs[0], "work_s": 0})

    async def cancel_a_step():
        t = asyncio.create_task(r.step())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(cancel_a_step())
    assert json.dumps({**r.jobs[0], "work_s": 0}) == before


def test_a_restart_carries_on(cfg, web):
    r = research(cfg)
    r.handle(START)

    async def part_way():
        for _ in range(4):
            await r.step()

    asyncio.run(part_way())
    stage = r.jobs[0]["stage"]
    again = research(cfg)
    assert again.jobs[0]["stage"] == stage
    asyncio.run(finish(again))
    assert again.shelf.find("heat pumps")


def test_nothing_found_is_given_up_and_announced(cfg, monkeypatch):
    monkeypatch.setitem(R.PROVIDERS, "ddgs", lambda q, c: [])
    monkeypatch.setattr(R, "wiki_search", lambda q, c: [])
    monkeypatch.setattr(R, "REFUSED_TRIES", 1)
    r = research(cfg)
    r.handle({"action": "start", "title": "Nothing", "request": "Something nobody has written about"})
    asyncio.run(finish(r))
    job = r.ready()
    assert job["stage"] == "failed" and "couldn't be finished" in R.announcement(job)


def test_failures_in_a_row_give_up(cfg, web):
    def broken(system, user):
        raise ConnectionError("LM Studio restarting")

    r = research(cfg, FakeLm(side=broken), max_errors=2)
    r.handle(START)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            asyncio.run(r.step())
    assert not r.has_work() and r.ready()["stage"] == "failed"


def test_shelving_failure_is_tried_again(cfg, web, monkeypatch):
    r = research(cfg)
    r.handle(START)

    async def until_keep():
        while r.jobs[0]["stage"] != "keep":
            await r.step()

    asyncio.run(until_keep())
    real = r.shelf.add
    monkeypatch.setattr(r.shelf, "add", lambda job, body="": (_ for _ in ()).throw(OSError("database is locked")))
    with pytest.raises(OSError):
        asyncio.run(r.step())
    assert r.jobs[0]["stage"] == "keep"  # still to keep, not lost as "done"
    monkeypatch.setattr(r.shelf, "add", real)
    asyncio.run(r.step())
    assert not r.jobs and r.ready()


def test_tool_and_steps_take_turns_with_the_queue(cfg):
    r = research(cfg)
    errors = []

    def start(i):
        try:
            r.handle({"action": "start", "title": f"Topic {i}", "request": f"About topic {i}"})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=start, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(r.jobs) == 20 and len({j["id"] for j in r.jobs}) == 20
    assert len(json.loads((cfg.research.dir / "jobs.json").read_text())) == 20


def test_clear(cfg, web):
    r = research(cfg)
    r.handle(START)
    asyncio.run(finish(r))
    r.handle({"action": "start", "title": "Llamas", "request": "llamas"})
    r.clear()
    assert not r.jobs and not list(cfg.research.dir.glob("*.md")) and r.handle({"action": "list"}).startswith(
        "There's no research")


def test_for_reading():
    assert R.for_reading("# T\n\n## Summary\nA [1, 2]. B [3].\n\n## Sources\n\n1. x") == "# T\n\n## Summary\nA. B."

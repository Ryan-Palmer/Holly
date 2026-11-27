"""Wikipedia first, in the search tool (app/search.py): a settled fact from Wikipedia's best
few articles; anything about now, or that changes, from the web. Wikipedia and the web are
stood in for."""

import dataclasses

import pytest

from app import search as S
from conftest import FakeLm

LIMA = [("Lima", "Lima is the capital and largest city of Peru."), ("Peru", "Peru is a country in South America.")]


def searcher(cfg, side, wiki=lambda q: LIMA, web=None) -> S.Searcher:
    s = S.Searcher(dataclasses.replace(cfg.search, wikipedia=True, provider="ddgs"), FakeLm(side=side), "m")
    s._wiki = wiki
    s._search = web or (lambda q: ([S.Result("Lima - web", "https://example.org", "Lima is Peru's capital.")], ""))
    return s


def judging(route="WIKIPEDIA", wiki="ANSWER: Lima is the capital of Peru.", web="ANSWER: Lima, says the web."):
    def side(system, user):
        return route if system == S.ROUTE_PROMPT else wiki if system.startswith("Today is") and "Wikipedia" in system \
            else web
    return side


def test_a_settled_fact_comes_from_wikipedia(cfg):
    s = searcher(cfg, judging(), web=lambda q: pytest.fail("searched the web"))
    found = s.look_up("capital of Peru", "What's the capital of Peru?")
    assert found.answer == "Lima is the capital of Peru." and found.queries == ["wikipedia: capital of Peru"]
    assert "These facts are right" in S.Searcher.for_holly(found)  # not hedged: it counts as found
    assert "Lima:\nLima is the capital" in s.lm.side_requests[-1]["user"]


def test_a_question_for_the_web_goes_to_the_web(cfg):
    for side in (judging(route="WEB"), judging(wiki="CHANGES"), judging(wiki="NO ANSWER")):
        found = searcher(cfg, side).look_up("capital of Peru", "What's the capital of Peru?")
        assert found.answer == "Lima, says the web." and found.queries[0] == "wikipedia: capital of Peru"


def test_about_now_never_asks_wikipedia(cfg):
    # Told so, the judge still answered "who won the Booker Prize this year?" from Wikipedia.
    s = searcher(cfg, judging(), wiki=lambda q: pytest.fail("asked Wikipedia"))
    for asked in ("Who won the Booker Prize this year?", "How much is a first class stamp now?",
                  "When does the tip open on Saturdays?"):
        found = s.look_up("q", asked)
        assert found.answer == "Lima, says the web." and found.queries == ["q"]
    assert not any(r["system"] == S.ROUTE_PROMPT for r in s.lm.side_requests)


def test_wikipedia_failing_is_the_web(cfg, capsys):
    def down(q):
        raise TimeoutError("slow")
    assert searcher(cfg, judging(), wiki=down).look_up("q", "Who wrote Middlemarch?").answer == "Lima, says the web."
    assert "Wikipedia: TimeoutError" in capsys.readouterr().out
    assert searcher(cfg, judging(), wiki=lambda q: []).look_up("q", "Who wrote Middlemarch?").answer \
        == "Lima, says the web."


def test_off_is_the_web_alone(cfg):
    s = searcher(cfg, judging(), wiki=lambda q: pytest.fail("asked Wikipedia"))
    s.cfg = dataclasses.replace(s.cfg, wikipedia=False)
    assert s.look_up("q", "Who wrote Middlemarch?").queries == ["q"]


def test_not_here_is_not_found(cfg):
    # With no way to say "these results don't have it" before the last round, it answered "The
    # provided search results do not contain information...", marked "these facts are right".
    s = searcher(cfg, judging(route="WEB", web="NOT HERE: nothing about the tip."))
    found = s.look_up("Keswick tip hours", "When does the tip open?")
    assert not found.results and "You don't know the answer" in found.answer
    assert "NOT HERE:" in s.lm.side_requests[-1]["system"]  # on offer before the last round

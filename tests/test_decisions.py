"""The side checks' registry (app/decisions.py): what needs no model."""

import pytest

from app import context, labelled
from app.decisions import Decision, all_decisions, numbered, starts, tool_named


class Lm:
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def ask(self, system, user, model=None, max_tokens=200, temperature=0.2, reasoning="none"):
        self.asked.append((system, user, max_tokens, temperature))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_every_check_is_registered_with_its_labelled_cases():
    decisions = all_decisions()
    assert set(decisions) == set(labelled.CASES) and all(labelled.CASES.values())
    assert {name for _, name, *_ in labelled.SIDE_CHECKS} <= set(decisions)
    # The names Holly asks them by, and the event log keeps their answers under.
    asked = {"taken_up", "offered_tool", "wants_report", "to_do", "job_tool", "about_the_past"}
    assert asked <= set(decisions)
    assert all(f'"{name}"' in open(context.__file__, encoding="utf-8").read() for name in asked)


def test_a_check_asks_once_and_reads_its_answer():
    d = Decision("test", "Is it a test? YES or NO.", lambda line: f"They say: {line}", "YES / NO", starts("YES"))
    lm = Lm(" yes, it is")
    assert d.ask(lm, "m", "Testing.") is True
    assert lm.asked == [("Is it a test? YES or NO.", "They say: Testing.", 4, 0)]
    assert d.ask(Lm("NO"), "m", "x") is False


def test_lm_studio_failing_is_the_fallback_quietly_and_raised_otherwise():
    d = Decision("test", "?", lambda x: x, "a tool's name", lambda text, x: text, fallback=None)
    assert d.ask(Lm(ConnectionError("gone")), "m", "x") is None
    with pytest.raises(ConnectionError):
        d.ask(Lm(ConnectionError("gone")), "m", "x", quiet=False)


def test_readers():
    assert tool_named(" Weather.", ("weather", "search")) == "weather"
    assert tool_named("NONE", ("weather",)) is None and tool_named("", ("weather",)) is None
    assert numbered("2.") == 2 and numbered("NONE") == 0 and numbered(" (3)") == 3
    assert starts("PAST")(" past, plainly") and not starts("PAST")("OTHER")

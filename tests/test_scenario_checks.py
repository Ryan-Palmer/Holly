"""The scenarios' own checks (app/scenarios.py), on made-up turns: no model. A labelled line
named OFFER once hid guards.OFFER there, and the in-character check crashed, found only an
hour into the evals."""

from app import guards, scenarios
from app.scenarios import Turn, in_character


def test_in_character_flags_offers_chirpiness_and_length():
    turns = [Turn("Thanks.", "Fifty-six. Anything else you need?", []),
             Turn("Hi.", "How lovely to hear from you!", []),
             Turn("Go on.", "word " * 31, []),
             Turn("Kettle's on.", "Thrilling.", [])]
    assert in_character(turns) == ["offered more help: 'Thanks.'", "chirpy ('How lovely'): 'Hi.'",
                                   "31 words: 'Go on.'"]


def test_what_scenarios_takes_from_guards_is_still_guards():
    for name in ("FORGOT", "NO_CLOCK", "OFFER", "without_offers"):
        assert getattr(scenarios, name) is getattr(guards, name), name

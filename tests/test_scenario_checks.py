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


def test_passes_on_what_the_search_found_by_name_not_the_crews_words():
    found = ("The provided search results do not identify a single \"best\" fish pie recipe, as multiple "
             "variations are listed. These include a recipe from BBC Food, a Luxe fish pie recipe, and "
             "Felicity Cloake's fish pie.")
    named = Turn("You can get it now if you like.", "I've found a few. There's one from BBC Food, a Luxe "
                 "version, and a Felicity Cloake one. Want the gist of one?", [("search", {})], told=[found])
    assert scenarios.passes_on(named)
    # A search answering their words has no names: theirs, echoed back, don't count.
    echoed = Turn("You can get it now if you like.", "You can, but there's nothing on that. Odd.",
                  [("search", {})], told=["The results don't contain the phrase 'You can get it now if you like'."])
    assert not scenarios.passes_on(echoed)
    # Nor do the names that start a sentence, his own name, or a reply passing on none of them.
    assert not scenarios.passes_on(Turn("Go on.", "These are grim. Holly here.", [],
                                        told=["These results mention Holly, from BBC Food."]))


def test_what_scenarios_takes_from_guards_is_still_guards():
    for name in ("FORGOT", "NO_CLOCK", "OFFER", "without_offers"):
        assert getattr(scenarios, name) is getattr(guards, name), name


def test_the_side_checks_keep_how_sure_each_answer_was():
    """Right isn't the whole story: a check answered at 55% is one a new model or a reworded
    prompt tips first. Each row keeps the least sure of its votes, and the words weighed."""
    edge = scenarios.MEMORY_LINES[0][0]  # the one answered near the edge

    class Odds:
        keep_odds, odds = False, None

        def ask(self, system, user, model=None, max_tokens=200, temperature=0.2, reasoning="none"):
            assert self.keep_odds  # asked for while the check runs
            near = f"They say: {edge}" in user
            self.odds = {"PAST": 0.55, "OTHER": 0.45} if near else {"OTHER": 0.97, "PAST": 0.03}
            return "PAST" if near else "OTHER"

    lm = Odds()
    rows = scenarios.memory_check(lm, None)
    assert not lm.keep_odds  # off again after
    assert (rows[0]["sure"], rows[0]["odds"]) == (0.55, {"PAST": 0.55, "OTHER": 0.45})
    assert {r["sure"] for r in rows[1:] if r["line"] != edge} == {0.97}
    side = scenarios.side_checks(lm, None)
    assert all(r["sure"] in (0.55, 0.97) for r in side)


def test_the_scorecard_shows_the_checks_answered_near_the_edge():
    from app import evals
    rows = [dict(line="a", right=True, sure=0.55, odds={"PAST": 0.55}), dict(line="b", right=True, sure=0.99),
            dict(line="c", right=False, sure=0.7), dict(line="d", right=True, sure=None)]
    c = evals.checks(rows)
    assert (c["right"], c["of"], c["least_sure"]) == (3, 4, 0.55)
    assert [r["line"] for r in c["unsure"]] == ["a", "c"]
    assert evals.sure_line(c) == "; 2 under 80% sure (least 55%)"
    assert evals.sure_line(evals.checks([dict(line="x", right=True)])) == ""  # no logprobs: nothing said

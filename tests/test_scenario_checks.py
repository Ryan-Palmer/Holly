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

"""The week of folds' checks (app/many_folds.py), on notes written by hand: what passes, and
what each check catches."""

import datetime as dt

from app import many_folds as F

START = dt.date(2026, 11, 21)  # a Saturday: Sam on the 28th, the dentist moved from the 26th to 1 December
BUDGET = 1125

GOOD = """About the crew:
The crew's sister is Jo, a vet.
Jo lives in Penrith, having moved from Keswick.
The crew has a ginger cat called Biscuit, and a kitten called Pickle.
The crew works at the library three days a week.
The crew dislikes mushrooms.

Lately:
Saturday 28 November 2026: Sam's visit is off, he's poorly.
Thursday 26 November 2026: The crew finished the lighthouse jigsaw.
Tuesday 1 December 2026: dentist at ten, moved from Thursday.
Sunday 29 November 2026: Jo is coming over for lunch."""


def test_notes_that_hold_up_pass():
    assert F.problems(GOOD, 6, START, BUDGET) == []


def test_each_check_catches_its_fault():
    def wrong(notes, day=6):
        return " | ".join(F.problems(notes, day, START, BUDGET))

    assert "lost: the kitten Pickle" in wrong(GOOD.replace(", and a kitten called Pickle", ""))
    assert "Jo still in Keswick" in wrong(GOOD.replace("Jo lives in Penrith, having moved from Keswick.",
                                                       "Jo is a vet in Keswick. Jo lives in Penrith."))
    assert "Sam's visit still on" in wrong(GOOD + "\nSaturday 28 November 2026: Sam is coming to stay.")
    assert "Sam's" not in wrong(GOOD + "\nThe dentist is at the same time.")  # "same" isn't Sam
    assert "the jigsaw still under way" in wrong(GOOD.replace("finished the lighthouse jigsaw",
                                                             "started a lighthouse jigsaw"))
    assert "the dentist still on the old day" in wrong(GOOD + "\nThursday 26 November 2026: dentist at ten.")
    assert "the boiler still broken" in wrong(GOOD + "\nFriday 27 November 2026: An engineer is coming to fix the boiler.")
    assert "not a note" in wrong(GOOD + "\nFriday 27 November 2026: The crew asked for the weather.")
    assert "not a note" in wrong(GOOD + "\nThe crew misses Lister.")
    assert "a name nobody said: Gerald" in wrong(GOOD + "\nThe crew's neighbour is Gerald.")
    assert "a date nobody gave" in wrong(GOOD + "\nMonday 12 January 2026: a party.")
    assert "over his budget" in wrong(GOOD + "\n" + "word " * BUDGET)


def test_checks_wait_for_the_day_a_change_is_said():
    early = GOOD.replace("Pickle", "nobody").replace("Penrith", "Keswick")
    assert F.problems(early.replace(", having moved from Keswick", ""), 1, START, BUDGET) == []

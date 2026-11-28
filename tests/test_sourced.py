"""Numbers he says after a tool must come from it (app/sourced.py)."""

from app import sourced

FORECAST = ("Tomorrow, Friday 27 November: rain, 8 to 12 degrees, 100 per cent chance of rain, wind up to 11, "
            "gusts 21. Saturday 28 November: rain, 8 to 13 degrees, 74 per cent chance of rain.")


def test_numbers_in_digits_and_words():
    assert sourced.numbers("Seventy-eight per cent, 3,000 years, 8.5 degrees") == {78, 3000, 8.5}
    assert sourced.from_words("a hundred and twenty") == [120]
    assert sourced.from_words("three thousand two hundred, then nineteen") == [3200, 19]
    assert {21, 15, 2115} <= sourced.numbers("It's 21:15.")


def test_times_and_prices_said_as_he_says_them():
    """"Twelve nineteen" was read as 31, and "one pound eighty" as 1 and 80, against the
    tool's £1.80: three false alarms on a real scorecard."""
    assert sourced.from_words("Twelve nineteen. Not that you're doing much.") == [12, 19]
    assert sourced.from_words("twenty five past") == [25]
    assert sourced.from_words("One pound eighty for a letter, three pounds thirty for a large one")[:2] == [1.8, 3.3]
    assert sourced.from_words("A standard letter's a pound eighty.") == [1.8]
    assert sourced.unsourced("Twelve nineteen.", ["Here: 12:19 on Thursday."]) == []
    assert sourced.unsourced("A pound eighty, or three pounds thirty.", ["First class: £1.80; large: £3.30."]) == []


def test_a_time_on_the_twelve_hour_clock_is_the_clocks():
    """Told 14:58, he said "2:58", flagged as 258 from nowhere (the evals, 28 November)."""
    told = ["UTC: 14:58 on Saturday 28 November 2026."]
    assert sourced.unsourced("2:58. Not that it matters.", told) == []
    assert sourced.unsourced("Two fifty-eight.", told) == []
    assert sourced.unsourced("12:24, if you must know.", ["It's 00:24 on Sunday."]) == []
    assert sourced.unsourced("3:58.", told) == [358]  # an hour out is still from nowhere


def test_a_number_from_the_tool_is_his_to_say():
    reply = "Rain. One hundred per cent chance of it. Eight to twelve degrees, gusts of twenty-one."
    assert sourced.unsourced(reply, [FORECAST]) == []


def test_a_number_from_nowhere_is_flagged():
    reply = "Rain, a hundred per cent of it, and twenty-six degrees, if you believe that."
    assert sourced.unsourced(reply, [FORECAST]) == [26]


def test_small_numbers_and_his_own_pass():
    assert sourced.unsourced("Two or three of them. With an IQ of six thousand, I'd know.", ["Nothing."]) == []


def test_only_replies_that_used_a_tool_are_checked():
    rows = [dict(part="conversations", said="Weather tomorrow?", reply="Twenty-six degrees. Sunburn.", told=[FORECAST]),
            dict(part="persona", said="How old are you?", reply="Three million and forty.", told=[]),
            dict(part="scenarios", said="Rain?", reply="A hundred per cent.", told=[FORECAST])]
    result = sourced.check(rows)
    assert result["checked"] == 2
    assert [(f["part"], f["numbers"]) for f in result["flagged"]] == [("conversations", [26])]

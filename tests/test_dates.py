"""Day words and times: worked out in code, since Gemma's own sums with dates were off."""

import datetime as dt

import pytest

from app.archive import days_in
from app.persona import date_days, day, resolve_day, when
from app.timers import parse_duration, parse_when, say_when

WED = dt.date(2026, 11, 18)  # a Wednesday
NOW = dt.datetime(2026, 11, 18, 10, 0)  # Wednesday morning


def test_day_and_when():
    assert day(dt.date(2026, 11, 1)) == "Sunday 1 November 2026"
    assert when(dt.datetime(2026, 11, 21, 21, 40)) == "Saturday 21 November 2026 at 9:40 pm"


@pytest.mark.parametrize("words, past, want", [
    ("today", False, WED),
    ("tonight", False, WED),
    ("yesterday", False, dt.date(2026, 11, 17)),
    ("tomorrow", False, dt.date(2026, 11, 19)),
    ("the day after tomorrow", False, dt.date(2026, 11, 20)),
    ("next Tuesday", False, dt.date(2026, 11, 24)),
    ("Wednesday", False, dt.date(2026, 11, 25)),  # said on a Wednesday: the next one
    ("on Monday", True, dt.date(2026, 11, 16)),  # about the past: the last one
    ("last Friday", False, dt.date(2026, 11, 13)),
    ("this Friday", True, dt.date(2026, 11, 20)),  # "this" is ahead, whatever the sentence
])
def test_resolve_day(words, past, want):
    assert resolve_day(words, WED, past) == want


def test_date_days_marks_each_day_word():
    assert date_days("Jo's coming over tomorrow.", dt.date(2026, 11, 21)) == \
        "Jo's coming over tomorrow [Sunday 22 November 2026]."
    # A past sentence takes the last Monday; a future word wins over a past one.
    assert "[Monday 16 November 2026]" in date_days("I went to the dentist on Monday.", WED)
    assert "[Monday 23 November 2026]" in date_days("I said I'll ring on Monday.", WED)


@pytest.mark.parametrize("text, seconds", [
    ("20 minutes", 1200),
    ("1 hour 30 minutes", 5400),
    ("1.5 hours", 5400),
    ("half an hour", 1800),
    ("an hour and a half", 5400),
    ("90 seconds", 90),
    ("1h30", 5400),
    ("15", 900),  # a bare number: minutes
])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == dt.timedelta(seconds=seconds)


@pytest.mark.parametrize("text", ["soon", "", "a while"])
def test_parse_duration_unreadable(text):
    assert parse_duration(text) is None


@pytest.mark.parametrize("at, day_text, want", [
    ("19:30", "", dt.datetime(2026, 11, 18, 19, 30)),
    ("7:30 pm", "", dt.datetime(2026, 11, 18, 19, 30)),
    ("9", "", dt.datetime(2026, 11, 19, 9, 0)),  # gone today: the next 9 o'clock
    ("noon", "", dt.datetime(2026, 11, 18, 12, 0)),
    ("07:00", "tomorrow", dt.datetime(2026, 11, 19, 7, 0)),
    ("12:00", "Friday", dt.datetime(2026, 11, 20, 12, 0)),
    ("10:30", "Wednesday", dt.datetime(2026, 11, 18, 10, 30)),  # today, still to come
    ("09:00", "Wednesday", dt.datetime(2026, 11, 25, 9, 0)),  # today's has gone: next week's
    ("12:00", "25 December", dt.datetime(2026, 12, 25, 12, 0)),
    ("12:00", "1 January", dt.datetime(2027, 1, 1, 12, 0)),
])
def test_parse_when(at, day_text, want):
    assert parse_when(at, day_text, NOW) == want


@pytest.mark.parametrize("at, day_text", [("25:00", ""), ("9", "today"), ("teatime", ""), ("12:00", "someday")])
def test_parse_when_refuses(at, day_text):
    assert parse_when(at, day_text, NOW) is None


def test_say_when():
    assert say_when(NOW + dt.timedelta(minutes=20), NOW) == "10:20 today, in 20 minutes"
    assert say_when(NOW + dt.timedelta(minutes=90), NOW) == "11:30 today, in 1 hour 30 minutes"
    assert say_when(dt.datetime(2026, 11, 19, 7, 0), NOW) == "07:00 tomorrow"
    assert say_when(dt.datetime(2026, 11, 20, 19, 30), NOW) == "19:30 on Friday 20 November"


def test_days_in_looks_back():
    assert days_in("what did we talk about yesterday?", WED) == {dt.date(2026, 11, 17)}
    assert days_in("what did I say on Tuesday?", WED) == {dt.date(2026, 11, 17)}
    assert days_in("on 21 November", WED) == {dt.date(2025, 11, 21)}  # still to come this year
    assert days_in("on 2 November", WED) == {dt.date(2026, 11, 2)}
    assert days_in("what's on tomorrow?", WED) == set()

"""Timers and reminders: set, listed, cancelled, due, and kept when an announcement fails."""

import datetime as dt

from app import timers
from app.timers import Timers

NOW = dt.datetime(2026, 11, 22, 12, 0)


def test_set_list_cancel(tmp_path):
    t = Timers(tmp_path / "timers.json")
    assert t.handle({"action": "list"}, NOW) == "Nothing is set."
    assert t.handle({"action": "set", "label": "pasta", "duration": "20 minutes"}, NOW).startswith(
        "Set: pasta, due 12:20 today, in 20 minutes.")
    assert t.handle({"action": "set", "label": "ring Mum", "at": "19:30", "day": "tomorrow"}, NOW).startswith(
        "Set: ring Mum, due 19:30 tomorrow")
    assert [i["kind"] for i in t.items] == ["timer", "reminder"]
    listed = t.handle({"action": "list"}, NOW)
    assert listed.index("pasta") < listed.index("ring Mum")
    # Kept on disk: a fresh Timers sees them.
    assert len(Timers(tmp_path / "timers.json").items) == 2
    assert t.handle({"action": "cancel", "label": "the pasta timer"}, NOW) == "Cancelled: pasta."
    assert t.handle({"action": "cancel", "label": "steak"}, NOW).startswith("There's nothing set called 'steak'")
    assert t.handle({"action": "cancel", "label": "all"}, NOW) == "Cancelled: ring Mum."


def test_cancel_matches_whole_words(tmp_path):
    t = Timers(tmp_path / "timers.json")
    t.handle({"action": "set", "label": "tea", "duration": "5 minutes"}, NOW)
    assert t.handle({"action": "cancel", "label": "steak"}, NOW).startswith("There's nothing")
    # No words to go on: the only one set.
    assert t.handle({"action": "cancel", "label": "the timer"}, NOW) == "Cancelled: tea."


def test_bad_requests_say_how_to_fix_them(tmp_path):
    t = Timers(tmp_path / "timers.json")
    assert "needs a 'duration'" in t.handle({"action": "set", "label": "x"}, NOW)
    assert t.handle({"action": "set", "label": "x", "at": "09:00", "day": "today"}, NOW).startswith("Nothing set")
    assert not t.items


def test_due_and_put_back(tmp_path):
    t = Timers(tmp_path / "t.json")
    t.handle({"action": "set", "label": "ring Mum", "at": "12:05"}, NOW)
    assert not t.due(NOW)
    due = t.due(NOW + dt.timedelta(minutes=5))
    assert [i["label"] for i in due] == ["ring Mum"] and not t.items
    # The announcement failed: kept, on disk too, and not due again at once.
    t.put_back(due, NOW + dt.timedelta(minutes=5))
    assert len(t.items) == 1 and Timers(tmp_path / "t.json").items
    assert not t.due(NOW + dt.timedelta(minutes=5, seconds=30))
    again = t.due(NOW + dt.timedelta(minutes=6, seconds=1))
    assert [i["label"] for i in again] == ["ring Mum"]
    said = timers.announcement(again, NOW + dt.timedelta(minutes=12))
    assert "came due at 12:05" in said and "couldn't say so then" in said
    # Twice the wait the second time.
    t.put_back(again, NOW + dt.timedelta(minutes=6))
    assert t.items[0]["retry"] == "2026-11-22T12:08:00"
    assert "hasn't been announced" in t.handle({"action": "list"}, NOW + dt.timedelta(minutes=7))


def test_announcement():
    item = {"label": "pasta", "due": NOW.isoformat(), "kind": "timer"}
    assert "it's due now" in timers.announcement([item], NOW + dt.timedelta(minutes=1))
    late = timers.announcement([item], NOW + dt.timedelta(days=1))
    assert "while you were switched off" in late and "on Sunday 22 November 2026" in late
    assert "call the tool now" in late  # a timer to play music plays it


def test_unreadable_file_starts_empty(tmp_path):
    (tmp_path / "t.json").write_text("{not json", encoding="utf-8")
    assert Timers(tmp_path / "t.json").items == []

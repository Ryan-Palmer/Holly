"""The reports' index: keeping finished reports, announcing them, and finding one from
the crew's words (by keyword here; by meaning, with stand-in vectors)."""

import datetime as dt
import re
import zlib

import numpy as np

from app import reports as RP
from app.reports import Reports

NOW = dt.datetime(2026, 11, 23, 20, 0)

SHELF = [  # (title, request, summary, days old)
    ("Beginner running plans", "How should a complete beginner start running?",
     "A nine-week plan, three runs a week, alternating running and walking.", 30),
    ("Budget espresso machines", "What's a good espresso machine for under 300 pounds?",
     "Entry-level machines with a proper portafilter make good espresso.", 10),
    ("Keeping chickens in the garden", "Could we keep a few hens in the back garden?",
     "Three or four hens need a secure coop and protection from foxes.", 5),
    ("Board games for ten-year-olds", "What are the best board games for a ten-year-old?",
     "Ticket to Ride and Carcassonne lead; Forbidden Island is the best cooperative game.", 2),
    ("Christmas presents for a ten-year-old", "What are good Christmas presents for a ten-year-old?",
     "Lego, walkie-talkies and craft kits are popular this year.", 1),
]


def job(i: int, title: str, request: str, summary: str, days: float, **extra) -> dict:
    when = (NOW - dt.timedelta(days=days)).isoformat(timespec="seconds")
    return {"id": f"r{i}", "title": title, "request": request, "asked": when, "finished": when, "stage": "done",
            "file": f"r{i}.md", "summary": summary, "announced": True, **extra}


def shelf(tmp_path, lm=None, expand=None) -> Reports:
    s = Reports(tmp_path / "reports.db", lm, "embedder" if lm else "", expand)
    for i, row in enumerate(SHELF):
        s.add(job(i, *row), f"{row[2]} More on {row[0].lower()}.")
    return s


def titles(picks: list[dict]) -> list[str]:
    return [p["title"] for p in picks]


def test_telling_words():
    assert RP.telling("the report on heat pumps") == {"heat", "pumps"}
    assert RP.stems("Bikes and glass") == {"bike", "and", "glass"}


def test_find_exact_latest_and_by_words(tmp_path):
    s = shelf(tmp_path)
    assert titles(s.find("Budget espresso machines", now=NOW)) == ["Budget espresso machines"]
    latest = s.find("the report", now=NOW)
    assert titles(latest) == ["Christmas presents for a ten-year-old"] and latest[0]["sure"]
    picks = s.find("the espresso one", now=NOW)
    assert titles(picks) == ["Budget espresso machines"] and picks[0]["sure"]
    assert s.find("llamas in Peru", now=NOW) == []


def test_unsure_gives_a_shortlist(tmp_path):
    picks = shelf(tmp_path).find("the ten-year-old one", now=NOW)
    assert set(titles(picks)) == {"Board games for ten-year-olds", "Christmas presents for a ten-year-old"}
    assert not any(p.get("sure") for p in picks)


def test_gemmas_words_count_for_less(tmp_path):
    # "the hens one" alone is sure; Gemma's words for "the fitness one" find running, unsure alone.
    s = shelf(tmp_path, expand=lambda words: "running, jogging, exercise, fitness")
    picks = s.find("the fitness one", now=NOW)
    assert titles(picks)[0] == "Beginner running plans"
    assert not picks[0].get("sure")  # none of the crew's own words in it


class BagOfWords:
    """Stand-in embeddings: a word's hash picks its dimension, so shared words mean closeness."""

    def embed(self, texts, model):
        out = []
        for t in texts:
            v = np.zeros(256, np.float32)
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                v[zlib.crc32(w[:5].encode()) % 256] += 1
            out.append(v + 0.01)
        return out


def test_by_meaning(tmp_path):
    s = shelf(tmp_path, BagOfWords())
    assert s.embed_missing() == len(SHELF)
    assert s.embed_missing() == 0  # done already
    close = s._by_meaning("coop foxes hens garden")
    best = max(close, key=close.get)
    assert s._rows("n = ?", (best,))[0]["title"] == "Keeping chickens in the garden"


def test_meaning_failing_leaves_keywords(tmp_path, capsys):
    class Broken:
        def embed(self, texts, model):
            raise ConnectionError("gone")

    s = shelf(tmp_path, Broken())
    assert titles(s.find("the espresso one", now=NOW)) == ["Budget espresso machines"]
    assert "off for 60 s" in capsys.readouterr().out


def test_add_again_keeps_number_and_announcement(tmp_path):
    s = Reports(tmp_path / "reports.db")
    first = s.add(job(9, "Fixing a dripping tap", "How do I fix a tap?", "Replace the washer.", 0, announced=False))
    s.announced(first, ok=True)
    again = s.add(job(9, "Fixing a dripping tap", "How do I fix a tap?", "Replace the washer or cartridge.", 0,
                      announced=False))
    assert again["n"] == first["n"] and again["announced"] == 1
    assert s.ready(NOW) is None
    assert len(s._rows()) == 1 and "cartridge" in s._rows()[0]["summary"]


def test_ready_announced_and_retry(tmp_path):
    s = Reports(tmp_path / "reports.db")
    row = s.add(job(1, "Puppy training", "How do we train a puppy?", "Short sessions.", 0, announced=False))
    assert s.ready(NOW)["title"] == "Puppy training"
    s.announced(row, ok=False, retry=NOW + dt.timedelta(minutes=1))
    assert s.ready(NOW) is None and s.ready(NOW + dt.timedelta(minutes=2))
    assert Reports(tmp_path / "reports.db").ready(NOW + dt.timedelta(minutes=2))  # after a restart too
    s.announced(row, ok=True)
    assert s.ready(NOW + dt.timedelta(hours=1)) is None
    assert Reports(tmp_path / "reports.db").ready(NOW + dt.timedelta(hours=1)) is None


def test_recent_and_clear(tmp_path):
    s = shelf(tmp_path)
    recent, older = s.recent(days=7, now=NOW)
    assert titles(recent) == ["Christmas presents for a ten-year-old", "Board games for ten-year-olds",
                              "Keeping chickens in the garden"] and older == 2
    assert s.titles(2) == ["Board games for ten-year-olds", "Christmas presents for a ten-year-old"]
    assert sorted(s.clear()) == [f"r{i}.md" for i in range(len(SHELF))]
    assert s.find("the espresso one", now=NOW) == []

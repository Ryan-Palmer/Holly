"""Recall: finding what was said (by keyword and day; meaning needs LM Studio), and who
a question is about."""

import datetime as dt

from app.archive import Archive, named_in
from conftest import FakeLm

HITS = [(None, "Crew: Jo came round with June and Ed.\nHolly: Noted.")]


def test_named_in():
    assert named_in("Did I meet Jo yesterday?", HITS) == ["Jo"]
    assert named_in("What did June say?", HITS) == ["June"]
    assert named_in("What happened on 22 June?", HITS) == []  # a date, not a name
    assert named_in("What did Ed say on Monday?", HITS) == ["Ed"]  # two letters count
    assert named_in("Who is Bob?", HITS) == []  # not in what was found
    # A sentence's first word isn't a name, unless it's recall's query (words, not a sentence).
    assert named_in("Jo is coming", HITS) == []
    assert named_in("Jo visit", HITS, keywords=True) == ["Jo"]


def archive(cfg, lm=None) -> Archive:
    return Archive(cfg.recall, lm or FakeLm(), "m")


def entry(at: dt.datetime, crew: str, holly: str = "Noted.") -> tuple[str, str]:
    return at.isoformat(), f"Crew: {crew}\nHolly: {holly}"


def test_search_by_keyword(cfg):
    a = archive(cfg)
    a.add([entry(dt.datetime(2026, 11, 1, 9), "June came round for tea."),
           entry(dt.datetime(2026, 11, 2, 9), "We made lasagna."),
           entry(dt.datetime(2026, 11, 3, 9), "It may rain.")])
    texts = lambda hits: " ".join(t for _, t in hits)
    assert "June came round" in texts(a.search("June", on=dt.date(2026, 11, 20)))  # a name, not a month
    assert "lasagna" in texts(a.search("lasagne", on=dt.date(2026, 11, 20)))  # near enough
    assert a.search("may", on=dt.date(2026, 11, 20)) == []  # the verb, not a name


def test_a_day_asked_about_is_the_answer(cfg):
    a = archive(cfg)
    yesterday = dt.datetime(2026, 11, 17, 8)
    # A busy day: the one asked about comes late, past the day's first DAY_ENTRIES.
    a.add([entry(yesterday + dt.timedelta(minutes=i), f"Chatter number {i}.") for i in range(30)])
    a.add([entry(yesterday + dt.timedelta(hours=11), "Jo came over to make lasagne.")])
    a.add([entry(dt.datetime(2026, 11, 12, 9), "We made lasagne once before.")])
    hits = a.search("lasagne", on=dt.date(2026, 11, 18), question="Who came over to make lasagne yesterday?")
    texts = [t for _, t in hits]
    assert any("Jo came over" in t for t in texts)
    assert not any("once before" in t for t in texts)  # other days aren't mixed in
    assert len(hits) == 20


def test_recall_asks_again_about_a_name(cfg):
    asked = []

    def judge(system, user):
        asked.append(user)
        return "NOTHING" if len(asked) == 1 else "ANSWER: Jo is the crew's sister, a vet."

    a = archive(cfg, FakeLm(side=judge))
    a.add([entry(dt.datetime(2026, 11, 1, 9), "My sister Jo is a vet.")])
    found = a.recall("Jo", "Did I meet Jo yesterday?")
    assert found.answer == "Jo is the crew's sister, a vet."
    assert "What do these entries say about Jo?" in asked[1]


def test_recall_nothing(cfg):
    a = archive(cfg, FakeLm(side=lambda s, u: "NOTHING"))
    assert a.recall("Kryten", "Who is Kryten?").results == 0
    a.add([entry(dt.datetime(2026, 11, 1, 9), "Kryten is a mechanoid.")])
    found = a.recall("Kryten", "When did I last see Kryten?")
    assert found.results == 0 and "doesn't say" in found.answer


def test_misheard_names(cfg):
    a = archive(cfg, FakeLm(side=lambda s, u: "ANSWER: A mechanoid."))
    a.add([entry(dt.datetime(2026, 11, 1, 9), "Kryten is a mechanoid.")])
    assert a.recall("Crichton", "Who is Crichton?").answer == "A mechanoid."


def test_clear(cfg):
    a = archive(cfg)
    a.add([entry(dt.datetime(2026, 11, 1, 9), "Hello.")])
    a.clear()
    assert a.count() == 0

"""Recall: finding what was said (by keyword and day; meaning needs LM Studio), and who
a question is about."""

import datetime as dt

from app.archive import Archive, added_names, in_their_words, named_in
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


def test_a_broad_day_question_sees_the_whole_day(cfg):
    a = archive(cfg)
    yesterday = dt.datetime(2026, 11, 17, 8)
    a.add([entry(yesterday + dt.timedelta(minutes=15 * i), f"Chatter number {i}.") for i in range(60)])  # to 22:45
    hits = a.search("yesterday", on=dt.date(2026, 11, 18), question="What did we do yesterday?")
    hours = sorted(int(at[11:13]) for at, _ in hits)
    assert len(hits) == 20 and hours[0] == 8 and hours[-1] >= 21  # not just its first hours


def test_the_judge_is_asked_about_the_date_not_yesterday(cfg):
    asked = []
    a = archive(cfg, FakeLm(side=lambda s, u: asked.append(u) or "ANSWER: Lasagne."))
    yesterday = dt.datetime.combine(dt.date.today() - dt.timedelta(days=1), dt.time(18))
    a.add([entry(yesterday, "We made lasagne yesterday [an earlier day], and tonight it's soup.")])
    a.recall("yesterday", "What did we do yesterday?")
    from app.persona import day
    assert f"Question: What did we do on {day(yesterday.date())}?" in asked[0]


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


def test_last_night_is_looked_for_in_the_evening(cfg):
    """A day of chatter about food, and tea at 9:39 pm: "last night" is the evening's."""
    a = archive(cfg)
    day_ = dt.datetime(2026, 11, 17, 8)
    a.add([entry(day_ + dt.timedelta(minutes=20 * i), f"Food chatter number {i}.") for i in range(27)])  # to 16:40
    a.add([entry(dt.datetime(2026, 11, 17, 21, 39), "I'm having fish pie for tea.")])
    texts = [t for _, t in a.search("what I had for food last night", on=dt.date(2026, 11, 18),
                                    question="What did I have for food last night?")]
    assert texts == ["Crew: I'm having fish pie for tea.\nHolly: Noted."]
    # Nothing said that night: the day it was, instead.
    a2 = archive(cfg)
    a2.add([entry(dt.datetime(2026, 11, 17, 12), "Soup for lunch.")])
    assert len(a2.search("last night", on=dt.date(2026, 11, 18), question="What did I eat last night?")) == 1


def test_the_crews_own_words_when_his_question_names_them(cfg):
    """His notes know the crew by name, and he put it in his question; his memory only
    ever says "Crew". After a miss, the crew's own words are tried, "I" as the crew."""
    asked = []

    def judge(system, user):
        asked.append(user)
        return "ANSWER: Fish pie." if "the crew had for food" in user else "NOTHING"

    a = archive(cfg, FakeLm(side=judge))
    a.add([entry(dt.datetime.combine(dt.date.today() - dt.timedelta(days=1), dt.time(21, 39)),
                 "I'm having fish pie for tea.")])
    found = a.recall("what Sam had for food last night", "What did Sam have for food last night?",
                     asked="Hey Holly, do you remember what I had for food last night?")
    assert found.answer == "Fish pie."
    assert "Question: do you remember what the crew had for food on the night of" in asked[1]
    # A name the crew said themselves isn't theirs: no second try, and no answer.
    asked.clear()
    found = a.recall("Jo tea last night", "What did Jo have for tea last night?",
                     asked="Do you remember what Jo had for tea last night?")
    assert found.results == 0 and len(asked) == 1


def test_in_their_words_and_added_names():
    assert in_their_words("Hey Holly, do you remember what I had for food last night?") == \
        "do you remember what the crew had for food last night?"
    assert in_their_words("Holly, did I tell you about my mum? I'm sure I did.") == \
        "did the crew tell you about the crew's mum? the crew is sure the crew did."
    assert added_names("what Sam had for tea What did Sam have for tea on Tuesday?", "what I had for tea") == ["Sam"]
    assert added_names("Jo tea What did Jo have?", "What did jo have?") == []
    assert added_names("When did Holly meet Kryten in November?", "when did you meet kryten") == []


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

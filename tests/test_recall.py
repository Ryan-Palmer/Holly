"""Recall: finding what was said (by keyword and day; meaning needs LM Studio), and who
a question is about."""

import asyncio
import dataclasses
import datetime as dt
from types import SimpleNamespace

from app.archive import QUERY_PREFIX, Archive, RecallTool, added_names, in_their_words, named_in
from app.search import Lookup
from app.tools import Turn
from conftest import FakeLm
from test_reports import BagOfWords

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
    assert added_names("what Sam had for tea", "What did Sam have for tea on Tuesday?", "what I had for tea") == ["Sam"]
    assert added_names("Jo tea", "What did Jo have?", "What did jo have?") == []
    assert added_names("Kryten", "When did Holly meet Kryten in November?", "when did you meet kryten") == []
    # Not a sentence's first word, nor the ship's records' words: no second try for those.
    assert added_names("cat race", "Lasagne aside. When did the crew of Red Dwarf meet the cat race?",
                       "do you remember the cat race?", known={"red", "dwarf"}) == []
    # "Holly's" isn't him being hailed; "the mine" isn't the crew's.
    assert in_their_words("Holly's memory's shot, what did I eat?") == "Holly's memory's shot, what did the crew eat?"
    assert in_their_words("What did I find in the mine?") == "What did the crew find in the mine?"


def test_a_name_for_someone_they_didnt_name_is_still_asked_about(cfg):
    """"Did I meet her yesterday?", put as "Did the crew meet Jo yesterday?": Jo isn't in
    the crew's words, but "her" is, so Jo isn't the crew, and is asked about plainly."""
    asked = []

    def judge(system, user):
        asked.append(user)
        return "ANSWER: Jo came round." if "say about Jo" in user else "NOTHING"

    a = archive(cfg, FakeLm(side=judge))
    a.add([entry(dt.datetime(2026, 11, 1, 9), "My sister Jo is a vet.")])
    found = a.recall("Jo", "Did the crew meet Jo yesterday?", asked="Did I meet her yesterday?")
    assert found.answer == "Jo came round." and "What do these entries say about Jo?" in asked[-1]


def test_tonight_said_earlier_that_day_is_last_night(cfg):
    """"I'm having fish pie tonight", said at 4:30, and the evening's own talk: asked about
    last night, the evening's talk alone missed it."""
    a = archive(cfg)
    a.add([entry(dt.datetime(2026, 11, 17, 16, 30), "I'm having fish pie tonight.")])
    a.add([entry(dt.datetime(2026, 11, 17, 21, 0), "Night, Holly.")])
    a.add([entry(dt.datetime(2026, 11, 17, 11, 0), "Nice morning for it.")])
    texts = " ".join(t for _, t in a.search("what I had for tea last night", on=dt.date(2026, 11, 18),
                                            question="What did I have for tea last night?"))
    assert "fish pie tonight" in texts and "Night, Holly" in texts and "morning" not in texts


def test_crew_is_a_word_worth_finding_in_the_records(cfg):
    """Every conversation entry says "Crew:", so it's no keyword there; in the ship's
    records ("Starbug's crew"), and in the reports' words, it is."""
    from app import archive as archive_module
    assert "crew" in archive(cfg).stopwords
    assert "crew" not in Archive(cfg.recall, FakeLm(), "m", cfg.recall.records_file, talk=False).stopwords
    assert "crew" not in archive_module.STOPWORDS


def test_recall_nothing(cfg):
    a = archive(cfg, FakeLm(side=lambda s, u: "NOTHING"))
    assert a.recall("Kryten", "Who is Kryten?").results == 0
    a.add([entry(dt.datetime(2026, 11, 1, 9), "Kryten is a mechanoid.")])
    found = a.recall("Kryten", "When did I last see Kryten?")
    assert found.results == 0 and "doesn't say" in found.answer


SAID = "Crew: Jo and her mum June are coming tomorrow."


def test_recall_reads_what_the_crew_said_in_this_conversation(cfg):
    """Asked who was coming over, he recalled, and the archive had nothing: the visit was
    still in his history. The judge reads the crew's lines there too."""
    asked = []
    a = archive(cfg, FakeLm(side=lambda s, u: asked.append(u) or "ANSWER: Jo and her mum June."))
    found = a.recall("visit coming over", "Who is coming over tomorrow?", said=SAID)
    assert f"(Said in this conversation, by the crew:)\n{SAID}" in asked[0]
    assert found.answer == "Jo and her mum June." and found.results == 1


def test_nothing_remembered_tells_him_to_answer_from_the_conversation_or_his_notes(cfg):
    """Told "say you don't remember", he did, with the answer in front of him."""
    a = archive(cfg, FakeLm(side=lambda s, u: "NOTHING"))
    found = a.recall("visit coming over", "Who is coming over tomorrow?", said=SAID)
    assert found.results == 0
    assert found.answer.startswith("Your archive doesn't say. If this conversation or your notes say, "
                                   "answer from them.")
    # Nothing found at all, and no judge asked: the same.
    assert "If this conversation or your notes say, answer from them." in a.recall(
        "boiler", "When's the boiler service?").answer


class Remembers:
    """An archive that records what recall was asked."""

    def __init__(self):
        self.calls = []

    def recall(self, query, question, said="", asked="", notes=""):
        self.calls.append((query, question, said, asked, notes))
        return Lookup(query, "Jo and her mum June are coming tomorrow.", 1, 1.0, 1.0, [query])


def test_the_recall_tool_passes_the_crews_lines_and_their_own_words():
    memory = Remembers()
    turn = Turn("Holly, who's coming over tomorrow?", 3, SimpleNamespace(search_ms=0.0), crew_lines=lambda: SAID,
                notes=lambda: "About the crew:\nJo is the crew's sister.")
    told = asyncio.run(RecallTool(memory).run(
        {"query": "visit tomorrow", "question": "Who is coming over tomorrow?"}, turn))
    assert memory.calls == [("visit tomorrow", "Who is coming over tomorrow?", SAID,
                             "Holly, who's coming over tomorrow?", "About the crew:\nJo is the crew's sister.")]
    assert told.startswith("Jo and her mum June are coming tomorrow.\n(This is your memory of it")


def test_the_episode_is_left_out_of_the_judges_question(cfg):
    """"The Cat's name in the episode Gunmen of the Apocalypse?" got NOTHING from the
    judge, taking it as the Cat's own name; without "the episode", the right one."""
    asked = []
    a = archive(cfg, FakeLm(side=lambda s, u: asked.append(u) or "ANSWER: The Riviera Kid."))
    a.add([entry(dt.datetime(2026, 11, 1, 9), "We watched Gunmen of the Apocalypse again.")])
    a.recall("Cat name Gunmen of the Apocalypse", "What was the Cat's name in the episode Gunmen of the Apocalypse?")
    assert "Question: What was the Cat's name in Gunmen of the Apocalypse?\n" in asked[0]


def test_a_days_search_by_meaning_leaves_the_day_words_out(cfg):
    """Searching a day by meaning, "yesterday" put the entries that said "yesterday" first,
    ahead of the album asked for: the day is settled, so its words are left out."""
    class Embedder(FakeLm):
        def __init__(self):
            super().__init__()
            self.texts = []

        def embed(self, texts, model):
            self.texts += texts
            return BagOfWords().embed(texts, model)

    lm = Embedder()
    a = Archive(dataclasses.replace(cfg.recall, embed_model="embedder"), lm, "m")
    yesterday = dt.datetime(2026, 11, 17, 9)
    a.add([entry(yesterday, "We had a lovely day yesterday at the lake."),
           entry(yesterday + dt.timedelta(hours=2), "Play that new album by the jazz band."),
           entry(yesterday + dt.timedelta(hours=4), "Jo's making lasagne for tea.")])
    hits = a.search("what music did I ask for yesterday?", on=dt.date(2026, 11, 18))
    assert len(hits) == 3  # the day asked about, all of it
    searched = [t for t in lm.texts if t.startswith(QUERY_PREFIX)]
    assert searched and all("music" in t and "yesterday" not in t.lower() for t in searched)


def test_misheard_names(cfg):
    a = archive(cfg, FakeLm(side=lambda s, u: "ANSWER: A mechanoid."))
    a.add([entry(dt.datetime(2026, 11, 1, 9), "Kryten is a mechanoid.")])
    assert a.recall("Crichton", "Who is Crichton?").answer == "A mechanoid."


def test_clear(cfg):
    a = archive(cfg)
    a.add([entry(dt.datetime(2026, 11, 1, 9), "Hello.")])
    a.clear()
    assert a.count() == 0


def test_only_new_vectors_are_read(cfg):
    """Read whole after each fold, the vectors took 190 ms at a year's archive and 1 s at five,
    on the first search after he slept. Now only the new ones are read."""
    class Embedder(FakeLm):
        def embed(self, texts, model):
            return BagOfWords().embed(texts, model)

    a = Archive(dataclasses.replace(cfg.recall, embed_model="embedder"), Embedder(), "m")
    day = dt.datetime(2026, 11, 1, 9)
    a.add([entry(day + dt.timedelta(hours=k), f"Entry number {k} about gardening and slugs.") for k in range(5)])
    a._by_meaning(["slugs"])
    assert a.vectors_read == 5 and len(a._matrix[1]) == 5
    a._by_meaning(["slugs"])
    assert a.vectors_read == 5  # nothing new: nothing read
    a.add([entry(day + dt.timedelta(days=1), "The lasagne was lovely, Jo made it."),
           entry(day + dt.timedelta(days=1, hours=1), "Lasagne again tomorrow.")])
    best = a._by_meaning(["lasagne lovely"])
    assert a.vectors_read == 7 and len(a._matrix[1]) == 7  # only the two new ones
    assert a.entry(best[0] - 1).startswith("Crew: The lasagne") or "lasagne" in a.entry(best[0] - 1).lower()
    a.clear()  # anything else: read whole
    a.add([entry(day, "Slugs ate the lettuces.")])
    a._by_meaning(["slugs"])
    assert a.vectors_read == 8 and len(a._matrix[1]) == 1


def test_warm_loads_the_vectors_before_any_search(cfg):
    class Embedder(FakeLm):
        def embed(self, texts, model):
            return BagOfWords().embed(texts, model)

    a = Archive(dataclasses.replace(cfg.recall, embed_model="embedder"), Embedder(), "m")
    a.add([entry(dt.datetime(2026, 11, 1, 9), "Slugs ate the lettuces.")])
    a.warm()
    assert a.vectors_read == 1 and len(a._matrix[1]) == 1
    a._by_meaning(["slugs"])
    assert a.vectors_read == 1  # the search read nothing
    Archive(dataclasses.replace(cfg.recall, embed_model=""), Embedder(), "m").warm()  # no model: nothing to do



def test_recall_reads_his_notes_with_the_archive(cfg):
    """His notes said the kit was for emptying the fire; the archive only that the hoover was
    "to suck up the dirt". Answered from the archive alone, he said the dirt."""
    seen = []

    def judge(system, user):
        seen.append(user)
        return "ANSWER: For emptying the fire."

    a = archive(cfg, FakeLm(side=judge))
    a.add([entry(dt.datetime(2026, 11, 23, 9), "I got a special hoover to suck up the dirt.")])
    notes = "About the crew:\nThe crew's kit for emptying the fire is a dust mask and a special hoover."
    found = a.recall("special hoover", "What's the special hoover for?", notes=notes)
    assert found.answer == "For emptying the fire." and found.results
    assert seen[-1].startswith("(His notes on earlier conversations:)\nAbout the crew:")
    assert "suck up the dirt" in seen[-1]
    # Nothing in the archive, the notes alone are read.
    found = a.recall("fireplace", "What do I clean with the hoover?", notes=notes)
    assert found.answer == "For emptying the fire." and found.results
    # No notes, nothing found: as before.
    assert not a.recall("llamas", "Who has llamas?").results

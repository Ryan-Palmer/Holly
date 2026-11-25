"""The status report (app/status.py): made up from the ship's records, out of his sight."""

import asyncio
import random

from app import status
from app.archive import Archive
from app.speech_text import starts_section
from conftest import FakeLm


def records(cfg) -> Archive:
    """A tiny ship's records: Rimmer's page, with a quote that mustn't be used."""
    r = Archive(cfg.recall, FakeLm(), "m", cfg.recall.records_file, talk=False)
    r.add([(None, "(Ship's records: Arnold Rimmer, Personality) Rimmer is a coward and a stickler for rules."),
           (None, "(Ship's records: Arnold Rimmer, Childhood) He had three brothers, all more successful."),
           (None, "(Ship's records: Arnold Rimmer, Quotes) A line from the show, word for word."),
           (None, "(Ship's records: Arnold Rimmer, Back to Earth onwards) Something that happened later."),
           (None, "(Ship's records: Skutters, Skutters) Small maintenance droids with a fondness for cowboys.")])
    return r


def test_notes_are_who_they_are_and_never_quotes(cfg):
    """Given lines from the show, he recited them; given plot, the writer retold it."""
    r = records(cfg)
    for seed in range(20):
        got = status.notes(r, "Arnold Rimmer", 3, random.Random(seed))
        assert len(got) == 3 and not any("word for word" in n for n in got)
        assert sum("coward" in n or "brothers" in n for n in got) >= 2  # mostly from who he is
        assert not any(n.startswith("(Ship's records") for n in got)
    assert status.notes(r, "Nobody", 3, random.Random(1)) == [] and status.notes(None, "Rimmer", 3, random.Random()) == []


def test_the_brief_has_each_of_them_and_whats_been_done(cfg):
    text = status.brief(records(cfg), ["Lister — Old news."], random.Random(3))
    for name in ("Lister:", "Rimmer:", "Cat:", "Kryten:", "Additional:"):
        assert name in text
    assert "coward" in text or "brothers" in text
    assert "Reports already given lately" in text and "Lister — Old news." in text


def test_tidy_and_items():
    report = status.tidy("Here you go:\n**Lister** - Lost a bet.\nRimmer: Filed a complaint.\nThe Cat — Slept.\n"
                         "Kryten — Ironed socks.\nAdditional — A skutter was promoted.\nHope that helps!")
    assert report.splitlines() == ["Lister — Lost a bet.", "Rimmer — Filed a complaint.", "Cat — Slept.",
                                   "Kryten — Ironed socks.", "Additional — A skutter was promoted."]
    assert status.items(report)["Cat"] == "Slept."
    assert status.tidy("Not a report at all.") == "Not a report at all."


def drafts_lm(picks: str = "Lister: 2\nRimmer: 3\nCat: 1\nKryten: 2\nAdditional: 3"):
    n = []

    def side(system, user):
        if system == status.WRITER_PROMPT:
            n.append(1)
            return "\n".join(f"{name} — Wrote draft {len(n)}." for name in status.NAMES)
        if system == status.PICK_PROMPT:
            return picks
        return ""
    return FakeLm(side=side)


def test_the_best_of_each_item_is_kept(cfg):
    """Alone, one item in three was a dud: three drafts, and the best of each, picked."""
    tool = status.StatusTool(drafts_lm(), "m", records(cfg))
    report = asyncio.run(tool.write())
    assert report.splitlines() == ["Lister wrote draft 2.", "Rimmer wrote draft 3.", "The Cat wrote draft 1.",
                                   "Kryten wrote draft 2.", "Additional — Wrote draft 3."]
    # A pick that's nonsense keeps the first draft's.
    report = asyncio.run(status.StatusTool(drafts_lm("No idea."), "m").write())
    assert report.splitlines()[0] == "Lister wrote draft 1."


def test_reports_are_remembered_between_sessions(cfg, tmp_path):
    """So tomorrow's isn't today's again."""
    file = tmp_path / "status.json"
    tool = status.StatusTool(drafts_lm(), "m", None, file)
    result = asyncio.run(tool.run({}, None))
    assert result.startswith(status.REPORT) and "Lister wrote draft 2." in result
    again = status.StatusTool(drafts_lm(), "m", None, file)
    assert list(again.recent) == list(tool.recent) and len(again.recent) == 1


def test_a_report_that_cant_be_written_isnt_made_up(cfg):
    def broken(system, user):
        raise ConnectionError("LM Studio restarting")

    result = asyncio.run(status.StatusTool(FakeLm(side=broken), "m").run({}, None))
    assert "didn't come through" in result and not result.startswith(status.REPORT)


def test_each_crew_member_starts_a_section():
    """A longer pause before each, as between subjects of the news."""
    for line in ("Lister — Lost a bet.", "Rimmer — Filed a complaint.", "The Cat — Slept.", "Kryten — Ironed socks.",
                 "Additional — A skutter was promoted."):
        assert starts_section(line)


def test_who_is_one_of_the_crew():
    assert status.who("Kryten") == "Kryten" and status.who("the Cat") == "Cat" and status.who("rimmer") == "Rimmer"
    assert status.who("Holly") is None and status.who(None) is None and status.who("") is None


def test_one_of_them_is_somewhere_doing_something(cfg):
    """Asked where one of them is: their notes, a place and a kind of thing, at random (left
    to choose, nearly all were in the galley), and what's been said of them lately kept out."""
    text = status.brief_one(records(cfg), "Rimmer", ["Rimmer — Polished his medals.\nCat — Slept."], "",
                            random.Random(2))
    assert text.startswith("Rimmer: in ") and ("coward" in text or "brothers" in text)
    assert "Not these again" in text and "Polished his medals." in text and "Slept." not in text


def test_after_a_report_one_of_them_carries_on_from_it(cfg):
    """Asked about one of them after a report, the answer moved on to something new."""
    text = status.brief_one(records(cfg), "Rimmer", [], "Filed a complaint about the complaints procedure.")
    assert "carries on from exactly that" in text and "Filed a complaint about the complaints procedure." in text

    asked = []

    def side(system, user):
        asked.append(system)
        if system.startswith(status.PICK_ONE_PROMPT):
            return "2"
        return f"He's in the drive room, draft {len(asked)}."

    tool = status.StatusTool(FakeLm(side=side), "m", records(cfg))
    tool.given = "Rimmer — Filed a complaint.\nCat — Slept."
    answer = asyncio.run(tool.write_one("Rimmer"))
    assert answer == "He's in the drive room, draft 2."  # three drafts, the second picked
    assert all("Filed a complaint." in s for s in asked)  # the writer and the picker both told


def test_an_answer_about_one_of_them(cfg, tmp_path):
    def side(system, user):
        return "1" if system.startswith(status.PICK_ONE_PROMPT) else "He's in the gym, losing to a rowing machine."

    tool = status.StatusTool(FakeLm(side=side), "m", None, tmp_path / "status.json")
    result = asyncio.run(tool.run({"who": "kryten"}, None))
    assert result.startswith(status.ANSWER) and status.report_of(result) == "He's in the gym, losing to a rowing machine."
    assert list(tool.recent) == ["Kryten — He's in the gym, losing to a rowing machine."]


def test_the_last_report_in_his_history_is_kept_to():
    """Given before a restart, it's what an answer about one of them carries on from."""
    def told(start: str, text: str, end: str) -> str:
        return "\n".join((start, text, end))

    history = [{"role": "user", "content": "Status report"},
               {"role": "tool", "tool_call_id": "1", "content": told(status.REPORT, "Rimmer — Old.", status.AFTER)},
               {"role": "tool", "tool_call_id": "2", "content": told(status.REPORT, "Rimmer — Newer.", status.AFTER)},
               {"role": "tool", "tool_call_id": "3",
                "content": told(status.ANSWER, "He's in the gym.", status.AFTER_ANSWER)}]
    assert status.last_given(history) == "Rimmer — Newer." and status.last_given([]) == ""


def test_the_crew_are_said_in_sentences_and_additional_is_a_heading():
    """Written "Lister — Lost a bet.", the voice paused at the dash: the names came out as headings."""
    assert status.said_as("Lister", "Lost a bet.") == "Lister lost a bet."
    assert status.said_as("Cat", "Refused to move.") == "The Cat refused to move."
    assert status.said_as("Kryten", "His spare heads walked out.") == "As for Kryten, his spare heads walked out."
    assert status.said_as("Cat", "DivaDroid sued him.") == "As for the Cat, DivaDroid sued him."
    assert status.said_as("Additional", "A skutter was promoted.") == "Additional — A skutter was promoted."
    said = "Lister lost a bet.\nThe Cat slept.\nAs for Kryten, his heads walked out.\nAdditional — A recall."
    assert status.items(said) == {"Lister": "lost a bet.", "Cat": "slept.", "Kryten": "his heads walked out.",
                                  "Additional": "A recall."}
    assert status.items("Rimmer — Filed a complaint.") == {"Rimmer": "Filed a complaint."}  # as written, too


def test_finding_one_of_them_has_its_own_stock_line():
    """"Hang on, I'll check the monitors" fits "where's Rimmer?", not a status report."""
    from app.tools import Toolbox
    box = Toolbox([status.StatusTool(FakeLm(), "m")], 3)
    assert box.filler([{"name": "status", "arguments": "{}"}]) == "status"
    assert box.filler([{"name": "status", "arguments": '{"who": "Rimmer"}'}]) == "crew"
    assert box.filler(["status"]) == "status"

"""The safety nets' patterns: what trips them, and what mustn't."""

import pytest

from app.guards import (CLAIMED_RECALL, CLOCK_TIME, COUNTDOWN, FORGOT, MEMORY_ASKED, NO_CLOCK, OFFER, OFFERED_REPORT,
                        PROMISED_RESEARCH, REPORT_ASKED, REPORT_WANTED, RESEARCH_ASKED, without_offers)

CASES = [
    (COUNTDOWN, ["Six minutes left.", "Four minutes until you're stirring that pasta.", "Two minutes to go.",
                 "You've got about a minute before you're back at it.", "Ten minutes left on the timer."],
     ["Two minutes. Don't let it burn.", "It's due at 22:13.", "It took me a few minutes to work out.",
      "Bake it in ten minutes."]),
    (OFFER, ["Anything else you need me to dig up?", "Any other bits of info you're after?",
             "Just let me know if you hit a snag.", "I'll be right here if you need anything.",
             "Shall I work out the rest, or are you all set?", "Does that clear things up?", "Is there anything else?",
             "Just give us a shout if you need anything.", "Just shout if you want me to do something else.",
             "Need me to do anything?", "Anything else for me?", "Anything else on your mind, or are you off?"],
     ["Want a summary?", "Quick search now, or shall I dig into it while I doze?", "How long for?",
      "It didn't list anything else.", "Is there anything in the fridge?", "It had no other data.",
      "Do you want me to open it?", "I'll be here, doing absolutely nothing."]),
    (CLAIMED_RECALL, ["I've used the recall tool, but nothing.", "I've checked my records.", "It's not in my files.",
                      "I have no record of that.", "It's not in my notes."],
     ["I remember it well.", "Let me think.", "The record player's broken.", "Take notes.",
      "I've checked my notes: Jo's a vet."]),
    (MEMORY_ASKED, ["Do you remember Jo?", "Who's Jo?", "Have you forgotten my sister?", "Do you know who June is?"],
     ["What's the weather like?", "Play some music."]),
    (FORGOT, ["I don't remember her.", "Name's gone a bit foggy.", "It's slipped my mind.", "Doesn't ring a bell.",
              "I can't quite place him.", "I don't have anything in my notes about a problem.",
              "I didn't have that in my files.", "I've got nothing on the company.", "Nothing in my notes."],
     ["I remember her well.", "Jo, your sister.", "I don't have anything on today.", "Nothing on telly.",
      "You've got nothing on me."]),
    (RESEARCH_ASKED, ["Can you research heat pumps?", "Look into slugs for me.", "Find out about e-bikes.",
                      "Write a report on chickens."],
     ["What's the time?", "Look at that!"]),
    (PROMISED_RESEARCH, ["I'll look into it.", "I'll get back to you.", "I'll do it while I doze.", "I'm on it.",
                         "I'll put together a proper report.", "I'll get onto that.",
                         "I'll have a report ready for you once I've finished my nap.", "After a quick snooze.",
                         "You'll have it when I wake."],
     ["Heat pumps move heat.", "Slugs like beer."]),
    (REPORT_WANTED, ["A proper report, please.", "Write it up.", "Do it properly.", "Research it while you doze."],
     ["Just a quick search.", "Search it now."]),
    (OFFERED_REPORT, ["Quick search now, or shall I dig into it while I doze?", "Want a proper write-up?"],
     ["Shall I search for it?"]),
    (NO_CLOCK, ["I've no clock of my own.", "I can't tell you the time.", "No idea what time it is.",
                "I haven't got a watch."],
     ["It's Tuesday.", "The clock's broken in the galley.", "Casino watch."]),
    (CLOCK_TIME, ["It's 10:14.", "About 10 pm.", "Ten o'clock.", "At 7:30 a.m. sharp", "It's 5:05 pm."],
     ["That's £12.50.", "Ten minutes.", "Two pmegs."]),
]


@pytest.mark.parametrize("pattern, hits, misses", CASES, ids=[c[0].pattern[:20] for c in CASES])
def test_pattern(pattern, hits, misses):
    assert [h for h in hits if not pattern.search(h)] == []
    assert [m for m in misses if pattern.search(m)] == []


def test_without_offers():
    assert without_offers("Up again. Since 2022, mostly. Anything else you need?") == "Up again. Since 2022, mostly."
    assert without_offers("Anything else?") == "Anything else?"  # all there is: kept
    assert without_offers("  Fifteen degrees. Hardly tropical.\n") == "Fifteen degrees. Hardly tropical."


def test_a_time_counts_as_given_only_if_it_was():
    from app.guards import guessed_time
    assert not guessed_time("It's 22:14.", ["What time is it?", "Here: 22:14 on Tuesday."])
    assert guessed_time("It's 22:15.", ["Lovely.", "Here: 22:14 on Tuesday."])  # a minute on from what he was told
    assert not guessed_time("Ready at 7:30 pm.", ["Remind me at 7:30 pm"])
    assert not guessed_time("Nothing timed.", [])


def test_a_time_is_given_however_hes_put_it():
    """Told "17:05" by the clock, or "sunset 16:02", his "5:05 pm", "five o'clock" and "just
    after 4 pm" were taken for guesses: not said, or redone and kept marked as wrong."""
    from app.guards import guessed_time
    clock = ["What time is it?", "Here: 17:05 on Wednesday 25 November 2026 (UTC+0)."]
    assert not guessed_time("It's 5:05 pm.", clock)
    assert not guessed_time("It's 5:05.", clock)
    assert not guessed_time("It's 17:05.", clock)
    assert not guessed_time("Five o'clock, near enough.", clock)
    assert not guessed_time("Sunset's just after 4 pm.", ["Tomorrow: sunrise 07:45, sunset 16:02."])
    assert not guessed_time("Ready at 19:30.", ["Remind me at 7:30"])
    assert guessed_time("It's 5:15.", clock)  # to the minute: not what he was told
    assert guessed_time("About 10 pm.", clock)  # an hour alone, nowhere near


def test_minutes():
    from app.guards import minutes
    assert minutes("17:05") == ({1025}, True)
    assert minutes("5:05") == ({305, 1025}, True)
    assert minutes("07:45") == ({465}, True)
    assert minutes("7:30 a.m.") == ({450}, True)
    assert minutes("4 pm") == ({960}, False)
    assert minutes("12 am") == ({0}, False)
    assert minutes("Ten o'clock") == ({600, 1320}, False)


def test_due_labels_however_theyre_punctuated():
    """A label with an apostrophe before its tool's word ("Jo's playlist") wasn't seen."""
    import datetime as dt

    from app.guards import ACTION_WORDS, DUE_LABEL
    from app.timers import announcement
    note = announcement([{"label": "Jo's playlist", "due": "2026-11-25T18:00:00", "kind": "timer"},
                         {"label": "pasta", "due": "2026-11-25T18:00:00", "kind": "reminder"}],
                        now=dt.datetime(2026, 11, 25, 18, 0, 30))
    assert DUE_LABEL.findall(note) == ["Jo's playlist", "pasta"]
    assert [label for label in DUE_LABEL.findall(note) if ACTION_WORDS.search(label)] == ["Jo's playlist"]


def test_a_crew_aside_and_the_rest_after_it_are_left_out():
    """"The Cat is currently refusing to leave his wardrobe. He's convinced the hangers are
    judging him.", after research queued: the second sentence, kept, and every Cat line
    after was about coat hangers."""
    reply = ("I've got it in hand. The Cat is currently refusing to leave his wardrobe. "
             "He's convinced the hangers are judging him.")
    assert without_offers(reply, asides=True) == "I've got it in hand."
    assert without_offers(reply) == reply  # asked about the crew: kept
    assert without_offers("The Cat slept.", asides=True) == "The Cat slept."  # all there is: kept
    assert without_offers("A jigsaw, maybe. The cat would hate it.", asides=True) == "A jigsaw, maybe. The cat would hate it."


@pytest.mark.parametrize("line", [
    "Can you look into the best dog breeds for a small house and write me a report?",
    "Holly, research the best electric bikes under a thousand pounds and write me a report.",
    "Could you do a report on heat pumps?", "I'd like a report on air fryers.", "Write a report on slugs.",
    "Holly, please write up a proper report on solar panels", "Go on then, write it up.",
    "Can you put together a short write-up on composting?"])
def test_a_report_asked_for_in_so_many_words(line):
    assert REPORT_ASKED.search(line)


@pytest.mark.parametrize("line", [
    "Status report, please.", "What did the board games report say?", "I need to write a report for work tonight.",
    "Could you look into slugs in the garden?", "My boss wants a status update by five.", "Can you write me a poem?",
    "Can you sum up the coffee machine report?", "The report was rubbish.", "Can you open the report?"])
def test_a_report_only_mentioned_isnt_one_asked_for(line):
    assert not REPORT_ASKED.search(line)

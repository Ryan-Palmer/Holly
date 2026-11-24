"""The safety nets' patterns: what trips them, and what mustn't."""

import pytest

from app.guards import (CLAIMED_RECALL, CLOCK_TIME, FORGOT, MEMORY_ASKED, NO_CLOCK, OFFER, RECALL_ASKED, OFFERED_REPORT,
                        PROMISED_RESEARCH, REPORT_WANTED, RESEARCH_ASKED, without_offers)

CASES = [
    (RECALL_ASKED, ["Do you remember what they thought of Lister?", "Can you remember using your tools?",
                    "Do you recall the Cat Fleet?", "Remember when we met the cats?", "Have you forgotten Jo?"],
     ["Who's cleverer, you or me?", "Remember to call June.", "Do you miss Lister?", "Who is Jo?"]),
    (OFFER, ["Anything else you need me to dig up?", "Any other bits of info you're after?",
             "Just let me know if you hit a snag.", "I'll be right here if you need anything.",
             "Shall I work out the rest, or are you all set?", "Does that clear things up?", "Is there anything else?",
             "Just give us a shout if you need anything.", "Just shout if you want me to do something else.",
             "Need me to do anything?", "Anything else for me?", "Anything else on your mind, or are you off?"],
     ["Want a summary?", "Quick search now, or shall I dig into it while I doze?", "How long for?",
      "It didn't list anything else.", "Is there anything in the fridge?", "It had no other data.",
      "Do you want me to open it?", "I'll be here, doing absolutely nothing."]),
    (CLAIMED_RECALL, ["I've used the recall tool, but nothing.", "I've checked my records.", "It's not in my files.",
                      "I have no record of that.", "I've checked my notes.", "It's not in my notes."],
     ["I remember it well.", "Let me think.", "The record player's broken.", "Take notes."]),
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
    (CLOCK_TIME, ["It's 10:14.", "About 10 pm.", "Ten o'clock.", "At 7:30 a.m. sharp"],
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

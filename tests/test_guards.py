"""The safety nets' patterns: what trips them, and what mustn't."""

import pytest

from app.speak import (CLAIMED_RECALL, CLOCK_TIME, FORGOT, MEMORY_ASKED, OFFERED_REPORT, PROMISED_RESEARCH,
                       REPORT_WANTED, RESEARCH_ASKED)

CASES = [
    (CLAIMED_RECALL, ["I've used the recall tool, but nothing.", "I've checked my records.", "It's not in my files.",
                      "I have no record of that."],
     ["I remember it well.", "Let me think.", "The record player's broken."]),
    (MEMORY_ASKED, ["Do you remember Jo?", "Who's Jo?", "Have you forgotten my sister?", "Do you know who June is?"],
     ["What's the weather like?", "Play some music."]),
    (FORGOT, ["I don't remember her.", "Name's gone a bit foggy.", "It's slipped my mind.", "Doesn't ring a bell.",
              "I can't quite place him."],
     ["I remember her well.", "Jo, your sister."]),
    (RESEARCH_ASKED, ["Can you research heat pumps?", "Look into slugs for me.", "Find out about e-bikes.",
                      "Write a report on chickens."],
     ["What's the time?", "Look at that!"]),
    (PROMISED_RESEARCH, ["I'll look into it.", "I'll get back to you.", "I'll do it while I doze.", "I'm on it.",
                         "I'll put together a proper report."],
     ["Heat pumps move heat.", "Slugs like beer."]),
    (REPORT_WANTED, ["A proper report, please.", "Write it up.", "Do it properly.", "Research it while you doze."],
     ["Just a quick search.", "Search it now."]),
    (OFFERED_REPORT, ["Quick search now, or shall I dig into it while I doze?", "Want a proper write-up?"],
     ["Shall I search for it?"]),
    (CLOCK_TIME, ["It's 10:14.", "About 10 pm.", "Ten o'clock.", "At 7:30 a.m. sharp"],
     ["That's £12.50.", "Ten minutes.", "Two pmegs."]),
]


@pytest.mark.parametrize("pattern, hits, misses", CASES, ids=[c[0].pattern[:20] for c in CASES])
def test_pattern(pattern, hits, misses):
    assert [h for h in hits if not pattern.search(h)] == []
    assert [m for m in misses if pattern.search(m)] == []

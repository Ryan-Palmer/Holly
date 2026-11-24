"""Holly's safety nets: sentences that mustn't be said, and times he guessed.

Each net is a pattern found in a sentence as it's written. A sentence that trips one
isn't said: the reply is asked again (app/speak.py), with a tool call required when
the trouble was a tool he claimed and didn't call. Every one is here because he did it,
over and over, when told not to in his rules alone.
"""

from __future__ import annotations

import re

# A claim to have looked in his memory: "I've used the recall tool, but...", "I've
# checked my records", "not in my files". Said without calling recall, it stuck: copied
# from his history, he said it a dozen times in a row and recalled only when told to.
# Checking his notes is fine (they're in his prompt): "I've checked my notes: Jo's a vet".
# Saying something isn't in them is a claim to have looked.
CLAIMED_RECALL = re.compile(
    r"\brecall tool\b|\b(?:used|checked|searched|looked (?:through|in)|consulted|scanned|gone through) "
    r"(?:through |in )?(?:my |the |your )?(?:own )?(?:records|archives?|files|memory banks?|data ?banks?|"
    r"memory)\b|\bnot in my (?:records|archives?|files|memory|notes)\b|\b(?:no|any) records? (?:of|about|on)\b",
    re.IGNORECASE)
# Asked if he remembered something, he said he didn't, without calling recall ("the name's
# gone a bit foggy", of Queeg; "lovely to meet you", to the crew's own girlfriend), and
# recalled only when told to use it: his senility is for trivia, not for what's in his
# archive. So asked about his memory, he can't say he doesn't remember until he's looked.
MEMORY_ASKED = re.compile(r"\b(?:remember|recall|forgot(?:ten)?|forget|do you know (?:who|what|about)|"
                          r"who(?:'s| is| was)|you met)\b", re.IGNORECASE)
FORGOT = re.compile(r"\b(?:don't|do not|can't|cannot|couldn't) (?:quite |really |honestly )?(?:remember|recall|"
                    r"place|bring (?:it|that|them|her|him) to mind)\b|\bslipped my mind\b|\bgone (?:a bit )?"
                    r"(?:foggy|fuzzy|blurry|hazy)\b|\bnot sticking\b|\bno idea who\b|\bnever heard of\b|"
                    r"\bdoesn't ring a bell\b|"
                    # Asked about the crew's news, from his notes alone: "I don't have anything in
                    # my notes about a problem", "I've got nothing on the company". It was in his
                    # archive.
                    r"\b(?:don't|didn't|haven't) (?:got |have )?(?:anything|that|it|much) (?:in|on) "
                    r"(?:my |the )?(?:notes|files|records|archives?|memory)\b|"
                    # His own, not banter: "you've got nothing on me", asked who's cleverer.
                    r"\bnothing in my (?:notes|files|records)\b|\bI(?:'ve| have)? got nothing (?:on|about)\b",
                    re.IGNORECASE)
# Asked to research something, he promised to without calling the research tool ("I'll get
# on with that later and come back to you"), and nothing was ever researched. So when the
# crew ask for research and he promises it, the research tool must have been called.
RESEARCH_ASKED = re.compile(r"\b(?:research|look(?:ing)? into|find out (?:all )?about|report on|dig into|"
                            r"read up on|look up everything)\b", re.IGNORECASE)
# ("I'll get onto that. I'll have a report ready for you once I've finished my nap", and
# no research started: so "get onto", "have... ready" and naps too.)
PROMISED_RESEARCH = re.compile(r"\b(?:look into (?:it|that|this)|get on(?:to| with)? (?:it|that|this)|get back to "
                               r"you|come back to you|let you know|report back|while (?:things are quiet|"
                               r"I doze|I'm dozing|I'm asleep|I sleep|you sleep)|have a proper look|dig into it|"
                               r"have (?:a|the|your|it|that) (?:proper |full )?(?:report |write-up )?ready|"
                               r"(?:my|a)(?: little| quick| short)? (?:nap|snooze|doze|kip)|when I wake|"
                               r"get(?:ting)? started|(?:get|got|start|started|starting|do|doing|write|writing|"
                               r"put together) (?:on )?(?:a|the|that|your) (?:proper |full |research |little )?"
                               r"(?:report|write-up)|report (?:started|going|under way)|it's in hand|get cracking|"
                               r"crack on|I'm on it)\b",
                               re.IGNORECASE)
# The crew asked for a report outright, or chose one when he asked: research may start.
# Anything less ("look into slugs", "find out about heat pumps"), he asks first: a quick
# search now, or a report while he dozes? Told to ask, in a rule alone, he mostly didn't.
REPORT_WANTED = re.compile(r"\b(?:research|report|write[- ]?up|write it (?:all )?up|in depth|in detail|properly|"
                           r"proper (?:look|one|job)|full one|long one|while you doze|while you sleep|later)\b",
                           re.IGNORECASE)
# His last reply offered the choice in his own words ("shall I dig into it while I doze?").
OFFERED_REPORT = re.compile(r"\b(?:report|write (?:it|that|them) up|write-up|proper|properly|research|digging|"
                            r"dig into|while I(?:'m)? doz|while I sleep)", re.IGNORECASE)
# Saying he can't tell the time, when he has the clock tool: told in his rules that the time
# came only from the clock, he said "I've no clock of my own" (asked what day tomorrow was,
# which he knew), and read the clock only when told to look at it.
NO_CLOCK = re.compile(r"\b(?:no|not got a|haven't (?:got )?a|don't have a|without a) (?:clock|watch)\b|"
                      r"\bcan(?:'t|not) (?:tell|see|know) (?:you )?(?:the|what) time\b|"
                      r"\bno (?:idea|way of knowing) what time\b", re.IGNORECASE)
# A time of day in a reply: "10:14", "10 pm", "ten o'clock". Not "12.50": that's money.
CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.)(?=\W|$)|"
                        r"\bo'clock\b", re.IGNORECASE)
# A tool call written out as text, in the model's own markup: asked "Do you miss Lister?",
# Gemma's reply was `recall{query:<|"|>Lister<|"|>,...}<tool_call|>`, to be read aloud.
# Matched as written, and as the sentence chunker leaves it (without "|" or "_"):
# `recall{query:<"Lister<"}<toolcall`.
TOOL_MARKUP = re.compile(r"<\||\|>|<tool_?call|<\"|\b\w+\{\w+:")
# A closing offer of more help, a receptionist's: "Anything else you need me to dig up?",
# "I'll be here if you need anything". At the end of reply after reply it became his
# voice, and he copied it from his history every turn. Not said, and kept out of what he
# sees of his history (app/persona.py); not the questions the talk needs, either: "want a
# summary?", "shall I dig into it while I doze?", "how long for?". Nor an answer that has
# the words in it: "It didn't list anything else" is what the report said.
OFFER = re.compile(r"\banything else\b[^.!?]*\?|\banything else (?:you|I can|I could|for (?:you|me))\b|"
                   r"\bany other (?:bits|questions|info\w*|data)\b[^.!?]*\?|"
                   r"\bis there anything (?:else|I can|you need|you'd like)\b|"
                   r"\b(?:let me know if|just let me know|I'll be (?:right )?here if|are you all set|shout if|"
                   r"(?:need|want) me to do anything|does that clear)\b", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def without_offers(text: str) -> str:
    """`text` without its sentences offering more help (OFFER), unless that's all it is."""
    text = text.strip()
    sentences = SENTENCE_END.split(text)
    kept = [s for s in sentences if not OFFER.search(s)]
    return " ".join(kept) if kept and len(kept) < len(sentences) else text


# Added to a reply that gave a time nobody gave him (never spoken: it's for his history).
GUESSED = " [That time was a guess, not from the clock: it's probably wrong.]"


def unsaid(sentence: str, user: str, *, stray: set[str], recalled: bool, researched: bool, can_recall: bool,
           can_research: bool, answering_choice: bool, clocked: bool = False, can_clock: bool = False) -> str | None:
    """Why `sentence`, from a reply to `user`, mustn't be said, if it mustn't. `recalled`,
    `researched`, `clocked`: he's called those tools in this reply; `can_...`: he has them;
    `answering_choice`: the crew are answering his search-or-report question."""
    if (word := sentence.strip().strip(".!").lower()) in stray:  # Gemma's lone "thought"
        return f"a stray {word!r}"
    if TOOL_MARKUP.search(sentence):
        return "he wrote a tool call out as text"
    if not clocked and can_clock and NO_CLOCK.search(sentence):
        return "he said he couldn't tell the time without calling the clock"
    if not recalled and can_recall and CLAIMED_RECALL.search(sentence):
        return "he said he'd looked in his memory without calling recall"
    if not recalled and can_recall and MEMORY_ASKED.search(user) and FORGOT.search(sentence):
        return "he said he didn't remember without calling recall"
    # Asked for research, or answering his search-or-report question ("go on then, write it
    # up"): he said he'd started a report and called nothing, four times in six. A question
    # isn't a promise: "shall I have a proper look while I doze?" is him asking which the
    # crew want, as he's meant to.
    if (not researched and can_research
            and (RESEARCH_ASKED.search(user) or REPORT_WANTED.search(user) or answering_choice)
            and PROMISED_RESEARCH.search(sentence) and not sentence.rstrip().endswith("?")):
        return "he promised research without calling the research tool"
    return None


def needs_tool(why: str) -> bool:
    """Is a sentence dropped for this reason (from unsaid) redone with a tool required?"""
    return any(w in why for w in ("recall", "research", "clock", "tool call"))


def guessed_time(said: str, told: list[str]) -> bool:
    """Did he give a time of day that nobody gave him (the clock, the weather's sunset, a
    search, the crew, a timer coming due)? Kept as fact in his history, a guess stuck:
    greeted, he said "10:14 PM" at 7:56, and repeated it four times, asked to use the
    clock or not."""
    return bool(CLOCK_TIME.search(said)) and not any(CLOCK_TIME.search(t) for t in told)

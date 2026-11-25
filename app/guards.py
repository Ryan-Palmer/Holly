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
# A report asked of him in so many words ("look into dog breeds for a small house and write
# me a report"): his reply must call a tool (app/speak.py). Well into an evening's talk he
# asked "a quick search now, or a proper report?" anyway, calling nothing, 5 times in 6 (on
# its own, never). Narrower than REPORT_WANTED, which only lets a start through: not
# "status report", "the board games report", or "I need to write a report for work".
_WRITE = r"(?:write|do|put together|make|draw up|prepare)"
_A_REPORT = r"(?:a|an)\s+(?:(?:proper|full|short|quick|little|detailed)\s+)?(?:report|write-?up)\b"
REPORT_ASKED = re.compile(
    rf"\b{_WRITE}\s+(?:me|us)\s+(?:up\s+)?{_A_REPORT}"  # "write me a report"
    rf"|\b(?:can|could|will|would)\s+you\b[^.?!]*?\b{_WRITE}\s+(?:up\s+)?{_A_REPORT}"  # "can you ... do a report"
    rf"|^\W*(?:holly\W+)?(?:please\s+)?{_WRITE}\s+(?:up\s+)?{_A_REPORT}"  # "Write a report on..."
    rf"|\b(?:want|like|need)\s+{_A_REPORT}\s+(?:on|about|into)\b"  # "I'd like a report on"
    r"|\bwrite (?:it|that|this|them) (?:all )?up\b",
    re.IGNORECASE)
# Timers and reminders come due, in timers.announcement's note: each label, however it's
# punctuated ("Jo's playlist").
DUE_LABEL = re.compile(r"\bThe (?:timer|reminder) '(.*?)': it(?:'s due now| came due)")
# A label that may be a tool's job: told in the note to do it with the tool, he announced
# "check the weather in Keswick" and didn't, one time in four. Words alone took "go to Jo's
# school play" and "search for my keys" for jobs too, so these are only asked about
# (timers.to_do); if the model says so, his reply must call a tool (app/speak.py).
ACTION_WORDS = re.compile(r"\b(?:play|put on|music|songs?|album|playlist|weather|forecast|news|headlines|"
                          r"time|look up|search|find out)\b", re.IGNORECASE)
# Words that may be a status report asked for: then asked about (status.wants_report), since
# "my boss wants a status update" isn't one. If it is, the news tool, every time.
STATUS_ASKED = re.compile(r"\bstatus (?:report|update)s?\b", re.IGNORECASE)
# His last reply may have offered research, in his own words ("shall I dig into it while I
# doze?", "I can do a proper research job on it if you want"): then whether the crew took
# him up on it is asked of the model (research.accepted, from app/speak.py).
OFFERED_REPORT = re.compile(r"\b(?:report|write (?:it|that|them) up|write-up|proper|properly|research|digging|"
                            r"dig into|while I(?:'m)? doz|while I sleep)", re.IGNORECASE)
# Saying he can't tell the time, when he has the clock tool: told in his rules that the time
# came only from the clock, he said "I've no clock of my own" (asked what day tomorrow was,
# which he knew), and read the clock only when told to look at it.
NO_CLOCK = re.compile(r"\b(?:no|not got a|haven't (?:got )?a|don't have a|without a) (?:clock|watch)\b|"
                      r"\bcan(?:'t|not) (?:tell|see|know) (?:you )?(?:the|what) time\b|"
                      r"\bno (?:idea|way of knowing) what time\b", re.IGNORECASE)
# A time of day in a reply: "10:14", "10:14 pm", "10 pm", "ten o'clock". Not "12.50": money.
_HOURS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
_AMPM = r"\s?(?:am|pm|a\.m\.|p\.m\.)(?=\W|$)"
CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}(?:" + _AMPM + r"|\b)|\b\d{1,2}" + _AMPM + r"|\b(?:\d{1,2}|"
                        + "|".join(_HOURS) + r")\s+o'clock\b", re.IGNORECASE)
NEAR_MIN = 15  # an hour alone ("4 pm", "five o'clock") is said of a time this close to it
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


def without_offers(text: str, countdowns: bool = False, told: list[str] | None = None, asides: bool = False) -> str:
    """`text` without its sentences offering more help (OFFER); with `countdowns`, those
    counting a timer down (COUNTDOWN); given `told`, those with a time of day nobody gave him
    (guessed_time); with `asides`, a later one about the crew (CREW_ASIDE) and the rest of the
    reply after it; unless that's all it is."""
    text = text.strip()
    sentences = SENTENCE_END.split(text)
    if asides:
        aside = next((i for i, s in enumerate(sentences) if i and CREW_ASIDE.search(s)), len(sentences))
    else:
        aside = len(sentences)
    kept = [s for i, s in enumerate(sentences) if i < aside and not OFFER.search(s)
            and not (countdowns and COUNTDOWN.search(s)) and not (told is not None and guessed_time(s, told))]
    return " ".join(kept) if kept and len(kept) < len(sentences) else text


# The crew of Red Dwarf, tacked on to a reply about something else: after a run of status
# reports in his history, a DeWalt gift list ended "The Cat's currently having a standoff
# with a mirror", and research queued, "The Cat is currently refusing to leave his
# wardrobe. He's convinced the hangers are judging him." And the rest after it: that second
# sentence, kept, and every Cat line after was about coat hangers. Not said, nor kept,
# unless the crew were asked about (CREW_ASKED) or a status report or recall is what's
# being given. Capitals: "the cat" may be the crew's own. Not "crew" for asked: they're
# "the crew" to him, and a report announced says "Tell the crew it's ready".
CREW_ASIDE = re.compile(r"^\W*(?:The Cat|Lister|Rimmer|Kryten)\b")
CREW_ASKED = re.compile(r"\b(?:Lister|Rimmer|Kryten|Cat|status|Red Dwarf)\b", re.IGNORECASE)


# A timer counted down in passing, "Six minutes left", "Two minutes to go", "about a minute
# before you're back at it". With a timer set and no tool called, he took a minute off his
# last count each reply: five replies in one minute, from six minutes to "about a minute",
# with five to go. He has no clock of his own; the timer tool knows.
_COUNT = (r"(?:\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
          r"thirty|forty|forty-five|sixty|a couple of|a few|about an?|half an)")
COUNTDOWN = re.compile(r"\b" + _COUNT + r"\s+(?:more\s+)?(?:minutes?|mins?|seconds?|secs?|hours?)\s+"
                       r"(?:left|to go|remaining|until|till|before)\b|\b(?:left|remaining) on (?:the|that|your) timer\b",
                       re.IGNORECASE)

# The crew asking about time or a timer: then a count down is his answer, and must come from
# the timer tool. Unasked, it's said in passing, and isn't said at all.
TIME_ASKED = re.compile(r"\b(?:how long|how much (?:time|longer)|time left|what time|when(?:'s| is| will| does)|"
                        r"timer|alarm|reminder|minutes?|due|left on)\b", re.IGNORECASE)

# Added to a reply that gave a time nobody gave him (never spoken: it's for his history).
GUESSED = " [That time was a guess, not from the clock: it's probably wrong.]"


# Every net on his replies, by the name it's counted under. Each makes up for something a
# model did wrong. app/speak.py counts each one as it fires, and the evals show how often:
# with --raw, how often they would have, acting on nothing (app/evals.py). One a new model
# never trips can come out. Asked again (unsaid, and a word alone):
STRAY = "a stray word"
MARKUP = "he wrote a tool call out as text"
COUNTED = "he counted a timer down without the timer tool"
TIMED = "he gave a time of day without the clock"
NO_CLOCK_SAID = "he said he couldn't tell the time without calling the clock"
CLAIMED = "he said he'd looked in his memory without calling recall"
FORGOTTEN = "he said he didn't remember without calling recall"
PROMISED = "he promised research without calling the research tool"
ALONE = "a word alone"
# Not said, nor kept (app/speak.py); and marked in his history:
OFFERED = "an offer, not said"
GUESS_CUT = "a guessed time, not said"
COUNT_CUT = "a countdown, not said"
ASIDE_CUT = "a crew aside, not said"
GUESS_MARKED = "a guessed time, marked as one in his history"
NETS = (STRAY, MARKUP, COUNTED, TIMED, NO_CLOCK_SAID, CLAIMED, FORGOTTEN, PROMISED, ALONE,
        OFFERED, GUESS_CUT, COUNT_CUT, ASIDE_CUT, GUESS_MARKED)


def unsaid(sentence: str, user: str, *, stray: set[str], recalled: bool, researched: bool, can_recall: bool,
           can_research: bool, answering_choice: bool, clocked: bool = False, can_clock: bool = False,
           counting: bool = False, told: list[str] | None = None) -> str | None:
    """Why `sentence`, from a reply to `user`, mustn't be said, if it mustn't. `recalled`,
    `researched`, `clocked`: he's called those tools in this reply; `can_...`: he has them;
    `answering_choice`: the crew are answering his search-or-report question; `counting`:
    a timer's set and he hasn't called the timer tool in this reply, so any count down is
    his guess; `told`: what he's been given this turn (the crew's line, tools' answers)."""
    if sentence.strip().strip(".!").lower() in stray:  # Gemma's lone "thought"
        return STRAY
    if TOOL_MARKUP.search(sentence):
        return MARKUP
    if counting and COUNTDOWN.search(sentence) and TIME_ASKED.search(user):
        return COUNTED  # (unasked, it isn't said: speak.py)
    if told is not None and can_clock and TIME_ASKED.search(user) and guessed_time(sentence, told):
        return TIMED  # (unasked, it isn't said: speak.py)
    if not clocked and can_clock and NO_CLOCK.search(sentence):
        return NO_CLOCK_SAID
    if not recalled and can_recall and CLAIMED_RECALL.search(sentence):
        return CLAIMED
    if not recalled and can_recall and MEMORY_ASKED.search(user) and FORGOT.search(sentence):
        return FORGOTTEN
    # Asked for research, or answering his search-or-report question ("go on then, write it
    # up"): he said he'd started a report and called nothing, four times in six. A question
    # isn't a promise: "shall I have a proper look while I doze?" is him asking which the
    # crew want, as he's meant to.
    if (not researched and can_research
            and (RESEARCH_ASKED.search(user) or REPORT_WANTED.search(user) or answering_choice)
            and PROMISED_RESEARCH.search(sentence) and not sentence.rstrip().endswith("?")):
        return PROMISED
    return None


def needs_tool(why: str) -> bool:
    """Is a sentence dropped for this reason (from unsaid) redone with a tool required?"""
    return any(w in why for w in ("recall", "research", "clock", "timer", "tool call"))


def minutes(time: str) -> tuple[set[int], bool]:
    """The times of day a CLOCK_TIME match can mean, in minutes past midnight, and whether
    it's to the minute: "17:05" {1025}; "5:05" {305, 1025}; "4 pm" {960}, not to the minute."""
    t = time.lower().replace(".", "")
    m = re.match(r"(\d{1,2}|[a-z]+)(?::(\d{2}))?\s*(am|pm|o'clock)?", t)
    hour = int(m[1]) if m[1].isdigit() else _HOURS.index(m[1]) + 1
    mins = int(m[2] or 0)
    if m[3] in ("am", "pm"):
        return {(hour % 12 + (12 if m[3] == "pm" else 0)) * 60 + mins}, m[2] is not None
    if hour > 12 or (len(m[1]) == 2 and m[1].startswith("0")):  # the 24-hour clock: "17:05", "07:45"
        return {hour * 60 + mins}, m[2] is not None
    return {(hour % 12) * 60 + mins, (hour % 12 + 12) * 60 + mins}, m[2] is not None


def guessed_time(said: str, told: list[str]) -> bool:
    """Did he give a time of day that nobody gave him (the clock, the weather's sunset, a
    search, the crew, a timer coming due)? Kept as fact in his history, a guess stuck:
    greeted, he said "10:14 PM" at 7:56, and repeated it four times, asked to use the
    clock or not. Each time he gave must be one he was given: any time at all, given a
    turn before, let him count on a minute a reply ("22:15", "22:16", "22:17"), unchecked.
    Given as he'd say it, though: told "17:05" by the clock, or "sunset 16:02", his "5:05
    pm", "five o'clock" and "just after 4 pm" were taken for guesses, and not said, or
    redone and kept marked as wrong. So times are compared as times of day: to the minute
    exactly, an hour alone within NEAR_MIN."""
    heard: set[int] = set()
    for t in CLOCK_TIME.findall(" ".join(told)):
        heard |= minutes(t)[0]
    for t in CLOCK_TIME.findall(said):
        times, exact = minutes(t)
        if exact and not times & heard:
            return True
        if not exact and not any(min(abs(a - b), 1440 - abs(a - b)) <= NEAR_MIN for a in times for b in heard):
            return True
    return False

"""What Holly is sent: every rule about his request, in one place.

His request is his system prompt (persona/system_prompt.md, his tools' rules, his examples,
the date, when he last talked and his notes), then his history as it's sent, not as it's
kept (his replies without what the nets cut, his tools' earlier answers stubbed), then the
crew's line with notes after it. Notes are lines in brackets, for this request only: to
stay in character, a rut to get out of, a memory he might bring up, an offer they passed
over, the tool to call. Their words are in persona/notes.toml, each with why.

prepare() makes a turn's request: the side checks that decide its notes (through
Holly._decide, so a replay gives the answers they gave then), and whether he must call a
tool. The reply loop (app/speak.py) adds a note when it asks again (with_note). What's kept
of his reply, for his history, is kept() here too: his history is what he's sent next."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import re
import time
import tomllib
from pathlib import Path

from . import archive, guards, offers, status, timers
from .console import DIM, RESET
from .persona import API_KEYS, about_the_crew, today, today_date, week_around, when


@dataclasses.dataclass(frozen=True)
class Notes:
    """persona/notes.toml: the lines added to what he's sent."""
    in_character: str
    speaking_up: str
    passed_over: str
    told: str
    worn: str
    memory: str
    taken_up_tool: str
    job_tool: str
    empty: str
    alone: str
    last_talked: str
    notes_heading: str
    examples_heading: str
    example: str


def load_notes(path: Path) -> Notes:
    with open(path, "rb") as f:
        return Notes(**tomllib.load(f))


# Whose answers are stubbed (Notes.told): lookups of the world, which can be looked up again
# and go stale. Not his memory's (recall, status, research), which follow-ups lean on: with
# recall's stubbed, "What was her problem at home?", after recall had found Jo's boiler,
# recalled again with a weaker question and missed it, 3 in 6 (6 in 6 whole).
LOOKUPS = {"weather", "search", "news", "clock", "timer", "spotify"}


# His system prompt ---------------------------------------------------------------


def system_prompt(text: str, extra: str, examples: list[dict], notes: Notes) -> str:
    """His prompt, his tools' rules (`extra`: app/search.md and the like) and his examples,
    inside the prompt, not as fake earlier turns (Notes.examples_heading)."""
    prompt = text.strip()
    if extra:
        prompt += "\n\n" + extra.strip()
    if examples:
        lines = [notes.example.format(user=ex["user"], holly=ex["holly"]) for ex in examples]
        prompt += f"\n\n{notes.examples_heading}\n\n" + "\n\n".join(lines)
    return prompt


def preamble(convo) -> list[dict]:
    """His system message: his prompt, the date (without it he searched for last year's
    results), when he last talked, and his notes on anything older than the history. All at
    its end, which changes only when one of them does (his notes only while he sleeps), so
    LM Studio's cache survives until then. Made again only then (kept on `convo`)."""
    key = (today(), convo.memory, convo.previous_end)
    if convo._made_for != key:
        convo._made_for = key
        content = f"{convo.system_prompt}\n\nToday is {key[0]}. {week_around(today_date())}"
        if convo.previous_end:
            content += convo.notes.last_talked.format(when=when(convo.previous_end))
        if convo.memory:
            content += f"\n\n{convo.notes.notes_heading}\n{convo.memory}"
        convo.preamble = [{"role": "system", "content": content}]
    return convo.preamble


# His history, as it's sent ---------------------------------------------------------


def history_as_sent(history: list[dict], clean: bool, notes: Notes) -> list[dict]:
    """His history as he's sent it: only the keys LM Studio takes; and, `clean` (the nets
    acting), his replies without their offers of more help (guards.OFFER), so he doesn't copy
    them, older ones included, and his tools' earlier answers, if they're lookups of the
    world (LOOKUPS), as Notes.told."""
    sent = [{k: m[k] for k in API_KEYS if k in m} for m in history]
    told: list[str] = []
    tools: set[str] = set()
    called = {c.get("id"): c["function"]["name"] for m in sent for c in m.get("tool_calls") or []}
    crew = crew_before = False  # this exchange about the crew of Red Dwarf; the one before
    for m in sent:
        if m["role"] == "user":
            told, tools, crew_before = [m.get("content") or ""], set(), crew
        elif m["role"] == "tool":
            told.append(m.get("content") or "")  # whole: a time he gives is checked against it
            # Never a status report's, whatever tool gave it: said as written, it's what he's told
            # to do if asked more (status.AFTER), and how an earlier one is found
            # (status.without_earlier_reports).
            if (clean and called.get(m.get("tool_call_id")) in LOOKUPS
                    and not told[-1].startswith(status.SAID + status.REPORTS)):
                m["content"] = notes.told
        elif m["role"] == "assistant" and m.get("tool_calls"):
            tools |= {c["function"]["name"] for c in m["tool_calls"]}
        if m["role"] == "assistant" and m.get("content") and clean:
            # And, unasked, without timers counted down or times of day nobody gave him:
            # with six in his history, a minute on each reply, he went on counting,
            # whatever the time. Nor the crew of Red Dwarf tacked on to something else:
            # copying those, he tacked them on to every reply (guards.CREW_ASIDE).
            unasked = not guards.TIME_ASKED.search(told[0] if told else "")
            crew = about_the_crew(told[0] if told else "", tools)
            m["content"] = guards.without_offers(m["content"], countdowns=unasked, told=told if unasked else None,
                                                 asides=not (crew or crew_before))
    return sent


def worn_phrases(history: list[dict], replies: int = 8, at_most: int = 4) -> list[str]:
    """Openings (three words) and phrases (four) in more than one of his last few replies, in
    `history` as he's sent it (history_as_sent), so what's cut from it isn't quoted back to
    him ("convinced the hangers are", from crew asides cut). Not status reports: said as
    written, they're not his wording, and "Lately you've said 'the cat refused to'" would
    only have him on about the crew again."""
    said = [m["content"] for i, m in enumerate(history) if m["role"] == "assistant" and m.get("content")
            and not (i and history[i - 1]["role"] == "tool"
                     and (history[i - 1].get("content") or "").startswith(status.SAID))][-replies:]
    openings: dict[str, int] = {}
    phrases: dict[str, int] = {}
    for text in said:
        words = re.findall(r"[a-z']+", text.lower())
        if len(words) >= 3:
            openings[" ".join(words[:3])] = openings.get(" ".join(words[:3]), 0) + 1
        # Each once a reply, in the order said: phrases that tie are then listed in that order.
        # From a set, they came in the order of Python's string hashes, new each process: the
        # same history gave "a"; "b" one time and "b"; "a" the next.
        for p in dict.fromkeys(" ".join(words[i:i + 4]) for i in range(len(words) - 3)):
            phrases[p] = phrases.get(p, 0) + 1
    worn = [p for p, n in openings.items() if n > 1]
    worn += [p for p, n in sorted(phrases.items(), key=lambda x: -x[1]) if n > 1 and p not in worn
             and not any(p in w or w in p for w in worn)]
    return worn[:at_most]


def messages_for(convo, user_text: str | None = None, taken_up: bool | None = None,
                 memory: str = "") -> list[dict]:
    """The full message list to send for a new user line. The time isn't here: he asks the
    clock tool for it (app/clock.py). Without `user_text`, just the preamble and history.
    After `user_text`, notes: to stay in character (Notes.in_character, or speaking_up for a
    note of his own), if he's fallen into a rut (worn_phrases), that he has, and if they've
    passed over what he offered (`taken_up` False, app/offers.py), to leave it; and a
    `memory` offered (memory_for), to bring up if it fits. They're in this request only: not
    kept in his history, and being last, they leave LM Studio's cache of the rest."""
    notes = convo.notes
    history = history_as_sent(convo.history, convo.clean, notes)
    tail = []
    if user_text is not None:
        note = notes.speaking_up if user_text.startswith("[") else notes.in_character
        if worn := worn_phrases(history):
            user_text += "\n" + notes.worn.format(phrases="; ".join(f'"{p}"' for p in worn))
        if memory:
            user_text += f"\n[{memory}]"
        if taken_up is False:
            note += f"\n{notes.passed_over}"
        tail = [{"role": "user", "content": f"{user_text}\n{note}"}]
    return [*preamble(convo), *history, *tail]


def with_note(messages: list[dict], asked: int, note: str) -> list[dict]:
    """`messages` with `note` after the crew's line (messages[asked - 1]); the rest as it was."""
    last = messages[asked - 1]
    return [*messages[:asked - 1], {**last, "content": last["content"] + "\n" + note}, *messages[asked:]]


# What's been said, for the side checks ------------------------------------------------


def last_exchange(history: list[dict], exchanges: int = 2, chars: int = 600) -> str:
    """The last couple of exchanges, the crew's lines and his replies, for judging a
    follow-up. Two, not one: after "Human Toes" / "I've already given you the news",
    "I just wanted to know more about the toe eating" was taken for a memory, the toe
    eating being his own made-up news a turn earlier. A long reply keeps its start and
    its end, where that story was."""
    said: list[str] = []
    users = 0
    for m in reversed(history):
        if m["role"] not in ("user", "assistant") or not m.get("content"):
            continue
        text = " ".join(m["content"].split())
        said.append(text if len(text) <= chars else f"{text[:chars // 2]} ... {text[-chars // 2:]}")
        if m["role"] == "user" and (users := users + 1) == exchanges:
            break
    return " / ".join(reversed(said))


def last_tools(history: list[dict]) -> list[str]:
    """The tools his last reply came from: those called since the crew last spoke."""
    used: list[str] = []
    for m in reversed(history):
        if m["role"] == "user":
            break
        used += [c["function"]["name"] for c in m.get("tool_calls") or [] if c["function"]["name"] not in used]
    return used[::-1]


def last_reply(history: list[dict]) -> str:
    return next((m.get("content") or "" for m in reversed(history)
                 if m["role"] == "assistant" and m.get("content")), "")


def taken_up(holly, user: str) -> bool | None:
    """Are the crew taking him up on something he offered or asked about doing in his last
    reply ("Want me to search for it now or later?" / "You can get it now if you like")?
    As the model judges (app/offers.py): True, or False, passed over. None, nothing to
    judge: that reply offered and asked nothing (guards.ASKED_OR_OFFERED)."""
    if not holly.tools or user.startswith("["):
        return None
    said = last_reply(holly.convo.history)
    if not guards.ASKED_OR_OFFERED.search(said):
        return None
    return offers.taken_up(holly.lm, holly.model, said, user)


async def memory_for(holly, user: str) -> str:
    """A memory to offer him with `user`: what the crew said on another day about the same
    thing (archive.memory_offered), as a note for his request (Notes.memory); or "". Not for
    a line too short to be about anything, nor more than once every `offer_every` replies,
    nor the same memory twice; and dropped, so as not to hold up his reply, if not found in
    `offer_ms` (one short question to the model: it ends soon after anyway)."""
    cfg = holly.cfg.recall
    if (not (cfg.offer_memories and holly.archive) or user.startswith("[") or len(user.split()) < 4
            or holly.turns - holly.offered_turn < cfg.offer_every):
        return ""

    def find() -> tuple[int, str, str] | None:
        found = [m for m in holly.archive.memories(user) if m[0] not in holly.offered]
        n = archive.memory_offered(holly.lm, holly.model, user, found)
        return next((m for m in found if m[0] == n), None)

    t0 = time.perf_counter()
    try:
        got = await asyncio.wait_for(asyncio.to_thread(find), cfg.offer_ms / 1000)
    except asyncio.TimeoutError:
        print(f"{DIM}  (a memory to offer: not found in {cfg.offer_ms} ms){RESET}", flush=True)
        return ""
    except Exception as e:  # LM Studio busy or gone: he answers as he would have
        print(f"{DIM}  (a memory to offer: {e!r}){RESET}", flush=True)
        return ""
    if not got:
        return ""
    n, day, said = got
    holly.offered.add(n)
    holly.offered_turn = holly.turns
    print(f"{DIM}  (a memory offered, {(time.perf_counter() - t0) * 1000:.0f} ms: {day}: {said[:80]}){RESET}",
          flush=True)
    return holly.convo.notes.memory.format(when=f"on {day} " if day else "", said=said)


# A turn's request -------------------------------------------------------------------


@dataclasses.dataclass
class Request:
    messages: list[dict]
    asked: int  # messages[:asked] are what he's sent before he answers; the crew's line last
    choice: str | None  # "required": he must call a tool
    required: str | None  # what to send for one, if the model takes it (cfg.llm.tool_choice)
    taken_up: bool | None
    held: bool  # answering his search-or-report question
    counting: bool  # timers going as it starts
    memory: str  # a memory offered, "" if none


async def prepare(holly, user: str) -> Request:
    """The request for `user`, and whether he must call a tool. The side checks go through
    holly._decide: a replay gives the answers they gave then."""
    notes = holly.convo.notes
    # Timers going, as the turn starts: a time he gives may be a countdown (guards.unsaid).
    # Only a tool call (then "timer" is in called) changes them mid-turn.
    counting = await holly._decide("counting", lambda: bool(holly.timers and holly.timers.items))
    # Their answer to his search-or-report question (the research tool held a start and had
    # him ask), or their taking him up on something he offered (app/offers.py): either way,
    # what they chose is theirs, and research isn't held for asking which. Or carried on past
    # it: then his request says to leave it (Notes.passed_over).
    held = await holly._decide("held", lambda: bool(holly.research_tool
                                                    and holly.research_tool.held_turn == holly.turns - 1))
    took_up, memory = await asyncio.gather(holly._decide("taken_up", taken_up, holly, user), holly._offer(user))
    messages = messages_for(holly.convo, user, taken_up=took_up, memory=memory)
    asked = len(messages)
    # Taken up: which tool, from what he offered (offers.offered_tool), named in his request,
    # as "required" alone can't: left to choose, a search taken up started a report.
    if took_up and (tool := await holly._decide(
            "offered_tool", offers.offered_tool, holly.lm, holly.model, last_reply(holly.convo.history), user,
            set(holly.toolbox.by_name), last_tools(holly.convo.history))):
        messages = with_note(messages, asked, notes.taken_up_tool.format(tool=tool))
    # "required": the reply must call a tool. So on a retry, and when the crew have just
    # chosen a report in answer to his search-or-report question: then he said "I'll get
    # cracking", "I'll get stuck into it", in ever new words, and started nothing. And
    # when he's asked for something from the past, as the model judges it
    # (archive.about_the_past): "do you remember Jo?", "what did we do yesterday?",
    # follow-ups. Asked from his notes alone, he said "I don't have anything in my notes
    # about a problem", with it in his archive; with a recall a turn earlier to riff on,
    # he guessed what the cats thought of Lister ("a walking disaster area"). (If the
    # model honours it: else a retry is just asked again.)
    toolbox = holly.toolbox
    required = "required" if holly.cfg.llm.tool_choice else None
    choice = (required if took_up
              or (held and "research" in toolbox and guards.REPORT_WANTED.search(user)) else None)
    # A status report, with one already in his history: he gave one without the tool,
    # three times in three, made up or the last one again; with none, he called it
    # every time. So he must call a tool, and the earlier reports are stubbed before he
    # sees them, not only once he's called it.
    if (not user.startswith("[") and "status" in toolbox and guards.STATUS_ASKED.search(user)
            and await holly._decide("wants_report", status.wants_report, holly.lm, holly.model, user)):
        messages = status.without_earlier_reports(messages, asked - 1)
        choice = choice or required
    # A report asked for in so many words (guards.REPORT_ASKED): well into a talk, he
    # asked "a quick search now, or a proper report?" anyway, calling nothing. With
    # something to report on (guards.reported_on): "write me a report", alone, isn't.
    if (not user.startswith("[") and "research" in toolbox and guards.REPORT_ASKED.search(user)
            and guards.reported_on(user)):
        choice = choice or required
    # A timer to do something with a tool ("check the weather in Keswick"), not just say,
    # and which tool, named in his request (timers.job_tool).
    if (user.startswith("[") and (labels := [label for label in guards.DUE_LABEL.findall(user)
                                             if guards.ACTION_WORDS.search(label)])
            and await holly._decide("to_do", timers.to_do, holly.lm, holly.model, labels)):
        choice = choice or required
        if tool := await holly._decide("job_tool", timers.job_tool, holly.lm, holly.model, labels[0],
                                       set(toolbox.by_name)):
            messages = with_note(messages, asked, notes.job_tool.format(tool=tool))
    # Which tool is his to choose: made to call one for a memory, he chose recall every
    # time (25 of 25 in the evals). A web search turned into recall only ever caught the
    # news, judged a memory by mistake.
    if required and not choice and "recall" in toolbox and not user.startswith("["):
        if await holly._decide("about_the_past", archive.about_the_past, holly.lm, holly.model, user,
                               last_exchange(holly.convo.history)):
            choice = required
            if memory:  # recall will answer it properly: no memory offered beside it
                last = messages[asked - 1]
                messages = [*messages[:asked - 1], {**last, "content": last["content"].replace(f"\n[{memory}]", "")}]
                memory = ""
    return Request(messages, asked, choice, required, took_up, held, counting, memory)


# What's kept --------------------------------------------------------------------------


def kept(said: str, user: str, steps: list[dict], called: set[str], history: list[dict], clean: bool,
         crew_before: bool, fired) -> str:
    """His reply as it's kept in his history. Cleaned (`clean`), as what wasn't said isn't
    kept either (guards.without_offers); and a time of day he gave, unasked and unchecked,
    marked as a guess (guards.GUESSED), if `fired(guards.GUESS_MARKED)` says to."""
    unasked_countdown = not guards.TIME_ASKED.search(user)
    this_turn = [user] + [m.get("content") or "" for m in steps if m["role"] == "tool"]
    if clean:
        said = guards.without_offers(said, countdowns=unasked_countdown,
                                     told=this_turn if unasked_countdown else None,
                                     asides=not (crew_before or about_the_crew(user, called)))
    # What he was told: this turn, and the last few (asked "you mean 24 minutes past
    # midnight?" a turn after the clock said 00:24, his "00:24" was no guess).
    told = this_turn + [m.get("content") or "" for m in history[-12:]]
    if guards.guessed_time(said, told) and fired(guards.GUESS_MARKED):
        said += guards.GUESSED
    return said

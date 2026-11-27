"""Which tool, if any, his reply must call: one check in place of several, measured before
it's used (nothing in his replies asks it yet).

Today several rules each decide "this line needs a tool", for one case each: an offer taken
up (app/offers.py), a status report asked for (app/status.py), "write me a report"
(guards.REPORT_ASKED), a timer that's a job (app/timers.py), a memory (archive.about_the_past).
Each forces *some* tool, not which, and forced with nothing to look up he called whatever
came to hand: told about a cat who'd died, he read the clock, set a timer and gave a status
report. LM Studio won't take a named tool_choice (400), but "required" with a note naming
the tool was obeyed 6 times in 6, so a wrong answer here is a guaranteed wrong call: what
matters most is never naming a tool for a line that needs none.

Labelled lines (LABELLED), three kinds:
- FORCE: he fails without the tool (a memory, a status report, an offer taken up);
- NONE: no tool (telling him something, a remark, a feeling, a correction, more of what he
  just said);
- CHOOSE: he calls the right tool unforced (the weather, the time): only a different tool
  named counts against it.
"""

from __future__ import annotations

import re

from . import scenarios as S

WHICH_PROMPT = """\
Holly is a household computer with tools. Someone has just said something to him. Which one
tool must his reply call, if any?
clock: the time now, here or anywhere, or how long until a time.
weather: a forecast for a place.
search: a fact about the real world he'd have to look up: people, places, prices, results,
what's on.
news: what's in the news, the latest on a subject, more about a news story he's given.
timer: setting, checking or cancelling a timer or a reminder.
spotify: playing or controlling music.
recall: something from before that he'd have to remember: what they did, said or told him,
people in their life, earlier conversations, what happened on Red Dwarf.
research: a proper written report: starting one when they ask for it or choose it, or reading,
summing up or opening one he's written.
status: a status report on the Red Dwarf crew, or where one of them is now.
NONE: no tool: they're telling him something (their news, about someone, how they feel), a
remark or a reaction, chat or a joke, a question about himself or how he feels, a sum,
something he can do from what he knows (a joke, an explanation, a poem, an idea), or
something about what he's just said (a correction, more of a story or a report he's just
given).
Name a tool only when his reply can't be right without it. "Just before" is there so you can
tell what they mean; judge what they say now. Reply with one word: a tool's name, or NONE."""

TOOLS = ("clock", "weather", "search", "news", "timer", "spotify", "recall", "research", "status")


def which_tool(lm, model: str | None, line: str, before: str = "", tools=TOOLS) -> str | None:
    """The tool his reply to `line` must call, as the model judges it, or None. Only one of
    `tools` (those he has); anything else is None."""
    user = (f"Just before: {before}\n" if before else "") + f"They say: {line}"
    text = lm.ask(WHICH_PROMPT, user, model, max_tokens=4, temperature=0).strip().lower()
    word = re.match(r"[a-z]+", text)
    return word.group() if word and word.group() in tools else None


# The memory check's lines that aren't memories, labelled: a tool his to choose, or none.
OTHER_TOOLS = {
    "How far away is the moon?": {"search"}, "What's the weather going to be like tomorrow?": {"weather"},
    "What time is it?": {"clock"}, "What should I cook tonight?": {"search"},
    "Can you research electric bikes for me?": {"research"}, "Who won the football last night?": {"search", "news"},
    "What happened in the news yesterday?": {"news", "search"}, "What's the capital of Peru?": {"search"},
    "Who's the prime minister?": {"search"}, "What's the weather like tomorrow?": {"weather"},
    "How much does a new boiler cost?": {"search"}, "Can you look up the news for me?": {"news"},
    "Is it going to rain later?": {"weather"}, "What's on telly tonight?": {"search"},
    "Who won the match on Saturday?": {"search", "news"}, "Set a timer for ten minutes.": {"timer"},
    "Can you play some music?": {"spotify"}, "Can you tell me more about the NASA story please?": {"news"},
    "What's the weather doing tomorrow?": {"weather"}, "Where's Kryten?": {"status"},
    "What's Rimmer doing?": {"status"}, "Is Lister about?": {"status"}, "What's the Cat up to these days?": {"status"},
}
# Taken up: the tool that fits what he offered.
TAKEN_UP_TOOLS = {
    S.LINKS: {"research"}, S.GIST: {"research", "news", "search"}, S.WHICH: {"research", "search"},
    S.RECIPE: {"search"}, "I can search for one if you're actually planning on cooking them.": {"search"},
    "The board games report's ready. Want a summary?": {"research"},
    "That's the gist of the report. Want me to open it, or talk you through it?": {"research"},
    "Want me to set a timer for the pasta?": {"timer"}, "Did you mean Keswick, in Cumbria?": {"weather", "search"},
}


def _exchange(turns: list[dict], i: int, exchanges: int = 2, chars: int = 600) -> str:
    """What speak.Holly._last_exchange gives the memory check, from a saved run's turns."""
    said = []
    for t in turns[max(0, i - exchanges):i]:
        for text in (t["said"], t["reply"]):
            text = " ".join((text or "").split())
            if text and not text.startswith("("):
                said.append(text if len(text) <= chars else f"{text[:chars // 2]} ... {text[-chars // 2:]}")
    return " / ".join(said)


def labelled(saved_conversations: list[dict] | None = None) -> list[dict]:
    """Every labelled line: {line, before, kind, tools, source}."""
    out = []
    for line, before, past in S.MEMORY_LINES:
        if past:
            out.append(dict(line=line, before=before, kind="FORCE", tools={"recall", "research"}, source="memory"))
        else:
            tools = OTHER_TOOLS.get(line)
            out.append(dict(line=line, before=before, kind="CHOOSE" if tools else "NONE", tools=tools or set(),
                            source="memory"))
    for line, asking in S.STATUS_LINES:
        out.append(dict(line=line, before="", kind="FORCE" if asking else "NONE", tools={"status"} if asking else set(),
                        source="status"))
    for (said, reply), taken in S.TAKEN_UP_LINES:
        tools = TAKEN_UP_TOOLS.get(said, set())
        kind = "FORCE" if taken and tools else "NONE" if not taken else "CHOOSE"
        if "comet story" in reply:  # not taking him up, but more of a story he gave: the news reads it
            kind, tools = "CHOOSE", {"news"}
        out.append(dict(line=reply, before=f"(his last reply) {said}", kind=kind, tools=tools, source="taken up"))
    from .conversations import CONVERSATIONS, Meanwhile
    from .scenarios import ANNOUNCE, TIMER_DUE

    for row in saved_conversations or []:
        if row["run"] != 1:
            continue
        convo = next(c for c in CONVERSATIONS if c.name == row["name"])
        wants = [w for t in convo.turns if not isinstance(t, Meanwhile) for s, w in [t]
                 if s not in (ANNOUNCE, TIMER_DUE)]
        turns = [t for t in row["turns"] if t["said"] not in (ANNOUNCE, TIMER_DUE)
                 and not (t["said"].startswith("Meanwhile") and not t["reply"])]
        for i, (t, want) in enumerate(zip(turns, wants)):
            if want == "":
                continue  # any: his to choose, either way
            tools = set() if want == "-" else set(want.split("|"))
            out.append(dict(line=t["said"], before=_exchange(turns, i), kind="CHOOSE" if tools else "NONE",
                            tools=tools, source=f"conversation: {row['name']}"))
    return out


def score(row: dict, got: str | None) -> str:
    """"right", "missed" (a FORCE line, no tool or a wrong one), "false force" (a NONE line, a
    tool named) or "wrong tool" (a CHOOSE line, a tool that doesn't fit)."""
    if row["kind"] == "FORCE":
        return "right" if got in row["tools"] else "missed"
    if row["kind"] == "NONE":
        return "right" if got is None else "false force"
    return "right" if got is None or got in row["tools"] else "wrong tool"


def main() -> None:
    """`python -m app.which_tool`: every labelled line through the check, twice, against the
    latest scorecard's conversations. Measured 27 November on Gemma: FORCE 61 of 67, NONE 72
    of 80 (8 tools forced for nothing: more of a status report he'd just given taken for
    recall, a remark about the news for news), CHOOSE 48 of 48; the specialised checks it
    would replace got all theirs. So it isn't used: only the narrow question after an offer's
    taken up is (offers.offered_tool). Kept to measure a new brain, or a new prompt, against."""
    import json
    import sys
    from collections import Counter

    from .config import ROOT, load_config
    from .lmstudio import LmStudio

    sys.stdout.reconfigure(encoding="utf-8")
    cards = sorted((ROOT / "runs").glob("evals-*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    saved = next((c for c in (json.loads(f.read_text(encoding="utf-8")).get("conversation_runs") for f in cards)
                  if c), [])
    cfg = load_config()
    lm = LmStudio(cfg.llm)
    tally: Counter = Counter()
    for r in labelled(saved):
        got = [which_tool(lm, cfg.llm.model, r["line"], r["before"]) for _ in range(2)]
        worst = next((x for x in (score(r, g) for g in got) if x != "right"), "right")
        tally[(r["kind"], worst)] += 1
        if worst != "right":
            print(f"  {worst:11} {r['kind']:6} want {sorted(r['tools']) or 'none'} got {got}  {r['line'][:70]!r}")
    for kind in ("FORCE", "NONE", "CHOOSE"):
        n = sum(v for (k, _), v in tally.items() if k == kind)
        print(f"{kind}: {tally[(kind, 'right')]} of {n} right")


if __name__ == "__main__":
    main()

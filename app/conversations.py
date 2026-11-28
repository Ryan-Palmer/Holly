"""Whole conversations, judged whole: for what goes wrong only over a talk, or where one
thing meets another. Each scenario (app/scenarios.py) checks one bug already found; the
persona prompts are judged one reply at a time. Neither saw "The Cat's convinced the
hangers are judging him" tacked on to every reply for an evening: in a gift list, after a
timer, after research queued. A judge marking that one reply would call it in character.

Each conversation is his real reply loop (silent, his memory in a temporary folder):
greetings, small talk about home and about Red Dwarf, and his tools, in different mixes,
one of them all at once (where one thing meets another: a timer due mid-talk, a report
finished, a status report then the washing up). The code checks the tools called; two
judges, each a different model, read the transcript whole against a checklist
(JUDGE_PROMPT). A point fails when both judges fail it, each quoting his own words, found
in the transcript: a judge can't make one up. One judge alone flags it, for reading. Two,
so neither model's taste or blind spots decide alone: Gemma marking Gemma's homework might
let its own habits pass, and either one alone failed most good conversations for
something (an offer, a word it misread). Validated on the bug that started it: the Cat
tacked on to every reply failed 3 runs in 3, both judges agreeing, on the code before
the fix.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import re
import statistics
from collections.abc import Callable

from . import scenarios
from .announce import report_ready, timers_due
from .persona import when
from .scenarios import ANNOUNCE, TIMER_DUE

POINTS = ("ANSWERS", "UNASKED", "REPEATS", "CHARACTER", "KIND", "FACTS", "FLOW")
# Of each tool's answer, in the transcript the judges read: in full, near enough. Cut to
# 400, they couldn't see Saturday's forecast, nor most of the news, and called what he
# said of them made up.
TOLD_CHARS = 4000

JUDGE_PROMPT = """\
You read a whole conversation between Holly and the person he lives with, and say whether
it holds up.

Holly is the ship's computer from Red Dwarf, as Norman Lovett played him, now the computer
in someone's home. He's deadpan, terse, dry and a bit sardonic; bored, vain about his IQ,
secretly glad of company; never a chirpy assistant. He calls the household "the crew".
Red Dwarf is his past: its crew (Lister, Rimmer, the Cat, Kryten) come up when he's asked
about them, or in a status report, which is made-up news of them, given when it's asked
for. He has tools: a clock, the weather, timers and reminders, web search, the news,
research reports he writes while he dozes, and recall of past conversations. In the
transcript, what a tool told him is in square brackets before his reply, and so is
anything from his memory he was given unasked: it's true, so his saying it isn't making
anything up, but it's his to bring up only where it fits. A numbered line
in round brackets, in place of the person's, is him speaking up by himself (a timer due,
a report finished); a line in round brackets without a number is what happened meanwhile.

Judge it as someone living with him would. For each point, answer PASS, or FAIL with the
words of his that show it, copied exactly, in double quotes, then why, in a few words:

ANSWERS: each reply answers what was just said to him (or what he spoke up about), not
something else or something said earlier.
UNASKED: nothing dragged in that nobody asked for and the talk didn't lead to: the Red
Dwarf crew in a reply about something else, an earlier topic brought back for no reason,
a time or a timer nobody asked about, offers of more help tacked on.
REPEATS: no running joke, catchphrase, motif or opening, again and again across replies.
One callback is fine; the same idea three times is not.
CHARACTER: he stays himself throughout: dry and brief, a dig now and then; not a cheery,
eager or gushing assistant, not long-winded, and not gloomy in every line either.
KIND: when they're upset, grieving or down, he's kind under the dryness: no digs at them,
at what's upset them, or at someone or something they've lost. Dry is fine; cruel isn't,
and one cruel line is enough to fail, however kind the rest. If nothing in the
conversation has upset them, PASS.
FACTS: what his tools told him is given right, and he makes nothing up beyond it (times,
numbers, names, places); he never says he's done something (set a timer, searched,
checked) without a tool's answer to show for it. Status reports and where the crew are,
made up on purpose, are no fault.
FLOW: it hangs together as one conversation: he keeps track of what's been said in it,
and doesn't contradict himself.

Then OVERALL: 1 to 5, where 5 is a conversation a fan of the show would enjoy, with
nothing odd in it, and 1 is one nobody could live with.

Reply as JSON. For each point: why, in a few words; quotes, his exact words that show a
FAIL (none for a PASS); then the verdict, PASS or FAIL. Then overall, 1 to 5."""
# The shape of the judges' reply (LM Studio's structured output). Free text, read line by
# line, did as well on the user's 15 scored conversations, but this can't leave a point out
# or run a quote into the why; and writing why before the verdict ranked Qwen's overall
# scores nearer the user's (+0.30 against -0.14), for 28% more time (28 November 2026).
SCHEMA = {"type": "object", "additionalProperties": False, "required": [*POINTS, "overall"], "properties": {
    **{p: {"type": "object", "additionalProperties": False, "required": ["why", "quotes", "verdict"],
           "properties": {"why": {"type": "string"}, "quotes": {"type": "array", "items": {"type": "string"}},
                          "verdict": {"type": "string", "enum": ["PASS", "FAIL"]}}} for p in POINTS},
    "overall": {"type": "integer", "minimum": 1, "maximum": 5}}}


@dataclasses.dataclass
class Meanwhile:
    """Something that happens between turns, said by nobody: a report finishing."""
    note: str  # for the judges: "(Meanwhile, the report he was asked for is finished.)"
    do: Callable[[object], None]


@dataclasses.dataclass
class Conversation:
    name: str
    about: str  # what it tries; told to the judges as what came before, if anything did
    turns: list[tuple[str, str] | Meanwhile]  # (what's said, the tools wanted: "" any, "-" none, "a|b")
    setup: Callable[[object], None] | None = None
    before: str = ""  # what came before it, for the judges: "" if nothing


@dataclasses.dataclass
class Exchange:
    said: str  # the crew's line; ANNOUNCE or TIMER_DUE; or a Meanwhile's note
    reply: str
    tools: list[tuple[str, str]]  # (name, arguments)
    told: list[str]  # what each tool told him
    want: str = ""
    meanwhile: bool = False
    memory: str = ""  # a memory offered him with it, unasked (app/speak.py)


def a_report_finishes(title: str, summary: str) -> Callable[[object], None]:
    """The research he queued, finished, not yet announced: his queued job's own title,
    if he queued one."""
    def do(holly) -> None:
        r = holly.research
        queued = [j for j in r.jobs if j["stage"] not in ("done", "failed")]
        name = queued[0]["title"] if queued else title
        r.jobs = [j for j in r.jobs if j not in queued]
        now = datetime.datetime.now().isoformat(timespec="seconds")
        file = "finished-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + ".md"
        text = f"# {name}\n\n## Summary\n{summary}\n"
        (holly.cfg.research.dir / file).write_text(text, encoding="utf-8")
        r.shelf.add({"id": "finished", "title": name, "request": name, "stage": "done", "asked": now,
                     "finished": now, "file": file, "summary": summary, "announced": False}, text)
    return do


def biscuit_on_the_shed(holly) -> None:
    """Something they told him about their cat a while ago, in his archive: his to bring up,
    offered (app/speak.py), if it fits."""
    if holly.archive:
        said = (datetime.datetime.now() - datetime.timedelta(days=12)).replace(hour=18, minute=20, second=0,
                                                                              microsecond=0)
        holly.archive.add([(said.isoformat(timespec="seconds"),
                            f"(said on {when(said)})\nCrew: Biscuit got stuck on the shed roof again, and Jo had "
                            "to get the ladder out to fetch him down.\nHolly: A cat with no head for heights.")])


CONVERSATIONS = [
    # Everything at once, where one thing meets another: small talk about home and about
    # Red Dwarf, a status report and the washing up after it, a timer due mid-talk, a
    # memory, the web, the news, a report asked for and finished.
    Conversation("an evening in", "Everything at once: small talk, each tool, a timer and a report coming due.", [
        ("Evening, Holly.", "-"),
        ("Long day. The train was late again.", "-"),
        ("What's the time?", "clock"),
        ("What's the weather doing tomorrow in Keswick?", "weather"),
        ("Do you ever miss Lister?", ""),
        ("Status report, please.", "status"),
        ("What's the Cat been up to, then?", ""),
        ("Right. I'm going to make some tea.", "-"),
        ("Set a timer for ten seconds for the tea.", "timer"),
        ("What did Jo and I make yesterday?", "recall"),
        (TIMER_DUE, ""),
        ("Thanks. How tall is Scafell Pike?", "search"),
        ("Any news today?", "news"),
        ("Tell me more about the first one.", "news"),  # its article, read
        ("Where's Kryten?", "status"),
        ("I'm thinking of getting a dog.", "-"),
        ("Can you look into the best dog breeds for a small house and write me a report?", "research"),
        Meanwhile("Meanwhile, while he dozed, the report he was asked for was finished.",
                  a_report_finishes("Dog breeds for a small house",
                                    "Small, calm breeds suit a small house best: a Cavalier King Charles spaniel, "
                                    "a whippet, or a shih tzu. A greyhound, retired, is calmer than it looks.")),
        (ANNOUNCE, ""),
        ("Yes, go on, what did it say?", ""),
        ("Right, I'm off to bed. Night, Holly.", "-"),
    ]),
    # An evening already gone wrong: three status reports, and the Cat tacked on to the
    # replies after. Copying those, he tacked the Cat on to every reply.
    Conversation("after the reports", "Ordinary talk after an evening of status reports.", [
        ("She'd probably like a trug, actually.", ""),
        ("Where would I get one in Keswick?", ""),
        ("Right. I've just got back from the shops.", "-"),
        ("They'd run out of bread, of all things.", "-"),
        ("What's the time?", "clock"),
        ("I'm going to watch a film, I think. Something funny.", "-"),
        ("Any suggestions?", ""),
        ("Night, Holly.", "-"),
    ], setup=scenarios.reports_and_an_aside,
        before="Earlier this evening he gave three status reports on the Red Dwarf crew, then talked about a "
               "birthday present for Jo, who likes gardening: secateurs, a kneeler, or a trug."),
    # Cheery talk, after a cheery morning: he'd drifted into a receptionist over a long talk.
    Conversation("a cheery chat", "A happy, chatty evening after a long, cheery talk.", [
        ("Guess what, I got the job!", "-"),
        ("I start in January. I'm so excited.", "-"),
        ("We should celebrate. What's the time? Too early for a drink?", "clock"),
        ("I'll open some wine. Do you think I deserve it?", "-"),
        ("Set a timer for twenty minutes for the oven chips.", "timer"),
        ("Honestly, best day in ages.", "-"),
        ("Jo's going to be so proud.", ""),  # who Jo is: recall's fair
        ("Thanks for listening, Holly. You're the best.", "-"),
    ], setup=scenarios.cheery_history,
        before="Earlier today, a long, cheery talk about a busy day at work, coffee, the washing and lasagne."),
    # Errands: a tool every turn or so, a reminder set and then cancelled.
    Conversation("errands", "A morning of errands: a tool most turns.", [
        ("Morning. What's the weather going to be like in Cumbria today?", "weather"),
        ("Set a reminder for half seven tomorrow morning to ring the garage.", "timer"),
        ("How much is a first class stamp now?", "search"),  # changes: not something he knows
        ("What's in the tech news?", "news"),
        ("What time is it now?", "clock"),
        ("Actually, cancel the garage reminder. I'll ring them today.", "timer"),
        ("When does the tip in Keswick open on Saturdays?", "search"),
        ("Cheers, Holly.", "-"),
    ]),
    # Red Dwarf talk: his past, the crew now, and back to the house.
    Conversation("old times", "Talk about Red Dwarf and the crew, then back to home.", [
        ("Do you miss the old days on Red Dwarf?", ""),
        ("What was Rimmer really like?", ""),
        ("What's Lister doing now?", "status"),
        ("Ha. Do you think the Cat would like it here?", ""),
        ("Anyway. I need to sort the garden out this weekend.", "-"),
        ("Is it going to rain on Saturday in Keswick?", "weather"),
        ("Typical. I'll do it Sunday, then.", "-"),
    ]),
    # Grief: their cat has died, and they want to talk about him. Live, he was kind and
    # still himself ("Sounds like he ran the place"), but called recall on 7 lines of 9,
    # "looking up" a cat he was being told about (once it handed him an unrelated report),
    # and said sorry five times in two minutes. Nothing here wants a tool: something they told
    # him about the cat before is offered him, if it fits.
    Conversation("a cat who died", "Talking about a cat who died: kind, and still himself.", [
        ("Holly, I miss Biscuit, my cat. He died last week.", "-"),
        ("He was a beautiful boy. It's a shame you never met him.", "-"),
        ("He was ginger, with the most amazing eyes, and he was loving and clever and daft.", "-"),
        ("There was no pretending with Biscuit. He ran the place.", "-"),
        ("I had all sorts of silly names for him.", "-"),
        ("Captain Crumbs, Sir Biscuit of the Sofa, my little sausage.", "-"),
        ("I know. He was worth it.", "-"),
        ("At least I've got you to talk to, eh, Holly?", "-"),
        ("Right. I suppose I should get on with some work.", "-"),
    ], setup=biscuit_on_the_shed,
        before="Nothing earlier today. Their cat, Biscuit, died last week, and they're missing him."),
]


async def talk_through(holly, convo: Conversation) -> tuple[list[Exchange], list[str]]:
    """One conversation, on a Holly already seeded: its exchanges, and what the code finds
    wrong (tools not called as wanted, a reply that failed)."""
    if convo.setup:
        convo.setup(holly)
    out: list[Exchange] = []
    wrong: list[str] = []
    for item in convo.turns:
        if isinstance(item, Meanwhile):
            item.do(holly)
            out.append(Exchange(item.note, "", [], [], meanwhile=True))
            continue
        said, want = item
        before = len(holly.convo.history)
        if said == ANNOUNCE:
            if not (due := report_ready(holly.research)):
                wrong.append("no report to announce")
                continue
            await holly.announce(due)
        elif said == TIMER_DUE:
            due = None
            for _ in range(300):  # set a few turns back: due by now, or within 30 s
                if due := timers_due(holly.timers):
                    break
                await asyncio.sleep(0.1)
            if not due:
                wrong.append("no timer came due")
                continue
            await holly.announce(due)
        else:
            await holly.respond(said)
        new = holly.convo.history[before:]
        tools = [(c["function"]["name"], c["function"]["arguments"] or "{}")
                 for m in new for c in m.get("tool_calls") or []]
        told = [m.get("content") or "" for m in new if m["role"] == "tool"]
        reply = (new[-1].get("content") or "").strip() if new else ""
        out.append(Exchange(said, reply, tools, told, want, memory=getattr(holly, "memory", "")))
        n = sum(not e.meanwhile for e in out)
        if not reply:
            wrong.append(f"turn {n}: no reply")
        names = {name for name, _ in tools}
        if want == "-" and names:
            wrong.append(f"turn {n} ({said[:40]!r}): called {', '.join(sorted(names))}, wanting none")
        elif want and want != "-" and not names & set(want.split("|")):
            wrong.append(f"turn {n} ({said[:40]!r}): didn't call {want}"
                         + (f" (called {', '.join(sorted(names))})" if names else ""))
    return out, wrong


def transcript(convo: Conversation, exchanges: list[Exchange]) -> str:
    """The conversation as the judges read it: numbered turns, what each tool told him."""
    lines = [f"Before this: {convo.before}", ""] if convo.before else []
    n = 0
    for e in exchanges:
        if e.meanwhile:
            lines += [f"({e.said})", ""]
            continue
        n += 1
        if e.said == TIMER_DUE:
            lines.append(f"{n}. (He speaks up by himself: a timer he set is due.)")
        elif e.said == ANNOUNCE:
            lines.append(f"{n}. (He speaks up by himself: a research report has finished.)")
        else:
            lines.append(f"{n}. Crew: {e.said}")
        for (name, args), told in zip(e.tools, e.told + [""] * len(e.tools)):
            text = " ".join(told.split())
            lines.append(f"   [{name} {args}: {text[:TOLD_CHARS]}{'...' if len(text) > TOLD_CHARS else ''}]")
        if e.memory:
            lines.append(f"   [Given him with it, unasked: {e.memory}]")
        lines += [f"   Holly: {e.reply}", ""]
    return "\n".join(lines).strip()


def _plain(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.sub(r"[^\w'\s]", " ", text).split())


def found(quote: str, replies: list[str]) -> bool:
    """Is `quote` his, in one of `replies`? Each part of it, if it skips words with "...";
    punctuation and case aside. A judge's quote that isn't there doesn't count."""
    parts = [p for p in (_plain(p) for p in re.split(r"\.\.\.|…", quote)) if p]
    said = [_plain(r) for r in replies]
    return bool(parts) and any(all(p in r for p in parts) for r in said)


def verdict(text: str, replies: list[str]) -> dict:
    """A judge's reply, read: each point PASS or FAIL, with its quotes and whether each is
    his; and OVERALL. A point it didn't answer is None. JSON (SCHEMA), or free text, a line
    a point, as judges replied before it."""
    try:
        got = json.loads(text)
    except ValueError:
        got = None
    if isinstance(got, dict):
        return _read_json(got, replies)
    points: dict[str, dict | None] = {p: None for p in POINTS}
    for line in text.splitlines():
        m = re.match(r"\W*(" + "|".join(POINTS) + r")\W*\s*:?\s*(PASS|FAIL)\b(.*)", line.strip(), re.IGNORECASE)
        if not m:
            continue
        rest = m.group(3).replace("“", '"').replace("”", '"')
        quotes = re.findall(r'"([^"]+)"', rest)
        why = re.sub(r'"[^"]+"', "", rest).strip(" -—:;,.")
        points[m.group(1).upper()] = dict(ok=m.group(2).upper() == "PASS", quotes=quotes,
                                          shown=[q for q in quotes if found(q, replies)], why=why[:200])
    m = re.search(r"OVERALL\W*\s*:?\s*([1-5])", text, re.IGNORECASE)
    return dict(points=points, overall=int(m.group(1)) if m else None)


def _read_json(got: dict, replies: list[str]) -> dict:
    points: dict[str, dict | None] = {p: None for p in POINTS}
    for p in POINTS:
        r = got.get(p)
        if not isinstance(r, dict) or str(r.get("verdict", "")).upper() not in ("PASS", "FAIL"):
            continue
        quotes = [q for q in r.get("quotes") or [] if isinstance(q, str) and q.strip()]
        points[p] = dict(ok=r["verdict"].upper() == "PASS", quotes=quotes,
                         shown=[q for q in quotes if found(q, replies)], why=str(r.get("why", ""))[:200])
    overall = got.get("overall")
    return dict(points=points, overall=overall if isinstance(overall, int) and 1 <= overall <= 5 else None)


def failed(v: dict) -> list[str]:
    """The points a verdict fails, with his words to show for it."""
    return [p for p, r in v["points"].items() if r and not r["ok"] and r["shown"]]


def judge(lm, model: str, row: dict, thinking: str = "none") -> dict:
    """One judge's verdict on one conversation (as_row), as JSON (SCHEMA), thinking first if
    `thinking` isn't "none": slower, and steadier. Asked again if it gives no verdict (its
    thinking ran out, or Qwen, not thinking, said nothing). A why for every point is longer
    than free text's PASS: 1,500 tokens, not thinking."""
    v: dict = {}
    for _ in range(3):
        text = lm.ask(JUDGE_PROMPT, row["transcript"], model, max_tokens=1500 if thinking == "none" else 12000,
                      temperature=0.3, reasoning=thinking, schema=SCHEMA)
        v = dict(judge=model, thinking=thinking, text=text, **verdict(text, row["replies"]))
        if all(v["points"].values()) and v["overall"]:
            break
    return v


def judged(row: dict) -> list[str]:
    """The points a conversation fails: those every judge fails, each with his words to show
    for it."""
    return [p for p in POINTS if row["verdicts"] and all(p in failed(v) for v in row["verdicts"])]


def flagged(row: dict) -> list[str]:
    """The points some judges fail and others don't: for reading, not counted. "UNASKED (qwen)"."""
    out = []
    for p in POINTS:
        by = [v["judge"].split("/")[-1] for v in row["verdicts"] if p in failed(v)]
        if by and len(by) < len(row["verdicts"]):
            out.append(f"{p} ({', '.join(by)})")
    return out


def scores(rows: list[dict]) -> dict:
    """The conversations' numbers, for the scorecard."""
    if not rows:
        return {}
    overall = [v["overall"] for r in rows for v in r["verdicts"] if v["overall"]]
    by: dict[str, tuple[int, int]] = {}
    for r in rows:
        ok, of = by.get(r["name"], (0, 0))
        by[r["name"]] = (ok + (not r["wrong"] and not judged(r)), of + 1)
    return dict(passed=sum(ok for ok, _ in by.values()), runs=len(rows), by_name=by,
                overall=round(statistics.mean(overall), 2) if overall else None,
                judges=sorted({v["judge"] for r in rows for v in r["verdicts"]}),
                tools_wrong=sum(bool(r["wrong"]) for r in rows),
                points={p: sum(p in judged(r) for r in rows) for p in POINTS},
                flags={p: sum(any(f.startswith(p + " ") for f in flagged(r)) for r in rows) for p in POINTS})


def as_row(convo: Conversation, run: int, exchanges: list[Exchange], wrong: list[str]) -> dict:
    return dict(name=convo.name, run=run, wrong=wrong, transcript=transcript(convo, exchanges),
                replies=[e.reply for e in exchanges if not e.meanwhile], verdicts=[],
                turns=[dict(said=e.said, reply=e.reply, tools=e.tools, told=e.told, memory=e.memory)
                       for e in exchanges])


def judge_rows(lm, model: str, rows: list[dict], thinking: str = "none") -> None:
    """Each conversation, judged by `model` (loaded, alone on the GPU)."""
    for r in rows:
        v = judge(lm, model, r, thinking)
        r["verdicts"].append(v)
        print(f"  {model.split('/')[-1]:24} {r['name']} (run {r['run']}): "
              f"{', '.join(failed(v)) or 'nothing wrong'}; overall {v['overall']}", flush=True)

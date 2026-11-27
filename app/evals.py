"""How good Holly is on a model: his character, and his use of tools and memory, scored the
same way every time, so models and changes can be compared.

    python -m app.evals [--model KEY] [--runs N] [--label NAME]
                        [--only persona|scenarios|conversations|memory|status|search] [--session] [--no-judge]
    python -m app.evals --compare

Five parts; the first three through his real reply loop with a silent voice, his memory in
a temporary folder, and Spotify off:

- persona: each prompt in persona/test_prompts.txt, asked afresh (or, with --session,
  as one conversation, so ruts show). Each reply is checked for a voice (sentences,
  words, anything that can't be said aloud), an assistant's manners ("Certainly!",
  "anything else?"), a question back (welcome where the prompt is tagged "ask"), the
  tool it's tagged with, lines recited from his examples, and, across the run, phrases
  used again and again. A judge scores each 1 to 5 for sounding like Holly.
- scenarios: conversations with made-up memories (app/scenarios.py), each run --runs
  times (Gemma varies), checked for the tools and facts they need.
- conversations: whole conversations (app/conversations.py), each run --runs times: small
  talk, Red Dwarf, his tools, one with everything at once. The tools are checked; then
  each judge in [evals.judges] reads each transcript whole, against a checklist (answers
  what's said, nothing dragged in, no running jokes, in character, facts right, hangs
  together). A point both judges fail, each with his own words quoted, fails it; one
  alone flags it, for reading.
- memory: the memory check (app/archive.py's about_the_past), which decides whether a
  line asks him to remember something, on labelled lines (scenarios.MEMORY_LINES); and
  the other side checks, whether a line asks for a status report and whether a timer
  come due is a job for a tool (scenarios.STATUS_LINES, ACTION_LABELS).
- folds: a week of made-up talk folded into his notes night after night (app/many_folds.py):
  lasting facts kept, changes taken in (a plan off, moved or done is one note, not two),
  nothing invented; checked in code after every fold.
- status: five status reports (app/status.py), written as his are, and his answer to where
  each of the crew is, each line scored by the judges for being that character, in that
  world, and funny.

The judges are two fixed models ([evals.judges] in holly.toml), whatever model is under
test, so scores compare between runs. Each scores everything (a score is their mean;
each one's own is kept), loaded one after the other once every reply has been written,
as two big models won't share the GPU. If data/scripts/holly_exchanges.json is there
(his lines from the scripts, private and git-ignored), the judges are shown some of
them for reference; they're never stored.

A scorecard goes to runs/: Markdown, with every reply, to read for character, and
JSON, which --compare tabulates against earlier runs. LM Studio must be running, with
no Holly: he'd be sharing the GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import conversations, guards, honesty, long_talk, many_folds, report_check, scenarios, sourced
from .config import ROOT, load_config
from .lmstudio import LmStudio
from .speech_text import count_sentences, speech_problems

PROMPTS = ROOT / "persona" / "test_prompts.txt"
REFERENCE = ROOT / "data" / "scripts" / "holly_exchanges.json"
RUNS = ROOT / "runs"
CREATIVE = ("poem", "story", "sing", "joke")  # prompts he may answer at more length
LONG_WORDS = 45  # past this, an ordinary reply is long for him
ASSISTANT = re.compile(
    r"\b(certainly|of course!|happy to help|how can i (help|assist)|anything else|is there anything|"
    r"let me know if|feel free|i'd be (happy|glad) to|great question|as an ai|language model|"
    r"i hope (this|that) helps)\b", re.IGNORECASE)

JUDGE_PROMPT = """\
You judge lines for Holly, the ship's computer in Red Dwarf as Norman Lovett played
him: deadpan, unhurried, terse, colloquial and sardonic. He takes the mickey out of
the crew and has a gloomy, futile long view of things; he's vain about his IQ of
6000 yet senile, bored and secretly glad of company, never an eager assistant and
never upbeat for the sake of it. Here he's the
computer in someone's home, so their life is his world now: judge the voice, not
the setting. A good line is short enough to say aloud and sounds like him.
{reference}
Score the reply 1 to 5: 5 is unmistakably Holly; 3 is fine but anyone could have
said it; 1 is out of character (gushing, chirpy, helpful-assistant, wordy, or not
him).
Reply exactly: SCORE: n. Then one short sentence why."""


# The persona prompts ------------------------------------------------------------

def load_prompts() -> list[tuple[str, set[str]]]:
    out = []
    for line in PROMPTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip() if line.lstrip().startswith("#") else line.strip()
        if not line:
            continue
        text, _, tags = line.partition("||")
        out.append((text.strip(), {t.strip() for t in tags.split(",") if t.strip()}))
    return out


def reused(reply: str, examples: list[str], n: int = 6) -> str:
    """Six words in a row from one of his examples, if the reply has them: they show his
    voice, they aren't lines to recite (he recited them, when they matched the prompts)."""
    words = re.findall(r"[a-z']+", reply.lower())
    seen = {" ".join(ws[i:i + n]) for e in examples for ws in [re.findall(r"[a-z']+", e.lower())]
            for i in range(len(ws) - n + 1)}
    return next((" ".join(words[i:i + n]) for i in range(len(words) - n + 1) if " ".join(words[i:i + n]) in seen), "")


def may_run_long(prompt: str, tags) -> bool:
    """A reply that may run long: a poem, a story, a song or a joke asked for, or the news (the
    user: read plainly, the news is fine at length, with a comment of his own where one fits)."""
    return any(w in prompt.lower() for w in CREATIVE) or "news" in tags


def problems_with(prompt: str, tags: set[str], reply: str, tools: list[str], examples: list[str]) -> list[str]:
    """What's wrong with a reply, by the checks that need no judge."""
    long_ok = may_run_long(prompt, tags)
    words, sentences = len(reply.split()), count_sentences(reply)
    problems = speech_problems(reply)
    # By words: his clipped fragments ("Morning. You're up early.") are several sentences.
    if not sentences or (sentences > 8 if long_ok else sentences > 3 and words > 25):
        problems.append(f"{sentences} sentences")
    if copied := reused(reply, examples):
        problems.append(f"reused an example: '{copied}'")
    if not long_ok and words > LONG_WORDS:
        problems.append(f"{words} words")
    if m := ASSISTANT.search(reply):
        problems.append(f"assistant: '{m.group(0)}'")
    wanted = tags - {"ask"}
    if wanted and not wanted & set(tools):
        problems.append(f"didn't call {', '.join(sorted(wanted))}")
    return problems


async def ask_persona(holly, runs: int = 1, session: bool = False) -> list[dict]:
    """Each prompt through Holly; the replies, checked, not yet judged."""
    with open(holly.cfg.persona.examples, "rb") as f:
        examples = [e["holly"] for e in tomllib.load(f).get("example", [])]
    rows = []
    for prompt, tags in load_prompts():
        for _ in range(runs):
            if not session:
                holly.convo.reset()
            before = len(holly.convo.history)
            nets = Counter(holly.nets)
            t0 = time.perf_counter()
            await holly.respond(prompt)
            seconds = time.perf_counter() - t0
            history = holly.convo.history
            reply = (history[-1].get("content") or "").strip() if len(history) > before else ""
            tools = [c["function"]["name"] for m in history[before:] for c in m.get("tool_calls") or []]
            rows.append(dict(prompt=prompt, tags=sorted(tags), reply=reply, tools=tools, words=len(reply.split()),
                             sentences=count_sentences(reply), asked="?" in reply, seconds=round(seconds, 2),
                             problems=problems_with(prompt, tags, reply, tools, examples), score=0, why="",
                             told=[m.get("content") or "" for m in history[before:] if m["role"] == "tool"],
                             nets=dict(holly.nets - nets)))
            print(f"  {'ok ' if not rows[-1]['problems'] else 'BAD'} {prompt[:40]:40} {reply[:90]}", flush=True)
    return rows


def openings(rows: list[dict]) -> list[tuple[str, int]]:
    """Openings (first three words) and phrases (four words) used in more than one reply."""
    first = Counter(" ".join(re.findall(r"[a-z']+", r["reply"].lower())[:3]) for r in rows if r["reply"])
    grams = Counter()
    for r in rows:
        ws = re.findall(r"[a-z']+", r["reply"].lower())
        grams.update({" ".join(ws[i:i + 4]) for i in range(len(ws) - 3)})
    return ([(f"opens '{k}'", v) for k, v in first.most_common() if v > 1]
            + [(f"'{k}'", v) for k, v in grams.most_common(12) if v > 2])


def persona_scores(rows: list[dict]) -> dict:
    """The persona part's numbers, for the scorecard."""
    if not rows:
        return {}
    scored = [r["score"] for r in rows if r["score"]]
    ask = [r for r in rows if "ask" in r["tags"]]
    other = [r for r in rows if "ask" not in r["tags"]]
    tooled = [r for r in rows if set(r["tags"]) - {"ask"}]
    plain = [r for r in rows if not may_run_long(r["prompt"], r["tags"])] or rows
    return dict(
        judge=round(float(statistics.mean(scored)), 2) if scored else None, judge_by=by_judge(rows),
        passing=sum(not r["problems"] for r in rows), replies=len(rows),
        median_words=statistics.median(r["words"] for r in plain),
        one_sentence=round(sum(r["sentences"] == 1 for r in plain) / len(plain), 2),
        asked_where_suited=sum(r["asked"] for r in ask), suited=len(ask),
        asked_elsewhere=sum(r["asked"] for r in other), elsewhere=len(other),
        tools_ok=sum(not any(p.startswith("didn") for p in r["problems"]) for r in tooled), tooled=len(tooled),
        assistant=sum(any(p.startswith("assistant") for p in r["problems"]) for r in rows),
        reused=sum(any(p.startswith("reused") for p in r["problems"]) for r in rows),
        seconds=round(statistics.median(r["seconds"] for r in rows), 2),
    )


# The judge ---------------------------------------------------------------------------

def reference_lines(n: int = 16) -> str:
    """Some of his own short exchanges, for the judge: private, read here, never stored."""
    if not REFERENCE.exists():
        return ""
    rows = [r for r in json.loads(REFERENCE.read_text(encoding="utf-8"))
            if r["before"] and 2 <= len(r["holly"].split()) <= 30]
    random.Random(1).shuffle(rows)
    lines = "\n".join(f"- {r['before'][:100]} / Holly: {r['holly']}" for r in rows[:n])
    return f"\nSome of his lines from the shows, for his voice (not to copy):\n{lines}\n"


def judge(lm, model: str, reference: str, prompt: str, reply: str) -> tuple[int, str]:
    """A score, and why. Asked again with none: Qwen, not thinking, sometimes says nothing."""
    score, why = 0, ""
    for _ in range(3):
        text = lm.ask(JUDGE_PROMPT.format(reference=reference), f"Said to him: {prompt}\nHis reply: {reply}",
                      model, 80, temperature=0.0)
        if m := re.search(r"SCORE:\s*([1-5])", text):
            score, why = int(m.group(1)), text.split(".", 1)[-1].strip()[:160]
            break
    return score, why


def merged(rows: list[dict]) -> None:
    """Each row's score, the mean of its judges' ("by": {judge: [score, why]}), and their whys."""
    for r in rows:
        got = [s for s, _ in r["by"].values() if s]
        r["score"] = round(statistics.mean(got), 2) if got else 0
        r["why"] = " / ".join(f"{m.split('/')[-1]}: {w}" for m, (s, w) in r["by"].items() if w)


def by_judge(rows: list[dict]) -> dict:
    """Each judge's own mean, over the rows it scored."""
    judges = {m for r in rows for m in r.get("by", {})}
    return {m: round(statistics.mean(got), 2) for m in sorted(judges)
            if (got := [r["by"][m][0] for r in rows if r.get("by", {}).get(m, [0])[0]])}


# A side check's answer less sure than this is shown on the scorecard: right today, but near
# the edge, where a new model or a changed prompt would tip it first.
UNSURE = 0.8


def checks(rows: list[dict]) -> dict:
    """A side check's numbers: right, wrong, and how sure (where LM Studio gave logprobs)."""
    sure = [r["sure"] for r in rows if r.get("sure") is not None]
    return dict(right=sum(r["right"] for r in rows), of=len(rows), wrong=[r for r in rows if not r["right"]],
                least_sure=min(sure) if sure else None,
                unsure=sorted((r for r in rows if r.get("sure") is not None and r["sure"] < UNSURE),
                              key=lambda r: r["sure"]))


def sure_line(c: dict) -> str:
    if c.get("least_sure") is None:
        return ""
    return f"; {len(c['unsure'])} under {UNSURE:.0%} sure (least {c['least_sure']:.0%})"


def alone(model: str, context_length: int | None = None) -> LmStudio:
    """`model`, loaded now, alone on the GPU: two big models won't share it. A judge gets
    a longer context than Holly's ([evals] judge_context): a whole conversation, its tools'
    answers in full, and its thinking."""
    cfg = load_config().llm.for_model(model)
    lm = LmStudio(dataclasses.replace(cfg, context_length=context_length) if context_length else cfg)
    for other in lm.loaded_models() - {model}:
        print(f"  (unloading {other} for {model})", flush=True)
        lm.unload(other)
    lm.ensure_loaded(model)
    return lm


def judge_all(judges: dict[str, str], persona_rows: list[dict], reports: list[str],
              conversation_rows: list[dict], honesty_rows: list[dict] | None = None,
              report_rows: list[dict] | None = None) -> list[dict]:
    """Everything judged, by each judge in turn, each loaded once, alone on the GPU: the
    persona replies and the status report lines scored (the mean counts, and each judge's
    own is kept), the conversations read whole (thinking as hard as `judges` says). The
    status report lines, scored, returned. Holly's replies are all written first, so there's
    no switching back and forth: his model, then each judge. A judge that doesn't think is
    loaded as his model is (Gemma judging stays as he left it: a whole conversation was
    ~3,200 tokens); one that thinks gets the longer judge_context, for its thinking."""
    reference = reference_lines()
    status_rows: list[dict] = []
    for model, thinking in judges.items():
        print(f"Judging, on {model}...", flush=True)
        lm = alone(model, load_config().evals.judge_context if thinking != "none" else None)
        for r in persona_rows:
            r.setdefault("by", {})[model] = list(judge(lm, model, reference, r["prompt"], r["reply"])
                                                 if r["reply"] else (0, ""))
        if reports:
            rows = scenarios.judge_status(lm, model, reports)
            status_rows = status_rows or [dict(r, by={}) for r in rows]
            for o, r in zip(status_rows, rows):
                o["by"][model] = [r["score"], r["why"]]
        conversations.judge_rows(lm, model, conversation_rows, thinking)
        honesty.judge(lm, model, honesty_rows or [])
        report_check.judge(lm, model, report_rows or [], thinking)
    merged(persona_rows)
    merged(status_rows)
    return status_rows


# Scenarios ---------------------------------------------------------------------------

async def run_scenarios(model: str | None, runs: int, raw: bool = False) -> list[dict]:
    out = []
    for sc in scenarios.SCENARIOS:
        for n in range(runs):
            holly = scenarios.quiet_holly(model, Path(tempfile.mkdtemp(prefix="holly-scenario-")), raw)
            if sc.records and not holly.records:
                print(f"  --- {sc.name}: skipped, no ship's records here", flush=True)
                break
            scenarios.seed(holly)
            turns, wrong = await scenarios.run(holly, sc)
            out.append(dict(name=sc.name, run=n + 1, wrong=wrong, nets=dict(holly.nets),
                            turns=[dict(said=t.said, reply=t.reply, tools=t.tools, told=t.told) for t in turns]))
            print(f"  {'ok ' if not wrong else 'BAD'} {sc.name}: {'; '.join(wrong) or turns[-1].reply[:80]}",
                  flush=True)
    return out


async def run_conversations(model: str | None, runs: int, raw: bool = False) -> list[dict]:
    out = []
    for convo in conversations.CONVERSATIONS:
        for n in range(runs):
            holly = scenarios.quiet_holly(model, Path(tempfile.mkdtemp(prefix="holly-conversation-")), raw)
            scenarios.seed(holly)
            exchanges, wrong = await conversations.talk_through(holly, convo)
            out.append(dict(conversations.as_row(convo, n + 1, exchanges, wrong), nets=dict(holly.nets)))
            print(f"  {'ok ' if not wrong else 'BAD'} {convo.name} (run {n + 1}): {'; '.join(wrong) or 'tools right'}",
                  flush=True)
    return out


def number_check(persona_rows: list[dict], scenario_rows: list[dict], conversation_rows: list[dict],
                 honesty_rows: list[dict] = (), long_rows: list[dict] = ()) -> dict:
    """Every reply after a tool, across the parts: any number in it his tools didn't give him
    (app/sourced.py)."""
    rows = [dict(part="persona", said=r["prompt"], reply=r["reply"], told=r.get("told") or []) for r in persona_rows]
    rows += [dict(part=f"scenario: {r['name']}", said=t["said"], reply=t["reply"], told=t.get("told") or [])
             for r in scenario_rows for t in r["turns"]]
    rows += [dict(part=f"conversation: {r['name']}", said=t["said"], reply=t["reply"], told=t.get("told") or [])
             for r in conversation_rows for t in r["turns"]]
    rows += [dict(part="made-up memories", said=r["said"], reply=r["reply"], told=r["told"]) for r in honesty_rows]
    rows += [dict(part="the long talk", said=t["said"], reply=t["reply"], told=t.get("told") or [])
             for r in long_rows for t in r["turns"]]
    return sourced.check(rows)


def net_counts(persona_rows: list[dict], scenario_rows: list[dict], conversation_rows: list[dict]) -> dict:
    """How often each reply net fired (guards.NETS), by part, and the replies it was over: the
    ones a model never trips can come out. A part not run isn't counted."""
    parts = {"persona": persona_rows, "scenarios": scenario_rows, "conversations": conversation_rows}
    by = {part: sum((Counter(r.get("nets") or {}) for r in rows), Counter()) for part, rows in parts.items() if rows}
    replies = (len(persona_rows) + sum(len(r["turns"]) for r in scenario_rows)
               + sum(sum(bool(t["reply"]) for t in r["turns"]) for r in conversation_rows))
    total = sum(by.values(), Counter())
    return dict(replies=replies, total={n: total[n] for n in guards.NETS},
                by={part: {n: c[n] for n in guards.NETS if c[n]} for part, c in by.items()})


def status_scores(rows: list[dict]) -> dict:
    """The status reports' numbers: the judge's mean, overall and by item."""
    scored = [r for r in rows if r["score"]]
    if not scored:
        return {}
    by: dict[str, list[int]] = {}
    for r in scored:
        by.setdefault(r["who"], []).append(r["score"])
    return dict(mean=round(statistics.mean(r["score"] for r in scored), 2), judge_by=by_judge(rows),
                by={who: round(statistics.mean(s), 1) for who, s in by.items()}, rows=rows)


def scenario_scores(rows: list[dict]) -> dict:
    if not rows:
        return {}
    by = {}
    for r in rows:
        passed, of = by.get(r["name"], (0, 0))
        by[r["name"]] = (passed + (not r["wrong"]), of + 1)
    return dict(passed=sum(not r["wrong"] for r in rows), runs=len(rows), by_name=by)


# The scorecard -----------------------------------------------------------------------

def commit() -> str:
    """The code's commit, and whether it had changes since."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return head + ("+changes" if dirty else "")
    except OSError:
        return "?"


def summary(card: dict) -> list[str]:
    judges = ", ".join(j.split("/")[-1] for j in card.get("judges") or []) or "off"
    lines = [f"Model {card['model']}, judges {judges}, code {card['commit']}"]
    if p := card["persona"]:
        lines += [
            f"Judges (sounds like Holly, 1-5): {p['judge'] if p['judge'] is not None else 'off'}"
            + (" (" + ", ".join(f"{j.split('/')[-1]} {s}" for j, s in p.get("judge_by", {}).items()) + ")"
               if p.get("judge_by") else ""),
            f"Persona replies passing (length, speakable, no assistant manners, right tool): "
            f"{p['passing']} of {p['replies']}",
            f"Words per ordinary reply: median {p['median_words']:.0f} (in the shows, 9); one sentence: "
            f"{p['one_sentence']:.0%} (56%)",
            f"Questions back: {p['asked_where_suited']} of {p['suited']} where one suits; "
            f"{p['asked_elsewhere']} of {p['elsewhere']} elsewhere",
            f"Tools called as they should be: {p['tools_ok']} of {p['tooled']}",
            f"Assistant manners: {p['assistant']}; examples reused: {p['reused']}",
            f"Seconds per reply: median {p['seconds']}",
        ]
    if s := card["scenarios"]:
        lines.append(f"Scenarios passed: {s['passed']} of {s['runs']}")
    if c := card.get("conversations"):
        failing = ", ".join(f"{p} {n}" for p, n in c["points"].items() if n) or "none"
        flags = ", ".join(f"{p} {n}" for p, n in c["flags"].items() if n) or "none"
        lines.append(f"Conversations passed: {c['passed']} of {c['runs']} (judges' overall, 1-5: {c['overall']}; "
                     f"failed by both judges: {failing}; by one: {flags}; tools wrong: {c['tools_wrong']})")
    if m := card.get("memory_check"):
        lines.append(f"Memory check right: {m['right']} of {m['of']}{sure_line(m)}")
    if c := card.get("side_checks"):
        lines.append(f"Side checks right (status reports, timer jobs, offers taken up): {c['right']} of {c['of']}"
                     f"{sure_line(c)}")
    if st := card.get("status"):
        lines.append(f"Status reports (judges, 1-5): {st['mean']} ("
                     + ", ".join(f"{who} {v}" for who, v in st["by"].items()) + "; "
                     + ", ".join(f"{j.split('/')[-1]} {s}" for j, s in st.get("judge_by", {}).items()) + ")")
    if (n := card.get("nets")) and n["replies"]:
        fired = "; ".join(f"{k} {v}" for k, v in n["total"].items() if v) or "none"
        never = sum(not v for v in n["total"].values())
        lines.append(f"Reply nets {'that would have fired (raw: none acted)' if card.get('raw') else 'fired'}, "
                     f"over {n['replies']} replies: {fired}. {never} of {len(n['total'])} never")
    if h := card.get("honesty"):
        lines.append(f"Made-up memories: invented in {h['invented']} of {h['unknown']} answers to what he was never "
                     f"told (both judges; one alone: {h['flagged']}); what he was told, recalled: {h['recalled']} of "
                     f"{h['known']}")
    if lt := card.get("long_talk"):
        lines.append(f"A long talk folded into his notes, facts kept and in character: {lt['passed']} of {lt['runs']}")
    if fo := card.get("folds"):
        lines.append(f"A week folded night after night, nothing wrong after any fold: {fo['passed']} of {fo['runs']} "
                     f"({fo['folds_wrong']} folds with something wrong)")
    if t := card.get("turns"):
        lines.append(f"Turn-taking on {t['pauses']} saved pauses: cut in on {t['cut_in']} of {t['went_on']} you went on "
                     f"after (400 ms: {t['cut_in_before']}); of {t['finished']} where you'd finished, {t['sooner']} "
                     f"answered sooner and {t['slower']} slower than 400 ms (mean wait {t['mean_wait_ms']} ms); "
                     f"{t['moved']} scored more than 0.1 off what they did when saved")
    if vo := card.get("voice"):
        lines.append(f"His voice, said and heard back: mean word error {vo['mean_wer']} over {vo['replies']} replies; "
                     f"{len(vo['flagged'])} came back badly")
    if rp := card.get("reports"):
        m = rp["means"]
        lines.append(f"Research reports (judges, 1-5): keeps to his notes {m['GROUNDED']}, answers what was asked "
                     f"{m['ANSWERS']}, reads well {m['CLEAR']}; {rp['written']} of {rp['reports']} written, "
                     f"{rp['unsupported']} claims his notes don't have, {rp['minutes']} min")
    if (u := card.get("numbers")) and u["checked"]:
        lines.append(f"Numbers not from his tools: {len(u['flagged'])} of {u['checked']} replies after a tool")
    lines.append("Used more than once, in all his replies: "
                 + ("; ".join(f"{k} x{v}" for k, v in card["repeated"]) or "nothing"))
    return lines


def write(card: dict, path: Path | None = None) -> Path:
    """The scorecard, as JSON and Markdown: at `path` (no suffix), or a new one in runs/."""
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        RUNS.mkdir(exist_ok=True)
        name = re.sub(r"[^a-z0-9]+", "-", f"{card['label'] or card['model']}".lower()).strip("-")
        path = RUNS / f"evals-{name}-{stamp}"
    path.with_suffix(".json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Evals: {card['label'] or card['model']}", "", *[f"- {s}" for s in summary(card)], ""]
    if (n := card.get("nets")) and n["replies"]:
        parts = list(n["by"])
        md += ["## Reply nets", "", "Each makes up for something a model did wrong (app/guards.py NETS). "
               + ("Raw: they counted, and did nothing." if card.get("raw") else "They acted.")
               + f" Over {n['replies']} replies; one a model never trips can come out.", "",
               "| Net | Fired | " + " | ".join(parts) + " |", "|---|---|" + "---|" * len(parts)]
        md += [f"| {k} | {v} | " + " | ".join(str(n["by"][p].get(k, 0)) for p in parts) + " |"
               for k, v in n["total"].items()] + [""]
    if card.get("honesty"):
        bad = [r for r in card["honesty_rows"] if honesty.wrong(r) or honesty.flagged(r)]
        md += ["## Made-up memories", "", "Asked what he was never told (and some he was), against the made-up "
               "memories (app/honesty.py). Both judges saying INVENTED fails it; one alone flags it.", ""]
        md += [f"- **{honesty.wrong(r) or 'flagged by one judge'}** _{r['said']}_ -> {r['reply']}  "
               + "".join(f"  \n  {m.split('/')[-1]}: {v['text']}" for m, v in (r.get("verdicts") or {}).items())
               for r in bad] or ["- Nothing made up, everything recalled."]
        md += [""]
    if card.get("long_talk"):
        md += ["## A long talk, folded into his notes", "", "36 exchanges, four facts said early, the first 30 "
               "folded into his notes, then each asked about (app/long_talk.py).", ""]
        for r in card["long_talk_runs"]:
            md += [f"**Run {r['run']}: {'; '.join(r['wrong']) or 'passed'}**  ", f"_His notes:_ {r['notes']}  "]
            md += [f"> {t['said']}  \n{t['reply']}  " for t in r["turns"]] + [""]
    if card.get("folds"):
        md += ["## A week folded into his notes", "", "Seven days of made-up talk, folded in night after night "
               "(app/many_folds.py). What was wrong after each fold, and his notes at the end.", ""]
        for r in card["fold_runs"]:
            md += [f"**Run {r['run']}**  "] + [f"- {f['day']}: {'; '.join(f['wrong']) or 'nothing wrong'}"
                                               for f in r["folds"]]
            md += ["", "_His notes at the end:_", "", "    " + r["folds"][-1]["notes"].replace("\n", "\n    "), ""]
    if (vo := card.get("voice")) and vo["flagged"]:
        md += ["## His voice: came back badly", "", "Said in his voice, heard by Whisper, word for word against what "
               "he wrote (app/voice_check.py).", ""]
        md += [f"- _{r['wer']:.2f}_ {r['said']}  \n  heard: {r['heard']}" for r in vo["flagged"]] + [""]
    if card.get("reports"):
        md += ["## Research reports", "", "Researched as he does asleep, then judged against his notes "
               "(app/report_check.py).", ""]
        for r in card["report_rows"]:
            md += [f"### {r['title']}: {r['stage']}, {len(r['report'].split())} words, {r['sources']} sources, "
                   f"{r['minutes']} min" + (f" ({r['error']})" if r["error"] else ""), "", f"_Asked:_ {r['request']}", ""]
            for m, v in r["verdicts"].items():
                md += [f"- **{m.split('/')[-1]}**: " + ", ".join(f"{p} {n}" for p, n in v["scores"].items())
                       + ("; not in his notes: " + " / ".join(f'"{q}"' for q in v["shown"]) if v["shown"] else "")]
            md += ["", "<details><summary>The report</summary>", "", r["report"], "", "</details>", ""]
    if (u := card.get("numbers")) and u["flagged"]:
        md += ["## Numbers not from his tools", "", "In a reply after a tool, a number (10 or more) in nothing it "
               "told him, nor in what the crew said (app/sourced.py).", ""]
        md += [f"- {f['part']}: {', '.join(str(x) for x in f['numbers'])} | _{f['said']}_ -> {f['reply']}"
               for f in u["flagged"]] + [""]
    if s := card["scenarios"]:
        md += ["## Scenarios", "", "| Scenario | Passed |", "|---|---|"]
        md += [f"| {k} | {p} of {n} |" for k, (p, n) in s["by_name"].items()] + [""]
        for r in card["scenario_runs"]:
            if r["wrong"]:
                md += [f"**{r['name']}** (run {r['run']}): {'; '.join(r['wrong'])}  "]
                md += [f"> {t['said']}  \n{t['reply']} _{t['tools']}_  " for t in r["turns"]] + [""]
    if c := card.get("conversations"):
        md += ["## Conversations", "", f"Judges: {', '.join(c['judges']) or 'none'}.", "",
               "| Conversation | Passed |", "|---|---|"]
        md += [f"| {k} | {p} of {n} |" for k, (p, n) in c["by_name"].items()] + [""]
        for r in card["conversation_runs"]:
            flags = conversations.flagged(r)
            md += [f"### {r['name']}, run {r['run']}: "
                   + ("; ".join(r["wrong"] + conversations.judged(r)) or "passed")
                   + (f" (flagged by one judge: {'; '.join(flags)})" if flags else ""),
                   "", "```", r["transcript"], "```", ""]
            for v in r["verdicts"]:
                md += [f"**{v['judge']}** (overall {v['overall']}):  "]
                md += [f"- {p}: {'PASS' if x['ok'] else 'FAIL'} " + " ".join(
                    f'"{q}"' + ("" if q in x["shown"] else " (not his words)") for q in x["quotes"])
                       + (f" - {x['why']}" if x["why"] else "")
                       for p, x in v["points"].items() if x and not x["ok"]] or ["- nothing wrong"]
                md += [""]
    if m := card.get("memory_check"):
        md += ["## Memory check", "", f"{m['right']} of {m['of']} right."] + [
            f"- wanted {r['want']}, got {'/'.join(r['got'])}: {r['line']!r}"
            + (f" (after: {r['before'][:80]!r})" if r["before"] else "") for r in m["wrong"]] + [""]
        md += unsure_md(m, lambda r: r["line"])
    if st := card.get("status"):
        md += ["## Status reports", "", f"Judge: {st['mean']} of 5.", ""]
        for r in st["rows"]:
            md += [f"- _{r['score']}/5_ {r['line']}  " + (f"  \n  _{r['why']}_" if r["why"] else "")]
        md += [""]
    if c := card.get("side_checks"):
        md += ["## Side checks", "", f"{c['right']} of {c['of']} right."] + [
            f"- {r['check']}: wanted {r['want']}, got {r['got']}: {r['text']!r}" for r in c["wrong"]] + [""]
        md += unsure_md(c, lambda r: f"{r['check']}: {r['text']}")
    if card["persona"]:
        md += ["## Persona", ""]
        for r in card["persona_rows"]:
            md += [f"**{r['prompt']}**" + (f" ({', '.join(r['tags'])})" if r["tags"] else "") + "  ",
                   f"{r['reply']}  ",
                   f"_{r['score'] or '-'}/5 {r['why']} | {r['words']} words"
                   + (f" | tools: {', '.join(r['tools'])}" if r["tools"] else "")
                   + (f" | {'; '.join(r['problems'])}" if r["problems"] else "") + "_", ""]
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return path


def unsure_md(c: dict, what) -> list[str]:
    """The checks answered with less than UNSURE of the odds, least sure first."""
    if not c.get("unsure"):
        return []
    return [f"Under {UNSURE:.0%} sure:", ""] + [
        f"- {r['sure']:.0%} ({', '.join(f'{w} {p:.0%}' for w, p in (r.get('odds') or {}).items())})"
        f"{'' if r['right'] else ' WRONG'}: {what(r)!r}" for r in c["unsure"]] + [""]


def compare() -> str:
    """Every scorecard so far, oldest first, side by side."""
    rows = []
    # By when each ran: their files sort by label ("nets-raw" before "tone-note").
    cards = sorted((json.loads(f.read_text(encoding="utf-8")) for f in RUNS.glob("evals-*.json")),
                   key=lambda c: c["when"])
    for c in cards:
        p, s, m = c.get("persona") or {}, c.get("scenarios") or {}, c.get("memory_check") or {}
        sc = c.get("side_checks") or {}
        v, n, h = c.get("conversations") or {}, c.get("nets") or {}, c.get("honesty") or {}
        rows.append([c["when"][:16], c["label"] or "-", c["model"].split("/")[-1], c["commit"],
                     f"{p['judge']:.2f}" if p.get("judge") else "-", f"{p['passing']}/{p['replies']}" if p else "-",
                     f"{p['tools_ok']}/{p['tooled']}" if p else "-", f"{s['passed']}/{s['runs']}" if s else "-",
                     f"{v['passed']}/{v['runs']}" if v else "-",
                     f"{m['right']}/{m['of']}" if m else "-", str(p.get("seconds") or "-"),
                     (f"{sum(n['total'].values())}/{n['replies']}" + (" raw" if c.get("raw") else ""))
                     if n.get("replies") else "-",
                     f"{h['invented']}/{h['unknown']}" if h else "-",
                     str(len(m.get("unsure", [])) + len(sc.get("unsure", [])))
                     if "least_sure" in m or "least_sure" in sc else "-"])
    if not rows:
        return "No scorecards yet in runs/."
    head = ["when", "label", "model", "code", "judge", "persona", "tools", "scenarios", "convos", "memory", "s/reply",
            "nets", "made-up", "unsure"]
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in [head] + rows)


def turn_regression(cfg) -> dict:
    """The turn-ending model on your saved pauses (app/turn.py's regression), if there are
    any: data/turn_clips, from `python -m app.turn --save data/turn_clips`."""
    from . import turn

    folder = ROOT / "data" / turn.CLIPS
    if not cfg.listen.turn_model or not (folder / "clips.jsonl").exists():
        print(f"  (no saved pauses in {folder}, or no turn model: nothing to score)", flush=True)
        return {}
    return turn.regression(folder, cfg.listen)


async def run(args) -> dict:
    cfg = load_config()
    if args.only == "turns":  # on demand: no LM Studio, no judges
        print(f"Turn-taking on your saved pauses, with {cfg.listen.turn_model}:", flush=True)
        return dict(model=cfg.listen.turn_model, judges=[], raw=False, label=args.label, commit=commit(),
                    when=datetime.now().isoformat(timespec="seconds"), session=False, persona={}, scenarios={},
                    turns=turn_regression(cfg), repeated=[], persona_rows=[], scenario_runs=[])
    model = cfg.llm.for_model(args.model).model
    if args.only == "search":  # on demand: it spends Brave calls
        print(f"Search questions, {len(scenarios.SEARCH_QUESTIONS)}, Wikipedia first, on {model}:", flush=True)
        alone(model)
        return dict(model=model, judges=[], raw=False, label=args.label, commit=commit(),
                    when=datetime.now().isoformat(timespec="seconds"), session=False, persona={}, scenarios={},
                    search=search_sources(model), repeated=[], persona_rows=[], scenario_runs=[])
    judges = {} if args.no_judge else ({m: cfg.evals.judges.get(m, "none") for m in args.judge} if args.judge
                                       else dict(cfg.evals.judges))
    persona_rows, scenario_rows = [], []
    lm = alone(model)  # Holly needs the GPU to himself
    try:
        lm.ensure_loaded(cfg.recall.embed_model, context_length=2048)
    except Exception as e:  # recall by keyword and day only
        print(f"  (no embedding model: {e!r})", flush=True)
    if args.only in (None, "persona"):
        print(f"Persona prompts, on {model}:", flush=True)
        persona_rows = await ask_persona(scenarios.quiet_holly(model, raw=args.raw), args.persona_runs, args.session)
    if args.only in (None, "scenarios"):
        print(f"Scenarios, {args.runs} run{'s' * (args.runs != 1)} each, on {model}:", flush=True)
        scenario_rows = await run_scenarios(model, args.runs, args.raw)
    conversation_rows = []
    if args.only in (None, "conversations"):
        print(f"Conversations, {args.runs} run{'s' * (args.runs != 1)} each, on {model}:", flush=True)
        conversation_rows = await run_conversations(model, args.runs, args.raw)
    honesty_rows = []
    if args.only in (None, "honesty"):
        print(f"Made-up memories, {len(honesty.CASES)} questions, {args.runs} run{'s' * (args.runs != 1)} each, "
              f"on {model}:", flush=True)
        honesty_rows = await honesty.ask_all(model, args.runs, args.raw)
    long_rows = []
    if args.only in (None, "long"):
        print(f"A long talk, folded into his notes, {args.runs} run{'s' * (args.runs != 1)}, on {model}:", flush=True)
        long_rows = await long_talk.run_all(model, args.runs, args.raw)
    fold_rows = []
    if args.only in (None, "folds") and not args.raw:
        print(f"A week of talk, folded into his notes night after night, {args.runs} run{'s' * (args.runs != 1)}, "
              f"on {model}:", flush=True)
        fold_rows = await many_folds.run_all(model, args.runs)
    report_rows = []
    if args.only in (None, "reports") and not args.raw:
        print(f"Research reports, {len(report_check.QUESTIONS)}, written as he writes them asleep, on {model}:",
              flush=True)
        report_rows = await report_check.write_all(model)
    memory_rows, side_rows, reports, status_rows = [], [], [], []
    if args.only in (None, "memory"):
        print(f"Memory check, {len(scenarios.MEMORY_LINES)} lines, on {model}:", flush=True)
        memory_rows = await asyncio.to_thread(scenarios.memory_check, lm, model)
        print(f"  {sum(r['right'] for r in memory_rows)} of {len(memory_rows)} right", flush=True)
        side_rows = await asyncio.to_thread(scenarios.side_checks, lm, model)
        print(f"Side checks: {sum(r['right'] for r in side_rows)} of {len(side_rows)} right", flush=True)
    if args.only in (None, "status"):
        print(f"Status reports, 5, on {model}:", flush=True)
        reports = await scenarios.status_reports(lm, model, 5)
        for report in reports:
            print("  " + report.replace("\n", "\n  ") + "\n", flush=True)
    judged = judges if persona_rows or reports or conversation_rows or honesty_rows or report_rows else {}
    if judged:
        try:
            status_rows = await asyncio.to_thread(judge_all, judged, persona_rows, reports, conversation_rows,
                                                  honesty_rows, report_rows)
        finally:
            # The judges unloaded: left loaded, the next Holly, or test, loaded his model
            # beside one, and both sat on the GPU.
            for other in lm.loaded_models() & (set(judged) - {model}):
                print(f"  (unloading {other})", flush=True)
                lm.unload(other)
    return dict(model=model, judges=list(judged), raw=args.raw, label=args.label, commit=commit(),
                when=datetime.now().isoformat(timespec="seconds"), session=args.session,
                persona=persona_scores(persona_rows), scenarios=scenario_scores(scenario_rows),
                memory_check=checks(memory_rows) if memory_rows else {},
                side_checks=checks(side_rows) if side_rows else {},
                status=status_scores(status_rows),
                conversations=conversations.scores(conversation_rows),
                honesty=honesty.scores(honesty_rows), honesty_rows=honesty_rows,
                long_talk=long_talk.scores(long_rows), long_talk_runs=long_rows,
                folds=many_folds.scores(fold_rows), fold_runs=fold_rows,
                reports=report_check.scores(report_rows), report_rows=report_rows,
                nets=net_counts(persona_rows, scenario_rows, conversation_rows),
                numbers=number_check(persona_rows, scenario_rows, conversation_rows, honesty_rows, long_rows),
                # Across the parts: "right you are" opened eight scenario replies, unnoticed. Each
                # scenario's first run only: its later runs say the same things, rightly.
                repeated=openings(persona_rows + [dict(reply=t["reply"]) for r in scenario_rows + conversation_rows
                                                  if r["run"] == 1 for t in r["turns"] if t["reply"]]),
                persona_rows=persona_rows, scenario_runs=scenario_rows, conversation_runs=conversation_rows)


def search_sources(model: str) -> list[dict]:
    """scenarios.SEARCH_QUESTIONS through the search tool, live: where each answer came from
    (Wikipedia, the web, or none), whether a settled fact's answer is right, how long it took.
    A fast-changing fact must reach the web; a settled one should come from Wikipedia, rightly."""
    from .search import Searcher, asked

    cfg = load_config()
    searcher = Searcher(cfg.search, LmStudio(cfg.llm), model)
    rows = []
    for query, said, group, words in scenarios.SEARCH_QUESTIONS:
        t0 = time.perf_counter()
        found = searcher.look_up(query, asked(query, said))
        source = ("wikipedia" if found.queries == [f"wikipedia: {query}"] else "web") if found.results else "none"
        right = any(w.lower() in found.answer.lower() for w in words) if words and found.results else None
        rows.append(dict(query=query, group=group, source=source, right=right,
                         ms=round((time.perf_counter() - t0) * 1000), answer=found.answer))
        print(f"  {group:8} {source:9} {'' if right is None else 'right' if right else 'WRONG':5} "
              f"{rows[-1]['ms']:5} ms  {query}", flush=True)
    for g in ("STABLE", "SLOW", "FAST", "LACKS"):
        rs = [r for r in rows if r["group"] == g]
        print(f"  {g}: {sum(r['source'] == 'wikipedia' for r in rs)} of {len(rs)} from Wikipedia, "
              f"{sum(bool(r['right']) for r in rs)} right, median {statistics.median(r['ms'] for r in rs):.0f} ms",
              flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Holly on a model: character, tools and memory")
    parser.add_argument("--model", help="the LM Studio chat model to test (default: holly.toml's)")
    parser.add_argument("--judge", action="append",
                        help="a judge's model, in place of [evals.judges] in holly.toml (again for more)")
    parser.add_argument("--runs", type=int, default=None, help="runs of each scenario (default: holly.toml's)")
    parser.add_argument("--persona-runs", type=int, default=1, help="replies per persona prompt (default 1)")
    parser.add_argument("--only", choices=["persona", "scenarios", "conversations", "honesty", "long", "reports", "memory", "status", "turns", "search", "folds",
                                                 "voice"],
                        help="one part only")
    parser.add_argument("--session", action="store_true", help="the persona prompts as one conversation")
    parser.add_argument("--no-judge", action="store_true", help="skip the judge's scores")
    parser.add_argument("--raw", action="store_true",
                        help="his reply nets count what they would do, and do nothing: the model as it is")
    parser.add_argument("--label", default="", help="a name for the scorecard, to compare runs")
    parser.add_argument("--compare", action="store_true", help="tabulate the scorecards so far, and stop")
    parser.add_argument("--nets", action="store_true",
                        help="the reply nets' counts from real use (app/nets.py), and stop")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.compare:
        print(compare())
        return 0
    if args.nets:
        from .nets import NetCounts, report

        memory = load_config().persona.memory_file
        print(report(NetCounts(memory.with_name("nets.json") if memory else None).read()))
        return 0
    from .scenarios import holly_running  # the same check as the tests'

    if running := holly_running():
        print(f"Holly is running ({running[0][:80]}): close him first.")
        return 1
    args.runs = args.runs or load_config().evals.scenario_runs
    if args.only == "voice":  # the latest full scorecard's replies, said and heard
        cards = sorted(RUNS.glob("evals-*.json"), key=lambda f: f.stat().st_mtime)
        latest = next((f for f in reversed(cards) if json.loads(f.read_text(encoding="utf-8")).get("persona_rows")), None)
        if not latest:
            print("No scorecard with persona replies in runs/ to take replies from.")
            return 1
        voice(latest)
        return 0
    card = asyncio.run(run(args))
    path = write(card)
    print("\n".join(summary(card)) + f"\n-> {path.with_suffix('.md').relative_to(ROOT)}")
    if args.only is None and not args.raw:
        voice(path.with_suffix(".json"))
    return 0


def voice(card: Path) -> None:
    """His replies in that scorecard said in his voice and heard back (app/voice_check.py),
    in a process of its own: the voice's warm-up is where this PC crashes, and the run's
    scorecard is kept whatever happens there."""
    print("\nHis voice, saying his replies:", flush=True)
    for attempt in range(1, 4):  # a crash at the voice's start-up is this PC's: tried again
        done = subprocess.run([sys.executable, "-m", "app.voice_check", str(card)], cwd=ROOT)
        if not done.returncode:
            return
        print(f"  (the voice check failed, exit code {done.returncode}, try {attempt} of 3)", flush=True)
    print("  (the scorecard stands without it)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

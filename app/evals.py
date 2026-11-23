"""How good Holly is on a model: his character, and his use of tools and memory, scored the
same way every time, so models and changes can be compared.

    python -m app.evals [--model KEY] [--runs N] [--label NAME] [--only persona|scenarios]
                        [--session] [--no-judge]
    python -m app.evals --compare

Two parts, both through his real reply loop with a silent voice, his memory in a
temporary folder, and Spotify off:

- persona: each prompt in persona/test_prompts.txt, asked afresh (or, with --session,
  as one conversation, so ruts show). Each reply is checked for a voice (sentences,
  words, anything that can't be said aloud), an assistant's manners ("Certainly!",
  "anything else?"), a question back (welcome where the prompt is tagged "ask"), the
  tool it's tagged with, lines recited from his examples, and, across the run, phrases
  used again and again. A judge scores each 1 to 5 for sounding like Holly.
- scenarios: conversations with made-up memories (app/scenarios.py), each run --runs
  times (Gemma varies), checked for the tools and facts they need.

The judge is one fixed model ([evals] judge in holly.toml), whatever model is under
test, so scores compare between runs; it's loaded after every reply has been written,
as two big models won't share the GPU. If data/scripts/holly_exchanges.json is there
(his lines from the scripts, private and git-ignored), the judge is shown some of
them for reference; they're never stored.

A scorecard goes to runs/: Markdown, with every reply, to read for character, and
JSON, which --compare tabulates against earlier runs. LM Studio must be running, with
no Holly: he'd be sharing the GPU.
"""

from __future__ import annotations

import argparse
import asyncio
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

from . import scenarios
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
him: deadpan, unhurried, terse, colloquial and a bit cheeky, vain about his IQ of
6000 yet senile, bored and glad of company, never an eager assistant. Here he's the
computer in someone's home, so their life is his world now: judge the voice, not
the setting. A good line is short enough to say aloud and sounds like him.
{reference}
Score the reply 1 to 5: 5 is unmistakably Holly; 3 is fine but anyone could have
said it; 1 is out of character (gushing, helpful-assistant, wordy, or not him).
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


def problems_with(prompt: str, tags: set[str], reply: str, tools: list[str], examples: list[str]) -> list[str]:
    """What's wrong with a reply, by the checks that need no judge."""
    creative = any(w in prompt.lower() for w in CREATIVE)
    words, sentences = len(reply.split()), count_sentences(reply)
    problems = speech_problems(reply)
    # By words: his clipped fragments ("Morning. You're up early.") are several sentences.
    if not sentences or (sentences > 8 if creative else sentences > 3 and words > 25):
        problems.append(f"{sentences} sentences")
    if copied := reused(reply, examples):
        problems.append(f"reused an example: '{copied}'")
    if not creative and words > LONG_WORDS:
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
            t0 = time.perf_counter()
            await holly.respond(prompt)
            seconds = time.perf_counter() - t0
            history = holly.convo.history
            reply = (history[-1].get("content") or "").strip() if len(history) > before else ""
            tools = [c["function"]["name"] for m in history[before:] for c in m.get("tool_calls") or []]
            rows.append(dict(prompt=prompt, tags=sorted(tags), reply=reply, tools=tools, words=len(reply.split()),
                             sentences=count_sentences(reply), asked="?" in reply, seconds=round(seconds, 2),
                             problems=problems_with(prompt, tags, reply, tools, examples), score=0, why=""))
            print(f"  {'ok ' if not rows[-1]['problems'] else 'BAD'} {prompt[:40]:40} {reply[:90]}", flush=True)
    return rows


def openings(rows: list[dict]) -> list[tuple[str, int]]:
    """Openings (first three words) and phrases (four words) used in more than one reply."""
    first = Counter(" ".join(r["reply"].lower().split()[:3]) for r in rows if r["reply"])
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
    plain = [r for r in rows if not any(w in r["prompt"].lower() for w in CREATIVE)] or rows
    return dict(
        judge=round(float(statistics.mean(scored)), 2) if scored else None,
        passing=sum(not r["problems"] for r in rows), replies=len(rows),
        median_words=statistics.median(r["words"] for r in plain),
        one_sentence=round(sum(r["sentences"] == 1 for r in plain) / len(plain), 2),
        asked_where_suited=sum(r["asked"] for r in ask), suited=len(ask),
        asked_elsewhere=sum(r["asked"] for r in other), elsewhere=len(other),
        tools_ok=sum(not any(p.startswith("didn") for p in r["problems"]) for r in tooled), tooled=len(tooled),
        assistant=sum(any(p.startswith("assistant") for p in r["problems"]) for r in rows),
        reused=sum(any(p.startswith("reused") for p in r["problems"]) for r in rows),
        repeated=openings(rows),
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
    text = lm.ask(JUDGE_PROMPT.format(reference=reference), f"Said to him: {prompt}\nHis reply: {reply}", model,
                  80, temperature=0.0)
    m = re.search(r"SCORE:\s*([1-5])", text)
    return (int(m.group(1)) if m else 0), text.split(".", 1)[-1].strip()[:160]


def judge_all(rows: list[dict], judge_model: str) -> None:
    """Score every reply with the judge, loaded now, alone on the GPU."""
    lm = LmStudio(load_config().llm.for_model(judge_model))
    for other in lm.loaded_models() - {judge_model}:
        print(f"  (unloading {other} for the judge)", flush=True)
        lm.unload(other)
    lm.ensure_loaded(judge_model)
    reference = reference_lines()
    for r in rows:
        r["score"], r["why"] = judge(lm, judge_model, reference, r["prompt"], r["reply"]) if r["reply"] else (0, "")


# Scenarios ---------------------------------------------------------------------------

async def run_scenarios(model: str | None, runs: int) -> list[dict]:
    out = []
    for sc in scenarios.SCENARIOS:
        for n in range(runs):
            holly = scenarios.quiet_holly(model, Path(tempfile.mkdtemp(prefix="holly-scenario-")))
            scenarios.seed(holly)
            turns, wrong = await scenarios.run(holly, sc)
            out.append(dict(name=sc.name, run=n + 1, wrong=wrong,
                            turns=[dict(said=t.said, reply=t.reply, tools=t.tools) for t in turns]))
            print(f"  {'ok ' if not wrong else 'BAD'} {sc.name}: {'; '.join(wrong) or turns[-1].reply[:80]}",
                  flush=True)
    return out


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
    lines = [f"Model {card['model']}, judge {card['judge'] or 'off'}, code {card['commit']}"]
    if p := card["persona"]:
        lines += [
            f"Judge (sounds like Holly, 1-5): {p['judge'] if p['judge'] is not None else 'off'}",
            f"Persona replies passing (length, speakable, no assistant manners, right tool): "
            f"{p['passing']} of {p['replies']}",
            f"Words per ordinary reply: median {p['median_words']:.0f} (in the shows, 9); one sentence: "
            f"{p['one_sentence']:.0%} (56%)",
            f"Questions back: {p['asked_where_suited']} of {p['suited']} where one suits; "
            f"{p['asked_elsewhere']} of {p['elsewhere']} elsewhere",
            f"Tools called as they should be: {p['tools_ok']} of {p['tooled']}",
            f"Assistant manners: {p['assistant']}; examples reused: {p['reused']}",
            "Used more than once: " + ("; ".join(f"{k} x{v}" for k, v in p["repeated"]) or "nothing"),
            f"Seconds per reply: median {p['seconds']}",
        ]
    if s := card["scenarios"]:
        lines.append(f"Scenarios passed: {s['passed']} of {s['runs']}")
    return lines


def write(card: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS.mkdir(exist_ok=True)
    name = re.sub(r"[^a-z0-9]+", "-", f"{card['label'] or card['model']}".lower()).strip("-")
    path = RUNS / f"evals-{name}-{stamp}"
    path.with_suffix(".json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Evals: {card['label'] or card['model']}", "", *[f"- {s}" for s in summary(card)], ""]
    if s := card["scenarios"]:
        md += ["## Scenarios", "", "| Scenario | Passed |", "|---|---|"]
        md += [f"| {k} | {p} of {n} |" for k, (p, n) in s["by_name"].items()] + [""]
        for r in card["scenario_runs"]:
            if r["wrong"]:
                md += [f"**{r['name']}** (run {r['run']}): {'; '.join(r['wrong'])}  "]
                md += [f"> {t['said']}  \n{t['reply']} _{t['tools']}_  " for t in r["turns"]] + [""]
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


def compare() -> str:
    """Every scorecard so far, oldest first, side by side."""
    rows = []
    for f in sorted(RUNS.glob("evals-*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        p, s = c.get("persona") or {}, c.get("scenarios") or {}
        rows.append([c["when"][:16], c["label"] or "-", c["model"].split("/")[-1], c["commit"],
                     f"{p['judge']:.2f}" if p.get("judge") else "-", f"{p['passing']}/{p['replies']}" if p else "-",
                     f"{p['tools_ok']}/{p['tooled']}" if p else "-", f"{s['passed']}/{s['runs']}" if s else "-",
                     str(p.get("seconds") or "-")])
    if not rows:
        return "No scorecards yet in runs/."
    head = ["when", "label", "model", "code", "judge", "persona", "tools", "scenarios", "s/reply"]
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in [head] + rows)


async def run(args) -> dict:
    cfg = load_config()
    model = cfg.llm.for_model(args.model).model
    judge_model = None if args.no_judge else (args.judge or cfg.evals.judge)
    persona_rows, scenario_rows = [], []
    lm = LmStudio(cfg.llm.for_model(model))
    for other in lm.loaded_models() - {model}:  # Holly needs the GPU to himself
        print(f"  (unloading {other})", flush=True)
        lm.unload(other)
    lm.ensure_loaded(model)
    try:
        lm.ensure_loaded(cfg.recall.embed_model, context_length=2048)
    except Exception as e:  # recall by keyword and day only
        print(f"  (no embedding model: {e!r})", flush=True)
    if args.only in (None, "persona"):
        print(f"Persona prompts, on {model}:", flush=True)
        persona_rows = await ask_persona(scenarios.quiet_holly(model), args.persona_runs, args.session)
    if args.only in (None, "scenarios"):
        print(f"Scenarios, {args.runs} run{'s' * (args.runs != 1)} each, on {model}:", flush=True)
        scenario_rows = await run_scenarios(model, args.runs)
    if judge_model and persona_rows:
        print(f"Judging, on {judge_model}...", flush=True)
        judge_all(persona_rows, judge_model)
    return dict(model=model, judge=judge_model if persona_rows else None, label=args.label, commit=commit(),
                when=datetime.now().isoformat(timespec="seconds"), session=args.session,
                persona=persona_scores(persona_rows), scenarios=scenario_scores(scenario_rows),
                persona_rows=persona_rows, scenario_runs=scenario_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Holly on a model: character, tools and memory")
    parser.add_argument("--model", help="the LM Studio chat model to test (default: holly.toml's)")
    parser.add_argument("--judge", help="the judge's model (default: [evals] judge in holly.toml)")
    parser.add_argument("--runs", type=int, default=None, help="runs of each scenario (default: holly.toml's)")
    parser.add_argument("--persona-runs", type=int, default=1, help="replies per persona prompt (default 1)")
    parser.add_argument("--only", choices=["persona", "scenarios"], help="one part only")
    parser.add_argument("--session", action="store_true", help="the persona prompts as one conversation")
    parser.add_argument("--no-judge", action="store_true", help="skip the judge's scores")
    parser.add_argument("--label", default="", help="a name for the scorecard, to compare runs")
    parser.add_argument("--compare", action="store_true", help="tabulate the scorecards so far, and stop")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.compare:
        print(compare())
        return 0
    from .scenarios import holly_running  # the same check as the tests'

    if running := holly_running():
        print(f"Holly is running ({running[0][:80]}): close him first.")
        return 1
    args.runs = args.runs or load_config().evals.scenario_runs
    card = asyncio.run(run(args))
    path = write(card)
    print("\n".join(summary(card)) + f"\n-> {path.with_suffix('.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""His research reports, written as he writes them while he sleeps (app/research.py), and
judged: the background work had no evals at all, and it's the slowest, most stepwise thing
he does, so the most to lose quietly when a model or a source changes (the GDELT idea, on
the to-do list, wants a before and after too).

Each question in QUESTIONS is researched the real way (web search and Wikipedia, pages
read, notes, a draft checked against the notes), capped at MAX_MINUTES. Then each judge
reads what was asked, his notes (what the pages said) and the report, and scores it 1 to 5
for keeping to his notes, answering what was asked, and reading well; and quotes up to
three claims in it his notes don't have (checked to be in the report).
"""

from __future__ import annotations

import dataclasses
import re
import tempfile
import time
from pathlib import Path

from . import conversations

MAX_MINUTES = 15.0  # of work per report (45 when he's asleep): an eval must end
QUESTIONS = [  # (title, what the crew asked): everyday, not personal, not tied to this week's news
    ("Board games for a ten-year-old", "What are the best board games for a family with a ten-year-old? Some we can "
     "all play together, and a quick one for a school night."),
    ("Heat pumps in an old house", "How do air source heat pumps work, and are they worth it in an old stone house "
     "with no cavity walls?"),
]
POINTS = ("GROUNDED", "ANSWERS", "CLEAR")

JUDGE_PROMPT = """\
You judge a research report that Holly, a household computer, wrote for the people he lives with. He searched the \
web and Wikipedia, read pages, and took notes; then wrote the report from his notes. You're given what they asked \
for, his notes, and the report.

Score it 1 to 5 on each:
GROUNDED: every claim in the report is in his notes (5); a few small ones aren't (3); much of it isn't (1).
ANSWERS: it answers what they asked, fully and directly (5); partly (3); hardly (1).
CLEAR: plain English, well organised, the right length for what was asked (5); hard work to read (1).
UNSUPPORTED: up to three claims in the report that his notes don't have, each copied exactly from the report, in \
double quotes; or none.

Reply in exactly this form, a line each:
GROUNDED: n
ANSWERS: n
CLEAR: n
UNSUPPORTED: "claim" "claim" or none"""


def verdict(text: str, report: str) -> dict:
    """A judge's reply, read: each score (0 if missing), and the unsupported claims, those
    found in the report."""
    scores = {}
    for p in POINTS:
        m = re.search(rf"{p}\W*\s*:?\s*([1-5])", text, re.IGNORECASE)
        scores[p] = int(m.group(1)) if m else 0
    line = next((ln for ln in text.splitlines() if ln.strip().upper().startswith("UNSUPPORTED")), "")
    quotes = re.findall(r'"([^"]+)"', line.replace("“", '"').replace("”", '"'))
    return dict(scores=scores, quotes=quotes, shown=[q for q in quotes if conversations.found(q, [report])],
                text=text[:600])


async def write_one(holly, title: str, request: str) -> dict:
    """One report, researched as he would asleep: every step, until it's done or given up."""
    research = holly.research
    research.handle({"action": "start", "title": title, "request": request})
    job = next(j for j in research.jobs if j["title"] == title)
    t0 = time.monotonic()
    steps = 0
    while job in research.jobs:
        try:
            print(f"    {await research.step()}"[:160], flush=True)
        except Exception as e:  # counted by the job; given up after max_errors in a row
            print(f"    (a step failed: {e!r})"[:160], flush=True)
        steps += 1
    notes = research._all_notes(job) if job.get("questions") else ""
    return dict(title=title, request=request, stage=job["stage"], report=job.get("report") or "",
                notes=notes, sources=len(job.get("sources") or []), steps=steps,
                minutes=round((time.monotonic() - t0) / 60, 1), error=job.get("error", ""), verdicts={})


async def write_all(model: str | None) -> list[dict]:
    from . import scenarios

    rows = []
    for title, request in QUESTIONS:
        holly = scenarios.quiet_holly(model, Path(tempfile.mkdtemp(prefix="holly-report-")))
        if not holly.research:
            print("  (research is off: no reports)", flush=True)
            return []
        holly.research.cfg = dataclasses.replace(holly.research.cfg, max_minutes=MAX_MINUTES)
        print(f"  {title}:", flush=True)
        rows.append(await write_one(holly, title, request))
        r = rows[-1]
        print(f"  -> {r['stage']}: {len(r['report'].split())} words from {r['sources']} sources, {r['minutes']} min"
              + (f" ({r['error']})" if r["error"] else ""), flush=True)
    return rows


def judge(lm, model: str, rows: list[dict], thinking: str = "none") -> None:
    for r in rows:
        if not r["report"]:
            continue
        text = lm.ask(JUDGE_PROMPT, f"What they asked for: {r['request']}\n\nHis notes:\n\n{r['notes']}\n\n"
                                    f"The report:\n\n{r['report']}", model,
                      max_tokens=400 if thinking == "none" else 12000, temperature=0.3, reasoning=thinking)
        v = verdict(text, r["report"])
        r["verdicts"][model] = v
        print(f"  {model.split('/')[-1]:24} {r['title']}: {v['scores']}; unsupported {v['shown']}", flush=True)


def scores(rows: list[dict]) -> dict:
    """Each point's mean over every report and judge; reports written; unsupported claims."""
    if not rows:
        return {}
    every = [v for r in rows for v in r["verdicts"].values()]
    means = {p: round(sum(v["scores"][p] for v in every if v["scores"][p]) /
                      max(1, sum(bool(v["scores"][p]) for v in every)), 2) for p in POINTS}
    return dict(reports=len(rows), written=sum(bool(r["report"]) for r in rows), means=means,
                unsupported=sum(len(v["shown"]) for v in every),
                minutes=round(sum(r["minutes"] for r in rows), 1))

"""How well Holly stays Holly: the test prompts through his real reply loop, scored.

    python -m app.persona_eval [--runs N] [--label NAME] [--no-judge] [--session]

Each prompt in persona/test_prompts.txt is asked afresh, through the same reply loop as
app.listen (his tools and their rules included), with a silent voice and his memory,
timers and research in a temporary folder; Spotify is left off, so nothing plays.
Each reply is checked:

- for a voice: sentences and words (Holly is terse: in the shows, half his lines are a
  single sentence, of nine words or so), and anything a voice can't say;
- for an assistant's manners ("Certainly!", "How can I help?", "anything else?");
- for a question back: welcome where the prompt is tagged "ask", and counted elsewhere;
- for the tool a prompt is tagged with, called or not;
- across the run, for openings and phrases used again and again;
- with --session, the prompts are one conversation instead, so ruts show (and his
  check on them, Conversation.worn_phrases, is put to work);
- by a judge (the same Gemma, out of his sight), 1 to 5 for sounding like Holly. If
  data/scripts/holly_exchanges.json is there (his lines from the scripts, private and
  git-ignored), the judge is shown some of them for reference.

A Markdown report goes to runs/, with every reply, to read for character too.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import re
import statistics
import sys
import tempfile
import time
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ROOT, load_config
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


class SilentVoice:
    sample_rate = 48000

    async def stream(self, text, trim_lead=False):
        yield np.zeros(480, np.float32)


class SilentPlayer:
    first_played = first_voice = last_played = None
    underruns = 0

    def begin_turn(self):
        self.done = asyncio.Event()

    def put(self, audio):
        pass

    def expect_gap(self):
        pass

    def end_turn(self):
        self.done.set()

    def wake(self):
        pass

    async def wait_done(self):
        await self.done.wait()


def load_prompts() -> list[tuple[str, set[str]]]:
    out = []
    for line in PROMPTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip() if line.lstrip().startswith("#") else line.strip()
        if not line:
            continue
        text, _, tags = line.partition("||")
        out.append((text.strip(), {t.strip() for t in tags.split(",") if t.strip()}))
    return out


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


async def run(args) -> dict:
    from .speak import Holly  # here: it imports torch and the rest

    tmp = Path(tempfile.mkdtemp(prefix="holly-eval-"))
    cfg = load_config()
    cfg = dataclasses.replace(
        cfg,
        persona=dataclasses.replace(cfg.persona, memory_file=tmp / "memory.json"),
        recall=dataclasses.replace(cfg.recall, file=tmp / "archive.db"),
        timers=dataclasses.replace(cfg.timers, file=tmp / "timers.json"),
        research=dataclasses.replace(cfg.research, dir=tmp / "research"),
        spotify=dataclasses.replace(cfg.spotify, enabled=False))
    holly = Holly(None, face=False, cfg=cfg)
    holly.tts, holly.player = SilentVoice(), SilentPlayer()
    holly.reminded = True  # no reminders from the records: the same prompt every time
    holly.lm.ensure_loaded(holly.model)
    reference = "" if args.no_judge else reference_lines()
    with open(cfg.persona.examples, "rb") as f:
        examples = [e["holly"] for e in tomllib.load(f).get("example", [])]
    rows = []
    for prompt, tags in load_prompts():
        for _ in range(args.runs):
            if not args.session:
                holly.convo.reset()
            before = len(holly.convo.history)
            t0 = time.perf_counter()
            await holly.respond(prompt)
            seconds = time.perf_counter() - t0
            history = holly.convo.history
            reply = (history[-1].get("content") or "").strip() if history else ""
            tools = [c["function"]["name"] for m in history[before:] for c in m.get("tool_calls") or []]
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
            score, why = judge(holly.lm, holly.model, reference, prompt, reply) if not args.no_judge else (0, "")
            rows.append(dict(prompt=prompt, tags=sorted(tags), reply=reply, tools=tools, words=words,
                             sentences=sentences, asked="?" in reply, problems=problems, score=score, why=why,
                             seconds=seconds))
            print(f"  {score or '-'} {'ok ' if not problems else 'BAD'} {prompt[:40]:40} {reply[:90]}", flush=True)
    return dict(rows=rows, label=args.label)


def reused(reply: str, examples: list[str], n: int = 6) -> str:
    """Six words in a row from one of his examples, if the reply has them: they show his
    voice, they aren't lines to recite (he recited them, when they matched the prompts)."""
    words = re.findall(r"[a-z']+", reply.lower())
    seen = {" ".join(ws[i:i + n]) for e in examples for ws in [re.findall(r"[a-z']+", e.lower())]
            for i in range(len(ws) - n + 1)}
    return next((" ".join(words[i:i + n]) for i in range(len(words) - n + 1) if " ".join(words[i:i + n]) in seen), "")


def openings(rows: list[dict]) -> list[tuple[str, int]]:
    """Openings (first three words) and phrases (four words) used in more than one reply."""
    first = Counter(" ".join(r["reply"].lower().split()[:3]) for r in rows if r["reply"])
    grams = Counter()
    for r in rows:
        ws = re.findall(r"[a-z']+", r["reply"].lower())
        grams.update({" ".join(ws[i:i + 4]) for i in range(len(ws) - 3)})
    return ([(f"opens '{k}'", v) for k, v in first.most_common() if v > 1]
            + [(f"'{k}'", v) for k, v in grams.most_common(12) if v > 2])


def report(result: dict) -> str:
    rows = result["rows"]
    scored = [r["score"] for r in rows if r["score"]]
    ask = [r for r in rows if "ask" in r["tags"]]
    other = [r for r in rows if "ask" not in r["tags"]]
    tooled = [r for r in rows if set(r["tags"]) - {"ask"}]
    plain = [r for r in rows if not any(w in r["prompt"].lower() for w in CREATIVE)]
    summary = [
        f"Judge (sounds like Holly, 1-5): mean {statistics.mean(scored):.2f}" if scored else "Judge: off",
        f"Passing (length, speakable, no assistant manners, right tool): "
        f"{sum(not r['problems'] for r in rows)} of {len(rows)}",
        f"Words per ordinary reply: median {statistics.median([r['words'] for r in plain]):.0f} "
        f"(in the shows, 9); one sentence: {sum(r['sentences'] == 1 for r in plain) / len(plain):.0%} (56%)",
        f"Questions back: {sum(r['asked'] for r in ask)} of {len(ask)} where one suits; "
        f"{sum(r['asked'] for r in other)} of {len(other)} elsewhere",
        f"Tools called as they should be: {sum(not any(p.startswith('didn') for p in r['problems']) for r in tooled)}"
        f" of {len(tooled)}",
        f"Assistant manners: {sum(any(p.startswith('assistant') for p in r['problems']) for r in rows)}; "
        f"examples reused: {sum(any(p.startswith('reused') for p in r['problems']) for r in rows)}",
        "Used more than once: " + ("; ".join(f"{k} x{v}" for k, v in openings(rows)) or "nothing"),
    ]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"persona-{result['label'] or 'run'}-{stamp}.md"
    lines = [f"# Persona eval: {result['label'] or stamp}", "", *[f"- {s}" for s in summary], ""]
    for r in rows:
        lines += [f"**{r['prompt']}**" + (f" ({', '.join(r['tags'])})" if r["tags"] else "") + "  ",
                  f"{r['reply']}  ",
                  f"_{r['score'] or '-'}/5 {r['why']} | {r['words']} words"
                  + (f" | tools: {', '.join(r['tools'])}" if r["tools"] else "")
                  + (f" | {'; '.join(r['problems'])}" if r["problems"] else "") + "_", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(summary) + f"\n-> {path.relative_to(ROOT)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score how well Holly stays Holly")
    parser.add_argument("--runs", type=int, default=1, help="replies per prompt (default 1)")
    parser.add_argument("--label", default="", help="a name for the report, to compare runs")
    parser.add_argument("--no-judge", action="store_true", help="skip the judge's scores")
    parser.add_argument("--session", action="store_true", help="one conversation, not each prompt afresh")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report(asyncio.run(run(args))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

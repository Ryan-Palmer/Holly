"""Milestone 1 check: run the test prompts through Holly and score the replies.

    python -m app.persona_eval [--model MODEL ...] [--runs N]

Each prompt is asked fresh, with no history. A reply passes when it is one to
three sentences (up to eight for a poem or story, which he may now tell) and
has nothing a TTS voice can't say. Writes a Markdown
report per model to runs/ so the replies can be read for character too, which
no script can judge.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime

from .config import ROOT, load_config
from .lmstudio import LmStudio, ReplyTiming
from .persona import Conversation
from .speech_text import count_sentences, speech_problems

PROMPTS = ROOT / "persona" / "test_prompts.txt"
CREATIVE = ("poem", "story", "limerick")  # prompts he may answer at more length
RUNS = ROOT / "runs"


def load_prompts() -> list[str]:
    lines = PROMPTS.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def evaluate(lm: LmStudio, convo: Conversation, model: str, prompts: list[str], runs: int) -> dict:
    rows = []
    for prompt in prompts:
        for _ in range(runs):
            timing = ReplyTiming()
            reply = "".join(lm.stream_reply(convo.messages_for(prompt), timing, model=model)).strip()
            sentences = count_sentences(reply)
            problems = speech_problems(reply)
            longest = 8 if any(w in prompt.lower() for w in CREATIVE) else 3
            if not 1 <= sentences <= longest:
                problems.append(f"{sentences} sentences")
            if timing.reasoning_chars:
                problems.append("thought before answering")
            rows.append(
                dict(prompt=prompt, reply=reply, problems=problems, first_ms=timing.first_token_ms or 0)
            )
            mark = "ok " if not problems else "FAIL"
            print(f"  {mark} {timing.first_token_ms or 0:5.0f} ms  {prompt[:50]}")
    return dict(model=model, rows=rows)


def write_report(result: dict) -> str:
    rows = result["rows"]
    passed = sum(1 for r in rows if not r["problems"])
    firsts = [r["first_ms"] for r in rows]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"persona-{result['model'].replace('/', '_')}-{stamp}.md"
    lines = [
        f"# Persona eval: {result['model']}",
        "",
        f"{passed} of {len(rows)} replies pass. First token median "
        f"{statistics.median(firsts):.0f} ms, worst {max(firsts):.0f} ms.",
        "",
    ]
    for r in rows:
        status = "pass" if not r["problems"] else "FAIL: " + ", ".join(r["problems"])
        lines += [f"**{r['prompt']}**  ", f"{r['reply']}  ", f"_{status}, {r['first_ms']:.0f} ms_", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"{passed}/{len(rows)} pass, first token median {statistics.median(firsts):.0f} ms -> {path.relative_to(ROOT)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Holly's replies to the test prompts")
    parser.add_argument("--model", action="append", help="model key; repeat to compare (default: holly.toml)")
    parser.add_argument("--runs", type=int, default=1, help="replies per prompt (default 1)")
    args = parser.parse_args()
    # The Windows console defaults to cp1252, which can't print emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config()
    models = args.model or [cfg.llm.model]
    lm = LmStudio(cfg.llm)
    convo = Conversation(cfg.persona)
    prompts = load_prompts()

    all_passed = True
    summaries = []
    for i, model in enumerate(models):
        if i > 0:
            # Two 30B-class models don't fit in VRAM together.
            lm.unload(models[i - 1])
        print(f"{model}: loading...")
        lm.ensure_loaded(model)
        # Warm up so the first prompt's timing isn't a cold start.
        "".join(lm.stream_reply(convo.messages_for("Hello."), ReplyTiming(), model=model))
        result = evaluate(lm, convo, model, prompts, args.runs)
        summaries.append(f"{model}: {write_report(result)}")
        all_passed &= all(not r["problems"] for r in result["rows"])

    print()
    print("\n".join(summaries))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

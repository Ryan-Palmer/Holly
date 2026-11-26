"""Can his voice say what he wrote? His replies from an evals run, said in his own voice
(the VoxCPM2 clone, sentence by sentence, as he speaks) and heard back by Whisper, word for
word against what he wrote. "Yeah?" alone came out garbled, found only by ear: a word, a
number, a name or a unit his voice can't manage shows up here, as words that don't come
back. Numbers count the same in digits or words; punctuation and case don't count.

    python -m app.voice_check runs/evals-<label>-<stamp>.json    # adds "voice" to that scorecard

Run by app/evals.py in a process of its own, after the scorecard is written: the voice's
warm-up is where this PC's crashes come (memory: the start-up crash), and the rest of a
long run shouldn't go with it.
"""

from __future__ import annotations

import re

from .sourced import from_words

LIMIT = 120  # replies said, at most: ~3 s each
FLAG_AT = 0.2  # word error rate at or over which a reply is listed ("Lasagne" garbled in four words: 0.25)


def words(text: str) -> list[str]:
    """`text` as words to compare: lower case, no punctuation, every number as digits (a
    run of number words, "seventy eight", as "78"; a time "20:39" as "20 39")."""
    # A price, "£1.80" from Whisper, "one pound eighty" from him: "1 pound 80" either way.
    text = re.sub(r"£(\d+)\.(\d{2})\b", r"\1 pound \2", text.lower().replace("’", "'"))
    text = re.sub(r"£(\d+)", r"\1 pound", re.sub(r"\bpounds\b", "pound", text))
    # A time, written "12:19" by him and "12.19" by Whisper: hours and minutes, either way.
    text = re.sub(r"\b(\d{1,2})[:.](\d{2})\b", r"\1 \2", text)
    text = text.replace("per cent", "percent").replace("%", " percent")  # as Whisper writes it
    out: list[str] = []
    run: list[str] = []
    numberish = re.compile(r"^(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                           r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                           r"eighty|ninety|hundred|thousand|million|and)$")

    def flush():
        if run:
            nums = from_words(" ".join(run))
            out.extend(str(int(n)) if n == int(n) else str(n) for n in nums)
            if not nums:
                out.extend(run)
            run.clear()

    for w in re.findall(r"[a-z']+|\d+(?:\.\d+)?", text.replace(",", "")):
        if numberish.match(w) and (run or w != "and"):
            run.append(w)
        else:
            flush()
            if w[0].isdigit():  # "07" and "7", "3.0" and "3": the same number
                n = float(w)
                w = str(int(n)) if n == int(n) else str(n)
            out.append(w.strip("'"))
    flush()
    return [w for w in out if w]


def wer(said: str, heard: str) -> float:
    """Word error rate of `heard` against `said`: edits over the words said."""
    a, b = words(said), words(heard)
    if not a:
        return 0.0 if not b else 1.0
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return round(row[-1] / len(a), 3)


def replies(card: dict) -> list[str]:
    """A scorecard's replies to say: the persona's, and the first run of each conversation,
    the long talk, the made-up-memory questions and each scenario; each once."""
    out: list[str] = [r["reply"] for r in card.get("persona_rows") or []]
    for key in ("conversation_runs", "long_talk_runs", "scenario_runs"):
        out += [t["reply"] for r in card.get(key) or [] if r.get("run", 1) == 1 for t in r["turns"]]
    out += [r["reply"] for r in card.get("honesty_rows") or [] if r.get("run", 1) == 1]
    seen, kept = set(), []
    for text in out:
        if text and text not in seen and not text.startswith("("):
            seen.add(text)
            kept.append(text)
    return kept[:LIMIT]


async def round_trip(cfg, texts: list[str]) -> list[dict]:
    """Each of `texts` said in his voice, sentence by sentence with a short gap, and heard."""
    import asyncio

    import numpy as np
    import torch
    import torchaudio

    from .speech_text import split_sentences
    from .stt import Transcriber
    from .tts_common import make_tts

    from . import cpu

    # The voice's start-up on the efficiency cores, whatever [cpu] says: on this PC's
    # performance cores it crashed (0xC0000005) half the time; an eval can wait.
    with cpu.starting_on("efficiency"):
        tts = await asyncio.to_thread(make_tts, cfg)
        await tts.warm_up()
        stt = Transcriber(cfg.listen)
        stt.warm_up()
    rows = []
    for text in texts:
        chunks: list[np.ndarray] = []
        for sentence in split_sentences(text) or [text]:
            async for chunk in tts.stream(sentence):
                chunks.append(chunk)
            chunks.append(np.zeros(int(0.25 * tts.sample_rate), np.float32))
        audio = torchaudio.functional.resample(torch.from_numpy(np.concatenate(chunks)), tts.sample_rate, 16_000)
        heard = await asyncio.to_thread(stt.transcribe, audio.numpy().astype(np.float32))
        rows.append(dict(said=text, heard=heard, wer=wer(text, heard)))
        print(f"  {rows[-1]['wer']:.2f}  {text[:70]}  |  {heard[:70]}", flush=True)
    return rows


def scores(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return dict(replies=len(rows), mean_wer=round(sum(r["wer"] for r in rows) / len(rows), 3),
                flagged=[r for r in rows if r["wer"] >= FLAG_AT], rows=rows)


def main() -> None:
    import asyncio
    import json
    import sys
    from pathlib import Path

    from .config import load_config
    from .evals import write

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1])
    card = json.loads(path.read_text(encoding="utf-8"))
    texts = replies(card)
    print(f"His voice, {len(texts)} replies, said and heard back:", flush=True)
    rows = asyncio.run(round_trip(load_config(), texts))
    card["voice"] = scores(rows)
    write(card, path.with_suffix(""))
    print(f"Mean word error {card['voice']['mean_wer']}; {len(card['voice']['flagged'])} replies at "
          f"{FLAG_AT} or more -> {path.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()

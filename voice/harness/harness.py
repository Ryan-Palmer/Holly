"""TTS evaluation harness: run candidates, score them blind, report.

Run with the app's venv:

    .venv\\Scripts\\python voice\\harness\\harness.py list            candidates and whether each is installed
    .venv\\Scripts\\python voice\\harness\\harness.py run [NAME ...]  synthesize the test sentences with each
    .venv\\Scripts\\python voice\\harness\\harness.py merge RUN RUN   combine runs, to score them together
                                                     (--into RUN adds to a scored run)
    .venv\\Scripts\\python voice\\harness\\harness.py score [RUN]     listen blind and rate each sample
    .venv\\Scripts\\python voice\\harness\\harness.py report [RUN]    reveal: scores and timings per candidate

Each candidate runs in its own venv at voice/harness/envs/NAME, through its
adapter in voice/harness/adapters/. They all read the reference clip from
voice/data/holly/ (made by voice/prep.py export). Runs go to
voice/data/harness/RUN/, which git ignores. RUN defaults to the latest.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import subprocess
import sys
import tomllib
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOICE = HERE.parent
REF_WAV = VOICE / "data" / "holly" / "reference.wav"
REF_TXT = VOICE / "data" / "holly" / "reference.txt"
RUNS = VOICE / "data" / "harness"
SENTENCES = HERE / "sentences.txt"


def candidates() -> dict[str, dict]:
    with open(HERE / "candidates.toml", "rb") as f:
        return tomllib.load(f)["candidate"]


def venv_python(name: str) -> Path:
    """Candidates can share a venv (e.g. two variants of one package)."""
    venv = candidates().get(name, {}).get("venv", name)
    return HERE / "envs" / venv / "Scripts" / "python.exe"


def run_dir(run: str | None) -> Path:
    if run:
        return RUNS / run
    runs = sorted(p for p in RUNS.glob("*") if p.is_dir())
    if not runs:
        sys.exit("No runs yet: use `run` first.")
    return runs[-1]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return audio, w.getframerate()


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def match_loudness(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Same RMS level for every blind sample: louder tends to sound better,
    which would bias the scores."""
    rms = np.sqrt(np.mean(audio**2)) if audio.size else 0.0
    if rms < 1e-6:
        return audio
    return audio * (10 ** (target_db / 20) / rms)


# Commands ------------------------------------------------------------------


def cmd_list() -> None:
    for name, c in candidates().items():
        ok = venv_python(name).exists()
        print(f"  {name:22s} {'installed' if ok else 'not installed':14s} {c.get('summary', '')}")


def cmd_run(names: list[str], keep_lm: bool) -> None:
    if not REF_WAV.exists():
        sys.exit(f"No reference clip at {REF_WAV}: run voice/prep.py export first.")
    all_c = candidates()
    names = names or [n for n in all_c if venv_python(n).exists()]
    if not names:
        sys.exit("No candidates installed. See voice/harness/README.md.")
    if not keep_lm and shutil.which("lms"):
        # Timings mean nothing if VRAM spills into system RAM.
        print("Unloading LM Studio models to free VRAM (--keep-lm to skip)...")
        subprocess.run(["lms", "unload", "--all"], capture_output=True)

    out = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in names:
        if name not in all_c:
            print(f"{name}: not in candidates.toml, skipped")
            continue
        py = venv_python(name)
        if not py.exists():
            print(f"{name}: not installed ({py.parent.parent} missing), skipped")
            continue
        print(f"{name}: running...", flush=True)
        r = subprocess.run([
            str(py), str(HERE / "adapters" / all_c[name]["adapter"]),
            "--ref", str(REF_WAV), "--ref-text", str(REF_TXT),
            "--sentences", str(SENTENCES), "--out", str(out / name),
        ])
        if r.returncode:
            print(f"{name}: FAILED (exit {r.returncode})")
            shutil.rmtree(out / name, ignore_errors=True)

    done = [p.name for p in out.glob("*") if (p / "timings.json").exists()]
    if not done:
        sys.exit("Nothing succeeded.")
    make_blind(out, done)
    print(f"\nRun {out.name}: {', '.join(done)}. Next: score")


def cmd_merge(runs: list[str], into: str | None) -> None:
    """Combine candidates from several runs into one new run, to score together.
    With `into`, add them to an existing run instead: its scores are kept, and
    `score` then asks only about the new samples."""
    if into:
        out = RUNS / into
        key = json.loads((out / "key.json").read_text())
        have = {k["candidate"] for k in key.values()}
        added = []
        for run in runs:
            for cand in sorted(p for p in (RUNS / run).glob("*") if (p / "timings.json").exists()):
                if cand.name not in have and cand.name not in added:
                    shutil.copytree(cand, out / cand.name)
                    added.append(cand.name)
        if not added:
            sys.exit("Nothing new to add.")
        make_blind(out, added, key)
        print(f"Added {', '.join(added)} to {out.name}. Next: score {out.name}")
        return
    out = RUNS / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-merged")
    names = []
    for run in runs:
        for cand in sorted(p for p in (RUNS / run).glob("*") if (p / "timings.json").exists()):
            if cand.name in names:
                print(f"{cand.name}: already taken from an earlier run, skipped from {run}")
                continue
            shutil.copytree(cand, out / cand.name)
            names.append(cand.name)
    if not names:
        sys.exit("Nothing to merge.")
    make_blind(out, names)
    print(f"Run {out.name}: {', '.join(names)}. Next: score")


def make_blind(out: Path, names: list[str], key: dict | None = None) -> None:
    """Copy every sample to blind/ under a random id; the key stays aside.
    Pass an existing key to add samples to a run without reusing its ids."""
    blind = out / "blind"
    blind.mkdir(exist_ok=True)
    items = [(n, w.name) for n in names for w in sorted((out / n).glob("*.wav"))]
    key = key or {}
    free = sorted(set(range(1000, 10000)) - {int(i) for i in key})
    ids = random.sample(free, len(items))
    for (name, wav), i in zip(items, ids):
        audio, rate = read_wav(out / name / wav)
        write_wav(blind / f"{i}.wav", match_loudness(audio), rate)
        key[str(i)] = {"candidate": name, "sentence": int(wav[:2])}
    (out / "key.json").write_text(json.dumps(key, indent=1))


def cmd_score(run: str | None) -> None:
    import msvcrt

    import sounddevice as sd

    out = run_dir(run)
    key = json.loads((out / "key.json").read_text())
    scores_path = out / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    todo = [i for i in key if i not in scores]
    random.shuffle(todo)
    ref, ref_rate = read_wav(REF_WAV)
    print(__doc_score__)
    print(f"{len(todo)} samples to score ({len(scores)} already done).\n")
    print("Reference (the real Holly):")
    sd.play(ref, ref_rate)

    def ask(label: str) -> str | None:
        print(f"  {label} 1-5: ", end="", flush=True)
        while True:
            k = msvcrt.getwch().lower()
            if k in "12345":
                print(k)
                return k
            if k in ("q", "r", " ", "\r", "p"):
                return k

    n = 0
    while n < len(todo):
        sid = todo[n]
        audio, rate = read_wav(out / "blind" / f"{sid}.wav")
        print(f"\n[{len(scores) + 1}/{len(key)}] sample {sid}")
        sd.play(audio, rate)
        got = {}
        for label in ("similarity", "naturalness"):
            while label not in got:
                k = ask(label)
                if k == "q":
                    sd.stop()
                    print("\nSaved. Run score again to carry on.")
                    return
                if k == "r":
                    sd.play(ref, ref_rate)
                elif k in (" ", "\r", "p"):
                    sd.play(audio, rate)
                else:
                    got[label] = int(k)
        scores[sid] = got
        scores_path.write_text(json.dumps(scores, indent=1))
        n += 1
    sd.stop()
    print(f"\nAll {len(key)} scored. Next: report")


__doc_score__ = """Rate each sample twice, 1 (bad) to 5 (great):
  similarity: does it sound like Norman's Holly?
  naturalness: does it sound like a person talking, with no glitches?
Keys: 1-5 score, space replay sample, r replay the reference, q save and quit."""


def cmd_report(run: str | None) -> None:
    out = run_dir(run)
    key = json.loads((out / "key.json").read_text())
    scores_path = out / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    rows = []
    for name in sorted({k["candidate"] for k in key.values()}):
        t = json.loads((out / name / "timings.json").read_text())
        res, meta = t["results"], t["meta"]
        mine = [scores[i] for i, k in key.items() if k["candidate"] == name and i in scores]
        mean = lambda xs: statistics.mean(xs) if xs else float("nan")  # noqa: E731
        rows.append(dict(
            name=name,
            sim=mean([s["similarity"] for s in mine]),
            nat=mean([s["naturalness"] for s in mine]),
            scored=len(mine),
            first=statistics.median(r["first_audio_ms"] for r in res),
            rtf=statistics.median(r["rtf"] for r in res if r["rtf"]),
            vram=meta["peak_vram_gb"],
            streams=meta["streams"],
        ))
    rows.sort(key=lambda r: -(r["sim"] + r["nat"]) if r["scored"] else 0)
    header = "| Candidate | Similarity | Naturalness | Scored | First audio (median) | RTF (median) | Peak VRAM | Streams |"
    lines = [header, "|" + "---|" * 8]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['sim']:.2f} | {r['nat']:.2f} | {r['scored']} | {r['first']:.0f} ms | "
            f"{r['rtf']:.2f} | {r['vram']:.1f} GB | {'yes' if r['streams'] else 'no'} |"
        )
    report = "\n".join(lines)
    (out / "report.md").write_text(f"# TTS harness run {out.name}\n\n{report}\n", encoding="utf-8")
    print(report)
    print("\nRTF is synthesis time over audio length: under 1 is faster than real time. "
          "First audio for a non-streaming model is the whole sentence's synthesis time.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("run")
    s.add_argument("names", nargs="*")
    s.add_argument("--keep-lm", action="store_true", help="don't unload LM Studio's models first")
    s = sub.add_parser("merge")
    s.add_argument("runs", nargs="+")
    s.add_argument("--into", help="add to this existing run instead of making a new one")
    sub.add_parser("score").add_argument("run", nargs="?")
    sub.add_parser("report").add_argument("run", nargs="?")
    a = p.parse_args()
    if a.cmd == "list":
        cmd_list()
    elif a.cmd == "run":
        cmd_run(a.names, a.keep_lm)
    elif a.cmd == "merge":
        cmd_merge(a.runs, a.into)
    elif a.cmd == "score":
        cmd_score(a.run)
    elif a.cmd == "report":
        cmd_report(a.run)


if __name__ == "__main__":
    main()

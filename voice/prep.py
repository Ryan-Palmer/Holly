"""Build Holly's voice dataset from recordings: one command per stage.

Run with the dataset tools' own venv (voice/.venv), not the app's:

    voice\\.venv\\Scripts\\python voice\\prep.py fetch URL         download a video's audio
    voice\\.venv\\Scripts\\python voice\\prep.py separate ID       pull the voice off music and laughter
    voice\\.venv\\Scripts\\python voice\\prep.py segment ID        cut into lines, transcribe, group by speaker
    voice\\.venv\\Scripts\\python voice\\prep.py trim ID           cut rejected lines down to their words (drops laughter)
    voice\\.venv\\Scripts\\python voice\\prep.py review ID         listen, and keep or reject each line
    voice\\.venv\\Scripts\\python voice\\prep.py export            write the kept lines and the reference clip

Everything lands under voice/data/, which git ignores. ID is the source's
folder name under voice/data/raw/ (for a YouTube video, its video id).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
WORK = DATA / "work"
OUT = DATA / "holly"

SEP_RATE = 44_100  # Demucs works at 44.1 kHz
VAD_RATE = 16_000
EXPORT_RATE = 24_000  # what most TTS models train on and Orpheus/SNAC uses


# Audio helpers -----------------------------------------------------------


def decode(path: Path, rate: int, channels: int) -> np.ndarray:
    """Any audio or video file to float32 (channels, samples), via PyAV."""
    import av

    layout = "mono" if channels == 1 else "stereo"
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    return np.concatenate(chunks, axis=1).astype(np.float32)


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(audio), src, dst).numpy()


def rms_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(audio**2)), 1e-9)))


def work_dir(source_id: str) -> Path:
    return WORK / source_id


def load_manifest(source_id: str) -> dict:
    return json.loads((work_dir(source_id) / "manifest.json").read_text(encoding="utf-8"))


def save_manifest(source_id: str, manifest: dict) -> None:
    path = work_dir(source_id) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


# Stages ------------------------------------------------------------------


def fetch(url: str) -> None:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    folder = RAW / info["id"]
    folder.mkdir(parents=True, exist_ok=True)
    opts = {"format": "bestaudio", "noplaylist": True, "outtmpl": str(folder / "source.%(ext)s"), "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    (folder / "info.txt").write_text(
        f"{info.get('title')} | {info.get('duration')} s | {info.get('uploader')} | {url}\n", encoding="utf-8"
    )
    print(f"Saved to {folder.relative_to(ROOT.parent)}. Next: separate {info['id']}")


def separate(source_id: str, model_name: str) -> None:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    source = next((RAW / source_id).glob("source.*"))
    mix = torch.from_numpy(decode(source, SEP_RATE, 2))
    model = get_model(model_name).eval()
    # Demucs expects the mix normalised, as its own CLI does.
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std()
    with torch.inference_mode():
        stems = apply_model(model, ((mix - mean) / std)[None], device="cuda", split=True, overlap=0.25)[0]
    vocals = stems[model.sources.index("vocals")] * std + mean
    out = work_dir(source_id)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "vocals.wav", vocals.mean(0).cpu().numpy(), SEP_RATE)
    print(f"Vocals: {mix.shape[1] / SEP_RATE:.0f} s -> {(out / 'vocals.wav').relative_to(ROOT.parent)}. "
          f"Next: segment {source_id}")


def segment(source_id: str, max_line_s: float, merge_gap_s: float) -> None:
    import torch
    from faster_whisper import WhisperModel
    from silero_vad import get_speech_timestamps, load_silero_vad

    out = work_dir(source_id)
    vocals, rate = sf.read(out / "vocals.wav", dtype="float32")
    v16 = resample(vocals, rate, VAD_RATE)

    spans = get_speech_timestamps(
        torch.from_numpy(v16), load_silero_vad(), sampling_rate=VAD_RATE,
        threshold=0.5, min_speech_duration_ms=300, min_silence_duration_ms=250,
        speech_pad_ms=80, return_seconds=True,
    )
    # Join pieces of one line split by a short pause.
    lines: list[list[float]] = []
    for s in spans:
        if lines and s["start"] - lines[-1][1] < merge_gap_s and s["end"] - lines[-1][0] <= max_line_s:
            lines[-1][1] = s["end"]
        else:
            lines.append([s["start"], s["end"]])

    whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    embedder = SpeakerEmbedder()
    clips_dir = out / "clips"
    clips_dir.mkdir(exist_ok=True)
    clips = []
    for i, (start, end) in enumerate(lines):
        audio = vocals[int(start * rate) : int(end * rate)]
        name = f"{source_id}_{i:03d}.wav"
        sf.write(clips_dir / name, audio, rate)
        a16 = v16[int(start * VAD_RATE) : int(end * VAD_RATE)]
        segs, _ = whisper.transcribe(a16, language="en", beam_size=5, without_timestamps=True,
                                     condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        clips.append(dict(file=name, start=round(start, 2), end=round(end, 2), duration=round(end - start, 2),
                          level_db=round(rms_db(audio), 1), text=text, status="pending", reference=False,
                          embedding=embedder(a16)))
        print(f"  {name} {end - start:5.1f}s  {text}")

    label_speakers(clips)
    for c in clips:
        del c["embedding"]
    save_manifest(source_id, {"source": source_id, "rate": rate, "clips": clips})
    counts = {}
    for c in clips:
        counts[c["speaker"]] = counts.get(c["speaker"], 0) + c["duration"]
    print(f"{len(clips)} lines. Speaker groups by total time: "
          + ", ".join(f"{k} {v:.0f}s" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"Next: review {source_id}")


def trim(source_id: str, statuses: list[str], pad_start: float, pad_end: float) -> None:
    """Cut lines down to their words, dropping laughter before and after.

    Demucs keeps audience laughter in the vocal stem (it's human voice) and
    the VAD hears it as speech, so lines run on into the laugh after the
    punchline. Whisper doesn't transcribe laughter as words, so its word
    timings mark where the speech really starts and ends. Trimmed lines go
    back to pending, to be reviewed again.
    """
    from faster_whisper import WhisperModel

    manifest = load_manifest(source_id)
    out = work_dir(source_id)
    vocals, rate = sf.read(out / "vocals.wav", dtype="float32")
    whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    fade = int(0.02 * rate)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    for c in manifest["clips"]:
        if c["status"] not in statuses or c["speaker"] not in ("A", "?"):
            continue
        a16 = resample(vocals[int(c["start"] * rate) : int(c["end"] * rate)], rate, VAD_RATE)
        segs, _ = whisper.transcribe(a16, language="en", beam_size=5, word_timestamps=True,
                                     condition_on_previous_text=False)
        # Whisper writes laughter, when it notices it, as "(laughs)" or "[Laughter]".
        words = [w for s in segs for w in (s.words or []) if any(ch.isalnum() for ch in w.word)
                 and not w.word.strip().startswith(("(", "["))]
        if not words:
            continue
        start = max(0.0, c["start"] + words[0].start - pad_start)
        end = min(c["end"], c["start"] + words[-1].end + pad_end)
        audio = vocals[int(start * rate) : int(end * rate)].copy()
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
        sf.write(out / "clips" / c["file"], audio, rate)
        cut = c["duration"] - (end - start)
        c.update(start=round(start, 2), end=round(end, 2), duration=round(end - start, 2),
                 level_db=round(rms_db(audio), 1), text="".join(w.word for w in words).strip(),
                 status="pending", trimmed=True)
        print(f"  {c['file']}  cut {cut:4.1f}s -> {c['duration']:4.1f}s  {c['text']}")
    save_manifest(source_id, manifest)
    print(f"Next: review {source_id}")


class SpeakerEmbedder:
    """ECAPA voice embeddings, to group lines by who's speaking."""

    def __init__(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.torch = torch
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(DATA / "models" / "ecapa"),
            run_opts={"device": "cuda"},
            local_strategy=LocalStrategy.COPY,  # Windows can't symlink without admin
        )

    def __call__(self, a16: np.ndarray) -> list[float] | None:
        if len(a16) < VAD_RATE:  # under a second is too little to tell voices apart
            return None
        with self.torch.inference_mode():
            e = self.model.encode_batch(self.torch.from_numpy(a16)[None].cuda())[0, 0]
        return (e / e.norm()).cpu().tolist()


def label_speakers(clips: list[dict], threshold: float = 0.45) -> None:
    """Greedy clustering on cosine similarity. Groups are named A, B, C... by
    total speaking time, so in a Holly compilation A should be Holly. Lines
    too short to embed are labelled '?'."""
    groups: list[dict] = []
    for c in clips:
        e = c["embedding"]
        if e is None:
            c["speaker"] = "?"
            continue
        e = np.array(e)
        best = max(groups, key=lambda g: float(g["centroid"] @ e), default=None)
        if best is not None and float(best["centroid"] @ e) / np.linalg.norm(best["centroid"]) >= threshold:
            best["members"].append(c)
            best["sum"] += e
            best["centroid"] = best["sum"] / len(best["members"])
        else:
            groups.append({"members": [c], "sum": e.copy(), "centroid": e.copy()})
    groups.sort(key=lambda g: -sum(m["duration"] for m in g["members"]))
    for label, g in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", groups):
        for m in g["members"]:
            m["speaker"] = label


def review(source_id: str, speaker: str | None, redo: bool) -> None:
    import msvcrt

    import sounddevice as sd

    manifest = load_manifest(source_id)
    clips_dir = work_dir(source_id) / "clips"
    todo = [c for c in manifest["clips"]
            if (redo or c["status"] == "pending") and (speaker is None or c["speaker"] == speaker)]
    print(__doc_review__)
    i = 0
    while 0 <= i < len(todo):
        c = todo[i]
        audio, rate = sf.read(clips_dir / c["file"], dtype="float32")
        print(f"\n[{i + 1}/{len(todo)}] {c['file']}  speaker {c['speaker']}  {c['duration']:.1f}s  "
              f"{c['level_db']:.0f} dB  status {c['status']}{'  REFERENCE' if c['reference'] else ''}")
        print(f"  \"{c['text']}\"")
        sd.play(audio, rate)
        while True:
            key = msvcrt.getwch().lower()
            if key in ("k", "r"):
                c["status"] = "keep" if key == "k" else "reject"
                if key == "r":
                    c["reference"] = False
                print(f"  -> {c['status']}")
                i += 1
                break
            if key == "f":
                c["reference"] = not c["reference"]
                if c["reference"]:
                    c["status"] = "keep"
                print(f"  -> reference {'on' if c['reference'] else 'off'} (and kept)")
                i += 1
                break
            if key == "e":
                sd.stop()
                new = input("  corrected text: ").strip()
                if new:
                    c["text"] = new
                print(f"  \"{c['text']}\"")
                continue
            if key == "b":
                i = max(0, i - 1)
                break
            if key in (" ", "\r", "p"):
                sd.play(audio, rate)
                continue
            if key == "q":
                i = -1
                break
        sd.stop()
        save_manifest(source_id, manifest)
    done = sum(c["status"] != "pending" for c in manifest["clips"])
    kept = [c for c in manifest["clips"] if c["status"] == "keep"]
    print(f"\nReviewed {done}/{len(manifest['clips'])}. Kept {len(kept)} lines, "
          f"{sum(c['duration'] for c in kept):.0f} s; {sum(c['reference'] for c in kept)} marked as reference.")


__doc_review__ = """Keys: k keep, r reject, f keep and use in the reference clip,
      e correct the transcript, space replay, b back, q quit (progress is saved)"""


def export(reference_max_s: float) -> None:
    """Kept lines from every source -> voice/data/holly/: clips at 24 kHz,
    loudness-matched, with metadata.csv (file|text) and reference.wav."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clips").mkdir(exist_ok=True)
    rows, refs = [], []
    for manifest_path in sorted(WORK.glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        for c in m["clips"]:
            if c["status"] != "keep":
                continue
            audio, rate = sf.read(manifest_path.parent / "clips" / c["file"], dtype="float32")
            audio = resample(audio, rate, EXPORT_RATE)
            audio = audio * 10 ** ((-20 - rms_db(audio)) / 20)  # all lines at -20 dB RMS
            audio = np.clip(audio, -0.99, 0.99)
            sf.write(OUT / "clips" / c["file"], audio, EXPORT_RATE, subtype="PCM_16")
            rows.append(f"{c['file']}|{c['text']}")
            if c["reference"]:
                refs.append((audio, c["text"]))
    (OUT / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    total = sum(sf.info(OUT / "clips" / r.split("|")[0]).duration for r in rows)
    print(f"Exported {len(rows)} lines, {total:.0f} s, to {OUT.relative_to(ROOT.parent)}")

    if refs:
        gap = np.zeros(int(0.3 * EXPORT_RATE), dtype=np.float32)
        parts, texts, length = [], [], 0.0
        for audio, text in refs:
            if length and length + len(audio) / EXPORT_RATE > reference_max_s:
                break
            parts += [audio, gap]
            texts.append(text)
            length += (len(audio) + len(gap)) / EXPORT_RATE
        sf.write(OUT / "reference.wav", np.concatenate(parts[:-1]), EXPORT_RATE, subtype="PCM_16")
        (OUT / "reference.txt").write_text(" ".join(texts) + "\n", encoding="utf-8")
        print(f"Reference clip: {length:.1f} s from {len(texts)} lines -> reference.wav and reference.txt")
    else:
        print("No lines marked as reference yet (press f in review).")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").add_argument("url")
    s = sub.add_parser("separate")
    s.add_argument("id")
    s.add_argument("--model", default="htdemucs_ft", help="Demucs model (htdemucs_ft is slower but cleaner)")
    s = sub.add_parser("segment")
    s.add_argument("id")
    s.add_argument("--max-line", type=float, default=12.0, help="longest line in seconds")
    s.add_argument("--merge-gap", type=float, default=0.35, help="join pieces closer than this, in seconds")
    s = sub.add_parser("trim")
    s.add_argument("id")
    s.add_argument("--status", nargs="+", default=["reject"],
                   help="trim lines with these statuses (pending, keep, reject)")
    s.add_argument("--pad-start", type=float, default=0.10, help="seconds kept before the first word")
    s.add_argument("--pad-end", type=float, default=0.12, help="seconds kept after the last word")
    s = sub.add_parser("review")
    s.add_argument("id")
    s.add_argument("--speaker", help="only this speaker group, e.g. A")
    s.add_argument("--redo", action="store_true", help="include lines already reviewed")
    s = sub.add_parser("export")
    s.add_argument("--reference-max", type=float, default=20.0, help="longest reference clip in seconds")
    a = p.parse_args()

    if a.cmd == "fetch":
        fetch(a.url)
    elif a.cmd == "separate":
        separate(a.id, a.model)
    elif a.cmd == "segment":
        segment(a.id, a.max_line, a.merge_gap)
    elif a.cmd == "trim":
        trim(a.id, a.status, a.pad_start, a.pad_end)
    elif a.cmd == "review":
        review(a.id, a.speaker, a.redo)
    elif a.cmd == "export":
        export(a.reference_max)


if __name__ == "__main__":
    main()

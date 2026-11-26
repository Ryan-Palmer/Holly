"""Has the crew finished speaking, or only paused? Smart Turn v3.2 (pipecat-ai/smart-turn-v3,
BSD-2-Clause) judges from the audio itself, its rhythm and pitch at the pause, whether the
turn is over: a Whisper-tiny encoder with a classifier, ~8 MB int8 on the CPU, ~35 ms a
check. Before it, a turn ended after 400 ms of silence whatever was said, and a pause
mid-sentence ("I was thinking we could... do the garden") was cut in on. With it, the VAD
(app/vad.py) asks it at the first short pause: finished, he answers sooner than before;
not, he waits, up to [listen] max_silence_ms, for the rest. A model, not rules about words:
it doesn't depend on Whisper's transcript or on Holly's LLM, so either can be swapped.

    python -m app.turn [--save FOLDER]    # talk to it: at each pause, its verdict and what was heard

The probe doesn't load Holly, nor reply: it's for seeing how it does on your voice (close
Holly first: the mic and the GPU are his).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
WINDOW = 8 * SAMPLE_RATE  # the model hears the last 8 s of the turn


class TurnModel:
    """Smart Turn, from `spec` ("<Hugging Face repo>/<file>"): fetched once into the Hugging
    Face cache (as Whisper is), then loaded from it with no network."""

    def __init__(self, spec: str):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        repo, _, name = spec.rpartition("/")
        try:
            path = hf_hub_download(repo, name, local_files_only=True)
        except Exception:  # not fetched yet
            path = hf_hub_download(repo, name)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1  # small: more threads only add overhead
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=8)
        self.finished(np.zeros(SAMPLE_RATE, np.float32))  # the first run is slow

    def finished(self, audio: np.ndarray) -> float:
        """How likely the turn in `audio` (16 kHz mono, all of it so far, with the pause) is
        over: 0 to 1. As pipecat-ai/smart-turn's inference.py: the last 8 s, padded at the
        start if shorter, Whisper's features, normalised."""
        audio = audio[-WINDOW:] if len(audio) > WINDOW else np.pad(audio, (WINDOW - len(audio), 0))
        feats = self.features(audio, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
                              max_length=WINDOW, truncation=True, do_normalize=True).input_features
        return float(self.session.run(None, {"input_features": feats.astype(np.float32)})[0][0, 0])


def answers_after(p: float, listen) -> int:
    """How long Holly waits, at a pause the model gives `p`, before answering (if you don't
    go on): as app/vad.py does."""
    return listen.turn_check_ms if p >= listen.turn_threshold else listen.max_silence_ms


def cuts_in(pause: dict, listen) -> bool:
    """Would Holly have answered during a pause you went on after ({"p", "pause_ms"})?"""
    return pause.get("pause_ms", 0) >= answers_after(pause["p"], listen)


def verdict(p: float, listen) -> str:
    """What Holly does with `p` at a pause (app/vad.py)."""
    if p >= listen.turn_threshold:
        return "finished: he answers now"
    return f"not sure you've finished: he waits up to {listen.max_silence_ms / 1000:g} s"


async def probe(save=None) -> None:
    """Listen, and at each pause say what the model makes of it (and what Holly would do),
    and what was said so far. It never ends a turn on the model's word: it always waits
    the longest, so whether you went on after each pause is known, and each is labelled by
    it when the turn's over. With `save` (a folder), each pause's audio is kept there, a
    WAV each, with clips.jsonl: its probability, what was heard, and whether you went on."""
    import asyncio
    import dataclasses
    import json
    import wave
    from pathlib import Path

    from .config import load_config
    from .console import DIM, RESET
    from .mic import Mic
    from .stt import Transcriber
    from .vad import TurnDetector

    cfg = load_config()
    listen = cfg.listen
    print(f"{DIM}Loading Whisper, Silero VAD and {listen.turn_model}...{RESET}", flush=True)
    stt = await asyncio.to_thread(Transcriber, listen)
    await asyncio.to_thread(stt.warm_up)
    checks: list[tuple[float, np.ndarray]] = []
    # Watching only: the turn never ends on the model's word, and always waits the longest.
    watch = dataclasses.replace(listen, turn_threshold=2.0)
    detector = TurnDetector(watch, TurnModel(listen.turn_model), on_check=lambda p, audio: checks.append((p, audio)))
    folder = Path(save) if save else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    n = len(list(folder.glob("*.wav"))) if folder else 0
    mic = Mic(cfg.audio.host_api, listen.input_device, listen.input_channels)
    mic.start()
    print(f"Listening on {mic.device_name}. Say things, finished and not, with pauses. Ctrl-C to stop.\n",
          flush=True)
    turn: list[dict] = []  # this turn's pauses so far
    try:
        while True:
            block, captured = await mic.queue.get()
            paused = detector.silence_ms
            utterance = detector.feed(block, captured)
            if turn and "pause_ms" not in turn[-1] and paused and detector.in_speech and not detector.silence_ms:
                turn[-1]["pause_ms"] = paused  # you went on, after this long
            while checks:
                p, audio = checks.pop(0)
                heard = await asyncio.to_thread(stt.transcribe, audio)
                print(f"  {p:.2f} {verdict(p, listen)}  | {heard}", flush=True)
                turn.append(dict(p=round(p, 4), heard=heard, audio=audio))
            if utterance and turn:
                # Every pause but the last, you went on after; after the last, you'd finished.
                for i, c in enumerate(turn):
                    c["went_on"] = i < len(turn) - 1
                cut = sum(cuts_in(c, listen) for c in turn[:-1])
                before = sum(c.get("pause_ms", 0) >= listen.end_silence_ms for c in turn[:-1])
                print(f"{DIM}  (turn over. Pauses you went on after: {len(turn) - 1}; he'd have cut in on {cut} "
                      f"(the 400 ms rule: {before}). You'd finished: he'd answer after "
                      f"{answers_after(turn[-1]['p'], listen)} ms (before: {listen.end_silence_ms})){RESET}\n",
                      flush=True)
                if folder:
                    with open(folder / "clips.jsonl", "a", encoding="utf-8") as index:
                        for c in turn:
                            n += 1
                            name = f"{n:04d}.wav"
                            with wave.open(str(folder / name), "wb") as w:
                                w.setnchannels(1)
                                w.setsampwidth(2)
                                w.setframerate(SAMPLE_RATE)
                                w.writeframes((np.clip(c["audio"], -1, 1) * 32767).astype(np.int16).tobytes())
                            index.write(json.dumps(dict(file=name, p=c["p"], heard=c["heard"], went_on=c["went_on"],
                                                        pause_ms=c.get("pause_ms"))) + "\n")
                turn = []
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        mic.close()


def main() -> None:
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(description="Try the turn-ending model on your voice (close Holly first)")
    parser.add_argument("--save", metavar="FOLDER", help="keep each pause's audio there, labelled, for tuning")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(probe(args.save))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

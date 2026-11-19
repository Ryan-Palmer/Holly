"""Shared by every adapter. Adapters run inside their candidate's own venv,
so this file depends only on the standard library, numpy and torch.

An adapter is a script that does:

    from common import Adapter
    class MyModel(Adapter):
        name = "my-model"
        def load(self): ...                    # load weights onto the GPU
        def synthesize(self, text): ...        # yield float32 mono chunks
        sample_rate = 24_000
    MyModel.main()

and is run as:

    <venv python> adapters/my_model.py --ref ref.wav --ref-text ref.txt \\
        --sentences sentences.txt --out OUTDIR

It writes OUTDIR/NN.wav per sentence and OUTDIR/timings.json. If the model
streams, `synthesize` yields chunks as they're ready and the first one sets
time to first audio; otherwise it yields once, at the end.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def read_sentences(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and not s.lstrip().startswith("#")]


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def to_mono_float(x) -> np.ndarray:
    """Tensors or arrays of any common shape -> 1-D float32."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > 1:
        x = x.mean(axis=0) if x.shape[0] < x.shape[-1] else x.mean(axis=-1)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if np.abs(x).max(initial=0) > 1.5:  # int16-scaled
        x = x / 32768.0
    return x


class Adapter:
    name = "unnamed"
    sample_rate = 24_000
    streams = False  # True if synthesize() yields chunks before the end

    def __init__(self, ref: Path, ref_text: str):
        self.ref = ref
        self.ref_text = ref_text

    def load(self) -> None:
        raise NotImplementedError

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def version(self) -> str:
        return "?"

    # ------------------------------------------------------------------

    @classmethod
    def main(cls) -> None:
        p = argparse.ArgumentParser()
        p.add_argument("--ref", type=Path, required=True)
        p.add_argument("--ref-text", type=Path, required=True)
        p.add_argument("--sentences", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        a = p.parse_args()
        a.out.mkdir(parents=True, exist_ok=True)

        import torch

        if not torch.cuda.is_available():
            sys.exit(f"{cls.name}: PyTorch can't see the GPU (torch {torch.__version__}). "
                     "Install a CUDA 12.8+ build in this venv.")
        gpu = torch.cuda.get_device_name(0)

        adapter = cls(a.ref, a.ref_text.read_text(encoding="utf-8").strip())
        t = time.perf_counter()
        adapter.load()
        load_s = time.perf_counter() - t
        # First inference is always slow; don't let it count.
        for _ in adapter.synthesize("This is a warm-up sentence."):
            pass
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        results = []
        for i, text in enumerate(read_sentences(a.sentences)):
            chunks = []
            t0 = time.perf_counter()
            first = None
            for chunk in adapter.synthesize(text):
                chunk = to_mono_float(chunk)
                if first is None and chunk.size:
                    first = time.perf_counter()
                chunks.append(chunk)
            total = time.perf_counter() - t0
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            write_wav(a.out / f"{i:02d}.wav", audio, adapter.sample_rate)
            dur = len(audio) / adapter.sample_rate
            results.append(dict(
                index=i, text=text, audio_s=round(dur, 2), total_ms=round(total * 1000),
                first_audio_ms=round(((first or t0 + total) - t0) * 1000),
                rtf=round(total / dur, 3) if dur else None,
            ))
            print(f"  {cls.name} {i:02d}: first audio {results[-1]['first_audio_ms']} ms, "
                  f"{dur:.1f} s audio in {total:.2f} s", flush=True)

        meta = dict(
            candidate=cls.name, version=adapter.version(), streams=cls.streams, gpu=gpu,
            torch=torch.__version__, python=platform.python_version(), load_s=round(load_s, 1),
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            sample_rate=adapter.sample_rate,
        )
        (a.out / "timings.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"  {cls.name}: done, peak VRAM {meta['peak_vram_gb']} GB (PyTorch allocations)")

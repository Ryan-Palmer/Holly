r"""Render a talking-head clip of Holly with Ditto, and time it.

Runs in face/.venv, with Ditto's code in face/ditto-src and its PyTorch
weights in face/checkpoints (see face/README.md).

    face\.venv\Scripts\python face\try_ditto.py --image face/data/holly_face.png --audio voice/data/holly/reference.wav
    face\.venv\Scripts\python face\try_ditto.py ... --online     # Ditto's streaming mode

The audio is fed in as fast as Ditto will take it, so "frames a second" is
how fast it can go, which must stay above 25 for live use. "First frame" is
counted from the first audio handed over.
"""

import argparse
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

FACE = Path(__file__).resolve().parent
DITTO = FACE / "ditto-src"
CHECKPOINTS = FACE / "checkpoints"
FPS = 25
CHUNK = (3, 5, 2)  # Ditto's streaming chunk: frames of history, new frames, lookahead


class VramPeak:
    """Samples nvidia-smi, as torch's own figures miss ONNX Runtime and are wrong under WDDM."""

    def __init__(self):
        self.base = self.peak = self._used()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _used():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        return int(out.split()[0])

    def _run(self):
        while not self._stop.wait(0.25):
            self.peak = max(self.peak, self._used())

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self.base, self.peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", default=None, help="default: face/data/out/<image>_<audio>[_online].mp4")
    p.add_argument("--online", action="store_true", help="Ditto's streaming mode")
    p.add_argument("--fast", action="store_true", help="compiled, GPU-resident stages (face/fast_ditto.py); implies --warmup")
    p.add_argument("--warmup", action="store_true", help="render 3 s first, untimed")
    p.add_argument("--steps", type=int, default=None, help="diffusion steps (default 50 offline, 10 online)")
    args = p.parse_args()

    sys.path.insert(0, str(DITTO))
    import imageio_ffmpeg
    import librosa
    import numpy as np

    if args.online:
        from stream_pipeline_online import StreamSDK
    else:
        from stream_pipeline_offline import StreamSDK

    image, audio_path = Path(args.image).resolve(), Path(args.audio).resolve()
    out = Path(args.out) if args.out else (
        FACE / "data" / "out" / f"{image.stem}_{audio_path.stem}{'_online' if args.online else ''}.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    setup_kwargs = {}
    if args.online:
        # What Ditto's own TensorRT streaming config sets; the PyTorch config ships offline only.
        setup_kwargs = {"online_mode": True, "sampling_timesteps": 10, "overlap_v2": 70}
    if args.steps:
        setup_kwargs["sampling_timesteps"] = args.steps

    vram = VramPeak()
    t0 = time.perf_counter()
    sdk = StreamSDK(str(CHECKPOINTS / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"), str(CHECKPOINTS / "ditto_pytorch"))
    if args.fast:
        from fast_ditto import speed_up

        speed_up(sdk)
        args.warmup = True
    t_load = time.perf_counter() - t0

    audio, _ = librosa.load(str(audio_path), sr=16000)
    t_warm = 0.0
    if args.warmup:
        t0 = time.perf_counter()
        # The first pass through each model pays for CUDA set-up; keep it out of the timings.
        render(sdk, np, image, audio[: 3 * 16000], out.with_suffix(".warmup.mp4"), setup_kwargs, args.online)
        out.with_suffix(".warmup.mp4.tmp.mp4").unlink(missing_ok=True)
        t_warm = time.perf_counter() - t0
    stats = render(sdk, np, image, audio, out, setup_kwargs, args.online)
    base, peak = vram.stop()

    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", sdk.tmp_output_path,
         "-i", str(audio_path), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(out)],
        check=True,
    )
    Path(sdk.tmp_output_path).unlink(missing_ok=True)

    frame_times, t_setup, t_start, t_end, stages = stats
    n = len(frame_times)
    print()
    print(f"mode           {'online (streaming)' if args.online else 'offline'}, "
          f"{setup_kwargs.get('sampling_timesteps', 50)} steps{', fast' if args.fast else ''}{', warmed up' if args.warmup else ''}")
    print(f"load models    {t_load:.1f} s")
    if t_warm:
        print(f"warm-up        {t_warm:.1f} s")
    print(f"set up face    {t_setup:.1f} s")
    print(f"audio          {len(audio) / 16000:.1f} s, {n} frames written")
    if n:
        print(f"first frame    {(frame_times[0] - t_start) * 1000:.0f} ms after the first audio")
        if n > 1:
            print(f"frames/second  {(n - 1) / (frame_times[-1] - frame_times[0]):.1f} (needs 25)")
    print(f"total          {t_end - t_start:.1f} s")
    print(f"VRAM           {base} MiB before, peak {peak} MiB (+{peak - base})")
    print("stages (each runs in its own thread; the slowest sets the frame rate):")
    for name, (calls, secs) in stages.items():
        if calls:
            print(f"  {name:14} {calls:5} calls  {secs:6.2f} s  {secs / max(n, 1) * 1000:6.1f} ms/frame")
    print(f"wrote          {out}")


class Timed:
    """Wraps one of Ditto's pipeline stages, adding up the time spent in it."""

    def __init__(self, inner, tally):
        self._inner, self._tally = inner, tally

    def __call__(self, *a, **kw):
        t = time.perf_counter()
        try:
            return self._inner(*a, **kw)
        finally:
            self._tally[0] += 1
            self._tally[1] += time.perf_counter() - t

    def __getattr__(self, name):
        return getattr(self._inner, name)


def render(sdk, np, image, audio, out, setup_kwargs, online):
    t0 = time.perf_counter()
    sdk.setup(str(image), str(out), **setup_kwargs)
    t_setup = time.perf_counter() - t0
    sdk.setup_Nd(N_d=math.ceil(len(audio) / 16000 * FPS))

    stages = {}
    for name in ("wav2feat", "audio2motion", "motion_stitch", "warp_f3d", "decode_f3d", "putback"):
        inner = getattr(sdk, name)
        inner = getattr(inner, "_inner", inner)
        stages[name] = [0, 0.0]
        setattr(sdk, name, Timed(inner, stages[name]))

    # Time each frame as it reaches the writer, and the writer's own encoding.
    frame_times = []
    write = sdk.writer
    stages["write mp4"] = [0, 0.0]
    timed = Timed(write, stages["write mp4"])

    def timed_write(img, fmt="bgr"):
        frame_times.append(time.perf_counter())
        timed(img, fmt=fmt)

    timed_write.close = write.close
    sdk.writer = timed_write

    t_start = time.perf_counter()
    if online:
        audio = np.concatenate([np.zeros(CHUNK[0] * 640, dtype=np.float32), audio])
        split_len = int(sum(CHUNK) * 0.04 * 16000) + 80
        for i in range(0, len(audio), CHUNK[1] * 640):
            piece = audio[i:i + split_len]
            if len(piece) < split_len:
                piece = np.pad(piece, (0, split_len - len(piece)))
            sdk.run_chunk(piece, CHUNK)
    else:
        sdk.audio2motion_queue.put(sdk.wav2feat.wav2feat(audio))
    sdk.close()
    return frame_times, t_setup, t_start, time.perf_counter(), {k: tuple(v) for k, v in stages.items()}

if __name__ == "__main__":
    main()

"""Holly's cloned voice: VoxCPM2, in this process, on the GPU.

VoxCPM2 won the blind test in voice/harness: closest to Norman's Holly,
most natural, and it streams. It clones from the reference clip and its
transcript made by voice/prep.py export, using both of its cloning inputs:
the isolated reference audio, and the clip as a prompt with its transcript
("continuation" mode), its highest-fidelity setting.

Its streaming generator is blocking GPU code, so it runs in a worker thread
and hands chunks to the event loop; otherwise the LLM stream would stall
while it worked.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

import numpy as np

from .config import VoxCpmConfig
from .tts_common import trim_leading_silence

_DONE = object()


class VoxCpmTts:
    def __init__(self, cfg: VoxCpmConfig):
        from voxcpm import VoxCPM
        import voxcpm.model.voxcpm2 as voxcpm2
        from tqdm import tqdm

        # VoxCPM wraps every sentence's generation in a progress bar, which cluttered the
        # console mid-reply. Only its own loop goes quiet: downloads keep their progress.
        voxcpm2.tqdm = lambda *args, **kwargs: tqdm(*args, **{**kwargs, "disable": True})

        if not cfg.reference_wav.exists():
            raise FileNotFoundError(
                f"No reference clip at {cfg.reference_wav}: run voice/prep.py export first."
            )
        self.cfg = cfg
        self.reference_text = cfg.reference_text.read_text(encoding="utf-8").strip()
        self.model = VoxCPM.from_pretrained(cfg.model, load_denoiser=False, optimize=cfg.compile, device="cuda")
        tts = getattr(self.model, "tts_model", self.model)
        self.sample_rate = int(getattr(tts, "sample_rate", 48_000))
        # One generation at a time: the model isn't safe to call concurrently.
        self._lock = threading.Lock()

    async def warm_up(self) -> None:
        """The first runs are slow, and with torch.compile the very first
        start after an install compiles for minutes. Pay for it now, with
        sentences of a few lengths, so compiled shapes are ready."""
        for text in ("Ready.", "I am the ship's computer.", "Three million years is a long time to be on your own."):
            async for _ in self.stream(text):
                pass

    def stream(self, text: str, trim_lead: bool = False) -> AsyncIterator[np.ndarray]:
        chunks = self._stream(text)
        return trim_leading_silence(chunks, self.sample_rate) if trim_lead else chunks

    async def _stream(self, text: str) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def work() -> None:
            try:
                with self._lock:
                    for chunk in self.model.generate_streaming(
                        text=text,
                        reference_wav_path=str(self.cfg.reference_wav),
                        prompt_wav_path=str(self.cfg.reference_wav),
                        prompt_text=self.reference_text,
                        inference_timesteps=self.cfg.inference_timesteps,
                    ):
                        if stop.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, np.asarray(chunk, dtype=np.float32).reshape(-1))
            except BaseException as e:  # noqa: BLE001 - re-raised in the event loop
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        threading.Thread(target=work, name="voxcpm", daemon=True).start()
        try:
            while (item := await queue.get()) is not _DONE:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()  # if the consumer gave up early (barge-in, later)

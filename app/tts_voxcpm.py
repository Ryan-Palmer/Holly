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
import re
import threading
from collections.abc import AsyncIterator

import numpy as np

from .config import VoxCpmConfig
from .tts_common import trim_leading_silence

_DONE = object()
# The voice model's own caches, in its tokens: the reference clip twice (~250 for
# Holly's 20 s clip), the texts, and the speech made (~6 a second). VoxCPM sizes
# them for 8,192, ~0.3 GB, and attends over all of it at every step; a sentence
# of Holly's needs well under 1,000.
CACHE_TOKENS = 2048


class VoxCpmTts:
    def __init__(self, cfg: VoxCpmConfig):
        from voxcpm import VoxCPM
        import voxcpm.model.voxcpm2 as voxcpm2
        from voxcpm.model.utils import get_dtype
        from tqdm import tqdm

        # VoxCPM wraps every sentence's generation in a progress bar, which cluttered the
        # console mid-reply. Only its own loop goes quiet: downloads keep their progress.
        voxcpm2.tqdm = lambda *args, **kwargs: tqdm(*args, **{**kwargs, "disable": True})

        if not cfg.reference_wav.exists():
            raise FileNotFoundError(
                f"No reference clip at {cfg.reference_wav}: run voice/prep.py export first."
            )
        self.cfg = cfg
        reference_text = cfg.reference_text.read_text(encoding="utf-8").strip()
        # Not compiled yet: the compiled steps keep the caches' addresses, so the
        # smaller caches go in first.
        self.model = VoxCPM.from_pretrained(cfg.model, load_denoiser=False, optimize=False, device="cuda")
        self.tts = tts = self.model.tts_model
        dtype = get_dtype(tts.config.dtype)
        for lm in (tts.base_lm, tts.residual_lm):
            lm.kv_cache = None  # the 8,192 one, freed before the new one is made
            lm.setup_cache(1, CACHE_TOKENS, tts.device, dtype)
        if cfg.compile:
            tts.optimize()
        self.sample_rate = int(getattr(tts, "sample_rate", 48_000))
        # The reference clip, encoded once. VoxCPM's own generate() reads it from disk
        # and encodes it twice (as reference and as prompt) for every sentence.
        self.prompt = tts.build_prompt_cache(prompt_text=reference_text,
                                             prompt_wav_path=str(cfg.reference_wav),
                                             reference_wav_path=str(cfg.reference_wav))
        # Its share of the cache: the clip twice, its transcript, and a margin.
        self.prompt_tokens = (self.prompt["ref_audio_feat"].shape[0] + self.prompt["audio_feat"].shape[0]
                              + len(tts.text_tokenizer(reference_text)) + 8)
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

    def _generate(self, text: str):
        """VoxCPM's own streaming generate, with the clip encoded once, and the speech
        kept within the cache: a sentence too long for it is cut short, not a crash."""
        text = re.sub(r"\s+", " ", text).strip()
        room = CACHE_TOKENS - self.prompt_tokens - len(self.tts.text_tokenizer(text))
        result = self.tts._generate_with_prompt_cache(
            target_text=text, prompt_cache=self.prompt, max_len=max(room, 16),
            inference_timesteps=self.cfg.inference_timesteps, streaming=True)
        try:
            for wav, _, _ in result:
                yield wav.squeeze(0).cpu().numpy()
        finally:
            result.close()

    async def _stream(self, text: str) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def work() -> None:
            try:
                with self._lock:
                    for chunk in self._generate(text):
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

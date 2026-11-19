"""Stock-voice TTS: Orpheus in LM Studio, decoded to audio with SNAC.

Orpheus is a Llama model that writes speech as audio codes. Through LM
Studio's completions endpoint it returns them as text, `<custom_token_N>`,
in groups of seven per SNAC frame. Each frame is 2048 samples, about 85 ms
at 24 kHz. Audio is decoded while codes are still arriving.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

import httpx
import numpy as np
import torch
from snac import SNAC

from .config import TtsConfig

SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 2048
CODES_PER_FRAME = 7
CODEBOOK_SIZE = 4096
# <custom_token_N> with N below this is a control token, not audio.
FIRST_AUDIO_TOKEN = 10
END_OF_SPEECH = "<custom_token_2>"
_TOKEN = re.compile(r"<custom_token_(\d+)>")

# Frames decoded either side of the frames being emitted. SNAC is
# convolutional, so a frame decoded alone clicks at its edges. Measured
# against decoding a whole utterance at once: lookahead 1 gives 26 dB SNR,
# 2 gives 32 dB, and more context than this doesn't improve on it.
LEFT_CONTEXT = 3
LOOKAHEAD = 2

# Leading-silence trim: anything quieter than -40 dBFS before the voice
# starts is cut down to a 20 ms lead-in.
SILENCE_LEVEL = 0.01
LEAD_IN = 480


class OrpheusTts:
    sample_rate = SAMPLE_RATE

    def __init__(self, cfg: TtsConfig, base_url: str):
        self.cfg = cfg
        self.http = httpx.AsyncClient(base_url=base_url, timeout=120)
        self.device = torch.device(cfg.device)
        self.snac = SNAC.from_pretrained(cfg.snac_model).eval().to(self.device)

    async def warm_up(self) -> None:
        """First inference is slow for both models, so pay for it at startup."""
        async for _ in self.stream("Ready."):
            pass

    async def stream(self, text: str, trim_lead: bool = False) -> AsyncIterator[np.ndarray]:
        """Yield float32 mono audio chunks at 24 kHz for one sentence.

        Orpheus opens every utterance with ~170 ms of silence. With
        `trim_lead`, that is cut to LEAD_IN, so the first chunk out is voice:
        worth doing for a reply's first sentence, where it is pure latency.
        """
        if not trim_lead:
            async for chunk in self._stream(text):
                yield chunk
            return
        held = np.zeros(0, dtype=np.float32)
        voiced = False
        async for chunk in self._stream(text):
            if voiced:
                yield chunk
                continue
            held = np.concatenate([held, chunk])
            loud = np.flatnonzero(np.abs(held) > SILENCE_LEVEL)
            if loud.size:
                voiced = True
                yield held[max(0, loud[0] - LEAD_IN) :]
            else:
                held = held[-LEAD_IN:]
        if not voiced and held.size:
            yield held

    async def _stream(self, text: str) -> AsyncIterator[np.ndarray]:
        frames: list[list[int]] = []
        current: list[int] = []
        emitted = 0
        async for code in self._codes(text):
            current.append(code)
            if len(current) == CODES_PER_FRAME:
                frames.append(current)
                current = []
                # The first frame goes out without waiting for lookahead: speech
                # starts from near-silence, so an edge click there is inaudible,
                # and it saves a frame's worth of generation (~50 ms).
                ready = len(frames) - (LOOKAHEAD if emitted else 0)
                if ready > emitted:
                    yield self._decode(frames, emitted, ready)
                    emitted = ready
        if len(frames) > emitted:
            yield self._decode(frames, emitted, len(frames))

    async def _codes(self, text: str) -> AsyncIterator[int]:
        """The audio codes for `text`, each mapped back into 0..4095."""
        body = {
            "model": self.cfg.model,
            "prompt": f"<|audio|>{self.cfg.voice}: {text}<|eot_id|>",
            "max_tokens": 2000,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "repeat_penalty": self.cfg.repeat_penalty,
            "stop": [END_OF_SPEECH],
            "stream": True,
        }
        index = 0
        async with self.http.stream("POST", "/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                piece = json.loads(line[6:])["choices"][0].get("text", "")
                for m in _TOKEN.finditer(piece):
                    n = int(m.group(1))
                    if n < FIRST_AUDIO_TOKEN:
                        continue
                    # Each of the 7 positions in a frame has its own offset.
                    code = n - FIRST_AUDIO_TOKEN - (index % CODES_PER_FRAME) * CODEBOOK_SIZE
                    if not 0 <= code < CODEBOOK_SIZE:
                        continue  # a stray token; skipping it keeps frames aligned
                    index += 1
                    yield code

    @torch.inference_mode()
    def _decode(self, frames: list[list[int]], start: int, end: int) -> np.ndarray:
        """Audio for frames[start:end], decoded with context either side."""
        lo = max(0, start - LEFT_CONTEXT)
        hi = min(len(frames), end + LOOKAHEAD)
        window = frames[lo:hi]
        # Orpheus interleaves SNAC's three layers: 1 coarse, 2 medium, 4 fine codes.
        l1 = [f[0] for f in window]
        l2 = [c for f in window for c in (f[1], f[4])]
        l3 = [c for f in window for c in (f[2], f[3], f[5], f[6])]
        codes = [torch.tensor([layer], dtype=torch.long, device=self.device) for layer in (l1, l2, l3)]
        audio = self.snac.decode(codes)[0, 0]
        a = (start - lo) * SAMPLES_PER_FRAME
        b = (end - lo) * SAMPLES_PER_FRAME
        return audio[a:b].float().cpu().numpy()

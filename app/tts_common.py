"""What a TTS engine provides, and making Holly's (VoxCPM2)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from .config import Config

# Leading-silence trim: anything quieter than -40 dBFS before the voice
# starts is cut down to a 20 ms lead-in.
SILENCE_LEVEL = 0.01
LEAD_IN_S = 0.02


class Tts(Protocol):
    sample_rate: int

    async def warm_up(self) -> None: ...

    def stream(self, text: str, trim_lead: bool = False) -> AsyncIterator[np.ndarray]:
        """Float32 mono chunks for one sentence, at `sample_rate`."""


async def trim_leading_silence(chunks: AsyncIterator[np.ndarray], sample_rate: int) -> AsyncIterator[np.ndarray]:
    """Drop the silence a TTS model opens with, down to a short lead-in, so
    the first chunk out is voice. Worth doing for a reply's first sentence,
    where that silence is pure latency."""
    lead_in = int(LEAD_IN_S * sample_rate)
    held = np.zeros(0, dtype=np.float32)
    voiced = False
    async for chunk in chunks:
        if voiced:
            yield chunk
            continue
        held = np.concatenate([held, chunk])
        loud = np.flatnonzero(np.abs(held) > SILENCE_LEVEL)
        if loud.size:
            voiced = True
            yield held[max(0, loud[0] - lead_in) :]
        else:
            held = held[-lead_in:]
    if not voiced and held.size:
        yield held


def make_tts(cfg: Config) -> Tts:
    from .tts_voxcpm import VoxCpmTts

    return VoxCpmTts(cfg.tts.voxcpm)

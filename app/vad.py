"""Turn detection: Silero VAD decides when you've started and finished speaking."""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad

from .config import ListenConfig

SAMPLE_RATE = 16_000
# Silero takes exactly 512 samples at 16 kHz: 32 ms.
BLOCK = 512
BLOCK_MS = BLOCK * 1000 // SAMPLE_RATE


@dataclass
class Utterance:
    audio: np.ndarray  # float32 mono at 16 kHz
    speech_end: float  # perf_counter() when the last speech left your mouth
    duration_ms: int


class TurnDetector:
    """Feed it 512-sample blocks; it returns an Utterance when a turn ends."""

    def __init__(self, cfg: ListenConfig):
        self.cfg = cfg
        torch.set_num_threads(1)  # Silero is tiny; more threads only add overhead
        self.model = load_silero_vad()
        self.preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, cfg.preroll_ms // BLOCK_MS)
        )
        self.reset()

    def reset(self) -> None:
        """Forget any speech in progress, e.g. after Holly has spoken."""
        self.model.reset_states()
        self.preroll.clear()
        self.blocks: list[np.ndarray] = []
        self.in_speech = False
        self.speech_ms = 0
        self.silence_ms = 0
        self.last_speech_end = 0.0

    def feed(self, block: np.ndarray, block_end: float) -> Utterance | None:
        """`block_end` is the perf_counter() time the block's last sample was captured."""
        prob = self.model(torch.from_numpy(block), SAMPLE_RATE).item()
        if not self.in_speech:
            self.preroll.append(block)
            if prob >= self.cfg.start_threshold:
                self.in_speech = True
                self.blocks = list(self.preroll)
                self.speech_ms = BLOCK_MS
                self.silence_ms = 0
                self.last_speech_end = block_end
            return None

        self.blocks.append(block)
        if prob >= self.cfg.end_threshold:
            self.speech_ms += BLOCK_MS + self.silence_ms
            self.silence_ms = 0
            self.last_speech_end = block_end
            return None

        self.silence_ms += BLOCK_MS
        if self.silence_ms < self.cfg.end_silence_ms:
            return None

        # The turn is over. Keep a little of the trailing silence; Whisper
        # transcribes a clipped final word worse than a quiet tail.
        keep = len(self.blocks) - self.silence_ms // BLOCK_MS + 4
        audio = np.concatenate(self.blocks[:keep])
        speech_ms, speech_end = self.speech_ms, self.last_speech_end
        self.reset()
        if speech_ms < self.cfg.min_speech_ms:
            return None
        return Utterance(audio=audio, speech_end=speech_end, duration_ms=speech_ms)

"""Turn detection: Silero VAD decides when you've started speaking and when you pause; at
a pause, a turn-ending model (app/turn.py), if there is one, decides whether you've
finished or will go on."""

from __future__ import annotations

import collections
from collections.abc import Callable
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
    """Feed it 512-sample blocks; it returns an Utterance when a turn ends.

    Without a turn model, a turn ends after end_silence_ms of silence, whatever was said.
    With one, it's asked once each pause, after turn_check_ms of silence: sure you've
    finished (at least turn_threshold), the turn ends then, sooner than end_silence_ms;
    otherwise it waits for more speech, up to max_silence_ms. Being cut off is worse than
    a slower answer: an "unsure" middle that ended at end_silence_ms, as before, had him
    answer "So, I was thinking..." live, a pause the model wasn't sure of. Unsure, he
    waits (half the finished questions scored only 0.14-0.38: those are answered after
    max_silence_ms). `on_check(probability, audio)` hears each answer (the probe,
    app/turn.py; the console, app/listen.py)."""

    def __init__(self, cfg: ListenConfig, turn_model=None,
                 on_check: Callable[[float, np.ndarray], None] | None = None):
        self.cfg = cfg
        self.turn_model = turn_model
        self.on_check = on_check
        torch.set_num_threads(1)  # Silero is tiny; more threads only add overhead
        self.model = load_silero_vad()
        self.preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, cfg.preroll_ms // BLOCK_MS)
        )
        self.ended_by_model = False  # how the last turn ended: the model said so, or the wait ran out
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
        self.checked = False  # the turn model's been asked about this pause
        self.wait_ms = self.cfg.end_silence_ms  # how long this pause may go on before the turn ends

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
            self.checked = False  # speaking again: the next pause is asked about afresh
            self.wait_ms = self.cfg.end_silence_ms
            return None

        self.silence_ms += BLOCK_MS
        if (self.turn_model is not None and not self.checked and self.silence_ms >= self.cfg.turn_check_ms
                and self.speech_ms >= self.cfg.min_speech_ms):
            # The whole turn so far, the pause included, as the model was trained.
            self.checked = True
            audio = np.concatenate(self.blocks)
            p = self.turn_model.finished(audio)
            if self.on_check:
                self.on_check(p, audio)
            if p >= self.cfg.turn_threshold:
                return self._end(True)
            self.wait_ms = self.cfg.max_silence_ms
        return self._end(False) if self.silence_ms >= self.wait_ms else None

    def _end(self, by_model: bool) -> Utterance | None:
        """The turn's over. Keep a little of the trailing silence; Whisper transcribes a
        clipped final word worse than a quiet tail."""
        keep = len(self.blocks) - self.silence_ms // BLOCK_MS + 4
        audio = np.concatenate(self.blocks[:keep])
        speech_ms, speech_end = self.speech_ms, self.last_speech_end
        self.ended_by_model = by_model
        self.reset()
        if speech_ms < self.cfg.min_speech_ms:
            return None
        return Utterance(audio=audio, speech_end=speech_end, duration_ms=speech_ms)

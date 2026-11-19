"""Speech to text with faster-whisper."""

from __future__ import annotations

import re

import numpy as np
import torch  # noqa: F401  (puts PyTorch's cuBLAS and cuDNN DLLs on the path for CTranslate2)
from faster_whisper import WhisperModel

from .config import ListenConfig

# What Whisper says when given noise or breath instead of speech, dropped
# only when it is the whole transcript. The VAD keeps most noise away from
# Whisper, so this stays short: "thank you" is a real thing to say to Holly.
_HALLUCINATIONS = {"you", "thanks for watching", "thank you for watching", "uh", "um", "hmm"}


class Transcriber:
    def __init__(self, cfg: ListenConfig):
        self.model = WhisperModel(
            cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type
        )

    def warm_up(self) -> None:
        """The first transcription in a process is slow (~0.5 s, or ~20 s the
        very first time CUDA compiles kernels for this GPU)."""
        rng = np.random.default_rng(0)
        # Twice: after one pass, the next transcription is still ~2-8x slow.
        for _ in range(2):
            self.transcribe((rng.standard_normal(16_000) * 0.01).astype(np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        bare = re.sub(r"[^\w\s]", "", text).strip().lower()
        if not bare or bare in _HALLUCINATIONS:
            return ""
        return text

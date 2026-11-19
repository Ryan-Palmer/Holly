"""Microphone capture: 16 kHz mono blocks, timestamped, into an asyncio queue."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import sounddevice as sd

from .player import find_device
from .vad import BLOCK, SAMPLE_RATE


class Mic:
    def __init__(self, host_api: str, device: str, channels: list[int]):
        self.queue: asyncio.Queue[tuple[np.ndarray, float]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        index, api_name = find_device("input", host_api, device)
        info = sd.query_devices(index)
        self.device_name = f"{info['name']} ({api_name})"
        n = info["max_input_channels"]
        self._channels = [c - 1 for c in channels if 1 <= c <= n] or list(range(n))
        extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in api_name else None
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=n,
            dtype="float32",
            blocksize=BLOCK,
            latency="low",
            device=index,
            extra_settings=extra,
            callback=self._callback,
        )

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.stream.start()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()

    def _callback(self, indata, frames, time_info, status) -> None:
        now = time.perf_counter()
        block = indata[:, self._channels].mean(axis=1).astype(np.float32)
        # When the block's last sample was captured, if the driver says.
        age = time_info.currentTime - time_info.inputBufferAdcTime
        captured = now - age + frames / SAMPLE_RATE if 0 < age < 1 else now
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self.queue.put_nowait, (block, captured))

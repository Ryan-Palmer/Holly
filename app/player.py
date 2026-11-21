"""Streaming audio playback: chunks go in a queue, sounddevice pulls them out."""

from __future__ import annotations

import asyncio
import collections
import threading
import time

import numpy as np
import sounddevice as sd

# -40 dBFS: well above the silence a voice pads with, well below speech.
VOICE_LEVEL = 0.01


def find_device(kind: str, host_api: str, device: str) -> tuple[int, str]:
    """An "input" or "output" device's index, and its host API's name.

    With a host API, the device is that API's default, or the first of its
    devices whose name contains `device`. Without one, it's sounddevice's
    own default, or a match across all APIs.
    """
    apis = sd.query_hostapis()
    candidates = [
        (i, d) for i, d in enumerate(sd.query_devices())
        if d[f"max_{kind}_channels"] > 0
        and (not host_api or apis[d["hostapi"]]["name"] == host_api)
    ]
    if not candidates:
        raise ValueError(f"No {kind} devices for host API {host_api!r}")
    if device:
        for i, d in candidates:
            if device.lower() in d["name"].lower():
                return i, apis[d["hostapi"]]["name"]
        raise ValueError(f"No {kind} device matching {device!r} in {host_api or 'any host API'}")
    if host_api:
        api = next(a for a in apis if a["name"] == host_api)
        index = api[f"default_{kind}_device"]
    else:
        index = sd.default.device[0 if kind == "input" else 1]
    return index, apis[sd.query_devices(index)["hostapi"]]["name"]


class Player:
    """Plays float32 mono chunks in order, as they arrive.

    The sounddevice callback runs on its own thread, so the queue is guarded
    by a lock. Between replies the stream keeps running and plays silence.
    """

    def __init__(self, sample_rate: int, blocksize: int, host_api: str = "", device: str = ""):
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._pos = 0
        self._lock = threading.Lock()
        self._ending = False
        self._done: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.first_played: float | None = None
        # When the first sample above VOICE_LEVEL left the speakers: later
        # than first_played when a sentence opens with silence.
        self.first_voice: float | None = None
        # When the reply's last sample leaves the speakers.
        self.last_played: float | None = None
        # Blocks that ran out of audio mid-reply: audible as stutters.
        self.underruns = 0
        self._gap = False  # a pause is expected (Holly is searching)
        index, api_name = find_device("output", host_api, device)
        self.device_name = f"{sd.query_devices(index)['name']} ({api_name})"
        extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in api_name else None
        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            latency="low",
            device=index,
            extra_settings=extra,
            callback=self._callback,
        )

    def start(self) -> None:
        self.stream.start()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()

    def begin_turn(self) -> None:
        """Reset for a new reply. Call from the event loop."""
        with self._lock:
            self._ending = False
            self.first_played = None
            self.first_voice = None
            self.last_played = None
            self.underruns = 0
            self._gap = False
        self._loop = asyncio.get_running_loop()
        self._done = asyncio.Event()

    def put(self, chunk: np.ndarray) -> None:
        if chunk.size:
            with self._lock:
                self._chunks.append(chunk)
                self._gap = False

    def wake(self) -> None:
        """No face to wake; FacePlayer's counterpart."""

    def expect_gap(self) -> None:
        """A pause follows the audio so far (a search), until more comes: the
        silence isn't counted as stutters."""
        with self._lock:
            self._gap = True

    def end_turn(self) -> None:
        """No more chunks are coming; `wait_done` returns once they've played."""
        with self._lock:
            self._ending = True
            if not self._chunks:
                self._signal_done()

    def clear(self) -> None:
        """Drop everything queued, for barge-in."""
        with self._lock:
            self._chunks.clear()
            self._pos = 0
            if self._ending:
                self._signal_done()

    async def wait_done(self) -> None:
        assert self._done is not None
        await self._done.wait()

    def _signal_done(self) -> None:
        if self._loop and self._done:
            self._loop.call_soon_threadsafe(self._done.set)

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                n = min(frames - filled, len(chunk) - self._pos)
                out[filled : filled + n] = chunk[self._pos : self._pos + n]
                filled += n
                self._pos += n
                if self._pos >= len(chunk):
                    self._chunks.popleft()
                    self._pos = 0
            if filled < frames and self.first_played is not None and not self._ending and not self._gap:
                self.underruns += 1
            # When this block will actually leave the speakers.
            dac = time.perf_counter() + max(time_info.outputBufferDacTime - time_info.currentTime, 0.0)
            if filled and (self.first_played is None or self.first_voice is None):
                if self.first_played is None:
                    self.first_played = dac
                loud = np.flatnonzero(np.abs(out[:filled]) > VOICE_LEVEL)
                if loud.size:
                    self.first_voice = dac + loud[0] / self.stream.samplerate
            if self._ending and not self._chunks:
                self.last_played = dac + filled / self.stream.samplerate
                self._ending = False
                self._signal_done()
        out[filled:] = 0

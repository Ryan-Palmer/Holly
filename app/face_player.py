"""Holly's face: starts face/server.py and plays Holly's voice through it.

The face server (in its own venv, face/.venv) shows Ditto's animated Holly in
a window and plays the audio itself, so voice and face stay in step.
FacePlayer stands in for Player: the same calls, but the audio goes over a
local socket. See face/server.py for how the two are kept in sync.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import subprocess
import threading
import time

import numpy as np

from .config import ROOT, FaceConfig

HEADER = struct.Struct("<cI")  # message type, payload length; see face/server.py


def start_server(cfg: FaceConfig, host_api: str, device: str) -> subprocess.Popen:
    """Start the face server; it warms up for about a minute before it listens."""
    (ROOT / "face" / "data").mkdir(parents=True, exist_ok=True)
    log = open(ROOT / "face" / "data" / "server.log", "w", encoding="utf-8")
    return subprocess.Popen(
        [str(ROOT / cfg.python), "-u", str(ROOT / "face" / "server.py"), "--image", str(ROOT / cfg.image),
         "--port", str(cfg.port), "--lead", str(cfg.lead_ms / 1000), "--height", str(cfg.height),
         "--idle-head", str(cfg.idle_head),
         "--host-api", host_api, "--device", device],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )


class FacePlayer:
    """Player's interface, sending the audio to the face server."""

    def __init__(self, sample_rate: int, port: int, server: subprocess.Popen, timeout: float = 600):
        self.server = server
        deadline = time.monotonic() + timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError("The face server stopped; see face/data/server.log")
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("The face server didn't start; see face/data/server.log")
                time.sleep(0.5)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.device_name = "Holly's face window"
        self.first_played: float | None = None
        self.first_voice: float | None = None
        self.underruns = 0
        self._turn = 0
        self._done: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._send(b"R", struct.pack("<I", sample_rate))

    def _send(self, kind: bytes, payload: bytes = b"") -> None:
        with self._lock:
            self.sock.sendall(HEADER.pack(kind, len(payload)) + payload)

    def start(self) -> None:
        threading.Thread(target=self._read, daemon=True, name="face-link").start()

    def close(self) -> None:
        try:
            self.sock.close()
        finally:
            self.server.terminate()

    def begin_turn(self) -> None:
        self._turn += 1
        self.first_played = self.first_voice = None
        self.underruns = 0
        self._loop = asyncio.get_running_loop()
        self._done = asyncio.Event()
        self._send(b"B", struct.pack("<I", self._turn))

    def put(self, chunk: np.ndarray) -> None:
        if chunk.size:
            self._send(b"A", np.ascontiguousarray(chunk, dtype=np.float32).tobytes())

    def end_turn(self) -> None:
        self._send(b"E")

    def clear(self) -> None:
        self._send(b"C")

    def expect_gap(self) -> None:
        """A pause follows the audio so far (a search): not a stutter."""
        self._send(b"G")

    async def wait_done(self) -> None:
        assert self._done is not None
        await self._done.wait()

    def _read(self) -> None:
        buf = b""
        try:
            while True:
                while len(buf) < HEADER.size:
                    buf += self._recv()
                kind, size = HEADER.unpack(buf[: HEADER.size])
                while len(buf) < HEADER.size + size:
                    buf += self._recv()
                msg = json.loads(buf[HEADER.size : HEADER.size + size])
                buf = buf[HEADER.size + size :]
                if msg.get("type") == "done" and msg["turn"] == self._turn:
                    self.first_played = msg["first_played"]
                    self.first_voice = msg["first_voice"]
                    self.underruns = msg["underruns"]
                    if self._loop and self._done:
                        self._loop.call_soon_threadsafe(self._done.set)
        except OSError:
            # The window was closed: don't leave a reply waiting forever.
            if self._loop and self._done:
                self._loop.call_soon_threadsafe(self._done.set)

    def _recv(self) -> bytes:
        data = self.sock.recv(65536)
        if not data:
            raise OSError("face server closed the connection")
        return data

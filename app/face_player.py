"""Holly's face: starts face/server.py and plays Holly's voice through it.

The face server (in its own venv, face/.venv) shows Ditto's animated Holly in
a window and plays the audio itself, so voice and face stay in step.
FacePlayer stands in for Player: the same calls, but the audio goes over a
local socket. See face/server.py for how the two are kept in sync.

If the face window is closed, Holly carries on without it: his voice plays
through his own Player from then on.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import subprocess
import threading
import time

import numpy as np

from .config import ROOT, FaceConfig
from .player import Player

HEADER = struct.Struct("<cI")  # message type, payload length; see face/server.py
# The face server takes audio off the socket as it comes, whatever it's playing, so a send
# waits a moment at most. Blocking without limit, a stalled sound driver froze Holly: the
# face stopped reading, the socket filled, and his whole loop stopped in sendall, so he
# heard nothing and said nothing. Past this, the face is given up as if its window closed.
SEND_TIMEOUT_S = 3.0


def start_server(cfg: FaceConfig, host_api: str, device: str, awake_s: float = 30.0) -> subprocess.Popen:
    """Start the face server; it warms up for about a minute before it listens."""
    (ROOT / "face" / "data").mkdir(parents=True, exist_ok=True)
    log = open(ROOT / "face" / "data" / "server.log", "w", encoding="utf-8")
    return subprocess.Popen(
        [str(ROOT / cfg.python), "-u", str(ROOT / "face" / "server.py"), "--image", str(ROOT / cfg.image),
         "--port", str(cfg.port), "--lead", str(cfg.lead_ms / 1000), "--height", str(cfg.height),
         "--idle-head", str(cfg.idle_head), "--sleep-after", str(awake_s),
         "--host-api", host_api, "--device", device, "--parent", str(os.getpid())],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )


class FacePlayer:
    """Player's interface, sending the audio to the face server; or, once the face
    window has been closed, to a Player of its own (`fallback`)."""

    def __init__(self, sample_rate: int, port: int, server: subprocess.Popen, host_api: str = "",
                 device: str = "", block: int = 960, timeout: float = 600):
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
        self.sock.settimeout(SEND_TIMEOUT_S)  # for sends; the reader waits on regardless (_recv)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.device_name = "Holly's face window"
        self.sample_rate, self.host_api, self.device, self.block = sample_rate, host_api, device, block
        self.first_played: float | None = None
        self.first_voice: float | None = None
        self.last_played: float | None = None  # when the reply's last sample leaves the speakers
        self.underruns = 0
        self._turn = 0
        self._in_turn = False  # between begin_turn and the turn's audio having played
        self._done: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._closed = False  # the face server has gone (its window closed)
        self.fallback: Player | None = None
        self.sock.sendall(HEADER.pack(b"R", 4) + struct.pack("<I", sample_rate))

    def _send(self, kind: bytes, payload: bytes = b"") -> bool:
        """Send to the face; False if it's gone, and the fallback Player is to be used."""
        if self.fallback is None and not self._closed:
            try:
                with self._lock:
                    self.sock.sendall(HEADER.pack(kind, len(payload)) + payload)
                return True
            except TimeoutError:  # an OSError too: caught first, to say why
                self._closed = True
                self._lose_face(f"the face stopped taking audio for {SEND_TIMEOUT_S:.0f} s (its sound "
                                "device may have stalled)")
                return False
            except OSError:
                self._closed = True
        if self.fallback is None:
            self._lose_face()
        return False

    def _lose_face(self, why: str = "the face window was closed") -> None:
        """The face is gone (its window closed, or it stalled): carry on with Holly's own
        speakers. Called from the event loop. A reply under way carries on there, less
        what the face had queued."""
        print(f"\n  ({why}: Holly's voice plays here from now on)", flush=True)
        self.fallback = Player(self.sample_rate, self.block, self.host_api, self.device)
        self.fallback.start()
        self.device_name = self.fallback.device_name
        if self._in_turn:
            self.fallback.begin_turn()
        if self.server.poll() is None:
            self.server.terminate()

    def start(self) -> None:
        threading.Thread(target=self._read, daemon=True, name="face-link").start()

    def close(self) -> None:
        try:
            self.sock.close()
        finally:
            self.server.terminate()
            if self.fallback:
                self.fallback.close()

    def begin_turn(self) -> None:
        self._turn += 1
        self.first_played = self.first_voice = self.last_played = None
        self.underruns = 0
        self._loop = asyncio.get_running_loop()
        self._done = asyncio.Event()
        self._in_turn = True
        if not self._send(b"B", struct.pack("<I", self._turn)):
            self.fallback.begin_turn()

    def put(self, chunk: np.ndarray) -> None:
        if chunk.size and not self._send(b"A", np.ascontiguousarray(chunk, dtype=np.float32).tobytes()):
            self.fallback.put(chunk)

    def end_turn(self) -> None:
        if not self._send(b"E"):
            self.fallback.end_turn()

    def clear(self) -> None:
        if not self._send(b"C"):
            self.fallback.clear()

    def wake(self) -> None:
        """Someone started speaking: wake the face if it's asleep."""
        self._send(b"W")

    def expect_gap(self) -> None:
        """A pause follows the audio so far (a search): not a stutter."""
        if not self._send(b"G"):
            self.fallback.expect_gap()

    async def wait_done(self) -> None:
        assert self._done is not None
        if self.fallback is None:
            await self._done.wait()  # also set if the face goes mid-turn (see _read)
        if self.fallback is not None:
            await self.fallback.wait_done()
            f = self.fallback
            self.first_played, self.first_voice, self.last_played, self.underruns = (
                f.first_played, f.first_voice, f.last_played, f.underruns)
        self._in_turn = False

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
                    self.last_played = msg.get("last_played")
                    self.underruns = msg["underruns"]
                    self._finished()
        except OSError:
            # The window was closed: the next call goes to the fallback, and a reply
            # waiting for the face to finish isn't left waiting forever.
            self._closed = True
            self._finished()

    def _finished(self) -> None:
        """From the reader thread: the turn's audio has played (or the face has gone)."""
        if self._loop and self._done:
            try:
                self._loop.call_soon_threadsafe(self._done.set)
            except RuntimeError:  # the loop has closed: Holly is shutting down
                pass

    def _recv(self) -> bytes:
        while True:
            try:
                data = self.sock.recv(65536)
                break
            except TimeoutError:  # the socket's send timeout: quiet for a while is fine here
                if self._closed:
                    raise OSError("face given up")
        if not data:
            raise OSError("face server closed the connection")
        return data

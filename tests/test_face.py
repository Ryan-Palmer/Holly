"""His face going away mid-reply (its window closed, or its sound stalled): his voice
carries on through the speakers, and nothing freezes."""

import asyncio
import socket
import subprocess
import sys
import threading

import numpy as np
import pytest

from app import face_player
from app.face_player import HEADER, FacePlayer
from conftest import SilentPlayer


class Speakers(SilentPlayer):
    """The fallback Player, without a sound device."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.chunks = 0
        Speakers.instances.append(self)

    def start(self):
        pass

    def put(self, audio):
        self.chunks += 1


def face_server(reads: int | None):
    """A stand-in face server: it reads `reads` messages, then closes; or with None,
    reads nothing at all, as when its sound device stalled. Returns its port."""
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        if reads is None:
            threading.Event().wait(30)  # stuck
        for _ in range(reads or 0):
            head = conn.recv(HEADER.size, socket.MSG_WAITALL)
            if not head:
                break
            kind, size = HEADER.unpack(head)
            if size:
                conn.recv(size, socket.MSG_WAITALL)
        conn.close()
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return port


@pytest.fixture
def face(monkeypatch):
    monkeypatch.setattr(face_player, "Player", Speakers)
    Speakers.instances.clear()
    made = []

    def make(reads):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        fp = FacePlayer(48000, face_server(reads), proc)
        made.append((fp, proc))
        fp.start()
        return fp, proc

    yield make
    for fp, proc in made:
        fp.close()
        proc.kill()


async def a_turn(fp: FacePlayer, chunks: int, size: int = 4800, pause: float = 0.0) -> None:
    fp.begin_turn()
    for i in range(chunks):
        fp.put(np.zeros(size, np.float32))
        if i == 0 and pause:
            await asyncio.sleep(pause)
    fp.end_turn()
    await asyncio.wait_for(fp.wait_done(), 10)


def test_window_closed_mid_reply(face):
    fp, proc = face(3)  # the sample rate, the turn's start, one chunk: then gone

    async def go():
        await a_turn(fp, 6, pause=0.5)
        assert fp.fallback is Speakers.instances[0] and Speakers.instances[0].chunks
        fp.wake()  # used to raise
        await a_turn(fp, 2)  # the next reply too

    asyncio.run(go())
    assert proc.wait(5) is not None  # the face's process is stopped


def test_stalled_face_doesnt_freeze_him(face, monkeypatch):
    monkeypatch.setattr(face_player, "SEND_TIMEOUT_S", 0.5)
    fp, proc = face(None)

    async def go():
        # Big chunks fill the socket's buffers; then a send times out and the speakers take over.
        await a_turn(fp, 200, size=48000)

    asyncio.run(asyncio.wait_for(go(), 20))
    assert fp.fallback is not None and "stopped taking audio" not in fp.device_name
    assert proc.wait(5) is not None


@pytest.mark.parametrize("crt", [True, False])
def test_the_crt_look_reaches_the_face(cfg, tmp_path, monkeypatch, crt):
    import dataclasses

    from app import face_player
    started = []
    monkeypatch.setattr(face_player, "ROOT", tmp_path)  # its log goes here, not over the real one
    monkeypatch.setattr(face_player.subprocess, "Popen", lambda args, **kw: started.append(args))
    face_player.start_server(dataclasses.replace(cfg.face, crt=crt), "WASAPI", "Speakers")
    assert ("--crt" in started[0]) == crt


def test_the_case_picture_reaches_the_face(cfg, tmp_path, monkeypatch):
    import dataclasses

    from app import face_player
    started = []
    monkeypatch.setattr(face_player, "ROOT", tmp_path)
    monkeypatch.setattr(face_player.subprocess, "Popen", lambda args, **kw: started.append(args))
    face_player.start_server(dataclasses.replace(cfg.face, crt=True, case="face/data/case.png"), "WASAPI", "Out")
    face_player.start_server(dataclasses.replace(cfg.face, crt=False, case="face/data/case.png"), "WASAPI", "Out")
    assert started[0][started[0].index("--case") + 1] == str(tmp_path / "face/data/case.png")
    assert "--case" not in started[1]  # only with the CRT look


def test_full_screen_reaches_the_face(cfg, tmp_path, monkeypatch):
    import dataclasses

    from app import face_player
    started = []
    monkeypatch.setattr(face_player, "ROOT", tmp_path)
    monkeypatch.setattr(face_player.subprocess, "Popen", lambda args, **kw: started.append(args))
    face_player.start_server(dataclasses.replace(cfg.face, full_screen=True), "WASAPI", "Out")
    face_player.start_server(cfg.face, "WASAPI", "Out")
    assert "--full-screen" in started[0] and "--full-screen" not in started[1]


def test_full_screen_turns_the_face_on(make_holly):
    from conftest import FakeLm

    holly = make_holly(FakeLm(), full_screen=True)
    assert holly.cfg.face.full_screen and holly.cfg.face.enabled
    assert not make_holly(FakeLm()).cfg.face.full_screen

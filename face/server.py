r"""Holly's face: a window that shows Ditto animating Holly in time with his voice.

Holly's app starts this in face/.venv and sends it his voice over a local
socket (app/face_player.py) instead of playing it himself. This process
plays the audio too, because audio and frames then come from one timeline
and can't drift apart:

- The timeline runs continuously in 0.2 s blocks: Holly's audio when he's
  speaking, silence otherwise, so the face idles and blinks between replies.
- Each block goes both to the speakers' queue and to Ditto's streaming mode.
  Ditto's frame k shows the audio at 0.04 k s into the timeline.
- The speakers' queue is kept `lead` seconds ahead of what's playing, which
  gives Ditto that long to draw each frame before its audio is heard. The
  lead is also how much later Holly starts speaking than he would without
  a face.
- The window shows whichever frame matches the audio leaving the speakers
  now, from the output stream's own clock.
- After --sleep-after seconds with nothing to say or hear, Ditto stops and the
  window holds its last frame: drawing 25 frames a second costs as much GPU
  idle as talking. Holly's app wakes it the moment anyone starts speaking.

    face\.venv\Scripts\python face\server.py --image face/data/holly_colour.jpg
"""

import argparse
import ctypes
import collections
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from PIL import Image
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly

FACE = Path(__file__).resolve().parent
sys.path.insert(0, str(DITTO := FACE / "ditto-src"))
CHECKPOINTS = FACE / "checkpoints"

SR = 48000  # the timeline's playback rate
DITTO_SR = 16000
BLOCK = 9600  # 0.2 s at 48 kHz: one Ditto chunk of 5 frames
BLOCK16 = BLOCK // 3
FRAME = SR // 25  # timeline samples per video frame
CHUNK = (3, 5, 2)  # Ditto's streaming chunk: frames of history, new frames, lookahead
VOICE_LEVEL = 0.01  # as app/player.py
# Lips rest at the still's own shape when Holly is silent (Ditto's motion
# model otherwise keeps the mouth moving slightly in silence). Per frame:
SPEECH_RMS = 0.01  # a frame with more than this is speech
LIPS_HANG = 6  # frames the lips keep moving after speech (pauses between words)
LIPS_AHEAD = 3  # frames they start moving before it
LIPS_STEP = 1 / 3  # how fast they ease between resting and moving, per frame
# The head: nearly still (near the photo's pose) while Holly is silent, by
# --idle-head, and moving as Ditto drives it while he speaks. It keeps moving through pauses between sentences
# and eases in and out more slowly than the lips, so it never snaps.
POSE_HANG = 25  # frames the head keeps moving after speech (1 s)
POSE_AHEAD = 5  # frames it starts moving before it (as far ahead as is known)
POSE_STEP = 1 / 12  # ~0.5 s from still to moving
POSE_OUT_S = 1.5  # back to the idle sway after speaking: slower, or he snapped back to idle
WINDOW_HEIGHT = 720  # the window opens this tall; it can then be any size
CRT_REDRAW_S = 1 / 60  # while the CRT look's bar rolls, the picture is redrawn this often, frame or
# no frame: asleep, redrawn only as each frame came due, it stuttered
DOZE_S = 1.5  # before sleeping, the face settles to the photo itself for this long
DOZE_STEP = 1 / 25  # ~1 s to settle into the photo, eyes closing, when dozing off
WAKE_STEP = 1 / 10  # ~0.4 s to open the eyes and come back to life on waking
# The eyes shut: Ditto's own deepest blink, as offsets to the photo's expression keypoints
# 11, 13, 15, 16 and 18 (x, y, z). Captured by running Ditto on silence and taking the
# frame whose eye keypoints moved furthest from the photo's; it showed the eyes fully closed.
EYES_CLOSED = np.zeros((21, 3), np.float32)
EYES_CLOSED[[11, 13, 15, 16, 18]] = [[0.0026, 0.0150, 0.0], [0.0024, -0.0048, 0.0], [-0.0005, 0.0182, 0.0],
                                     [-0.0013, -0.0014, 0.0], [-0.0008, -0.0022, 0.0]]
WAKE_GRACE_S = 10.0  # speech wakes the face this long, in case Holly answers it
POSE_KEYS = ("pitch", "yaw", "roll", "t")  # t: the head's position in the frame
HEADER = struct.Struct("<cI")  # message type, payload length; see app/face_player.py


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def exit_with(pid: int) -> None:
    """Wait for Holly's process to end (however it ends: a crash runs no cleanup), then exit
    at once. Left behind, the face stayed on screen holding its VRAM, looking like a frozen
    Holly."""
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE: enough to wait on it
        if handle:
            kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)  # INFINITE
        elif (error := ctypes.get_last_error()) != 87:  # 87, ERROR_INVALID_PARAMETER: no such process
            # Couldn't watch it (access denied, say): Holly may well be running, so the face
            # stays, closed with him as before (when his socket goes) or by hand.
            log(f"Can't watch Holly's process (error {error}): the face won't close by itself if he crashes")
            return
    else:
        while True:
            try:
                os.kill(pid, 0)  # (on POSIX, signal 0 only checks it's there)
            except ProcessLookupError:
                break
            except PermissionError:
                pass
            time.sleep(1)
    log("Holly has gone: closing the face")
    os._exit(0)


class Decimator:
    """48 kHz to 16 kHz for Ditto, filtered across block boundaries."""

    def __init__(self):
        self.taps = firwin(63, 7200, fs=SR)
        self.zi = lfilter_zi(self.taps, 1.0) * 0.0

    def __call__(self, block):
        out, self.zi = lfilter(self.taps, 1.0, block, zi=self.zi)
        return out[::3].astype(np.float32)


class Turn:
    """One reply from Holly, and where it landed on the timeline (in 48 kHz samples)."""

    def __init__(self, number):
        self.number = number
        self.pending = collections.deque()  # audio not yet on the timeline
        self.pending_len = 0
        self.ended = False
        self.start = self.voice = self.last = None
        self.underruns = 0
        self.gap = False  # a pause is expected (Holly is searching): not a stutter
        self.times = {}  # "start", "voice", "end": when each left the speakers
        self.reported = False


class Timeline:
    """The continuous audio stream: fed to Ditto and the speakers, in step."""

    def __init__(self, sdk, lead, send, idle_head=0.2, sleep_after=30.0):
        self.sdk = sdk
        # Asleep: Ditto is skipped and the window holds its last frame. Ditto numbers its
        # frames (and looks up their controls) by its own count, which falls behind the
        # timeline while it sleeps; chunk_offset[c] is how far, for each chunk sent.
        self.sleep_after = sleep_after
        self.last_active = time.perf_counter()
        self.asleep = False
        # Timeline frames from held_from up to held_until were skipped (held_until is None
        # while still asleep): the window shows the last frame Ditto drew before them.
        self.held_from = self.held_until = None
        self.chunk_offset: dict[int, int] = {}
        self.idle_head = idle_head  # how much of Ditto's head movement is kept in silence
        self.pose_out_step = max(1 - idle_head, 0.05) / (POSE_OUT_S * 25)  # per frame
        self.lead = lead
        self.send = send
        self.lock = threading.Lock()
        self.turn: Turn | None = None
        self.finishing: list[Turn] = []  # ended turns still playing out
        self.fed = 0  # timeline samples handed to the speakers' queue
        self.blocks = collections.deque()  # (timeline position, samples) not yet fully played
        self.block_pos = 0  # samples of blocks[0] already played
        self.played = 0  # timeline position of the next sample out
        self.clock = (0, time.perf_counter())  # (timeline position, when it leaves the speakers)
        self.decimate = Decimator()
        self.hist16 = collections.deque([np.zeros(BLOCK16, np.float32)] * 2, maxlen=3)
        self.chunks = 0
        self.voiced = {}  # frame -> speech in it?
        self.lips = 0.0  # 0 resting, 1 following the audio
        self.pose = idle_head  # 0 still, 1 moving as Ditto drives it
        self.doze = 0.0  # 0 as Ditto draws it, 1 the photo with eyes closed (asleep)

    # -- from Holly ---------------------------------------------------------

    def begin(self, number):
        with self.lock:
            self._end_current()
            self.turn = Turn(number)
            self.last_active = time.perf_counter()

    def wake(self):
        """Someone started speaking: wake for WAKE_GRACE_S in case Holly answers; his reply
        then keeps him awake. Never shortens the time already left, and doesn't add a full
        sleep_after either: speech that isn't for Holly shouldn't keep his face awake when
        his app has stopped listening without the wake word."""
        now = time.perf_counter()
        self.last_active = max(self.last_active, now - self.sleep_after + WAKE_GRACE_S)

    def held(self, k):
        """Is timeline frame k one Ditto skipped, so the window should hold its last frame?
        Only those: the frames drawn before sleeping still play out (the feeder runs a lead
        ahead of the speakers), or he froze mid-word, 0.6 s before his reply ended."""
        return self.held_from is not None and k >= self.held_from and (
            self.held_until is None or k < self.held_until)

    def _quiet_for(self):
        """Seconds since anything was said or played; 0 while a reply is under way."""
        if self.turn is not None or self.finishing:
            return 0.0
        return time.perf_counter() - self.last_active

    def _sleepy(self):
        return self.sleep_after > 0 and self._quiet_for() > self.sleep_after

    def audio(self, samples, rate):
        if rate != SR:
            samples = resample_poly(samples, SR, rate).astype(np.float32)
        with self.lock:
            if self.turn and not self.turn.ended:
                self.turn.pending.append(samples)
                self.turn.pending_len += len(samples)
                self.turn.gap = False

    def expect_gap(self):
        """The audio so far is followed by a pause (a search), until more comes."""
        with self.lock:
            if self.turn:
                self.turn.gap = True

    def end(self):
        with self.lock:
            self._end_current()

    def clear(self):
        with self.lock:
            if self.turn:
                self.turn.pending.clear()
                self.turn.pending_len = 0
            self._end_current()

    def _end_current(self):
        if self.turn and not self.turn.ended:
            self.turn.ended = True
            self.finishing.append(self.turn)
        self.turn = None

    # -- the feeder: keeps the speakers' queue `lead` seconds ahead ---------

    def run_feeder(self):
        while True:
            with self.lock:
                ahead = self.fed - self.played
            if ahead < self.lead * SR:
                self._feed_block()
            else:
                time.sleep(0.005)

    def _take(self, turn, n):
        out = []
        while n and turn.pending:
            head = turn.pending[0]
            if len(head) <= n:
                out.append(turn.pending.popleft())
                n -= len(head)
            else:
                out.append(head[:n])
                turn.pending[0] = head[n:]
                n = 0
        taken = np.concatenate(out) if out else np.zeros(0, np.float32)
        turn.pending_len -= len(taken)
        return taken

    def _feed_block(self):
        pos = self.fed
        block = np.zeros(BLOCK, np.float32)
        with self.lock:
            for turn in [*self.finishing, self.turn]:
                if turn is None:
                    continue
                if not turn.pending_len:
                    if turn is self.turn and turn.start is not None and not turn.gap:
                        turn.underruns += 1  # Holly's voice ran out mid-reply: a stutter
                    continue
                taken = self._take(turn, BLOCK)
                block[: len(taken)] = taken
                if turn.start is None:
                    turn.start = pos
                if turn.voice is None:
                    loud = np.flatnonzero(np.abs(taken) > VOICE_LEVEL)
                    if loud.size:
                        turn.voice = pos + int(loud[0])
                turn.last = pos + len(taken)
                if len(taken) < BLOCK and not turn.ended and not turn.gap:
                    turn.underruns += 1  # Holly's voice ran out mid-reply: a stutter
                break  # one turn per block; a new turn waits for the old one's audio
            self.blocks.append((pos, block))
            self.fed += BLOCK
        # Ditto: block j completes the lookahead for chunk j-1 (none for block 0).
        self.hist16.append(self.decimate(block))
        first = pos // FRAME
        for k in range(first, first + BLOCK // FRAME):
            seg = block[(k - first) * FRAME:(k - first + 1) * FRAME]
            self.voiced[k] = float(np.sqrt(np.mean(seg * seg))) > SPEECH_RMS
        if pos == 0:
            return
        chunk_first = first - BLOCK // FRAME  # timeline frame of this chunk's first frame
        if self._sleepy():
            if not self.asleep:
                log("Asleep: Ditto paused")
                self.asleep = True
                self.held_from, self.held_until = chunk_first, None
            return
        offset = chunk_first - self.chunks * (BLOCK // FRAME)
        if self.asleep:
            log(f"Awake after {chunk_first - self.held_from} skipped frames")
            self.asleep = False
            self.held_until = chunk_first
        self.chunk_offset[self.chunks] = offset
        self._set_lips(chunk_first, first, offset)
        window = np.concatenate(self.hist16)[1280:1280 + 6480]
        self.sdk.run_chunk(window, CHUNK)
        self.chunks += 1

    def _set_lips(self, start, stop, offset=0):
        """Tell Ditto how much the lips and head follow the audio for timeline frames
        start..stop-1, which are Ditto's frames start-offset..stop-1-offset."""
        ctrl = self.sdk.ctrl_info

        def speaking(k, hang, ahead):
            return any(self.voiced.get(i, False) for i in range(k - hang, k + ahead + 1))

        for k in range(start, stop):
            target = 1.0 if speaking(k, LIPS_HANG, LIPS_AHEAD) else 0.0
            self.lips += max(-LIPS_STEP, min(LIPS_STEP, target - self.lips))
            # Dozing off: the head returns to the photo's pose and the face to the photo's
            # expression with the eyes shut ("doze", see face_controls), so he sleeps as the
            # still, eyes closed. With the pose alone, he froze on whatever expression the
            # animation had reached. The eyes close slowly and open quickly on waking.
            dozing = self.sleep_after > 0 and self._quiet_for() > self.sleep_after - DOZE_S
            target = 1.0 if speaking(k, POSE_HANG, POSE_AHEAD) else 0.0 if dozing else self.idle_head
            step = POSE_STEP if target > self.pose else self.pose_out_step
            self.pose += max(-step, min(step, target - self.pose))
            self.doze = min(1.0, self.doze + DOZE_STEP) if dozing else max(0.0, self.doze - WAKE_STEP)
            ctrl[k - offset] = {"vad_alpha": self.lips, "pose_alpha": self.pose, "doze": self.doze}
        for old in [i for i in self.voiced if i < start - max(LIPS_HANG, POSE_HANG)]:
            del self.voiced[old]
        for old in [i for i in ctrl if isinstance(i, int) and i < start - offset - 250]:
            del ctrl[old]
        for old in [c for c in self.chunk_offset if c < self.chunks - 100]:
            del self.chunk_offset[old]

    # -- the speakers -------------------------------------------------------

    def callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        dac = time.perf_counter() + max(time_info.outputBufferDacTime - time_info.currentTime, 0.0)
        with self.lock:
            start = self.played
            filled = 0
            while filled < frames and self.blocks:
                bpos, block = self.blocks[0]
                n = min(frames - filled, BLOCK - self.block_pos)
                out[filled:filled + n] = block[self.block_pos:self.block_pos + n]
                filled += n
                self.block_pos += n
                if self.block_pos >= BLOCK:
                    self.blocks.popleft()
                    self.block_pos = 0
            out[filled:] = 0
            self.played += filled
            self.clock = (start, dac)
            end = self.played
            for turn in [*self.finishing, *([self.turn] if self.turn else [])]:
                for key, at in (("start", turn.start), ("voice", turn.voice)):
                    if at is not None and key not in turn.times and at <= end:
                        turn.times[key] = dac + max(at - start, 0) / SR
            # A turn is done once it has ended, all its audio is on the timeline, and the
            # last of it is in this callback. "end" used to be stamped the first time
            # playback caught up with the audio so far, which happens mid-reply whenever
            # the voice runs dry (most replies, and every search). So "done" came as soon
            # as the last audio was queued, up to the 0.6 s lead early, and the mic,
            # listening again 0.3 s after, heard Holly's last words and answered them.
            done = [t for t in self.finishing if t.pending_len == 0 and (t.last is None or t.last <= end)]
            if done:
                self.last_active = time.perf_counter()  # the quiet spell starts when he stops
            for turn in done:
                if turn.last is not None:
                    turn.times["end"] = dac + max(turn.last - start, 0) / SR
                self.finishing.remove(turn)
        for turn in done:
            self.send({
                "type": "done", "turn": turn.number,
                "first_played": turn.times.get("start"), "first_voice": turn.times.get("voice"),
                "last_played": turn.times.get("end"),
                "underruns": turn.underruns,
            })

    def now(self):
        """The timeline position leaving the speakers at this moment."""
        with self.lock:
            pos, dac = self.clock
        return pos + (time.perf_counter() - dac) * SR


class Frames:
    """Ditto's finished frames, numbered from the start of the timeline."""

    def __init__(self, chunk_offset=None):
        self.lock = threading.Lock()
        self.frames = {}
        self.count = 0  # frames Ditto has made
        self.latest = -1  # timeline frame of the newest
        self.chunk_offset = {} if chunk_offset is None else chunk_offset  # Ditto chunk -> frames skipped before it

    def __call__(self, img, fmt="rgb"):
        with self.lock:
            k = self.count + self.chunk_offset.get(self.count // (BLOCK // FRAME), 0)
            self.frames[k] = img
            self.latest = k
            self.count += 1

    close = staticmethod(lambda: None)

    def at(self, k):
        """Frame k, or the latest before it if Ditto is behind; older frames are dropped."""
        with self.lock:
            for old in [i for i in self.frames if i < k]:
                if len(self.frames) > 1:
                    del self.frames[old]
            if k in self.frames:
                return self.frames[k], True
            if self.frames:
                return self.frames[max(self.frames)], False
            return None, False


class Link:
    """The socket to Holly's app."""

    def __init__(self, port, timeline_ref):
        self.server = socket.create_server(("127.0.0.1", port))
        self.conn = None
        self.send_lock = threading.Lock()
        self.timeline_ref = timeline_ref

    def send(self, msg):
        data = json.dumps(msg).encode()
        with self.send_lock:
            if self.conn:
                try:
                    self.conn.sendall(HEADER.pack(b"J", len(data)) + data)
                except OSError:
                    self.conn = None

    def run(self):
        while True:
            conn, _ = self.server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.send_lock:
                self.conn = conn
            log("Holly connected")
            try:
                self._serve(conn)
            except OSError:
                pass
            log("Holly disconnected")
            self.timeline_ref().clear()
            with self.send_lock:
                self.conn = None

    def _recv(self, conn, n):
        buf = bytearray()
        while len(buf) < n:
            part = conn.recv(n - len(buf))
            if not part:
                raise OSError("closed")
            buf += part
        return bytes(buf)

    def _serve(self, conn):
        rate = SR
        timeline = self.timeline_ref()
        while True:
            kind, size = HEADER.unpack(self._recv(conn, HEADER.size))
            payload = self._recv(conn, size) if size else b""
            if kind == b"R":
                rate = struct.unpack("<I", payload)[0]
            elif kind == b"B":
                timeline.begin(struct.unpack("<I", payload)[0])
            elif kind == b"A":
                timeline.audio(np.frombuffer(payload, np.float32).copy(), rate)
            elif kind == b"E":
                timeline.end()
            elif kind == b"C":
                timeline.clear()
            elif kind == b"G":
                timeline.expect_gap()
            elif kind == b"W":
                timeline.wake()


def lips_only_vad(x_d_info, dst, alpha):
    """Ditto's ctrl_vad blends the whole expression towards dst, though it builds a lips-only
    mask; it was evidently meant to move the lips alone. This does, so blinks carry on."""
    lips = np.zeros((21, 3), dtype=np.float32)
    lips[[6, 12, 14, 17, 19, 20]] = 1
    lips = lips.reshape(1, -1)
    exp = x_d_info["exp"]
    x_d_info["exp"] = exp * (1 - lips) + (exp * alpha + dst["exp"] * (1 - alpha)) * lips
    return x_d_info


def face_controls(motion_stitch):
    """Two per-frame controls added to Ditto's motion step:
    - "pose_alpha": Ditto sways the head while Holly is silent, too. 1 leaves the head as
      Ditto drives it, 0 holds it at the source photo's pose.
    - "doze": 0 leaves the expression as Ditto drives it, 1 is the photo's own expression
      with the eyes closed (EYES_CLOSED): he falls asleep as the still.
    ctrl_motion gets the frame's controls but not the source, so __call__ notes it first
    (one motion-stitch thread, so a plain dict is safe)."""
    eyes_closed = EYES_CLOSED.reshape(1, -1)
    call, ctrl_motion = motion_stitch.MotionStitch.__call__, motion_stitch.ctrl_motion
    source = {}

    def noting_call(self, x_s_info, x_d_info, **kwargs):
        source["info"] = x_s_info
        return call(self, x_s_info, x_d_info, **kwargs)

    def holding_ctrl_motion(x_d_info, **kwargs):
        alpha = kwargs.get("pose_alpha", 1.0)
        if alpha < 1 and "info" in source:
            src = source["info"]
            for k in POSE_KEYS:
                x_d_info[k] = src[k] + (x_d_info[k] - src[k]) * alpha
        doze = kwargs.get("doze", 0.0)
        if doze > 0 and "info" in source:
            asleep = source["info"]["exp"] + eyes_closed.reshape(source["info"]["exp"].shape)
            x_d_info["exp"] = x_d_info["exp"] * (1 - doze) + asleep * doze
        return ctrl_motion(x_d_info, **kwargs)

    motion_stitch.MotionStitch.__call__ = noting_call
    motion_stitch.ctrl_motion = holding_ctrl_motion


def quiet_onnxruntime(threads=4):
    """Ditto runs HuBERT (its audio features, a chunk every 0.2 s, silence included) and its
    face detectors through onnxruntime, asking for the GPU first. onnxruntime-gpu 1.32 is
    built for CUDA 13 and this venv's CUDA is torch's 12.8, so the GPU provider never loads
    and HuBERT runs on the CPU, with a thread per physical core (24) that spins between
    chunks: 72% of the machine, all the time. 4 threads that sleep instead: 61 ms a chunk
    against 40, and ~3% of the machine. Kept on the CPU, it costs no VRAM or GPU time."""
    import onnxruntime

    session = onnxruntime.InferenceSession

    def quiet_session(path, sess_options=None, providers=None, **kwargs):
        so = sess_options or onnxruntime.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return session(path, sess_options=so, providers=["CPUExecutionProvider"], **kwargs)

    onnxruntime.InferenceSession = quiet_session


def build_ditto(image, height, overlap):
    import core.atomic_components.motion_stitch as motion_stitch
    from fast_ditto import speed_up
    from stream_pipeline_online import StreamSDK

    motion_stitch.ctrl_vad = lips_only_vad
    face_controls(motion_stitch)
    quiet_onnxruntime()

    sdk = StreamSDK(str(CHECKPOINTS / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"), str(CHECKPOINTS / "ditto_pytorch"))
    speed_up(sdk, height)
    # overlap_v2: of the motion model's 80-frame window, how many frames overlap the previous
    # call. Ditto's streaming config uses 70, making motion 10 frames at a time and trailing
    # the audio by 10 more; 75 makes it 5 at a time (one chunk), for less delay.
    kwargs = {"online_mode": True, "sampling_timesteps": 10, "overlap_v2": overlap}
    out = FACE / "data" / "out" / "live.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    def setup(sink):
        sdk.setup(str(image), str(out), **kwargs)
        try:
            sdk.writer.close()  # Ditto opens an mp4 writer; the window takes the frames instead
        except Exception:
            pass
        Path(sdk.tmp_output_path).unlink(missing_ok=True)
        sdk.writer = sink
        sdk.writer_pbar.close()
        sdk.writer_pbar = type("Quiet", (), {"update": lambda *a: None, "close": lambda *a: None})()
        sdk.setup_Nd(N_d=10**8)  # effectively endless

    # Warm up: the first frames compile and record each model (about a minute).
    log("Warming up Ditto (compiling, about a minute)...")
    t = time.perf_counter()
    warm = Frames()
    setup(warm)
    silence = np.zeros(6480, np.float32)
    for _ in range(20):
        sdk.run_chunk(silence, CHUNK)
    sdk.close()
    log(f"Warm-up done in {time.perf_counter() - t:.0f} s ({warm.count} frames)")
    return sdk, setup


def main():
    p = argparse.ArgumentParser(description="Holly's face window")
    p.add_argument("--image", default=str(FACE / "data" / "holly_colour.jpg"))
    p.add_argument("--port", type=int, default=50555)
    p.add_argument("--lead", type=float, default=0.6, help="seconds of audio queued ahead of the speakers")
    p.add_argument("--overlap", type=int, default=75, help="Ditto's overlap_v2 (its own streaming config: 70)")
    p.add_argument("--height", type=int, default=0,
                   help="height Ditto's frames are drawn at; 0 for the still's own (the window scales them)")
    p.add_argument("--idle-head", type=float, default=0.2,
                   help="head movement kept while Holly is silent: 0 still, 1 as Ditto drives it")
    p.add_argument("--sleep-after", type=float, default=30.0,
                   help="seconds after Holly last spoke before Ditto pauses to spare the GPU (0: never)")
    p.add_argument("--host-api", default="Windows WASAPI")
    p.add_argument("--device", default="", help="output device name (or part of it)")
    p.add_argument("--parent", type=int, default=0, help="Holly's process ID: the face exits when it ends")
    p.add_argument("--crt", action="store_true", help="an old monitor's look (face/screen.py)")
    p.add_argument("--case", default="", help="with --crt: a picture of the monitor, its screen opening see-through")
    p.add_argument("--full-screen", action="store_true", help="open full screen (F11 or Escape leaves it)")
    args = p.parse_args()
    if args.parent:
        # First, before Ditto's minute of warm-up: Holly can crash during his own.
        threading.Thread(target=exit_with, args=(args.parent,), daemon=True, name="parent").start()

    sdk, setup = build_ditto(Path(args.image).resolve(), args.height or None, args.overlap)
    frames = Frames()
    setup(frames)
    # What the warm-up used and let go: ~190 MB of VRAM back for good, measured, at no
    # cost to speed. Ditto's working memory comes straight back, so only now, once.
    import torch
    torch.cuda.empty_cache()

    timeline = None
    link = Link(args.port, lambda: timeline)
    timeline = Timeline(sdk, args.lead, link.send, args.idle_head, args.sleep_after)
    frames.chunk_offset = timeline.chunk_offset  # before the feeder sends Ditto anything

    sys.path.insert(0, str(FACE.parent))
    from app.player import find_device  # same device choice as Holly's own player

    index, api = find_device("output", args.host_api, args.device)
    extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in api else None
    stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=SR // 50, latency="low",
                             device=index, extra_settings=extra, callback=timeline.callback)
    stream.start()
    threading.Thread(target=timeline.run_feeder, daemon=True, name="feeder").start()
    threading.Thread(target=link.run, daemon=True, name="link").start()
    log(f"Ready on port {args.port}, playing through {sd.query_devices(index)['name']} ({api}), "
        f"lead {args.lead:.2f} s")

    # The window: pygame's, drawn with OpenGL (face/screen.py). Each frame goes to a GPU
    # texture that a shader draws as large as fits the window, centred on black, or, with
    # --crt, on an old monitor: ~1 ms a frame at any size, full screen included. Tk's
    # window, scaling in Python, took 47 ms a frame at 2560x1440 and 97 ms at 4K (past the
    # 40 ms there is). OpenCV's scaled well but showed a crosshair cursor and left gaps when
    # snapped to a screen edge in Windows 11. pygame's own SCALED mode only scales by whole
    # numbers. SDL2's 2D renderer scaled well, but can't run a shader: its CRT look was
    # layers blended over the picture, and its glow read as a halo.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import moderngl
    import pygame
    from screen import Screen, opening

    still = Image.open(args.image).convert("RGB")
    w, h = still.size
    fw, fh = (round(args.height * w / h), args.height) if args.height else (w, h)  # Ditto's frames
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    window = pygame.Window("Holly", (round(WINDOW_HEIGHT * w / h), WINDOW_HEIGHT), opengl=True, resizable=True)
    face = still.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize((64, 64))
    window.set_icon(pygame.image.frombuffer(face.tobytes(), face.size, "RGB"))
    case = None
    if args.crt and args.case:
        if Path(args.case).exists():
            case = np.asarray(Image.open(args.case).convert("RGBA"))
            try:
                opening(case)
            except ValueError as e:  # a JPEG, or a PNG without its screen cut out: it took the face down
                log(f"Can't use the monitor picture ({e}): a plain bezel instead")
                case = None
        else:  # it's yours, kept out of git: a fresh copy has the plain bezel
            log(f"No monitor picture at {args.case}: a plain bezel instead")
    screen = Screen(moderngl.create_context(), (fw, fh), crt=args.crt, case=case)
    winmm = ctypes.windll.winmm if sys.platform == "win32" else None
    if winmm:
        winmm.timeBeginPeriod(1)  # 1 ms sleeps, not Windows' default 15.6 ms ticks
    shown = {"k": -1, "late": 0, "frames": 0, "skipped": 0, "draw": 0.0, "ahead": [], "made": 0,
             "report": time.perf_counter(), "click": 0.0, "full": False, "shown_at": 0.0}

    def show(img=None):
        """Draw the frame (updated to `img`, (h, w, 3) RGB, if given), as large as fits."""
        if img is not None:
            screen.update(img)
        screen.draw(window.size, time.perf_counter())
        window.flip()
        shown["shown_at"] = time.perf_counter()

    def draw(img):
        show(np.ascontiguousarray(img))  # Ditto's frames aren't always laid out in one block

    def full_screen(on=None):
        on = not shown["full"] if on is None else on
        if on:
            window.set_fullscreen(desktop=True)  # on whichever screen the window is on
        else:
            window.set_windowed()
        shown["full"] = on

    if args.full_screen:
        full_screen(True)
    show(np.asarray(still.resize((fw, fh))))  # until Ditto's first frame

    def handle_events():
        """Returns False once the window is closed."""
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                full_screen()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                full_screen(False)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                now = time.perf_counter()
                if now - shown["click"] < 0.4:  # a double-click
                    full_screen()
                    now = 0.0
                shown["click"] = now
            elif event.type in (pygame.WINDOWRESIZED, pygame.WINDOWSIZECHANGED, pygame.WINDOWEXPOSED):
                show()  # refit; asleep, nothing else redraws the held frame
        return True

    def tick():
        k = int(timeline.now() // FRAME)
        if k != shown["k"]:
            shown["skipped"] += max(k - shown["k"] - 1, 0) if shown["k"] >= 0 else 0
            if timeline.held(k):  # asleep: keep showing the last frame
                shown["k"] = k
                return
            shown["ahead"].append(frames.latest - k)  # frames Ditto has ready beyond this one
            img, on_time = frames.at(k)
            if img is not None:
                t = time.perf_counter()
                draw(img)
                shown["draw"] += time.perf_counter() - t
                shown["frames"] += 1
                shown["late"] += not on_time
            shown["k"] = k
        now = time.perf_counter()
        if screen.rolling(now) and now - shown["shown_at"] > CRT_REDRAW_S:
            show()  # the bar rolls on smoothly, between frames and while he sleeps
        if now - shown["report"] > 10:
            n = max(shown["frames"], 1)
            ahead = sorted(shown["ahead"]) or [0]
            log(f"shown {shown['frames']}, late {shown['late']}, skipped {shown['skipped']}, "
                f"Ditto made {frames.count - shown['made']} in {now - shown['report']:.0f} s, "
                f"ready ahead min {ahead[0]} median {ahead[len(ahead) // 2]}, "
                f"draw {shown['draw'] / n * 1000:.1f} ms")
            shown.update(late=0, frames=0, skipped=0, draw=0.0, ahead=[], made=frames.count, report=now)

    try:
        while handle_events():
            tick()
            time.sleep(0.002)
    finally:
        if winmm:
            winmm.timeEndPeriod(1)
        pygame.quit()
    stream.stop()


if __name__ == "__main__":
    main()

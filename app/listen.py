"""Milestone 3: the full voice loop. Speak to Holly; he answers out loud.

    python -m app.listen [--model MODEL] [--face] [--no-search] [--no-wake] [--forget]

Silero VAD hears you start and stop, faster-whisper transcribes what you
said, and the milestone 2 pipeline answers. Timings count from the moment
you stopped speaking.

Holly answers only when you say his name near the start or end of what you
say ("Holly, what's...", "...tomorrow, Holly?"), or carry on within a few
seconds of his last reply. Anything else is ignored. The name is found in
Whisper's transcript, which is made anyway, so it costs nothing.

This is half-duplex: the mic is ignored while Holly talks and for a moment
after, so he doesn't hear himself. Interrupting him is milestone 5. Use a
headset anyway, as the plan advises.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
import sys
import subprocess
import time

from .announce import Announcement
from .console import DIM, RESET, YELLOW, stamp
from .mic import Mic
from .speak import Holly, TurnTimes
from .yours import FORGET_HELP, RESET_HELP
from .stt import Transcriber
from .turn import TurnModel, verdict
from .vad import TurnDetector

# After Holly finishes, mic audio is ignored this long, so the tail of his
# voice in the room isn't taken for you speaking.
ECHO_TAIL_S = 0.3
# Below this much free VRAM after loading, a browser tab or the next
# allocation spike can spill GPU memory into system RAM.
LOW_VRAM_GB = 1.5
# Asleep, his notes are written once the room has been quiet this long. Any
# speech cancels them (they're redone after the next quiet spell): with one
# try per sleep, a cough or the TV could stop them for good.
COMPACT_QUIET_S = 5.0


def gpu_free_gb() -> float | None:
    """Free VRAM across all processes, from nvidia-smi. torch.cuda.mem_get_info
    can't be trusted here: under Windows' driver model it reported 14 GB free
    when under 1 GB was."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return int(out.split()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def addressed(text: str, wake_words: set[str], edge: int) -> bool:
    """Is Holly's name among the first or last `edge` words of `text`?"""
    words = [w.removesuffix("'s") for w in re.findall(r"[a-z']+", text.lower())]
    return any(w in wake_words for w in words[:edge] + words[-edge:])


def for_holly(text: str, started: float, awake_until: float, wake_words: set[str], edge: int) -> bool:
    """Does he answer `text`, begun at `started`? Always with no wake words; awake (begun
    by `awake_until`), with no name needed; asleep, only if he's `addressed`."""
    return not wake_words or started <= awake_until or addressed(text, wake_words, edge)


def after_speaking(now: float, last_played: float | None, awake_s: float) -> tuple[float, float]:
    """(deaf_until, awake_until) once he's replied, counted from when his last sample left
    the speakers (`last_played`), or `now` if that's later: the mic is deaf for the echo
    tail after it, and he's awake, no name needed, for `awake_s`."""
    spoke_until = max(now, last_played or 0.0)
    return spoke_until + ECHO_TAIL_S, spoke_until + awake_s


class ListeningHolly(Holly):
    HINT = ""

    def __init__(self, *args, wake: bool | None = None, mic: str | None = None, **kwargs):
        """`mic`: the microphone, by name or part of it, in place of holly.toml's input_device
        (--mic "Steam Streaming" for a phone over Steam Link)."""
        super().__init__(*args, **kwargs)
        if wake is False:
            self.cfg = dataclasses.replace(self.cfg, listen=dataclasses.replace(self.cfg.listen, wake_words=()))
        if mic is not None:
            self.cfg = dataclasses.replace(self.cfg, listen=dataclasses.replace(self.cfg.listen, input_device=mic))
        self.wake_words = set(self.cfg.listen.wake_words)
        self.awake_until = 0.0  # the follow-up window: no name needed until then
        self.quiet_since = 0.0  # when the room last went quiet; notes wait for COMPACT_QUIET_S of it

    async def start(self) -> None:
        await super().start()
        listen = self.cfg.listen
        print(f"{DIM}Loading Whisper {listen.whisper_model} and Silero VAD...{RESET}", flush=True)
        self.stt = await asyncio.to_thread(Transcriber, listen)
        await asyncio.to_thread(self.stt.warm_up)
        turn_model = None
        if listen.turn_model:
            # Whether a pause is the end of what you're saying (app/turn.py). If it can't be
            # had, a turn ends after a pause, as it did before it.
            try:
                turn_model = await asyncio.to_thread(TurnModel, listen.turn_model)
            except Exception as e:
                print(f"{YELLOW}The turn-ending model ({listen.turn_model}) couldn't be loaded ({e!r}): a turn "
                      f"ends after {listen.end_silence_ms} ms of silence.{RESET}", flush=True)
        self.detector = TurnDetector(listen, turn_model, on_check=self._turn_checked)
        self.mic = Mic(self.cfg.audio.host_api, listen.input_device, listen.input_channels)
        self.mic.start()
        free_gb = gpu_free_gb()
        if free_gb is not None and free_gb < LOW_VRAM_GB:
            print(
                f"{YELLOW}Only {free_gb:.1f} GB of VRAM is free. If it runs out, Windows moves GPU memory "
                f"into system RAM and Holly slows to a crawl (replies of 5-10 s). Close other GPU-heavy apps, "
                f"such as browser tabs, games or another Holly.{RESET}"
            )
        elif free_gb is not None:
            print(f"{DIM}VRAM free: {free_gb:.1f} GB.{RESET}")
        how = (f"Say \"Holly\" to talk to him; for {listen.awake_s:.0f} s after he answers, he's awake "
               "and you don't need to." if self.wake_words else "Speak to Holly.")
        print(f"{DIM}Listening on {self.mic.device_name}. {how} Ctrl-C to quit.{RESET}\n")

    async def next_utterance(self):
        """Wait for you to say something; returns the detector's Utterance, or an
        Announcement (timers come due, a report finished), checked while nobody's speaking."""
        hearing = False
        while True:
            block, captured = await self.mic.queue.get()
            if not hearing:
                if announcement := self.pending():
                    self.background.wake()  # as for speech: he's about to talk
                    self.detector.reset()
                    return announcement
                self._maybe_doze()
            if captured < self.deaf_until:
                continue
            utterance = self.detector.feed(block, captured)
            if self.detector.in_speech and not hearing:
                hearing = True
                self.player.wake()  # the face may be asleep; it needs ~0.6 s to come round
                self.background.wake()
                print(f"{DIM}(listening){RESET}", end="\r", flush=True)
            if utterance:
                return utterance
            if hearing and not self.detector.in_speech:
                hearing = False
                self.quiet_since = time.perf_counter()

    def _turn_checked(self, p: float, audio) -> None:
        """At a pause, the turn model's answer, and what he does with it, kept on screen: on the
        first live try, overwritten, there was no telling why a fragment got an answer."""
        print(f"\r{DIM}  (a pause: {p:.2f}, {verdict(p, self.cfg.listen)}){RESET}", flush=True)

    def _maybe_doze(self) -> None:
        """Once he's asleep and the room is quiet: his notes, then any research
        (app/background.py)."""
        now = time.perf_counter()
        if now >= self.awake_until and now - self.quiet_since >= COMPACT_QUIET_S:
            self.background.doze()

    async def run(self) -> None:
        await self.start()
        self.deaf_until = 0.0
        try:
            while True:
                utterance = await self.next_utterance()
                if isinstance(utterance, Announcement):  # timers due, a report finished: he speaks up
                    await self.announce(utterance)
                else:
                    text = await asyncio.to_thread(self.stt.transcribe, utterance.audio)
                    times = TurnTimes(start=utterance.speech_end, transcribed=time.perf_counter())
                    if not text:
                        continue
                    started = utterance.speech_end - utterance.duration_ms / 1000
                    if not for_holly(text, started, self.awake_until, self.wake_words,
                                     self.cfg.listen.wake_edge_words):
                        print(f"{stamp()}{DIM}(not for Holly) {text}{RESET}")
                        continue
                    print(f"{stamp()}You: {text}")
                    try:
                        await self.respond(text, times)
                    except Exception as e:  # the voice or the speakers failed: report it and carry on
                        print(f"{YELLOW}  (that reply failed: {e!r}){RESET}\n", flush=True)
                # Drop what the mic heard while Holly spoke, until his last sample has
                # left the speakers, plus the echo tail. Awake for the same span his face
                # stays awake for (the face server's --sleep-after): awake, no wake word
                # needed; asleep, it is.
                self.deaf_until, self.awake_until = after_speaking(
                    time.perf_counter(), self.player.last_played, self.cfg.listen.awake_s)
                self.detector.reset()
        except KeyboardInterrupt:
            print()
        finally:
            self.background.stop()
            self.mic.close()
            self.player.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Holly out loud")
    parser.add_argument("--model", help="LM Studio chat model key (default: holly.toml)")
    parser.add_argument("--face", action=argparse.BooleanOptionalAction, help="show Holly's face (default: holly.toml)")
    parser.add_argument("--full-screen", action=argparse.BooleanOptionalAction,
                        help="open his face full screen, the face on too (default: holly.toml)")
    parser.add_argument("--search", action=argparse.BooleanOptionalAction,
                        help="let Holly search the web (default: holly.toml)")
    parser.add_argument("--wake", action=argparse.BooleanOptionalAction,
                        help="answer only when you say his name (default: on, unless wake_words is empty)")
    parser.add_argument("--mic", metavar="NAME",
                        help="the microphone, by name or part of it (default: holly.toml's input_device, else "
                             "Windows' default); \"Steam Streaming\" for a phone over Steam Link")
    parser.add_argument("--forget", action="store_true", help=FORGET_HELP)
    parser.add_argument("--reset", action="store_true", help=RESET_HELP)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(ListeningHolly(args.model, args.face, args.search, args.forget, args.reset,
                                   wake=args.wake, mic=args.mic, full_screen=args.full_screen).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

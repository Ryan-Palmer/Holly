"""Milestone 3: the full voice loop. Speak to Holly; he answers out loud.

    python -m app.listen [--model MODEL] [--engine voxcpm|orpheus] [--voice VOICE] [--no-wake]

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

from .mic import Mic
from .speak import DIM, RESET, Holly, TurnTimes
from .stt import Transcriber
from .vad import TurnDetector

# After Holly finishes, mic audio is ignored this long, so the tail of his
# voice in the room isn't taken for you speaking.
ECHO_TAIL_S = 0.3
# Below this much free VRAM after loading, a browser tab or the next
# allocation spike can spill GPU memory into system RAM.
LOW_VRAM_GB = 1.5
YELLOW = "\033[33m"


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


class ListeningHolly(Holly):
    HINT = ""

    def __init__(self, *args, wake: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if wake is False:
            self.cfg = dataclasses.replace(self.cfg, listen=dataclasses.replace(self.cfg.listen, wake_words=()))
        self.wake_words = set(self.cfg.listen.wake_words)
        self.awake_until = 0.0  # the follow-up window: no name needed until then

    async def start(self) -> None:
        await super().start()
        listen = self.cfg.listen
        print(f"{DIM}Loading Whisper {listen.whisper_model} and Silero VAD...{RESET}", flush=True)
        self.stt = await asyncio.to_thread(Transcriber, listen)
        await asyncio.to_thread(self.stt.warm_up)
        self.detector = TurnDetector(listen)
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
        how = (f"Say \"Holly\" to talk to him; after he answers, you have {listen.follow_up_s:.0f} s "
               "to carry on without it." if self.wake_words else "Speak to Holly.")
        print(f"{DIM}Listening on {self.mic.device_name}. {how} Ctrl-C to quit.{RESET}\n")

    async def next_utterance(self):
        """Wait for you to say something; returns the detector's Utterance."""
        hearing = False
        while True:
            block, captured = await self.mic.queue.get()
            if captured < self.deaf_until:
                continue
            utterance = self.detector.feed(block, captured)
            if self.detector.in_speech and not hearing:
                hearing = True
                print(f"{DIM}(listening){RESET}", end="\r", flush=True)
            if utterance:
                return utterance
            if not self.detector.in_speech:
                hearing = False

    async def run(self) -> None:
        await self.start()
        self.deaf_until = 0.0
        try:
            while True:
                utterance = await self.next_utterance()
                text = await asyncio.to_thread(self.stt.transcribe, utterance.audio)
                times = TurnTimes(start=utterance.speech_end, transcribed=time.perf_counter())
                if not text:
                    continue
                started = utterance.speech_end - utterance.duration_ms / 1000
                if self.wake_words and started > self.awake_until and not addressed(
                        text, self.wake_words, self.cfg.listen.wake_edge_words):
                    print(f"{DIM}(not for Holly) {text}{RESET}")
                    continue
                print(f"You: {text}")
                await self.respond(text, times)
                # Drop what the mic heard while Holly spoke, until his last sample has
                # left the speakers, plus the echo tail.
                spoke_until = max(time.perf_counter(), self.player.last_played or 0.0)
                self.deaf_until = spoke_until + ECHO_TAIL_S
                self.awake_until = spoke_until + self.cfg.listen.follow_up_s
                self.detector.reset()
        except KeyboardInterrupt:
            print()
        finally:
            self.mic.close()
            self.player.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Holly out loud")
    parser.add_argument("--model", help="LM Studio chat model key (default: holly.toml)")
    parser.add_argument("--engine", choices=["voxcpm", "orpheus"], help="TTS engine (default: holly.toml)")
    parser.add_argument("--voice", help="Orpheus preset voice, with --engine orpheus")
    parser.add_argument("--face", action=argparse.BooleanOptionalAction, help="show Holly's face (default: holly.toml)")
    parser.add_argument("--search", action=argparse.BooleanOptionalAction,
                        help="let Holly search the web (default: holly.toml)")
    parser.add_argument("--wake", action=argparse.BooleanOptionalAction,
                        help="answer only when you say his name (default: on, unless wake_words is empty)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(ListeningHolly(args.model, args.engine, args.voice, args.face, args.search, wake=args.wake).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

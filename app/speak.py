"""Milestone 2: type to Holly, and she answers out loud in a stock voice.

    python -m app.speak [--model MODEL] [--voice VOICE]

The reply streams: the LLM's tokens are cut into sentences, each sentence
goes to TTS as soon as it's complete, and audio plays as it's decoded. So
Holly starts talking while the LLM is still writing.

Commands: /reset clears the conversation, /quit leaves.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import time

import torch

from .config import load_config
from .lmstudio import LmStudio, ReplyTiming
from .persona import Conversation
from .player import Player
from .speech_text import SentenceChunker
from .tts_orpheus import OrpheusTts

DIM, RESET, CYAN = "\033[2m", "\033[0m", "\033[36m"


@dataclasses.dataclass
class TurnTimes:
    """perf_counter() stamps for one reply, from pressing Enter."""

    enter: float
    first_token: float | None = None
    first_sentence: float | None = None
    first_audio: float | None = None
    first_played: float | None = None
    first_voice: float | None = None
    underruns: int = 0

    def report(self) -> str:
        def ms(t: float | None) -> str:
            return f"{(t - self.enter) * 1000:.0f}" if t else "-"

        return (
            f"first token {ms(self.first_token)}, first sentence {ms(self.first_sentence)}, "
            f"first audio {ms(self.first_audio)}, first sound {ms(self.first_played)}, "
            f"first voice {ms(self.first_voice)} ms"
            + (f", {self.underruns} underruns" if self.underruns else "")
        )


class Holly:
    def __init__(self, model: str | None, voice: str | None):
        cfg = load_config()
        if voice:
            cfg = dataclasses.replace(cfg, tts=dataclasses.replace(cfg.tts, voice=voice))
        if cfg.tts.device.startswith("cuda") and not torch.cuda.is_available():
            sys.exit("PyTorch can't see the GPU. Install a CUDA 12.8+ build (see README).")
        self.cfg = cfg
        self.model = model or cfg.llm.model
        self.lm = LmStudio(cfg.llm)
        self.convo = Conversation(cfg.persona)
        self.tts: OrpheusTts | None = None
        self.player: Player | None = None

    async def start(self) -> None:
        print(f"{DIM}Loading {self.model} and {self.cfg.tts.model}...{RESET}", flush=True)
        self.lm.ensure_loaded(self.model)
        self.lm.ensure_loaded(self.cfg.tts.model, self.cfg.tts.context_length)
        self.tts = OrpheusTts(self.cfg.tts, self.cfg.llm.base_url)
        await self.tts.warm_up()
        async for _ in self.lm.astream_reply(self.convo.messages_for("Hello."), ReplyTiming(), self.model):
            pass
        audio = self.cfg.audio
        self.player = Player(self.tts.sample_rate, audio.blocksize, audio.host_api, audio.output_device)
        self.player.start()
        print(
            f"{DIM}Holly is on {self.model}, voice {self.cfg.tts.voice}, "
            f"playing through {self.player.device_name}. /reset to start over, /quit to leave.{RESET}\n"
        )

    async def respond(self, user: str) -> None:
        assert self.tts and self.player
        times = TurnTimes(enter=time.perf_counter())
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        parts: list[str] = []
        self.player.begin_turn()

        async def think() -> None:
            chunker = SentenceChunker()
            timing = ReplyTiming()
            print(f"{CYAN}Holly:{RESET} ", end="", flush=True)
            async for delta in self.lm.astream_reply(self.convo.messages_for(user), timing, self.model):
                times.first_token = times.first_token or time.perf_counter()
                parts.append(delta)
                print(delta, end="", flush=True)
                for sentence in chunker.feed(delta):
                    times.first_sentence = times.first_sentence or time.perf_counter()
                    await sentences.put(sentence)
            for sentence in chunker.flush():
                times.first_sentence = times.first_sentence or time.perf_counter()
                await sentences.put(sentence)
            await sentences.put(None)
            print()

        async def speak() -> None:
            first = True
            while (sentence := await sentences.get()) is not None:
                # The first sentence's opening silence is pure latency; later
                # sentences keep theirs as the pause between sentences.
                async for audio in self.tts.stream(sentence, trim_lead=first):
                    first = False
                    times.first_audio = times.first_audio or time.perf_counter()
                    self.player.put(audio)
            self.player.end_turn()

        await asyncio.gather(think(), speak())
        await self.player.wait_done()
        times.first_played = self.player.first_played
        times.first_voice = self.player.first_voice
        times.underruns = self.player.underruns
        self.convo.record(user, "".join(parts).strip())
        print(f"{DIM}  {times.report()}{RESET}\n")

    async def run(self) -> None:
        await self.start()
        try:
            while True:
                user = (await asyncio.to_thread(input, "You: ")).strip()
                if not user:
                    continue
                if user in ("/quit", "/exit"):
                    return
                if user == "/reset":
                    self.convo.reset()
                    print(f"{DIM}Conversation cleared.{RESET}\n")
                    continue
                await self.respond(user)
        except (EOFError, KeyboardInterrupt):
            print()
        finally:
            if self.player:
                self.player.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Holly; she answers out loud")
    parser.add_argument("--model", help="LM Studio chat model key (default: holly.toml)")
    parser.add_argument("--voice", help="Orpheus voice (default: holly.toml)")
    args = parser.parse_args()
    # The Windows console defaults to cp1252, which can't print emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(Holly(args.model, args.voice).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Milestone 2: type to Holly, and he answers out loud in a stock voice.

    python -m app.speak [--model MODEL] [--engine voxcpm|orpheus] [--voice VOICE] [--no-search]

The reply streams: the LLM's tokens are cut into sentences, each sentence
goes to TTS as soon as it's complete, and audio plays as it's decoded. So
Holly starts talking while the LLM is still writing.

Holly can search the web (app/search.py). When his reply starts with a
search instead of words, he says a stock line ("Hang on, let me look that
up"), rendered at startup so it plays at once, while the search runs.

Commands: /reset clears the conversation, /quit leaves.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import json
import random
import sys
import time

import numpy as np
import torch

from .config import load_config
from .lmstudio import LmStudio, ReplyTiming
from .persona import Conversation
from .face_player import FacePlayer, start_server
from .player import Player
from . import clock, weather
from .search import TOOL, Searcher
from .speech_text import SentenceChunker
from .tts_common import Tts, make_tts

DIM, RESET, CYAN = "\033[2m", "\033[0m", "\033[36m"


@dataclasses.dataclass
class TurnTimes:
    """perf_counter() stamps for one reply, from pressing Enter or, when
    talking, from the end of your speech."""

    start: float
    transcribed: float | None = None
    first_token: float | None = None
    first_sentence: float | None = None
    first_audio: float | None = None
    first_played: float | None = None
    first_voice: float | None = None
    underruns: int = 0
    search_ms: float = 0.0  # searching and summarising, all searches

    def report(self) -> str:
        def ms(t: float | None) -> str:
            return f"{(t - self.start) * 1000:.0f}" if t else "-"

        return (
            (f"transcribed {ms(self.transcribed)}, " if self.transcribed else "")
            + f"first token {ms(self.first_token)}, first sentence {ms(self.first_sentence)}, "
            f"first audio {ms(self.first_audio)}, first sound {ms(self.first_played)}, "
            f"first voice {ms(self.first_voice)} ms"
            + (f", searching {self.search_ms:.0f} ms" if self.search_ms else "")
            + (f", {self.underruns} underruns" if self.underruns else "")
        )


class Holly:
    HINT = " /reset to start over, /quit to leave."

    def __init__(self, model: str | None, engine: str | None = None, voice: str | None = None,
                 face: bool | None = None, search: bool | None = None):
        cfg = load_config()
        if face is not None:
            cfg = dataclasses.replace(cfg, face=dataclasses.replace(cfg.face, enabled=face))
        if search is not None:
            cfg = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, enabled=search))
        tts = cfg.tts
        if engine:
            tts = dataclasses.replace(tts, engine=engine)
        if voice:
            tts = dataclasses.replace(tts, orpheus=dataclasses.replace(tts.orpheus, voice=voice))
        cfg = dataclasses.replace(cfg, tts=tts)
        if not torch.cuda.is_available():
            sys.exit("PyTorch can't see the GPU. Install a CUDA 12.8+ build (see README).")
        self.cfg = cfg
        self.model = model or cfg.llm.model
        self.lm = LmStudio(cfg.llm)
        # Every request sends the same tools: they're part of the start of the
        # prompt, so changing them between requests would break LM Studio's cache.
        self.searcher = Searcher(cfg.search, self.lm, self.model) if cfg.search.enabled else None
        self.tools = ([TOOL, clock.TOOL] + ([weather.TOOL] if cfg.search.weather else [])) if self.searcher else None
        search_prompt = cfg.search.prompt.read_text(encoding="utf-8") if self.searcher else ""
        self.convo = Conversation(cfg.persona, search_prompt)
        self.fillers: list[np.ndarray] = []
        self.last_filler = -1
        self.tts: Tts | None = None
        self.player: Player | FacePlayer | None = None

    def voice_name(self) -> str:
        t = self.cfg.tts
        return "Holly (VoxCPM2 clone)" if t.engine == "voxcpm" else f"Orpheus {t.orpheus.voice}"

    async def start(self) -> None:
        tts = self.cfg.tts
        audio = self.cfg.audio
        print(f"{DIM}Loading {self.model} and the {self.voice_name()} voice...{RESET}", flush=True)
        # Holly needs nearly all of the GPU. Anything else left loaded in LM
        # Studio (another chat model, or Orpheus when the clone is in use)
        # pushes VRAM into system RAM and slows every reply.
        keep = {self.model} | ({tts.orpheus.model} if tts.engine == "orpheus" else set())
        for other in self.lm.loaded_models() - keep:
            print(f"{DIM}Unloading {other} from LM Studio to free VRAM.{RESET}", flush=True)
            self.lm.unload(other)
        # Always a fresh load, now the GPU is clear: an instance loaded while
        # other models held VRAM may be part-way on the CPU, invisibly.
        self.lm.ensure_loaded(self.model, fresh=True)
        if tts.engine == "orpheus":
            self.lm.ensure_loaded(tts.orpheus.model, tts.orpheus.context_length)
        face_server = None
        if self.cfg.face.enabled:
            # Its own process and venv; it warms up while the voice loads. Only
            # now, after the chat model: Ditto's VRAM, taken during that load,
            # would push part of the model onto the CPU (see ensure_loaded).
            print(f"{DIM}Starting Holly's face (log: face/data/server.log)...{RESET}", flush=True)
            face_server = start_server(self.cfg.face, audio.host_api, audio.output_device,
                                       self.cfg.listen.awake_s)
        try:
            await self._start_voice(face_server)
        except BaseException:
            if face_server:  # don't leave it holding the GPU
                face_server.terminate()
            raise

    async def _start_voice(self, face_server) -> None:
        tts = self.cfg.tts
        audio = self.cfg.audio
        self.tts = await asyncio.to_thread(make_tts, self.cfg)
        if tts.engine == "voxcpm" and tts.voxcpm.compile:
            print(f"{DIM}Warming up the voice (the first start after an install compiles for a few "
                  f"minutes)...{RESET}", flush=True)
        await self.tts.warm_up()
        async for _ in self.lm.astream_reply(self.convo.messages_for("Hello."), ReplyTiming(), self.model,
                                             self.tools):
            pass
        if self.searcher:
            for line in self.cfg.search.fillers:
                chunks = [a async for a in self.tts.stream(line, trim_lead=True)]
                if chunks:
                    self.fillers.append(np.concatenate(chunks))
        if face_server:
            print(f"{DIM}Waiting for the face to warm up...{RESET}", flush=True)
            self.player = await asyncio.to_thread(FacePlayer, self.tts.sample_rate, self.cfg.face.port, face_server)
        else:
            block = self.tts.sample_rate * audio.block_ms // 1000
            self.player = Player(self.tts.sample_rate, block, audio.host_api, audio.output_device)
        self.player.start()
        print(
            f"{DIM}Holly is on {self.model}, voice {self.voice_name()}, "
            f"playing through {self.player.device_name}.{self.HINT}{RESET}\n"
        )

    async def respond(self, user: str, times: TurnTimes | None = None) -> TurnTimes:
        """Answer `user` out loud. Pass `times` to count from an earlier start."""
        assert self.tts and self.player
        times = times or TurnTimes(start=time.perf_counter())
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        parts: list[str] = []
        steps: list[dict] = []  # his searches and their results, for the history
        last_said: list[str] = []
        thought = asyncio.Event()
        wait_for_reply = self.cfg.tts.engine == "voxcpm" and self.cfg.tts.voxcpm.wait_for_reply
        self.player.begin_turn()

        async def think() -> None:
            chunker = SentenceChunker()
            timing = ReplyTiming()
            print(f"{CYAN}Holly:{RESET} ", end="", flush=True)
            messages = self.convo.messages_for(user)
            asked = len(messages)
            searches = 0
            while True:
                calls: list[dict] = []
                said: list[str] = []
                async for delta in self.lm.astream_reply(messages, timing, self.model, self.tools, calls):
                    times.first_token = times.first_token or time.perf_counter()
                    parts.append(delta)
                    said.append(delta)
                    print(delta, end="", flush=True)
                    for sentence in chunker.feed(delta):
                        times.first_sentence = times.first_sentence or time.perf_counter()
                        await sentences.put(sentence)
                last_said[:] = said
                if not calls or searches > self.cfg.search.max_searches:
                    break  # (past the limit, he was told to stop; stop asking him)
                # He wants to look something up. Say anything he wrote first;
                # if he wrote nothing yet, a stock line covers the wait.
                for sentence in chunker.flush():
                    times.first_sentence = times.first_sentence or time.perf_counter()
                    await sentences.put(sentence)
                if not parts and any(c["name"] != "clock" for c in calls):  # the clock is instant
                    self._say_filler(times)
                self.player.expect_gap()
                messages = [*messages, {
                    "role": "assistant",
                    "content": "".join(said),
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]}}
                                   for c in calls],
                }]
                for call in calls:
                    searches += 1
                    result = await self._run_tool(call, user, searches, times)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                print(f"{CYAN}Holly:{RESET} ", end="", flush=True)
            steps.extend(messages[asked:])
            for sentence in chunker.flush():
                times.first_sentence = times.first_sentence or time.perf_counter()
                await sentences.put(sentence)
            await sentences.put(None)
            thought.set()
            print()

        async def speak() -> None:
            if wait_for_reply:
                # Sharing the GPU with the LLM slows VoxCPM ~8x; take turns.
                await thought.wait()
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
        self.convo.record(user, "".join(last_said).strip(), steps)
        print(f"{DIM}  {times.report()}{RESET}\n")
        return times

    def _say_filler(self, times: TurnTimes) -> None:
        """Play a stock line ("Hang on, let me look that up") in his voice."""
        if not self.fillers:
            return
        pick = random.choice([i for i in range(len(self.fillers)) if i != self.last_filler] or [0])
        self.last_filler = pick
        times.first_audio = times.first_audio or time.perf_counter()
        audio = self.fillers[pick]
        step = self.tts.sample_rate // 10  # 100 ms pieces, like the voice's own stream
        for i in range(0, len(audio), step):
            self.player.put(audio[i:i + step])

    async def _run_tool(self, call: dict, user: str, n: int, times: TurnTimes) -> str:
        """Run one tool call; returns what goes back to the model."""
        print()
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        if call["name"] == "clock":
            zone = str(args.get("timezone") or "").strip() or None
            result = clock.now(zone)
            print(f"{DIM}  (clock{': ' + zone if zone else ''}) {result.splitlines()[-1]}{RESET}", flush=True)
            # Given the right time, he still hedged it ("11:42 AM. Or 11:42 PM").
            return result + "\n(This is the exact time. Say it plainly, without doubting it.)"
        if n > self.cfg.search.max_searches:
            return "That's enough searching. Answer with what you have."
        if call["name"] == "search" and self.searcher:
            if not (query := str(args.get("query", "")).strip()):
                return "The search needs a query."
            label, job = f"searching: {query}", functools.partial(self.searcher.look_up, query, user)
        elif call["name"] == "weather" and self.searcher and self.cfg.search.weather:
            if not (place := str(args.get("place", "")).strip()):
                return "The weather tool needs a place."
            label, job = f"weather: {place}", functools.partial(weather.forecast, place, self.cfg.search)
        else:
            return f"There is no tool called {call['name']}."
        print(f"{DIM}  ({label}){RESET}", flush=True)
        found = await asyncio.to_thread(job)
        times.search_ms += found.search_ms + found.summary_ms
        again = f"; searched again: {'; '.join(found.queries[1:])}" if len(found.queries) > 1 else ""
        print(f"{DIM}  ({found.results} results{again}, {found.search_ms:.0f} + {found.summary_ms:.0f} ms: "
              f"{found.answer}){RESET}", flush=True)
        return self.searcher.for_holly(found)

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
    parser = argparse.ArgumentParser(description="Talk to Holly; he answers out loud")
    parser.add_argument("--model", help="LM Studio chat model key (default: holly.toml)")
    parser.add_argument("--engine", choices=["voxcpm", "orpheus"], help="TTS engine (default: holly.toml)")
    parser.add_argument("--voice", help="Orpheus preset voice, with --engine orpheus")
    parser.add_argument("--face", action=argparse.BooleanOptionalAction, help="show Holly's face (default: holly.toml)")
    parser.add_argument("--search", action=argparse.BooleanOptionalAction,
                        help="let Holly search the web (default: holly.toml)")
    args = parser.parse_args()
    # The Windows console defaults to cp1252, which can't print emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(Holly(args.model, args.engine, args.voice, args.face, args.search).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Milestone 2: type to Holly, and he answers out loud in a stock voice.

    python -m app.speak [--model MODEL] [--no-search] [--face] [--forget]

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
import re
import sys
import time

import numpy as np
import torch

from .config import load_config
from .lmstudio import LmStudio, ReplyTiming
from .persona import Conversation, set_aside, when
from .face_player import FacePlayer, start_server
from .player import Player
from . import archive, clock, memory, timers, weather
from .records import reminders as pick_reminders, source_of
from .search import TOOL, Searcher
from .speech_text import SentenceChunker
from .tts_common import Tts, make_tts

DIM, RESET, CYAN, YELLOW = "\033[2m", "\033[0m", "\033[36m", "\033[33m"
# Tools that answer at once, from this machine: no stock line, and they don't count
# towards the lookups allowed per reply.
LOCAL_TOOLS = ("clock", "timer")
COMPACT_RETRY_S = 60.0  # after his notes failed to be written, wait this long before trying again
# Said when a reply fails part-way (LM Studio restarted, a request timed out).
LOST_THREAD = "Sorry. I seem to have lost my train of thought."
# A time of day in a reply: "10:14", "10 pm", "ten o'clock". Not "12.50": that's money.
CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.)(?=\W|$)|"
                        r"\bo'clock\b", re.IGNORECASE)


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
    context_tokens: int | None = None  # the last request's prompt and reply, from LM Studio

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


def quiet_connection_resets(loop: asyncio.AbstractEventLoop) -> None:
    """Don't print asyncio's harmless report when LM Studio has closed an idle connection.

    The OpenAI client keeps its connection to LM Studio open between replies. When LM
    Studio has closed its end, Windows' asyncio, tidying the connection up, prints
    "Exception in callback _ProactorBasePipeTransport._call_connection_lost" with a
    ConnectionResetError (WinError 10054), though nothing failed: the next request opens a
    new connection. Only that case is dropped; everything else is reported as before."""
    default = loop.get_exception_handler()

    def handler(loop, context):
        if (isinstance(context.get("exception"), ConnectionResetError)
                and "_call_connection_lost" in repr(context.get("handle"))):
            return
        if default:
            default(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


class Holly:
    HINT =" /reset to start over, /quit to leave."

    def __init__(self, model: str | None, face: bool | None = None, search: bool | None = None,
                 forget: bool = False):
        cfg = load_config()
        if face is not None:
            cfg = dataclasses.replace(cfg, face=dataclasses.replace(cfg.face, enabled=face))
        if search is not None:
            cfg = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, enabled=search))
        if not torch.cuda.is_available():
            sys.exit("PyTorch can't see the GPU. Install a CUDA 12.8+ build (see README).")
        self.cfg = cfg
        self.model = model or cfg.llm.model
        self.lm = LmStudio(cfg.llm)
        # Every request sends the same tools: they're part of the start of the
        # prompt, so changing them between requests would break LM Studio's cache.
        self.searcher = Searcher(cfg.search, self.lm, self.model) if cfg.search.enabled else None
        memory_file = cfg.persona.memory_file
        if forget:
            # Set aside, not deleted, in case it was a mistake; each with its own
            # time, so a second --forget doesn't overwrite the first.
            asides = [set_aside(memory_file, "forgotten")] if memory_file and memory_file.exists() else []
            if cfg.recall.enabled and (a := archive.set_aside_archive(cfg.recall.file)):
                asides.append(a)
            if asides:
                print(f"{DIM}Holly's memory of earlier sessions is set aside "
                      f"({', '.join(a.name for a in asides)}); he starts afresh.{RESET}")
        # The ship's records (app/records.py), if they've been built: searched by recall, and a
        # few, fitting what's been said, put in his prompt while he sleeps to bring up now and then.
        self.records = (archive.Archive(cfg.recall, self.lm, self.model, cfg.recall.records_file)
                        if cfg.recall.enabled and cfg.recall.records_file.exists() else None)
        self.reminded = False  # the reminders fit the talk so far
        self.archive = (archive.Archive(cfg.recall, self.lm, self.model, records=self.records)
                        if cfg.recall.enabled else None)
        tools, prompts = [], []
        if self.searcher:
            tools += [TOOL, clock.TOOL] + ([weather.TOOL] if cfg.search.weather else [])
            prompts.append(cfg.search.prompt.read_text(encoding="utf-8"))
        if self.archive:
            tools.append(archive.TOOL)
            prompts.append(cfg.recall.prompt.read_text(encoding="utf-8"))
        self.timers = timers.Timers(cfg.timers.file) if cfg.timers.enabled else None
        if self.timers:
            tools.append(timers.TOOL)
            prompts.append(cfg.timers.prompt.read_text(encoding="utf-8"))
        self.tools = tools or None
        self.turn_lock = asyncio.Lock()  # one reply at a time: yours, or his speaking up
        self.convo = Conversation(cfg.persona, "\n\n".join(prompts), memory_file, self.archive)
        # Stock lines in his voice for while a tool works, by kind: "search" (web
        # search and weather) and "recall" (his archive).
        self.fillers: dict[str, list[np.ndarray]] = {"search": [], "recall": []}
        self.last_filler = -1
        self.tts: Tts | None = None
        self.player: Player | FacePlayer | None = None
        self.writing_notes = False
        self.resuming: asyncio.Task | None = None  # at startup: the last session into his notes
        self.compaction: asyncio.Task | None = None  # later: older history into his notes
        self.compact_retry_at = 0.0  # after a failed try, not before this

    def voice_name(self) -> str:
        return "Holly (VoxCPM2 clone)"

    async def start(self) -> None:
        audio = self.cfg.audio
        quiet_connection_resets(asyncio.get_running_loop())
        print(f"{DIM}Loading {self.model} and the {self.voice_name()} voice...{RESET}", flush=True)
        # Holly needs nearly all of the GPU. Anything else left loaded in LM
        # Studio pushes VRAM into system RAM and slows every reply.
        for other in self.lm.loaded_models() - {self.model}:
            print(f"{DIM}Unloading {other} from LM Studio to free VRAM.{RESET}", flush=True)
            self.lm.unload(other)
        # Always a fresh load, now the GPU is clear: an instance loaded while
        # other models held VRAM may be part-way on the CPU, invisibly.
        self.lm.ensure_loaded(self.model, fresh=True)
        if self.archive and (embedder := self.cfg.recall.embed_model):
            # Now, before the face takes its VRAM, rather than when he first recalls.
            try:
                self.lm.ensure_loaded(embedder, context_length=2048)
            except Exception as e:  # not downloaded: recall still works by keyword and day
                print(f"{YELLOW}Couldn't load {embedder} for recall by meaning ({e!r}); "
                      f"recall uses keywords and days only.{RESET}", flush=True)
        # Last session's history goes into his notes while the voice loads.
        if self.convo.previous_end:
            print(f"{DIM}He remembers talking until {when(self.convo.previous_end)}"
                  f"{', and has notes' if self.convo.memory else ''}.{RESET}", flush=True)
            if end := self.convo.resume_split():
                self.resuming = asyncio.create_task(self.compact(end, "from last time"))
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
            if self.resuming:  # nor the notes half-written
                self.resuming.cancel()
            raise

    async def _start_voice(self, face_server) -> None:
        tts = self.cfg.tts
        audio = self.cfg.audio
        self.tts = await asyncio.to_thread(make_tts, self.cfg)
        if tts.voxcpm.compile:
            print(f"{DIM}Warming up the voice (the first start after an install compiles for a few "
                  f"minutes)...{RESET}", flush=True)
        await self.tts.warm_up()
        if self.resuming:
            try:
                await self.resuming
            except Exception as e:  # he starts with the history as it was
                print(f"{DIM}  (couldn't write his notes: {e!r}){RESET}", flush=True)
        if self.records:
            try:
                await self.refresh_reminders(prewarm=False)  # the warm-up below reads them
            except Exception as e:
                print(f"{DIM}  (couldn't pick reminders from the ship's records: {e!r}){RESET}", flush=True)
        async for _ in self.lm.astream_reply(self.convo.messages_for("Hello."), ReplyTiming(), self.model,
                                             self.tools):
            pass
        lines = {"search": self.cfg.search.fillers if self.searcher else (),
                 "recall": self.cfg.recall.fillers if self.archive else ()}
        for kind, said in lines.items():
            for line in said:
                chunks = [a async for a in self.tts.stream(line, trim_lead=True)]
                if chunks:
                    self.fillers[kind].append(np.concatenate(chunks))
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
        wait_for_reply = self.cfg.tts.voxcpm.wait_for_reply
        self.player.begin_turn()

        failed: list[Exception] = []

        async def think() -> None:
            """The reply, from LM Studio and his tools. If either fails (LM Studio
            restarted, a request timed out), he says so in his own way and the
            turn ends there, rather than the whole program."""
            try:
                await think_through()
            except Exception as e:
                failed.append(e)
                print(f"\n{YELLOW}  (the reply failed: {e!r}){RESET}", flush=True)
                await sentences.put(LOST_THREAD)
            finally:
                await sentences.put(None)
                thought.set()

        async def think_through() -> None:
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
                slow = [c["name"] for c in calls if c["name"] not in LOCAL_TOOLS]  # these are instant
                if not parts and slow:
                    self._say_filler(times, "recall" if all(n == "recall" for n in slow) else "search")
                self.player.expect_gap()
                messages = [*messages, {
                    "role": "assistant",
                    "content": "".join(said),
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]}}
                                   for c in calls],
                }]
                for call in calls:
                    if call["name"] not in LOCAL_TOOLS:  # instant and local: no limit
                        searches += 1
                    result = await self._run_tool(call, user, searches, times)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                print(f"{CYAN}Holly:{RESET} ", end="", flush=True)
            steps.extend(messages[asked:])
            if timing.prompt_tokens is not None:
                times.context_tokens = timing.prompt_tokens + (timing.completion_tokens or 0)
            for sentence in chunker.flush():
                times.first_sentence = times.first_sentence or time.perf_counter()
                await sentences.put(sentence)
            print()

        async def speak() -> None:
            try:
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
            finally:
                self.player.end_turn()

        await asyncio.gather(think(), speak())
        await self.player.wait_done()
        times.first_played = self.player.first_played
        times.first_voice = self.player.first_voice
        times.underruns = self.player.underruns
        if failed:  # not kept: half a turn would confuse the next one
            print()
            return times
        said = "".join(last_said).strip()
        if CLOCK_TIME.search(said) and not any(c["function"]["name"] in LOCAL_TOOLS for m in steps
                                                for c in (m.get("tool_calls") or [])):
            # He gave a time without the clock: a guess. Kept as fact in his history, it
            # stuck: greeted, he said "10:14 PM" at 7:56, and repeated it four times,
            # asked to use the clock or not. (The brackets are never spoken.)
            said += " [That time was a guess, not from the clock: it's probably wrong.]"
        dropped = self.convo.record(user, said, steps)
        print(f"{DIM}  {times.report()}{RESET}")
        if dropped:
            print(f"{DIM}  (history past {self.cfg.persona.history_hard_tokens:,} tokens: dropped the "
                  f"oldest {dropped} exchanges, without notes){RESET}")
        print(f"{DIM}  {self.context_report(times.context_tokens)}{RESET}\n")
        return times

    def context_report(self, used: int | None) -> str:
        """How full his context is, and how far his history is from being
        folded into notes (see compact)."""
        size = self.cfg.llm.context_length
        soft = self.cfg.persona.history_soft_tokens
        history = self.convo.tokens()
        # `used` is LM Studio's count; the history's is characters / 4, which runs ~10% high.
        report = f"context {used:,} of {size:,} tokens" if used else f"context ? of {size:,} tokens"
        if history > soft:
            return report + "; older history goes into his notes when he next sleeps"
        return report + f"; ~{soft - history:,} more before older history goes into his notes"

    async def compact(self, end: int | None = None, label: str = "asleep") -> bool:
        """Fold the older history, history[:end], into his notes
        (app/memory.py), then have LM Studio read the new, shorter prompt, so
        his next reply doesn't wait for it. Meant for while he sleeps, and
        for startup; `self.writing_notes` is True while the notes are being
        written, the part that's safe to cancel. Returns True if the history
        was compacted."""
        convo = self.convo
        if not (end := end or convo.compaction_split()):
            return False
        dropped = convo.history[:end]
        exchanges = sum(m["role"] == "user" for m in dropped)
        before = convo.tokens()
        t0 = time.perf_counter()
        self.writing_notes = True
        try:
            notes = await memory.write_notes(self.lm, self.model, convo.memory, convo.transcript(end),
                                             self.cfg.persona.memory_tokens)
        finally:
            self.writing_notes = False
        if not notes or not convo.apply_compaction(end, dropped, notes):
            return False
        self.reminded = False  # the talk has moved on: reminders to fit it
        t1 = time.perf_counter()
        prompt = await self.lm.prewarm(convo.messages_for(), self.model, self.tools)
        if self.archive:  # the archived exchanges' vectors, now rather than at his next recall
            try:
                await asyncio.to_thread(self.archive.embed_missing)
            except Exception as e:
                print(f"{DIM}  (couldn't embed the archive: {e!r}){RESET}", flush=True)
        print(f"{DIM}  ({label}: folded {exchanges} exchanges into his notes, ~{len(notes) // 4} tokens; "
              f"history ~{before:,} -> ~{convo.tokens():,} tokens; {t1 - t0:.1f} s to write, "
              f"{time.perf_counter() - t1:.1f} s to reread)\n  {self.context_report(prompt)}{RESET}",
              flush=True)
        return True

    def _say_filler(self, times: TurnTimes, kind: str = "search") -> None:
        """Play a stock line ("Hang on, let me look that up") in his voice."""
        if not (fillers := self.fillers[kind]):
            return
        pick = random.choice([i for i in range(len(fillers)) if i != self.last_filler] or [0])
        self.last_filler = pick
        times.first_audio = times.first_audio or time.perf_counter()
        audio = fillers[pick]
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
        if call["name"] == "timer" and self.timers:
            result = self.timers.handle(args)
            print(f"{DIM}  (timer) {result.split('. Tell the crew')[0]}{RESET}", flush=True)
            return result
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
        elif call["name"] == "recall" and self.archive:
            if not (query := str(args.get("query", "")).strip()):
                return "Recall needs a query."
            label, job = f"recalling: {query}", functools.partial(self.archive.recall, query, user)
        else:
            return f"There is no tool called {call['name']}."
        print(f"{DIM}  ({label}){RESET}", flush=True)
        try:
            found = await asyncio.to_thread(job)
        except Exception as e:  # LM Studio, a provider or the weather service broke: he says he couldn't
            print(f"{YELLOW}  ({call['name']} failed: {e!r}){RESET}", flush=True)
            return (f"The {call['name']} tool broke ({type(e).__name__}). Tell the crew you couldn't find "
                    "out, in your own words, and give no time, number, date or name for it.")
        times.search_ms += found.search_ms + found.summary_ms
        again = f"; searched again: {'; '.join(found.queries[1:])}" if len(found.queries) > 1 else ""
        print(f"{DIM}  ({found.results} results{again}, {found.search_ms:.0f} + {found.summary_ms:.0f} ms: "
              f"{found.answer}){RESET}", flush=True)
        return Searcher.for_holly(found)

    def start_compaction(self) -> None:
        """In the background, if due: fold older history into his notes (see
        compact), and pick fresh reminders from the ship's records. Cancel
        `self.compaction` while `self.writing_notes` to leave the history as
        it was."""
        due = self.convo.needs_compacting() or (self.records and not self.reminded)
        if self.compaction or time.perf_counter() < self.compact_retry_at or not due:
            return
        self.compaction = asyncio.create_task(self.sleep_work())
        self.compaction.add_done_callback(self._compacted)

    async def sleep_work(self) -> bool:
        """What's done while he sleeps. False if the notes couldn't be written."""
        ok = True
        if self.convo.needs_compacting():
            ok = await self.compact()
        if self.records and not self.reminded:
            await self.refresh_reminders()
        return ok

    async def refresh_reminders(self, prewarm: bool = True) -> None:
        """A few entries from the ship's records for his system prompt, close to
        what's been said lately (app/records.py); then, if they changed, LM
        Studio rereads the prompt, as after his notes."""
        picks = await asyncio.to_thread(pick_reminders, self.records, self.convo.recent_topics(),
                                        self.cfg.recall.reminders)
        self.reminded = True
        if picks == self.convo.reminders:
            return
        self.convo.reminders = picks
        names = "; ".join(source_of(p) for p in picks)
        print(f"{DIM}  (from the ship's records, on his mind: {names}){RESET}", flush=True)
        if prewarm:
            await self.lm.prewarm(self.convo.messages_for(), self.model, self.tools)

    def _compacted(self, task: asyncio.Task) -> None:
        self.compaction = None
        if task.cancelled():
            return  # someone spoke: tried again later, no wait
        if (e := task.exception()) or not task.result():
            # LM Studio refused, or the notes came back empty: not again for a
            # while, rather than once per mic block.
            self.compact_retry_at = time.perf_counter() + COMPACT_RETRY_S
            print(f"{DIM}  (couldn't write his notes{f': {e!r}' if e else ''}; trying again in "
                  f"{COMPACT_RETRY_S:.0f} s){RESET}", flush=True)

    async def announce(self, due: list[dict]) -> TurnTimes | None:
        """Speak up about timers and reminders that have come due: his reply to a
        note saying so, which is kept in his history like any turn."""
        print(f"{DIM}  (due: {', '.join(i['label'] for i in due)}){RESET}", flush=True)
        self.player.wake()
        try:
            return await self.respond(timers.announcement(due))
        except Exception as e:
            print(f"{YELLOW}  (couldn't announce it: {e!r}){RESET}\n", flush=True)
            return None

    async def _watch_timers(self) -> None:
        """Typed: there's no listening loop to check between, so a task checks
        every second and speaks up between your turns."""
        while True:
            await asyncio.sleep(1)
            if self.timers and (due := self.timers.due()):
                async with self.turn_lock:
                    await self.announce(due)
                print("You: ", end="", flush=True)

    async def run(self) -> None:
        await self.start()
        watcher = asyncio.create_task(self._watch_timers())
        try:
            while True:
                # Typed, there's no sleep to wait for: his notes are written while
                # you type, and dropped if you send before they're done.
                self.start_compaction()
                user = (await asyncio.to_thread(input, "You: ")).strip()
                if self.compaction and self.writing_notes:
                    self.compaction.cancel()
                if not user:
                    continue
                if user in ("/quit", "/exit"):
                    return
                if user == "/reset":
                    self.convo.reset()
                    print(f"{DIM}Conversation and notes cleared, on disk too.{RESET}\n")
                    continue
                try:
                    async with self.turn_lock:
                        await self.respond(user)
                except Exception as e:  # the voice or the speakers failed: report it and carry on
                    print(f"{YELLOW}  (that reply failed: {e!r}){RESET}\n", flush=True)
        except (EOFError, KeyboardInterrupt):
            print()
        finally:
            watcher.cancel()
            if self.player:
                self.player.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Holly; he answers out loud")
    parser.add_argument("--model", help="LM Studio chat model key (default: holly.toml)")
    parser.add_argument("--face", action=argparse.BooleanOptionalAction, help="show Holly's face (default: holly.toml)")
    parser.add_argument("--search", action=argparse.BooleanOptionalAction,
                        help="let Holly search the web (default: holly.toml)")
    parser.add_argument("--forget", action="store_true",
                        help="start afresh, setting aside his memory of earlier sessions")
    args = parser.parse_args()
    # The Windows console defaults to cp1252, which can't print emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(Holly(args.model, args.face, args.search, args.forget).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

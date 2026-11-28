"""Milestone 2: type to Holly, and he answers out loud in a stock voice.

    python -m app.speak [--model MODEL] [--no-search] [--face] [--forget]

The reply streams: the LLM's tokens are cut into sentences, each sentence
goes to TTS as soon as it's complete, and audio plays as it's decoded. So
Holly starts talking while the LLM is still writing.

Holly can search the web (app/search.py). When his reply starts with a
search instead of words, he says a stock line ("Hang on, let me look that
up"), rendered at startup so it plays at once, while the search runs.

What he's sent, and the side checks that decide it, is built in app/context.py;
this is the reply loop around it. His tools are in app/tools.py and beside
their code; the safety nets that make him redo a reply are in app/guards.py;
what he does while he sleeps is in app/background.py; speaking up by himself,
in app/announce.py; every turn, written down, in app/events.py; --forget and
--reset, in app/yours.py.

Commands: /reset clears the conversation, /quit leaves.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import random
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
import torch

from . import archive, context, cpu, guards, news, research, spotify, status, timers, yours
from .announce import Announcement, pending
from .background import Background, context_report
from .clock import ClockTool
from .config import load_config
from .console import CYAN, DIM, RESET, YELLOW, note, stamp
from .events import EventLog
from .face_player import FacePlayer, start_server
from .lmstudio import LmStudio, ReplyTiming
from .nets import NetCounts
from .persona import Conversation, about_the_crew, when
from .player import Player
from .search import SearchTool, Searcher
from .speech_text import SentenceChunker, starts_section
from .tools import Toolbox, Turn
from .tts_common import Tts, make_tts
from .news import NewsTool
from .weather import WeatherTool

__all__ = ["CYAN", "DIM", "RESET", "YELLOW", "stamp", "Holly", "TurnTimes"]  # app/listen.py uses these

# Said when a reply fails part-way (LM Studio restarted, a request timed out).
LOST_THREAD = "Sorry. I seem to have lost my train of thought."
SECTION_PAUSE_S = 0.6  # before a status report's next section


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
    failed: bool = False  # the reply broke off (LM Studio restarted, a timeout): not kept

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
    HINT = " /reset to start over, /quit to leave."

    def __init__(self, model: str | None, face: bool | None = None, search: bool | None = None,
                 forget: bool = False, reset: bool = False, cfg=None):
        """`cfg`, if given, instead of holly.toml's (app/scenarios.py: his memory elsewhere)."""
        cfg = cfg or load_config()
        # The model's own settings and quirks ([models."key"] in holly.toml).
        cfg = dataclasses.replace(cfg, llm=cfg.llm.for_model(model))
        if face is not None:
            cfg = dataclasses.replace(cfg, face=dataclasses.replace(cfg.face, enabled=face))
        if search is not None:
            cfg = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, enabled=search))
        if not torch.cuda.is_available():
            sys.exit("PyTorch can't see the GPU. Install a CUDA 12.8+ build (see README).")
        self.cfg = cfg
        self.model = cfg.llm.model
        self.stray_words = {w.lower() for w in cfg.llm.stray_words}
        self.lm = LmStudio(cfg.llm)
        self.searcher = Searcher(cfg.search, self.lm, self.model) if cfg.search.enabled else None
        memory_file = cfg.persona.memory_file
        yours.start_afresh(cfg, forget, reset)
        # The ship's records (app/records.py), if they've been built: searched by recall, and a
        # few, fitting what's been said, put in his prompt while he sleeps to bring up now and then.
        self.records = (archive.Archive(cfg.recall, self.lm, self.model, cfg.recall.records_file, talk=False)
                        if cfg.recall.enabled and cfg.recall.records_file.exists() else None)
        self.archive = (archive.Archive(cfg.recall, self.lm, self.model, records=self.records)
                        if cfg.recall.enabled else None)
        self.timers = timers.Timers(cfg.timers.file) if cfg.timers.enabled else None
        self.spotify = (spotify.Spotify(cfg.spotify)
                        if cfg.spotify.enabled and spotify.available() else None)
        # Research jobs (app/research.py): queued by the tool, worked on while he sleeps.
        # Finished reports are found by meaning too, with recall's embedding model, if it's on.
        self.research = (research.Research(cfg.research, cfg.search, self.lm, self.model,
                                           cfg.recall.embed_model if cfg.recall.enabled else "")
                         if cfg.research.enabled and self.searcher else None)
        self.research_tool = research.ResearchTool(self.research) if self.research else None
        tools = []
        if self.searcher:
            tools += [SearchTool(self.searcher), ClockTool()] + ([WeatherTool(cfg.search)] if cfg.search.weather else [])
            tools += [NewsTool(cfg.search, self.searcher)] if cfg.search.news else []
        if self.archive:
            tools.append(archive.RecallTool(self.archive, self.research))
        if self.timers:
            tools.append(timers.TimerTool(self.timers))
        if self.spotify:
            tools.append(spotify.SpotifyTool(self.spotify))
        if self.research_tool:
            tools.append(self.research_tool)
        if cfg.status.enabled:  # with the ship's records, if built, to write from
            tools.append(status.StatusTool(self.lm, self.model, self.records, cfg.status.file))
        self.toolbox = Toolbox(tools, cfg.search.max_searches)
        # Every request sends the same tools: they're part of the start of the
        # prompt, so changing them between requests would break LM Studio's cache.
        self.tools = self.toolbox.schemas()
        self.turn_lock = asyncio.Lock()  # one reply at a time: yours, or his speaking up
        self.convo = Conversation(cfg.persona, self.toolbox.rules(), memory_file, self.archive)
        if "status" in self.toolbox:  # a report given before a restart: what's asked of the crew carries on from it
            self.toolbox.by_name["status"].given = status.last_given(self.convo.history)
        self.background = Background(cfg, self.convo, self.lm, self.model, self.tools, self.archive, research=self.research)
        # Stock lines in his voice for while a tool works, by kind: "search" (web
        # search and weather), "recall" (his archive) and "spotify" (its app opening).
        self.fillers: dict[str, list[np.ndarray]] = {"search": [], "recall": [], "spotify": [], "status": [],
                                                     "crew": []}
        self.last_filler = -1
        self.tts: Tts | None = None
        self.player: Player | FacePlayer | None = None
        self.resuming: asyncio.Task | None = None  # at startup: the last session into his notes
        self.turns = 0  # replies so far this session
        # Memories offered him unasked (context.memory_for): each archive entry once a session, and
        # when the last was; this turn's, for the evals and the console.
        self.offered: set[int] = set()
        self.offered_turn = -10**6
        self.memory = ""
        # The nets on his replies (guards.NETS), each as it fires: they make up for a model's
        # habits, and the evals show which a model still trips. With nets_act off (evals
        # --raw) they're counted, and do nothing: the model as it is.
        self.nets: Counter[str] = Counter()
        self.nets_act = True
        # ...and kept between sessions, per model, beside his memory (app/nets.py): what real
        # talk trips, not only the evals'.
        self.net_counts = NetCounts(memory_file.with_name("nets.json") if memory_file else None)
        # Empty replies after a tool, as the model wrote them (guards.EMPTY), to see what they are.
        self.empty_log = memory_file.with_name("empty_replies.jsonl") if memory_file else None
        # Every turn as it happens, beside his memory (app/events.py), to replay (app/replay.py).
        self.events = EventLog(yours.events_folder(memory_file))
        # Replaying a turn: its side checks' answers, by name, given rather than asked
        # (_decide), so his request is the one he was sent. None: asked.
        self.replaying: dict | None = None
        self._turn: dict = {}  # this turn's event, filled as it goes

    async def _decide(self, name: str, fn, *args):
        """A side check's answer (fn(*args), in a worker thread), kept with the turn; or,
        replaying, the one it gave then."""
        if self.replaying is not None and name in self.replaying:
            got = self.replaying[name]
        else:
            got = await asyncio.to_thread(fn, *args)
        self._turn.setdefault("decisions", {})[name] = got
        return got

    def _log_empty(self, user: str, called: set[str], raw: str, timing) -> None:
        """An empty reply after a tool, as the model wrote it, and any reasoning LM Studio sent
        with it: on the console, and appended to empty_replies.jsonl beside his nets.json."""
        print(f"\n{DIM}  (empty after {', '.join(sorted(called))}: wrote {raw!r}; reasoning "
              f"{timing.reasoning_chars} characters){RESET}", flush=True)
        if not self.empty_log:
            return
        try:
            with self.empty_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(at=datetime.datetime.now().isoformat(timespec="seconds"), model=self.model,
                                        said=user, tools=sorted(called), raw=raw,
                                        reasoning=timing.reasoning[-500:]), ensure_ascii=False) + "\n")
        except OSError:
            pass  # a log only: never in the way of his reply

    def fired(self, net: str) -> bool:
        """Count a net firing; True if it's to act."""
        self.nets[net] += 1
        return self.nets_act

    def voice_name(self) -> str:
        return "Holly (VoxCPM2 clone)"

    async def start(self) -> None:
        # [cpu] startup = "efficiency": loaded and warmed up on the efficiency cores, a
        # stopgap for an unstable CPU (app/cpu.py); then on them all.
        if self.cfg.cpu.startup != "all":
            print(f"{DIM}Starting up on the {self.cfg.cpu.startup} cores ([cpu] in holly.toml).{RESET}", flush=True)
        with cpu.starting_on(self.cfg.cpu.startup):
            await self._start()

    async def _start(self) -> None:
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
                self.resuming = asyncio.create_task(self.background.compact(end, "from last time"))
        face_server = None
        if self.cfg.face.enabled:
            # Its own process and venv; it warms up while the voice loads. Only
            # now, after the chat model: Ditto's VRAM, taken during that load,
            # would push part of the model onto the CPU (see ensure_loaded).
            print(f"{DIM}Starting Holly's face (log: face/data/server.log)...{RESET}", flush=True)
            face_server = start_server(self.cfg.face, audio.host_api, audio.output_device,
                                       self.cfg.listen.awake_s)
            cpu.free(face_server)  # not held to Holly's start-up cores
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
        if self.archive:  # his archive's vectors, loaded while the voice warms up (archive.warm)
            self._warming = asyncio.create_task(asyncio.to_thread(self._warm_archive))
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
        async for _ in self.lm.astream_reply(self.convo.messages_for("Hello."), ReplyTiming(), self.model,
                                             self.tools):
            pass
        lines = {"search": self.cfg.search.fillers if self.searcher else (),
                 "recall": self.cfg.recall.fillers if self.archive else (),
                 "spotify": self.cfg.spotify.fillers if self.spotify else (),
                 "status": self.cfg.status.fillers if self.cfg.status.enabled else (),
                 "crew": self.cfg.status.crew_fillers if self.cfg.status.enabled else ()}
        for kind, said in lines.items():
            for line in said:
                chunks = [a async for a in self.tts.stream(line, trim_lead=True)]
                if chunks:
                    self.fillers[kind].append(np.concatenate(chunks))
        if face_server:
            print(f"{DIM}Waiting for the face to warm up...{RESET}", flush=True)
        block = self.tts.sample_rate * audio.block_ms // 1000
        if face_server:
            # With its own speakers to fall back on, should the face window be closed.
            self.player = await asyncio.to_thread(FacePlayer, self.tts.sample_rate, self.cfg.face.port, face_server,
                                                  audio.host_api, audio.output_device, block)
        else:
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
        self.turns += 1
        self.events.before_turn(self.convo)
        self._turn = dict(turn=self.turns, at=datetime.datetime.now().isoformat(timespec="seconds"),
                          model=self.model, nets_act=self.nets_act, said=user, replies=[], tools=[])
        nets_before = Counter(self.nets)  # this turn's, for the counts kept (app/nets.py)
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        parts: list[str] = []
        steps: list[dict] = []  # his searches and their results, for the history
        last_said: list[str] = []
        spoken: list[str] = []  # the sentences said this turn
        aside: list[str] = []  # the crew, tacked on unasked: the rest isn't said (say)
        crew_before = self.convo.last_about_the_crew()  # "what's he doing there?", after "where's Kryten?"
        called: set[str] = set()  # the tools called this reply (and seen by speak(): news means sections)
        unasked_countdown = not guards.TIME_ASKED.search(user)  # a timer counted down, if he does, is in passing
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
            # Before anything's asked of it: his model, still as he loaded it (see keep_loaded).
            if await self.lm.keep_loaded(self.model):
                note("LM Studio had loaded his model again with its own settings (its engine crashed?): "
                     "loaded again with his. Restart Holly when you can: loaded beside the voice, "
                     "it may be slower", YELLOW)
            chunker = SentenceChunker()
            timing = ReplyTiming()
            print(f"{stamp()}{CYAN}Holly:{RESET} ", end="", flush=True)
            req = await context.prepare(self, user)
            messages, asked, choice, required, self.memory = (req.messages, req.asked, req.choice, req.required,
                                                              req.memory)
            lookups = 0  # slow tools called this reply
            turn = Turn(user, self.turns, times, lambda kind: self._say_filler(times, kind),
                        req.held or bool(req.taken_up), self.convo.crew_lines, lambda: self.convo.memory)
            retried = False  # a reply was dropped part-way and asked again (once a turn)

            def told() -> list[str]:
                """What he's been given this turn: the crew's line, a memory offered, and his
                tools' answers."""
                return ([user] + ([self.memory] if self.memory else [])
                        + [m.get("content") or "" for m in messages[asked:] if m["role"] == "tool"])

            def crew_unasked() -> bool:
                """Nobody asked about the crew of Red Dwarf, nor just before, and he's giving no
                report or memory of them (persona.about_the_crew)."""
                return not (crew_before or about_the_crew(user, called))

            async def say(sentence: str) -> None:
                # A chunk can hold more than one sentence (a word alone goes with the next:
                # app/speech_text.py), so each is looked at in turn; any after something said:
                kept: list[str] = []
                for part in guards.SENTENCE_END.split(sentence):
                    after = bool(spoken or kept)
                    rest = bool(aside)  # the rest of a crew aside: cut with it, counted once
                    if rest:
                        net = guards.ASIDE_CUT
                    elif after and guards.OFFER.search(part):
                        # "Anything else you need me to dig up?": not said, nor kept (below).
                        net = guards.OFFERED
                    elif after and unasked_countdown and guards.guessed_time(part, told()):
                        # "It's 22:15", unasked and unchecked: a minute on from his last, each reply.
                        net = guards.GUESS_CUT
                    elif after and unasked_countdown and guards.COUNTDOWN.search(part):
                        # "Six minutes left", unasked: not said, nor kept (below). It came every
                        # reply, a minute less each time, whatever the time.
                        net = guards.COUNT_CUT
                    elif after and crew_unasked() and guards.CREW_ASIDE.search(part):
                        # "The Cat is currently refusing to leave his wardrobe", after a gift list:
                        # not said, nor kept (below); nor the rest, "He's convinced the hangers
                        # are judging him" (guards.CREW_ASIDE).
                        aside.append(part)
                        net = guards.ASIDE_CUT
                    else:
                        net = None
                    if net and (self.nets_act if rest else self.fired(net)):
                        print(f"{DIM}[{net}]{RESET}", end="", flush=True)
                    else:
                        kept.append(part)
                if not kept:
                    return
                sentence = " ".join(kept)
                spoken.append(sentence)
                times.first_sentence = times.first_sentence or time.perf_counter()
                await sentences.put(sentence)

            def dropped(sentence: str) -> str | None:
                """Why this sentence mustn't be said, if it mustn't (app/guards.py); counted,
                and nothing if the nets don't act."""
                if retried:
                    return None
                why = guards.unsaid(sentence, user, stray=self.stray_words,
                                    researched="research" in called,
                                    can_research="research" in self.toolbox,
                                    answering_choice=turn.answering_choice, clocked="clock" in called,
                                    can_clock="clock" in self.toolbox,
                                    counting=req.counting and "timer" not in called,
                                    told=told())
                return why if why and self.fired(why) else None

            while True:
                calls: list[dict] = []
                said: list[str] = []
                redo = None
                self.events.request(self.turns, messages, choice)
                async for delta in self.lm.astream_reply(messages, timing, self.model, self.tools, calls, choice):
                    if redo:
                        # Let it run out, unsaid: closing LM Studio's stream part-way left
                        # httpx complaining ("async generator already executing").
                        continue
                    times.first_token = times.first_token or time.perf_counter()
                    parts.append(delta)
                    said.append(delta)
                    print(delta, end="", flush=True)
                    for sentence in chunker.feed(delta):
                        if redo := dropped(sentence):
                            break
                        await say(sentence)
                wrote = dict(raw="".join(said), calls=[dict(name=c["name"], arguments=c["arguments"]) for c in calls])
                self._turn["replies"].append(wrote)
                if not wrote["raw"] and not calls:  # what it thought instead, if anything
                    wrote["reasoning"] = timing.reasoning[-500:]
                if redo:
                    calls.clear()  # whatever it went on to ask for, it's asked again
                if not redo and not calls:  # the end of the reply: its last words
                    last = chunker.flush()
                    lone = not retried and not spoken and len(last) == 1 and len(last[0].split()) == 1
                    if not retried and not spoken and not last and called:
                        # Nothing to say after a tool's answer (guards.EMPTY): logged as written,
                        # then asked again, once.
                        self._log_empty(user, called, "".join(said), timing)
                        if self.fired(guards.EMPTY):
                            redo = guards.EMPTY
                            messages = context.with_note(messages, asked, self.convo.notes.empty)
                    for sentence in last:
                        if redo := dropped(sentence):
                            break
                        if lone and self.fired(guards.ALONE):
                            # A word alone, his whole reply ("Yeah?", to "Hey Holly"): the voice garbled
                            # it. Asked again, for a few words; a word again, it's said.
                            redo = guards.ALONE
                            messages = context.with_note(messages, asked, self.convo.notes.alone)
                            break
                        await say(sentence)
                wrote["redo"] = redo
                if redo:
                    # Not said: asked again, with a tool required if he claimed one.
                    print(f"\n{DIM}  ({redo}: not said; asking again){RESET}", flush=True)
                    retried, chunker = True, SentenceChunker()
                    aside.clear()
                    choice = required if guards.needs_tool(redo) else None
                    print(f"{stamp()}{CYAN}Holly:{RESET} ", end="", flush=True)
                    continue
                choice = None
                last_said[:] = said
                if not calls or lookups > self.cfg.search.max_searches:
                    break  # (past the limit, he was told to stop; stop asking him)
                # He wants to look something up. Say anything he wrote first;
                # if he wrote nothing yet, a stock line covers the wait.
                for sentence in chunker.flush():
                    await say(sentence)
                if not parts and (kind := self.toolbox.filler(calls)):
                    self._say_filler(times, kind)
                self.player.expect_gap()
                messages = [*messages, {
                    "role": "assistant",
                    "content": "".join(said),
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]}}
                                   for c in calls],
                }]
                for call in calls:
                    if not self.toolbox.local(call["name"]):  # instant and local: no limit
                        lookups += 1
                    called.add(call["name"])
                    result = await self.toolbox.run(call, turn, lookups)
                    self._turn["tools"].append(dict(name=call["name"], arguments=call["arguments"], result=result))
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                    if result.startswith(status.REPORTS):  # a new report: not the last one again
                        messages = status.without_earlier_reports(messages, asked - 1)
                aside.clear()  # what he says with his tools' answers is said
                if report := next((status.report_of(m["content"]) for m in messages[-len(calls):]
                                   if m["role"] == "tool" and m["content"].startswith(status.SAID)), ""):
                    # His status report (or his answer about one of the crew), said as written. Told
                    # to give it as written, he retold it
                    # shorter, his own way, and the jokes came out mangled ("Tried his screwdriver. It
                    # gave him a disciplinary hearing."). It's in his voice already: his history has it
                    # as his reply, for anything asked about it after.
                    print(f"{stamp()}{CYAN}Holly:{RESET} {report}", flush=True)
                    parts.append(report)
                    for sentence in chunker.feed(report + "\n") + chunker.flush():
                        await say(sentence)
                    last_said[:] = [report]
                    break
                print(f"{stamp()}{CYAN}Holly:{RESET} ", end="", flush=True)
            steps.extend(messages[asked:])
            if timing.prompt_tokens is not None:
                times.context_tokens = timing.prompt_tokens + (timing.completion_tokens or 0)
            for sentence in chunker.flush():
                await say(sentence)
            print()

        async def speak() -> None:
            try:
                if wait_for_reply:
                    # Sharing the GPU with the LLM slows VoxCPM ~8x; take turns.
                    await thought.wait()
                first = True
                while (sentence := await sentences.get()) is not None:
                    if not first and called & {"news", "status"} and starts_section(sentence):
                        # A report's next section ("Politics — ...", "Rimmer — ..."): a longer pause
                        # before it, as between items on the news. Punctuation couldn't
                        # give one: a dash, two, or a full stop and a dash, ~0.1 s.
                        self.player.put(np.zeros(int(self.tts.sample_rate * SECTION_PAUSE_S), np.float32))
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
        self._turn.update(spoken=spoken, times=dict(report=times.report(), context_tokens=times.context_tokens))
        if failed:  # not kept: half a turn would confuse the next one
            print()
            times.failed = True
            self.events.turn(dict(self._turn, failed=repr(failed[0]), nets=dict(self.nets - nets_before)), [])
            return times
        # What wasn't said (say, above) isn't kept either.
        said = context.kept("".join(last_said).strip(), user, steps, called, self.convo.history, self.nets_act,
                            crew_before, self.fired)
        dropped = self.convo.record(user, said, steps)
        self.events.turn(dict(self._turn, kept=said, nets=dict(self.nets - nets_before), dropped=dropped),
                         self.convo.history[-(len(steps) + 2):])
        self.events.after_turn(self.convo)
        if self.nets_act:  # (raw, they're the model's own, not kept with real use's)
            self.net_counts.add(self.model, self.nets - nets_before)
        print(f"{DIM}  {times.report()}{RESET}")
        if dropped:
            print(f"{DIM}  (history past {self.cfg.persona.history_hard_tokens:,} tokens: dropped the "
                  f"oldest {dropped} exchanges, without notes){RESET}")
        print(f"{DIM}  {context_report(self.cfg, self.convo, times.context_tokens)}{RESET}\n")
        return times

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

    async def _offer(self, user: str) -> str:
        """context.memory_for, kept with the turn; or, replaying, the memory offered then."""
        if self.replaying is not None and "memory" in self.replaying:
            got = self.replaying["memory"]
        else:
            got = await context.memory_for(self, user)
        self._turn.setdefault("decisions", {})["memory"] = got
        return got

    def _warm_archive(self) -> None:
        try:
            self.archive.warm()
        except Exception as e:  # LM Studio's embedding model not up: the first search does it
            print(f"{DIM}  (the archive's vectors, not loaded: {e!r}){RESET}", flush=True)

    # Speaking up by himself -----------------------------------------------------

    def pending(self) -> Announcement | None:
        """Timers come due, or a report finished, to speak up about now (app/announce.py)."""
        return pending(self.timers, self.research)

    async def announce(self, announcement: Announcement) -> TurnTimes | None:
        """Speak up: his reply to a note saying what's happened, kept in his history like
        any turn. If it fails, what it was about is kept, to try again later."""
        note(announcement.what)
        times = None
        try:
            self.player.wake()
            times = await self.respond(announcement.note)
        except Exception as e:
            print(f"{YELLOW}  (couldn't announce it: {e!r}){RESET}\n", flush=True)
        ok = times is not None and not times.failed
        announcement.done(ok)
        if not ok:
            print(f"{DIM}  (not announced: {announcement.what}; trying again later){RESET}\n", flush=True)
        return times

    async def _watch(self) -> None:
        """Typed: there's no listening loop to check between, so a task checks every
        second, speaks up between your turns, and otherwise lets him doze."""
        while True:
            await asyncio.sleep(1)
            if announcement := self.pending():
                self.background.wake()
                async with self.turn_lock:
                    await self.announce(announcement)
                print("You: ", end="", flush=True)
            elif not self.turn_lock.locked():
                self.background.doze()  # while you're typing, or not

    async def run(self) -> None:
        await self.start()
        watcher = asyncio.create_task(self._watch())
        try:
            while True:
                # Typed, there's no sleep to wait for: his notes are written while
                # you type, and dropped if you send before they're done.
                self.background.doze()
                user = (await asyncio.to_thread(input, "You: ")).strip()
                self.background.wake()
                if not user:
                    continue
                if user in ("/quit", "/exit"):
                    return
                if user == "/reset":
                    async with self.turn_lock:  # not mid-way through a timer's announcement
                        self.convo.reset()
                        self.events.clear()
                        if self.timers:
                            self.timers.clear()
                        if self.research:
                            self.research.clear()
                    print(f"{DIM}History, notes, archive, timers, reminders, research and event log cleared, "
                          f"on disk too.{RESET}\n")
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
            self.background.stop()
            if self.player:
                self.player.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Holly; he answers out loud")
    parser.add_argument("--model", help="LM Studio chat model key (default: holly.toml)")
    parser.add_argument("--face", action=argparse.BooleanOptionalAction, help="show Holly's face (default: holly.toml)")
    parser.add_argument("--search", action=argparse.BooleanOptionalAction,
                        help="let Holly search the web (default: holly.toml)")
    parser.add_argument("--forget", action="store_true", help=yours.FORGET_HELP)
    parser.add_argument("--reset", action="store_true", help=yours.RESET_HELP)
    args = parser.parse_args()
    # The Windows console defaults to cp1252, which can't print emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(Holly(args.model, args.face, args.search, args.forget, args.reset).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

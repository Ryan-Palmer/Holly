"""Milestone 2: type to Holly, and he answers out loud in a stock voice.

    python -m app.speak [--model MODEL] [--no-search] [--face] [--forget]

The reply streams: the LLM's tokens are cut into sentences, each sentence
goes to TTS as soon as it's complete, and audio plays as it's decoded. So
Holly starts talking while the LLM is still writing.

Holly can search the web (app/search.py). When his reply starts with a
search instead of words, he says a stock line ("Hang on, let me look that
up"), rendered at startup so it plays at once, while the search runs.

His tools are in app/tools.py and beside their code; the safety nets that
make him redo a reply are in app/guards.py; what he does while he sleeps is
in app/background.py; speaking up by himself, in app/announce.py.

Commands: /reset clears the conversation, /quit leaves.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import random
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from . import archive, cpu, guards, news, offers, research, spotify, status, timers
from .announce import Announcement, pending
from .background import Background, context_report
from .clock import ClockTool
from .config import load_config
from .console import CYAN, DIM, RESET, YELLOW, note, stamp
from .face_player import FacePlayer, start_server
from .lmstudio import LmStudio, ReplyTiming
from .nets import NetCounts
from .persona import Conversation, about_the_crew, set_aside, when
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
        if reset:
            # Everything of yours, for good: history and notes, the archive, timers and
            # reminders, and whatever --forget or an unreadable file set aside. Not the
            # ship's records.
            gone = wipe_yours([memory_file, cfg.recall.file, cfg.timers.file]) + wipe_folder(cfg.research.dir)
            print(f"{DIM}Deleted {len(gone)} file{'s' * (len(gone) != 1)} of Holly's memory"
                  f"{': ' + ', '.join(p.name for p in gone) if gone else ''}; he starts afresh.{RESET}")
        elif forget:
            # Everything of yours (history and notes, the archive, timers and reminders), but
            # not the ship's records. Set aside, not deleted, in case it was a mistake; each
            # with its own time, so a second --forget doesn't overwrite the first.
            yours = [memory_file, cfg.recall.file, cfg.timers.file, cfg.research.dir]
            asides = [set_aside_with_sidecars(f, "forgotten") for f in yours if f and f.exists()]
            if asides:
                print(f"{DIM}Holly's memory of earlier sessions is set aside "
                      f"({', '.join(a.name for a in asides)}); he starts afresh.{RESET}")
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
        # The nets on his replies (guards.NETS), each as it fires: they make up for a model's
        # habits, and the evals show which a model still trips. With nets_act off (evals
        # --raw) they're counted, and do nothing: the model as it is.
        self.nets: Counter[str] = Counter()
        self.nets_act = True
        # ...and kept between sessions, per model, beside his memory (app/nets.py): what real
        # talk trips, not only the evals'.
        self.net_counts = NetCounts(memory_file.with_name("nets.json") if memory_file else None)

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
            messages = self.convo.messages_for(user)
            asked = len(messages)
            lookups = 0  # slow tools called this reply
            # Their answer to his search-or-report question (the research tool held a start and
            # had him ask), or their taking him up on something he offered (app/offers.py):
            # either way, what they chose is theirs, and research isn't held for asking which.
            held = bool(self.research_tool and self.research_tool.held_turn == self.turns - 1)
            took_up = await asyncio.to_thread(self._taken_up, user)
            if took_up and offers.NOTE:
                # Which tool is his to choose: told what they took up, in this request only.
                messages = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + "\n" + offers.NOTE}]
            turn = Turn(user, self.turns, times, lambda kind: self._say_filler(times, kind),
                        held or took_up, self.convo.crew_lines)
            retried = False  # a reply was dropped part-way and asked again (once a turn)
            # "required": the reply must call a tool. So on a retry, and when the crew have just
            # chosen a report in answer to his search-or-report question: then he said "I'll get
            # cracking", "I'll get stuck into it", in ever new words, and started nothing. And
            # when he's asked for something from the past, as the model judges it
            # (archive.about_the_past): "do you remember Jo?", "what did we do yesterday?",
            # follow-ups. Asked from his notes alone, he said "I don't have anything in my notes
            # about a problem", with it in his archive; with a recall a turn earlier to riff on,
            # he guessed what the cats thought of Lister ("a walking disaster area"). (If the
            # model honours it: else a retry is just asked again.)
            required = "required" if self.cfg.llm.tool_choice else None
            choice = (required if took_up
                      or (held and "research" in self.toolbox and guards.REPORT_WANTED.search(user)) else None)
            # A status report, with one already in his history: he gave one without the tool,
            # three times in three, made up or the last one again; with none, he called it
            # every time. So he must call a tool, and the earlier reports are stubbed before he
            # sees them, not only once he's called it.
            if (not user.startswith("[") and "status" in self.toolbox and guards.STATUS_ASKED.search(user)
                    and await asyncio.to_thread(status.wants_report, self.lm, self.model, user)):
                messages = status.without_earlier_reports(messages, asked - 1)
                choice = choice or required
            # A report asked for in so many words (guards.REPORT_ASKED): well into a talk, he
            # asked "a quick search now, or a proper report?" anyway, calling nothing. With
            # something to report on (guards.reported_on): "write me a report", alone, isn't.
            if (not user.startswith("[") and "research" in self.toolbox and guards.REPORT_ASKED.search(user)
                    and guards.reported_on(user)):
                choice = choice or required
            # A timer to do something with a tool ("check the weather in Keswick"), not just say.
            if (user.startswith("[") and (labels := [label for label in guards.DUE_LABEL.findall(user)
                                                     if guards.ACTION_WORDS.search(label)])
                    and await asyncio.to_thread(timers.to_do, self.lm, self.model, labels)):
                choice = choice or required
            # Which tool is his to choose: made to call one for a memory, he chose recall every
            # time (25 of 25 in the evals). A web search turned into recall only ever caught the
            # news, judged a memory by mistake.
            if required and not choice and "recall" in self.toolbox and not user.startswith("["):
                if await asyncio.to_thread(archive.about_the_past, self.lm, self.model, user,
                                           self._last_exchange()):
                    choice = required

            def told() -> list[str]:
                """What he's been given this turn: the crew's line, and his tools' answers."""
                return [user] + [m.get("content") or "" for m in messages[asked:] if m["role"] == "tool"]

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
                why = guards.unsaid(sentence, user, stray=self.stray_words, recalled="recall" in called,
                                    researched="research" in called, can_recall="recall" in self.toolbox,
                                    can_research="research" in self.toolbox,
                                    answering_choice=turn.answering_choice, clocked="clock" in called,
                                    can_clock="clock" in self.toolbox,
                                    counting=bool(self.timers and self.timers.items) and "timer" not in called,
                                    told=told())
                return why if why and self.fired(why) else None

            while True:
                calls: list[dict] = []
                said: list[str] = []
                redo = None
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
                if redo:
                    calls.clear()  # whatever it went on to ask for, it's asked again
                if not redo and not calls:  # the end of the reply: its last words
                    last = chunker.flush()
                    lone = not retried and not spoken and len(last) == 1 and len(last[0].split()) == 1
                    for sentence in last:
                        if redo := dropped(sentence):
                            break
                        if lone and self.fired(guards.ALONE):
                            # A word alone, his whole reply ("Yeah?", to "Hey Holly"): the voice garbled
                            # it. Asked again, for a few words; a word again, it's said.
                            redo = guards.ALONE
                            messages = [*messages[:asked - 1], {**messages[asked - 1], "content": (
                                messages[asked - 1]["content"] + "\n[A word alone doesn't come out right in "
                                "your voice: say a few words.]")}, *messages[asked:]]
                            break
                        await say(sentence)
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
        if failed:  # not kept: half a turn would confuse the next one
            print()
            times.failed = True
            return times
        this_turn = [user] + [m.get("content") or "" for m in steps if m["role"] == "tool"]
        said = "".join(last_said).strip()
        if self.nets_act:  # what wasn't said (say, above) isn't kept either
            said = guards.without_offers(said, countdowns=unasked_countdown,
                                         told=this_turn if unasked_countdown else None,
                                         asides=not (crew_before or about_the_crew(user, called)))
        # What he was told: this turn, and the last few (asked "you mean 24 minutes past
        # midnight?" a turn after the clock said 00:24, his "00:24" was no guess).
        told = this_turn + [m.get("content") or "" for m in self.convo.history[-12:]]
        if guards.guessed_time(said, told) and self.fired(guards.GUESS_MARKED):
            said += guards.GUESSED
        dropped = self.convo.record(user, said, steps)
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

    def _last_exchange(self, exchanges: int = 2, chars: int = 600) -> str:
        """The last couple of exchanges, the crew's lines and his replies, for judging a
        follow-up. Two, not one: after "Human Toes" / "I've already given you the news",
        "I just wanted to know more about the toe eating" was taken for a memory, the toe
        eating being his own made-up news a turn earlier. A long reply keeps its start and
        its end, where that story was."""
        said: list[str] = []
        users = 0
        for m in reversed(self.convo.history):
            if m["role"] not in ("user", "assistant") or not m.get("content"):
                continue
            text = " ".join(m["content"].split())
            said.append(text if len(text) <= chars else f"{text[:chars // 2]} ... {text[-chars // 2:]}")
            if m["role"] == "user" and (users := users + 1) == exchanges:
                break
        return " / ".join(reversed(said))

    def _taken_up(self, user: str) -> bool:
        """Are the crew taking him up on something he offered or asked about doing in his last
        reply ("Want me to search for it now or later?" / "You can get it now if you like")?
        As the model judges (app/offers.py); asked only if that reply offered or asked
        something (guards.ASKED_OR_OFFERED)."""
        if not self.tools or user.startswith("["):
            return False
        said = next((m.get("content") or "" for m in reversed(self.convo.history)
                     if m["role"] == "assistant" and m.get("content")), "")
        return bool(guards.ASKED_OR_OFFERED.search(said)) and offers.taken_up(self.lm, self.model, said, user)

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
                        if self.timers:
                            self.timers.clear()
                        if self.research:
                            self.research.clear()
                    print(f"{DIM}History, notes, archive, timers, reminders and research cleared, on disk "
                          f"too.{RESET}\n")
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


FORGET_HELP = ("start afresh, setting aside (renamed, not deleted) his history, notes, archive, timers, "
               "reminders and research; the ship's records stay")
RESET_HELP = ("start afresh, deleting his history, notes, archive, timers, reminders and research for good, "
              "and anything --forget set aside; the ship's records stay")


def sidecars(f: Path) -> list[Path]:
    """SQLite's files beside a database ("archive.db-journal"). They belong to it: a
    journal left by a crash would be rolled back into a new database of the same name."""
    return [s for s in (f.with_name(f.name + x) for x in ("-journal", "-wal", "-shm")) if s.exists()]


def set_aside_with_sidecars(f: Path, why: str) -> Path:
    aside = set_aside(f, why)
    for s in sidecars(f):
        s.replace(aside.with_name(aside.name + s.name[len(f.name):]))
    return aside


def wipe_yours(files: list[Path | None]) -> list[Path]:
    """Delete these files and any copies set aside from them (".forgotten-<time>",
    ".unreadable-<time>"), their half-written ".tmp" files, and SQLite's files
    beside them. Returns what went."""
    gone = []
    for f in files:
        if not f:
            continue
        for p in [f, f.with_suffix(".tmp"), *f.parent.glob(f"{f.stem}.forgotten-*{f.suffix}"),
                  *f.parent.glob(f"{f.stem}.unreadable-*{f.suffix}")]:
            for q in [p, *sidecars(p)] if p.exists() else sidecars(p):
                q.unlink()
                gone.append(q)
    return gone


def wipe_folder(d: Path) -> list[Path]:
    """Delete a folder of his (his research) and any copies --forget set aside."""
    gone = []
    for p in [d, *d.parent.glob(f"{d.name}.forgotten-*")] if d.parent.exists() else []:
        if p.is_dir():
            shutil.rmtree(p)
            gone.append(p)
    return gone


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Holly; he answers out loud")
    parser.add_argument("--model", help="LM Studio chat model key (default: holly.toml)")
    parser.add_argument("--face", action=argparse.BooleanOptionalAction, help="show Holly's face (default: holly.toml)")
    parser.add_argument("--search", action=argparse.BooleanOptionalAction,
                        help="let Holly search the web (default: holly.toml)")
    parser.add_argument("--forget", action="store_true", help=FORGET_HELP)
    parser.add_argument("--reset", action="store_true", help=RESET_HELP)
    args = parser.parse_args()
    # The Windows console defaults to cp1252, which can't print emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(Holly(args.model, args.face, args.search, args.forget, args.reset).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

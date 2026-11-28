"""Loads holly.toml from the repo root."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "holly.toml"


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    context_length: int
    parallel: int
    eval_batch_size: int
    reasoning_effort: str
    temperature: float
    max_tokens: int
    # A model's quirks, usually set in its [models."key"] table (see for_model):
    stray_words: tuple[str, ...] = ()  # a sentence of only one of these is dropped, unsaid
    tool_choice: bool = True  # it honours tool_choice "required" (a reply that must call a tool)
    # A model that can't stop thinking (LM Studio offers it no "off"): its least effort, sent
    # when it's asked not to think, and the tokens added for that thinking (see LmStudio._reasoning).
    least_thinking: str = ""
    thinking_tokens: int = 0
    # Each chat model's settings: [llm]'s, with its [models."key"] table over them ("": none).
    profiles: dict = field(default_factory=dict, compare=False, repr=False)

    def for_model(self, model: str | None = None) -> LlmConfig:
        """These settings for `model` (by default, [llm]'s own): a chat model picked with
        --model gets its own context length, quirks and the rest."""
        model = model or self.model
        return dataclasses.replace(self, model=model, **self.profiles.get(model, self.profiles.get("", {})))


# What a [models."key"] table may set.
PROFILE_KEYS = ("context_length", "reasoning_effort", "temperature", "max_tokens", "stray_words", "tool_choice",
                "least_thinking", "thinking_tokens")


@dataclass(frozen=True)
class VoxCpmConfig:
    model: str
    reference_wav: Path
    reference_text: Path
    compile: bool
    wait_for_reply: bool
    inference_timesteps: int


@dataclass(frozen=True)
class TtsConfig:
    voxcpm: VoxCpmConfig


@dataclass(frozen=True)
class AudioConfig:
    host_api: str
    output_device: str
    block_ms: int


@dataclass(frozen=True)
class ListenConfig:
    input_device: str
    input_channels: list[int]
    start_threshold: float
    end_threshold: float
    end_silence_ms: int
    min_speech_ms: int
    preroll_ms: int
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_hotwords: str = "Holly"  # names Whisper is told to expect
    wake_words: tuple[str, ...] = ("holly", "hollie", "holley")
    wake_edge_words: int = 3
    awake_s: float = 30.0  # after he last spoke: no wake word needed, and his face stays awake
    # The turn-ending model (app/turn.py): "<Hugging Face repo>/<file>", or "" for none (a
    # turn ends after end_silence_ms, whatever's said).
    turn_model: str = ""
    turn_check_ms: int = 200  # silence before it's asked
    turn_threshold: float = 0.5  # finished, at or above this: he answers then
    max_silence_ms: int = 1500  # below it, he waits for more, up to this


@dataclass(frozen=True)
class PersonaConfig:
    system_prompt: Path
    examples: Path
    notes: Path = ROOT / "persona/notes.toml"  # the lines added to what he's sent (app/context.py)
    # Token budgets (~4 characters a token). Past soft, the older history is folded
    # into his notes while he sleeps; past hard, the oldest is dropped at once.
    history_soft_tokens: int = 20_000
    history_hard_tokens: int = 26_000
    history_keep_tokens: int = 8_000  # kept word for word when compacting
    history_resume_tokens: int = 1_500  # of the last session's history, kept word for word at startup
    memory_tokens: int = 1_500  # his notes' length, at most
    memory_file: Path | None = None  # where history and notes are kept between sessions


@dataclass(frozen=True)
class FaceConfig:
    enabled: bool = False
    python: str = "face/.venv/Scripts/python.exe"
    image: str = "face/data/holly_colour.jpg"
    port: int = 50555
    lead_ms: int = 600
    height: int = 0  # 0: the still's own size
    idle_head: float = 0.2
    crt: bool = False  # an old monitor's look: a curved screen, scanlines, a rolling bar
    case: str = ""  # with crt: a picture of the monitor, its screen opening see-through; "" for a plain bezel


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = True
    provider: str = "ddgs"  # "ddgs" or "brave"
    backend: str = "yahoo,brave,duckduckgo"  # ddgs engines, or "auto"
    region: str = "uk-en"
    max_results: int = 6
    timeout_s: float = 5.0
    max_searches: int = 3  # lookups per reply; the clock doesn't count
    summary_max_tokens: int = 160
    max_rounds: int = 3  # searches and page reads one lookup may make
    max_lookup_s: float = 5.0  # no new search or read is started after this long
    context_tokens: int = 2048  # page text brave_context returns per search, in all
    summary_reasoning: str = "none"  # Gemma's thinking when judging results: "none", "low"...
    weather: bool = True  # the weather tool (Open-Meteo), with search
    news: bool = True  # the news tool (the BBC's headlines), with search
    home_country: str = "GB"  # places here come first in weather lookups
    fillers: tuple[str, ...] = ()
    # Wikipedia first, for settled facts (app/search.py): the opening paragraphs of its best
    # few articles, in one request; the web only if they don't answer, or it's about now.
    wikipedia: bool = True
    wiki_lang: str = "en"
    wiki_articles: int = 3
    wiki_timeout_s: float = 2.0
    # Wikimedia refuses requests without a contact: the repo's URL, nothing personal.
    user_agent: str = "Holly/0.1 (https://github.com/Ryan-Palmer/Holly)"


@dataclass(frozen=True)
class RecallConfig:
    enabled: bool = True
    file: Path = ROOT / "data/archive.db"  # every exchange that leaves his history
    records_file: Path = ROOT / "data/records.db"  # the ship's records (app/records.py), if built
    max_entries: int = 12  # keyword matches the judge reads, best first
    # Matches by meaning, from an embedding model in LM Studio ("" for keywords only).
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    meaning_entries: int = 8  # the closest in meaning the judge reads, at least min_similarity
    min_similarity: float = 0.5
    summary_max_tokens: int = 200
    fillers: tuple[str, ...] = ()
    # A memory offered, unasked: what the crew said on another day about what they're telling
    # him now, found as they finish (app/speak.py), to bring up if it fits. At most one every
    # offer_every replies, each memory once a session; dropped if not found in offer_ms.
    offer_memories: bool = True
    offer_every: int = 3
    offer_ms: int = 400


@dataclass(frozen=True)
class TimersConfig:
    enabled: bool = True
    file: Path = ROOT / "data/timers.json"  # what's set, kept between sessions


@dataclass(frozen=True)
class SpotifyConfig:
    enabled: bool = True  # and only if SPOTIFY_CLIENT_ID is set
    token_file: Path = ROOT / "data/spotify_token.json"  # the login, from `python -m app.spotify login`
    redirect_uri: str = "http://127.0.0.1:8888/callback"  # as registered with the Spotify app
    device: str = ""  # with nothing playing, play here (part of its name); else this PC, else any
    launch: bool = True  # with Spotify open nowhere, open the desktop app on this PC
    launch_wait_s: float = 10.0  # and wait this long for it to appear
    volume_step: int = 15  # "turn it up": by this many percent
    market: str = "GB"  # the country whose catalogue to search (Spotify's own "from_token" needs more access)
    timeout_s: float = 5.0
    fillers: tuple[str, ...] = ()  # said while the desktop app is being opened


@dataclass(frozen=True)
class ResearchConfig:
    enabled: bool = True  # and only with search on
    dir: Path = ROOT / "data/research"  # the jobs, and each report as markdown
    provider: str = "ddgs"  # web search for research: free; the choices are as under [search]
    wikipedia: str = "en"  # Wikipedia's language, searched alongside the web; "" for none
    user_agent: str = "HollyResearch/0.1 (https://github.com/Ryan-Palmer/Holly)"  # Wikipedia wants a contact
    questions: int = 5  # a job's research questions, at most
    gaps: int = 2  # questions the check may add
    searches: int = 3  # per question
    reads: int = 4  # pages read per question
    results: int = 8  # per web search
    page_chars: int = 12_000  # of each page, for the notes
    notes: int = 12  # per page, at most
    report_words: int = 900
    pause_s: float = 4.0  # between searches, so the free engines don't take it for a bot
    timeout_s: float = 15.0  # for a search or a page
    max_minutes: float = 45.0  # of work per job; then the report is written from what's found
    max_errors: int = 5  # failed steps in a row before a job is given up


@dataclass(frozen=True)
class StatusConfig:
    enabled: bool = True  # the status report (app/status.py): the crew's doings, made up
    file: Path = ROOT / "data/status.json"  # the last few reports, to write something different from
    fillers: tuple[str, ...] = ()  # said while a report's written
    crew_fillers: tuple[str, ...] = ()  # said while an answer about one of them is ("where's Kryten?")


@dataclass(frozen=True)
class EvalsConfig:
    scenario_runs: int = 3  # runs of each scenario (and conversation): Gemma varies
    # The judges, whatever model is under test: each scores everything judged, and how hard
    # each thinks over a whole conversation (app/conversations.py).
    judges: dict = field(default_factory=lambda: {"google/gemma-4-26b-a4b-qat": "none",
                                                  "qwen/qwen3.8-27b": "medium"}, compare=False)
    judge_context: int = 49152  # tokens: a whole conversation, its tools' answers, and thinking
    reply_judge: str = "qwen/qwen3.8-27b"  # scores the persona replies alone, as the user would


@dataclass(frozen=True)
class CpuConfig:
    startup: str = "all"  # "efficiency": start up on the efficiency cores only (see app/cpu.py)


@dataclass(frozen=True)
class Config:
    llm: LlmConfig
    tts: TtsConfig
    audio: AudioConfig
    listen: ListenConfig
    persona: PersonaConfig
    face: FaceConfig
    search: SearchConfig
    recall: RecallConfig = RecallConfig()
    timers: TimersConfig = TimersConfig()
    spotify: SpotifyConfig = SpotifyConfig()
    research: ResearchConfig = ResearchConfig()
    status: StatusConfig = StatusConfig()
    evals: EvalsConfig = EvalsConfig()
    cpu: CpuConfig = CpuConfig()


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    base = path.parent
    p = raw["persona"]
    t = raw["tts"]
    v = t["voxcpm"]
    tts = TtsConfig(
        voxcpm=VoxCpmConfig(
            model=v["model"],
            reference_wav=base / v["reference_wav"],
            reference_text=base / v["reference_text"],
            compile=v["compile"],
            wait_for_reply=v.get("wait_for_reply", False),
            inference_timesteps=v.get("inference_timesteps", 10),
        ),
    )
    return Config(
        llm=_llm_config(raw["llm"], raw.get("models", {})),
        tts=tts,
        audio=AudioConfig(**raw["audio"]),
        listen=ListenConfig(**raw["listen"]),
        persona=PersonaConfig(
            system_prompt=base / p["system_prompt"],
            examples=base / p["examples"],
            **({"notes": base / p["notes"]} if p.get("notes") else {}),
            memory_file=base / p["memory_file"] if p.get("memory_file") else None,
            **{k: p[k] for k in ("history_soft_tokens", "history_hard_tokens", "history_keep_tokens",
                                 "history_resume_tokens", "memory_tokens") if k in p},
        ),
        face=FaceConfig(**raw.get("face", {})),
        search=_search_config(raw.get("search", {}), base),
        recall=_recall_config(raw.get("recall", {}), base),
        timers=TimersConfig(**{k: (base / v if k == "file" else v)
                               for k, v in raw.get("timers", {}).items()}),
        spotify=SpotifyConfig(**{k: (base / v if k == "token_file" else
                                     tuple(v) if k == "fillers" else v)
                                 for k, v in raw.get("spotify", {}).items()}),
        research=ResearchConfig(**{k: (base / v if k == "dir" else v)
                                   for k, v in raw.get("research", {}).items()}),
        status=StatusConfig(**{k: (base / v if k == "file" else tuple(v) if k.endswith("fillers") else v)
                               for k, v in raw.get("status", {}).items()}),
        evals=EvalsConfig(**raw.get("evals", {})),
        cpu=CpuConfig(**raw.get("cpu", {})),
    )


def _llm_config(llm: dict, models: dict) -> LlmConfig:
    """[llm], with the [models."key"] table of its model over it."""
    def settings(table: dict) -> dict:
        return {k: tuple(v) if k == "stray_words" else v for k, v in table.items()}

    for key, table in models.items():
        if unknown := set(table) - set(PROFILE_KEYS):
            raise ValueError(f"[models.{key!r}] in holly.toml can't set {', '.join(sorted(unknown))}; "
                             f"only {', '.join(PROFILE_KEYS)}")
    # Every key, defaults too: else a model without a table kept the last one's quirks.
    defaults = {f.name: f.default for f in dataclasses.fields(LlmConfig) if f.name in PROFILE_KEYS}
    base = settings(defaults | {k: llm[k] for k in PROFILE_KEYS if k in llm})
    profiles = {"": base} | {key: base | settings(table) for key, table in models.items()}
    return LlmConfig(**settings(llm), profiles=profiles).for_model()


def _search_config(s: dict, base: Path) -> SearchConfig:
    s = dict(s)
    if "fillers" in s:
        s["fillers"] = tuple(s["fillers"])
    return SearchConfig(**s)


def _recall_config(r: dict, base: Path) -> RecallConfig:
    r = dict(r)
    for key in ("file", "records_file"):
        if key in r:
            r[key] = base / r[key]
    if "fillers" in r:
        r["fillers"] = tuple(r["fillers"])
    return RecallConfig(**r)

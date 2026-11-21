"""Loads holly.toml from the repo root."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PersonaConfig:
    system_prompt: Path
    examples: Path
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
    prompt: Path = ROOT / "persona/search_prompt.md"
    weather: bool = True  # the weather tool (Open-Meteo), with search
    home_country: str = "GB"  # places here come first in weather lookups
    fillers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecallConfig:
    enabled: bool = True
    file: Path = ROOT / "data/archive.db"  # every exchange that leaves his history
    records_file: Path = ROOT / "data/records.db"  # the ship's records (app/records.py), if built
    reminders: int = 5  # entries from the records in his prompt, refreshed while he sleeps
    max_entries: int = 12  # keyword matches the judge reads, best first
    # Matches by meaning, from an embedding model in LM Studio ("" for keywords only).
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    meaning_entries: int = 8  # the closest in meaning the judge reads, at least min_similarity
    min_similarity: float = 0.5
    summary_max_tokens: int = 200
    prompt: Path = ROOT / "persona/recall_prompt.md"
    fillers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimersConfig:
    enabled: bool = True
    file: Path = ROOT / "data/timers.json"  # what's set, kept between sessions
    prompt: Path = ROOT / "persona/timers_prompt.md"


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
    prompt: Path = ROOT / "persona/spotify_prompt.md"
    fillers: tuple[str, ...] = ()  # said while the desktop app is being opened


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
        llm=LlmConfig(**raw["llm"]),
        tts=tts,
        audio=AudioConfig(**raw["audio"]),
        listen=ListenConfig(**raw["listen"]),
        persona=PersonaConfig(
            system_prompt=base / p["system_prompt"],
            examples=base / p["examples"],
            memory_file=base / p["memory_file"] if p.get("memory_file") else None,
            **{k: p[k] for k in ("history_soft_tokens", "history_hard_tokens", "history_keep_tokens",
                                 "history_resume_tokens", "memory_tokens") if k in p},
        ),
        face=FaceConfig(**raw.get("face", {})),
        search=_search_config(raw.get("search", {}), base),
        recall=_recall_config(raw.get("recall", {}), base),
        timers=TimersConfig(**{k: (base / v if k in ("file", "prompt") else v)
                               for k, v in raw.get("timers", {}).items()}),
        spotify=SpotifyConfig(**{k: (base / v if k in ("token_file", "prompt") else
                                     tuple(v) if k == "fillers" else v)
                                 for k, v in raw.get("spotify", {}).items()}),
    )


def _search_config(s: dict, base: Path) -> SearchConfig:
    s = dict(s)
    if "prompt" in s:
        s["prompt"] = base / s["prompt"]
    if "fillers" in s:
        s["fillers"] = tuple(s["fillers"])
    return SearchConfig(**s)


def _recall_config(r: dict, base: Path) -> RecallConfig:
    r = dict(r)
    for key in ("file", "records_file", "prompt"):
        if key in r:
            r[key] = base / r[key]
    if "fillers" in r:
        r["fillers"] = tuple(r["fillers"])
    return RecallConfig(**r)

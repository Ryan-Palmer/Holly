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
class OrpheusConfig:
    model: str
    context_length: int
    voice: str
    temperature: float
    top_p: float
    repeat_penalty: float
    snac_model: str
    device: str


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
    engine: str  # "voxcpm" or "orpheus"
    voxcpm: VoxCpmConfig
    orpheus: OrpheusConfig


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
    whisper_hotwords: str = "Holly"
    wake_words: tuple[str, ...] = ("holly", "hollie", "holley")
    wake_edge_words: int = 3
    awake_s: float = 30.0  # after he last spoke: no wake word needed, and his face stays awake


@dataclass(frozen=True)
class PersonaConfig:
    system_prompt: Path
    examples: Path
    history_turns: int
    history_trim_to: int


@dataclass(frozen=True)
class FaceConfig:
    enabled: bool = False
    python: str = "face/.venv/Scripts/python.exe"
    image: str = "face/data/holly_colour.jpg"
    port: int = 50555
    lead_ms: int = 600
    height: int = 720
    idle_head: float = 0.2


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = True
    provider: str = "ddgs"  # "ddgs" or "brave"
    backend: str = "yahoo,brave,duckduckgo"  # ddgs engines, or "auto"
    region: str = "uk-en"
    max_results: int = 6
    timeout_s: float = 5.0
    max_searches: int = 2
    summary_max_tokens: int = 160
    max_rounds: int = 3  # searches and page reads one lookup may make
    max_lookup_s: float = 5.0  # no new search or read is started after this long
    summary_reasoning: str = "none"  # Gemma's thinking when judging results: "none", "low"...
    prompt: Path = ROOT / "persona/search_prompt.md"
    weather: bool = True  # the weather tool (Open-Meteo), with search
    home_country: str = "GB"  # places here come first in weather lookups
    fillers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    llm: LlmConfig
    tts: TtsConfig
    audio: AudioConfig
    listen: ListenConfig
    persona: PersonaConfig
    face: FaceConfig
    search: SearchConfig


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    base = path.parent
    p = raw["persona"]
    t = raw["tts"]
    v = t["voxcpm"]
    tts = TtsConfig(
        engine=t["engine"],
        voxcpm=VoxCpmConfig(
            model=v["model"],
            reference_wav=base / v["reference_wav"],
            reference_text=base / v["reference_text"],
            compile=v["compile"],
            wait_for_reply=v.get("wait_for_reply", False),
            inference_timesteps=v.get("inference_timesteps", 10),
        ),
        orpheus=OrpheusConfig(**t["orpheus"]),
    )
    return Config(
        llm=LlmConfig(**raw["llm"]),
        tts=tts,
        audio=AudioConfig(**raw["audio"]),
        listen=ListenConfig(**raw["listen"]),
        persona=PersonaConfig(
            system_prompt=base / p["system_prompt"],
            examples=base / p["examples"],
            history_turns=p["history_turns"],
            history_trim_to=p.get("history_trim_to", p["history_turns"] * 2 // 3),
        ),
        face=FaceConfig(**raw.get("face", {})),
        search=_search_config(raw.get("search", {}), base),
    )


def _search_config(s: dict, base: Path) -> SearchConfig:
    s = dict(s)
    if "prompt" in s:
        s["prompt"] = base / s["prompt"]
    if "fillers" in s:
        s["fillers"] = tuple(s["fillers"])
    return SearchConfig(**s)

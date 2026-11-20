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


@dataclass(frozen=True)
class PersonaConfig:
    system_prompt: Path
    examples: Path
    history_turns: int
    history_trim_to: int


@dataclass(frozen=True)
class Config:
    llm: LlmConfig
    tts: TtsConfig
    audio: AudioConfig
    listen: ListenConfig
    persona: PersonaConfig


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
    )

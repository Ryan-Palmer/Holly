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
    reasoning_effort: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class TtsConfig:
    model: str
    context_length: int
    voice: str
    temperature: float
    top_p: float
    repeat_penalty: float
    snac_model: str
    device: str


@dataclass(frozen=True)
class AudioConfig:
    host_api: str
    output_device: str
    blocksize: int


@dataclass(frozen=True)
class PersonaConfig:
    system_prompt: Path
    examples: Path
    history_turns: int


@dataclass(frozen=True)
class Config:
    llm: LlmConfig
    tts: TtsConfig
    audio: AudioConfig
    persona: PersonaConfig


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    base = path.parent
    p = raw["persona"]
    return Config(
        llm=LlmConfig(**raw["llm"]),
        tts=TtsConfig(**raw["tts"]),
        audio=AudioConfig(**raw["audio"]),
        persona=PersonaConfig(
            system_prompt=base / p["system_prompt"],
            examples=base / p["examples"],
            history_turns=p["history_turns"],
        ),
    )

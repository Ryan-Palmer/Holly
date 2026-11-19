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
class PersonaConfig:
    system_prompt: Path
    examples: Path
    history_turns: int


@dataclass(frozen=True)
class Config:
    llm: LlmConfig
    persona: PersonaConfig


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    base = path.parent
    llm = LlmConfig(**raw["llm"])
    p = raw["persona"]
    persona = PersonaConfig(
        system_prompt=base / p["system_prompt"],
        examples=base / p["examples"],
        history_turns=p["history_turns"],
    )
    return Config(llm=llm, persona=persona)

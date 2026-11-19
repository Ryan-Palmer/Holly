"""LM Studio client: model loading over its REST API, chat over its OpenAI API."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx
from openai import OpenAI

from .config import LlmConfig


@dataclass
class ReplyTiming:
    """Milliseconds from sending the request."""

    first_token_ms: float | None = None
    total_ms: float | None = None
    reasoning_chars: int = 0


class LmStudio:
    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.base_url, timeout=300)
        self.openai = OpenAI(base_url=f"{cfg.base_url}/v1", api_key="lm-studio")

    # Model management -------------------------------------------------

    def loaded_instances(self, model: str) -> list[dict]:
        resp = self.http.get("/api/v1/models")
        resp.raise_for_status()
        for m in resp.json()["models"]:
            if m["key"] == model:
                return m.get("loaded_instances", [])
        raise ValueError(f"LM Studio has no model called {model!r}")

    def unload(self, model: str) -> None:
        for inst in self.loaded_instances(model):
            self.http.post("/api/v1/models/unload", json={"instance_id": inst["id"]}).raise_for_status()

    def ensure_loaded(self, model: str | None = None) -> bool:
        """Load the model with our context length unless it already is.

        A model that LM Studio loaded on demand gets its default context,
        which for these models is huge and pushes the KV cache out of VRAM.
        So an instance with a different context length is reloaded.
        Returns True if a load happened.
        """
        model = model or self.cfg.model
        instances = self.loaded_instances(model)
        if any(i["config"].get("context_length") == self.cfg.context_length for i in instances):
            return False
        if instances:
            self.unload(model)
        self.http.post(
            "/api/v1/models/load",
            json={"model": model, "context_length": self.cfg.context_length, "flash_attention": True},
        ).raise_for_status()
        return True

    # Chat ------------------------------------------------------------

    def stream_reply(
        self, messages: list[dict], timing: ReplyTiming, model: str | None = None
    ) -> Iterator[str]:
        """Yield the spoken reply as text deltas, filling in `timing` as it goes."""
        t0 = time.perf_counter()
        stream = self.openai.chat.completions.create(
            model=model or self.cfg.model,
            messages=messages,
            stream=True,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            reasoning_effort=self.cfg.reasoning_effort,
        )
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                extra = delta.model_extra or {}
                reasoning = extra.get("reasoning_content") or extra.get("reasoning")
                if reasoning:
                    timing.reasoning_chars += len(reasoning)
                if delta.content:
                    if timing.first_token_ms is None:
                        timing.first_token_ms = (time.perf_counter() - t0) * 1000
                    yield delta.content
        finally:
            stream.close()
            timing.total_ms = (time.perf_counter() - t0) * 1000

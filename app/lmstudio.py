"""LM Studio client: model loading over its REST API, chat over its OpenAI API."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI, OpenAI

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
        self.aopenai = AsyncOpenAI(base_url=f"{cfg.base_url}/v1", api_key="lm-studio")

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

    def ensure_loaded(self, model: str | None = None, context_length: int | None = None) -> bool:
        """Load the model with the given context length unless it already is.

        A model that LM Studio loaded on demand gets its default context,
        which for the chat models is huge and pushes the KV cache out of
        VRAM. So an instance with a different context length is reloaded.
        Returns True if a load happened.
        """
        model = model or self.cfg.model
        context_length = context_length or self.cfg.context_length
        instances = self.loaded_instances(model)
        if any(i["config"].get("context_length") == context_length for i in instances):
            return False
        if instances:
            self.unload(model)
        self.http.post(
            "/api/v1/models/load",
            json={"model": model, "context_length": context_length, "flash_attention": True},
        ).raise_for_status()
        return True

    # Chat ------------------------------------------------------------

    def _chat_args(self, messages: list[dict], model: str | None) -> dict:
        return dict(
            model=model or self.cfg.model,
            messages=messages,
            stream=True,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            reasoning_effort=self.cfg.reasoning_effort,
        )

    @staticmethod
    def _content(chunk, timing: ReplyTiming, t0: float) -> str | None:
        if not chunk.choices:
            return None
        delta = chunk.choices[0].delta
        extra = delta.model_extra or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning")
        if reasoning:
            timing.reasoning_chars += len(reasoning)
        if delta.content and timing.first_token_ms is None:
            timing.first_token_ms = (time.perf_counter() - t0) * 1000
        return delta.content

    # With thinking off, Qwen 3.8 sometimes answers with nothing but an
    # end-of-turn token (half the time, given some histories). An empty reply
    # costs ~150 ms, so it is simply asked again.
    EMPTY_REPLY_ATTEMPTS = 3

    def stream_reply(
        self, messages: list[dict], timing: ReplyTiming, model: str | None = None
    ) -> Iterator[str]:
        """Yield the spoken reply as text deltas, filling in `timing` as it goes."""
        t0 = time.perf_counter()
        try:
            for _ in range(self.EMPTY_REPLY_ATTEMPTS):
                stream = self.openai.chat.completions.create(**self._chat_args(messages, model))
                got_text = False
                try:
                    for chunk in stream:
                        if text := self._content(chunk, timing, t0):
                            got_text = True
                            yield text
                finally:
                    stream.close()
                if got_text:
                    return
        finally:
            timing.total_ms = (time.perf_counter() - t0) * 1000

    async def astream_reply(
        self, messages: list[dict], timing: ReplyTiming, model: str | None = None
    ) -> AsyncIterator[str]:
        """The async version of `stream_reply`."""
        t0 = time.perf_counter()
        try:
            for _ in range(self.EMPTY_REPLY_ATTEMPTS):
                stream = await self.aopenai.chat.completions.create(**self._chat_args(messages, model))
                got_text = False
                try:
                    async for chunk in stream:
                        if text := self._content(chunk, timing, t0):
                            got_text = True
                            yield text
                finally:
                    await stream.close()
                if got_text:
                    return
        finally:
            timing.total_ms = (time.perf_counter() - t0) * 1000

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
    # From LM Studio, for the last request: the prompt, and what it wrote.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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

    def loaded_models(self) -> set[str]:
        """Keys of every model LM Studio has loaded, embedding models aside."""
        resp = self.http.get("/api/v1/models")
        resp.raise_for_status()
        return {m["key"] for m in resp.json()["models"]
                if m.get("loaded_instances") and m.get("type") != "embedding"}

    def unload(self, model: str) -> None:
        for inst in self.loaded_instances(model):
            self.http.post("/api/v1/models/unload", json={"instance_id": inst["id"]}).raise_for_status()

    def ensure_loaded(
        self, model: str | None = None, context_length: int | None = None, fresh: bool = False
    ) -> bool:
        """Load the model with our settings unless it already is.

        A model that LM Studio loaded on demand gets its defaults: for the
        chat models a 100k+ context that pushes the KV cache out of VRAM. So
        an instance loaded any other way is reloaded. Chat models (called
        without a context_length) get all the [llm] load settings and flash
        attention; others (the embedding model for recall) just the context
        length: LM Studio refuses flash attention for embedding models.

        `fresh` reloads even a matching instance. LM Studio splits a model
        between GPU and CPU at load time, by the VRAM free then, and its
        reported config doesn't show the split. An instance loaded while
        another model held the GPU read prompts 6x slower than a clean load
        (640 against 3,700 tokens/s), with settings that looked identical.
        Returns True if a load happened.
        """
        chat = context_length is None
        model = model or self.cfg.model
        want = {"context_length": context_length or self.cfg.context_length}
        if chat:
            want |= {"parallel": self.cfg.parallel, "eval_batch_size": self.cfg.eval_batch_size}
        instances = self.loaded_instances(model)
        if not fresh and any(all(i["config"].get(k) == v for k, v in want.items()) for i in instances):
            return False
        if instances:
            self.unload(model)
        self.http.post(
            "/api/v1/models/load", json={"model": model, **want, **({"flash_attention": True} if chat else {})}
        ).raise_for_status()
        return True

    # Chat ------------------------------------------------------------

    def _chat_args(self, messages: list[dict], model: str | None, tools: list[dict] | None = None,
                   tool_choice: str | None = None) -> dict:
        args = dict(
            model=model or self.cfg.model,
            messages=messages,
            stream=True,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            reasoning_effort=self.cfg.reasoning_effort,
            stream_options={"include_usage": True},  # token counts, in a last chunk
        )
        if tools:
            args["tools"] = tools
            if tool_choice:  # "required": he must call one (the prompt is the same, so the cache holds)
                args["tool_choice"] = tool_choice
        return args

    @staticmethod
    def _tool_calls(chunk, calls: dict[int, dict]) -> None:
        """Gather streamed tool-call fragments into `calls`, by index."""
        if not chunk.choices:
            return
        for tc in chunk.choices[0].delta.tool_calls or []:
            c = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            c["id"] = tc.id or c["id"]
            if tc.function:
                c["name"] += tc.function.name or ""
                c["arguments"] += tc.function.arguments or ""

    @staticmethod
    def _content(chunk, timing: ReplyTiming, t0: float) -> str | None:
        if usage := getattr(chunk, "usage", None):
            timing.prompt_tokens, timing.completion_tokens = usage.prompt_tokens, usage.completion_tokens
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

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """One vector per text, from an embedding model in LM Studio."""
        resp = self.openai.embeddings.create(model=model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    # Side tasks ------------------------------------------------------
    # Work out of Holly's sight (recall's judge, search's, his notes, research): one
    # system prompt and one user message, no tools. They don't think ("none") unless
    # asked to: thinking costs seconds, and these are short.

    def _side_args(self, system: str, user: str, model: str | None, max_tokens: int, temperature: float,
                   reasoning: str) -> dict:
        return dict(model=model or self.cfg.model, temperature=temperature, max_tokens=max_tokens,
                    reasoning_effort=reasoning, messages=[{"role": "system", "content": system},
                                                         {"role": "user", "content": user}])

    def ask(self, system: str, user: str, model: str | None = None, max_tokens: int = 200,
            temperature: float = 0.2, reasoning: str = "none") -> str:
        """A whole reply, blocking: call it from a worker thread."""
        reply = self.openai.chat.completions.create(
            **self._side_args(system, user, model, max_tokens, temperature, reasoning))
        return (reply.choices[0].message.content or "").strip()

    async def aask(self, system: str, user: str, model: str | None = None, max_tokens: int = 800,
                   temperature: float = 0.2, reasoning: str = "none") -> tuple[str, bool]:
        """A whole reply, streamed, so cancelling it stops LM Studio writing. Returns
        (text, cut): cut if it ran out of tokens."""
        parts: list[str] = []
        cut = False
        stream = await self.aopenai.chat.completions.create(
            **self._side_args(system, user, model, max_tokens, temperature, reasoning), stream=True)
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                parts.append(chunk.choices[0].delta.content or "")
                cut = cut or chunk.choices[0].finish_reason == "length"
        finally:
            await stream.close()
        return "".join(parts).strip(), cut

    async def prewarm(self, messages: list[dict], model: str | None = None,
                      tools: list[dict] | None = None) -> int | None:
        """Have LM Studio read `messages` into its cache now, so the next reply
        that starts with them doesn't wait for the read. Writes one token.
        Returns the prompt's length in tokens."""
        args = self._chat_args([*messages, {"role": "user", "content": "..."}], model, tools)
        args |= {"stream": False, "max_tokens": 1}
        del args["stream_options"]
        reply = await self.aopenai.chat.completions.create(**args)
        return reply.usage.prompt_tokens if reply.usage else None

    async def astream_reply(
        self, messages: list[dict], timing: ReplyTiming, model: str | None = None,
        tools: list[dict] | None = None, calls: list[dict] | None = None, tool_choice: str | None = None,
    ) -> AsyncIterator[str]:
        """The async version of `stream_reply`. With `tools`, any tool calls the
        model makes are added to `calls` as {"id", "name", "arguments"}."""
        t0 = time.perf_counter()
        try:
            for _ in range(self.EMPTY_REPLY_ATTEMPTS):
                stream = await self.aopenai.chat.completions.create(
                    **self._chat_args(messages, model, tools, tool_choice))
                got_text = False
                gathered: dict[int, dict] = {}
                try:
                    async for chunk in stream:
                        self._tool_calls(chunk, gathered)
                        if text := self._content(chunk, timing, t0):
                            got_text = True
                            yield text
                finally:
                    await stream.close()
                if gathered and calls is not None:
                    calls.extend(gathered[i] for i in sorted(gathered))
                if got_text or gathered:
                    return
        finally:
            timing.total_ms = (time.perf_counter() - t0) * 1000

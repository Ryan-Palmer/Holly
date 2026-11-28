"""LM Studio client: model loading over its REST API, chat over its OpenAI API."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI, BadRequestError, OpenAI

from .config import LlmConfig


CHECK_TIMEOUT_S = 5.0  # for LM Studio to say what's loaded (~2 ms when it's well)


def first_word_odds(choice, text: str) -> dict[str, float] | None:
    """How sure the model was of the first word of its answer `text`, from LM Studio's
    logprobs: each word it weighed there and its chance, most likely first, a word in other
    cases or spacing added together ("YES", "Yes" and " yes" are one). None if there are none."""
    logprobs = getattr(choice, "logprobs", None)
    for t in (logprobs.content or []) if logprobs else []:
        word = t.token.strip().upper()
        if word and text.upper().startswith(word):  # the answer's own first word
            odds: dict[str, float] = {}
            for alt in t.top_logprobs or [t]:
                if w := alt.token.strip().upper():
                    odds[w] = odds.get(w, 0.0) + math.exp(alt.logprob)
            return {w: round(p, 4) for w, p in sorted(odds.items(), key=lambda x: -x[1])}
    return None


@dataclass
class ReplyTiming:
    """Milliseconds from sending the request."""

    first_token_ms: float | None = None
    total_ms: float | None = None
    reasoning_chars: int = 0
    reasoning: str = ""  # its last 2,000 characters, kept to show what came instead of a reply
    # From LM Studio, for the last request: the prompt, and what it wrote.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LmStudio:
    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.base_url, timeout=300)
        self.openai = OpenAI(base_url=f"{cfg.base_url}/v1", api_key="lm-studio")
        self.aopenai = AsyncOpenAI(base_url=f"{cfg.base_url}/v1", api_key="lm-studio")
        self._loading = threading.Lock()  # a reply's check and a background job's, not both loading
        # The evals' window on a yes or no (scenarios.memory_check, side_checks): with keep_odds
        # on, each ask also brings back the odds of its answer's first word, in `odds`.
        self.keep_odds = False
        self.odds: dict[str, float] | None = None

    # Model management -------------------------------------------------

    def loaded_instances(self, model: str, timeout: float | None = None) -> list[dict]:
        """`timeout`: seconds to wait for LM Studio to answer, if not the client's 300 (made
        for loading a model)."""
        resp = self.http.get("/api/v1/models", **({"timeout": timeout} if timeout else {}))
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
        self, model: str | None = None, context_length: int | None = None, fresh: bool = False,
        check_timeout: float | None = None,
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
        Returns True if a load happened. `check_timeout`: how long to wait for LM Studio to
        say what's loaded (CHECK_TIMEOUT_S for the checks before a reply and after a failure).
        """
        chat = context_length is None
        model = model or self.cfg.model
        want = {"context_length": context_length or self.cfg.context_length}
        if chat:
            want |= {"parallel": self.cfg.parallel, "eval_batch_size": self.cfg.eval_batch_size}
        with self._loading:
            instances = self.loaded_instances(model, check_timeout)
            if not fresh and any(all(i["config"].get(k) == v for k, v in want.items()) for i in instances):
                return False
            if instances:
                self.unload(model)
            self.http.post(
                "/api/v1/models/load", json={"model": model, **want, **({"flash_attention": True} if chat else {})}
            ).raise_for_status()
            return True

    async def keep_loaded(self, model: str | None = None) -> bool:
        """The chat model, still loaded as Holly loaded it: checked before each reply and
        background job (~2 ms), and loaded again if not (~10 s). True if it was.

        When LM Studio's engine crashes, its next request loads the model again, with LM
        Studio's defaults: on 25 November, four slots of 8,192 tokens, not one of 30,208.
        Every prompt longer than that was cut down to fit, with no error. He lost the
        crew's question, and the forecasts he'd just looked up, and answered the day's
        first "Hey Holly" instead ("I'm here. What's up?"), turn after turn, for the
        rest of the afternoon."""
        return await asyncio.to_thread(self.ensure_loaded, model, None, False, CHECK_TIMEOUT_S)

    # Chat ------------------------------------------------------------

    def _reasoning(self, model: str | None, reasoning: str) -> tuple[str, int]:
        """The effort to send for `reasoning`, and the tokens to add for thinking. A model
        that can't stop thinking (its table's least_thinking) thinks at its least when asked
        not to, with thinking_tokens more to think in: sent "none", Inkling 2 Mini thought
        at its default, high, and a 4-token yes or no came back empty, all of it thinking."""
        cfg = self.cfg if model in (None, self.cfg.model) else self.cfg.for_model(model)
        if reasoning == "none" and cfg.least_thinking:
            return cfg.least_thinking, cfg.thinking_tokens
        return reasoning, 0

    def _chat_args(self, messages: list[dict], model: str | None, tools: list[dict] | None = None,
                   tool_choice: str | None = None) -> dict:
        effort, more = self._reasoning(model, self.cfg.reasoning_effort)
        args = dict(
            model=model or self.cfg.model,
            messages=messages,
            stream=True,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens + more,
            reasoning_effort=effort,
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
            timing.reasoning = (timing.reasoning + reasoning)[-2000:]
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
        effort, more = self._reasoning(model, reasoning)
        return dict(model=model or self.cfg.model, temperature=temperature, max_tokens=max_tokens + more,
                    reasoning_effort=effort, messages=[{"role": "system", "content": system},
                                                         {"role": "user", "content": user}])

    def ask(self, system: str, user: str, model: str | None = None, max_tokens: int = 200,
            temperature: float = 0.2, reasoning: str = "none", schema: dict | None = None) -> str:
        """A whole reply, blocking: call it from a worker thread. With `schema`, the reply is
        JSON in that shape (LM Studio's structured output), for a judge with several answers."""
        args = self._side_args(system, user, model, max_tokens, temperature, reasoning)
        if schema:
            args["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "strict": True, "schema": schema}}
        extra = dict(logprobs=True, top_logprobs=10) if self.keep_odds and not schema else {}
        try:
            try:
                reply = self.openai.chat.completions.create(**args, **extra)
            except BadRequestError:
                if not extra:
                    raise
                extra = {}  # a model that won't give odds still answers: the check, without them
                reply = self.openai.chat.completions.create(**args)
        except Exception:
            self._load_after_failure(model)
            raise
        text = (reply.choices[0].message.content or "").strip()
        if self.keep_odds:
            self.odds = first_word_odds(reply.choices[0], text) if extra else None
        return text

    def _load_after_failure(self, model: str | None) -> None:
        """A request failed: LM Studio's engine may have crashed. The model's loaded again
        now, as he loads it, before any other request (a retry, his reply after a side task
        that failed quietly) has LM Studio load it with its defaults (see keep_loaded). Asked
        with a short timeout: a hung LM Studio had him wait its 300 s more before saying the
        reply had failed."""
        try:
            self.ensure_loaded(model, check_timeout=CHECK_TIMEOUT_S)
        except Exception:
            pass  # LM Studio's gone, or not answering: the request's own error says why

    async def aask(self, system: str, user: str, model: str | None = None, max_tokens: int = 800,
                   temperature: float = 0.2, reasoning: str = "none") -> tuple[str, bool]:
        """A whole reply, streamed, so cancelling it stops LM Studio writing. Returns
        (text, cut): cut if it ran out of tokens."""
        parts: list[str] = []
        cut = False
        try:
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
        except Exception:
            await asyncio.to_thread(self._load_after_failure, model)
            raise
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
        except Exception:
            await asyncio.to_thread(self._load_after_failure, model)
            raise
        finally:
            timing.total_ms = (time.perf_counter() - t0) * 1000

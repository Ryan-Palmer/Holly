"""Shared test helpers: a config whose files are all in a temporary folder, a stand-in
for LM Studio, a silent voice and speakers, and a Holly built from them.

Tests marked `lmstudio` talk to the real LM Studio and are skipped by default; run
them with `pytest -m lmstudio`, with LM Studio up and no Holly running."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re

import numpy as np
import pytest

from app.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """holly.toml's settings, with every file of his in tmp_path, no ship's records,
    no embedding model (keywords only), no Spotify, and research unpaced."""
    c = load_config()
    return dataclasses.replace(
        c,
        persona=dataclasses.replace(c.persona, memory_file=tmp_path / "memory.json"),
        recall=dataclasses.replace(c.recall, file=tmp_path / "archive.db", records_file=tmp_path / "records.db",
                                   embed_model=""),
        timers=dataclasses.replace(c.timers, file=tmp_path / "timers.json"),
        research=dataclasses.replace(c.research, dir=tmp_path / "research", pause_s=0.0),
        spotify=dataclasses.replace(c.spotify, enabled=False),
    )


class FakeLm:
    """Stands in for app.lmstudio.LmStudio.

    `replies`: what each of Holly's replies (astream_reply) gives, in turn: a string he
    says; a list of tool calls, [(name, args)]; a (text, calls) pair; or an exception,
    raised after a few words. `side(system, user)`: the answer to any side task (the
    recall judge, search, research steps), from its system prompt and user message."""

    def __init__(self, replies=(), side=None):
        self.replies = list(replies)
        self.side = side or (lambda system, user: "")
        self.requests: list[dict] = []  # each reply's request: messages, tools, tool_choice
        self.side_requests: list[dict] = []
        self.prewarms = 0

    async def astream_reply(self, messages, timing, model=None, tools=None, calls=None, tool_choice=None):
        self.requests.append(dict(messages=messages, tools=tools, tool_choice=tool_choice))
        reply = self.replies.pop(0) if self.replies else "Nothing more to say."
        if isinstance(reply, BaseException):
            yield "Let me "
            raise reply
        text, tool_calls = (reply, []) if isinstance(reply, str) else reply if isinstance(reply, tuple) else ("", reply)
        for word in re.findall(r"\S+\s*", text):
            yield word
        if tool_calls and calls is not None:
            n = len(self.requests)
            calls.extend({"id": f"call{n}_{i}", "name": name, "arguments": json.dumps(args)}
                         for i, (name, args) in enumerate(tool_calls))

    def ask(self, system, user, model=None, max_tokens=200, temperature=0.2, reasoning="none"):
        self.side_requests.append(dict(system=system, user=user, model=model, max_tokens=max_tokens,
                                       temperature=temperature, reasoning=reasoning))
        return self.side(system, user).strip()

    async def aask(self, system, user, model=None, max_tokens=800, temperature=0.2, reasoning="none"):
        return self.ask(system, user, model, max_tokens, temperature, reasoning), False

    async def prewarm(self, messages, model=None, tools=None):
        self.prewarms += 1
        return 1000

    def embed(self, texts, model):
        raise ConnectionError("no embedding model in tests")

    def ensure_loaded(self, *a, **k):
        return False

    def loaded_models(self):
        return set()


class SilentVoice:
    """The voice, without sound: records each sentence it was given."""
    sample_rate = 48000

    def __init__(self):
        self.said: list[str] = []

    async def stream(self, text, trim_lead=False):
        self.said.append(text)
        yield np.zeros(480, np.float32)

    async def warm_up(self):
        pass


class SilentPlayer:
    """The speakers, without sound."""
    device_name = "nowhere"
    first_played = first_voice = last_played = None
    underruns = 0

    def begin_turn(self):
        self.done = asyncio.Event()

    def put(self, audio):
        pass

    def expect_gap(self):
        pass

    def end_turn(self):
        self.done.set()

    def wake(self):
        pass

    def close(self):
        pass

    async def wait_done(self):
        await self.done.wait()


@pytest.fixture
def make_holly(cfg, monkeypatch):
    """make_holly(lm) -> a Holly on `cfg`, talking to `lm`, with a silent voice."""
    from app import speak

    def make(lm, **kwargs):
        monkeypatch.setattr(speak.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(speak, "LmStudio", lambda llm_cfg: lm)
        holly = speak.Holly(kwargs.pop("model", None), face=False, cfg=kwargs.pop("cfg", cfg), **kwargs)
        holly.tts, holly.player = SilentVoice(), SilentPlayer()
        holly.background.reminded = True
        return holly

    return make


@pytest.fixture(scope="session")
def lmstudio():
    """For tests that use the real LM Studio: it must be up, with no Holly using it."""
    from app.lmstudio import LmStudio
    from app.scenarios import holly_running

    if running := holly_running():
        pytest.exit(f"Holly is running ({running[0][:80]}): close him before these tests", returncode=1)
    lm = LmStudio(load_config().llm)
    try:
        lm.loaded_models()
    except Exception as e:
        pytest.skip(f"LM Studio isn't answering: {e!r}")
    return lm

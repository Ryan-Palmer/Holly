"""The LM Studio client's model loading: kept as Holly loaded it."""

import asyncio
import json

import httpx
import pytest

from app.config import LlmConfig
from app.lmstudio import LmStudio, ReplyTiming

CFG = LlmConfig(base_url="http://lm", model="gemma", context_length=30208, parallel=1, eval_batch_size=2048,
                reasoning_effort="none", temperature=0.7, max_tokens=450)
OURS = {"context_length": 30208, "parallel": 1, "eval_batch_size": 2048, "flash_attention": True}
# What LM Studio loaded by itself after its engine crashed, on 25 November.
ITS_OWN = {"context_length": 8192, "parallel": 4, "eval_batch_size": 2048, "flash_attention": True}


def client(loaded: list[dict]) -> tuple[LmStudio, list[tuple[str, dict]]]:
    """A client whose LM Studio has `loaded` (instance configs) of the model, and the
    loads and unloads it was asked for."""
    asked: list[tuple[str, dict]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            instances = [{"id": f"gemma:{i}", "config": c} for i, c in enumerate(loaded)]
            return httpx.Response(200, json={"models": [{"key": "gemma", "loaded_instances": instances}]})
        body = json.loads(request.content)
        asked.append((request.url.path, body))
        if request.url.path.endswith("/load"):
            loaded[:] = [{k: v for k, v in body.items() if k != "model"}]
        else:
            loaded.clear()
        return httpx.Response(200, json={})

    lm = LmStudio(CFG)
    lm.http = httpx.Client(base_url=CFG.base_url, transport=httpx.MockTransport(handle))
    return lm, asked


def test_the_model_loaded_as_he_loads_it_is_left_alone():
    lm, asked = client([dict(OURS)])
    assert asyncio.run(lm.keep_loaded()) is False
    assert asked == []


def test_the_model_lm_studio_loaded_its_own_way_is_loaded_again_as_his():
    lm, asked = client([dict(ITS_OWN)])
    assert asyncio.run(lm.keep_loaded()) is True
    assert [path for path, _ in asked] == ["/api/v1/models/unload", "/api/v1/models/load"]
    assert asked[1][1] == {"model": "gemma", **OURS}


def test_a_model_that_isnt_loaded_is_loaded_as_his():
    lm, asked = client([])
    assert asyncio.run(lm.keep_loaded()) is True
    assert asked == [("/api/v1/models/load", {"model": "gemma", **OURS})]


def test_a_failed_reply_has_the_model_loaded_again_before_anything_else_asks(monkeypatch):
    """The engine crashed mid-reply: his model's loaded his way before the retry, or a
    side task, has LM Studio load it with its defaults."""
    lm, asked = client([])  # the crash took it with it

    async def crash(**kwargs):
        raise RuntimeError("Engine protocol predict request failed: fetch failed")

    monkeypatch.setattr(lm.aopenai.chat.completions, "create", crash)

    async def reply():
        return [t async for t in lm.astream_reply([{"role": "user", "content": "Hey Holly"}], ReplyTiming())]

    with pytest.raises(RuntimeError, match="fetch failed"):
        asyncio.run(reply())
    assert asked == [("/api/v1/models/load", {"model": "gemma", **OURS})]


def test_a_failed_side_task_has_the_model_loaded_again_too(monkeypatch):
    """A side task (the memory check, a research step) that fails quietly: else his reply,
    the next request, would have LM Studio load it with its defaults."""
    lm, asked = client([dict(ITS_OWN)])

    def crash(**kwargs):
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(lm.openai.chat.completions, "create", crash)
    with pytest.raises(RuntimeError):
        lm.ask("Sort this.", "They say: hello")
    assert [path for path, _ in asked] == ["/api/v1/models/unload", "/api/v1/models/load"]

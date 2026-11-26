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


def test_what_is_loaded_is_asked_with_a_short_timeout():
    """A hung LM Studio had him wait the client's 300 s more, after a reply had failed,
    before saying so: the checks before a reply and after a failure wait CHECK_TIMEOUT_S."""
    from app.lmstudio import CHECK_TIMEOUT_S
    waits: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        waits.append(request.extensions["timeout"]["read"])
        instances = [{"id": "gemma:0", "config": dict(OURS)}]
        return httpx.Response(200, json={"models": [{"key": "gemma", "loaded_instances": instances}]})

    lm = LmStudio(CFG)
    lm.http = httpx.Client(base_url=CFG.base_url, timeout=300, transport=httpx.MockTransport(handle))
    asyncio.run(lm.keep_loaded())
    lm._load_after_failure(None)
    lm.ensure_loaded()  # at startup: the client's own
    assert waits == [CHECK_TIMEOUT_S, CHECK_TIMEOUT_S, 300]


def test_a_hung_lm_studio_after_a_failure_is_left_be():
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    lm = LmStudio(CFG)
    lm.http = httpx.Client(base_url=CFG.base_url, transport=httpx.MockTransport(handle))
    lm._load_after_failure(None)  # no exception: the request's own error says why


def _answer(text: str, alts: list[tuple[str, float]], first: str | None = None):
    """A chat completion as LM Studio gives one with logprobs: `alts`, the first word's top
    candidates and their chances."""
    import math
    from types import SimpleNamespace as NS
    top = [NS(token=t, logprob=math.log(p)) for t, p in alts]
    tokens = [NS(token=first or alts[0][0], logprob=top[0].logprob, top_logprobs=top)] if alts else []
    return NS(choices=[NS(message=NS(content=text), logprobs=NS(content=tokens) if alts else None)])


def test_how_sure_the_first_word_was_adds_its_spellings_together():
    from app.lmstudio import first_word_odds
    odds = first_word_odds(_answer("YES", [("YES", 0.6), ("NO", 0.3), (" yes", 0.1)]).choices[0], "YES")
    assert odds == {"YES": 0.7, "NO": 0.3} and list(odds) == ["YES", "NO"]
    assert first_word_odds(_answer("YES", []).choices[0], "YES") is None  # no logprobs given


def test_a_side_check_asks_for_odds_only_when_the_evals_keep_them():
    from types import SimpleNamespace as NS
    sent: list[dict] = []

    def create(**kwargs):
        sent.append(kwargs)
        return _answer("PAST", [("PAST", 0.55), ("OTHER", 0.45)])

    lm = LmStudio(CFG)
    lm.openai = NS(chat=NS(completions=NS(create=create)))
    assert lm.ask("PAST or OTHER?", "What did I have for lunch?", max_tokens=4) == "PAST"
    assert "logprobs" not in sent[-1] and lm.odds is None  # Holly himself: as ever
    lm.keep_odds = True
    lm.ask("PAST or OTHER?", "What did I have for lunch?", max_tokens=4)
    assert sent[-1]["logprobs"] is True and lm.odds == {"PAST": 0.55, "OTHER": 0.45}


def test_a_model_that_wont_give_odds_still_answers_the_check():
    """The odds are extra: refused, the check is asked again without them, not lost as an error."""
    from types import SimpleNamespace as NS
    from openai import BadRequestError
    sent: list[dict] = []

    def create(**kwargs):
        sent.append(kwargs)
        if kwargs.get("logprobs"):
            raise BadRequestError("no logprobs here", body=None,
                                  response=httpx.Response(400, request=httpx.Request("POST", "http://lm/v1")))
        return _answer("YES", [])

    lm = LmStudio(CFG)
    lm.openai = NS(chat=NS(completions=NS(create=create)))
    lm.keep_odds = True
    assert lm.ask("YES or NO?", "Yes please.", max_tokens=4) == "YES" and lm.odds is None
    assert [bool(s.get("logprobs")) for s in sent] == [True, False]

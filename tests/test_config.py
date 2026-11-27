"""holly.toml: each chat model's profile, and the one way side tasks reach the model."""

import asyncio
from types import SimpleNamespace

import pytest

from app.config import _llm_config, load_config
from app.lmstudio import LmStudio
from conftest import FakeLm

LLM = dict(base_url="http://127.0.0.1:1234", model="a", context_length=30208, parallel=1, eval_batch_size=2048,
           reasoning_effort="none", temperature=0.8, max_tokens=300)


def test_profiles_override_llm():
    cfg = _llm_config(LLM, {"a": {"stray_words": ["thought"]}, "b": {"context_length": 16384, "tool_choice": False}})
    assert cfg.model == "a" and cfg.stray_words == ("thought",) and cfg.context_length == 30208
    b = cfg.for_model("b")
    assert (b.model, b.context_length, b.tool_choice, b.stray_words) == ("b", 16384, False, ())  # not a's quirks
    other = cfg.for_model("c")  # no table: [llm] alone
    assert (other.model, other.stray_words, other.tool_choice) == ("c", (), True)
    assert cfg.for_model(None) == cfg


def test_a_profile_cant_set_anything_else():
    with pytest.raises(ValueError, match="can't set base_url"):
        _llm_config(LLM, {"a": {"base_url": "elsewhere"}})


def test_holly_toml_loads():
    cfg = load_config()
    assert cfg.llm.model in cfg.llm.profiles and "thought" in cfg.llm.stray_words


def test_holly_uses_his_models_profile(make_holly, cfg):
    holly = make_holly(FakeLm(), model="google/gemma-4-31b-qat")
    assert holly.model == holly.cfg.llm.model == "google/gemma-4-31b-qat"
    assert holly.stray_words == {"thought"}


def test_without_tool_choice_a_redo_is_just_asked_again(make_holly, cfg):
    import dataclasses

    lm = FakeLm(["I'll look into it while I doze.", "No idea, sorry."])
    no_choice = dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, tool_choice=False,
                                                                 profiles={"": {"tool_choice": False}}))
    holly = make_holly(lm, cfg=no_choice)
    asyncio.run(holly.respond("Can you research slugs for me?"))
    assert [r["tool_choice"] for r in lm.requests] == [None, None]


class Recorder:
    """The OpenAI client's chat.completions, recording what it's sent."""

    def __init__(self):
        self.sent = []

    def create(self, **kw):
        self.sent.append(kw)
        if kw.get("stream"):
            async def chunks():
                for text, end in (("Hello", None), (" there", "length")):
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text),
                                                                   finish_reason=end)])

            class Stream:
                def __aiter__(self):
                    return chunks()

                async def close(self):
                    pass

            async def opened():
                return Stream()
            return opened()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  An answer. "))])


def test_side_tasks():
    lm = LmStudio(load_config().llm)
    rec = Recorder()
    lm.openai = lm.aopenai = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    assert lm.ask("Be brief.", "Why?", max_tokens=50) == "An answer."
    assert rec.sent[0]["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Why?"}]
    assert (rec.sent[0]["model"], rec.sent[0]["reasoning_effort"]) == (lm.cfg.model, "none")
    assert asyncio.run(lm.aask("Be brief.", "Why?", "m", 10, temperature=0.0)) == ("Hello there", True)
    assert rec.sent[1]["stream"] and rec.sent[1]["model"] == "m" and rec.sent[1]["temperature"] == 0.0


def test_a_model_that_cant_stop_thinking_thinks_least_with_room_to_answer():
    """Inkling 2 Mini, sent "none", thought at its default, high: a 4-token yes or no came
    back empty. Its table's least effort is sent instead, with room to think in."""
    cfg = _llm_config(LLM, {"inkling": {"least_thinking": "low", "thinking_tokens": 1500}})
    gemma = LmStudio(cfg)
    side = gemma._side_args("Answer YES or NO.", "Is Keswick in England?", "inkling", 4, 0.0, "none")
    assert (side["reasoning_effort"], side["max_tokens"]) == ("low", 1504)
    asked = gemma._side_args("", "", "inkling", 12000, 0.3, "medium")  # a judge told to think: as asked
    assert (asked["reasoning_effort"], asked["max_tokens"]) == ("medium", 12000)
    own = gemma._side_args("", "", None, 4, 0.0, "none")  # a model with an "off": unchanged
    assert (own["reasoning_effort"], own["max_tokens"]) == ("none", 4)
    inkling = LmStudio(cfg.for_model("inkling"))  # his replies, on it
    chat = inkling._chat_args([{"role": "user", "content": "Morning"}], None)
    assert (chat["reasoning_effort"], chat["max_tokens"]) == ("low", 300 + 1500)

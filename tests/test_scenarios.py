"""Holly on the real model (app/scenarios.py): skipped unless run with `pytest -m lmstudio`,
with LM Studio up and no Holly running. Gemma varies: a failure now and then is worth a
second run before it's taken for a regression."""

import asyncio

import pytest

from app import scenarios

pytestmark = pytest.mark.lmstudio


@pytest.fixture(scope="module")
def loaded(lmstudio):
    from app.config import load_config

    from app.evals import alone

    cfg = load_config()
    alone(cfg.llm.model)  # with anything else unloaded: a judge left from the evals, say
    try:
        lmstudio.ensure_loaded(cfg.recall.embed_model, context_length=2048)
    except Exception:  # recall by keyword and day only
        pass


@pytest.mark.parametrize("scenario", scenarios.SCENARIOS, ids=[s.name for s in scenarios.SCENARIOS])
def test_scenario(loaded, tmp_path, scenario):
    holly = scenarios.quiet_holly(tmp=tmp_path)
    if scenario.records and not holly.records:
        pytest.skip("no ship's records here")
    scenarios.seed(holly)
    turns, wrong = asyncio.run(scenarios.run(holly, scenario))
    assert not wrong, f"{wrong}: " + " / ".join(f"{t.said!r} -> {t.reply!r} {t.tools}" for t in turns)


def test_ordinal():
    from app.scenarios import ordinal
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 30)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "30th"]

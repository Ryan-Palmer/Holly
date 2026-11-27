"""Memories offered, on the real model (archive.memory_offered): is one of the crew's lines
from other days about the same thing as what's just been said? Each labelled case in
scenarios.MEMORY_OFFERS, asked twice. Skipped unless run with `pytest -m lmstudio`, with LM
Studio up and no Holly running."""

import pytest

from app import scenarios

pytestmark = pytest.mark.lmstudio


@pytest.fixture(scope="module")
def lm(lmstudio):
    from app.config import load_config
    from app.evals import alone

    alone(load_config().llm.model)
    return lmstudio


@pytest.mark.parametrize("line, said, which", scenarios.MEMORY_OFFERS,
                         ids=[f"{w or 'NONE'}: {line[:50]}" for line, _, w in scenarios.MEMORY_OFFERS])
def test_memory_offered(lm, line, said, which):
    from app.archive import memory_offered

    found = [(k, "Tuesday 24 November", s) for k, s in enumerate(said, 1)]
    got = [memory_offered(lm, lm.cfg.model, line, found) for _ in range(2)]
    assert got == [which, which], f"wanted {which}, got {got}"

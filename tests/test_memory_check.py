"""The memory check (app/archive.py's about_the_past) on the real model: does a line ask
Holly to remember something? Each labelled line in scenarios.MEMORY_LINES, asked twice.
Skipped unless run with `pytest -m lmstudio`, with LM Studio up and no Holly running."""

import pytest

from app import scenarios

pytestmark = pytest.mark.lmstudio


@pytest.fixture(scope="module")
def lm(lmstudio):
    from app.config import load_config

    lmstudio.ensure_loaded(load_config().llm.model)
    return lmstudio


@pytest.mark.parametrize("line, before, past", scenarios.MEMORY_LINES,
                         ids=[f"{'PAST' if p else 'OTHER'}: {line[:50]}" + (" (after talk)" if b else "")
                              for line, b, p in scenarios.MEMORY_LINES])
def test_memory_check(lm, line, before, past):
    from app.archive import about_the_past

    got = [about_the_past(lm, lm.cfg.model, line, before) for _ in range(2)]
    assert got == [past, past], f"wanted {'PAST' if past else 'OTHER'}, got {got}"

"""The side checks on the real model: is a line asking for a status report
(news.wants_report), and is a timer come due a job for one of his tools (timers.to_do)?
Each labelled case in scenarios.STATUS_LINES and scenarios.ACTION_LABELS, asked twice.
Skipped unless run with `pytest -m lmstudio`, with LM Studio up and no Holly running."""

import pytest

from app import scenarios

pytestmark = pytest.mark.lmstudio


@pytest.fixture(scope="module")
def lm(lmstudio):
    from app.config import load_config

    lmstudio.ensure_loaded(load_config().llm.model)
    return lmstudio


@pytest.mark.parametrize("line, wanted", scenarios.STATUS_LINES,
                         ids=[f"{'YES' if w else 'NO'}: {line[:50]}" for line, w in scenarios.STATUS_LINES])
def test_status_report_asked(lm, line, wanted):
    """"My boss wants a status update by five" had him made to give the news."""
    from app.news import wants_report

    assert [wants_report(lm, lm.cfg.model, line, quiet=False) for _ in range(2)] == [wanted, wanted]


@pytest.mark.parametrize("label, wanted", scenarios.ACTION_LABELS,
                         ids=[f"{'DO' if w else 'SAY'}: {label[:50]}" for label, w in scenarios.ACTION_LABELS])
def test_timer_job(lm, label, wanted):
    """"Go to Jo's school play" would have had him made to call a tool (music, at worst)."""
    from app.timers import to_do

    assert [to_do(lm, lm.cfg.model, [label], quiet=False) for _ in range(2)] == [wanted, wanted]

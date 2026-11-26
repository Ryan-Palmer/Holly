"""The side checks on the real model: is a line asking for a status report
(status.wants_report), is a timer come due a job for one of his tools (timers.to_do), and are
the crew taking him up on something he offered (offers.taken_up)?
Each labelled case in scenarios.STATUS_LINES, ACTION_LABELS and TAKEN_UP_LINES, asked twice.
Skipped unless run with `pytest -m lmstudio`, with LM Studio up and no Holly running."""

import pytest

from app import scenarios

pytestmark = pytest.mark.lmstudio


@pytest.fixture(scope="module")
def lm(lmstudio):
    from app.config import load_config
    from app.evals import alone

    alone(load_config().llm.model)  # with anything else unloaded: a judge left from the evals, say
    return lmstudio


@pytest.mark.parametrize("line, wanted", scenarios.STATUS_LINES,
                         ids=[f"{'YES' if w else 'NO'}: {line[:50]}" for line, w in scenarios.STATUS_LINES])
def test_status_report_asked(lm, line, wanted):
    """"My boss wants a status update by five" had him made to give the news."""
    from app.status import wants_report

    assert [wants_report(lm, lm.cfg.model, line, quiet=False) for _ in range(2)] == [wanted, wanted]


@pytest.mark.parametrize("label, wanted", scenarios.ACTION_LABELS,
                         ids=[f"{'DO' if w else 'SAY'}: {label[:50]}" for label, w in scenarios.ACTION_LABELS])
def test_timer_job(lm, label, wanted):
    """"Go to Jo's school play" would have had him made to call a tool (music, at worst)."""
    from app.timers import to_do

    assert [to_do(lm, lm.cfg.model, [label], quiet=False) for _ in range(2)] == [wanted, wanted]


@pytest.mark.parametrize("pair, wanted", scenarios.TAKEN_UP_LINES,
                         ids=[f"{'YES' if w else 'NO'}: {p[1][:30]} / {p[0][:30]}" for p, w in scenarios.TAKEN_UP_LINES])
def test_an_offer_taken_up(lm, pair, wanted):
    """"Yes, please." to "I can do a proper research job on it if you want.", and "You can get
    it now if you like" to "Want me to search for it now or later?": he promised, and called
    nothing. A question mark and a word, alone, missed the first."""
    from app.offers import taken_up

    said, reply = pair
    assert [taken_up(lm, lm.cfg.model, said, reply, quiet=False) for _ in range(2)] == [wanted, wanted]

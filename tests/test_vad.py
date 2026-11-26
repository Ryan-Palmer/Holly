"""Turn detection (app/vad.py): when a turn ends, with and without the turn-ending model.
Silero and the model are stood in for: what they say is set per test."""

import dataclasses

import numpy as np
import pytest

from app import vad

SPEECH, QUIET = 0.9, 0.05


class FakeSilero:
    """Speech probability per block, in turn; quiet once they run out."""
    def __init__(self):
        self.probs: list[float] = []

    def __call__(self, block, rate):
        p = self.probs.pop(0) if self.probs else QUIET
        return type("P", (), {"item": lambda self_: p})()

    def reset_states(self):
        pass


class FakeTurnModel:
    """Says each pause is finished or not, in turn; records the audio it was given."""
    def __init__(self, *answers: float):
        self.answers, self.heard = list(answers), []

    def finished(self, audio):
        self.heard.append(len(audio))
        return self.answers.pop(0)


@pytest.fixture
def detector(cfg, monkeypatch):
    silero = FakeSilero()
    monkeypatch.setattr(vad, "load_silero_vad", lambda: silero)
    listen = dataclasses.replace(cfg.listen, start_threshold=0.5, end_threshold=0.35, end_silence_ms=400,
                                 min_speech_ms=250, preroll_ms=96, turn_check_ms=200, turn_threshold=0.5,
                                 max_silence_ms=1500)

    def make(turn_model=None, checks=None):
        d = vad.TurnDetector(listen, turn_model, on_check=(lambda p, a: checks.append(p)) if checks is not None
                             else None)
        return d, silero
    return make


def run(d, silero, probs: list[float], blocks: int):
    """Feed `blocks` blocks, Silero saying `probs` then quiet; the block an utterance ended on,
    and it."""
    silero.probs = list(probs)
    for i in range(blocks):
        if u := d.feed(np.zeros(vad.BLOCK, np.float32), float(i)):
            return i, u
    return None, None


def test_without_the_model_a_pause_ends_the_turn(detector):
    d, silero = detector()
    at, u = run(d, silero, [SPEECH] * 10, 60)
    assert at == 10 + 400 // vad.BLOCK_MS  # 13 quiet blocks: 416 ms, past 400
    assert u.duration_ms == 10 * vad.BLOCK_MS and not d.ended_by_model


def test_finished_it_ends_sooner_than_before(detector):
    model, checks = FakeTurnModel(0.9), []
    d, silero = detector(model, checks)
    at, u = run(d, silero, [SPEECH] * 10, 60)
    assert at == 10 + 7 - 1  # asked at 224 ms of quiet, past 200: ended then, not at 416
    assert d.ended_by_model and checks == [0.9] and u.duration_ms == 10 * vad.BLOCK_MS


def test_mid_sentence_it_waits_for_the_rest(detector):
    """"I was thinking we could..." and a pause: 0.01, so he waits, up to max_silence_ms."""
    d, silero = detector(FakeTurnModel(0.01))
    at, u = run(d, silero, [SPEECH] * 10, 80)
    assert at == 10 + -(-1500 // vad.BLOCK_MS) - 1  # 47 quiet blocks: 1504 ms
    assert u and not d.ended_by_model


def test_unsure_it_waits_too(detector):
    """"So, I was thinking..." live, a pause the model wasn't sure of: answered at 400 ms, as
    before, it was cut in on. Being cut off is worse than a slower answer: it waits."""
    d, silero = detector(FakeTurnModel(0.29))
    at, u = run(d, silero, [SPEECH] * 10, 80)
    assert at == 10 + -(-1500 // vad.BLOCK_MS) - 1 and u and not d.ended_by_model


def test_speech_after_a_pause_is_the_same_turn(detector):
    """The rest of the sentence after the pause: one turn, and the new pause asked about afresh."""
    model, checks = FakeTurnModel(0.01, 0.8), []
    d, silero = detector(model, checks)
    at, u = run(d, silero, [SPEECH] * 10 + [QUIET] * 15 + [SPEECH] * 10, 80)
    assert checks == [0.01, 0.8]  # once each pause
    assert at == 35 + 7 - 1 and d.ended_by_model
    assert u.duration_ms == (10 + 15 + 10) * vad.BLOCK_MS  # the pause inside counts as the turn's
    assert model.heard[1] > model.heard[0]  # the whole turn so far, each time


def test_a_cough_isnt_asked_about(detector):
    model = FakeTurnModel()
    d, silero = detector(model)
    at, u = run(d, silero, [SPEECH] * 3, 80)  # 96 ms: under min_speech_ms
    assert model.heard == [] and u is None

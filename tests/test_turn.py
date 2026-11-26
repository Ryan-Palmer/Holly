"""The turn-ending model itself (app/turn.py), if it's been fetched into the Hugging Face
cache (python -m app.turn, or Holly's first start, fetches it): skipped where it hasn't."""

import numpy as np
import pytest

from app.config import load_config


@pytest.fixture(scope="module")
def model():
    from huggingface_hub import hf_hub_download

    spec = load_config().listen.turn_model
    if not spec:
        pytest.skip("no turn model in holly.toml")
    repo, _, name = spec.rpartition("/")
    try:
        hf_hub_download(repo, name, local_files_only=True)
    except Exception:
        pytest.skip("the turn model hasn't been fetched here")
    from app.turn import TurnModel
    return TurnModel(spec)


def test_it_answers_with_a_probability_for_any_length(model):
    rng = np.random.default_rng(0)
    for seconds in (0.5, 3, 12):  # shorter than its 8 s window, and longer
        p = model.finished((rng.standard_normal(int(16000 * seconds)) * 0.01).astype(np.float32))
        assert 0.0 <= p <= 1.0


def test_what_holly_does_at_a_pause(cfg):
    """The probe's sums (app/turn.py), as app/vad.py does."""
    from app.turn import answers_after, cuts_in

    listen = cfg.listen
    assert answers_after(0.9, listen) == listen.turn_check_ms
    assert answers_after(0.29, listen) == answers_after(0.01, listen) == listen.max_silence_ms
    assert cuts_in({"p": 0.01, "pause_ms": 900}, listen) is False  # not sure you'd finished: he waited
    assert cuts_in({"p": 0.29, "pause_ms": 900}, listen) is False
    assert cuts_in({"p": 0.9, "pause_ms": 300}, listen) is True


def test_saved_pauses_are_scored_again(cfg, tmp_path):
    """`--only turns`: the probe's labelled pauses, by the model as it is now."""
    import json
    import wave

    from app.turn import regression

    lines = [dict(file="0001.wav", p=0.01, heard="I was thinking", went_on=True, pause_ms=900),
             dict(file="0002.wav", p=0.9, heard="I was thinking we could go", went_on=False, pause_ms=None)]
    for line in lines:
        with wave.open(str(tmp_path / line["file"]), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes(np.zeros(16000, np.int16).tobytes())
    (tmp_path / "clips.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    class Model:
        def __init__(self, *answers):
            self.answers = list(answers)

        def finished(self, audio):
            return self.answers.pop(0)

    r = regression(tmp_path, cfg.listen, Model(0.02, 0.95))
    assert (r["cut_in"], r["cut_in_before"], r["sooner"], r["slower"], r["moved"]) == (0, 1, 1, 0, 0)
    r = regression(tmp_path, cfg.listen, Model(0.8, 0.2))  # a worse model: cuts in, answers late
    assert (r["cut_in"], r["sooner"], r["slower"], r["moved"]) == (1, 0, 1, 2)

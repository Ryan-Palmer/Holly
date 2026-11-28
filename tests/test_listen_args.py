"""app.listen's own options."""

from conftest import FakeLm


def test_mic_picks_the_microphone_for_this_run(cfg, monkeypatch):
    from app import listen, speak

    monkeypatch.setattr(speak.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(speak, "LmStudio", lambda llm_cfg: FakeLm())
    phone = listen.ListeningHolly(None, face=False, cfg=cfg, mic="Steam Streaming")
    assert phone.cfg.listen.input_device == "Steam Streaming"
    desk = listen.ListeningHolly(None, face=False, cfg=cfg)  # holly.toml's, or Windows' default
    assert desk.cfg.listen.input_device == cfg.listen.input_device

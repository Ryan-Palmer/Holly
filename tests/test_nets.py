"""The reply nets' counts from real use, kept per model between sessions (app/nets.py)."""

from collections import Counter

from app import guards
from app.nets import NetCounts, report
from conftest import FakeLm
from test_reply_loop import respond


def test_counts_add_up_across_sessions_per_model(tmp_path):
    f = tmp_path / "nets.json"
    NetCounts(f).add("google/gemma", Counter({guards.OFFERED: 2}))
    later = NetCounts(f)  # another session
    later.add("google/gemma", Counter({guards.OFFERED: 1, guards.ALONE: 1}))
    later.add("qwen/qwen", Counter())
    counts = later.read()
    assert counts["google/gemma"]["replies"] == 2 and counts["google/gemma"]["nets"] == {guards.OFFERED: 3,
                                                                                         guards.ALONE: 1}
    assert counts["qwen/qwen"] == dict(counts["qwen/qwen"], replies=1, nets={})
    text = report(counts)
    assert "google/gemma: 2 replies" in text and f"{len(guards.NETS) - 2} of {len(guards.NETS)} never fired" in text


def test_an_unreadable_file_is_set_aside_not_lost(tmp_path):
    f = tmp_path / "nets.json"
    f.write_text("{not json", encoding="utf-8")
    counts = NetCounts(f)
    assert counts.read() == {}
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("nets.unreadable")]
    counts.add("google/gemma", Counter({guards.OFFERED: 1}))
    assert counts.read()["google/gemma"]["nets"] == {guards.OFFERED: 1}


def test_hollys_turns_are_counted_beside_his_memory(make_holly, cfg):
    """Real use: each turn's nets, and the reply, kept per model."""
    holly = make_holly(FakeLm(["Fifty-six. Anything else you need me to work out?", "Thrilling stuff."]))
    respond(holly, "What's seven times eight?", "Kettle's on.")
    counts = NetCounts(cfg.persona.memory_file.with_name("nets.json")).read()
    assert counts[holly.model]["replies"] == 2 and counts[holly.model]["nets"] == {guards.OFFERED: 1}


def test_raw_turns_arent_kept_with_real_uses(make_holly, cfg):
    holly = make_holly(FakeLm(["Fifty-six. Anything else you need me to work out?"]))
    holly.nets_act = False
    respond(holly, "What's seven times eight?")
    assert holly.nets == {guards.OFFERED: 1}
    assert not cfg.persona.memory_file.with_name("nets.json").exists()

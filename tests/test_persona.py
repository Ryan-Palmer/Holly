"""His conversation: the prompt he's sent, his history, its compacting, and keeping it on disk."""

import json

from app.persona import IN_CHARACTER, SPEAKING_UP, Conversation


def convo(cfg, memory_file=None) -> Conversation:
    return Conversation(cfg.persona, "Some tool rules.", memory_file)


def say(c: Conversation, n: int, crew: str = "Tell me something.", holly: str = "Something.") -> None:
    for _ in range(n):
        c.record(crew, holly)


def test_prompt_has_rules_examples_and_date(cfg):
    c = convo(cfg)
    system = c.messages_for("Hello.")[0]["content"]
    assert "Some tool rules." in system and "Examples of how you talk" in system and "Today is " in system
    assert c.messages_for("Hello.")[-1] == {"role": "user", "content": f"Hello.\n{IN_CHARACTER}"}


def test_in_character_note_in_the_request_only(cfg):
    c = convo(cfg)
    c.record("Hi.", "Hello.")
    assert c.messages_for("Again?")[-1]["content"].endswith(IN_CHARACTER)
    assert all(IN_CHARACTER not in (m.get("content") or "") for m in c.messages_for())  # none without a new line
    assert all("Stay yourself" not in (m.get("content") or "") for m in c.history)


def test_speaking_up_has_its_own_note(cfg):
    c = convo(cfg)
    last = c.messages_for("[The timer 'pizza' is due now. Tell the crew, in a sentence.]")[-1]["content"]
    assert last.endswith(SPEAKING_UP) and IN_CHARACTER not in last  # not told "no offers" against the note


def test_offers_of_more_help_left_out_of_what_he_sees(cfg):
    c = convo(cfg)
    c.record("Coal prices?", "Up again. Anything else you need me to dig up?")
    c.record("Thanks.", "Anything else?")  # all there is: kept
    sent = [m["content"] for m in c.messages_for() if m["role"] == "assistant"]
    assert sent == ["Up again.", "Anything else?"]
    assert c.history[1]["content"] == "Up again. Anything else you need me to dig up?"  # his history as it was


def test_worn_phrases_noted_in_the_request_only(cfg):
    c = convo(cfg)
    c.record("One?", "Well, that's a bit odd, isn't it.")
    c.record("Two?", "Well, that's a bit much really.")
    c.record("Three?", "No.")
    worn = c.worn_phrases()
    assert worn[0] == "well that's a"  # an opening, three words
    assert "that's a bit" not in worn  # inside the opening already
    last = c.messages_for("Four?")[-1]["content"]
    assert last.startswith("Four?\n[Lately you've said these more than once")
    assert all("Lately" not in (m.get("content") or "") for m in c.history)


def test_messages_leave_out_private_keys(cfg):
    c = convo(cfg)
    c.record("Hi.", "Hello.")
    assert all("at" not in m for m in c.messages_for())


def test_hard_limit_drops_the_oldest_in_one_go(cfg):
    c = convo(cfg)
    c.hard_chars = 1000
    say(c, 8, "x" * 100, "y" * 100)  # 1,600 characters
    assert c.tokens() * 4 <= 1000
    assert c.history[0]["role"] == "user"


def test_compaction_split_and_apply(cfg):
    c = convo(cfg)
    say(c, 10, "a" * 400, "b" * 400)
    c.soft_chars, c.keep_chars = 4000, 2000
    assert c.needs_compacting()
    end = c.compaction_split()
    assert c.history[end]["role"] == "user" and c._chars(end) <= 2000
    dropped = c.history[:end]
    c.history.append({"role": "user", "content": "new"})  # said meanwhile: still applies
    assert c.apply_compaction(end, dropped, "About the crew:\n- They type a lot.")
    assert c.memory.startswith("About the crew") and "notes on your earlier" in c.messages_for()[0]["content"]
    # The history moved on (it was reset meanwhile): the notes aren't applied.
    c.history = [{"role": "user", "content": "x"}] * 5
    assert not c.apply_compaction(2, dropped[:2], "notes")


def test_transcript_cuts_recall_to_the_crews_line(cfg):
    c = convo(cfg)
    c.record("Remember Jo? She's coming on Friday now.", "Jo. The one who robbed the Cat.", steps=[
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "recall", "arguments": '{"query": "Jo"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Jo is the crew's sister; the Cat ate her sandwich."}])
    c.record("What's 2 and 2?", "Five.")
    tx = c.transcript(len(c.history))
    assert "Crew: Remember Jo? She's coming on Friday" in tx
    assert "sandwich" not in tx and "Holly: Five." in tx
    assert "sandwich" in c.exchanges(len(c.history))[0][1]  # the archive still gets it all


def test_crew_lines_skip_his_notes(cfg):
    c = convo(cfg)
    c.record("[The timer 'pasta': it's due now.]", "Pasta's ready.")
    c.record("Jo's coming over.", "Noted.")
    assert c.crew_lines().endswith("Jo's coming over.") and "pasta" not in c.crew_lines()
    assert c.recent_topics() == ["Jo's coming over."]


def test_kept_on_disk(cfg, tmp_path):
    f = tmp_path / "memory.json"
    c = convo(cfg, f)
    c.record("Hi.", "Hello.")
    c.memory = "About the crew:\n- Likes tea."
    c.save()
    again = convo(cfg, f)
    assert again.history == c.history and again.memory == c.memory and again.previous_end == c.last_spoke
    again.reset()
    assert json.loads(f.read_text(encoding="utf-8"))["history"] == []


def test_unreadable_file_is_set_aside(cfg, tmp_path):
    f = tmp_path / "memory.json"
    f.write_text('{"memory": "x", "history": [,]}', encoding="utf-8")
    c = convo(cfg, f)
    assert c.history == [] and c.memory == ""
    assert [p.name.split("-")[0] for p in tmp_path.iterdir()] == ["memory.unreadable"]


def test_the_date_in_his_prompt_moves_on_at_midnight(cfg, monkeypatch):
    from app import persona

    c = convo(cfg)
    monkeypatch.setattr(persona, "today", lambda: "Monday 23 November 2026")
    assert "Today is Monday 23 November 2026." in c.messages_for("Hello.")[0]["content"]
    monkeypatch.setattr(persona, "today", lambda: "Tuesday 24 November 2026")  # past midnight, same session
    assert "Today is Tuesday 24 November 2026." in c.messages_for("Still up?")[0]["content"]


def test_his_prompt_has_the_week_around_today(cfg):
    """The days ahead, each with how far off it is: he counted them a day short himself."""
    import datetime as dt
    from app.persona import today, week_around

    assert f"Today is {today()}. {week_around(dt.date.today())}" in convo(cfg).messages_for("Hi.")[0]["content"]

"""His conversation: the prompt he's sent, his history, its compacting, and keeping it on disk."""

import json

from app.config import load_config
from app.context import load_notes
from app.persona import Conversation

NOTES = load_notes(load_config().persona.notes)  # persona/notes.toml
IN_CHARACTER, PASSED_OVER, SPEAKING_UP, TOLD = NOTES.in_character, NOTES.passed_over, NOTES.speaking_up, NOTES.told


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


def test_an_offer_passed_over_is_noted_for_one_request_and_kept(cfg):
    """A lasagne recipe, offered and passed over, came up in 6 replies in 8. The note says to
    leave it; the offer stays in his history, to be taken up later."""
    c = convo(cfg)
    c.record("I'm making lasagne tonight.", "Lasagne. Want me to find a recipe?")
    last = c.messages_for("Guess what, I got the job!", taken_up=False)[-1]["content"]
    assert last == f"Guess what, I got the job!\n{IN_CHARACTER}\n{PASSED_OVER}"
    assert c.messages_for("Go on then.", taken_up=True)[-1]["content"] == f"Go on then.\n{IN_CHARACTER}"
    assert PASSED_OVER not in c.messages_for("Guess what, I got the job!")[-1]["content"]  # nothing to judge
    assert "Lasagne. Want me to find a recipe?" in [m["content"] for m in c.messages_for("Hi.", taken_up=False)]
    assert all(PASSED_OVER not in (m.get("content") or "") for m in c.history)


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

    import datetime as dt

    c = convo(cfg)
    monkeypatch.setattr(persona, "PINNED", dt.date(2026, 11, 23))
    assert "Today is Monday 23 November 2026." in c.messages_for("Hello.")[0]["content"]
    monkeypatch.setattr(persona, "PINNED", dt.date(2026, 11, 24))  # past midnight, same session
    assert "Today is Tuesday 24 November 2026." in c.messages_for("Still up?")[0]["content"]


def test_his_prompt_has_the_week_around_today(cfg):
    """The days ahead, each with how far off it is: he counted them a day short himself."""
    import datetime as dt
    from app.persona import today, week_around

    assert f"Today is {today()}. {week_around(dt.date.today())}" in convo(cfg).messages_for("Hi.")[0]["content"]


def report_turn(crew: str, report: str) -> list[dict]:
    from app import status
    return [{"role": "user", "content": crew},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                                                                 "function": {"name": "status", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": status.REPORTS[0] + "\n" + report},
            {"role": "assistant", "content": report}]


def test_crew_asides_left_out_of_what_he_sees(cfg):
    """With them kept, he copied them: every Cat line after was about coat hangers."""
    c = convo(cfg)
    c.record("Gift ideas?", "A jigsaw. The Cat's convinced the hangers are forming a union. He's livid.")
    c.record("Where's the Cat?", "In the wardrobe. The Cat's convinced the hangers are judging him.")  # asked: kept
    c.history += report_turn("Status report.", "Lister lost a bet. The Cat slept.")  # a report: kept
    sent = [m["content"] for m in c.messages_for() if m["role"] == "assistant" and m["content"]]
    assert sent == ["A jigsaw.", "In the wardrobe. The Cat's convinced the hangers are judging him.",
                    "Lister lost a bet. The Cat slept."]


def test_a_report_announced_isnt_asking_about_the_crew(cfg):
    """"Tell the crew it's ready": they're the crew to him, not Red Dwarf's."""
    c = convo(cfg)
    c.record("[Your research on 'Gifts' is finished. Tell the crew it's ready.]",
             "The gifts report is ready. The Cat's convinced the hangers are judging him.")
    assert [m["content"] for m in c.messages_for() if m["role"] == "assistant"] == ["The gifts report is ready."]


def test_what_he_doesnt_see_isnt_quoted_back_as_worn(cfg):
    """"Lately you've said: 'convinced the hangers are'": crew asides, cut from what he sees,
    were quoted back to him from his history, and he went on about coat hangers."""
    c = convo(cfg)
    c.record("Gift ideas?", "A jigsaw. The Cat's convinced the hangers are judging him.")
    c.record("And for June?", "A blower. The Cat's convinced the hangers are forming a union.")
    assert "hangers" not in c.messages_for("Thanks.")[-1]["content"]


def test_status_reports_arent_worn_phrases(cfg):
    """"Lately you've said 'an earlier report', 'report old news now'", from the line earlier
    reports are cut to: he gave a status report, unasked, three times in four."""
    c = convo(cfg)
    c.history = (report_turn("Status report.", "The Cat slept. Lister lost a bet.")
                 + report_turn("Another.", "The Cat sulked. Lister lost a bet.")
                 + report_turn("And another.", "The Cat sulked. Lister lost a bet again."))
    last = c.messages_for("She'd like a trug.")[-1]["content"]
    assert "Lately" not in last


def test_the_crew_after_asking_about_them_are_no_aside(cfg):
    """"What's he doing there?", after "Where's Kryten?": about him still."""
    c = convo(cfg)
    c.record("Where's Kryten?", "In the laundry.")
    c.record("What's he doing there?", "Ironing. Lister's helping, badly.")
    c.record("Right. Kettle's on.", "Thrilling. Lister's helping with that too.")
    sent = [m["content"] for m in c.messages_for() if m["role"] == "assistant"]
    assert sent == ["In the laundry.", "Ironing. Lister's helping, badly.", "Thrilling."]
    assert c.last_about_the_crew() is False
    c.record("Where's Rimmer?", "Saluting.")
    assert c.last_about_the_crew() is True



def test_his_tools_earlier_answers_are_sent_stubbed(cfg):
    """A forecast kept whole in what he saw, "Fair enough" got the forecast again (27 November,
    9 replays in 12). His history keeps it; what he's sent has TOLD in its place, but not a
    status report's (said as written, and how earlier ones are found)."""

    c = convo(cfg)
    c.record("Weather tomorrow?", "Rain, 7 to 14 degrees.", steps=[
        {"role": "assistant", "content": "", "tool_calls": [{"id": "w", "type": "function",
                                                             "function": {"name": "weather", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "w", "content": "Tomorrow: rain, 7 to 14 degrees, gusts 37."}])
    c.history += report_turn("Status report.", "Lister lost a bet. The Cat slept.")
    c.record("What's up with Jo?", "Boiler trouble.", steps=[
        {"role": "assistant", "content": "", "tool_calls": [{"id": "r", "type": "function",
                                                             "function": {"name": "recall", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "r", "content": "Jo's had no gas since Monday; her boiler's knackered."}])
    sent = [m["content"] for m in c.messages_for("Fair enough.") if m["role"] == "tool"]
    assert sent[0] == TOLD and sent[1].endswith("Lister lost a bet. The Cat slept.")
    assert sent[2].startswith("Jo's had no gas")  # his memory, for what's asked next: whole
    assert any("gusts 37" in (m.get("content") or "") for m in c.history)  # kept whole
    c.clean = False
    assert [m["content"] for m in c.messages_for() if m["role"] == "tool"][0].startswith("Tomorrow: rain")


def test_a_time_from_a_tool_is_still_his_once_stubbed(cfg):
    """Times he gives are checked against what his tools told him, whole, before it's stubbed."""
    c = convo(cfg)
    c.record("Kettle's on.", "Tea, then. It's 22:14.", steps=[
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                                                             "function": {"name": "clock", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "Here: 22:14 on Friday."}])
    c.record("Lovely.", "Tea, then. It's 22:20.")  # nobody gave him that: left out
    assert [m["content"] for m in c.messages_for() if m["role"] == "assistant" and m.get("content")] == [
        "Tea, then. It's 22:14.", "Tea, then."]


def test_raw_his_history_is_sent_as_it_was(cfg):
    """evals --raw: nothing cut from what he sees either."""
    c = convo(cfg)
    c.clean = False
    c.record("Coal prices?", "Up again. Anything else you need me to dig up?")
    assert [m["content"] for m in c.messages_for() if m["role"] == "assistant"] == [
        "Up again. Anything else you need me to dig up?"]

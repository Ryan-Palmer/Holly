"""The evals' checks and scorecards, without a model."""

from app import conversations, evals

EXAMPLES = ["Morning. You're up early, the sun's barely had a cup of tea."]


def test_checks_need_no_judge():
    ok = evals.problems_with("Morning, Holly.", set(), "Morning. Bit early.", [], EXAMPLES)
    assert ok == []
    assert evals.problems_with("Morning, Holly.", set(), "Certainly! How can I help?", [], EXAMPLES) == [
        "assistant: 'Certainly'"]
    assert "didn't call clock" in evals.problems_with("What time is it?", {"clock"}, "Late.", [], EXAMPLES)
    assert evals.problems_with("What time is it?", {"clock", "ask"}, "Ten.", ["clock"], EXAMPLES) == []
    recited = evals.problems_with("Hi.", set(), "You're up early, the sun's barely had a cup.", [], EXAMPLES)
    assert recited and recited[0].startswith("reused an example")
    assert "60 words" in evals.problems_with("Tell me about stars.", set(), "word " * 60, [], EXAMPLES)


def row(prompt, reply, tags=(), tools=(), score=4, problems=()):
    return dict(prompt=prompt, tags=sorted(tags), reply=reply, tools=list(tools), words=len(reply.split()),
                sentences=reply.count("."), asked="?" in reply, seconds=1.0, problems=list(problems), score=score,
                why="")


def test_scorecard_and_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "RUNS", tmp_path)
    rows = [row("Morning.", "Morning. Up early?", tags={"ask"}),
            row("What time is it?", "Late.", tags={"clock"}, problems=["didn't call clock"], score=2)]
    persona = evals.persona_scores(rows)
    assert persona["judge"] == 3.0 and (persona["passing"], persona["replies"]) == (1, 2)
    assert (persona["asked_where_suited"], persona["tools_ok"], persona["tooled"]) == (1, 0, 1)
    runs = [dict(name="a timer", run=1, wrong=[], turns=[]),
            dict(name="a timer", run=2, wrong=["nothing set"], turns=[dict(said="x", reply="y", tools=[])])]
    scen = evals.scenario_scores(runs)
    assert (scen["passed"], scen["runs"], scen["by_name"]["a timer"]) == (1, 2, (1, 2))
    replies = ["Evening.", "Green. The Cat's convinced the hangers are judging him."]
    talk = dict(name="after the reports", run=1, wrong=[], transcript="1. Crew: Evening.\n   Holly: Evening.",
                replies=replies, turns=[dict(said="Evening.", reply="Evening.", tools=[])], verdicts=[
                    dict(judge="google/judge", overall=4, **{k: v for k, v in conversations.verdict(
                        "UNASKED: PASS", replies).items() if k != "overall"}),
                    dict(judge="qwen/judge", overall=2, **{k: v for k, v in conversations.verdict(
                        'UNASKED: FAIL - "The Cat\'s convinced the hangers" - the Cat, unasked', replies).items()
                        if k != "overall"})])
    card = dict(model="google/some-model", judges=["google/judge", "qwen/judge"], label="try", commit="abc123",
                when="2026-11-23T23:00", session=False, persona=persona, scenarios=scen, persona_rows=rows,
                scenario_runs=runs, conversations=conversations.scores([talk]), conversation_runs=[talk],
                repeated=evals.openings(rows + [dict(reply="Right you are. Done."), dict(reply="Right you are. Set.")]))
    path = evals.write(card)
    md = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "| a timer | 1 of 2 |" in md and "nothing set" in md and "Late." in md
    assert "opens 'right you are' x2" in md
    assert "### after the reports, run 1: passed (flagged by one judge: UNASKED (judge))" in md and "- UNASKED: FAIL \"The Cat's convinced" in md
    assert "Conversations passed: 1 of 1" in "\n".join(evals.summary(card))
    table = evals.compare().splitlines()
    assert table[0].split()[:3] == ["when", "label", "model"]
    assert "some-model" in table[1] and "1/2" in table[1] and "3.00" in table[1] and "1/1" in table[1]


def test_two_judges_scores_are_averaged_and_each_kept():
    rows = [dict(by={"google/a": [4, "dry"], "qwen/b": [2, "chirpy"]}), dict(by={"google/a": [3, ""], "qwen/b": [0, ""]})]
    evals.merged(rows)
    assert rows[0]["score"] == 3 and rows[0]["why"] == "a: dry / b: chirpy"
    assert rows[1]["score"] == 3  # no score from one judge: the other's
    assert evals.by_judge(rows) == {"google/a": 3.5, "qwen/b": 2}


def test_compare_with_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "RUNS", tmp_path)
    assert evals.compare() == "No scorecards yet in runs/."

"""Research reports judged (app/report_check.py): reading the judges, and the sums, without a model."""

from app import report_check as rc

REPORT = "## Games\nTicket to Ride is easy to learn. Forbidden Island is cooperative. It costs 40 pounds."


def test_a_judges_scores_and_claims_read():
    v = rc.verdict('GROUNDED: 4\nANSWERS: 5\nCLEAR: 3\nUNSUPPORTED: "It costs 40 pounds." "It won awards."', REPORT)
    assert v["scores"] == {"GROUNDED": 4, "ANSWERS": 5, "CLEAR": 3}
    assert v["shown"] == ["It costs 40 pounds."]  # the other isn't in the report: doesn't count
    assert rc.verdict("GROUNDED: 5\nUNSUPPORTED: none", REPORT)["scores"]["ANSWERS"] == 0


def test_the_sums_over_reports_and_judges():
    v = rc.verdict('GROUNDED: 4\nANSWERS: 5\nCLEAR: 3\nUNSUPPORTED: "It costs 40 pounds."', REPORT)
    w = rc.verdict("GROUNDED: 2\nANSWERS: 3\nCLEAR: 5\nUNSUPPORTED: none", REPORT)
    rows = [dict(title="Games", report=REPORT, minutes=6.0, verdicts={"g": v, "q": w}),
            dict(title="Pumps", report="", minutes=2.5, verdicts={})]
    s = rc.scores(rows)
    assert s["means"] == {"GROUNDED": 3.0, "ANSWERS": 4.0, "CLEAR": 4.0}
    assert (s["reports"], s["written"], s["unsupported"], s["minutes"]) == (2, 1, 1, 8.5)

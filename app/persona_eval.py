"""The persona part of the evals (app/evals.py), as before:

    python -m app.persona_eval [--runs N] [--label NAME] [--no-judge] [--session]
"""

from __future__ import annotations

import argparse
import sys

from .evals import main as evals


def main() -> int:
    parser = argparse.ArgumentParser(description="Score how well Holly stays Holly (app.evals --only persona)")
    parser.add_argument("--runs", type=int, default=1, help="replies per prompt (default 1)")
    parser.add_argument("--label", default="", help="a name for the report, to compare runs")
    parser.add_argument("--no-judge", action="store_true", help="skip the judge's scores")
    parser.add_argument("--session", action="store_true", help="one conversation, not each prompt afresh")
    args = parser.parse_args()
    return evals(["--only", "persona", "--persona-runs", str(args.runs), "--label", args.label]
                 + ["--no-judge"] * args.no_judge + ["--session"] * args.session)


if __name__ == "__main__":
    sys.exit(main())

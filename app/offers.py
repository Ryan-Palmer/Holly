"""Have the crew just taken him up on something he offered to do? Then his reply must do it.

He offers more than he does: "Want me to search for it now or later?" / "You can get it now
if you like" had him say "I'll get on with it then. Give me a minute." and call nothing (26
November; replayed, 1 in 6). "I can do a proper research job on it if you want." / "Yes,
please." had him say a report was coming and start none, 5 times in 6 (25 November). Told
by text (a question mark, a word like "report"), the second was missed; patterns on the
first would only grow. So his model judges, from what he last said and their reply,
whatever the wording on either side (taken_up), and app/speak.py has the reply call a tool
if so. Which tool is his to choose, as the talk makes plain: a research start isn't held
for asking which (asked to tell research from the rest as well, the model called "Go on
then" to a report offer something else, and he'd have asked all over again).

Asked only after a reply that offered or asked something (guards.ASKED_OR_OFFERED): ~0.25 s then.
"""

from __future__ import annotations

# Added to the crew's line in that request (app/speak.py), not kept in his history. Made to
# call a tool, with nothing said of which, he picked the wrong one 2 times in 6: a news
# story read for "Yes, please" to his offer of a proper research job, a report started for
# "You can get it now if you like" to "search for it now or later?".
NOTE = ("[They're taking you up on what you just offered or asked about: do exactly that, the way they chose, "
        "now, with the tool for it.]")

TAKEN_UP_PROMPT = """\
Holly, the household computer, does things with his tools: he searches the web, reads or sums up his \
reports, writes a report on something while he dozes, sets timers, gives the weather, plays music, and so \
on. You're given what he last said, and the crew's reply to it.
YES: they're taking him up on, or choosing, something he offered or asked about doing that needs one of his \
tools ("yes, please", "go on then", "get it now", "a proper report", "later's fine", "yes, set it", "yes, \
that one", "talk me through it").
NO: anything else: no, not now, that's enough, another subject, a remark, or yes to something needing none \
of his tools.
Reply with one word: YES or NO."""


def taken_up(lm, model: str | None, said: str, reply: str, quiet: bool = True) -> bool:
    """Is the crew's `reply` to his `said` taking him up on something he offered with his
    tools? If it can't be asked, False (as before), or with `quiet` off (the evals), the
    error."""
    try:
        return lm.ask(TAKEN_UP_PROMPT, f"Holly said: {said}\nThey reply: {reply}", model, max_tokens=4,
                      temperature=0).strip().upper().startswith("YES")
    except Exception:
        if not quiet:
            raise
        return False

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

Only a reply to what he offered counts: a new request of theirs, or their own news, is
theirs to have answered as any other. A full eval run found "I start in January. I'm so
excited." taken for yes to a lasagne recipe, and "Can you tell me more about the comet
story?" for yes to "Not much to do, is there?": a tool forced for nothing, and a news
story searched for, not read. A note telling him they'd taken it up, for picking the
tool, made no difference: 2 wrong in 16 with it and without (a report started when told
"you can get it now"), so it went. Carried on past, though, his request says to leave it
(persona.PASSED_OVER): a lasagne recipe, passed over, was offered in 6 replies in 8.

Asked only after a reply that offered or asked something (guards.ASKED_OR_OFFERED): ~0.25 s then.
"""

from __future__ import annotations

TAKEN_UP_PROMPT = """\
Holly, the household computer, does things with his tools: he searches the web, reads or sums up his \
reports, writes a report on something while he dozes, sets timers, gives the weather, plays music, and so \
on. You're given what he last said, and the crew's reply to it.
YES: their reply says yes to, or chooses, something he offered or asked about doing, there in what he said, \
that needs one of his tools ("yes, please", "go on then", "get it now", "a proper report", "later's fine", \
"yes, set it", "yes, that one", "talk me through it").
NO: anything else: no, not now, that's enough; their own news, a remark, or carrying on as if he hadn't \
offered; a new question or request of their own, not what he offered; or yes to something needing none of \
his tools.
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

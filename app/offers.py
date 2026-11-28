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
(persona/notes.toml's passed_over): a lasagne recipe, passed over, was offered in 6 replies in 8.

Asked only after a reply that offered or asked something (guards.ASKED_OR_OFFERED): ~0.25 s then.
"""

from __future__ import annotations

from .decisions import Decision, register, starts, tool_named

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


# Which tool, once they've taken him up (taken_up): asked as a question of its own, with what
# he offered in front of it. Forced to call "a tool", he chose for himself, and "You can get it
# now if you like", to a search he'd offered, started a report 2 times in 16. One check for
# every line ("which tool, if any?") got 8 of 80 lines needing none wrong (app/which_tool.py);
# this is the narrow question, asked only after a yes.
OFFERED_TOOL_PROMPT = """\
Holly, the household computer, offered or asked about doing something, and the crew have just
said yes to it, or chosen. Which of his tools does doing it need?
search: looking something up on the web now (a recipe, a fact, a price).
research: a proper written report, started now and written while he dozes; or reading, summing
up, talking them through or opening a report he wrote himself.
news: the news, or reading more of a story from the news, as given by the news tool.
weather: a forecast. clock: the time. timer: setting a timer or a reminder.
spotify: music. recall: something from before. status: a status report on the Red Dwarf crew.
If he offered a choice (a quick search now, or a proper report), the one they chose. "Get it
now", "now's fine" or "later's fine", to a search he offered, is search.
Reply with one word: the tool's name, or NONE if it's unclear."""


def _after(used) -> str:
    return f" (after using the {' and '.join(used)} tool{'s' * (len(used) > 1)})" if used else ""


TAKEN_UP = register(Decision(
    "taken_up", TAKEN_UP_PROMPT, lambda said, reply: f"Holly said: {said}\nThey reply: {reply}", "YES / NO",
    starts("YES")))
OFFERED_TOOL = register(Decision(
    "offered_tool", OFFERED_TOOL_PROMPT, lambda said, reply, tools, used=(): f"Holly said{_after(used)}: {said}\n"
                                                                             f"They reply: {reply}",
    "a tool's name, or NONE", lambda text, said, reply, tools, used=(): tool_named(text, tools), fallback=None))


def offered_tool(lm, model: str | None, said: str, reply: str, tools, used=()) -> str | None:
    """The tool what he offered in `said`, taken up by `reply`, needs: one of `tools` (those he
    has), or None if the model can't say (then he's left to choose, as before). `used`: the
    tools his reply came from: "Shall I read you the rest of that story?" / "Go on." was
    research, from his words alone; it was a news story, as the news tool said."""
    return OFFERED_TOOL.ask(lm, model, said, reply, tools, used)


def taken_up(lm, model: str | None, said: str, reply: str, quiet: bool = True) -> bool:
    """Is the crew's `reply` to his `said` taking him up on something he offered with his
    tools? If it can't be asked, False (as before), or with `quiet` off (the evals), the
    error."""
    return TAKEN_UP.ask(lm, model, said, reply, quiet=quiet)

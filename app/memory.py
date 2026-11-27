"""Holly's notes on his older conversations, written while he sleeps.

When his history passes `history_soft_tokens`, the older exchanges are folded
into notes at the end of his system prompt, by the same Gemma (no other model
to load), and only the newest are kept word for word. This happens while he's
asleep, so the wait for it, and for LM Studio to reread the shorter prompt,
falls in a silence rather than in a reply. At startup, the last session's
history is folded in the same way (Holly.start), so the notes carry on from
one session to the next.
"""

from __future__ import annotations

from .lmstudio import LmStudio
from .persona import today

NOTES_PROMPT = """\
You keep the memory notes of Holly, the ship's computer from Red Dwarf, who \
talks with the crew over days and weeks. Older conversation is about to leave \
his memory, and your notes are all he will keep of it. Today is {today}.

Write plain notes, at most {words} words in all, one short line each, in two parts.

Under "About the crew:", only what stays true for months: who he talks to \
(by name, if they gave one), and what they've told him about themselves: \
family and friends, pets, work, home, likes and dislikes. Running jokes go \
here too.

Under "Lately:", the crew's own news and plans, one line each, starting with \
the date it happens (not the day it was mentioned): visits, appointments, trips, things they're making or doing, \
things left unfinished (a game, a question left open), and anything he found \
out for them that they'll want again (a forecast, a result). Name who and \
what exactly. A plan goes here only, not under "About the crew:" as well.

Dates: write the real date, like "Sunday 22 November", never "tomorrow" or \
"next Tuesday". The conversation shows when each exchange was said, and \
each day word in a crew line is followed by the date it means, in square \
brackets, worked out for you: "coming over tomorrow [Sunday 22 November \
2026]" happens on Sunday 22 November. Copy those dates; don't work out your \
own. If it doesn't show when something was said, leave it undated: never \
guess a date.

"The crew" are the real people he talks with. Red Dwarf's own crew \
(Lister, Rimmer, the Cat, Kryten and the rest) and their adventures are in \
his ship's records, not his notes: never write a note about them, and if his \
notes so far have any (a line naming one of them, or "The crew recalled..."), \
delete it. Name the crew only as they've named themselves: someone who hasn't \
given a name is "the crew" (never Lister or anyone else from the show, however \
much they talk about it), and their family by how they're related to them \
("the crew's sister is Jo").

Leave out what the crew asked him for (recipes, stories, poems, \
explanations, how-tos): what he said back needn't be remembered. Leave out \
that they asked him for something at all (the weather, the time, the news, a \
status report, a search, a report): "asked for the weather in Keswick" or \
"asked for a report on X" is never a note, and if his notes so far have any \
such lines, delete them. Leave out his own jokes and insults about them: \
they aren't facts. Note only what was \
said: never guess likes or dislikes from what someone asked about.

If he has notes already, start from them: keep every note that still holds, \
change any the conversation has changed, drop past events once they no \
longer matter, then add what's new. Leave out greetings, small talk, and any \
kind of note there's nothing for (never write "none"). Write only the notes, \
with those two labels and no other headings or markdown."""


async def write_notes(lm: LmStudio, model: str, earlier: str, transcript: str, max_tokens: int) -> str:
    """Fold `transcript` into the `earlier` notes; returns the new notes.
    Streamed, so cancelling it stops LM Studio writing."""
    words = max_tokens * 3 // 4
    # His earlier notes go last, nearest the writing. Before the transcript
    # (8K tokens of it), Gemma kept only what was in the transcript.
    user = f"The conversation to fold in:\n{transcript}\n\n" + (
        f"His notes so far:\n{earlier}" if earlier else "He has no notes yet.")
    # Temperature 0: at 0.3, one run in two dropped the cat from a fold. A little over
    # the budget, so a note isn't cut off mid-line; any part line is dropped.
    notes, cut = await lm.aask(NOTES_PROMPT.format(words=words, today=today()), user, model,
                               max_tokens + max_tokens // 4, temperature=0.0)
    return notes.rsplit("\n", 1)[0].strip() if cut else notes

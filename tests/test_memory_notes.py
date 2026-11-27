"""Writing his notes (app/memory.py), with the model stood in for: what it's asked."""

import asyncio

from app import memory
from conftest import FakeLm


class Asked(FakeLm):
    async def aask(self, system, user, model=None, max_tokens=800, temperature=0.2, reasoning="none"):
        self.side_requests.append(dict(system=system, user=user, max_tokens=max_tokens))
        return "About the crew:\nThe crew's sister is Jo.", False


def test_nearly_full_notes_are_told_to_make_room():
    """Full, they made no room, and crept towards the cut that loses their newest lines."""
    lm = Asked()
    budget = 1500 * 3 // 4
    for earlier, told in (("word " * int(budget * 0.9), True), ("word " * int(budget * 0.5), False), ("", False)):
        asyncio.run(memory.write_notes(lm, "m", earlier.strip(), "Crew: Hello.", 1500))
        user = lm.side_requests[-1]["user"]
        assert ("Make the room" in user) is told, earlier[:20]
        if told:
            assert f"{int(budget * 0.9)} words, nearly all of the {budget}" in user


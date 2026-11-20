"""Holly's persona: the system prompt, example exchanges and rolling history."""

from __future__ import annotations

import datetime
import tomllib

from .config import PersonaConfig


def today() -> str:
    return datetime.date.today().strftime("%A %d %B %Y").replace(" 0", " ")


class Conversation:
    def __init__(self, cfg: PersonaConfig, extra_prompt: str = ""):
        """`extra_prompt` is added to the system prompt, before the examples:
        the rules for looking things up, when Holly can search."""
        self.max_turns = cfg.history_turns
        self.trim_to = min(cfg.history_trim_to, cfg.history_turns)
        self.system_prompt = cfg.system_prompt.read_text(encoding="utf-8").strip()
        if extra_prompt:
            self.system_prompt += "\n\n" + extra_prompt.strip()
        with open(cfg.examples, "rb") as f:
            examples = tomllib.load(f).get("example", [])
        # The examples go inside the system prompt, not as fake earlier turns.
        # As turns, they made Qwen 3.8 (thinking off) answer with nothing at
        # all once the real history passed about three exchanges.
        if examples:
            lines = [f'Crew: "{ex["user"]}"\nHolly: "{ex["holly"]}"' for ex in examples]
            self.system_prompt += "\n\nExamples of how you talk:\n\n" + "\n\n".join(lines)
        self.history: list[dict] = []
        self._date = ""
        self.preamble: list[dict] = []
        self._date_preamble()

    def _date_preamble(self) -> None:
        """Holly knows the date (without it he searched for last year's
        results). It goes at the end of the system prompt, which changes only
        when the date does, so LM Studio's cache survives all day."""
        if self._date != (d := today()):
            self._date = d
            self.preamble = [{"role": "system", "content": f"{self.system_prompt}\n\nToday is {d}."}]

    def messages_for(self, user_text: str) -> list[dict]:
        """The full message list to send for a new user line."""
        self._date_preamble()
        return [*self.preamble, *self.history, {"role": "user", "content": user_text}]

    # History budget, in characters (~4 a token): ~5K tokens of the 8K
    # context, leaving room for the persona (~1K) and the reply. Searches make
    # exchanges longer, so the count of exchanges alone no longer bounds it.
    MAX_HISTORY_CHARS = 20_000

    def record(self, user_text: str, holly_text: str, steps: list[dict] | None = None) -> None:
        """Add an exchange: what was said, plus any `steps` between (his
        searches and their results, as sent to the model).

        The steps stay in the history. Without them, his history showed
        factual questions answered straight off, and after a few exchanges he
        stopped searching and said "I've searched the internet" instead.

        Past `history_turns` exchanges or MAX_HISTORY_CHARS, the oldest are
        dropped in one chunk, down to `history_trim_to` and two thirds of the
        budget, not one at a time: LM Studio reuses its cache only while the
        start of the prompt is unchanged, and dropping one exchange per turn
        changed it every turn, so the whole history was reread every time."""
        self.history.append({"role": "user", "content": user_text})
        self.history.extend(steps or [])
        self.history.append({"role": "assistant", "content": holly_text})
        starts = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(starts) <= self.max_turns and self._chars(0) <= self.MAX_HISTORY_CHARS:
            return
        keep = starts[max(0, len(starts) - self.trim_to)]
        while self._chars(keep) > self.MAX_HISTORY_CHARS * 2 // 3 and keep != starts[-1]:
            keep = next(i for i in starts if i > keep)
        del self.history[:keep]

    def _chars(self, start: int) -> int:
        return sum(len(m.get("content") or "") + len(str(m.get("tool_calls", "")))
                   for m in self.history[start:])

    def reset(self) -> None:
        self.history.clear()

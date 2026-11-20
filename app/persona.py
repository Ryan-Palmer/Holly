"""Holly's persona: the system prompt, example exchanges and rolling history."""

from __future__ import annotations

import tomllib

from .config import PersonaConfig


class Conversation:
    def __init__(self, cfg: PersonaConfig):
        self.max_turns = cfg.history_turns
        self.trim_to = min(cfg.history_trim_to, cfg.history_turns)
        self.system_prompt = cfg.system_prompt.read_text(encoding="utf-8").strip()
        with open(cfg.examples, "rb") as f:
            examples = tomllib.load(f).get("example", [])
        # The examples go inside the system prompt, not as fake earlier turns.
        # As turns, they made Qwen 3.8 (thinking off) answer with nothing at
        # all once the real history passed about three exchanges.
        if examples:
            lines = [f'Crew: "{ex["user"]}"\nHolly: "{ex["holly"]}"' for ex in examples]
            self.system_prompt += "\n\nExamples of how you talk:\n\n" + "\n\n".join(lines)
        self.preamble: list[dict] = [{"role": "system", "content": self.system_prompt}]
        self.history: list[dict] = []

    def messages_for(self, user_text: str) -> list[dict]:
        """The full message list to send for a new user line."""
        return [*self.preamble, *self.history, {"role": "user", "content": user_text}]

    def record(self, user_text: str, holly_text: str) -> None:
        """Add an exchange. Past `history_turns`, the oldest are dropped in one
        chunk, down to `history_trim_to`, not one at a time: LM Studio reuses
        its cache only while the start of the prompt is unchanged, and
        dropping one exchange per turn changed it every turn, so the whole
        history was reread every time."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": holly_text})
        if len(self.history) > 2 * self.max_turns:
            del self.history[: len(self.history) - 2 * self.trim_to]

    def reset(self) -> None:
        self.history.clear()

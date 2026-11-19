"""Checks and cleaning for text that is about to be spoken by a TTS voice."""

from __future__ import annotations

import re

# Emoji and pictographs, dingbats, flags, variation selectors and joiners.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️‍]"
)
_ACTION = re.compile(r"\*[^*]+\*|\[[^\]]+\]")
_LIST_LINE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s*#+\s*", re.MULTILINE)
_MARKDOWN_CHARS = re.compile(r"[*_`#|>~]")
_URL = re.compile(r"https?://\S+")
_SENTENCE_END = re.compile(r"[.!?]+[\"')”’]*(?=\s|$)")

# Full stops that don't end a sentence: common abbreviations, and initials
# such as "S.S." or "J. Smith".
_ABBREVIATIONS = re.compile(
    r"\b(?i:Mr|Mrs|Ms|Dr|St|vs|etc|e\.g|i\.e)\.|\b(?:[A-Z]\.)+"
)


def speech_problems(text: str) -> list[str]:
    """Things in `text` a TTS voice can't say, or shouldn't."""
    problems = []
    if _ACTION.search(text):
        problems.append("action in asterisks or brackets")
    if _EMOJI.search(text):
        problems.append("emoji")
    if _LIST_LINE.search(text):
        problems.append("list")
    if _HEADING.search(text):
        problems.append("heading")
    if "```" in text or "`" in text:
        problems.append("code")
    if _URL.search(text):
        problems.append("URL")
    elif _MARKDOWN_CHARS.search(_ACTION.sub("", text)):
        problems.append("markdown symbol")
    return problems


def count_sentences(text: str) -> int:
    text = _ABBREVIATIONS.sub("", text)
    # Pieces between sentence ends; a trailing fragment with no full stop counts too.
    return sum(1 for piece in _SENTENCE_END.split(text) if re.search(r"\w", piece))


def clean_for_speech(text: str) -> str:
    """The safety net before TTS: strip what a voice can't say."""
    text = _ACTION.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _URL.sub("", text)
    text = _HEADING.sub("", text)
    text = _LIST_LINE.sub("", text)
    text = _MARKDOWN_CHARS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

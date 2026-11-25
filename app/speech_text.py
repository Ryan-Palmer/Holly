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
# A status report's "Additional" (app/search.md), however he punctuates it, said with a
# dash: alone, VoxCPM garbled the word; after a full stop or comma, in the same breath,
# it ran on with no pause (0 ms, 3 times in 3); with a dash it paused 240-880 ms and said
# it clearly each time. An ellipsis lost the word.
_ADDITIONAL = re.compile(r"\bAdditional\s*(?:[.:;,!]|\.\.\.|…|—|–|-)+\s*(?=\w)")
# A status report's sections, and their names as he might start one with them: "Politics —
# ministers...", "Politics' got the age of consent...", "In tech, ...", "Top stories are...".
_SECTION_NAMES = ("top stories", "top", "headlines", "politics", "world", "world news", "business", "money",
                  "the economy", "finance", "technology", "tech", "science", "sport", "sports", "health",
                  "entertainment", "additional")
_SECTION = re.compile(r"^(?:and )?(?:in )?(?:the )?(?:" + "|".join(re.escape(n) for n in _SECTION_NAMES)
                      + r")(?![a-z])", re.IGNORECASE)
# A heading at the start of a sentence, a status report's "Politics: ministers want...":
# said with a dash, for the same pause after it.
_HEADED = re.compile(r"^([A-Z][\w'&]*(?: (?:and )?[\w'&]+){0,2}):\s+(?=\w)")
_SENTENCE_END = re.compile(r"[.!?]+[\"')”’]*(?=\s|$)")
_CLAUSE_BREAK = re.compile(r"[,;:—–](?=\s)")

# A full stop that doesn't end a sentence: after a common abbreviation, or
# after initials such as "S.S." or "J. Smith" (but "I." and "A." can end one).
_NOT_AN_END = re.compile(
    r"(?:\b(?i:mr|mrs|ms|dr|st|vs|etc|e\.g|i\.e)|\b(?:[A-Z]\.)+[A-Z]|\b(?![AI]\b)[A-Z])\.$"
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


class SentenceChunker:
    """Cuts a streamed reply into speakable chunks, a sentence at a time.

    A sentence end only counts once the following whitespace has arrived,
    since "3." might be "3.5". The first chunk may also be cut at a clause
    break (a comma, say) once it has `first_clause_words` words, so the
    voice can start sooner. Chunks come back cleaned for speech.
    """

    def __init__(self, first_clause_words: int = 6, min_words: int = 2):
        self.first_clause_words = first_clause_words
        self.min_words = min_words  # a shorter sentence goes with the next
        self.buffer = ""
        self.cuts = 0

    def feed(self, delta: str) -> list[str]:
        self.buffer += delta
        raw = []
        while (cut := self._find_cut()) is not None:
            raw.append(self.buffer[:cut])
            self.buffer = self.buffer[cut:]
            self.cuts += 1
        return [c for c in map(clean_for_speech, raw) if c]

    def flush(self) -> list[str]:
        rest = clean_for_speech(self.buffer)
        self.buffer = ""
        return [rest] if rest else []

    def _find_cut(self) -> int | None:
        buf = self.buffer
        for m in _SENTENCE_END.finditer(buf):
            if m.end() >= len(buf):
                break  # wait for the next character to be sure
            if _NOT_AN_END.search(buf[: m.start() + 1]):
                continue
            if len(buf[: m.end()].split()) < self.min_words:
                # A word alone isn't sent on its own: VoxCPM, given just "Additional.", made
                # a garbled mumble of it. With the next sentence it says it, and pauses after
                # the full stop as it does between any two.
                continue
            return m.end()
        if self.cuts == 0:
            for m in _CLAUSE_BREAK.finditer(buf):
                if len(buf[: m.start()].split()) >= self.first_clause_words:
                    return m.end()
        return None


def starts_section(chunk: str) -> bool:
    """Does this chunk start with a status report's section ("Politics — ...", "Politics' got
    ...", "In tech, ...", "Additional — ...")? Then, in a reply with news in it, it's said
    after a longer pause (app/speak.py). Asked for the dash, he wrote "Politics' got..." all
    the same, copying his last report: what's looked for is the name, not the punctuation."""
    return bool(_SECTION.match(chunk))


def split_sentences(text: str) -> list[str]:
    chunker = SentenceChunker(first_clause_words=10**9)
    return chunker.feed(text) + chunker.flush()


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def clean_for_speech(text: str) -> str:
    """The safety net before TTS: strip what a voice can't say."""
    text = _ACTION.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _URL.sub("", text)
    text = _HEADING.sub("", text)
    text = _LIST_LINE.sub("", text)
    text = _MARKDOWN_CHARS.sub("", text)
    text = _ADDITIONAL.sub("Additional — ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _HEADED.sub(r"\1 — ", text)

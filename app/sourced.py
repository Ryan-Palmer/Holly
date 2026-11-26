"""Numbers he says after a tool must come from it (an eval check, app/evals.py; no model).

Told a forecast, a price or the time, he gives it back: a number in his reply that's in
nothing his tools told him that turn, nor in what the crew said, he made up. Only replies
where a tool was called, and only numbers of 10 or more: "two or three", "a couple of",
the small change of talk, would only cry wolf. It reads digits ("78", "3,000", "21:15",
"8.5") and words ("seventy-eight per cent", "a hundred", "three thousand"). His character's
own numbers (an IQ of 6,000; three million years in deep space) are his to say.

It can't tell which of a tool's numbers belongs to what: told a week's forecast, Saturday's
78 per cent said of Sunday passes. The conversations' judges look for that (app/conversations.py).
"""

from __future__ import annotations

import re

LEAST = 10
HIS_OWN = {6000, 3_000_000, 3_000}  # his IQ; years in deep space (and as "three thousand", he rounds)

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_WORD = re.compile(r"[a-z]+", re.IGNORECASE)
_DIGITS = re.compile(r"\d+:\d{2}|\d[\d,]*(?:\.\d+)?")


def from_words(text: str) -> list[float]:
    """The numbers written in words in `text`: "seventy-eight" 78, "a hundred" 100, "three
    thousand two hundred" 3200, "one hundred and twenty" 120; "twelve nineteen" (a time) 12
    and 19; "one pound eighty" 1.8 (as a price is written)."""
    out: list[float] = []
    total = current = 0
    started = False
    last = ""  # the kind of the word before: "unit", "tens", "scale"
    words = _WORD.findall(text.lower().replace("-", " "))

    def close() -> None:
        nonlocal total, current, started, last
        if started:
            out.append(total + current)
        total = current = 0
        started, last = False, ""

    for i, w in enumerate(words):
        if w in _UNITS:
            if last == "unit" or (last == "tens" and current % 10):
                close()  # "twelve nineteen": two numbers
            current += _UNITS[w]
            started, last = True, "unit"
        elif w in _TENS:
            if last in ("unit", "tens"):
                close()  # "twelve twenty": two
            current += _TENS[w]
            started, last = True, "tens"
        elif w in _SCALES and (started or (i and words[i - 1] in ("a", "one"))):
            current = max(current, 1) * _SCALES[w]
            if _SCALES[w] >= 1_000:
                total, current = total + current, 0
            started, last = True, "scale"
        elif w == "and" and started:
            continue
        elif w in ("pound", "pounds", "quid") and (started or (i and words[i - 1] == "a")):
            pounds = (total + current) if started else 1  # "a pound eighty"
            total = current = 0
            started, last = False, ""
            pence = from_words(" ".join(words[i + 1:i + 3]))  # "one pound eighty"
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if pence and (nxt in _UNITS or nxt in _TENS) and pence[0] < 100:
                out.append(pounds + pence[0] / 100)
                words[i + 1:i + 3] = ["~"] * len(words[i + 1:i + 3])  # its pence, taken
            else:
                out.append(pounds)
        else:
            close()
    close()
    return out


def numbers(text: str) -> set[float]:
    """Every number in `text`, in digits or words; a time ("21:15") as itself and its parts."""
    found: set[float] = set()
    for d in _DIGITS.findall(text):
        if ":" in d:
            h, m = d.split(":")
            found |= {float(h), float(m), float(h + m)}
        else:
            try:
                found.add(float(d.replace(",", "")))
            except ValueError:
                pass
    found |= {float(n) for n in from_words(text)}
    return found


def unsourced(reply: str, told: list[str]) -> list[float]:
    """The numbers in `reply` (10 or more, not his own) that aren't in `told`: what his tools
    told him that turn, and what the crew said."""
    known = set().union(*(numbers(t) for t in told)) if told else set()
    return sorted(n for n in numbers(reply) if n >= LEAST and n not in HIS_OWN and n not in known)


def check(rows: list[dict]) -> dict:
    """Over `rows` ({"part", "said", "reply", "told"}: his replies, with what his tools told
    him that turn), the ones that used a tool, and those with a number from nowhere."""
    checked = [r for r in rows if r["told"] and r["reply"]]
    flagged = []
    for r in checked:
        if bad := unsourced(r["reply"], r["told"] + [r["said"]]):
            flagged.append(dict(part=r["part"], said=r["said"], reply=r["reply"],
                                numbers=[int(n) if n == int(n) else n for n in bad]))
    return dict(checked=len(checked), flagged=flagged)

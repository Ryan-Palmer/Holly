"""Holly's clock: the date and time, here or in any time zone, as a tool he calls.

Gemma has no sense of time; it only knows what it's told. The PC's clock is
kept to internet time by Windows, so reading it is exact and instant. A web
search for the time would be slow and wrong: snippets show the time a page
was indexed.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # time zone data from tzdata on Windows

TOOL = {
    "type": "function",
    "function": {
        "name": "clock",
        "description": "The current date and time here and in UTC, or in a given time zone.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description":
                "Optional: an IANA time zone such as 'America/New_York' or 'Asia/Tokyo', for the time "
                "somewhere else. Leave it out for the time here."}},
        },
    },
}


def _when(t: datetime.datetime) -> str:
    return f"{t:%H:%M} on {t:%A} {t.day} {t:%B} {t:%Y}"


def now(timezone: str | None = None) -> str:
    """The tool's answer, as plain facts."""
    here = datetime.datetime.now().astimezone()
    utc = here.astimezone(datetime.timezone.utc)
    lines = [f"Here: {_when(here)} (UTC{here.utcoffset().total_seconds() / 3600:+g}).",
             f"UTC: {_when(utc)}."]
    if timezone:
        try:
            there = here.astimezone(ZoneInfo(timezone.strip()))
            lines.append(f"{timezone}: {_when(there)} (UTC{there.utcoffset().total_seconds() / 3600:+g}).")
        except (ZoneInfoNotFoundError, ValueError):
            lines.append(f"There is no time zone called {timezone!r}: use an IANA name like 'Europe/Paris', "
                         "or the weather tool, which gives a place's local time.")
    return "\n".join(lines)

"""Holly speaking up by himself: a timer or reminder come due, or a research report
finished. Each is a note he's handed as the crew's turn; if his reply fails (LM Studio
restarting, say), it's kept and tried again, after a minute, then two, four... up to a
quarter of an hour apart. Taken off the list and never said, a reminder was lost."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

RETRY_S, RETRY_MAX_S = 60, 900


def retry_wait(tries: int) -> float:
    """Seconds to wait before trying an announcement again, after `tries` failed."""
    return min(RETRY_S * 2 ** tries, RETRY_MAX_S)


@dataclasses.dataclass
class Announcement:
    what: str  # for the console: "due: pasta"
    note: str  # what Holly is told, as the crew's turn
    done: Callable[[bool], None]  # said (True), or not: kept, to be tried again


def timers_due(timers) -> Announcement | None:
    """Timers and reminders come due (taken off the list: put back if not said)."""
    from .timers import announcement

    if not timers or not (due := timers.due()):
        return None

    def done(ok: bool) -> None:
        if not ok:
            timers.put_back(due)

    return Announcement(f"due: {', '.join(i['label'] for i in due)}", announcement(due), done)


def report_ready(research) -> Announcement | None:
    """A research report finished (or given up), not yet announced, whose time has come."""
    from .research import announcement

    if not research or not (job := research.ready()):
        return None
    what = f"research {'finished' if job['stage'] == 'done' else 'given up'}: {job['title']}"
    return Announcement(what, announcement(job), lambda ok: research.announced(job, ok))


def pending(timers, research) -> Announcement | None:
    """What he should speak up about now, if anything: timers first, then reports."""
    return timers_due(timers) or report_ready(research)

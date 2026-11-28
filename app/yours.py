"""Everything of yours Holly keeps, and starting afresh without it: --forget sets it aside
(renamed, not deleted, in case it was a mistake), --reset deletes it for good. His history
and notes, the archive, timers and reminders, research and the event log. Not the ship's
records, which are the show's, nor the nets' counts, which are the models'."""

from __future__ import annotations

import shutil
from pathlib import Path

from .console import DIM, RESET
from .persona import set_aside

FORGET_HELP = ("start afresh, setting aside (renamed, not deleted) his history, notes, archive, timers, "
               "reminders, research and event log; the ship's records stay")
RESET_HELP = ("start afresh, deleting his history, notes, archive, timers, reminders, research and event log "
              "for good, and anything --forget set aside; the ship's records stay")


def events_folder(memory_file: Path | None) -> Path | None:
    """His event log (app/events.py): beside his memory, or none without it."""
    return memory_file.with_name("events") if memory_file else None


def start_afresh(cfg, forget: bool, reset: bool) -> None:
    """--reset or --forget, before he loads anything."""
    memory_file = cfg.persona.memory_file
    if reset:
        # Everything of yours, for good, and whatever --forget or an unreadable file set aside.
        gone = wipe_yours([memory_file, cfg.recall.file, cfg.timers.file]) + wipe_folder(cfg.research.dir)
        if events_folder(memory_file):
            gone += wipe_folder(events_folder(memory_file))
        print(f"{DIM}Deleted {len(gone)} file{'s' * (len(gone) != 1)} of Holly's memory"
              f"{': ' + ', '.join(p.name for p in gone) if gone else ''}; he starts afresh.{RESET}")
    elif forget:
        # Each set aside with its own time, so a second --forget doesn't overwrite the first.
        yours = [memory_file, cfg.recall.file, cfg.timers.file, cfg.research.dir, events_folder(memory_file)]
        asides = [set_aside_with_sidecars(f, "forgotten") for f in yours if f and f.exists()]
        if asides:
            print(f"{DIM}Holly's memory of earlier sessions is set aside "
                  f"({', '.join(a.name for a in asides)}); he starts afresh.{RESET}")


def sidecars(f: Path) -> list[Path]:
    """SQLite's files beside a database ("archive.db-journal"). They belong to it: a
    journal left by a crash would be rolled back into a new database of the same name."""
    return [s for s in (f.with_name(f.name + x) for x in ("-journal", "-wal", "-shm")) if s.exists()]


def set_aside_with_sidecars(f: Path, why: str) -> Path:
    aside = set_aside(f, why)
    for s in sidecars(f):
        s.replace(aside.with_name(aside.name + s.name[len(f.name):]))
    return aside


def wipe_yours(files: list[Path | None]) -> list[Path]:
    """Delete these files and any copies set aside from them (".forgotten-<time>",
    ".unreadable-<time>"), their half-written ".tmp" files, and SQLite's files
    beside them. Returns what went."""
    gone = []
    for f in files:
        if not f:
            continue
        for p in [f, f.with_suffix(".tmp"), *f.parent.glob(f"{f.stem}.forgotten-*{f.suffix}"),
                  *f.parent.glob(f"{f.stem}.unreadable-*{f.suffix}")]:
            for q in [p, *sidecars(p)] if p.exists() else sidecars(p):
                q.unlink()
                gone.append(q)
    return gone


def wipe_folder(d: Path) -> list[Path]:
    """Delete a folder of his (his research, his event log) and any copies --forget set aside."""
    gone = []
    for p in [d, *d.parent.glob(f"{d.name}.forgotten-*")] if d.parent.exists() else []:
        if p.is_dir():
            shutil.rmtree(p)
            gone.append(p)
    return gone

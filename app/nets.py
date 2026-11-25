"""The reply nets' counts from real use (app/guards.py NETS), kept per model between
sessions, beside his memory (data/nets.json). The evals count them in made-up talk; this is
what real talk trips, over afternoons, which is how several of them were found. A net a
model never trips in real use is one to take out when the brain changes (README: Evals).
It's about the model, not the crew: --forget and --reset leave it alone.

    python -m app.evals --nets    # the counts so far
"""

from __future__ import annotations

import datetime
import json
import os
from collections import Counter
from pathlib import Path

from .guards import NETS
from .persona import set_aside


class NetCounts:
    def __init__(self, file: Path | None):
        self.file = file
        self._failed = False

    def read(self) -> dict:
        """{model: {"since", "last", "replies", "nets": {net: count}}}; empty if there's none
        yet. An unreadable file is set aside, not lost."""
        if not self.file or not self.file.exists():
            return {}
        try:
            return json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"Couldn't read the reply nets' counts ({e}): set aside as {set_aside(self.file, 'unreadable').name}",
                  flush=True)
            return {}

    def add(self, model: str, fired: Counter, replies: int = 1) -> None:
        """A turn's nets, and its reply, added to `model`'s counts. Written whole (to a
        temporary file, then renamed over), so a crash can't leave half of it; a failure is
        said once, not every turn, and the turn goes on."""
        if not self.file:
            return
        counts = self.read()
        today = datetime.date.today().isoformat()
        m = counts.setdefault(model, {"since": today, "last": today, "replies": 0, "nets": {}})
        m["last"], m["replies"] = today, m["replies"] + replies
        for net, n in fired.items():
            if n:
                m["nets"][net] = m["nets"].get(net, 0) + n
        tmp = self.file.with_suffix(".tmp")
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(counts, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.file)
            self._failed = False
        except OSError as e:
            if not self._failed:
                print(f"Couldn't save the reply nets' counts to {self.file}: {e}", flush=True)
            self._failed = True


def report(counts: dict) -> str:
    """The counts, a table per model: every net, the ones at nought too."""
    if not counts:
        return "No reply nets counted in real use yet."
    out = []
    for model, m in counts.items():
        width = max(len(n) for n in NETS)
        out += [f"{model}: {m['replies']} replies, {m['since']} to {m['last']}"]
        out += [f"  {net.ljust(width)}  {m['nets'].get(net, 0)}" for net in NETS]
        never = sum(not m["nets"].get(net) for net in NETS)
        out += [f"  {never} of {len(NETS)} never fired", ""]
    return "\n".join(out).rstrip()

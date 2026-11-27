"""Holly's finished research reports, and finding one from the crew's words.

The reports themselves are markdown files beside the research jobs (app/research.py
writes them). Here, in data/research/reports.db (SQLite), is what finds them: each
one's title, request, summary and text, indexed for keywords (FTS5, stemmed), and a
vector of its title, request and summary from the same small embedding model recall
uses, for finding by meaning ("the one about getting fit" finds "Beginner running
plans"). Finished jobs given up are kept too, to be announced.

A few reports a day make a thousand a year, so finding one is a search, not a scan:
the keyword matches, the closest in meaning and the newest few are scored, each by
the words it shares with what was said (in its title most), how close it is in
meaning, and how recent it is (the crew mostly mean a recent one). Where up to four
score about as well as each other, Holly asks which.
"""

from __future__ import annotations

import datetime
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .archive import _DATE, DOC_PREFIX, QUERY_PREFIX, STOPWORDS, _unit, days_in
from .persona import _DAY_WORDS

# Words that don't say which report: "the report on heat pumps" is "heat pumps".
REPORT_NOISE = {"the", "a", "an", "on", "about", "of", "for", "my", "that", "this", "one", "report", "reports",
                "research", "summary", "open", "show", "me", "it", "you", "did", "do", "into", "read", "found",
                "asked", "wrote", "written", "finished", "got", "gave", "made", "from"}
# The best is taken as the one meant only if it scores SURE and the next best under MARGIN of it:
# "Strength Training for Longevity" at 10.9 beat "Beginner running plans" for "getting fit", and
# by a score alone, the wrong one would have been read. Short of that, Holly gets a shortlist,
# those scoring at least SHORTLIST of the best (0.3 let in noise), up to CHOICES of them; knowing what's
# being talked about, picks one, or asks the crew (offering at most four), or says there's none.
SURE, MARGIN = 3.0, 0.5
# Unless the crew's own words are in its title ("the espresso one", "Budget espresso machines"):
# then a lead of TITLED_MARGIN will do.
TITLED, TITLED_MARGIN = 2.0, 0.75
SHORTLIST, CHOICES = 0.5, 6
OFFER = 1.0  # nothing scoring less is offered at all
EXTRA_WEIGHT = 0.5  # Gemma's words for what's meant ("puppy" for "the dog one"), against the crew's own
OWN = 1.0  # a sure pick matches the crew's own words at least this much (one in its request or summary)
# With a sure pick, up to RUNNERS_UP next best scoring ALSO of it are named too, for a correction.
RUNNERS_UP, ALSO = 2, 0.3
CANDIDATES = 40  # from each of keywords and meaning, before scoring
NEWEST = 10  # and the newest few, always
# Meaning: nomic-embed's similarities bunch between 0.4 and 0.65 whether related or not, and
# a short "the dog one" scored "Puppy training basics" 0.46, below unrelated reports. So
# what counts is how far a report stands out from all the others (its z-score): past
# MEANING_Z, MEANING_WEIGHT a standard deviation.
MEANING_Z, MEANING_WEIGHT = 2.5, 1.0
# Recency: a report from today scores up to RECENT_BOOST more; half that a week on.
RECENT_BOOST, RECENT_HALF_DAYS = 0.3, 7.0
EMBED_RETRY_S = 60.0  # after the embedding model failed, finding by meaning is off this long
COLUMNS = ("n", "id", "title", "request", "summary", "file", "asked", "finished", "stage", "error", "announced",
           "tries", "retry", "work_s")


# A day the crew name a report by ("yesterday's report", "the one from Wednesday", "this
# morning's", "last week's"): one asked for or finished then. The day words were dropped
# from the search, so "yesterday's report" was the newest, whichever day that was; and
# "this morning" searched for "morning", and found nothing.
_TODAY = re.compile(r"\b(?:earlier\s+)?this\s+(?:morning|afternoon|evening)\b|\bearlier\s+on\b", re.IGNORECASE)
_WEEK = re.compile(r"\b(this|last)\s+week\b", re.IGNORECASE)
PART_OF_DAY = {"morning", "afternoon", "evening", "night", "earlier"}  # "yesterday morning" is yesterday's


def days_named(words: str, on: datetime.date) -> list[tuple[datetime.date, datetime.date]]:
    """The days `words` name, as spans (first, last), looking back from `on`."""
    spans = [(d, d) for d in days_in(words, on)]
    if _TODAY.search(words):
        spans.append((on, on))
    for m in _WEEK.finditer(words):
        monday = on - datetime.timedelta(days=on.weekday())
        spans.append((monday, on) if m.group(1).lower() == "this"
                     else (monday - datetime.timedelta(days=7), monday - datetime.timedelta(days=1)))
    return spans


def without_days(words: str) -> str:
    """What's left to say which report, the days taken out."""
    for pattern in (_WEEK, _TODAY, _DATE, _DAY_WORDS):
        words = pattern.sub(" ", words)
    return " ".join(w for w in words.split() if w.lower().strip("'s") not in PART_OF_DAY)


def telling(words: str) -> set[str]:
    """The words that say which report: "in" or "what" matched every one."""
    return {w for w in re.findall(r"\w+", words.lower()) if len(w) > 2} - REPORT_NOISE - STOPWORDS


def stems(text: str) -> set[str]:
    """Its words, plurals made single: "bike" matched "bikes" only as a near miss."""
    return {stem(w) for w in re.findall(r"\w+", text.lower())}


def stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def sure(scored: list) -> bool:
    """Is the best clearly the one: scoring SURE, the next best under MARGIN of it, and matching
    some of the crew's own words? Gemma's alone weren't enough: "llamas in Peru", widened to
    "Andes, South America", was sure of a report on places to stargaze."""
    if not scored or scored[0][0] < SURE or scored[0][2].get("own", 1) < OWN:
        return False
    margin = TITLED_MARGIN if scored[0][2].get("own", 0) >= TITLED else MARGIN
    return len(scored) == 1 or scored[1][0] < scored[0][0] * margin


class Reports:
    def __init__(self, path: Path, lm=None, embed_model: str = "", expand=None):
        """`expand(words)`, if given, says what a report the crew mean by `words` would
        contain, for a second look when the first isn't sure (Research._expand)."""
        self.expand = expand
        self.last_scores: list = []
        self.lm = lm
        self.embed_model = embed_model
        path.parent.mkdir(parents=True, exist_ok=True)
        # Added to from the event loop (as a job finishes), searched from a worker thread
        # (Holly's tool calls): one connection, one lock, as for his archive.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS reports (n INTEGER PRIMARY KEY, id TEXT UNIQUE, title TEXT, "
                            "request TEXT, summary TEXT, file TEXT, asked TEXT, finished TEXT, stage TEXT, "
                            "error TEXT, announced INTEGER DEFAULT 0, tries INTEGER DEFAULT 0, retry TEXT, "
                            "work_s REAL)")
            # rowid = reports.n
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS report_text USING fts5(title, request, summary, "
                            "body, tokenize='porter unicode61')")
            self.db.execute("CREATE TABLE IF NOT EXISTS report_vectors (n INTEGER, model TEXT, vec BLOB, "
                            "PRIMARY KEY (n, model))")
            self.db.commit()
            # Not yet announced: checked many times a second while he listens, so kept in memory.
            self.unannounced = [dict(r) for r in self.db.execute(
                f"SELECT {', '.join(COLUMNS)} FROM reports WHERE announced = 0 ORDER BY finished")]
        self._embed_retry_at = 0.0

    # Keeping ----------------------------------------------------------------

    def add(self, job: dict, body: str = "") -> dict:
        """A finished (or given up) job, with its report's text. Returns its row."""
        row = {k: job.get(k) for k in COLUMNS if k != "n"}
        row["announced"] = int(bool(row["announced"]))
        row["tries"] = row["tries"] or 0
        # Added again (its queue's save failed, and it was shelved again at the next start), a
        # report keeps its number and whether it was announced: replaced outright, it was given
        # a new number, its old text and vector were left behind, and it was announced twice.
        keep = ("n", "announced", "tries", "retry")
        with self.lock:
            self.db.execute(f"INSERT INTO reports ({', '.join(row)}) VALUES ({', '.join('?' * len(row))}) "
                            "ON CONFLICT(id) DO UPDATE SET "
                            + ", ".join(f"{k} = excluded.{k}" for k in row if k not in keep + ("id",)),
                            list(row.values()))
            stored = self.db.execute(f"SELECT {', '.join(keep)} FROM reports WHERE id = ?", (row["id"],)).fetchone()
            row.update(dict(stored))
            self.db.execute("DELETE FROM report_text WHERE rowid = ?", (row["n"],))
            self.db.execute("DELETE FROM report_vectors WHERE n = ?", (row["n"],))  # remade from what it is now
            self.db.execute("INSERT INTO report_text (rowid, title, request, summary, body) VALUES (?, ?, ?, ?, ?)",
                            (row["n"], row["title"], row["request"], row["summary"] or "", body))
            self.db.commit()
        self.unannounced = [r for r in self.unannounced if r["n"] != row["n"]]
        if not row["announced"]:
            self.unannounced.append(row)
        return row

    def embed_missing(self) -> int:
        """Vectors for finished reports that have none. Blocking: call it from a thread."""
        if not (self.embed_model and self.lm):
            return 0
        with self.lock:
            rows = self.db.execute("SELECT n, title, request, summary FROM reports WHERE stage = 'done' AND n NOT IN "
                                   "(SELECT n FROM report_vectors WHERE model = ?)", (self.embed_model,)).fetchall()
        for i in range(0, len(rows), 32):
            batch = rows[i:i + 32]
            vecs = self.lm.embed([DOC_PREFIX + about(r) for r in batch], self.embed_model)
            with self.lock:
                self.db.executemany("INSERT OR REPLACE INTO report_vectors (n, model, vec) VALUES (?, ?, ?)",
                                    [(r["n"], self.embed_model, _unit(v).tobytes()) for r, v in zip(batch, vecs)])
                self.db.commit()
        return len(rows)

    def clear(self) -> list[str]:
        """Every report's row, for good. Returns their files, for the caller to delete."""
        with self.lock:
            files = [r[0] for r in self.db.execute("SELECT file FROM reports WHERE file IS NOT NULL")]
            for table in ("reports", "report_text", "report_vectors"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.commit()
        self.unannounced = []
        return files

    # Listing ----------------------------------------------------------------

    def _rows(self, where: str = "1", args: tuple = (), order: str = "finished DESC", limit: int = -1) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(
                f"SELECT {', '.join(COLUMNS)} FROM reports WHERE {where} ORDER BY {order} LIMIT ?", (*args, limit))]

    def recent(self, days: float = 7, at_most: int = 10, now: datetime.datetime | None = None) -> tuple[list[dict], int]:
        """The reports finished in the last `days`, newest first (at most `at_most`), and how
        many others there are."""
        since = ((now or datetime.datetime.now()) - datetime.timedelta(days=days)).isoformat(timespec="seconds")
        rows = self._rows("finished >= ?", (since,), limit=at_most)
        with self.lock:
            total = self.db.execute("SELECT count(*) FROM reports").fetchone()[0]
        return rows, total - len(rows)

    def on(self, spans: list[tuple[datetime.date, datetime.date]], at_most: int = -1) -> list[dict]:
        """The finished reports asked for or finished on those days (days_named), newest first."""
        if not spans:
            return []
        either = " OR ".join(["substr(asked, 1, 10) BETWEEN ? AND ? OR substr(finished, 1, 10) BETWEEN ? AND ?"]
                             * len(spans))
        args = tuple(d.isoformat() for a, b in spans for d in (a, b, a, b))
        return self._rows(f"stage = 'done' AND ({either})", args, limit=at_most)

    def has(self, report_id: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM reports WHERE id = ?", (report_id,)).fetchone() is not None

    def titles(self, at_most: int = 6) -> list[str]:
        """The newest finished reports' titles, newest last."""
        return [r["title"] for r in reversed(self._rows("stage = 'done'", limit=at_most))]

    # Announcing -------------------------------------------------------------

    def ready(self, now: datetime.datetime | None = None) -> dict | None:
        """One finished (or given up) and not yet announced, whose time to try has come."""
        now = now or datetime.datetime.now()
        for r in self.unannounced:
            if not r["retry"] or datetime.datetime.fromisoformat(r["retry"]) <= now:
                return r
        return None

    def announced(self, row: dict, ok: bool, retry: datetime.datetime | None = None) -> None:
        """Said; or not (the reply failed), and to be tried again at `retry`."""
        if ok:
            self.unannounced = [r for r in self.unannounced if r["n"] != row["n"]]
            row["announced"] = 1
        else:
            row["tries"] = (row["tries"] or 0) + 1
            row["retry"] = retry.isoformat(timespec="seconds") if retry else None
        with self.lock:
            self.db.execute("UPDATE reports SET announced = ?, tries = ?, retry = ? WHERE n = ?",
                            (row["announced"], row["tries"], row["retry"], row["n"]))
            self.db.commit()

    # Finding ----------------------------------------------------------------

    def find(self, words: str, least: float = 0.0, now: datetime.datetime | None = None,
             widen: bool = True) -> list[dict]:
        """The finished reports the crew may mean, best first: the latest, with no words
        that say which; the one with exactly that title; else those scoring best (see the
        module's notes) and above `least`. One only when it's clearly the one (see sure);
        else a shortlist of up to CHOICES, for Holly to pick from, or ask about."""
        words = words.strip().strip("'\"")
        if exact := self._rows("stage = 'done' AND lower(title) = ?", (words.lower(),), limit=1):
            exact[0]["sure"] = True
            return exact
        now = now or datetime.datetime.now()
        spans = days_named(words, now.date())
        if spans:
            words = without_days(words)
        asked = telling(words)
        then = self.on(spans) if spans else []
        if not asked and spans:
            # Only a day: the one from then, sure if it's the only one; else those, newest first.
            if len(then) == 1:
                then[0]["sure"] = True
            return then[:CHOICES]
        if not asked:
            latest = self._rows("stage = 'done'", limit=1)
            for r in latest:
                r["sure"] = True  # "the report": the latest
            return latest
        # "The dog one" for "Puppy training basics": none of the crew's words in it, and too
        # short for its meaning to stand out. So Gemma says what such a report would contain
        # ("dog, puppy, training, pet"), and those are looked for too, counting for less than
        # the crew's own words. Always, not only when unsure: "the coffee report" was sure of
        # the caffeine report, and never looked at "Budget espresso machines".
        extra: set[str] = set()
        if self.expand and widen:
            try:
                extra = telling(self.expand(words)) - asked
            except Exception as e:  # LM Studio busy or gone: the crew's words alone
                print(f"  (couldn't widen the search for a report: {e!r})", flush=True)
        scored = self._scored(f"{words}: {', '.join(sorted(extra))}" if extra else words, asked, extra, now)
        if spans:
            # A day and what it was about ("the ten-year-old one from Saturday"): those from then,
            # if any match; if none do, the day was misremembered, and what it was about decides.
            ns = {r["n"] for r in then}
            scored = [x for x in scored if x[2]["n"] in ns] or scored
        self.last_scores = [(round(s, 2), r["title"]) for s, _, r in scored[:5]]  # for the console and tests
        floor = max(least, OFFER)
        if not scored or scored[0][0] <= floor:
            return []
        if sure(scored):
            picks = [scored[0][2]]
            picks[0]["sure"] = True  # else, even alone, Holly confirms it
            # Sure, but not alone: the next best are named, so "no, the running one" is easy.
            best = scored[0][0]
            picks[0]["also"] = [r["title"] for s, _, r in scored[1:1 + RUNNERS_UP] if s > floor and s >= best * ALSO]
        else:
            best = scored[0][0]
            picks = [r for s, _, r in scored if s > floor and s >= best * SHORTLIST][:CHOICES]
        for r in picks:
            r.pop("body", None)
            r.pop("own", None)
        return picks

    def _scored(self, words: str, asked: set[str], extra: set[str],
                now: datetime.datetime) -> list[tuple[float, str, dict]]:
        """The candidates (keyword matches, the closest in meaning, the newest) scored, best
        first; on a tie, the newest."""
        close = self._by_meaning(words)
        with self.lock:
            matched = [r[0] for r in self.db.execute(
                "SELECT rowid FROM report_text WHERE report_text MATCH ? ORDER BY bm25(report_text) LIMIT ?",
                (" OR ".join(f'"{w}"' for w in asked | extra), CANDIDATES))]
        newest = [r["n"] for r in self._rows("stage = 'done'", limit=NEWEST)]
        ns = set(matched) | set(close) | set(newest)
        if not ns:
            return []
        with self.lock:
            marks = ",".join("?" * len(ns))
            rows = [dict(r) for r in self.db.execute(
                f"SELECT {', '.join('r.' + c for c in COLUMNS)}, t.body FROM reports r JOIN report_text t "
                f"ON t.rowid = r.n WHERE r.stage = 'done' AND r.n IN ({marks})", list(ns))]
        return sorted(((score(r, asked, extra, close.get(r["n"], 0.0), now), r["finished"], r) for r in rows),
                      key=lambda x: (-x[0], -_stamp(x[1])))

    def _by_meaning(self, words: str) -> dict[int, float]:
        """The finished reports closest in meaning to `words`: {n: how far each stands out
        from all the reports, in standard deviations (its z-score)}."""
        if not (self.embed_model and self.lm) or time.monotonic() < self._embed_retry_at:
            return {}
        try:
            self.embed_missing()
            query = _unit(self.lm.embed([QUERY_PREFIX + words], self.embed_model)[0])
        except Exception as e:  # keywords and recency still work
            self._embed_retry_at = time.monotonic() + EMBED_RETRY_S
            print(f"  (finding reports by meaning is off for {EMBED_RETRY_S:.0f} s: {e!r})", flush=True)
            return {}
        with self.lock:
            rows = self.db.execute("SELECT v.n, v.vec FROM report_vectors v JOIN reports r ON r.n = v.n "
                                   "WHERE v.model = ? AND r.stage = 'done'", (self.embed_model,)).fetchall()
        if not rows:
            return {}
        sims = np.stack([np.frombuffer(r[1], np.float32) for r in rows]) @ query
        # With only a few reports their spread says little: nomic's typical one instead.
        mean, spread = (float(sims.mean()), max(float(sims.std()), 0.03)) if len(sims) >= 8 else (0.5, 0.05)
        best = np.argsort(-sims)[:CANDIDATES]
        return {int(rows[i][0]): (float(sims[i]) - mean) / spread for i in best}


def about(r) -> str:
    """What a report's vector is made from: what it's about, not all it says."""
    return f"{r['title']}. {r['request']}. {r['summary'] or ''}"


def score(r: dict, asked: set[str], extra: set[str], z: float, now: datetime.datetime) -> float:
    title = stems(r["title"])
    rest = stems(f"{r['request']} {r['summary'] or ''}")
    text = stems(r.get("body") or "")

    def shared(ws: set[str]) -> float:
        # A word stem in common counts too: "pumps" and "pump".
        ws = {stem(w) for w in ws}
        near = {a for a in ws if any(w.startswith(a[:4]) for w in title | rest)}
        return 2 * len(ws & title) + len(ws & rest) + 0.5 * len(near) + 0.5 * len(ws & text)

    r["own"] = shared(asked)  # the crew's own words alone, for whether a pick can be sure
    words = r["own"] + EXTRA_WEIGHT * shared(extra)
    meaning = max(0.0, z - MEANING_Z) * MEANING_WEIGHT
    age = max(0.0, (now - datetime.datetime.fromisoformat(r["finished"])).total_seconds() / 86400)
    return (words + meaning) * (1 + RECENT_BOOST * 0.5 ** (age / RECENT_HALF_DAYS))


def _stamp(iso: str | None) -> float:
    return datetime.datetime.fromisoformat(iso).timestamp() if iso else 0.0

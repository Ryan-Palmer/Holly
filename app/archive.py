"""Holly's archive: every exchange that leaves his history, kept and searchable.

His history holds recent conversation word for word, and his notes (app/memory.py)
a few lines on everything older. Neither keeps the detail: what a story was about,
which recipe he gave, what was said on Tuesday. So each exchange that leaves his
history (folded into notes, or dropped) is stored here, in SQLite, with when it
was said, and he can look things up with the `recall` tool, the way he searches
the web: the matching exchanges go to the same Gemma, which answers in two or
three sentences, out of his sight, so his context stays small.

Matching is three ways, merged:
- by keyword (SQLite's FTS5, with stemming: "cookies" finds "cookie");
- by meaning: each entry's vector from a small embedding model in LM Studio
  (nomic-embed, ~0.3 GB), so "her mother" finds "Jo's mum" and "what did we
  cook" finds the lasagne. Vectors are kept in the same SQLite file, and compared
  with numpy: thousands of entries take milliseconds, so no vector database;
- by day ("yesterday", "on Tuesday", "21 November"): all that was said then.
The judge sorts out the near misses each way brings.
"""

from __future__ import annotations

import datetime
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .config import RecallConfig
from .lmstudio import LmStudio
from .persona import WEEKDAYS, _DAY_WORDS, day, resolve_day, today
from .search import Lookup

TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Search your memory: earlier conversations with the crew, older than what you "
                       "remember word for word (what they told you, asked, planned or decided, what you "
                       "told them, what you found out), and the ship's records of Red Dwarf: its crew, "
                       "what has happened on board, places, creatures and technology.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description":
                "Key words for what to remember: names, things, places, and the day if it was "
                "asked about ('lasagne Jo','boiler service', 'what we talked about yesterday', "
                "'Rimmer hologram death', 'polymorph')."}},
            "required": ["query"],
        },
    },
}

JUDGE_PROMPT = """\
Today is {today}. You are the memory of Holly, the ship's computer from Red Dwarf.
Below are entries from his past conversations with the crew, each with when it
was said, and from the ship's records about Red Dwarf and its crew, then a
question about them. Use only what the entries say, never anything else. Reply
in exactly one of these forms:

ANSWER: two or three short, plain sentences that answer the question from the
entries, for someone to read aloud: no lists, no markdown. Say when it was, if
that helps. Names, numbers and dates exactly as the entries give them.
NOTHING: if the entries don't answer the question."""

# Words that match everything, so they're left out of the keyword search.
STOPWORDS = set("""
a an and are as at be been but by can could did do does for from had has have he her him his how i if in
is it its me my of on or our she so that the their them then there they this to was we were what when
where which who why will with would you your about tell remember recall said say talk talked told
holly ago last time conversation conversations earlier before again know did do any some
""".split()) | set(WEEKDAYS) | {"today", "tonight", "yesterday", "tomorrow"}

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
_DATE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\b", re.IGNORECASE)

ENTRY_CHARS = 1500  # of each entry, for the judge: a long story needn't be read whole
EMBED_CHARS = 2000  # of each entry, for its vector: nomic-embed reads 2,048 tokens at most
EMBED_BATCH = 32
# nomic-embed wants to be told which side of a search a text is on.
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "
DAY_ENTRIES = 20  # at most, for a day asked about


class Archive:
    def __init__(self, cfg: RecallConfig, lm: LmStudio, model: str, file: Path | None = None,
                 records: Archive | None = None):
        """His conversations' archive (`cfg.file`), or another store of entries
        with `file` (the ship's records, app/records.py). `records`, if given, is
        searched by recall too."""
        self.cfg = cfg
        self.lm = lm
        self.model = model
        self.records = records
        file = file or cfg.file
        file.parent.mkdir(parents=True, exist_ok=True)
        # Written from the event loop (as history is folded), searched from a worker
        # thread (as Holly recalls): one connection, one lock.
        self.db = sqlite3.connect(file, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS entries "
                            "USING fts5(text, at UNINDEXED, tokenize='porter unicode61')")
            # Each entry's vector, by the entry's rowid and the model that made it.
            self.db.execute("CREATE TABLE IF NOT EXISTS vectors "
                            "(id INTEGER, model TEXT, vec BLOB, PRIMARY KEY (id, model))")
            self.db.commit()
        self._matrix: tuple[int, np.ndarray, np.ndarray] | None = None  # (count, ids, unit vectors)
        self._embed_failed = False

    # Keeping --------------------------------------------------------------

    def add(self, exchanges: list[tuple[str | None, str]]) -> None:
        """Store exchanges: (when it was said, ISO, or None; the exchange as text)."""
        if not exchanges:
            return
        with self.lock:
            self.db.executemany("INSERT INTO entries (at, text) VALUES (?, ?)", exchanges)
            self.db.commit()

    def entry(self, n: int) -> str | None:
        """The nth entry's text (for picking one at random)."""
        with self.lock:
            row = self.db.execute("SELECT text FROM entries LIMIT 1 OFFSET ?", (n,)).fetchone()
        return row[0] if row else None

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT count(*) FROM entries").fetchone()[0]

    def clear(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM entries")
            self.db.execute("DELETE FROM vectors")
            self.db.commit()
            self._matrix = None

    def embed_missing(self) -> int:
        """Make vectors for entries that have none (new ones, or ones from before
        the embedding model was set or changed). Blocking: call it from a thread.
        Returns how many were made."""
        model = self.cfg.embed_model
        if not model:
            return 0
        with self.lock:
            rows = self.db.execute("SELECT rowid, text FROM entries WHERE rowid NOT IN "
                                   "(SELECT id FROM vectors WHERE model = ?)", (model,)).fetchall()
        for i in range(0, len(rows), EMBED_BATCH):
            batch = rows[i:i + EMBED_BATCH]
            vecs = self.lm.embed([DOC_PREFIX + text[:EMBED_CHARS] for _, text in batch], model)
            with self.lock:
                self.db.executemany("INSERT OR REPLACE INTO vectors (id, model, vec) VALUES (?, ?, ?)",
                                    [(rid, model, _unit(v).tobytes()) for (rid, _), v in zip(batch, vecs)])
                self.db.commit()
        return len(rows)

    def _by_meaning(self, texts: list[str]) -> list[int]:
        """Rowids of the entries closest in meaning to any of `texts`, best first."""
        model = self.cfg.embed_model
        if not model or self._embed_failed:
            return []
        try:
            self.embed_missing()
            queries = np.stack([_unit(v) for v in self.lm.embed([QUERY_PREFIX + t for t in texts], model)])
        except Exception as e:  # the model isn't in LM Studio: keywords and days still work
            self._embed_failed = True
            print(f"  (recall by meaning is off: {e!r})", flush=True)
            return []
        with self.lock:
            count = self.db.execute("SELECT count(*) FROM vectors WHERE model = ?", (model,)).fetchone()[0]
            if not self._matrix or self._matrix[0] != count:
                rows = self.db.execute("SELECT id, vec FROM vectors WHERE model = ?", (model,)).fetchall()
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                mat = (np.stack([np.frombuffer(r[1], np.float32) for r in rows]) if rows
                       else np.zeros((0, queries.shape[1]), np.float32))
                self._matrix = (count, ids, mat)
        _, ids, mat = self._matrix
        if not len(ids):
            return []
        # The closest for any of the texts (the search words, and what was asked).
        sims = (mat @ queries.T).max(axis=1)
        best = np.argsort(-sims)[: self.cfg.meaning_entries]
        return [int(ids[i]) for i in best if sims[i] >= self.cfg.min_similarity]

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # Finding --------------------------------------------------------------

    def search(self, query: str, on: datetime.date | None = None,
               question: str = "") -> list[tuple[str | None, str]]:
        """Entries from days `query` names, matching its key words, and closest in
        meaning to it or to `question` (what was asked). `on` is today, for working
        out the days."""
        on = on or datetime.date.today()
        words = []
        for w in re.findall(r"[a-z0-9']+", query.lower()):
            w = w.removesuffix("'s").strip("'")
            if len(w) > 1 and w not in STOPWORDS and w not in MONTHS and w not in words:
                words.append(w)
        found: dict[int, tuple[str | None, str]] = {}
        with self.lock:
            for d in sorted(days_in(query, on)):
                rows = self.db.execute("SELECT rowid, at, text FROM entries WHERE substr(at, 1, 10) = ? "
                                       "ORDER BY at LIMIT ?", (d.isoformat(), DAY_ENTRIES)).fetchall()
                found |= {r[0]: (r[1], r[2]) for r in rows}
            if found:
                # A day was asked about, and he talked then: that day is the answer. Close
                # matches from other days made the judge answer "yesterday" with Thursday.
                return list(found.values())
            if words:
                # Each word, and for longer ones any word starting the same, less its last
                # letter: "lasagna" finds "lasagne", "colour" finds "colours".
                match = " OR ".join(f'"{w}"' + (f' OR "{w[:-1]}"*' if len(w) >= 5 else "") for w in words)
                rows = self.db.execute("SELECT rowid, at, text FROM entries WHERE entries MATCH ? "
                                       "ORDER BY bm25(entries) LIMIT ?", (match, self.cfg.max_entries)).fetchall()
                for r in rows:
                    found.setdefault(r[0], (r[1], r[2]))
        if close := [i for i in self._by_meaning([t for t in (query, question) if t]) if i not in found]:
            with self.lock:
                marks = ",".join("?" * len(close))
                rows = self.db.execute(f"SELECT rowid, at, text FROM entries WHERE rowid IN ({marks})",
                                       close).fetchall()
            found |= {r[0]: (r[1], r[2]) for r in rows}
        return list(found.values())

    def recall(self, query: str, question: str) -> Lookup:
        """Look `query` up, and answer `question` (what was asked) from what's found."""
        t0 = time.perf_counter()
        hits = self.search(query, question=question)
        days = sorted(days_in(f"{query} {question}", datetime.date.today()))
        if self.records and not days:  # a day asked about is about a conversation
            hits += self.records.search(query, question=question)
        t1 = time.perf_counter()
        if not hits:
            return self._nothing(query, t1 - t0, 0)
        entries = "\n\n".join(text[:ENTRY_CHARS] for _, text in sorted(hits, key=lambda h: h[0] or ""))
        about = f"\n(The question is about {', '.join(day(d) for d in days)}.)" if days else ""
        reply = self.lm.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_PROMPT.format(today=today())},
                      {"role": "user", "content": f"{entries}\n\nQuestion: {question}{about}"}],
            temperature=0.2,
            max_tokens=self.cfg.summary_max_tokens,
            reasoning_effort="none",
        )
        text = (reply.choices[0].message.content or "").strip()
        t2 = time.perf_counter()
        head, _, rest = text.partition(":")
        if not text or head.strip().upper().startswith("NOTHING"):
            return self._nothing(query, t1 - t0, t2 - t1, len(hits))
        answer = rest.strip() if head.strip().upper() == "ANSWER" else text
        return Lookup(query, answer, len(hits), (t1 - t0) * 1000, (t2 - t1) * 1000, [query])

    @staticmethod
    def _nothing(query: str, search_s: float, judge_s: float, n: int = 0) -> Lookup:
        # As with search: told only that nothing was found, he'd fill the gap himself.
        why = "Your memory has nothing on that." if not n else "Your memory doesn't say."
        return Lookup(query, f"{why} Tell the crew you don't remember, in your own words. Don't make "
                             "up what was said, or when.", 0, search_s * 1000, judge_s * 1000, [query])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def days_in(text: str, on: datetime.date) -> set[datetime.date]:
    """The days `text` names, looking back from `on`: "yesterday", "on Tuesday" (the last
    one), "21 November" (this year's, or last year's if that's still to come)."""
    days = set()
    for m in _DAY_WORDS.finditer(text):
        w = m.group(1).lower()
        if "tomorrow" not in w:
            days.add(resolve_day(m.group(1), on, past=True))
    for m in _DATE.finditer(text):
        try:
            d = datetime.date(on.year, MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)))
        except ValueError:
            continue
        days.add(d if d <= on else d.replace(year=on.year - 1))
    return days

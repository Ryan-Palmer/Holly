"""Holly's archive: every exchange that leaves his history, kept and searchable.

His history holds recent conversation word for word, and his notes (app/memory.py)
a few lines on everything older. Neither keeps the detail: what a story was about,
which recipe he gave, what was said on Tuesday. So each exchange that leaves his
history (folded into notes, or dropped) is stored here, in SQLite, with when it
was said, and he can look things up with the `recall` tool, the way he searches
the web: the matching exchanges go to the same Gemma, which answers in two or
three sentences, out of his sight, so his context stays small.

Matching is four ways, merged:
- by name: a page of the ship's records that the search or the question names
  ("Better Than Life", "Kryten"), its opening and the parts that fit, first.
  Without it, "when I met Kryten" found Nova 5 and the Series 4000s, every
  page saying "met" and "Kryten", and not the episode where they met;
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
            "properties": {
                "query": {"type": "string", "description":
                    "Key words for what to remember: names as they're spelt, things, places, and the "
                    "day if it was asked about ('lasagne Jo', 'boiler service', 'what we talked "
                    "about yesterday', 'Rimmer hologram death', 'Kryten first meeting', 'Better Than "
                    "Life')."},
                "question": {"type": "string", "description":
                    "The question to answer from what's found, in full: 'When did the crew first "
                    "meet Kryten?', 'What happened in Better Than Life?'"},
            },
            "required": ["query", "question"],
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
that helps. Names, numbers and dates exactly as the entries give them. For a
broad question ("tell me about your adventures with Lister"), give two or three
of the main things the entries say. If they answer only part of it, answer
that part. Asked for the worst, best, strangest or most dangerous, pick from
what the entries show, as Holly's own choice: no entry needs to call it that.
The ship's records describe Red Dwarf from outside, as a television series
with novels and other versions; Holly lived it. Tell it as what happened, and
never mention episodes, series, the show, novels, books, actors or versions.
Where the entries differ, go with the television account, without saying so.
A question about an episode is about what happened then; a name can be the
one someone went by there (in a game, a disguise, another reality).
The question is Holly's, put to his own memory: "you" in it is Holly, and "we"
Holly and the crew, so "when did you first meet Kryten?" asks when Holly and
the crew did, as the ship's records tell it.
NOTHING: only if none of the entries is about the question at all."""

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
NAMED_PAGES = 3  # pages of the ship's records found by name, at most
PAGE_OPENING = 3  # entries from the start of each: what it is, the episode's overview
PAGE_MATCHES = 3  # and its entries that best fit the other words asked
RECORD = "(Ship's records: "
THE_EPISODE = re.compile(r"\s*\bthe episode\b", re.IGNORECASE)
# How speech recognition has spelt names, should one get past Whisper's hints (holly.toml).
MISHEARD = re.compile(r"\b(?:Crichton|Critton|Crittan|Kryton|Kriten|Krighton)\b", re.IGNORECASE)
EMBED_RETRY_S = 60.0  # after the embedding model failed (LM Studio restarting, a timeout), try again then


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
        self._pages: list[tuple[str, re.Pattern, list[str]]] | None = None  # (name, its pattern, its pages)
        self._embed_retry_at = 0.0  # after a failure, recall by meaning is off until then

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
        if not model or time.monotonic() < self._embed_retry_at:
            return []
        try:
            self.embed_missing()
            queries = np.stack([_unit(v) for v in self.lm.embed([QUERY_PREFIX + t for t in texts], model)])
        except Exception as e:  # keywords and days still work; a passing failure shouldn't end it for good
            self._embed_retry_at = time.monotonic() + EMBED_RETRY_S
            print(f"  (recall by meaning is off for {EMBED_RETRY_S:.0f} s: {e!r})", flush=True)
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

    def _page_index(self) -> list[tuple[str, re.Pattern, list[str]]]:
        """The ship's records' pages by name, longest first: ("better than life",
        its pattern, ["Better Than Life (episode)", "Better Than Life", ...]), the
        episode first. A surname is a name too, for the biggest page of that name
        ("rimmer": Arnold Rimmer, not Ace or Howard). Names under 3 letters ("Up",
        "Io") and disambiguation pages are left out. Empty for his conversations'
        archive."""
        if self._pages is None:
            with self.lock:
                rows = self.db.execute("SELECT substr(text, 1, 300) FROM entries WHERE text LIKE ? ORDER BY rowid",
                                       (RECORD + "%",)).fetchall()
            pages: dict[str, list[str]] = {}
            size: dict[str, int] = {}  # entries per page
            for (head,) in rows:
                title = head.removeprefix(RECORD).split(") ", 1)[0].split(", ")[0]
                name = re.sub(r"\s*\([^)]*\)", "", title).strip().lower()
                size[title] = size.get(title, 0) + 1
                if len(name) >= 3 and "disambiguation" not in title.lower() and title not in pages.get(name, []):
                    pages.setdefault(name, []).append(title)
            # "Lister", "Rimmer", "Kochanski": the biggest page of a person's name ending so.
            people: dict[str, str] = {}
            for title in size:
                words = re.sub(r"\s*\([^)]*\)", "", title).split()
                if len(words) > 1 and all(w[:1].isupper() for w in words) and len(words[-1]) >= 4:
                    last = words[-1].lower()
                    if last not in pages and size[title] > size.get(people.get(last, ""), 0):
                        people[last] = title
            for last, title in people.items():
                pages[last] = [title]
            self._pages = sorted(
                ((name, re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE),
                  sorted(titles, key=lambda t: (not t.endswith("(episode)"), "(" in t)))
                 for name, titles in pages.items()), key=lambda x: -len(x[0]))
        return self._pages

    def _named(self, texts: list[str]) -> list[str]:
        """Pages of the records named in `texts`, longest name first. A one-word name
        counts only with a capital ("Kryten", "Polymorph"), so "the death of" isn't
        the page Death, and a three-letter one only mid-sentence too ("the Cat", not
        "Out of interest"); a name inside a longer one found ("Life" in "Better
        Than Life") doesn't count."""
        names: list[str] = []
        titles: list[str] = []
        for name, pattern, pages in self._page_index():
            if any(name in n for n in names):
                continue
            if not any((m := pattern.search(t)) and (" " in name or m.group(0)[0].isupper())
                       and (len(name) > 3 or m.start() > 0) for t in texts):
                continue
            names.append(name)
            titles += pages[:2]
            if len(titles) >= NAMED_PAGES:
                break
        return titles[:NAMED_PAGES]

    def _page_entries(self, title: str, words: list[str]) -> list[tuple[int, str | None, str]]:
        """A page's opening entries, then those fitting `words` best."""
        def like(x: str) -> str:
            return x.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        on_page = "(text LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\')"
        args = (like(RECORD + title) + ") %", like(RECORD + title) + ", %")
        with self.lock:
            rows = self.db.execute(f"SELECT rowid, at, text FROM entries WHERE {on_page} ORDER BY rowid LIMIT ?",
                                   (*args, PAGE_OPENING)).fetchall()
            if words:
                rows += self.db.execute(
                    f"SELECT rowid, at, text FROM entries WHERE entries MATCH ? AND {on_page} "
                    "ORDER BY bm25(entries) LIMIT ?", (" OR ".join(f'"{w}"' for w in words), *args,
                                                       PAGE_MATCHES)).fetchall()
        return rows

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # Finding --------------------------------------------------------------

    def search(self, query: str, on: datetime.date | None = None,
               question: str = "") -> list[tuple[str | None, str]]:
        """Entries from days `query` or `question` (what was asked) names, matching
        `query`'s key words, and closest in meaning to either. `on` is today, for
        working out the days."""
        on = on or datetime.date.today()

        def key_words(text: str) -> list[str]:
            # A month is left out only as part of a date ("21 November"): April, May and
            # June are names too, and a search for one alone had no words left.
            # "may", without its capital, is the verb.
            out = []
            for w in re.findall(r"[A-Za-z0-9']+", _DATE.sub(" ", text)):
                if w == "may":
                    continue
                w = w.lower().removesuffix("'s").strip("'")
                if len(w) > 1 and w not in STOPWORDS and w not in out:
                    out.append(w)
            return out

        words = key_words(query)
        found: dict[int, tuple[str | None, str]] = {}
        with self.lock:
            # The day may be in either: asked "what did we talk about yesterday?", he
            # searched for "conversation topics", and yesterday was never looked at.
            for d in sorted(days_in(f"{query} {question}", on)):
                rows = self.db.execute("SELECT rowid, at, text FROM entries WHERE substr(at, 1, 10) = ? "
                                       "ORDER BY at LIMIT ?", (d.isoformat(), DAY_ENTRIES)).fetchall()
                found |= {r[0]: (r[1], r[2]) for r in rows}
            if found:
                # A day was asked about, and he talked then: that day is the answer. Close
                # matches from other days made the judge answer "yesterday" with Thursday.
                return list(found.values())
        texts = [t for t in (query, question) if t]
        if titles := self._named(texts):
            # The pages named, first: their opening, and their parts that fit the rest.
            named = {w for t in titles for w in key_words(re.sub(r"\([^)]*\)", "", t))}
            rest = [w for w in key_words(" ".join(texts)) if w not in named]
            for title in titles:
                for r in self._page_entries(title, rest):
                    found.setdefault(r[0], (r[1], r[2]))
        with self.lock:
            if words:
                # Each word, and for longer ones any word starting the same, less its last
                # letter: "lasagna" finds "lasagne", "colour" finds "colours".
                match = " OR ".join(f'"{w}"' + (f' OR "{w[:-1]}"*' if len(w) >= 5 else "") for w in words)
                rows = self.db.execute("SELECT rowid, at, text FROM entries WHERE entries MATCH ? "
                                       "ORDER BY bm25(entries) LIMIT ?", (match, self.cfg.max_entries)).fetchall()
                for r in rows:
                    found.setdefault(r[0], (r[1], r[2]))
        if close := [i for i in self._by_meaning(texts) if i not in found]:
            with self.lock:
                marks = ",".join("?" * len(close))
                rows = self.db.execute(f"SELECT rowid, at, text FROM entries WHERE rowid IN ({marks})",
                                       close).fetchall()
            found |= {r[0]: (r[1], r[2]) for r in rows}
        return list(found.values())

    def recall(self, query: str, question: str, said: str = "") -> Lookup:
        """Look `query` up, and answer `question` (what Holly asks, or failing that
        what the crew said) from what's found, and from `said`: the crew's lines
        still in his history (Conversation.crew_lines), which aren't archived yet."""
        t0 = time.perf_counter()
        query, question = MISHEARD.sub("Kryten", query), MISHEARD.sub("Kryten", question)
        # "The Cat's name in the episode Gunmen of the Apocalypse?" got NOTHING from the
        # judge, taking it as the Cat's own name ("no name other than The Cat"); without
        # "the episode", "The Riviera Kid".
        question = THE_EPISODE.sub(" ", question).replace("  ", " ")
        hits = self.search(query, question=question)
        days = sorted(days_in(f"{query} {question}", datetime.date.today()))
        if self.records and not days:  # a day asked about is about a conversation
            hits += self.records.search(query, question=question)
        if said:
            hits.insert(0, (None, f"(Said in this conversation, by the crew:)\n{said}"))
        t1 = time.perf_counter()
        if not hits:
            return self._nothing(query, t1 - t0, 0)
        entries = "\n\n".join(text[:ENTRY_CHARS] for _, text in sorted(hits, key=lambda h: h[0] or ""))
        about = f"\n(The question is about {', '.join(day(d) for d in days)}.)" if days else ""
        reply = self.lm.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_PROMPT.format(today=today())},
                      {"role": "user", "content": f"{entries}\n\nQuestion: {question}\n(Searched for: "
                                                  f"{query}){about}"}],
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
        # "Tell the crew you don't remember", said outright, he obeyed even with the
        # answer in his history: asked who was coming over, he said he didn't know.
        why = "Your archive has nothing on that." if not n else "Your archive doesn't say."
        return Lookup(query, f"{why} If this conversation or your notes say, answer from them. If not, "
                             "tell the crew you don't remember, in your own words. Don't make up what was "
                             "said, or when.", 0, search_s * 1000, judge_s * 1000, [query])


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

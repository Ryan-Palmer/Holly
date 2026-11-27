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

import asyncio
import datetime
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .config import RecallConfig
from .console import DIM, RESET
from .lmstudio import LmStudio
from .persona import WEEKDAYS, _DAY_WORDS, day, resolve_day, today
from .search import Lookup, Searcher
from .tools import Lookup as LookupTool, Turn

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
                    "about yesterday', 'what the crew had for tea last night', 'Rimmer hologram "
                    "death', 'Kryten first meeting', 'Better Than Life'). Whoever you're talking "
                    "to is 'the crew' here, never their name: that's what your memory calls them."},
                "question": {"type": "string", "description":
                    "The question to answer from what's found, in full: 'When did the crew first "
                    "meet Kryten?', 'What did the crew have for tea last night?', 'What happened "
                    "in Better Than Life?'"},
            },
            "required": ["query", "question"],
        },
    },
}

JUDGE_PROMPT = """\
Today is {today}. You are the memory of Holly, the ship's computer from Red Dwarf.
Below are his notes on earlier conversations, if he has any (kept up to date,
and right), entries from his past conversations with the crew, each with when
it was said, and from the ship's records about Red Dwarf and its crew, then a
question about them. What the notes say counts as much as any entry, and
where an entry tells only part of something, the notes may tell the rest. Use only what the entries say, never anything else. Reply
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
The question is Holly's, put to his own memory: "you", "I" and "me" in it are
Holly, and "we" Holly and the crew, so "when did you first meet Kryten?" asks
when Holly and the crew did, as the ship's records tell it. Holly meets people
by being talked to and told about them: "did I meet Jo?" is answered yes if
she was there or talked about, and what was said of her.
Asked about a day, answer with what was said and done in that day's entries, across
the whole day if the question is broad. Words like "yesterday" or "tonight" count
from when they were said: the question's from today, an entry's from its own day,
and the date in square brackets after one says which day it meant. If that's the
day asked about, it happened then; if it's another day, it's that day's news, only
mentioned then: "we made lasagne yesterday [Sunday 8 March]", said on Monday,
happened on Sunday, so it isn't what was done on Monday.
If the entries mention who or what was asked about but don't answer it exactly,
say what they do say about them.
NOTHING: only if none of the entries mentions who or what was asked about."""

# Words that match everything, so they're left out of the keyword search.
STOPWORDS = set("""
a an and are as at be been but by can could did do does for from had has have he her him his how i if in
is it its me my of on or our she so that the their them then there they this to was we were what when
where which who why will with would you your about tell remember recall said say talk talked told
holly ago last time conversation conversations earlier before again know did do any some
""".split()) | set(WEEKDAYS) | {"today", "tonight", "yesterday", "tomorrow"}
# And in his conversations, where every entry says "Crew:" (not in the ship's records, where
# "crew" is a word worth finding, nor in reports.py, which shares STOPWORDS).
TALK_STOPWORDS = STOPWORDS | {"crew"}
EARLIER_PICKED = 4  # lines said earlier on a day about "tonight", for a question about that night

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
# "21 November", "the 21st of November": said the second way, it wasn't read as a day.
_DATE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(MONTHS) + r")\b", re.IGNORECASE)

ENTRY_CHARS = 1500  # of each entry, for the judge: a long story needn't be read whole
EMBED_CHARS = 2000  # of each entry, for its vector: nomic-embed reads 2,048 tokens at most
EMBED_BATCH = 32
# nomic-embed wants to be told which side of a search a text is on.
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "
OFFER_EACH = 3  # memories to offer him: the best this many by keyword, and in meaning
OFFER_WORDS = 4  # the crew's words in one, at least
OFFER_CHARS = 300  # of the crew's words in one, at most
DAY_ENTRIES = 20  # at most, for a day asked about
DAY_PICKED = 12  # of them, at most, matching what was asked: the rest spread across the day
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
                 records: Archive | None = None, talk: bool = True):
        """His conversations' archive (`cfg.file`), or another store of entries
        with `file` (the ship's records, app/records.py: `talk` False). `records`,
        if given, is searched by recall too."""
        self.cfg = cfg
        self.lm = lm
        self.model = model
        self.records = records
        self.stopwords = TALK_STOPWORDS if talk else STOPWORDS
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
        self.vectors_read = 0  # from the database, in all (_vectors): for the tests
        self._embedded_to: tuple[str, int | None] | None = None  # (model, newest rowid) all embedded up to
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
            self._embedded_to = None

    def embed_missing(self) -> int:
        """Make vectors for entries that have none (new ones, or ones from before
        the embedding model was set or changed). Blocking: call it from a thread.
        Returns how many were made."""
        model = self.cfg.embed_model
        if not model:
            return 0
        with self.lock:
            # Asked before every search by meaning (a memory offered: every reply): looking for
            # entries without a vector was 81 ms at five years' archive. Nothing added since it
            # last looked (the newest rowid the same), nothing to look for.
            top = (model, self.db.execute("SELECT max(rowid) FROM entries").fetchone()[0])
            if top == self._embedded_to:
                return 0
            rows = self.db.execute("SELECT rowid, text FROM entries WHERE rowid NOT IN "
                                   "(SELECT id FROM vectors WHERE model = ?)", (model,)).fetchall()
        for i in range(0, len(rows), EMBED_BATCH):
            batch = rows[i:i + EMBED_BATCH]
            vecs = self.lm.embed([DOC_PREFIX + text[:EMBED_CHARS] for _, text in batch], model)
            with self.lock:
                self.db.executemany("INSERT OR REPLACE INTO vectors (id, model, vec) VALUES (?, ?, ?)",
                                    [(rid, model, _unit(v).tobytes()) for (rid, _), v in zip(batch, vecs)])
                self.db.commit()
        self._embedded_to = top
        return len(rows)

    def warm(self) -> None:
        """Vectors made and loaded now (while he sleeps, after a fold; at startup), not on the
        next search: a first search at a year's archive took ~0.2 s, at five ~1 s."""
        model = self.cfg.embed_model
        if not model:
            return
        self.embed_missing()
        with self.lock:
            count = self.db.execute("SELECT count(*) FROM vectors WHERE model = ?", (model,)).fetchone()[0]
            if count and (not self._matrix or self._matrix[0] != count):
                dims = len(self.db.execute("SELECT vec FROM vectors WHERE model = ? LIMIT 1", (model,)).fetchone()[0]) // 4
                self._matrix = self._vectors(model, count, dims)

    def _vectors(self, model: str, count: int, dims: int) -> tuple[int, np.ndarray, np.ndarray]:
        """Every entry's vector, as (count, ids, matrix): only the new ones read, if the old are
        held. Read whole after each fold, it was 190 ms at a year's archive (45,000 entries)
        and 1 s at five, on the first search after he slept: often the memory offered, which
        gives up at 400 ms. New entries come last (rowids rise); anything else (vectors made
        for old entries, a new model, a clear) and it's read whole. Called holding the lock."""
        held = self._matrix
        if held and count > held[0] and len(held[1]):
            rows = self.db.execute("SELECT id, vec FROM vectors WHERE model = ? AND id > ? ORDER BY id",
                                   (model, int(held[1].max()))).fetchall()
            if held[0] + len(rows) == count:
                self.vectors_read += len(rows)
                return (count, np.concatenate([held[1], np.array([r[0] for r in rows], dtype=np.int64)]),
                        np.vstack([held[2], np.stack([np.frombuffer(r[1], np.float32) for r in rows])]))
        rows = self.db.execute("SELECT id, vec FROM vectors WHERE model = ?", (model,)).fetchall()
        self.vectors_read += len(rows)
        return (count, np.array([r[0] for r in rows], dtype=np.int64),
                np.stack([np.frombuffer(r[1], np.float32) for r in rows]) if rows
                else np.zeros((0, dims), np.float32))

    def _by_meaning(self, texts: list[str], among: set[int] | None = None) -> list[int]:
        """Rowids of the entries closest in meaning to any of `texts`, best first; only
        those in `among`, if given (a day's entries)."""
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
                self._matrix = self._vectors(model, count, queries.shape[1])
        _, ids, mat = self._matrix
        if not len(ids):
            return []
        # The closest for any of the texts (the search words, and what was asked).
        sims = (mat @ queries.T).max(axis=1)
        if among is not None:
            sims = np.where(np.isin(ids, np.fromiter(among, np.int64, len(among))), sims, -1.0)
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

    def page(self, title: str) -> list[str]:
        """Every entry of a page of the ship's records, in order, each as "(Ship's records:
        Title, Section) text"; none if there's no such page."""
        def like(x: str) -> str:
            return x.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        with self.lock:
            rows = self.db.execute("SELECT text FROM entries WHERE text LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\' "
                                   "ORDER BY rowid", (like(RECORD + title) + ") %", like(RECORD + title) + ", %"))
            return [r[0] for r in rows.fetchall()]

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # Finding --------------------------------------------------------------

    def _key_words(self, text: str) -> list[str]:
        """The words of `text` to search for."""
        # A month is left out only as part of a date ("21 November"): April, May and
        # June are names too, and a search for one alone had no words left.
        # "may", without its capital, is the verb.
        # Day words too ("last night", "until Monday"): the day is found by days_in, and
        # "night" as a word to match would pick that day's "good night" first.
        out = []
        for w in re.findall(r"[A-Za-z0-9']+", _DATE.sub(" ", _DAY_WORDS.sub(" ", text))):
            if w == "may":
                continue
            w = w.lower().removesuffix("'s").strip("'")
            if len(w) > 1 and w not in self.stopwords and w not in out:
                out.append(w)
        return out

    def memories(self, line: str, n: int = OFFER_EACH) -> list[tuple[int, str, str]]:
        """What the crew said on other days that might go with `line`, to offer him: the
        best `n` by keyword and the closest `n` in meaning, each (rowid, its day, their
        words). Their words only: not his old replies, to copy, nor a tool's answer with its
        instructions to him then; and the day, not the time (a time in his reply is cut
        unless asked for). Nothing too short to be about anything ("Evening, Holly")."""
        ids: list[int] = []
        if words := self._key_words(line):
            with self.lock:
                ids += [r[0] for r in self.db.execute(
                    "SELECT rowid FROM entries WHERE entries MATCH ? ORDER BY bm25(entries) LIMIT ?",
                    (" OR ".join(f'"{w}"' for w in words), n))]
        ids += [i for i in self._by_meaning([line])[:n] if i not in ids]
        if not ids:
            return []
        with self.lock:
            texts = dict(self.db.execute(f"SELECT rowid, text FROM entries WHERE rowid IN ({','.join('?' * len(ids))})",
                                         ids).fetchall())
        out = []
        for i in ids:
            day, said = crew_said(texts.get(i, ""))
            if len(said.split()) >= OFFER_WORDS:
                out.append((i, day, said))
        return out

    def search(self, query: str, on: datetime.date | None = None,
               question: str = "") -> list[tuple[str | None, str]]:
        """Entries from days `query` or `question` (what was asked) names, matching
        `query`'s key words, and closest in meaning to either. `on` is today, for
        working out the days."""
        on = on or datetime.date.today()
        key_words = self._key_words
        words = key_words(query)
        found: dict[int, tuple[str | None, str]] = {}
        # The day may be in either: asked "what did we talk about yesterday?", he searched
        # for "conversation topics", and yesterday was never looked at. Of that day's
        # entries, those matching the words asked about come first, then those closest in
        # meaning ("music" finds "the album by..."), up to DAY_PICKED; the rest of the
        # DAY_ENTRIES are spread across the day. Taken in time order, a busy day (136
        # entries one Monday) gave the judge its first hour, and "what did we do
        # yesterday?" was answered from twenty minutes of lunchtime. "Last night" is the
        # evening (spans_in): "what did I have for food last night?" missed the tea, said at
        # 9:39 pm, among a day of 223 entries; among the evening's 109, it came first. What
        # was said earlier that day about "tonight" goes with it ("fish pie tonight", at 4:30).
        asked = words + [w for w in key_words(question) if w not in words]
        for spans in spans_in(f"{query} {question}", on):
            earlier = []
            with self.lock:
                for start, end in spans:  # the night asked about, or if nothing was said then, its day
                    day = self.db.execute("SELECT rowid, at, text FROM entries WHERE at >= ? AND at < ? ORDER BY at",
                                          (start, end)).fetchall()
                    if day:
                        break
                if day and len(spans) > 1 and (start, end) == spans[0]:
                    earlier = self.db.execute(
                        "SELECT rowid, at, text FROM entries WHERE entries MATCH '\"tonight\"' AND at >= ? AND at < ? "
                        "ORDER BY at DESC LIMIT ?", (spans[1][0], start, EARLIER_PICKED)).fetchall()
                matched = self.db.execute(
                    "SELECT rowid FROM entries WHERE entries MATCH ? AND at >= ? AND at < ? "
                    "ORDER BY bm25(entries) LIMIT ?",
                    (" OR ".join(f'"{w}"' for w in asked), start, end, DAY_PICKED)).fetchall() if asked and day else []
            if not day:
                continue
            by_id = {r[0]: (r[1], r[2]) for r in day + earlier}
            picked = [r[0] for r in earlier] + [r[0] for r in matched]
            if len(picked) < DAY_PICKED:
                # Without the day's words: the day is settled, and "yesterday" in the search
                # put the entries that said "yesterday" first, ahead of the album asked about.
                topic = [t for t in (_DATE.sub(" ", _DAY_WORDS.sub(" ", s)).strip() for s in (query, question))
                         if len(t.split()) > 1 or t and t.lower() not in self.stopwords]
                close = self._by_meaning(topic, among=set(by_id)) if topic else []
                picked += [i for i in close if i not in picked][: DAY_PICKED - len(picked)]
            rest = [r[0] for r in day if r[0] not in picked]
            room = min(DAY_ENTRIES - len(picked), len(rest))
            picked += [rest[i * len(rest) // room] for i in range(room)] if room > 0 else []
            found |= {i: by_id[i] for i in picked}
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

    def recall(self, query: str, question: str, said: str = "", asked: str = "", notes: str = "") -> Lookup:
        """Look `query` up, and answer `question` (what Holly asks, or failing that
        what the crew said) from what's found, and from `said`: the crew's lines
        still in his history (Conversation.crew_lines), which aren't archived yet.
        `asked`: the crew's own words, tried too if his question finds nothing.
        `notes`: his notes on older talk, read with what's found: his notes said "the kit
        for emptying the fire is a dust mask and a special hoover", the archive only that
        the hoover was "to suck up the dirt", and recall answered from the archive alone;
        told "this is your memory of it, and it's right", he said the dirt, and of what,
        "I don't know. You didn't say"."""
        t0 = time.perf_counter()
        query, question = MISHEARD.sub("Kryten", query), MISHEARD.sub("Kryten", question)
        # "The Cat's name in the episode Gunmen of the Apocalypse?" got NOTHING from the
        # judge, taking it as the Cat's own name ("no name other than The Cat"); without
        # "the episode", "The Riviera Kid".
        question = THE_EPISODE.sub(" ", question).replace("  ", " ")
        hits, entries, about, dated = self._found(query, question, said, notes)
        t1 = time.perf_counter()
        if not hits and not notes:
            return self._nothing(query, t1 - t0, 0)
        text = self._judge(entries, dated, query, about)
        missed = text.upper().startswith("NOTHING") and bool(asked)
        # Words of the ship's records' pages he names ("Red Dwarf"): not names he added.
        known = ({w.lower() for t in self.records._named([f"{query} {question}"]) for w in re.findall(r"[A-Za-z]+", t)}
                 if missed and self.records else set())
        if missed and added_names(query, question, asked, known):
            # His notes know the crew by name, and he put it in: "do you remember what I had
            # for food last night?" became "what did Sam have for food last night?". His
            # memory says "Crew", never the name, so the tea was missed, or worse, the name
            # found where Holly had used it, and some other story told. The crew's own words,
            # "I" as the crew, found it. Only after a miss: his question is usually better
            # ("what was it called?" names what "it" is).
            own = in_their_words(asked)
            found = self._found(own, own, said, notes)
            if found[0] and not (again := self._judge(found[1], found[3], own, found[2])).upper().startswith(
                    "NOTHING"):
                hits, text = found[0], again
        names = list(dict.fromkeys(named_in(query, hits, keywords=True) + named_in(question, hits)))[:3]
        if asked and not THIRD_PERSON.search(asked):
            # A name he added, with nobody in the crew's words it could stand for, is theirs,
            # from his notes (see above): asked about it, the judge told some other story.
            # "Did I meet her yesterday?", as "Did the crew meet Jo yesterday?", keeps Jo.
            names = [n for n in names if re.search(rf"\b{n}\b", asked, re.IGNORECASE)]
        if text.upper().startswith("NOTHING") and names:
            # "Did I meet Jo yesterday?" got NOTHING every way it was put, with 14 entries
            # about her; "who is Jo?" got a full answer. Yes-or-no questions about someone
            # trip it up, so it's asked plainly what the entries say about them.
            text = self._judge(entries, f"What do these entries say about {' and '.join(names)}?", query, about)
        t2 = time.perf_counter()
        head, _, rest = text.partition(":")
        if not text or head.strip().upper().startswith("NOTHING"):
            return self._nothing(query, t1 - t0, t2 - t1, len(hits))
        answer = rest.strip() if head.strip().upper() == "ANSWER" else text
        # Answered from his notes alone, it's still found.
        return Lookup(query, answer, max(len(hits), 1), (t1 - t0) * 1000, (t2 - t1) * 1000, [query])

    def _found(self, query: str, question: str, said: str,
               notes: str = "") -> tuple[list[tuple[str | None, str]], str, str, str]:
        """What a search finds, and how the judge is shown it: (the hits, the entries
        as text, which days it's about, and the question with its days as dates)."""
        hits = self.search(query, question=question)
        days = sorted(days_in(f"{query} {question}", datetime.date.today()))
        if self.records and not days:  # a day asked about is about a conversation
            hits += self.records.search(query, question=question)
        if said:
            hits.insert(0, (None, f"(Said in this conversation, by the crew:)\n{said}"))
        entries = "\n\n".join(text[:ENTRY_CHARS] for _, text in sorted(hits, key=lambda h: h[0] or ""))
        if notes:  # first, and whole: they're short, and kept to a budget
            entries = f"(His notes on earlier conversations:)\n{notes}" + (f"\n\n{entries}" if entries else "")
        about = f"\n(The question is about {', '.join(day(d) for d in days)}.)" if days else ""

        # The question's "yesterday" as its date, for the judge: told the day, and how an
        # entry's "yesterday" counts from its own, it still matched "what did we do
        # yesterday?" to Monday's "we had a lovely day yesterday [Sunday]", three times in
        # three. "On Monday 23 November", it didn't.
        def on(m: re.Match) -> str:
            night = "the night of " if m.group(1).lower() in NIGHT_WORDS else ""
            return f"on {night}{day(resolve_day(m.group(1), datetime.date.today(), True))}"

        return hits, entries, about, _DAY_WORDS.sub(on, question) if days else question

    def _judge(self, entries: str, question: str, query: str, about: str) -> str:
        return self.lm.ask(JUDGE_PROMPT.format(today=today()),
                           f"{entries}\n\nQuestion: {question}\n(Searched for: {query}){about}",
                           self.model, self.cfg.summary_max_tokens)

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


# Short words capitalised mid-sentence that aren't names (STOPWORDS has the rest).
NOT_NAMES = {"ok", "oh", "no", "up", "us", "ah", "eh", "hi", "yes", "yeah"}


def named_in(text: str, hits: list[tuple[str | None, str]], keywords: bool = False) -> list[str]:
    """Names (capitalised words, not the first of a sentence or a question word) in
    `text` that the entries found mention: who a question is about. With `keywords`,
    `text` is recall's query, words not a sentence, so its first word counts too: there,
    "Loki" first was the name the question ("do I remember him?") left out."""
    lead = r"" if keywords else r"(?<![.?!]\s)(?<!^)"
    words = re.findall(lead + r"\b([A-Z][a-z]+)\b", text)  # two letters or more: "Jo", "Ed"
    in_dates = {m.group(2).lower() for m in _DATE.finditer(text)}  # "22 June" is a date; "June" a name
    words = [w for w in dict.fromkeys(words) if w.lower() not in STOPWORDS and w.lower() not in WEEKDAYS
             and w.lower() not in in_dates and w.lower() not in NOT_NAMES]
    return [w for w in words if any(re.search(rf"\b{w}\b", t) for _, t in hits)][:3]


def added_names(query: str, question: str, asked: str, known: set[str] = frozenset()) -> list[str]:
    """Names in Holly's search that aren't in `asked`, the crew's own words: capitalised
    words, in his query (words, so its first counts) or his question (not a sentence's
    first), bar question words, days, months and `known` (the records' page words)."""
    words = (re.findall(r"\b([A-Z][a-z]+)\b", query)
             + re.findall(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]+)\b", question))
    return [w for w in dict.fromkeys(words) if w.lower() not in TALK_STOPWORDS | NOT_NAMES | set(MONTHS) | known
            and not re.search(rf"\b{w}\b", asked, re.IGNORECASE)]


# Someone the crew's words point at without naming: a name he puts for it isn't theirs.
THIRD_PERSON = re.compile(r"\b(?:he|him|his|she|her|hers|they|them|their|theirs)\b", re.IGNORECASE)

# The crew's words as a question to Holly's memory, which calls them "Crew": "I", "me"
# and "my" are the crew. "You" is Holly, as the judge takes it anyway. ("Mine" isn't:
# "the mine" is a place.)
_THEIRS = [(re.compile(p), said) for p, said in (
    (r"\bI'm\b", "the crew is"), (r"\bI've\b", "the crew has"), (r"\bI'd\b", "the crew would"),
    (r"\bI'll\b", "the crew will"), (r"\bI\b", "the crew"), (r"(?i)\b(?:me|myself)\b", "the crew"),
    (r"(?i)\bmy\b", "the crew's"))]
# "Hey Holly," before what they asked; not "Holly's" ("Holly's memory's shot, ...").
_HAILED = re.compile(r"^\W*(?:(?:hey|hi|oi|ok|okay|oh|so),?\s+)?holly\b(?!['’]s)\W*", re.IGNORECASE)


def in_their_words(line: str) -> str:
    """"Hey Holly, do you remember what I had for tea?" -> "do you remember what the crew had for tea?"."""
    line = _HAILED.sub("", line)
    for pattern, said in _THEIRS:
        line = pattern.sub(said, line)
    return line.strip()


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


NIGHT_WORDS = ("last night", "tonight")
NIGHT = (datetime.time(17), datetime.time(5))  # an evening, and the small hours after it


def spans_in(text: str, on: datetime.date) -> list[list[tuple[str, str]]]:
    """For each day `text` names (days_in), the times to look in, as ISO strings, narrowest
    first: a night asked about ("last night", "tonight"), then its whole day; a day, just the
    day. A day asked about outright ("yesterday") as well as its night means the whole day."""
    nights = {resolve_day(m.group(1), on, past=True) for m in _DAY_WORDS.finditer(text)
              if m.group(1).lower() in NIGHT_WORDS}
    days = {resolve_day(m.group(1), on, past=True) for m in _DAY_WORDS.finditer(text)
            if m.group(1).lower() not in NIGHT_WORDS and "tomorrow" not in m.group(1).lower()}

    def span(start: datetime.datetime, end: datetime.datetime) -> tuple[str, str]:
        return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")

    out = []
    for d in sorted(days_in(text, on)):
        whole = span(datetime.datetime.combine(d, datetime.time()),
                     datetime.datetime.combine(d + datetime.timedelta(days=1), datetime.time()))
        if d in nights and d not in days:
            out.append([span(datetime.datetime.combine(d, NIGHT[0]),
                             datetime.datetime.combine(d + datetime.timedelta(days=1), NIGHT[1])), whole])
        else:
            out.append([whole])
    return out


# Is the crew's line asking him to remember something? Then his reply must call a tool
# (app/speak.py), recall in practice. Words alone didn't do: "what did we do yesterday?"
# is a memory, "what happened in the news yesterday?" isn't. Shown the exchange before (for
# follow-ups), it took remarks and the news straight after a memory for more of it ("ha,
# that sounds about right" had him read the clock, being made to call something), till
# told to judge the line itself. On 49 made-up lines, 49 right, and 16 of 16 it had never
# seen, in ~85 ms. Live, though, it took what had just been said for memories, and made
# him call something: a correction ("no, I meant you didn't make me listen to the news")
# had him read the clock, and more about his own made-up news, search the BBC for it. Told
# that what's just been said is in front of him, and shown the last two exchanges (app/
# speak.py), a day's 52 real lines had 8 taken for memories, not 15 (two of them new: a
# forecast and a game's lore, each looked up, which did no harm); the 49 and 16 still right.
# The labelled lines are in app/scenarios.py (MEMORY_LINES), checked by `python -m app.evals
# --only memory` and `pytest -m lmstudio tests/test_memory_check.py`. Once there, more about
# his made-up news, after "No, I need another status report please", was still a memory,
# twice in two, till told the news he's just given counts however it was asked for: 85 of 85.
PAST_PROMPT = """\
You sort what someone says to Holly, the household computer (and the ship's computer from Red Dwarf).
PAST: they're asking Holly for something from before, which he'd have to remember: what they did, said, \
decided or told him, people in their life and their news, earlier conversations, reports he wrote, or what \
happened on Red Dwarf (its crew, places, adventures).
OTHER: everything else. Things they're telling him now, remarks and reactions, how Holly feels about \
something, facts about the world, news, weather, the time, sums, requests and chat.
Telling isn't asking. Someone telling him about the past (what a person or a pet was like, what they did, \
what they used to call them, someone who's died, their own childhood) is telling him now: OTHER, however \
long ago it was, and so is carrying on telling him, after he's answered (the names they called them, more \
of what they were like). It's PAST only when they want him to remember it: a question, or "do you \
remember...", "remind me...", "did I tell you...".
Judge only what they say now. "Just before" is there so you can tell what a follow-up question refers to \
("and who was in charge of them?" asks more about that memory: PAST). A remark or reaction ("fair enough", \
"typical", "ha, that sounds right") is OTHER, and so is a question or remark about the news, weather, time \
or the world (what was said in Parliament, at a match), however soon after a memory it comes.
What's just been said in this conversation is in front of him, with nothing to remember: asking about \
something he's just said (what he meant, more of a story or a news item he's just told), correcting him \
("no, I meant..."), or answering him is OTHER. So is anything about the news or the status report he's just \
given (his status report is what Lister, Rimmer, the Cat and Kryten have been up to lately, all his own \
invention), however it was asked for ("another status report"): "more about that story", "what's Rimmer been \
up to?" after it is OTHER. A research report he wrote before is PAST. Asking where one of the crew is now, \
or what they're doing ("where's Kryten?", "is Lister about?"), is OTHER: he makes that up, with nothing to \
remember.
Reply with one word: PAST or OTHER."""


# A memory offered, unasked (app/speak.py): what the crew said on another day, given him with
# what they've just said, to bring up if it fits. Telling him about their cat, he'd called
# recall on line after line to look it up; better he's handed what he knows, and needn't.
# But a search always finds something ("a new drill" found "Left, left, left"), and a memory
# dragged in is worse than none: so the model says whether one is about the same thing. Told
# only that, it matched a place named in passing: the weather in Keswick brought up Jo, who
# lives there, 6 times in 12 conversations, and he worked her in every time. How sure it was
# (log probabilities) didn't tell those apart: 97 to 99%, as sure as of the right ones.
OFFER_PROMPT = """\
Holly, a household computer, is about to answer what someone has just said to him. Below it are some \
things they said to him on other days. Is one of them about the very same particular thing in their \
own life (the same person, pet, plan, object, dish or event) as what they say now, so that remembering \
it now would come naturally, and show he listens? Sharing a word, a place or a general subject isn't \
enough, nor is the same kind of thing (another pet, another car, another trip): "I need a new kettle" \
and "the kettle's broken again" are the same thing; "I'm tired" and "I was tired after the gym on \
Monday" aren't, and nor are "what's the weather in Leeds?" and "my brother lives in Leeds", or "we \
might get a dog" and their cat. NONE, too, when what they say now asks him for something (the weather, the news, the \
time, a search) or is about something just said ("tell me more about that one", "I'll do it Sunday, \
then", where what "it" is was said just before): what's in front of him \
is the answer. Red Dwarf and its crew (Lister, Rimmer, the Cat, Kryten) are never their own life.
Reply with its number, or NONE."""


def crew_said(entry: str) -> tuple[str, str]:
    """An archive entry's day ("Tuesday 24 November") and what the crew said in it."""
    m = re.search(r"\(said on (\w+ \d+ \w+) \d{4}", entry)
    day = m.group(1) if m else ""
    said = " ".join(line[len("Crew: "):].strip() for line in entry.splitlines() if line.startswith("Crew: "))
    said = " ".join(said.split())
    return day, said if len(said) <= OFFER_CHARS else said[:OFFER_CHARS].rsplit(" ", 1)[0] + " ..."


def memory_offered(lm: LmStudio, model: str | None, line: str, memories: list[tuple[int, str, str]]) -> int | None:
    """Which of `memories` (Archive.memories) is about what's just been said, as the model
    judges it: its rowid, or None. Raises if LM Studio can't answer."""
    if not memories:
        return None
    listed = "\n".join(f"{k}. ({day}) {said}" if day else f"{k}. {said}"
                        for k, (_, day, said) in enumerate(memories, 1))
    text = lm.ask(OFFER_PROMPT, f"They say now: {line}\n\nOn other days they said:\n{listed}", model,
                  max_tokens=4, temperature=0)
    m = re.match(r"\W*(\d+)", text)
    k = int(m.group(1)) if m else 0
    return memories[k - 1][0] if 1 <= k <= len(memories) else None


def judged_past(lm: LmStudio, model: str | None, line: str, before: str = "") -> bool:
    """Is `line` asking him for something from the past? `before`: the exchange just before,
    for follow-ups ("and who was in charge of them?"). Raises if LM Studio can't answer."""
    user = (f"Just before: {before}\n" if before else "") + f"They say: {line}"
    return lm.ask(PAST_PROMPT, user, model, max_tokens=4, temperature=0).strip().upper().startswith("PAST")


def about_the_past(lm: LmStudio, model: str | None, line: str, before: str = "") -> bool:
    """judged_past, or False if it can't be asked: he answers as he would have."""
    try:
        return judged_past(lm, model, line, before)
    except Exception:  # LM Studio busy or gone
        return False


class RecallTool(LookupTool):
    """Recall, which also finds his research reports: they aren't in his archive."""

    name, schema, rules, filler = "recall", TOOL, "archive", "recall"

    def __init__(self, archive: Archive, research=None):
        self.archive = archive
        self.research = research  # app/research.py's Research, if he has it

    async def run(self, args: dict, turn: Turn) -> str:
        if not (query := str(args.get("query", "")).strip()):
            return "Recall needs a query."
        # His question, not the crew's last words: after "you need to use the recall
        # tool", the judge was asked that, and found nothing.
        question = str(args.get("question") or "").strip() or turn.user
        research = self.research
        asked = f"{query} {question} {turn.user}"
        if research and "report" in asked.lower() and (
                reports := await asyncio.to_thread(research.find, asked)) and len(reports) == 1 and reports[0].get("sure"):
            # Asked what a report said, he recalled (fragments, or nothing) rather than reading
            # it, and wasn't moved by a hint to read it: so it's read for him. Only when one
            # report is clearly meant, though: "what did the weather report say?" is a
            # question about a conversation, and given a shortlist of research reports instead
            # of recall, he'd have said he had no such report. Anything less sure, recall runs.
            print(f"{DIM}  (recalling a report: reading '{reports[0]['title']}' instead){RESET}", flush=True)
            return research.handle({"action": "read", "title": reports[0]["title"]})
        said = turn.crew_lines()
        notes = turn.notes()
        found = await self.look_up(f"recalling: {query}",
                                   lambda: self.archive.recall(query, question, said, turn.user, notes), turn)
        if isinstance(found, str):
            return found
        if found.results:
            # Told to "state them plainly", he recited the judge's sentences word for word.
            return (f"{found.answer}\n(This is your memory of it, and it's right. Tell it in your own "
                    "words, as something you remember, keeping the facts as they are.)")
        if research and research.finished_titles() and (
                reports := await asyncio.to_thread(
                    research.find, f"{args.get('query', '')} {args.get('question', '')} {turn.user}", 1.0, False)
                ) and len(reports) == 1 and reports[0].get("sure"):
            # Nothing in his archive, and a report clearly covers it ("that marshmallow toy you
            # found"): told he had reports to read, he said "I'll have to check" and didn't. Only
            # one clear winner, though: anything less, and he's told their titles (below). Not
            # widened by Gemma (Reports.find's expand), and not at all with no reports: every
            # miss paid for it, a third of a second or more before "I don't remember".
            print(f"{DIM}  (not in his archive: reading the report '{reports[0]['title']}'){RESET}", flush=True)
            return ("Your archive has nothing on it, but this research report of yours may be what's meant. "
                    "If it doesn't answer what was asked (what was said or done, say), you don't remember: "
                    "say so.\n\n" + research.handle({"action": "read", "title": reports[0]["title"]}))
        if research and (titles := research.finished_titles()):
            # Asked what a report said, he recalled rather than reading it, and found nothing.
            return (f"{found.answer}\n(But if it's about one of your research reports ("
                    + "; ".join(f"'{t}'" for t in titles[-6:])
                    + "), call the research tool to read that report, and answer from it.)")
        return Searcher.for_holly(found)

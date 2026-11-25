"""The ship's records: Holly's memory of Red Dwarf itself, from the Red Dwarf fandom wiki.

    python -m app.records fetch    # download the wiki's articles (once; ~10 min, cached)
    python -m app.records build    # turn them into data/records.db, with vectors

Holly is the ship's computer, so the show's events, people, places and kit are
his own memories. The wiki's articles (reddwarf.fandom.com, CC BY-SA) are
fetched through its API, cut into paragraph-sized entries, and kept in an
archive like his conversations (app/archive.py): his recall tool searches both.
Only what happens in the show is kept: pages about actors, production and fan
works, and sections such as "Trivia" and "Behind the scenes", are left out, so
he doesn't talk about the people who played him. A few entries related to what
you've been talking about also go into his system prompt while he sleeps, so he
can drop a reference in now and then (see reminders()).

Everything stays in data/ (git-ignored): the show isn't ours to publish.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
from html.parser import HTMLParser

import httpx

from .config import ROOT, load_config

API = "https://reddwarf.fandom.com/api.php"
# The wiki asks scripts to say who they are. A name and the project, no personal details.
USER_AGENT = "HollyVoiceAssistant/0.1 (home project; +https://github.com/Ryan-Palmer/Holly)"
RAW = ROOT / "data" / "records_raw"
PAUSE_S = 0.3  # between requests: the wiki is a shared, free service

# Pages about the real world, not the ship's: left out whole.
SKIP_CATEGORIES = {"Real People", "Actors", "Behind the Scenes", "Fan Media", "Videos", "Games",
                   "Candidates for deletion", "Fair use files", "Public domain files", "Browse",
                   "Music", "Merchandise", "Books", "DVDs", "Crew", "Writers", "Directors"}
# Pages from the novels, comics and books only, with no series or episode: Holly is the
# television Holly, and the novels' versions differ (told "go with the television
# account", the judge still answered Better Than Life from the novel's page).
OTHER_MEDIA = {"Novels", "Comics", "Books", "Red Dwarf Smegazine", "Smegazine"}
ON_TV = re.compile(r"^(?:Series [IVX]+|.*Episodes|Back to Earth|The Promised Land|Main Characters|The Tank)$")
# Sections about production, other media or the wiki itself.
SKIP_SECTIONS = re.compile(r"trivia|behind the scenes|production|cast|crew|references|external links|"
                           r"see also|gallery|notes|appearances|novels?|comics?|smegazine|games?|"
                           r"ratings|reception|dvd|remaster|bloopers|deleted scenes|merchandise|"
                           r"background|credits|quotes? from|video|audio|audiobook|book|real.?life|"
                           r"speculation|fan theor|career|filmography|television series|written by|"
                           r"artwork|mentions|continuity|errors|goofs|inconsistenc|guest stars|"
                           r"contemporary|statement|filming|actress|actors?\b|filmed|rpg",
                           re.IGNORECASE)
# Pages that are the real world's, though not filed under a skipped category: days of the
# year (cast birthdays), years, and people (a career or filmography section).
SKIP_TITLES = re.compile(r"^\d{1,2} [A-Z][a-z]+$|^\d{3,4}s?\b")
REAL_PERSON = re.compile(r">\s*(Selected )?Filmography\s*<|>\s*Career\s*<", re.IGNORECASE)
CHUNK_CHARS = 900  # an entry: a paragraph or a few


class _Text(HTMLParser):
    """The readable prose of a wiki page, as (section, paragraph) pairs: infoboxes,
    tables, reference marks, captions and edit links dropped."""

    SKIP_TAGS = {"aside", "table", "sup", "script", "style", "figure", "figcaption", "noscript", "nav"}
    SKIP_CLASSES = ("toc", "navbox", "reference", "mw-editsection", "dablink", "hatnote", "gallery",
                    "wikia-gallery", "noprint", "printfooter", "mw-empty-elt", "quote-source")
    BLOCKS = {"p", "li", "dd", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[tuple[str, str]] = []
        self.section = ""
        self.skip_depth = 0  # >0 while inside something skipped
        self.stack: list[tuple[str, bool]] = []  # (tag, skipping because of it)
        self.buf: list[str] = []
        self.heading: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "img", "hr", "input", "meta", "link", "wbr"):
            return
        cls = dict(attrs).get("class") or ""
        skip = tag in self.SKIP_TAGS or any(c in cls for c in self.SKIP_CLASSES)
        self.stack.append((tag, skip))
        if skip:
            self.skip_depth += 1
        elif tag in ("h2", "h3", "h4") and not self.skip_depth:
            self._flush()
            self.heading = []

    def handle_startendtag(self, tag, attrs):
        """<br />, <img ... />: nothing to open or close."""

    def handle_endtag(self, tag):
        if tag in ("br", "img", "hr", "input", "meta", "link", "wbr"):
            return
        # Pop to the matching tag (HTML from the wiki is well formed, mostly).
        while self.stack:
            t, skip = self.stack.pop()
            if skip:
                self.skip_depth -= 1
            if t == tag:
                break
        if tag in ("h2", "h3", "h4") and self.heading is not None:
            self.section = " ".join("".join(self.heading).split())
            self.heading = None
        elif tag in self.BLOCKS and not self.skip_depth:
            self._flush()

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.heading is not None:
            self.heading.append(data)
        else:
            self.buf.append(data)

    def _flush(self):
        text = " ".join("".join(self.buf).split())
        self.buf = []
        text = re.sub(r"\[\s*(src|edit)?\s*\]", "", text).strip()
        if len(text) >= 40:  # captions, stubs of lists and stray labels are shorter
            self.out.append((self.section, text))


def page_entries(title: str, page_html: str) -> list[str]:
    """A page's prose as entries for the archive, each saying where it's from."""
    parser = _Text()
    parser.feed(page_html)
    parser._flush()
    name = title.removeprefix("RD: ")
    label = f"{name} (episode)" if title.startswith("RD: ") else name
    entries, current, section = [], "", None
    for sec, para in parser.out:
        if sec and SKIP_SECTIONS.search(sec):
            continue
        if para.startswith("This article is about"):
            continue
        if not sec and title.startswith("RD: "):
            # An episode's opening is the real world's ("the third episode in Series
            # VI... first broadcast on BBC2"); its Overview and Summary tell the story.
            continue
        if sec != section or len(current) + len(para) > CHUNK_CHARS:
            if current:
                entries.append(current)
            section = sec
            current = f"(Ship's records: {label}{', ' + sec if sec else ''}) {para}"
        else:
            current += " " + para
    if current:
        entries.append(current)
    return entries


# Fetching -----------------------------------------------------------------

def _get(client: httpx.Client, **params) -> dict:
    for attempt in range(4):
        resp = client.get(API, params={**params, "format": "json"})
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def fetch() -> None:
    """Every article (not redirects), saved as data/records_raw/<n>.json. Pages
    already saved are skipped, so an interrupted fetch carries on."""
    RAW.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        titles, more = [], {}
        while True:
            data = _get(client, action="query", list="allpages", apnamespace=0, apfilterredir="nonredirects",
                        aplimit=500, **more)
            titles += [p["title"] for p in data["query"]["allpages"]]
            if "continue" not in data:
                break
            more = {"apcontinue": data["continue"]["apcontinue"]}
        print(f"{len(titles)} articles", flush=True)
        (RAW / "titles.json").write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
        for i, title in enumerate(titles):
            out = RAW / f"{i}.json"
            if out.exists():
                continue
            try:
                data = _get(client, action="parse", page=title, prop="text|categories", redirects=1)["parse"]
            except (httpx.HTTPError, KeyError) as e:
                print(f"  skipped {title}: {e!r}", flush=True)
                continue
            out.write_text(json.dumps({"title": data["title"], "categories": [c["*"] for c in data["categories"]],
                                       "html": data["text"]["*"]}, ensure_ascii=False), encoding="utf-8")
            if i % 100 == 0:
                print(f"  {i}/{len(titles)}", flush=True)
            time.sleep(PAUSE_S)


def build() -> None:
    """data/records.db from the fetched pages, with vectors for recall by meaning."""
    from .archive import Archive
    from .lmstudio import LmStudio

    cfg = load_config()
    files = sorted(RAW.glob("[0-9]*.json"), key=lambda p: int(p.stem))
    entries, pages = [], 0
    for f in files:
        page = json.loads(f.read_text(encoding="utf-8"))
        cats = {c.replace("_", " ") for c in page["categories"]}
        if cats & SKIP_CATEGORIES or SKIP_TITLES.match(page["title"]) or REAL_PERSON.search(page["html"]):
            continue
        if cats & OTHER_MEDIA and not any(ON_TV.match(c) for c in cats):
            continue
        got = page_entries(page["title"], page["html"])
        pages += bool(got)
        entries += [(None, html.unescape(e)) for e in got]
    lm = LmStudio(cfg.llm)
    if cfg.recall.embed_model:
        lm.ensure_loaded(cfg.recall.embed_model, context_length=2048)
    path = cfg.recall.records_file
    if path.exists():
        path.unlink()
    records = Archive(cfg.recall, lm, cfg.llm.model, path, talk=False)
    records.add(entries)
    print(f"{len(entries)} entries from {pages} pages; embedding...", flush=True)
    t = time.perf_counter()
    records.embed_missing()
    print(f"done in {time.perf_counter() - t:.0f} s: {path}", flush=True)


# Reminders ----------------------------------------------------------------

REMINDER_CHARS = 320


def source_of(entry: str) -> str:
    """Where an entry is from: "D.N.A. (episode), Summary" from
    "(Ship's records: D.N.A. (episode), Summary) Lister ..."."""
    return entry.removeprefix("(Ship's records: ").split(") ", 1)[0]


def reminders(records, topics: list[str], n: int, rng: random.Random | None = None) -> list[str]:
    """`n` entries for his system prompt: the closest to each of `topics` (what's
    been said lately, newest first) in turn, one per page, and one at random, so
    he has something to bring up even when the talk is about nothing on the
    ship. Each cut to a sentence or two. Each topic is searched on its own: run
    together, one strong word ("vindaloo") took every place and the guitar got none."""
    rng = rng or random.Random()
    picked: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        source = source_of(text).split(",")[0]  # the page: one entry from each
        if source in seen:
            return
        seen.add(source)
        cut = text[:REMINDER_CHARS]
        if len(text) > REMINDER_CHARS:
            cut = cut[: max(cut.rfind(". "), cut.rfind("! "), 120) + 1]
        picked.append(cut.strip())

    found = [[text for _, text in records.search(t, question=t)] for t in topics if t.strip()]
    for rank in range(n * 2):  # each topic's best, then each one's second best, ...
        for hits in found:
            if rank < len(hits) and len(picked) < n - 1:
                add(hits[rank])
    if (total := records.count()) and len(picked) < n:
        for _ in range(5):
            row = records.entry(rng.randrange(total))
            if row:
                add(row)
                if len(picked) >= n:
                    break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Holly's ship's records, from the Red Dwarf fandom wiki")
    parser.add_argument("step", choices=["fetch", "build"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"fetch": fetch, "build": build}[args.step]()


if __name__ == "__main__":
    main()

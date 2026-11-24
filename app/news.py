"""Holly's news: the BBC's headlines, from its RSS feeds (free, no key, feeds.bbci.co.uk),
and Google News's RSS search for what those don't have.

Asked what's in the news, web search gave him a search engine's guess at what's recent,
from snippets. The BBC's feeds are its own headlines, newest first, each with a line about
it and when it went up (~0.25 s a feed). They also keep stories for months, so only the
last two days' are given him, or the last week's if that's too few. Feeds are kept for a
few minutes, so a run of questions doesn't fetch them each time.

Asked about something in particular, the BBC's feeds (a few dozen stories each) often
had nothing on it: a week-old dig at Sutton Hoo wasn't there. Then Google News's search
(its UK edition) finds it across publishers: headlines and dates only, no summary.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import email.utils
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx

from .config import SearchConfig
from .search import Lookup, Searcher
from .tools import Lookup as LookupTool, Turn

BBC = "https://feeds.bbci.co.uk"
FEEDS = {
    "top": "/news/rss.xml", "uk": "/news/uk/rss.xml", "world": "/news/world/rss.xml",
    "business": "/news/business/rss.xml", "politics": "/news/politics/rss.xml",
    "technology": "/news/technology/rss.xml", "science": "/news/science_and_environment/rss.xml",
    "health": "/news/health/rss.xml", "education": "/news/education/rss.xml",
    "entertainment": "/news/entertainment_and_arts/rss.xml", "sport": "/sport/rss.xml",
}
BBC_PAGES = ("https://www.bbc.co.uk/", "https://www.bbc.com/")
GOOGLE = "https://news.google.com/rss/search"
HEADLINES = 6  # given him at most: he'll say two or three
FRESH = datetime.timedelta(days=2)
STALE = datetime.timedelta(days=7)  # if two days give too few
KEEP_S = 300  # a feed is fetched again after this

TOOL = {
    "type": "function",
    "function": {
        "name": "news",
        "description": "The BBC's latest news headlines, newest first: by topic, or about something in the "
                       "news (words to look for in all the BBC's feeds).",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": list(FEEDS), "description": "'top' for the main stories."},
                "about": {"type": "string", "description":
                          "Optional: a few words to look for, for news about something in particular "
                          "('Sutton Hoo', 'interest rates')."},
            },
        },
    },
}

_kept: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


def headlines(topic: str, timeout: float) -> list[dict]:
    """A feed's stories, newest first: title, summary, published (aware datetime), link."""
    with _lock:
        if (got := _kept.get(topic)) and time.monotonic() - got[0] < KEEP_S:
            return got[1]
    resp = httpx.get(BBC + FEEDS[topic], timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    stories = []
    for item in ET.fromstring(resp.content).findall("./channel/item"):
        try:
            published = email.utils.parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        stories.append({"title": (item.findtext("title") or "").strip(),
                        "summary": (item.findtext("description") or "").strip(),
                        "published": published, "link": item.findtext("link") or ""})
    stories.sort(key=lambda s: s["published"], reverse=True)
    with _lock:
        _kept[topic] = (time.monotonic(), stories)
    return stories


def google_news(words: str, timeout: float) -> list[dict]:
    """Google News's stories matching `words` from the past week, UK edition: title,
    summary (who published it), published, link; newest first."""
    key = "google:" + words.lower()
    with _lock:
        if (got := _kept.get(key)) and time.monotonic() - got[0] < KEEP_S:
            return got[1]
    resp = httpx.get(GOOGLE, params={"q": f"{words} when:7d", "hl": "en-GB", "gl": "GB", "ceid": "GB:en"},
                     timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    stories = []
    for item in ET.fromstring(resp.content).findall("./channel/item"):
        try:
            published = email.utils.parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        source = (item.findtext("source") or "").strip()
        title = (item.findtext("title") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        stories.append({"title": title, "summary": f"from {source}" if source else "", "published": published,
                        "link": item.findtext("link") or ""})
    stories.sort(key=lambda s: s["published"], reverse=True)
    with _lock:
        _kept[key] = (time.monotonic(), stories)
    return stories


def ago(when: datetime.datetime, now: datetime.datetime) -> str:
    mins = int((now - when).total_seconds() // 60)
    if mins < 60:
        return f"{max(mins, 1)} minute{'s' * (max(mins, 1) != 1)} ago"
    if mins < 60 * 24:
        return f"{mins // 60} hour{'s' * (mins // 60 != 1)} ago"
    days = mins // (60 * 24)
    return f"{days} day{'s' * (days != 1)} ago"


def news(topic: str, about: str, cfg: SearchConfig, now: datetime.datetime | None = None) -> Lookup:
    """The latest headlines on `topic`, or about `about` in any feed, as a few lines for Holly."""
    t0 = time.perf_counter()
    now = now or datetime.datetime.now(datetime.timezone.utc)
    words = [w for w in re.findall(r"[a-z0-9']+", about.lower()) if len(w) > 2]
    if words:
        pool, seen = [], set()
        with concurrent.futures.ThreadPoolExecutor(len(FEEDS)) as pool_of:  # all at once: ~0.3 s, not ~3
            feeds = list(pool_of.map(lambda f: headlines(f, cfg.timeout_s), FEEDS))
        for stories in feeds:
            for s in stories:
                text = f"{s['title']} {s['summary']}".lower()
                if s["link"] not in seen and all(re.search(rf"\b{re.escape(w)}", text) for w in words):
                    seen.add(s["link"])
                    pool.append(s)
        if len([s for s in pool if now - s["published"] <= STALE]) < 3:
            # Too little in the BBC's feeds: Google News's search, each story once (the BBC's
            # own turn up there too, under the same headline).
            titles = {s["title"].lower() for s in pool}
            pool += [s for s in google_news(about.strip(), cfg.timeout_s)[:10] if s["title"].lower() not in titles]
        pool.sort(key=lambda s: s["published"], reverse=True)
        what = f"about {about.strip()!r}"
    else:
        topic = topic if topic in FEEDS else "top"
        pool = headlines(topic, cfg.timeout_s)
        what = "top stories" if topic == "top" else f"{topic} news"
    recent = [s for s in pool if now - s["published"] <= FRESH]
    if len(recent) < 3:
        recent = [s for s in pool if now - s["published"] <= STALE]
    ms = (time.perf_counter() - t0) * 1000
    if not recent:
        why = (f"There's nothing in the news {what} this past week." if words
               else f"The BBC's {what} had nothing from this past week.")
        return Lookup(about or topic, why + " Say so; if they want more, you could search the web.", 0, ms, 0)
    lines = [f"- {s['title']}: {s['summary']} ({ago(s['published'], now)})" for s in recent[:HEADLINES]]
    bbc = all(s["link"].startswith(BBC_PAGES) for s in recent[:HEADLINES])
    answer = (f"{'The BBC' + chr(39) + 's' if bbc else 'The'} latest {what}, newest first:\n" + "\n".join(lines)
              + "\nGive them the two or three that matter most, briefly, in your own words.")
    return Lookup(about or topic, answer, len(recent[:HEADLINES]), ms, 0)


class NewsTool(LookupTool):
    name, schema = "news", TOOL  # his rules for it are in search.md

    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg

    async def run(self, args: dict, turn: Turn) -> str:
        topic = str(args.get("topic") or "top").strip().lower()
        about = str(args.get("about") or "").strip()
        found = await self.look_up(f"news: {about or topic}", lambda: news(topic, about, self.cfg), turn)
        return found if isinstance(found, str) else Searcher.for_holly(found)

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
import random
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
# A status report: the top stories, then the freshest of these, a couple each.
ROUNDUP = {"top": "Top stories", "politics": "Politics", "world": "World", "business": "Business and money",
           "technology": "Technology", "science": "Science", "sport": "Sport"}
ROUNDUP_EACH = 2
# The report's "Additional" item, news from the Red Dwarf universe, is about one of these,
# drawn at random: left to choose, four reports in four had a mechanoid "detained" in some
# "sector", or the Jupiter Mining Corporation "announcing" something.
LORE = ("the ship's crew", "the Jupiter Mining Corporation", "the Space Corps", "a rival mining ship",
        "a hologram", "a mechanoid", "the cat race", "a skutter", "the ship's computer systems",
        "a vending machine", "a GELF tribe", "an android colony", "a holo-ship", "a deep-space trading post",
        "the ship's food dispensers", "a space station", "a simulant", "the ship's medical unit")
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
                "topic": {"type": "string", "enum": list(FEEDS) + ["roundup"], "description":
                          "'top' for the main stories; 'roundup' for a status report: the latest in each "
                          "of the main subjects."},
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


def roundup(cfg: SearchConfig, now: datetime.datetime | None = None) -> Lookup:
    """A status report's news: the freshest couple of stories in each of the main subjects,
    each story once (the top stories turn up again under their subjects)."""
    t0 = time.perf_counter()
    now = now or datetime.datetime.now(datetime.timezone.utc)
    with concurrent.futures.ThreadPoolExecutor(len(ROUNDUP)) as pool_of:  # all at once: ~0.3 s
        feeds = dict(zip(ROUNDUP, pool_of.map(lambda f: headlines(f, cfg.timeout_s), ROUNDUP)))
    seen, parts, count = set(), [], 0
    for feed, name in ROUNDUP.items():
        fresh = []
        for s in feeds[feed]:  # each story once: a feed can list one twice, and subjects share them
            if len(fresh) < ROUNDUP_EACH and now - s["published"] <= FRESH and s["title"].lower() not in seen:
                seen.add(s["title"].lower())
                fresh.append(s)
        if not fresh:
            continue
        count += len(fresh)
        parts.append(f"{name}:\n" + "\n".join(f"- {s['title']}: {s['summary']} ({ago(s['published'], now)})"
                                               for s in fresh[:ROUNDUP_EACH]))
    ms = (time.perf_counter() - t0) * 1000
    if not parts:
        return Lookup("roundup", "The BBC's feeds had nothing from the past two days. Say so.", 0, ms, 0)
    answer = ("The BBC's latest, by subject, for a status report:\n" + "\n".join(parts)
              # The form here, the last thing he reads: in his rules alone, he copied an older
              # report from his history instead ("Politics' got... Tech's got...").
              + "\nRound it up subject by subject, a sentence each, in your own words, each starting with "
                "its name and a dash ('Politics — ministers want...'); then the 'Additional' item, as your "
                f"rules say: today's is news about {random.choice(LORE)}.")
    return Lookup("roundup", answer, count, ms, 0)


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
        if topic == "roundup" and not about:
            found = await self.look_up("news: a status report", lambda: roundup(self.cfg), turn)
            return found if isinstance(found, str) else Searcher.for_holly(found)
        found = await self.look_up(f"news: {about or topic}", lambda: news(topic, about, self.cfg), turn)
        return found if isinstance(found, str) else Searcher.for_holly(found)

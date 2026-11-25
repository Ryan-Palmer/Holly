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
from .console import note
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
REPORT = "The BBC's latest, by subject, for a status report:"  # how a round-up starts
# An earlier report, as he's sent it on a turn with a new one.
EARLIER = "(An earlier status report: old news now.)"
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


def _all(fetch, names: list[str], timeout: float) -> list[list[dict]]:
    """Each feed's stories, fetched all at once (~0.3 s, not ~3). A feed that fails gives
    none: one that timed out or answered 503 failed the whole status report, the other
    six with it. If every one fails, the first one's error is raised."""
    def one(name: str) -> tuple[list[dict], Exception | None]:
        try:
            return fetch(name, timeout), None
        except Exception as e:
            note(f"news: the {name} feed failed: {e!r}")
            return [], e

    with concurrent.futures.ThreadPoolExecutor(len(names)) as pool_of:
        got = list(pool_of.map(one, names))
    if all(error for _, error in got):
        raise got[0][1]
    return [stories for stories, _ in got]


def roundup(cfg: SearchConfig, now: datetime.datetime | None = None, fetch=None) -> Lookup:
    """A status report's news: the freshest couple of stories in each of the main subjects,
    each story once (the top stories turn up again under their subjects). `fetch`: how a
    feed's stories are got (headlines, unless the evals give made-up ones)."""
    fetch = fetch or headlines
    t0 = time.perf_counter()
    now = now or datetime.datetime.now(datetime.timezone.utc)
    feeds = dict(zip(ROUNDUP, _all(fetch, list(ROUNDUP), cfg.timeout_s)))
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
    answer = (REPORT + "\n" + "\n".join(parts)
              # The form here, the last thing he reads: in his rules alone, he copied an older
              # report from his history instead ("Politics' got... Tech's got...").
              + "\nRound it up subject by subject, a sentence each, in your own words, each starting with "
                "its name and a dash ('Politics — ministers want...'); then the 'Additional' item, as your "
                f"rules say: today's is news about {random.choice(LORE)}.")
    return Lookup("roundup", answer, count, ms, 0)


# Is the crew's line asking for a status report? Asked only when it says "status report" or
# "status update" (guards.STATUS_ASKED): words alone took "my boss wants a status update
# by five" for one, and he'd have been made to give the news.
REPORT_PROMPT = """\
You sort what someone says to Holly, the household computer. He gives a "status report" when asked: a round-up \
of today's news, subject by subject.
YES: they're asking him for one now ("status report, please", "give me a status update, Holly", "another status \
report", "what's the status report today?").
NO: anything else: a status report or update of their own or someone else's ("I've got a status report to \
write", "my boss wants a status update by five"), or a remark about one he's given.
Reply with one word: YES or NO."""


def wants_report(lm, model: str | None, line: str, quiet: bool = True) -> bool:
    """Is `line` asking him for a status report? If it can't be asked, False (he answers as
    he would have), or with `quiet` off (the evals), the error."""
    try:
        return lm.ask(REPORT_PROMPT, f"They say: {line}", model, max_tokens=4,
                      temperature=0).strip().upper().startswith("YES")
    except Exception:
        if not quiet:
            raise
        return False


def without_earlier_reports(messages: list[dict], before: int) -> list[dict]:
    """`messages`, with the status reports in the first `before` (his history) cut to a
    line: their news, and what he said of it. Sent on a turn with a new report. Asked
    again six minutes after the last, the news unchanged, he gave the last one again word
    for word ("Politics' got..."), not as the round-up asked ("Politics — ..."). Kept in
    his history as they were: only this request leaves them out."""
    out, report = [], False
    for i, m in enumerate(messages):
        if i < before and m["role"] == "tool" and (m.get("content") or "").startswith(REPORT):
            m, report = {**m, "content": EARLIER}, True
        elif i < before and report and m["role"] == "assistant" and not m.get("tool_calls"):
            m, report = {**m, "content": EARLIER}, False
        elif m["role"] == "user":
            report = False
        out.append(m)
    return out


def ago(when: datetime.datetime, now: datetime.datetime) -> str:
    mins = int((now - when).total_seconds() // 60)
    if mins < 60:
        return f"{max(mins, 1)} minute{'s' * (max(mins, 1) != 1)} ago"
    if mins < 60 * 24:
        return f"{mins // 60} hour{'s' * (mins // 60 != 1)} ago"
    days = mins // (60 * 24)
    return f"{days} day{'s' * (days != 1)} ago"


def news(topic: str, about: str, cfg: SearchConfig, now: datetime.datetime | None = None, fetch=None) -> Lookup:
    """The latest headlines on `topic`, or about `about` in any feed, as a few lines for Holly."""
    fetch = fetch or headlines
    t0 = time.perf_counter()
    now = now or datetime.datetime.now(datetime.timezone.utc)
    words = [w for w in re.findall(r"[a-z0-9']+", about.lower()) if len(w) > 2]
    if words:
        pool, seen = [], set()
        for stories in _all(fetch, list(FEEDS), cfg.timeout_s):
            for s in stories:
                text = f"{s['title']} {s['summary']}".lower()
                if s["link"] not in seen and all(re.search(rf"\b{re.escape(w)}", text) for w in words):
                    seen.add(s["link"])
                    pool.append(s)
        if len([s for s in pool if now - s["published"] <= STALE]) < 3:
            # Too little in the BBC's feeds: Google News's search, each story once (the BBC's
            # own turn up there too, under the same headline).
            titles = {s["title"].lower() for s in pool}
            try:
                more = google_news(about.strip(), cfg.timeout_s)[:10]
            except Exception as e:  # what the BBC had still stands
                note(f"news: Google News failed: {e!r}")
                more = []
            pool += [s for s in more if s["title"].lower() not in titles]
        pool.sort(key=lambda s: s["published"], reverse=True)
        what = f"about {about.strip()!r}"
    else:
        topic = topic if topic in FEEDS else "top"
        pool = fetch(topic, cfg.timeout_s)
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
        self.fetch = headlines  # a feed's stories: made-up ones in the evals (app/scenarios.py)

    async def run(self, args: dict, turn: Turn) -> str:
        topic = str(args.get("topic") or "top").strip().lower()
        about = str(args.get("about") or "").strip()
        if topic == "roundup" and not about:
            found = await self.look_up("news: a status report", lambda: roundup(self.cfg, fetch=self.fetch), turn)
            return found if isinstance(found, str) else Searcher.for_holly(found)
        found = await self.look_up(f"news: {about or topic}", lambda: news(topic, about, self.cfg, fetch=self.fetch), turn)
        return found if isinstance(found, str) else Searcher.for_holly(found)

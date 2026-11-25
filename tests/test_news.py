"""The news tool (app/news.py), on made-up feeds: no network."""

import datetime as dt

import pytest

from app import news

NOW = dt.datetime(2026, 11, 24, 22, 0, tzinfo=dt.timezone.utc)


def rss(*items: tuple[str, str, dt.datetime, str], source: str = "") -> bytes:
    body = "".join(
        f"<item><title>{t}</title><description>{d}</description><link>{link}</link>"
        f"<pubDate>{when.strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate>"
        + (f"<source>{source}</source>" if source else "") + "</item>"
        for t, d, when, link in items)
    return f"<rss><channel>{body}</channel></rss>".encode()


class Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


@pytest.fixture
def feeds(monkeypatch, cfg):
    """BBC feeds and Google News answers, by URL, set per test; nothing kept between tests."""
    served: dict[str, bytes] = {}
    asked: list[str] = []

    def get(url, params=None, **kw):
        asked.append(url)
        return Response(served.get(url, rss()))

    monkeypatch.setattr(news.httpx, "get", get)
    monkeypatch.setattr(news, "_kept", {})
    return served, asked


def bbc(path: str) -> str:
    return news.BBC + news.FEEDS[path]


def test_top_stories_newest_first_and_only_recent(feeds, cfg):
    served, _ = feeds
    served[bbc("top")] = rss(("Old story", "From the spring.", NOW - dt.timedelta(days=200), "https://www.bbc.co.uk/1"),
                             ("Second", "Earlier today.", NOW - dt.timedelta(hours=5), "https://www.bbc.co.uk/2"),
                             ("First", "Just now.", NOW - dt.timedelta(minutes=10), "https://www.bbc.co.uk/3"),
                             ("Third", "Yesterday.", NOW - dt.timedelta(hours=30), "https://www.bbc.co.uk/4"))
    found = news.news("top", "", cfg.search, NOW)
    assert found.answer.startswith("The BBC's latest top stories, newest first:")
    assert found.answer.index("First") < found.answer.index("Second") < found.answer.index("Third")
    assert "Old story" not in found.answer and "(10 minutes ago)" in found.answer and found.results == 3


def test_about_something_looks_in_every_feed(feeds, cfg):
    served, asked = feeds
    served[bbc("science")] = rss(("Grave found at Sutton Hoo", "A sand body.", NOW - dt.timedelta(hours=2),
                                  "https://www.bbc.co.uk/9"),
                                 ("Badgers", "Cull ruled out.", NOW - dt.timedelta(hours=1), "https://www.bbc.co.uk/8"))
    served[bbc("uk")] = rss(("Sutton Hoo ship replica launched", "In Woodbridge.", NOW - dt.timedelta(hours=3),
                             "https://www.bbc.co.uk/7"),
                            ("More on Sutton Hoo", "Again.", NOW - dt.timedelta(hours=4), "https://www.bbc.co.uk/6"))
    found = news.news("top", "Sutton Hoo", cfg.search, NOW)
    assert "Grave found" in found.answer and "replica" in found.answer and "Badgers" not in found.answer
    assert news.GOOGLE not in asked  # three from the BBC: enough


def test_too_little_from_the_bbc_asks_google_news(feeds, cfg):
    served, _ = feeds
    served[news.GOOGLE] = rss(("Ghostly sand body found at Sutton Hoo - Live Science", "", NOW - dt.timedelta(hours=7),
                               "https://news.google.com/a"), source="Live Science")
    found = news.news("top", "Sutton Hoo", cfg.search, NOW)
    assert found.answer.startswith("The latest about 'Sutton Hoo'")
    assert "Ghostly sand body found at Sutton Hoo: from Live Science (7 hours ago)" in found.answer


def test_nothing_anywhere(feeds, cfg):
    found = news.news("top", "zzqx", cfg.search, NOW)
    assert found.results == 0 and found.answer.startswith("There's nothing in the news about 'zzqx'")


def test_a_status_report_takes_each_subject_once(feeds, cfg):
    served, _ = feeds
    story = ("Energy payment explained", "Millions will get it.", NOW - dt.timedelta(hours=3), "https://www.bbc.co.uk/e")
    served[bbc("top")] = rss(("Big story", "Today.", NOW - dt.timedelta(hours=1), "https://www.bbc.co.uk/b"))
    served[bbc("business")] = rss(story, story, ("Big story", "Today.", NOW - dt.timedelta(hours=1),
                                                 "https://www.bbc.co.uk/b"))
    served[bbc("science")] = rss(("Badgers", "Cull ruled out.", NOW - dt.timedelta(hours=6), "https://www.bbc.co.uk/s"))
    found = news.roundup(cfg.search, NOW)
    assert found.answer.count("Energy payment explained") == 1  # listed twice in its feed
    assert found.answer.count("Big story") == 1  # under top stories, not again under business
    assert "Business and money:" in found.answer and "Science:" in found.answer and "Sport:" not in found.answer
    assert "Additional" not in found.answer  # the news, just: the made-up item is the status report's


class Broken(Exception):
    pass


def test_a_feed_that_fails_leaves_the_rest(feeds, cfg, monkeypatch):
    """One feed timing out (or a 503) failed the whole status report, the six that answered
    with it; a Google News failure, the BBC's own stories."""
    served, _ = feeds
    served[bbc("top")] = rss(("Big story", "Today.", NOW - dt.timedelta(hours=1), "https://www.bbc.co.uk/b"))
    served[bbc("science")] = rss(("Sutton Hoo grave", "Sand.", NOW - dt.timedelta(hours=2), "https://www.bbc.co.uk/s"))
    get = news.httpx.get

    def flaky(url, params=None, **kw):
        if url in (bbc("sport"), news.GOOGLE):
            raise Broken("timed out")
        return get(url, params, **kw)

    monkeypatch.setattr(news.httpx, "get", flaky)
    found = news.roundup(cfg.search, NOW)
    assert "Big story" in found.answer and "Sport:" not in found.answer
    found = news.news("top", "Sutton Hoo", cfg.search, NOW)  # too little from the BBC: Google, which fails
    assert "Sutton Hoo grave" in found.answer


def test_every_feed_failing_is_an_error(feeds, cfg, monkeypatch):
    def down(url, params=None, **kw):
        raise Broken("no network")

    monkeypatch.setattr(news.httpx, "get", down)
    with pytest.raises(Broken):
        news.roundup(cfg.search, NOW)

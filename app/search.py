"""Holly looks things up: a web search, boiled down to a few sentences.

Holly's chat model asks for a search with a tool call. The results go to the
same model in a separate request, which answers the question from them in
two or three plain sentences, and only that answer goes back into Holly's
conversation. Raw results would fill his 8K context and tempt him to read
out URLs. Using the same model avoids loading a second one, and LM Studio
restores the conversation's cache after the side request (first token
~150 ms against ~80 fully cached, measured).
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
from urllib.parse import urlparse
from dataclasses import dataclass, field

import httpx

from .config import SearchConfig
from .lmstudio import LmStudio
from .persona import today

TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for facts about the real world: news, sport, weather, "
                       "people, places, science, numbers, dates and prices.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description":
                "What to search for. For anything recent, use today's date from your "
                "instructions, never a year from memory."}},
            "required": ["query"],
        },
    },
}

# Offered NO ANSWER as an equal choice, Gemma always took it rather than search
# again, so it's only on offer once the searches have run out.
SUMMARY_PROMPT = """\
Today is {today}. You answer a question from web search results. Use only
what the results say, never your own memory. Reply in exactly one of these
forms:

ANSWER: two or three short, plain sentences of fact that answer the
question, for someone to read aloud: no lists, no links, no markdown.
Numbers, names and dates exactly as the results give them.
{otherwise}"""

SEARCH_AGAIN = """\
READ: the number of one result whose page probably holds the answer, when
its title or snippet suggests so but doesn't give it (a price, a time, a
date or a figure is often only on the page).
SEARCH: if no result looks likely to hold the answer, a different web
search that is more likely to: more specific, worded differently, or
naming a site that would know."""

PAGE_CHARS = 6000  # of a page's text for the judge: ~1,500 tokens

LAST_ROUND = """\
NO ANSWER: if the results don't answer the question."""

NO_ANSWER = "NO ANSWER"


@dataclass
class Result:
    title: str
    url: str
    snippet: str


@dataclass
class Lookup:
    query: str
    answer: str
    results: int
    search_ms: float
    summary_ms: float
    queries: list[str] = field(default_factory=list)  # every search run, the first included


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def search_ddgs(query: str, cfg: SearchConfig) -> list[Result]:
    from ddgs import DDGS  # imported here: it's slow to import and only needed for this provider

    from ddgs.exceptions import DDGSException

    def text(backend: str) -> list[dict]:
        return DDGS(timeout=cfg.timeout_s).text(query, region=cfg.region, max_results=cfg.max_results,
                                                backend=backend)

    try:
        hits = text(cfg.backend)
    except DDGSException:
        # The engines refuse now and then (CAPTCHAs, rate limits): one more
        # go with every engine ddgs has, which is slower (~2 s).
        if cfg.backend == "auto":
            raise
        hits = text("auto")
    return [Result(_clean(h.get("title", "")), h.get("href", ""), _clean(h.get("body", ""))) for h in hits]


def search_brave(query: str, cfg: SearchConfig) -> list[Result]:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        raise RuntimeError("set BRAVE_API_KEY to use the Brave search provider")
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": cfg.max_results, "country": cfg.region.split("-")[0].upper()},
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    hits = resp.json().get("web", {}).get("results", [])
    return [
        Result(_clean(h.get("title", "")), h.get("url", ""),
               _clean(" ".join([h.get("description", ""), *h.get("extra_snippets", [])[:2]])))
        for h in hits[: cfg.max_results]
    ]


PROVIDERS = {"ddgs": search_ddgs, "brave": search_brave}


def fetch_page(url: str, cfg: SearchConfig) -> str:
    """A result's page as plain text, menus dropped, cut to PAGE_CHARS.

    Only public web pages: a page could name an address on this machine or
    network, and Holly shouldn't fetch those."""
    from ddgs import DDGS

    host = urlparse(url).hostname or ""
    try:
        private = ipaddress.ip_address(socket.gethostbyname(host)).is_private
    except (OSError, ValueError):
        private = True
    if urlparse(url).scheme not in ("http", "https") or private or host == "localhost":
        return "(not a public web page)"
    try:
        text = DDGS(timeout=cfg.timeout_s).extract(url, fmt="text_plain").get("content", "")
    except Exception as e:  # sites that refuse bots, timeouts
        return f"(this page couldn't be read ({type(e).__name__}): pick another result or search)"
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    # Menus and buttons are short lines with no numbers: "News Home", "Sign in".
    lines = [ln.strip() for ln in text.splitlines()]
    kept = [ln for ln in lines if len(ln.split()) >= 5 or re.search(r"\d", ln)]
    return "\n".join(kept)[:PAGE_CHARS] or "(this page had no text: pick another result or search)"


class Searcher:
    def __init__(self, cfg: SearchConfig, lm: LmStudio, model: str):
        if cfg.provider not in PROVIDERS:
            raise ValueError(f"unknown search provider {cfg.provider!r}; use one of {', '.join(PROVIDERS)}")
        self.cfg = cfg
        self.lm = lm
        self.model = model

    def _search(self, query: str) -> tuple[list[Result], str]:
        """Results, or none and what went wrong."""
        try:
            results = PROVIDERS[self.cfg.provider](query, self.cfg)
            return results, "" if results else "it found nothing"
        except Exception as e:  # a failed search is something Holly can say, not a crash
            # ddgs says "No results found." when the engines answer with CAPTCHAs.
            return [], f"it failed: {type(e).__name__}: {str(e)[:120]}"

    def look_up(self, query: str, question: str) -> Lookup:
        """Search for `query` and answer `question` (what was asked) from the results.

        The same Gemma judges what it has: it answers, or reads the page of a
        promising result, or names a better search, up to `max_rounds`
        searches and reads in all, or gives up. Holly sees only the outcome,
        so his context stays small however much it took. Steps show in
        `queries` ("read: <url>" for a page)."""
        search_s = summary_s = 0.0
        started = time.perf_counter()
        queries: list[str] = []
        found: list[str] = []  # each search's results and each page read, for the judge
        everything: list[Result] = []  # every result so far, numbered from 1 across searches
        next_step = ("SEARCH", query)
        while True:
            kind, text = next_step
            t0 = time.perf_counter()
            if kind == "SEARCH":
                queries.append(text)
                results, problem = self._search(text)
                first = len(everything) + 1
                everything += results
                listing = "\n\n".join(f"{i}. {r.title}\n{r.snippet}" for i, r in enumerate(results, first))
                found.append(f"Search: {text}\n\n{listing or '(no results: ' + problem + ')'}")
                if not everything:  # nothing at all to judge: the search itself failed
                    return self._not_found(queries, f"The search didn't work ({problem}).",
                                           time.perf_counter() - t0, 0)
            else:  # READ
                r = everything[int(text) - 1]
                queries.append(f"read: {r.url}")
                found.append(f"Page of result {text} ({r.title}):\n\n{fetch_page(r.url, self.cfg)}")
            search_s += time.perf_counter() - t0
            # Past the time budget, no more searches or reads: the stock line only
            # covers ~1.5 s, and one sunset question once took 10 s.
            late = time.perf_counter() - started > self.cfg.max_lookup_s
            last = len(queries) >= self.cfg.max_rounds or late
            t1 = time.perf_counter()
            next_step = self._judge(question, found, last, len(everything))
            summary_s += time.perf_counter() - t1
            kind, text = next_step
            if kind == "ANSWER":
                return Lookup(queries[0], text, len(everything), search_s * 1000, summary_s * 1000, queries)
            repeat = kind == "SEARCH" and text.lower() in (q.lower() for q in queries)
            read_twice = kind == "READ" and f"read: {everything[int(text) - 1].url}" in queries
            if kind == "NO ANSWER" or last or repeat or read_twice:
                # Given "the results don't say what the weather will be", he
                # forecast it anyway, so this counts as a failed search.
                return self._not_found(queries, "The search results didn't have the answer.",
                                       search_s, summary_s)

    def _judge(self, question: str, found: list[str], last: bool, n_results: int) -> tuple[str, str]:
        """("ANSWER", text), ("READ", result number), ("SEARCH", query) or ("NO ANSWER", "")."""
        thinking = self.cfg.summary_reasoning != "none"
        reply = self.lm.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT.format(
                    today=today(), otherwise=LAST_ROUND if last else SEARCH_AGAIN)},
                {"role": "user", "content": f"Question: {question}\n\n" + "\n\n".join(found)},
            ],
            temperature=0.2,
            # Thinking spends tokens before the reply.
            max_tokens=self.cfg.summary_max_tokens + (2000 if thinking else 0),
            reasoning_effort=self.cfg.summary_reasoning,
        )
        text = (reply.choices[0].message.content or "").strip()
        head, _, rest = text.partition(":")
        head = head.strip().upper()
        if NO_ANSWER in text.upper()[:20] or not text:
            return "NO ANSWER", ""
        if head == "SEARCH" and rest.strip():
            return "SEARCH", rest.strip().strip('"')
        if head == "READ":
            number = re.search(r"\d+", rest)
            if number and 1 <= int(number.group()) <= n_results:
                return "READ", number.group()
            return "NO ANSWER", ""
        if head == "ANSWER":
            return "ANSWER", rest.strip()
        return "ANSWER", text  # no label: take it as the answer

    @staticmethod
    def _not_found(queries: list[str], why: str, search_s: float, summary_s: float) -> Lookup:
        # Told only that the search failed, he answered from memory anyway,
        # confidently and wrongly; told not to guess, he still gave a sunset time.
        return Lookup(queries[0], f"{why} You don't know the answer. Tell the crew you couldn't "
                                  "find out, in your own words, and give no time, number, date or name "
                                  "for it, not even as a joke.", 0, search_s * 1000, summary_s * 1000, queries)

    @staticmethod
    def for_holly(found: Lookup) -> str:
        """The tool result Holly sees. Given the bare answer ("97 moons"), he
        hedged the fact itself ("Ninety-seven. Or perhaps ninety-eight")."""
        if not found.results:
            return found.answer
        return (f"{found.answer}\n(These facts are right. State them plainly, without "
                "changing or doubting them; any joke goes around them.)")

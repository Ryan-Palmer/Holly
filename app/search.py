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

import os
import re
import time
from dataclasses import dataclass

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

SUMMARY_PROMPT = """\
Today is {today}. Answer the question using only the search results below,
in two or three short, plain sentences of fact, for someone to read aloud:
no lists, no links, no markdown. Give numbers, names and dates exactly. If
the results don't contain the answer, reply with exactly: NO ANSWER"""

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


class Searcher:
    def __init__(self, cfg: SearchConfig, lm: LmStudio, model: str):
        if cfg.provider not in PROVIDERS:
            raise ValueError(f"unknown search provider {cfg.provider!r}; use one of {', '.join(PROVIDERS)}")
        self.cfg = cfg
        self.lm = lm
        self.model = model

    def look_up(self, query: str, question: str) -> Lookup:
        """Search for `query` and answer `question` (what was asked) from the results."""
        t0 = time.perf_counter()
        try:
            results = PROVIDERS[self.cfg.provider](query, self.cfg)
            problem = "" if results else "it found nothing"
        except Exception as e:  # a failed search is something Holly can say, not a crash
            # ddgs says "No results found." when the engines answer with CAPTCHAs.
            results, problem = [], f"it failed: {type(e).__name__}: {str(e)[:120]}"
        t1 = time.perf_counter()
        if problem:
            return self._not_found(query, f"The search didn't work ({problem}).", t1 - t0, 0)
        listing = "\n\n".join(f"{i}. {r.title}\n{r.snippet}" for i, r in enumerate(results, 1))
        reply = self.lm.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT.format(today=today())},
                {"role": "user", "content": f"Question: {question}\nSearched for: {query}\n\n"
                                            f"Results:\n\n{listing}"},
            ],
            temperature=0.2,
            max_tokens=self.cfg.summary_max_tokens,
            reasoning_effort="none",
        )
        answer = (reply.choices[0].message.content or "").strip()
        t2 = time.perf_counter()
        if not answer or NO_ANSWER in answer.upper():
            # Given "the results don't say what the weather will be", he
            # forecast it anyway, so this counts as a failed search.
            return self._not_found(query, "The search results didn't have the answer.", t1 - t0, t2 - t1)
        return Lookup(query, answer, len(results), (t1 - t0) * 1000, (t2 - t1) * 1000)

    @staticmethod
    def _not_found(query: str, why: str, search_s: float, summary_s: float) -> Lookup:
        # Told only that the search failed, he answered from memory anyway,
        # confidently and wrongly.
        return Lookup(query, f"{why} Tell the crew you couldn't find out, in your own words. "
                             "Don't guess the answer.", 0, search_s * 1000, summary_s * 1000)

    @staticmethod
    def for_holly(found: Lookup) -> str:
        """The tool result Holly sees. Given the bare answer ("97 moons"), he
        hedged the fact itself ("Ninety-seven. Or perhaps ninety-eight")."""
        if not found.results:
            return found.answer
        return (f"Search results: {found.answer}\n(These facts are right. State them plainly, without "
                "changing or doubting them; any joke goes around them.)")

"""Holly's research: reports written while he sleeps.

Asked to look into something properly, Holly calls the `research` tool, which
queues a job (data/research/jobs.json). While he's asleep, Holly works through
it one step at a time (Holly.start_research): plan the research questions;
research each in turn (search the web and Wikipedia, read the likeliest pages,
take notes, each tied to its source); check the notes for gaps; write the
report from the notes; check it against them; sum it up. Each step is saved
when it's done, so speech can cancel the one in hand and the next sleep picks
up after the last. The report is kept as markdown beside the jobs, for people,
and section by section in his archive, so recall finds it. When it's done, he
speaks up.

Every request goes to the same Gemma as his replies, one at a time. On one GPU,
subagents side by side would share its speed and split its context; what one
gives is a fresh context per question, and researching them in turn gives that
too. And a fixed sequence of steps suits a 26B model better than an open-ended
agent loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import re
import time
from urllib.parse import quote, urlparse

import httpx

from .config import ResearchConfig, SearchConfig
from .lmstudio import LmStudio
from .persona import day, today
from .search import PROVIDERS, _clean, fetch_page

TOOL = {
    "type": "function",
    "function": {
        "name": "research",
        "description": "Research a topic properly, for a written report: done in the background by you, "
                       "while you doze between conversations, and you're told when the report is ready. Also: "
                       "list research under way or "
                       "done, cancel some, sum up a finished report, or open one on the crew's screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "list", "cancel", "summary", "open"]},
                "title": {"type": "string", "description":
                          "A short title: 'Heat pumps for an old house'. For cancel, summary or open: which "
                          "one, in the crew's words ('the heat pump one'); empty for the latest report; to "
                          "cancel everything, 'all'."},
                "request": {"type": "string", "description":
                            "For start: what the crew want to know, in full, with anything they asked to be "
                            "covered or left out."},
            },
            "required": ["action"],
        },
    },
}

PLAN_PROMPT = """\
Today is {today}. You plan the research for a report the crew have asked for.
Split what they want into at most {n} research questions that together cover
it: each specific, answerable from web pages, and not overlapping the others,
the most important first. Reply with only the questions, one per line,
numbered."""

NAVIGATE_PROMPT = """\
Today is {today}. You're researching one question for a report. Below are the
question, your notes so far, and search results you haven't read yet. Reply in
exactly one of these forms:

{options}
DONE: if your notes answer the question well, or nothing here is likely to add
to them."""

READ_OPTION = """\
READ: the numbers of up to {k} results whose pages most likely hold facts your
notes lack, like "READ: 2, 5". Prefer sources that would know: official
sites, encyclopedias, established publications."""

SEARCH_OPTION = """\
SEARCH: a new web search that would find what your notes still lack, worded
differently from the searches already made."""

NOTES_PROMPT = """\
Today is {today}. You take notes for a report. From the page below, write down
the facts that help answer the question, one per line, each starting "- ": in
your own words, short, with names, numbers and dates exactly as the page gives
them, and when something was true if the page says. Only what the page says,
never your own knowledge. At most {k} notes. If the page has nothing that
helps, reply NOTHING."""

CHECK_PROMPT = """\
Below is what the crew asked for, and the notes gathered for each research
question. Reply with up to {k} lines of the form "GAP: a research question",
for important parts of what they asked that the notes leave unanswered and a
web search could answer. If nothing important is missing, reply COMPLETE."""

WRITE_PROMPT = """\
Today is {today}. Write a report for the crew on what they asked for, from the
notes below, in markdown: a "# " title, a "## Summary" of three or four
sentences, then a "## " section for each part, in plain prose (a short list
only where it helps). Use only the notes, never your own knowledge: every fact
must come from them, with its source number in square brackets after it, like
[3]. Where notes disagree, say so; where a question isn't answered, say so in a
sentence. Don't add a list of sources: it's added for you. At most {words}
words."""

VERIFY_PROMPT = """\
Below are notes, each ending with its source number, and a report written from
them. Check every statement in the report against the notes. Remove or correct
any that the notes don't support, and any source number that doesn't match a
note saying it. Change nothing else. Reply with the whole report and nothing
else."""

SUMMARY_PROMPT = """\
Sum up the report below in two or three short, plain sentences, for someone to
read aloud: its main findings, with no markdown, no source numbers and no
links."""

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
WIKI_RESULTS = 3  # per search, alongside the web's
RETRY_S, RETRY_MAX_S = 60, 900  # an announcement that failed: tried again after this, doubling
# A web search the engines refused (ddgs says "No results found" to a CAPTCHA): research is
# in no hurry, so it waits and tries again, this long and then twice it, before giving up.
REFUSED_WAIT_S, REFUSED_TRIES = 20.0, 3
ENTRY_CHARS = 1400  # of a report section, per archive entry (recall reads 1,500 of each)
CITATION = re.compile(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]")
UNFINISHED = ("plan", "research", "check", "write", "verify", "summary", "keep")
# What a finished job keeps: enough to list, sum up, open and announce it. Its working
# papers (every question's results and notes, the draft, the report's text, already in
# its .md) are dropped: they were ~85% of jobs.json.
KEEP = ("id", "title", "request", "asked", "stage", "finished", "file", "summary", "error", "announced",
        "tries", "retry", "work_s", "errors")
# Words that don't say which report: "the report on heat pumps" is "heat pumps".
REPORT_NOISE = {"the", "a", "an", "on", "about", "of", "for", "my", "that", "this", "one", "report", "research",
                "summary", "open", "show", "me", "it", "you", "did", "do", "into"}


def new_question(q: str, gap: bool = False) -> dict:
    return {"q": q, "queries": [], "results": [], "read": [], "notes": [], "done": False,
            "next": ["SEARCH", q], "gap": gap}


def wiki_search(query: str, cfg: ResearchConfig) -> list[dict]:
    resp = httpx.get(WIKI_API.format(lang=cfg.wikipedia), timeout=cfg.timeout_s,
                     headers={"User-Agent": cfg.user_agent},
                     params={"action": "query", "list": "search", "srsearch": query, "srlimit": WIKI_RESULTS,
                             "format": "json", "formatversion": 2})
    resp.raise_for_status()
    return [{"title": f"{h['title']} (Wikipedia)", "snippet": _clean(h.get("snippet", "")), "wiki": h["title"],
             "url": f"https://{cfg.wikipedia}.wikipedia.org/wiki/{quote(h['title'].replace(' ', '_'))}"}
            for h in resp.json().get("query", {}).get("search", [])]


def wiki_page(title: str, cfg: ResearchConfig) -> str:
    """A Wikipedia article as plain text, from its API rather than its HTML."""
    resp = httpx.get(WIKI_API.format(lang=cfg.wikipedia), timeout=cfg.timeout_s,
                     headers={"User-Agent": cfg.user_agent},
                     params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                             "titles": title, "format": "json", "formatversion": 2})
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", [])
    text = (pages[0].get("extract") or "") if pages else ""
    return text[:cfg.page_chars] or "(this page had no text)"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "report"


def announcement(job: dict) -> str:
    """What Holly is told, as the crew's turn, when a job is finished (or given up)."""
    if job["stage"] == "failed":
        return (f"[Your research on '{job['title']}' couldn't be finished: {job.get('error') or 'it went wrong'}. "
                "Tell the crew, in a sentence or two, in your own words. Nobody has spoken: you're speaking up "
                "by yourself.]")
    return (f"[Your research on '{job['title']}' is finished, and the report is written. Tell the crew it's "
            "ready, in a sentence, and ask if they'd like a summary; don't give it yet. If they would, here it "
            f"is: {job['summary']} Give it in two or three sentences in your own words, keeping the facts as "
            "they are, then ask if they'd like you to open the report. Nobody has spoken: you're speaking up "
            "by yourself.]")


class Research:
    """The research jobs, saved to `cfg.dir`/jobs.json after every step."""

    def __init__(self, cfg: ResearchConfig, search_cfg: SearchConfig, lm: LmStudio, model: str, archive=None):
        self.cfg = cfg
        # Research isn't in a hurry: more results, and longer for a page to answer.
        self.search_cfg = dataclasses.replace(search_cfg, max_results=cfg.results,
                                              timeout_s=max(search_cfg.timeout_s, cfg.timeout_s))
        self.providers = [p.strip() for p in cfg.provider.split(",") if p.strip()]
        if unknown := [p for p in self.providers if p not in PROVIDERS]:
            raise ValueError(f"unknown research provider {unknown[0]!r}; use any of {', '.join(PROVIDERS)}")
        self.lm = lm
        self.model = model
        self.archive = archive
        self.file = cfg.dir / "jobs.json"
        self.jobs: list[dict] = []
        self.last_search = 0.0  # time.monotonic() of the last web search, for pacing
        # Wikimedia refuses a User-Agent with no way to reach whoever runs it (403).
        self.wiki = bool(cfg.wikipedia) and ("@" in cfg.user_agent or "http" in cfg.user_agent)
        if cfg.wikipedia and not self.wiki:
            print("  (research: Wikipedia is skipped until [research] user_agent in holly.toml has a contact "
                  "address or URL, which Wikimedia requires)", flush=True)
        if self.file.exists():
            try:
                self.jobs = json.loads(self.file.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                print(f"Couldn't read Holly's research jobs ({e}); starting with none.", flush=True)
            # Finished jobs from before they were slimmed when finishing.
            if sum(slim(j) for j in self.jobs if j["stage"] not in UNFINISHED):
                self._save()

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        try:
            self.cfg.dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self.jobs, indent=1), encoding="utf-8")
            os.replace(tmp, self.file)
        except OSError as e:
            print(f"Couldn't save Holly's research: {e}", flush=True)

    # The tool -------------------------------------------------------------

    def handle(self, args: dict, now: datetime.datetime | None = None) -> str:
        """Do what the tool call asks; returns what Holly is told."""
        now = now or datetime.datetime.now()
        action = str(args.get("action") or "start").lower()
        title = " ".join(str(args.get("title") or "").split())
        if action == "list":
            return self._listing()
        if action == "cancel":
            gone = [j for j in self._matching(title) if j["stage"] in UNFINISHED]
            if not gone:
                return f"No research under way called {title!r}. " + self._listing()
            self.jobs = [j for j in self.jobs if j not in gone]
            self._save()
            return "Cancelled: " + ", ".join(j["title"] for j in gone) + "."
        if action in ("open", "summary"):
            if not (job := self._report(title)):
                return f"There's no finished report on {title!r}. " + self._listing()
            if action == "summary":
                return (f"The report '{job['title']}', finished {day(datetime.date.fromisoformat(job['finished'][:10]))}, "
                        f"says: {job['summary']}\n(Give the crew the gist in two or three sentences, in your own "
                        "words, keeping the facts as they are; then ask if they'd like you to open the report.)")
            path = self.cfg.dir / job["file"]
            try:
                os.startfile(path)  # the crew's own app for markdown
            except (OSError, AttributeError) as e:
                return f"The report couldn't be opened ({e}). It's at {path}. Tell the crew, briefly."
            return f"Opened the report '{job['title']}' on the crew's screen. Say so in a few words."
        request = " ".join(str(args.get("request") or "").split()) or title
        if not request:
            return "Research needs a title and what the crew want to know. Call the research tool again with them."
        title = title or request[:60]
        if [j for j in self.jobs if j["stage"] in UNFINISHED and j["title"].lower() == title.lower()]:
            return f"Research on '{title}' is already under way. Tell the crew, in a sentence."
        ahead = sum(j["stage"] in UNFINISHED for j in self.jobs)
        self.jobs.append({"id": now.strftime("%Y%m%d-%H%M%S"), "title": title, "request": request,
                          "asked": now.isoformat(timespec="seconds"), "stage": "plan", "questions": [],
                          "sources": [], "checked": False, "work_s": 0.0, "errors": 0, "announced": False})
        self._save()
        return (f"Queued: research on '{title}'" + (f", after {ahead} other job{'s' * (ahead != 1)}" if ahead else "")
                + ". You'll work on it yourself while you doze between conversations (your sleep, not the "
                "crew's), and you'll be told when the report is ready. Tell the crew, in a sentence, that you'll "
                "look into it while things are quiet and get back to them. You have no findings yet: give none.")

    def _listing(self) -> str:
        if not self.jobs:
            return "There's no research under way or finished."
        lines = []
        for j in self.jobs:
            if j["stage"] == "done":
                lines.append(f"'{j['title']}': finished {day(datetime.date.fromisoformat(j['finished'][:10]))}")
            elif j["stage"] == "failed":
                lines.append(f"'{j['title']}': given up ({j.get('error')})")
            elif j["stage"] == "plan":
                lines.append(f"'{j['title']}': queued")
            else:
                qs = j["questions"]
                lines.append(f"'{j['title']}': under way, {sum(q['done'] for q in qs)} of {len(qs)} questions "
                             "researched" + ("; writing the report" if j["stage"] not in ("research", "check")
                                             else ""))
        return "Research: " + "; ".join(lines) + "."

    def _report(self, words: str) -> dict | None:
        """The finished report the crew mean: the latest, with no words; else the one whose
        title, request and summary share the most of them ("the heat pump one")."""
        done = [j for j in self.jobs if j["stage"] == "done" and j.get("file")]
        asked = set(re.findall(r"\w+", words.lower())) - REPORT_NOISE
        if not asked:
            return done[-1] if done else None

        def score(j: dict) -> float:
            title = set(re.findall(r"\w+", j["title"].lower()))
            rest = set(re.findall(r"\w+", f"{j['request']} {j.get('summary', '')}".lower()))
            # A word stem in common counts too: "pumps" and "pump".
            near = {a for a in asked if any(w.startswith(a[:4]) for w in title | rest)}
            return 2 * len(asked & title) + len(asked & rest) + 0.5 * len(near)

        best = max(reversed(done), key=score, default=None)  # on a tie, the latest
        return best if best and score(best) > 0 else None

    def _matching(self, title: str) -> list[dict]:
        """Jobs a title means: 'all', the one called exactly that, else those sharing all its words."""
        if title.lower() == "all":
            return list(self.jobs)
        if exact := [j for j in self.jobs if j["title"].lower() == title.lower()]:
            return exact
        asked = set(re.findall(r"\w+", title.lower())) - {"the", "a", "an", "on", "about", "report", "research"}
        return [j for j in self.jobs if asked and asked <= set(re.findall(r"\w+", j["title"].lower()))]

    # Announcing -----------------------------------------------------------

    def ready(self, now: datetime.datetime | None = None) -> dict | None:
        """A job finished (or given up) and not yet announced, whose time to try has come."""
        now = now or datetime.datetime.now()
        for j in self.jobs:
            if (j["stage"] in ("done", "failed") and not j["announced"]
                    and (not j.get("retry") or datetime.datetime.fromisoformat(j["retry"]) <= now)):
                return j
        return None

    def announced(self, job: dict, ok: bool, now: datetime.datetime | None = None) -> None:
        """Said, or not (the reply failed): then tried again in a minute, two, four..."""
        if ok:
            job["announced"] = True
        else:
            tries = job.get("tries", 0)
            job["tries"] = tries + 1
            wait = min(RETRY_S * 2 ** tries, RETRY_MAX_S)
            job["retry"] = ((now or datetime.datetime.now()) + datetime.timedelta(seconds=wait)).isoformat()
        self._save()

    def clear(self) -> None:
        """Every job and report, for good."""
        for j in self.jobs:
            if j.get("file"):
                (self.cfg.dir / j["file"]).unlink(missing_ok=True)
        self.jobs = []
        self._save()

    # The work ---------------------------------------------------------------

    def has_work(self) -> bool:
        return any(j["stage"] in UNFINISHED for j in self.jobs)

    async def step(self) -> str:
        """One step of the first unfinished job; returns what was done, for the console.
        Each step changes the job only once its work is done, and saves it, so a
        step cancelled part-way leaves the job as it was. A failure is counted, and
        after `max_errors` in a row the job is given up; it's raised either way."""
        job = next(j for j in self.jobs if j["stage"] in UNFINISHED)
        t0 = time.monotonic()
        try:
            label = await getattr(self, f"_{job['stage']}")(job)
            job["errors"] = 0
        except Exception as e:
            job["errors"] = job.get("errors", 0) + 1
            if job["errors"] >= self.cfg.max_errors:
                self._give_up(job, f"it kept failing ({type(e).__name__})")
            raise
        finally:
            job["work_s"] = job.get("work_s", 0.0) + time.monotonic() - t0
            self._save()
        if job["stage"] in ("research", "check") and job["work_s"] > self.cfg.max_minutes * 60:
            # Out of time: written up from what's been found.
            for q in job["questions"]:
                q["done"] = True
            job["stage"] = "write"
            label += f"; out of time ({self.cfg.max_minutes:.0f} minutes): writing it up"
            self._save()
        return f"'{job['title']}': {label}"

    def _give_up(self, job: dict, why: str) -> None:
        job["stage"], job["error"] = "failed", why
        job["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        slim(job)

    async def _ask(self, system: str, user: str, max_tokens: int) -> str:
        text, _ = await self.lm.complete([{"role": "system", "content": system},
                                          {"role": "user", "content": user}], self.model, max_tokens)
        return text

    async def _plan(self, job: dict) -> str:
        text = await self._ask(PLAN_PROMPT.format(today=today(), n=self.cfg.questions),
                               f"Title: {job['title']}\nWhat the crew asked for: {job['request']}", 400)
        qs = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln).strip() for ln in text.splitlines()]
        qs = [q for q in qs if len(q) > 10][: self.cfg.questions] or [job["request"]]
        job["questions"] = [new_question(q) for q in qs]
        job["stage"] = "research"
        return f"planned {len(qs)} questions: " + " | ".join(qs)

    async def _research(self, job: dict) -> str:
        q = next((q for q in job["questions"] if not q["done"]), None)
        if not q:
            job["stage"] = "write" if job["checked"] else "check"
            return "every question researched"
        n = job["questions"].index(q) + 1
        where = f"{n}/{len(job['questions'])}"
        kind, arg = q["next"] or ("DECIDE", None)
        if kind == "SEARCH":
            found = await self._search(arg)
            known = {r["url"] for r in q["results"]}
            q["results"] += [r for r in found if r["url"] not in known]
            q["queries"].append(arg)
            q["next"] = None
            return f"{where} searched: {arg} ({len(found)} results)"
        if kind == "READ":
            i, rest = arg[0], arg[1:]
            r = q["results"][i - 1]
            notes = await self._notes(q["q"], r)
            q["read"].append(i)
            if notes:
                n_src = self._source(job, r)
                q["notes"] += [f"{note} [{n_src}]" for note in notes]
            q["next"] = ["READ", rest] if rest else None
            return f"{where} read: {r['url']} ({len(notes)} notes)"
        return f"{where} " + await self._decide(q)

    async def _search(self, query: str) -> list[dict]:
        """The web's results (the first provider to give any), then Wikipedia's. Paced:
        the free engines answer a burst of searches with CAPTCHAs; refused all the
        same, it waits and tries again, rather than spend one of the question's searches."""

        def web() -> tuple[list[dict], bool]:
            """Results, and whether every provider refused or failed (rather than finding nothing)."""
            refused = True
            for name in self.providers:
                try:
                    if results := PROVIDERS[name](query, self.search_cfg):
                        return [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results], False
                    refused = False
                except Exception as e:  # a refusal (CAPTCHA) or an outage: the next, or nothing this time
                    print(f"  (research search, {name}: {type(e).__name__}: {str(e)[:80]})", flush=True)
            return [], refused

        for attempt in range(REFUSED_TRIES):
            if (wait := self.last_search + self.cfg.pause_s - time.monotonic()) > 0:
                await asyncio.sleep(wait)
            self.last_search = time.monotonic()
            found, refused = await asyncio.to_thread(web)
            if found or not refused or attempt == REFUSED_TRIES - 1:
                break
            await asyncio.sleep(REFUSED_WAIT_S * 2 ** attempt)

        def wiki() -> list[dict]:
            if not self.wiki:
                return []
            try:
                return wiki_search(query, self.cfg)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:  # refused, User-Agent and all: not again this session
                    self.wiki = False
                print(f"  (research search, Wikipedia: HTTP {e.response.status_code})", flush=True)
            except Exception as e:
                print(f"  (research search, Wikipedia: {type(e).__name__}: {str(e)[:80]})", flush=True)
            return []

        return found + [r for r in await asyncio.to_thread(wiki) if r["url"] not in {f["url"] for f in found}]

    async def _notes(self, question: str, r: dict) -> list[str]:
        if r.get("wiki"):
            text = await asyncio.to_thread(wiki_page, r["wiki"], self.cfg)
        else:
            text = await asyncio.to_thread(fetch_page, r["url"], self.search_cfg, self.cfg.page_chars)
        if text.startswith("(") and len(text) < 200:  # couldn't be read
            return []
        reply = await self._ask(NOTES_PROMPT.format(today=today(), k=self.cfg.notes),
                                f"Question: {question}\n\nPage: {r['title']}\n{r['url']}\n\n{text}",
                                60 * self.cfg.notes)
        notes = [ln.strip()[2:].strip() for ln in reply.splitlines() if ln.strip().startswith(("- ", "* "))]
        return [n for n in notes if n][: self.cfg.notes]

    @staticmethod
    def _source(job: dict, r: dict) -> int:
        """The job's number for this source, adding it if new."""
        for i, s in enumerate(job["sources"], 1):
            if s["url"] == r["url"]:
                return i
        job["sources"].append({"title": r["title"], "url": r["url"]})
        return len(job["sources"])

    async def _decide(self, q: dict) -> str:
        """READ, SEARCH or DONE for this question, within its limits."""
        unread = [i for i in range(1, len(q["results"]) + 1) if i not in q["read"]]
        reads_left = self.cfg.reads - len(q["read"])
        can_read = bool(unread) and reads_left > 0
        can_search = len(q["queries"]) < self.cfg.searches
        if not can_read and not can_search:
            q["done"] = True
            return "done (limits reached)"
        options = "\n".join(([READ_OPTION.format(k=min(2, reads_left))] if can_read else [])
                            + ([SEARCH_OPTION] if can_search else []))
        listing = "\n\n".join(f"{i}. {q['results'][i - 1]['title']} ({urlparse(q['results'][i - 1]['url']).netloc})"
                              f"\n{q['results'][i - 1]['snippet']}" for i in unread) or "(none)"
        user = (f"Question: {q['q']}\n\nSearches made: {'; '.join(q['queries'])}\n\nNotes so far:\n"
                + ("\n".join(f"- {n}" for n in q["notes"]) or "(none yet)")
                + f"\n\nResults not yet read:\n\n{listing}")
        text = await self._ask(NAVIGATE_PROMPT.format(today=today(), options=options), user, 80)
        head, _, rest = text.partition(":")
        head = head.strip().upper()
        if head == "READ" and can_read:
            picks = [int(x) for x in re.findall(r"\d+", rest) if int(x) in unread][: min(2, reads_left)]
            if picks:
                q["next"] = ["READ", picks]
                return f"to read {', '.join(map(str, picks))}"
        if head == "SEARCH" and can_search:
            query = rest.strip().strip('"')
            if query and query.lower() not in (x.lower() for x in q["queries"]):
                q["next"] = ["SEARCH", query]
                return f"to search: {query}"
        q["done"] = True
        return f"done ({len(q['notes'])} notes)"

    @staticmethod
    def _all_notes(job: dict) -> str:
        return "\n\n".join(f"Question: {q['q']}\n" + ("\n".join(f"- {n}" for n in q["notes"]) or "(nothing found)")
                           for q in job["questions"])

    async def _check(self, job: dict) -> str:
        text = await self._ask(CHECK_PROMPT.format(k=self.cfg.gaps),
                               f"What the crew asked for: {job['request']}\n\n{self._all_notes(job)}", 200)
        gaps = [ln.split(":", 1)[1].strip() for ln in text.splitlines()
                if ln.strip().upper().startswith("GAP:") and len(ln.split(":", 1)[1].strip()) > 10]
        gaps = gaps[: self.cfg.gaps]
        job["questions"] += [new_question(g, gap=True) for g in gaps]
        job["checked"] = True
        job["stage"] = "research" if gaps else "write"
        return "checked: " + ("gaps: " + " | ".join(gaps) if gaps else "nothing missing")

    async def _write(self, job: dict) -> str:
        if not any(q["notes"] for q in job["questions"]):
            self._give_up(job, "nothing useful turned up in the searches")
            return "given up: nothing found"
        job["draft"] = await self._ask(WRITE_PROMPT.format(today=today(), words=self.cfg.report_words),
                                       f"What the crew asked for: {job['request']}\n\nNotes:\n\n"
                                       f"{self._all_notes(job)}", self.cfg.report_words * 2 + 200)
        job["stage"] = "verify"
        return f"report drafted, {len(job['draft'].split())} words"

    async def _verify(self, job: dict) -> str:
        text = await self._ask(VERIFY_PROMPT, f"Notes:\n\n{self._all_notes(job)}\n\nReport:\n\n{job['draft']}",
                               self.cfg.report_words * 2 + 200)
        # A checked report that lost most of the draft, or its headings, went wrong: keep the draft.
        ok = text.lstrip().startswith("#") and len(text) > len(job["draft"]) * 0.6
        job["report"] = text if ok else job["draft"]
        job["stage"] = "summary"
        return f"report checked against the notes, {len(job['report'].split())} words" + ("" if ok else
                                                                                          " (the check went wrong: draft kept)")

    async def _summary(self, job: dict) -> str:
        job["summary"] = await self._ask(SUMMARY_PROMPT, job["report"], 200)
        job["stage"] = "keep"
        return f"summed up: {job['summary']}"

    async def _keep(self, job: dict) -> str:
        """The report to a markdown file, with its sources, and into his archive."""
        finished = datetime.datetime.now()
        sources = "\n".join(f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(job["sources"], 1))
        asked = datetime.datetime.fromisoformat(job["asked"])
        text = (f"{job['report'].rstrip()}\n\n## Sources\n\n{sources}\n\n---\n\n*Asked for {day(asked.date())}: "
                f"\"{job['request']}\". Researched by Holly while he slept.*\n")
        name = f"{job['id']}-{slug(job['title'])}.md"
        self.cfg.dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.dir / name).write_text(text, encoding="utf-8")
        if self.archive:
            head = f"(Research report: {job['title']}, written {day(finished.date())}) "
            entries = [(finished.isoformat(timespec="seconds"), head + part[:ENTRY_CHARS])
                       for part in sections(job["report"])]
            self.archive.add(entries)
            try:
                await asyncio.to_thread(self.archive.embed_missing)
            except Exception as e:  # recall still finds them by keyword; vectors are made at the next recall
                print(f"  (couldn't embed the report: {e!r})", flush=True)
        job["file"] = name
        job["finished"] = finished.isoformat(timespec="seconds")
        job["stage"] = "done"
        slim(job)
        return f"report kept: {self.cfg.dir / name}"


def slim(job: dict) -> bool:
    """Drop a finished job's working papers, in place (see KEEP). True if any went."""
    gone = [k for k in job if k not in KEEP]
    for k in gone:
        del job[k]
    return bool(gone)


def sections(report: str) -> list[str]:
    """The report's sections for his archive, source numbers dropped (they mean
    nothing out of the report): the summary, then each part, under its heading."""
    title = next((ln.lstrip("# ").strip() for ln in report.splitlines() if ln.startswith("# ")), "")
    parts = re.split(r"\n(?=## )", "\n" + report)
    out = []
    for part in parts:
        body = CITATION.sub("", part).strip()
        if not body or body.startswith("# ") and "\n" not in body:
            continue
        body = re.sub(r"^#+\s*", "", body, flags=re.MULTILINE)
        out.append(f"{title}: {body}" if title and not body.startswith(title) else body)
    return out

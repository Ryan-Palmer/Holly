"""Holly's research: reports written while he sleeps.

Asked to look into something properly, Holly calls the `research` tool, which
queues a job (data/research/jobs.json). While he's asleep, Holly works through
it one step at a time (app/background.py): plan the research questions;
research each in turn (search the web and Wikipedia, read the likeliest pages,
take notes, each tied to its source); check the notes for gaps; write the
report from the notes; check it against them; sum it up. Each step is saved
when it's done, so speech can cancel the one in hand and the next sleep picks
up after the last. The report is kept as markdown beside the jobs. When it's
done, he speaks up; asked about it later, the tool finds it from the crew's
words (its title, request, summary or text) and he reads the whole of it. It
isn't put in his archive: what he and the crew say about it goes there, as
any talk does, and a second copy of the report, in fragments, only led recall
to thin answers.

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
import threading
import time
from urllib.parse import quote, urlparse

import httpx

from .announce import retry_wait
from .config import ResearchConfig, SearchConfig
from .console import DIM, RESET
from .guards import REPORT_WANTED
from .lmstudio import LmStudio
from .persona import day, today
from .reports import Reports, telling
from .search import PROVIDERS, _clean, fetch_page
from .tools import Tool, Turn

ACTIONS = ("start", "list", "cancel", "summary", "read", "open")

TOOL = {
    "type": "function",
    "function": {
        "name": "research",
        "description": "Research a topic properly, for a written report: done in the background by you, "
                       "while you doze between conversations, and you're told when the report is ready. Also: "
                       "sum up a finished report, read the whole of one (to talk the crew through it or answer "
                       "details), or open one on the crew's screen, found from the crew's own words for it, "
                       "among all your reports; list research under way or finished this week; or cancel some.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "title": {"type": "string", "description":
                          "A short title: 'Heat pumps for an old house'. For cancel, summary, read or open: which "
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
[3]. Where sources disagree, say so; where a question isn't answered, say so in
a sentence. Write as the report itself: never mention the notes, or how it was
researched. Don't add a list of sources: it's added for you. At most {words}
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

EXPAND_PROMPT = """\
Someone is referring, in a few words, to a research report written for them
earlier ("the dog one", "that coffee report"). List up to eight words such a
report's title or text would likely contain: the things meant, and other words
for them ("dog" might be "dog, puppy, training, pet"), in British English
("tap", not "faucet"). Reply with only the words, separated by commas."""

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
WIKI_RESULTS = 3  # per search, alongside the web's
# A web search the engines refused (ddgs says "No results found" to a CAPTCHA): research is
# in no hurry, so it waits and tries again, this long and then twice it, before giving up.
REFUSED_WAIT_S, REFUSED_TRIES = 20.0, 3
CITATION = re.compile(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]")
UNFINISHED = ("plan", "research", "check", "write", "verify", "summary", "keep")
LIST_DAYS, LIST_MAX = 7, 10  # listing: the reports finished this lately, at most this many


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
            "they are, then ask if they'd like you to open the report or talk them through it. Nobody has "
            "spoken: you're speaking up by yourself.]")


class Research:
    """The research jobs under way, saved to `cfg.dir`/jobs.json after every step; once
    finished, each goes to the reports' index (app/reports.py), `cfg.dir`/reports.db."""

    def __init__(self, cfg: ResearchConfig, search_cfg: SearchConfig, lm: LmStudio, model: str,
                 embed_model: str = ""):
        self.cfg = cfg
        # Research isn't in a hurry: more results, and longer for a page to answer.
        self.search_cfg = dataclasses.replace(search_cfg, max_results=cfg.results,
                                              timeout_s=max(search_cfg.timeout_s, cfg.timeout_s))
        self.providers = [p.strip() for p in cfg.provider.split(",") if p.strip()]
        if unknown := [p for p in self.providers if p not in PROVIDERS]:
            raise ValueError(f"unknown research provider {unknown[0]!r}; use any of {', '.join(PROVIDERS)}")
        self.lm = lm
        self.model = model
        self.file = cfg.dir / "jobs.json"
        self.jobs: list[dict] = []
        # The tool runs in a worker thread (Holly._run_tool), research steps on the event loop:
        # both change the queue and write jobs.json, so they take turns.
        self._lock = threading.RLock()
        self.last_search = 0.0  # time.monotonic() of the last web search, for pacing
        self.shelf = Reports(cfg.dir / "reports.db", lm, embed_model, self._expand)  # the finished ones
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
            # Finished jobs from before reports.db: into it, out of the queue.
            if done := [j for j in self.jobs if j["stage"] not in UNFINISHED]:
                for j in done:
                    self._shelve(j)
                self._save()

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        try:
            with self._lock:
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
        # He tried "find" (the tool says reports are found from the crew's words): it means sum up.
        action = {"find": "summary", "search": "summary", "summarise": "summary", "show": "open"}.get(action, action)
        if action not in ACTIONS:
            # Anything unknown used to start research: "find" the bike report queued a job on it.
            return f"The research tool has no action {action!r}; use one of: {', '.join(ACTIONS)}."
        title = " ".join(str(args.get("title") or "").split())
        if action == "list":
            return self._listing()
        if action == "cancel":
            with self._lock:
                gone = [j for j in self._matching(title) if j["stage"] in UNFINISHED]
                if gone:
                    self.jobs = [j for j in self.jobs if j not in gone]
                    self._save()
            if not gone:
                return f"No research under way called {title!r}. " + self._listing()
            return "Cancelled: " + ", ".join(j["title"] for j in gone) + "."
        if action in ("open", "summary", "read"):
            if not (picks := self.shelf.find(title)):
                return f"There's no finished report on {title!r}. " + self._listing()
            if len(picks) > 1 or (action == "open" and not picks[0].get("sure")):
                return self.which(picks)
            job = picks[0]
            finished = day(datetime.date.fromisoformat(job["finished"][:10]))
            path = self.cfg.dir / job["file"]
            # Others nearly as likely: if the crew say it's the wrong one, Holly knows the right title.
            also = (" (If the crew meant another: there's also " + " and ".join(f"'{t}'" for t in job["also"])
                    + ".)") if job.get("also") else ""
            if not job.get("sure"):
                # The only likely one, but not certain ("the plumbing one", "Fixing a dripping tap"):
                # asked to confirm it, he dithered and made up others. Given it, he can tell.
                also = (" (This was the only likely match for what the crew said, not a certain one: if it "
                        "plainly isn't what they meant, say you have no report on that.)")
            if action == "summary":
                return (f"The report '{job['title']}', finished {finished}, says: {job['summary']}\n(Give the crew "
                        "the gist in two or three sentences, in your own words, keeping the facts as they are; "
                        "then ask if they'd like you to open the report or talk them through it.)" + also)
            if action == "read":
                try:
                    text = for_reading(path.read_text(encoding="utf-8"))
                except OSError as e:
                    return f"The report couldn't be read ({e}). Tell the crew, briefly."
                return (f"The report '{job['title']}', finished {finished}, in full:\n\n{text}\n\n(This is the "
                        "whole report, and it's right. Answer the crew from it, keeping the facts as they are. "
                        "Talking them through it, take a few points at a time, in your own words, and ask if they "
                        "want more; never read out the markdown or list everything at once.)" + also)
            try:
                os.startfile(path)  # the crew's own app for markdown
            except (OSError, AttributeError) as e:
                return f"The report couldn't be opened ({e}). It's at {path}. Tell the crew, briefly."
            return f"Opened the report '{job['title']}' on the crew's screen. Say so in a few words." + also
        request = " ".join(str(args.get("request") or "").split()) or title
        if not request:
            return "Research needs a title and what the crew want to know. Call the research tool again with them."
        title = title or request[:60]
        with self._lock:
            if [j for j in self.jobs if j["stage"] in UNFINISHED and j["title"].lower() == title.lower()]:
                return f"Research on '{title}' is already under way. Tell the crew, in a sentence."
            ahead = sum(j["stage"] in UNFINISHED for j in self.jobs)
            # Unique, though two may be asked for in one breath ("research X and Y"): the second
            # of the same second, its id reused, replaced the first in the reports' index.
            jid, k = now.strftime("%Y%m%d-%H%M%S"), 2
            while any(j["id"] == jid for j in self.jobs) or self.shelf.has(jid):
                jid, k = f"{now:%Y%m%d-%H%M%S}-{k}", k + 1
            self.jobs.append({"id": jid, "title": title, "request": request,
                              "asked": now.isoformat(timespec="seconds"), "stage": "plan", "questions": [],
                              "sources": [], "checked": False, "work_s": 0.0, "errors": 0, "announced": False})
            self._save()
        return (f"Queued: research on '{title}'" + (f", after {ahead} other job{'s' * (ahead != 1)}" if ahead else "")
                + ". You'll work on it yourself while you doze between conversations (your sleep, not the "
                "crew's), and you'll be told when the report is ready. Let the crew know it's in hand, briefly "
                "and in your own words. You have no findings yet: give none.")

    def find(self, words: str, least: float = 0.0, widen: bool = True) -> list[dict]:
        """The finished reports these words may mean (app/reports.py), scoring above
        `least` (1 asks for more than one of a report's words in passing); none
        without words that say which. `widen`: let Gemma add words for what's meant."""
        return self.shelf.find(words, least, widen=widen) if telling(words) else []

    def finished_titles(self) -> list[str]:
        return self.shelf.titles()

    def _listing(self) -> str:
        """What's under way, and what's finished lately: not every report ever, which
        at a few a day would soon be hundreds."""
        lines = []
        for j in self.jobs:
            if j["stage"] == "plan":
                lines.append(f"'{j['title']}': queued")
            else:
                qs = j["questions"]
                lines.append(f"'{j['title']}': under way, {sum(q['done'] for q in qs)} of {len(qs)} questions "
                             "researched" + ("; writing the report" if j["stage"] not in ("research", "check")
                                             else ""))
        recent, older = self.shelf.recent(days=LIST_DAYS, at_most=LIST_MAX)
        for r in recent:
            when = day(datetime.date.fromisoformat(r["finished"][:10]))
            lines.append(f"'{r['title']}': finished {when}" if r["stage"] == "done" else
                         f"'{r['title']}': given up {when} ({r['error']})")
        if not lines and not older:
            return "There's no research under way or finished."
        more = (f" And {older} older report{'s' * (older != 1)}, not listed. To find one, call the research tool "
                "to read, sum up or open it with the crew's own words for it as its title: it searches them all."
                if older else "")
        return ("Research: " + "; ".join(lines) + "." if lines else "Nothing under way or finished lately.") + more

    @staticmethod
    def which(reports: list[dict]) -> str:
        """What Holly is told when it isn't clear which report is meant: he picks, knowing
        what's being talked about ("the dog one" is "Puppy training basics"), or asks."""
        names = [f"'{j['title']}'" for j in reports]
        if len(names) == 1:  # only for open: it shows on the crew's screen, so asked first
            return (f"The likeliest report is {names[0]}, but it isn't certain. Ask the crew if that's the one, "
                    "and if so, call the research tool again with its exact title.")
        return (f"Not sure which report is meant. The best match is {names[0]}, then " + ", ".join(names[1:])
                + ". If one of them plainly fits what the crew said, call the research tool again at once with its "
                "exact title, without asking. Ask the crew only if several really might be meant, naming at most "
                "four, briefly. If none fits, say you have no report on that.")

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
        return self.shelf.ready(now)

    def announced(self, report: dict, ok: bool, now: datetime.datetime | None = None) -> None:
        """Said, or not (the reply failed): then tried again in a minute, two, four..."""
        wait = retry_wait(report["tries"] or 0)
        self.shelf.announced(report, ok, (now or datetime.datetime.now()) + datetime.timedelta(seconds=wait))

    def clear(self) -> None:
        """Every job and report, for good."""
        for f in self.shelf.clear():
            (self.cfg.dir / f).unlink(missing_ok=True)
        with self._lock:
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
        was = job["stage"]
        job["stage"], job["error"] = "failed", why
        job["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            self._shelve(job)
        except Exception:
            job["stage"] = was  # still in the queue, to be given up (and announced) next time
            raise

    def _shelve(self, job: dict) -> dict:
        """A finished (or given up) job into the reports' index, with its report's text, and
        out of the queue. Returns its row."""
        body = ""
        if job.get("file"):
            try:
                body = for_reading((self.cfg.dir / job["file"]).read_text(encoding="utf-8"))
            except OSError:
                pass
        row = self.shelf.add(job, body)
        with self._lock:
            self.jobs = [j for j in self.jobs if j is not job]
        return row

    async def _ask(self, system: str, user: str, max_tokens: int) -> str:
        text, _ = await self.lm.aask(system, user, self.model, max_tokens)
        return text

    def _expand(self, words: str) -> str:
        """What a report the crew mean by `words` would contain (see Reports.find). Blocking,
        ~0.3 s: called from the research tool, in a worker thread."""
        return self.lm.ask(EXPAND_PROMPT, words, self.model, 60, temperature=0.0)

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
        # "- NOTHING" is no note: kept, the report said a source stated "NOTHING".
        return [n for n in notes if n and n.strip(" .*").upper() != "NOTHING"][: self.cfg.notes]

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
        """The report to a markdown file, with its sources."""
        finished = datetime.datetime.now()
        sources = "\n".join(f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(job["sources"], 1))
        asked = datetime.datetime.fromisoformat(job["asked"])
        text = (f"{job['report'].rstrip()}\n\n## Sources\n\n{sources}\n\n---\n\n*Asked for {day(asked.date())}: "
                f"\"{job['request']}\". Researched by Holly while he slept.*\n")
        name = f"{job['id']}-{slug(job['title'])}.md"
        self.cfg.dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.dir / name).write_text(text, encoding="utf-8")
        job["file"] = name
        job["finished"] = finished.isoformat(timespec="seconds")
        job["stage"] = "done"
        try:
            self._shelve(job)
        except Exception:
            # Not in the index (its database locked, say): still to keep, so the next step
            # tries again. Left "done" in the queue, it was skipped for ever: never announced
            # or found, though its file was written.
            job["stage"] = "keep"
            raise
        try:  # its vector, for finding it by meaning; else made when next looked for
            await asyncio.to_thread(self.shelf.embed_missing)
        except Exception as e:
            print(f"  (couldn't embed the report: {e!r})", flush=True)
        return f"report kept: {self.cfg.dir / name}"


def for_reading(md: str) -> str:
    """A report's file as Holly reads it: no sources list, footer or source numbers,
    which mean nothing to say aloud."""
    md = re.split(r"\n## Sources\s*\n", md)[0]
    return CITATION.sub("", md).strip()


# Did the crew just take him up on researching something properly? Told it only by text (a
# question mark, and a word like "report" in his last reply), "Yes, please." to "I can do a
# proper research job on it if you want." wasn't taken for one: he said he'd started and
# called nothing, 5 times in 6, or was held and asked all over again. So the model judges,
# from what he said and their reply, whatever the wording (accepted).
ACCEPTED_PROMPT = """\
Holly, the household computer, can research something properly while he dozes and write a report on it, or \
search the web for it now. You're given what he last said, and the crew's reply to it.
YES: they're taking him up on researching it properly or writing a report, or asking him to ("yes, please", \
"go on then", "a report, please", "do it properly"), where what he said offered that, or asked which they'd \
like.
NO: anything else: a quick search now ("just search it"), something else, not now, another question, or yes \
to something else he asked (a summary, opening a report, reading it out).
Reply with one word: YES or NO."""


def accepted(lm, model: str | None, said: str, reply: str, quiet: bool = True) -> bool:
    """Are the crew, replying `reply` to his `said`, taking him up on research? If it can't be
    asked, False (as before), or with `quiet` off (the evals), the error."""
    try:
        return lm.ask(ACCEPTED_PROMPT, f"Holly said: {said}\nThey reply: {reply}", model, max_tokens=4,
                      temperature=0).strip().upper().startswith("YES")
    except Exception:
        if not quiet:
            raise
        return False


class ResearchTool(Tool):
    """Starting research is held unless the crew asked for a report outright, or chose one
    when he asked: then he asks them whether they'd like a quick search now or a report."""

    name, schema, rules, local = "research", TOOL, "research", True

    def __init__(self, research: Research):
        self.research = research
        self.held_turn = -2  # the reply a start was last held in: the crew's next turn answers his question

    async def run(self, args: dict, turn: Turn) -> str:
        if (str(args.get("action") or "start").lower() == "start" and not REPORT_WANTED.search(turn.user)
                and not turn.answering_choice):
            print(f"{DIM}  (research start held: not asked for outright; he asks which){RESET}", flush=True)
            self.held_turn = turn.number
            # No words given him to say: handed "a quick search now or a report while you
            # doze", he said exactly that, every time.
            return ("Not started: the crew didn't say they wanted a report. Ask them which they'd like: "
                    "you searching it now, or researching it properly while you doze and writing it up. "
                    "Put it your own way, about what they asked, and not as you've put it before. Then do "
                    "what they choose. Don't say you've started anything.")
        # In a thread: finding a report by meaning asks LM Studio for a vector.
        result = await asyncio.to_thread(self.research.handle, args)
        shown = result.split(". Tell the crew")[0].split("\n")[0]  # a report read: its first line
        print(f"{DIM}  (research {args.get('action') or 'start'}) {shown}{RESET}", flush=True)
        return result

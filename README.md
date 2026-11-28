# Holly

A voice assistant in the character of Holly, the ship's computer from Red
Dwarf. Speak to him and he answers out loud, in a cloned voice, with an
optional animated face. Everything runs locally on one Windows PC.

He can search the web, give the weather, the news and the time, set timers
and reminders, play music on Spotify, and remember what you've told him
across sessions, as well as everything that happened on Red Dwarf.

## Requirements

- Windows, with an NVIDIA GPU with about 28 GB of VRAM for everything,
  including the face (built on an RTX 5090, 32 GB).
- Python 3.11 or newer.
- [LM Studio](https://lmstudio.ai), with its server on port 1234 and these
  models downloaded:
  - `google/gemma-4-26b-a4b-qat` (the chat model)
  - `text-embedding-nomic-embed-text-v1.5` (optional, for recall by meaning)

  Holly loads and unloads them himself. If LM Studio's engine crashes, it
  loads the chat model again by itself, with its own defaults (a short
  context, which cuts his prompts). Holly notices before his next reply and
  loads it his way. To make LM Studio's own reload right too, set the
  model's defaults there to a 30,208-token context and one parallel slot.
- A microphone and speakers. A headset is best: while Holly talks, the mic
  is ignored.

## Install

```
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -e .
```

Install PyTorch first, from PyTorch's own index. Newer NVIDIA cards need a
build for CUDA 12.8 or later, and without one everything silently runs on
the CPU.

## Data

None of this is in the repo: voices, faces and the show aren't ours to
publish. Everything below goes in git-ignored folders.

- **Holly's voice** (required). A clean clip of about 20 s of the voice to
  clone, and its transcript, at `voice/data/holly/reference.wav` and
  `voice/data/holly/reference.txt`. `voice/prep.py` builds them from
  recordings, in its own venv (setup in `voice/requirements.txt`):

  ```
  voice\.venv\Scripts\python voice\prep.py fetch URL       # or put a file at voice/data/raw/ID/source.*
  voice\.venv\Scripts\python voice\prep.py separate ID     # take the voice off music and laughter
  voice\.venv\Scripts\python voice\prep.py segment ID      # cut into lines and transcribe
  voice\.venv\Scripts\python voice\prep.py trim ID
  voice\.venv\Scripts\python voice\prep.py review ID       # keep lines, and mark some as the reference
  voice\.venv\Scripts\python voice\prep.py export          # writes voice/data/holly/
  ```

- **The ship's records** (optional). Holly's memory of Red Dwarf, from the
  Red Dwarf fandom wiki. With LM Studio running:

  ```
  .venv\Scripts\python -m app.records fetch    # ~25 min, once
  .venv\Scripts\python -m app.records build    # data/records.db
  ```

- **The face** (optional). A still of Holly's face at
  `face/data/holly_colour.jpg`, plus Ditto and its own venv:
  see `face/README.md`.

Holly keeps his memory in `data/`: his history and notes, an archive of
past conversations, timers, research reports, and the Spotify login.

## Run

```
.venv\Scripts\python -m app.listen            # talk to him
.venv\Scripts\python -m app.listen --face     # with his face
.venv\Scripts\python -m app.speak             # type to him, and he answers out loud
.venv\Scripts\python -m app.chat              # text only, no tools or memory
```

Run one Holly at a time: two won't fit on the GPU.

The first start after installing spends a few minutes compiling the voice.
Later starts take about a minute.

Say "Holly" near the start or end of what you say. For 30 s after he last
spoke he's awake and listens without it.

You can pause mid-sentence. At each pause, a small model (Smart Turn, on
the CPU, fetched once from Hugging Face) judges from your voice whether
you've finished. If it's sure you have, he answers after 200 ms. If not
("I was thinking we could… go to the beach"), he waits up to 1.5 s for the
rest, because being cut off is worse than a slower answer. With Holly
closed, `python -m app.turn` tries it on your voice and shows what it makes
of each pause. On a first try it cut in on 10 of 34 pauses mid-thought,
where waiting 400 ms cut in on 30. It answered finished questions sooner
when it was sure, and the rest after 1.5 s. Settings: `[listen]` in
`holly.toml`.

| Option | |
|---|---|
| `--face` / `--no-face` | show his face |
| `--no-search` | no web search, weather, news or clock |
| `--no-wake` | answer everything, without his name (`app.listen`) |
| `--forget` | start afresh, setting his memory of you aside |
| `--reset` | start afresh, deleting his memory of you |
| `--model KEY` | another LM Studio chat model (its settings: `[models]` in `holly.toml`) |

Typed, `/reset` clears his memory and `/quit` leaves.

## Features

- **Conversation.** He answers in character, in one to three sentences,
  speaking as the reply is written.
- **Memory.** Recent talk is kept word for word. Older talk is folded into
  dated notes while he's idle, and kept whole in an archive, which his
  recall tool searches by keyword, meaning and day. It carries over between
  sessions.
- **The ship's records.** Recall searches them too, and status reports are
  written from them.
- **Web search.** Settled facts come from Wikipedia first. What changes
  from week to week (prices, rates, the latest winner) goes to the web: free
  (DuckDuckGo and others) without setup. For better results, put a
  [Brave Search API](https://brave.com/search/api/) key in the Windows user
  environment variable `BRAVE_API_KEY`.
- **Weather.** Forecasts for any town, from Open-Meteo. No setup.
- **Clock.** The time here or anywhere, and how many days until a day.
- **News.** The BBC's latest headlines, by subject or on anything you name
  (Google News for a story the BBC hasn't got), or a round-up of the main
  subjects, a sentence each. Ask for more about a story ("tell me more about
  the Nasa one") and he reads its article on the BBC. No setup.
- **Status reports.** Ask for a status report and he gives a line each on
  what Lister, Rimmer, the Cat and Kryten have been up to lately, then an
  "Additional" item from the wider Red Dwarf universe, all made up. It's
  written from their pages of the ship's records (three drafts, the best of
  each line kept), so it's better with the records built. Ask where one of
  them is ("where's Kryten?", "what's Rimmer up to?") and he makes up a line
  about just them, the same way, carrying on from his last report. Settings
  under `[status]` in `holly.toml`.
- **Timers and reminders.** "Set a timer for 20 minutes", "remind me at half
  seven tomorrow to ring Mum". He speaks up when one is due, even after a
  restart.
- **Spotify.** Play, pause, skip, volume, what's playing. Needs Spotify
  Premium. Create an app at
  [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
  with the redirect URI `http://127.0.0.1:8888/callback` and the Web API,
  put its client ID in the Windows user environment variable
  `SPOTIFY_CLIENT_ID`, then sign in once:

  ```
  .venv\Scripts\python -m app.spotify login
  ```

- **Research.** "Holly, can you research heat pumps for an old house?" Ask
  him to look into something without saying which you want, and he asks
  whether you'd like a quick search now or a report. Take him up on
  anything he offers, however you put it ("yes, please", "go on then",
  "get it now"), and he does it. He works on reports
  only while he's asleep, a step at a time, and stops whenever anyone
  speaks. He splits the request into questions, searches the web
  (free, through ddgs) and Wikipedia, reads the likeliest pages, keeps notes
  with their sources, and writes a report from the notes. When it's done he
  says so and asks if you'd like a summary, then whether to open it. The
  report is saved as markdown, with its sources, in `data/research/`. Any
  time later, ask him to sum one up, talk you through it, answer a detail
  from it, or open it. He finds it from your words or its day, and if more
  than one might be meant, he asks which. No setup. Limits are under
  `[research]` in `holly.toml`.
- **His face.** Ditto animates the still from his voice, in a window that
  scales to any size (F11 or a double-click for full screen), on an old
  monitor if you like (see `face/README.md`). It dozes off when he does. If the window is closed, or its sound stalls, his voice
  carries on through the speakers.

## Configuration

`holly.toml` holds every setting, with comments: models (and each chat
model's quirks), microphone and speakers, wake words, memory budgets, search providers and more. Holly's
character is in `persona/`: `system_prompt.md` and example exchanges in
`examples.toml`. Each tool's rules for when and how he uses it sit beside its
code in `app/`, as a `.md` with the same name (`search.md` for `search.py`).

## Evals

To check a change to his character, or another model, score him:

```
.venv\Scripts\python -m app.evals                      # on holly.toml's model
.venv\Scripts\python -m app.evals --model KEY --label try-31b
.venv\Scripts\python -m app.evals --compare            # every scorecard so far
```

A full run takes a couple of hours, and puts him through these, with a
silent voice and his memory in a temporary folder:

- **Persona**: the prompts in `persona/test_prompts.txt`, each scored for
  length, anything a voice can't say, an assistant's manners, questions back,
  the tool it needs, lines recited from his examples, and repetition. The
  reply judge scores each reply 1-5 as you would: it's shown your own scored
  replies and notes (`persona/judge_examples.json`) and no rules of ours.
- **Scenarios** (`app/scenarios.py`): conversations with made-up memories
  that need his tools and memory (recall, the clock and the days ahead,
  timers set and come due, status reports on made-up news, search or
  report, reading and announcing reports, weather, a failed search), each
  run three times, since the model varies.
- **Conversations** (`app/conversations.py`): whole conversations, each run
  three times. One evening has everything in it (small talk, Red Dwarf,
  every tool, a timer and a report coming due mid-talk). The others are
  ordinary talk after an evening of status reports, a cheery chat, errands,
  old times, and talk about a cat who died. The code checks the tools. Each
  judge reads each transcript whole, against a checklist: he answers what's
  said, drags nothing in, has no running jokes, stays in character, is kind
  when you're upset, gets the facts right, and keeps track. A point fails
  when both judges fail it, each quoting his own words (checked to be his).
  One judge alone flags it, for reading. The judges answer in JSON, so every
  point gets a verdict. Scenarios catch bugs already
  found. These catch the ones nobody has found yet, like the Cat tacked on
  to every reply for an evening, which a judge scoring one reply calls in
  character. (On the code before that fix, both judges failed it, 3 runs
  in 3.)
- **Made-up memories** (`app/honesty.py`): asked about things nobody ever
  told him (lunch on Tuesday, Jo's husband's name, what the boiler engineer
  found), against the made-up memories, he must say he doesn't know. Both
  judges read each answer for anything he presents as remembered that
  isn't, and both saying so fails it. Some he was told are asked too, so "I
  don't know" to everything doesn't pass.
- **A long talk** (`app/long_talk.py`): an evening's talk with four facts
  said early, the start folded into his notes as it is while he sleeps,
  then each fact asked about: in his notes, in his answer, and in
  character.
- **A week of folds** (`app/many_folds.py`): seven days of made-up talk
  folded into his notes night after night, checked in code after every
  fold: lasting facts kept, a changed plan rewritten rather than added
  beside the old one, nothing invented.
- **Research reports** (`app/report_check.py`): two everyday questions
  researched the real way (capped at 15 minutes each), then both judges
  score each report for keeping to his notes, answering what was asked and
  reading well, and quote claims his notes don't have.
- **Numbers** (`app/sourced.py`), over every reply above that used a tool:
  a number of 10 or more he said that nothing his tools told him had.
- **His voice** (`app/voice_check.py`): after the scorecard's written, up
  to 120 of his replies said in his own voice and heard back by Whisper,
  word for word: what his voice garbles ("Lasagne" came back "Lazane")
  shows. In a process of its own, as the voice's start-up is where this PC
  crashes (it's tried three times).
- **Memory check**: the side task that decides whether a line asks him to
  remember something (then he must look it up), on labelled lines
  (`MEMORY_LINES` in `app/scenarios.py`), each asked twice. Also the other
  side checks: whether a line asks for a status report, whether a timer
  come due is a job for one of his tools (and which), and whether a reply takes him up
  on an offer. Each answer's odds (LM Studio's logprobs) say how sure the
  model was: the scorecard lists those under 80%, right today but the first
  a new model or a reworded prompt would tip. A model that thinks first is
  always sure, so they tell you nothing there.
- **Status reports**: five status reports, written as his are, and his
  answer to where each of the crew is, each line scored by the judges for
  being that character, in that world, and funny.

Every problem found by talking to him, once fixed, gets a check: a unit
test if the fix is plain code, or a scenario or labelled line here if it
depends on what the model says.

The judges are two fixed models, Gemma and Qwen (`[evals.judges]` in
`holly.toml`), whatever model is being tested, so scores compare between
runs. There are two so that neither one's taste or blind spots decide
alone: Gemma marking Gemma's replies might let its own habits pass. Each
scores everything. A score is their mean, and each one's own is kept. Each
is loaded once every reply is written, one after the other, with a longer
context than Holly's (`judge_context`). Each thinks as hard as `holly.toml`
says over a whole conversation. Gemma doesn't think at all: its thinking
ran to 28,000 tokens of "I'll check one more time", with no verdict.
`--judge MODEL`, given once or more, judges with others instead, and
`--no-judge` skips them. Scorecards from before the second judge (25
November 2026) are Gemma's alone.

The persona replies are the exception. Since 27 November 2026, one judge,
Qwen (`reply_judge`), scores them, calibrated on your own scores.
`--compare` marks the older persona scores "old", because they came from
rules of ours and don't compare. Of four judges tried, it agreed with your
order best, and adding Gemma or a thinking judge pulled it down. Like them
all, it still scores a quip that makes no sense well, so your ear is the
check for that.

**Reply nets.** Holly's reply loop has nets that make up for what a model
gets wrong: an offer of more help not said, a timer counted down unasked, a
promise of research with no research started, the crew tacked on, and so
on (`NETS` in `app/guards.py`). Each is counted as it fires, and the
scorecard lists every one with its count, the ones at nought too. `--raw`
lets the nets count what they would do and do nothing (nor clean his
history), to see the model as it is. Holly keeps the counts from real use
too, per model, in `data/nets.json` (`--forget` and `--reset` leave them):
`python -m app.evals --nets` shows them. A net a new model never trips, in
the evals or in real talk, can come out: fixes go first into what he's given (prompt, history, notes),
then into routing (a tool required when the ask is plain), and only last
into nets on what he says.

A scorecard goes to `runs/`: Markdown with every reply and transcript, and
JSON for `--compare`. `--only` runs one part: `persona`, `scenarios`,
`conversations`, `honesty`, `long`, `folds`, `reports`, `memory`, `status`,
`voice` (the latest scorecard's replies, said and heard), or, on demand
only, `search` (58 labelled questions, where each answer came from; it
spends Brave searches) and `turns` (below). `--session` asks
the persona prompts as one conversation, which shows ruts. Run it with LM
Studio up and no Holly running. A new model also wants its quirks in a
`[models]` table in `holly.toml`, including, for one that can't stop
thinking (LM Studio's reasoning options for it have no "off"), its least
effort and the tokens that thinking takes.

**Turn-taking**, only on demand: `--only turns` scores your own saved
pauses with the turn-ending model and settings as they are, against the
old 400 ms rule. Save them with `python -m app.turn --save data/turn_clips`,
labelled by whether you went on. They're your voice, so they stay out of
git. Run it before
changing the model or its settings.

## Tests

```
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m pytest                 # fast: no GPU, LM Studio or network
.venv\Scripts\python -m pytest -m lmstudio     # Holly on the real model
```

The fast tests stand in for the model, the web and the speakers: dates and
timers, recall, the reports and research, his history, weather places, the
clock, the ship's records, the wake word, the face going away, and his reply
loop with its safety nets. The `lmstudio` ones use the real model: the
scenarios through his real reply loop (`app/scenarios.py`), with made-up
memories in a temporary folder, the memory check's labelled lines, and the
notes he writes. They need LM Studio, and no Holly running. Gemma varies, so
one of them may fail now and then: run it again before taking it for a
regression.

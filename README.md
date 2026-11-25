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
  - `text-embedding-nomic-embed-text-v1.5` (for recall by meaning; optional)

  Holly loads and unloads them himself. If LM Studio's engine crashes, it
  loads the chat model again by itself, with its own defaults (a short
  context, cutting his prompts); Holly notices before his next reply and
  loads it his way. To have LM Studio's own reload right too, set the
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
.venv\Scripts\python -m app.speak             # type to him; he answers out loud
.venv\Scripts\python -m app.chat              # text only, no tools or memory
```

Run one Holly at a time: two won't fit on the GPU.

The first start after installing spends a few minutes compiling the voice;
later starts take about a minute.

Say "Holly" near the start or end of what you say. For 30 s after he last
spoke he's awake and listens without it.

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
- **The ship's records.** Recall searches them too, and a few that fit the
  conversation come back to him now and then.
- **Web search.** Free (DuckDuckGo and others) without setup. For better
  results, put a [Brave Search API](https://brave.com/search/api/) key in
  the Windows user environment variable `BRAVE_API_KEY`.
- **Weather.** Forecasts for any town, from Open-Meteo. No setup.
- **Clock.** The time here or anywhere, and how many days until a day.
- **News.** The BBC's latest headlines, by subject or on anything you name
  (Google News for a story the BBC hasn't got). Ask for a status report and
  he rounds up the main subjects, a sentence each, then an "Additional"
  item: made-up news from the Red Dwarf universe. No setup.
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
  whether you'd like a quick search now or a report. He works on reports
  only while he's asleep, a step at a time, and stops whenever anyone
  speaks. He splits the request into questions, searches the web
  (free, through ddgs) and Wikipedia, reads the likeliest pages, keeps notes
  with their sources, and writes a report from the notes. When it's done he
  says so and asks if you'd like a summary, then whether to open it. The
  report is saved as markdown, with its sources, in `data/research/`. Any
  time later, ask him to sum one up, talk you through it, answer a detail
  from it, or open it; he finds it from your words, and if more than one
  might be meant, he asks which. No setup; limits are under `[research]` in
  `holly.toml`.
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

It puts him through three things, with a silent voice and his memory in a
temporary folder:

- **Persona**: the prompts in `persona/test_prompts.txt`, each scored for
  length, anything a voice can't say, an assistant's manners, questions back,
  the tool it needs, lines recited from his examples, and repetition. A judge
  scores each reply 1-5 for sounding like Holly.
- **Scenarios** (`app/scenarios.py`): conversations with made-up memories
  that need his tools and memory (recall, the clock and the days ahead,
  timers set and come due, status reports on made-up news, search or
  report, reading and announcing reports, weather, a failed search), each
  run three times, since the model varies.
- **Memory check**: the side task that decides whether a line asks him to
  remember something (then he must look it up), on labelled lines
  (`MEMORY_LINES` in `app/scenarios.py`), each asked twice; and the other
  side checks, whether a line asks for a status report and whether a timer
  come due is a job for one of his tools.

Every problem found by talking to him, once fixed, gets a check: a unit
test if the fix is plain code, or a scenario or labelled line here if it
depends on what the model says.

The judge is one fixed model (`[evals]` in `holly.toml`), whatever model is
being tested, so scores compare between runs; it's loaded once every reply
is written. A scorecard goes to `runs/`: Markdown with every reply, and JSON
for `--compare`. `--only persona`, `--only scenarios` or `--only memory`
runs one part; `--session` asks the persona prompts as one conversation,
which shows ruts. Run it with LM Studio up and no Holly running. A new model also wants its
quirks in a `[models]` table in `holly.toml`.

If you have Holly's lines from the show in `data/scripts/holly_exchanges.json`
(not supplied), the judge is shown some of them for reference.

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

# Holly

A voice assistant in the character of Holly, the ship's computer from Red
Dwarf. Speak to him and he answers out loud, in a cloned voice, with an
optional animated face. Everything runs locally on one Windows PC.

He can search the web, give the weather and the time, set timers and
reminders, play music on Spotify, and remember what you've told him across
sessions, as well as everything that happened on Red Dwarf.

## Requirements

- Windows, with an NVIDIA GPU with about 28 GB of VRAM for everything,
  including the face (built on an RTX 5090, 32 GB).
- Python 3.11 or newer.
- [LM Studio](https://lmstudio.ai), with its server on port 1234 and these
  models downloaded:
  - `google/gemma-4-26b-a4b-qat` (the chat model)
  - `text-embedding-nomic-embed-text-v1.5` (for recall by meaning; optional)

  Holly loads and unloads them himself.
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
| `--no-search` | no web search, weather or clock |
| `--no-wake` | answer everything, without his name (`app.listen`) |
| `--forget` | start afresh, setting his memory of you aside |
| `--reset` | start afresh, deleting his memory of you |
| `--model KEY` | another LM Studio chat model |

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
- **Clock.** The time here or anywhere.
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

- **Research.** "Holly, can you research heat pumps for an old house?" He
  works on it only while he's asleep, a step at a time, and stops whenever
  anyone speaks. He splits the request into questions, searches the web
  (free, through ddgs) and Wikipedia, reads the likeliest pages, keeps notes
  with their sources, and writes a report from the notes. When it's done he
  says so and asks if you'd like a summary, then whether to open it. The
  report is saved as markdown, with its sources, in `data/research/`. Any
  time later, ask him to sum one up, talk you through it, answer a detail
  from it, or open it; he finds it from your words, and if more than one
  might be meant, he asks which. No setup; limits are under `[research]` in
  `holly.toml`.
- **His face.** Ditto animates the still from his voice, in a window that
  scales to any size (F11 or a double-click for full screen). It dozes off
  when he does.

## Configuration

`holly.toml` holds every setting, with comments: models, microphone and
speakers, wake words, memory budgets, search providers and more. Holly's
character is in `persona/`: `system_prompt.md` and example exchanges in
`examples.toml`. Each tool's rules for when and how he uses it sit beside its
code in `app/`, as a `.md` with the same name (`search.md` for `search.py`).

# Holly

A local, voice-in, voice-out Holly, the ship's computer. The plan, the stages
and the milestones are in the Holly Voice Pipeline Plan doc.

Everything runs on this machine. LM Studio serves the LLM on port 1234.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -e .
```

Install PyTorch first, from PyTorch's own index. The RTX 5090 needs a build
for CUDA 12.8 or newer. The default build from PyPI doesn't support it, and
then everything silently falls back to the CPU.

## Milestone 1: text Holly

Talk to Holly at the command line:

```
.venv\Scripts\python -m app.chat
```

`/reset` clears the conversation and `/quit` leaves. After each reply, the
chat shows the time to the first token, and it warns if the reply holds
anything a TTS voice couldn't say.

Score the persona against the 20 test prompts in `persona/test_prompts.txt`:

```
.venv\Scripts\python -m app.persona_eval
.venv\Scripts\python -m app.persona_eval --model qwen/qwen3.8-27b --model google/gemma-4-31b-qat --runs 3
```

Reports go to `runs/`. Read them for character as well as for the pass count.

## Milestone 2: Holly speaks

Type to Holly, and he answers out loud in a stock voice:

```
.venv\Scripts\python -m app.speak
.venv\Scripts\python -m app.speak --voice dan
```

The reply is cut into sentences as the LLM writes it. Each sentence goes to
Orpheus, a TTS model running in LM Studio, and the audio plays as it's
decoded. After each reply the app shows these times, counted from pressing
Enter:

- first token: the LLM's first word of the reply
- first sentence: the first chunk handed to TTS
- first audio: the first decoded audio
- first sound: that audio leaving the speakers
- first voice: the first audible speech, which is the number that matters

Underruns are moments when playback ran out of audio, which you'd hear as
stutters.

Audio plays through WASAPI, because Windows' default MME path adds about
100 ms. Orpheus opens every utterance with about 170 ms of silence, so that
silence is trimmed from the first sentence of each reply.

## Milestone 3: the voice loop

Speak to Holly, and he answers out loud:

```
.venv\Scripts\python -m app.listen
```

Silero VAD hears you start and stop. You've finished once you've been quiet
for 400 ms (`end_silence_ms` in `holly.toml`; raise it if Holly cuts in
while you pause). faster-whisper then transcribes what you said, and the
milestone 2 pipeline answers. Timings count from when you stopped speaking.

He answers only when you say his name among your first or last three
words ("Holly, what's...", "Hey Holly", "...tomorrow, Holly?"), or carry on
within 8 s of his last reply; anything else shows as "(not for Holly)" and
is ignored. The name is found in Whisper's transcript, which is made
anyway, so it costs nothing, and Whisper is told to expect it (it heard
"You alright, Holly?" as "You alright, ho?"). The hint doesn't make Whisper
hear "Holly" in silence or noise. `--no-wake` (or an empty `wake_words`
under `[listen]`) makes him answer everything.

It's half-duplex for now: the mic is ignored while Holly talks and for
0.3 s after, so he doesn't hear himself. Interrupting him is milestone 5.
Use a headset anyway.

The microphone is the WASAPI default input, with all its channels mixed. Set
`input_device` and `input_channels` under `[listen]` to choose.

Run one Holly at a time. The full loop uses about 25 GB of the 5090's
32 GB, and Windows takes a few more for the desktop. If VRAM runs out,
Windows quietly moves GPU memory into system RAM and everything crawls, so
the app warns at startup when less than 1 GB is free.

The first run on a new machine (or after a driver update) takes about 20 s
longer while CUDA compiles Whisper's kernels for the GPU.

## Milestone 4: Holly's own voice

Holly speaks with a clone of Norman Lovett's voice, made by VoxCPM2 from a
20 s reference clip. VoxCPM2 won a blind test of five open models (see
`voice/harness/README.md`). It runs in the app's own process, compiled with
`torch.compile` (needs `triton-windows`). The first start after an install
spends a few minutes compiling; later starts take about 30 s.

```
.venv\Scripts\python -m pip install "kaldifst==1.9.0" voxcpm "triton-windows<3.8"
.venv\Scripts\python -m app.listen                     # Holly's voice
.venv\Scripts\python -m app.listen --engine orpheus    # the stock voice
```

`kaldifst` is pinned because 1.9.1 has no Windows wheel and fails to build.
The reference clip and its transcript come from `voice/prep.py export`, and
`[tts.voxcpm]` in `holly.toml` points at them.

The chat model and VoxCPM share the GPU, which decides both the timing and
the choice of chat model:

| Chat model | Speaking | End of speech to Holly's voice | Stutters (5 replies) | VRAM free |
|---|---|---|---|---|
| **Gemma 4 26B-A4B** (default: 8K context, 2048-token batches) | **streaming** | **median 1.16 s (0.92-1.30)** | 4 | 5.0 GB |
| Gemma 4 26B-A4B (4K, 256-token batches) | streaming | median 1.74 s (1.22-1.88) | 0 | 10.6 GB |
| Gemma 4 26B-A4B (4K, 256-token batches) | after the full reply | median 2.03 s (1.31-2.10) | 0 | 10.6 GB |
| Gemma 4 31B | after the full reply | median 2.7 s (2.0-3.4) | 2 | under 1 GB |
| Gemma 4 31B | streaming | median 3.7 s | 581 | under 1 GB |

- **Streaming** (`wait_for_reply = false`) speaks each sentence as soon as it's
  written. While the chat model is still writing, VoxCPM slows down, because
  the two processes share the GPU. With 31B it fell from about 26 to 3
  generation steps a second and stuttered. 26B-A4B, a mixture-of-experts
  model with 4B parameters active per word, writes about 3 times faster, so
  the overlap is short and streaming wins.
- **VRAM.** Gemma 31B left under 1 GB free, and in live use the desktop's
  own usage grew enough to push it into system RAM, giving replies of 5-10 s.
  26B-A4B needs about 4 GB less. Holly unloads any other model left in LM
  Studio when he starts, then **always reloads his chat model fresh**. LM
  Studio splits a model between GPU and CPU at load time, by the VRAM free
  then, and doesn't report the split. An instance loaded while another
  model held the GPU read prompts 6 times slower (640 against 3,700 tokens a
  second), with settings that looked identical. Holly warns at startup if
  under 1.5 GB is free.
- **History and LM Studio's cache.** LM Studio reuses its cache only while
  the start of the prompt is unchanged. Holly used to drop the oldest
  exchange every turn once his history was full, which changed that start
  every turn, so the whole history was reread each time (2.6 s per turn in a
  long live session). History now runs to 48 exchanges and is then cut to 32
  in one go, so a full reread comes once every 16 exchanges. Over a 60-turn
  test, the first word came in 110-230 ms on every turn.
- **Character.** Both Gemmas passed 39 of 40 persona prompts. 26B-A4B is a
  little chattier and more whimsical; 31B is slightly drier. To switch back,
  set `model = "google/gemma-4-31b-qat"` and `wait_for_reply = true`.

## Phase 2: Holly's face

Ditto animates a still of Holly's screen face from his voice, live, in a
window. Setup, its own venv and the speed-ups are in `face/README.md`.

```
.venv\Scripts\python -m app.listen --face
```

`--face` (or `enabled = true` under `[face]` in `holly.toml`) starts
`face/server.py` in `face/.venv`. The face window also plays Holly's voice,
so the two come from one timeline and can't drift apart; between replies
the face blinks and his head barely moves. After 30 s with nothing said, the
face holds still to spare the GPU, and wakes when anyone speaks. Its log is `face/data/server.log`.

First live test (7 replies): no face frame late, one stutter, and Holly's
voice starting 1.9-2.6 s after you stop, against about 1.2 s without the
face. Two costs: the audio is queued 0.6 s ahead (`lead_ms`) so Ditto can
draw each frame in time, and VoxCPM runs about half speed while Ditto shares
the GPU. VRAM free with everything loaded: 3.3 GB.

The face server starts only once the chat model has loaded. Started
earlier, its VRAM made LM Studio put part of the model in system RAM
(see above), slowing every reply, most often on a restart.

## Looking things up

Holly searches the web for questions about the real world: news, sport,
people, numbers, dates. He has a weather tool for forecasts. Small talk and
jokes don't trigger either.

```
.venv\Scripts\python -m pip install ddgs
.venv\Scripts\python -m app.listen               # searching is on by default
.venv\Scripts\python -m app.listen --no-search
```

- **How.** Gemma gets a `search` tool, and `persona/search_prompt.md` tells
  him to use it rather than answer from his senile memory. The same Gemma
  then answers the question from the results in two or three sentences
  (`app/search.py`), and only that answer goes back into Holly's
  conversation. Raw results would fill his 8K context. Using the same model
  saves loading another, and LM Studio restores the conversation's cache
  after the side request (first word in ~150 ms, against ~80 fully cached).
- **A second look.** That summary step also judges the results, out of
  Holly's sight: it answers, reads the page of a promising result (ddgs
  `extract`, menus stripped, 6,000 characters), or runs a better search, up
  to 3 searches and reads (`max_rounds`), starting none after 5 s
  (`max_lookup_s`). Offered "no answer" as an equal choice, it always took
  it, so that's only allowed on the last round. Reading a page got a sunset
  time the snippets lacked. Pages only load from public addresses, never
  this machine or network. Gemma's thinking for the judgement took 3-11 s
  instead of ~0.3 s and answered no better, so it's off
  (`summary_reasoning`).
- **The wait.** When his reply starts with a search, he says a stock line
  ("Hang on. I'll look it up.", `fillers` under `[search]`), rendered in his
  voice at startup, so his voice starts ~0.3 s after you stop (~1.2 s with
  the face). The answer follows once the search (0.8-1.3 s, sometimes 4 s)
  and the summary (~0.3 s) are done.
- **Facts stay facts.** Given a bare answer, he hedged the very fact he'd
  found ("Ninety-seven. Or perhaps ninety-eight"), so the result tells him
  the facts are right, and the prompt bans "or perhaps" on them. When the
  search fails or the results don't hold the answer, he's told to say he
  couldn't find out; told only that it failed, he guessed, wrongly.
- **He knows the date.** It goes at the end of his system prompt. Without
  it he searched for the 2024 championship.
- **Search engines.** `ddgs` (free, no key) asks Yahoo, Brave and
  DuckDuckGo, then every engine it has if those fail. The engines answer
  bots with CAPTCHAs now and then, and always while a VPN was on, so turn
  any VPN off. For something dependable, use the Brave Search API
  (`provider = "brave"`, key in `BRAVE_API_KEY`; about 1,000 searches a
  month free, then metered).
- **Weather.** Search can't do it: the results are links to weather sites,
  not forecasts. So a `weather` tool (`app/weather.py`) gets the place and
  the week's forecast from Open-Meteo, which is free with no key (~0.7 s).
  Its place lookup knows towns, villages and parks, but not counties or
  most beaches, so a miss tells him to try again with the nearest town:
  Kennack Sands became Helston, Cornwall became Truro, the Lake District
  Keswick. Told he could also just say he couldn't find out, he always did.
  Without a county or country, places in `home_country` (GB) come first.
  It includes sunrise and sunset, so those come from it too, and the local
  time there: asked the time in New York, he otherwise worked it out
  himself, half an hour out.
- **The time.** A `clock` tool (`app/clock.py`) reads the PC's clock, which
  Windows keeps to internet time: the time here, in UTC, or in any IANA
  time zone. It's instant, so no stock line plays; the tool's round trip
  adds ~200 ms before his first word. With only the tool's description he
  used it for Tokyo but guessed the time here, wrongly, so his rules say to
  call it whenever he's asked the time. Given the right time he still
  hedged it ("11:42 AM. Or 11:42 PM"), so the result says it's exact. The
  time can't go in the system prompt, which would change every minute and
  break LM Studio's cache.
- **Limits.** Prices are still hard: they're rarely in snippets or on the
  pages the free engines return. Those engines also fail outright on some
  searches; the Brave API is the fix. When nothing is found he's told not
  to give any time, number or name: told only "don't guess", he once made
  up a sunset time.
- **Searches stay in his history.** With only his replies kept, his history
  showed questions answered straight off, and after a few exchanges he
  stopped searching and said "I've searched the internet" instead, even when
  asked to search. Kept, he searches every time and remembers what he found.
  History is trimmed by size as well as by exchanges, since searches make
  exchanges longer.

## Voice dataset

`voice/prep.py` turns recordings of Holly into a dataset for voice cloning.
It runs in its own venv: see `voice/requirements.txt` for setup.

```
voice\.venv\Scripts\python voice\prep.py fetch URL       # or put a file at voice/data/raw/ID/source.*
voice\.venv\Scripts\python voice\prep.py separate ID     # Demucs: voice off music and laughter
voice\.venv\Scripts\python voice\prep.py segment ID      # cut into lines, transcribe, group by speaker
voice\.venv\Scripts\python voice\prep.py trim ID --status pending   # cut lines down to their words
voice\.venv\Scripts\python voice\prep.py review ID       # listen; keep, reject, mark reference, fix text
voice\.venv\Scripts\python voice\prep.py export          # all sources -> voice/data/holly/
```

Demucs leaves audience laughter in the vocal stem, because laughter is human
voice. The VAD hears it as speech, so lines run on into the laugh after a
punchline. `trim` uses Whisper's word timings to cut each line from just
before its first word to just after its last.

`export` writes 24 kHz clips loudness-matched to -20 dB, `metadata.csv`
(`file|text`), and `reference.wav` and `reference.txt`, built from the lines
marked as reference, up to 20 s. Everything under `voice/data/` is
git-ignored. None of it is to be published.

## Tuning Holly

- `persona/system_prompt.md` holds the character and the rules for speech.
- `persona/examples.toml` holds example exchanges. They steer tone more than
  the prompt does, so rewrite them in your own words.
- `holly.toml` sets the model, sampling and history length.

## LM Studio notes

- Use `127.0.0.1`, not `localhost`. On Windows, `localhost` tries IPv6 first
  and adds about 2 seconds to every request.
- The app loads its model with an 8K context. LM Studio's default context for
  these models is over 100K tokens, and the KV cache for that spills out of
  VRAM and slows everything down.
- Thinking is off (`reasoning_effort = "none"`). With thinking on, a reply
  takes 0.5 to 6 seconds longer to start.

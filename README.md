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

Type to Holly, and he answers out loud:

```
.venv\Scripts\python -m app.speak
```

The reply is cut into sentences as the LLM writes it. Each sentence goes to
the voice (milestone 4), and the audio plays as it's made. This milestone
began with Orpheus, a stock voice running in LM Studio; it was removed once
the clone had replaced it. After each reply the app shows these times,
counted from pressing Enter:

- first token: the LLM's first word of the reply
- first sentence: the first chunk handed to TTS
- first audio: the first decoded audio
- first sound: that audio leaving the speakers
- first voice: the first audible speech, which is the number that matters

Underruns are moments when playback ran out of audio, which you'd hear as
stutters.

Audio plays through WASAPI, because Windows' default MME path adds about
100 ms. A voice may open a sentence with silence, so that silence is
trimmed from the first sentence of each reply.

If a reply fails part-way (LM Studio restarted, a request timed out), Holly
says he's lost his train of thought, the turn isn't kept in his history, and
he carries on; a tool that fails tells him he couldn't find out.

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
while he's awake, within 30 s of his last reply (`awake_s`, also when his
face sleeps); anything else shows as "(not for Holly)" and is ignored. The name is found in Whisper's transcript, which is made
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
.venv\Scripts\python -m app.listen
```

`kaldifst` is pinned because 1.9.1 has no Windows wheel and fails to build.
The reference clip and its transcript come from `voice/prep.py export`, and
`[tts.voxcpm]` in `holly.toml` points at them.

The chat model and VoxCPM share the GPU, which decides both the timing and
the choice of chat model:

| Chat model | Speaking | End of speech to Holly's voice | Stutters (5 replies) | VRAM free |
|---|---|---|---|---|
| **Gemma 4 26B-A4B** (8K context, 2048-token batches; now 30K) | **streaming** | **median 1.16 s (0.92-1.30)** | 4 | 5.0 GB |
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
  long live session). History is now cut in large pieces, rarely, so the
  start stays put. Over a 60-turn test at 8K, the first word came in
  110-230 ms on every turn.
- **A long memory.** Gemma is loaded with a 30,208-token context (LM Studio
  rounds 30,000 up to that; ~29.8 GB of VRAM in use with the face up). The
  persona and tools take ~1K tokens, leaving room for ~23K of history.
  Past ~20K (`history_soft_tokens`), once Holly is asleep (`awake_s` after
  he last spoke) and the room has been quiet for 5 s, the same Gemma folds the older exchanges into a few lines
  of notes (`app/memory.py`), which go at the end of his system prompt, and
  keeps the newest ~8K word for word. LM Studio then rereads the shorter
  prompt, still while he sleeps, so his next reply starts cached. On a
  16.8K-token test conversation, writing the notes took 1.5-2.7 s and the
  reread 0.6 s, and afterwards he still knew the crew member's name, the
  cat, the holiday dates, a reminder and a football score from the start
  of it. If anyone speaks while the notes are being written (even a cough,
  or talk not meant for Holly), the job is dropped and the history left as
  it was; the reply doesn't wait for it, and the notes are tried again after
  the next quiet spell. Typed (`app.speak`), they're written while you type.
  Past ~26K (`history_hard_tokens`) without a sleep, the oldest is dropped
  at once, without notes. After each reply the console shows the context in
  use (LM Studio's own count) and how far the history is from being folded.
- **From one session to the next.** History and notes are saved to
  `data/memory.json` (git-ignored) after every reply, written to a temporary
  file and renamed over the old one, so a crash can't leave half a file. At
  startup, all but the newest ~1,500 tokens (`history_resume_tokens`) of the
  last session go into his notes while the voice loads (~1-3 s), and his
  system prompt says when he last talked with you. `--forget` sets the file
  aside as `memory.forgotten-<date>-<time>.json` and starts afresh (a file
  that won't read, after a slip in a hand edit, is set aside the same way); `/reset` (typed) wipes
  it; `memory_file = ""` keeps nothing. In a test across three days he
  answered "Tuesday 24 November at 10:00" to a boiler service booked for
  "next Tuesday" on the Wednesday before, said that Jo, coming "on Saturday",
  was visiting today, and knew they'd left a chess game unfinished.
- **What the notes hold.** Up to 1,500 tokens (`memory_tokens`, ~5% of the
  context) of plain lines in two parts: "About the crew:" (people, pets,
  work, running jokes) and "Lately:" (visits, appointments, plans and what
  he found out, each with its real date). Requests for poems and stories are
  left out: a "topics" line grew to 1,700 tokens and pushed out a visit.
- **Writing the notes.** Each exchange in the transcript carries its date
  and time, and day words in your lines ("tomorrow", "next Tuesday", "on
  Saturday") get their dates worked out in code, in brackets, to be copied:
  Gemma's own sums put "next Tuesday", said on Wednesday 18 November, on the
  25th, even with a calendar to hand. His earlier notes go after the
  transcript, nearest where Gemma writes: before it, the second fold kept
  only what was in the 8K-token transcript and lost the rest. Temperature 0:
  at 0.3, one run in two left out the cat. The token count is characters /
  4, which ran about 10% high (16.8K counted, 15.1K real), so the limits are
  safe.
- **Reading is cheap.** 26B-A4B rereads 16K tokens of history in a few
  hundred ms: a reply's first word came ~100-150 ms later than with the
  history cached.
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
the face blinks and his head barely moves. 30 s after he last spoke, the
same moment he stops listening without his name, he nods off: his face
settles into the still with his eyes closed, and pauses to spare the GPU.
He opens his eyes when anyone speaks. The window can be any size: the face
scales to fit, and F11 or a double-click goes full screen (Escape leaves).
Its log is `face/data/server.log`.

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
  conversation. Raw results would soon fill his context. Using the same model
  saves loading another, and LM Studio restores the conversation's cache
  after the side request (first word in ~150 ms, against ~80 fully cached).
- **A second look.** That summary step also judges the results, out of
  Holly's sight: it answers, reads the page of a promising result (fetched with
  ddgs's HTTP client, menus stripped, 6,000 characters), or runs a better search, up
  to 3 searches and reads (`max_rounds`), starting none after 5 s
  (`max_lookup_s`). Offered "no answer" as an equal choice, it always took
  it, so that's only allowed on the last round. Reading a page got a sunset
  time the snippets lacked. Pages only load from public addresses, never
  this machine or network, and each redirect is checked the same way
  before it's followed. Gemma's thinking for the judgement took 3-11 s
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
- **Search engines.** `provider` is a list, tried in order:
  `"brave_context,ddgs"` (not `brave` between them: one service and one
  key, so if one fails so will the other). The Brave ones need a key in
  `BRAVE_API_KEY` (read from your saved Windows user variables too, so no
  restart after setting it); Brave's $5 monthly credit covers ~1,000
  searches. `brave_context` is Brave's LLM Context API: passages from each
  result's page instead of a snippet (2,048 tokens in all,
  `context_tokens`). `brave` is its web search. `ddgs` is free, with no key:
  it asks Yahoo, Brave and DuckDuckGo, which answer bots with CAPTCHAs now
  and then, and always while a VPN was on.

  | 16 varied questions | Answered | Median | Slowest |
  |---|---|---|---|
  | `brave_context` | 16 | 1.66 s | 2.0 s |
  | `brave` | 15 | 1.60 s | 2.7 s |

  Context found this week's cinema listings, which web search couldn't,
  and never needed a second search. It said the only cricket results it had
  were from August and September, where web search passed an August Test
  off as live. On 8 other questions the free engines answered 5 (median
  3.6 s) and Brave 8. Brave's Answers service, which writes the answer
  itself, was ruled out: ~4.5 s, paid per answer, and no "no answer"
  signal.
- **Weather.** Search can't do it: the results are links to weather sites,
  not forecasts. So a `weather` tool (`app/weather.py`) gets the place and
  the week's forecast from Open-Meteo, which is free with no key (~0.7 s).
  Its place lookup knows towns, villages and parks, but not counties or
  most beaches, so a miss tells him to try again with the nearest town:
  a beach became the town beside it, a county its main town, the Lake District
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
- **Limits.** More page text gives the judge more to misread: once,
  context took a BBC2 repeat for a new series of Red Dwarf. When nothing is
  found he's told not to give any time, number or name: told only "don't
  guess", he once made up a sunset time.
- **Searches stay in his history.** With only his replies kept, his history
  showed questions answered straight off, and after a few exchanges he
  stopped searching and said "I've searched the internet" instead, even when
  asked to search. Kept, he searches every time and remembers what he found.
  History is budgeted in tokens, not exchanges, since searches make
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
- `holly.toml` sets the model, sampling, context and history budgets.

## LM Studio notes

- Use `127.0.0.1`, not `localhost`. On Windows, `localhost` tries IPv6 first
  and adds about 2 seconds to every request.
- The app loads its model with a 30K context. LM Studio's default context for
  these models is over 100K tokens, and the KV cache for that spills out of
  VRAM and slows everything down.
- Thinking is off (`reasoning_effort = "none"`). With thinking on, a reply
  takes 0.5 to 6 seconds longer to start.

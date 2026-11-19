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

Type to Holly, and she answers out loud in a stock voice:

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

Speak to Holly, and she answers out loud:

```
.venv\Scripts\python -m app.listen
```

Silero VAD hears you start and stop. You've finished once you've been quiet
for 400 ms (`end_silence_ms` in `holly.toml`; raise it if Holly cuts in
while you pause). faster-whisper then transcribes what you said, and the
milestone 2 pipeline answers. Timings count from when you stopped speaking.

It's half-duplex for now: the mic is ignored while Holly talks and for
0.3 s after, so she doesn't hear herself. Interrupting her is milestone 5.
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

Holly speaks with a clone of Norman Lovell's voice, made by VoxCPM2 from a
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

Two things decide the timing:

- **Holly speaks only once Gemma has finished the reply** (`wait_for_reply`).
  Run alongside Gemma, VoxCPM drops from about 26 to 3 generation steps a
  second, because the two processes share the GPU and its many small steps
  lose to Gemma's big ones. Taking turns, Holly's voice starts 2.0-3.4 s after
  you stop speaking (median 2.7 s), with almost no stutters. Most of that time
  is Gemma writing the reply, so a faster chat model is the next lever.
- **VRAM is nearly full.** The desktop (2.2 GB), Gemma (20.0 GB), Whisper
  (1.1 GB) and VoxCPM (5.5 GB) leave about 3 GB. Gemma is loaded with one
  parallel slot and 256-token batches (`[llm]` in `holly.toml`); LM Studio's
  defaults cost 2.5 GB more, which was enough to tip it over.

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

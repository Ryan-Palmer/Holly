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

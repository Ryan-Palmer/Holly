# TTS evaluation harness

Compares voice-cloning TTS models for Holly's voice, by ear and by latency.
Every candidate gets the same reference clip
(`voice/data/holly/reference.wav`, from `voice/prep.py export`) and the same
test sentences (`sentences.txt`).

```
.venv\Scripts\python voice\harness\harness.py list               what's installed
.venv\Scripts\python voice\harness\harness.py run [NAME ...]     synthesize with each (all installed by default)
.venv\Scripts\python voice\harness\harness.py score              rate each sample blind, 1-5
.venv\Scripts\python voice\harness\harness.py report             reveal the scores and timings
```

`run` unloads LM Studio's models first, so timings aren't skewed by VRAM
running out (`--keep-lm` to skip). Results go to `voice/data/harness/RUN/`.
Samples are scored blind, for **similarity** (does it sound like Holly?)
and **naturalness**. While scoring, `r` replays the reference.

## Adding a candidate

1. Create its venv at `voice/harness/envs/NAME` and install a CUDA 12.8 PyTorch
   in it first:
   `envs\NAME\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
2. Install the model's package. If it pins an old torch, install it with
   `--no-deps` and add its other dependencies by hand.
3. Write `adapters/NAME_adapter.py`, subclassing `Adapter` from `adapters/common.py`.
   The `_adapter` suffix stops it shadowing the package it imports.
4. Add it to `candidates.toml`.

## Installing the candidates

VoxCPM2 is the voice Holly uses, in the app's own venv. The other
candidates aren't installed. To compare one, build its environment as below;
its weights download on the first run.

| Candidate | Install |
|---|---|
| voxcpm2 | `pip install "kaldifst==1.9.0" voxcpm` (1.9.1 has no Windows wheel) |
| chatterbox, chatterbox-turbo | `pip install --no-deps chatterbox-tts==0.1.7`, then its dependencies except torch, torchaudio and gradio |
| qwen3-tts | `pip install qwen-tts` (skip flash-attn on Windows) |
| moss-tts-realtime | `git clone https://github.com/OpenMOSS/MOSS-TTS envs/moss-src`, then `pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e "envs/moss-src[torch-runtime]"` |

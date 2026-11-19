# TTS evaluation harness

Picks Holly's cloned voice (milestone 4) by ear and by latency. Every
candidate gets the same reference clip (`voice/data/holly/reference.wav`, made
by `voice/prep.py export`) and the same test sentences (`sentences.txt`),
which are written in Holly's style but aren't lines from the show.

```
.venv\Scripts\python voice\harness\harness.py list               what's installed
.venv\Scripts\python voice\harness\harness.py run [NAME ...]     synthesize with each (all installed by default)
.venv\Scripts\python voice\harness\harness.py score              rate each sample blind, 1-5, twice
.venv\Scripts\python voice\harness\harness.py report             reveal scores and timings per candidate
```

`run` unloads LM Studio's models first, because if VRAM spills into system
RAM the timings mean nothing (`--keep-lm` to skip). Each adapter warms its
model up once before timing. Results go to `voice/data/harness/RUN/`, and the
outputs are copied to `blind/` under random ids, with the key kept aside
until `report`.

The scores are **similarity** (does it sound like Norman's Holly?) and
**naturalness** (a person talking, without glitches). While scoring, `r`
replays the real Holly reference.

## Results

**2026-11-19.** Reference: 19.7 s, from 2 lines of the Non-Sequiturs
compilation. Scored blind, 10 samples each, 1 to 5.

| Candidate | Similarity | Naturalness | First audio (median) | RTF (median) | Peak VRAM | Streams |
|---|---|---|---|---|---|---|
| voxcpm2 | 3.80 | 4.00 | 299 ms | 0.94 | 6.1 GB | yes |
| moss-tts-realtime | 3.40 | 3.20 | 413 ms | 1.50 | 11.5 GB | yes |
| chatterbox | 3.00 | 3.90 | 1962 ms | 0.57 | 3.6 GB | no |
| qwen3-tts | 3.00 | 3.20 | 5407 ms | 1.72 | 4.6 GB | no |
| chatterbox-turbo | 1.60 | 1.60 | 1172 ms | 0.33 | 3.2 GB | no |

**VoxCPM2 is the choice for milestone 4.** It leads on both sound and latency.
Its RTF of 0.94 is only just faster than real time. It ran without
`torch.compile`, because Triton wasn't installed, so that is the first thing to
try for speed.

MOSS-TTS-Realtime was timed uncompiled. Compiled (with `triton-windows`
3.5.1), its first sentence took 33 s of compiling, and the next ran at about
0.9 RTF with first audio in 289 ms. The run was stopped there, by mistake. Even
compiled, it scored lower than VoxCPM2 on naturalness and needs nearly twice
the VRAM.

## Adding a candidate

1. Create its venv at `voice/harness/envs/NAME`. Install the CUDA 12.8 PyTorch
   **first**, since the 5090 needs it and older pins won't run:
   `envs\NAME\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
2. Install the model's package. If it pins an old torch, install it with
   `--no-deps` and add its other dependencies by hand.
3. Write `adapters/NAME_adapter.py`, subclassing `Adapter` from `common.py`.
   The file name needs the `_adapter` suffix, so it can't shadow the package
   it imports.
4. Add it to `candidates.toml`.

## Install notes (checked 2026-11-19)

| Candidate | Install | Notes |
|---|---|---|
| chatterbox, chatterbox-turbo | `pip install --no-deps chatterbox-tts==0.1.7`, then its dependencies except torch, torchaudio and gradio | Pins torch 2.6, which can't drive a 5090. Archived by Resemble in Aug 2026. MIT. Watermarked output. |
| qwen3-tts | `pip install qwen-tts` | No pin. Skip flash-attn on Windows. Apache-2.0. |
| voxcpm2 | `pip install "kaldifst==1.9.0" voxcpm` | kaldifst 1.9.1 has no Windows wheel and fails to build; 1.9.0 has one. Apache-2.0, but check the LICENSE. |

| moss-tts-realtime | `git clone https://github.com/OpenMOSS/MOSS-TTS envs/moss-src`, then `pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e "envs/moss-src[torch-runtime]"` | Pins torch 2.9.1+cu128. Without flash-attention it `torch.compile`s, which needs `triton-windows<3.6`; the adapter runs it uncompiled. Its own audio loader needs FFmpeg, so the adapter reads the WAV itself. Apache-2.0. |

Not yet added:
- **IndexTTS-2.5**: needs Python 3.11 and uv. Under the bilibili licence, so not free for commercial use.
- **CosyVoice 3**: repo pins torch 2.3.1 and is Linux-leaning.
- **Fish S2-Pro**: Linux or WSL only, and its weights are non-commercial.

# Holly's face (phase 2)

[Ditto](https://github.com/antgroup/ditto-talkinghead) (Apache-2.0)
animates a still of Holly from his voice. It uses Ditto's PyTorch weights,
not its TensorRT engines, which were built before the RTX 5090.

Ditto's code, its venv, the weights and everything under `face/data/` are
git-ignored. Nothing made with Norman Lovett's likeness is to be published.

## Setup

```
git clone --depth 1 https://github.com/antgroup/ditto-talkinghead face/ditto-src
git -C face/ditto-src apply ../ditto-windows.patch
py -3.12 -m venv face/.venv
face\.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
face\.venv\Scripts\python -m pip install onnxruntime-gpu librosa tqdm filetype imageio imageio-ffmpeg opencv-python-headless scikit-image mediapipe colored "numpy>=2" einops huggingface_hub soundfile av
face\.venv\Scripts\python -c "from huggingface_hub import snapshot_download as d; d('digital-avatar/ditto-talkinghead', local_dir='face/checkpoints', allow_patterns=['ditto_pytorch/**', 'ditto_cfg/*'])"
```

The weights come to 2.3 GB, most of it the HuBERT speech model.
`ditto-windows.patch` swaps a Cython blending step, which needs a C compiler
on Windows, for numpy. `try_ditto.py` uses the ffmpeg that imageio-ffmpeg
ships, as none is on the PATH.

## Try it

```
face\.venv\Scripts\python face\try_ditto.py --image face/data/holly_colour.jpg --audio voice/data/holly/reference.wav
face\.venv\Scripts\python face\try_ditto.py --image face/data/holly_colour.jpg --audio voice/data/holly/reference.wav --online --warmup
face\.venv\Scripts\python face\try_ditto.py --image face/data/holly_colour.jpg --audio voice/data/holly/reference.wav --online --fast
```

Clips go to `face/data/out/`. `--online` is Ditto's streaming mode, which
is what live use would need: it takes audio in 0.2 s chunks. `--fast` adds
the speed-ups below. `holly_colour.jpg` is a colour still of Holly's
screen face, 1820x1024.

## Live, with Holly

`app.listen --face` starts `server.py` (see the main README). The server:

- runs one continuous timeline in 0.2 s blocks (Holly's voice, or silence),
  feeding each block both to the speakers and to Ditto, so frame k always
  shows the audio 0.04 k s in, and the window picks the frame for whatever
  is leaving the speakers;
- keeps the speakers' queue `--lead` seconds ahead (0.6 s), giving Ditto
  time to draw each frame;
- runs Ditto's motion model 5 frames at a time (`--overlap 75`) rather than
  the 10 of Ditto's own streaming config (70). With 10, the motion trailed
  the audio by about 0.7 s and frames came out barely in time even with a
  0.8 s lead; with 5 they're ready about 0.36 s early.

- rests the lips at the still's own shape when Holly is silent. Ditto's
  motion model otherwise keeps the mouth moving slightly in silence, even on
  digital zeros. It uses Ditto's per-frame `vad_alpha` control, with
  `ctrl_vad` replaced by a lips-only version: Ditto's own builds a lips mask
  but never applies it, which would freeze blinks too. The lips keep moving
  for 0.24 s after speech and start 0.12 s before it, easing over 3 frames.
- calms the head when Holly is silent. Ditto otherwise sways it between
  replies as much as while he talks. A per-frame `pose_alpha` control added
  to Ditto's motion step (`hold_pose_when_told`) blends pitch, yaw, roll and
  position from the photo's (0) to Ditto's (1): 1 while he speaks,
  `idle_head` under `[face]` in `holly.toml` (0.2) while he's silent. The head keeps
  moving for 1 s after speech, so it doesn't stop between sentences, and
  eases in and out over about 0.5 s.
- sleeps 30 s after Holly last spoke (`awake_s` under `[listen]`, the same
  span in which he needs no wake word). Drawing 25 frames a second costs as
  much GPU idle as talking: ~40% of the 5090 and ~200 W, against ~0% and
  ~50 W asleep. First he nods off over ~1 s: the head returns to the
  photo's pose, and a per-frame "doze" control (`face_controls`) blends his
  expression to the photo's with the eyes shut. With the pose alone he froze
  on whatever expression the animation had reached. The eyes-shut offsets
  (`EYES_CLOSED`) are Ditto's own deepest blink: run on silence, the frame
  whose eye keypoints (11, 13, 15, 16, 18) moved furthest from the photo's
  showed the eyes fully closed. Then Ditto pauses and the window holds that
  frame. Holly's app wakes it the moment its VAD hears anyone start
  speaking, a second or more before he could answer; Ditto needs ~0.6 s,
  and his eyes open over ~0.4 s. Speech wakes it for 10 s only, so talk
  that isn't for Holly doesn't keep it awake; his reply does. Ditto numbers
  frames, and looks up their controls, by its own count, which falls behind
  the timeline while it sleeps, so each chunk sent is tagged with the frames
  skipped before it. Frames drawn before sleeping still play out: the feeder
  runs the lead ahead of the speakers.
- keeps HuBERT, Ditto's audio-feature model, on the CPU with 4 threads
  that sleep between chunks (`quiet_onnxruntime`). Ditto asks onnxruntime
  for the GPU, but onnxruntime-gpu 1.32 is built for CUDA 13 and this
  venv's CUDA is torch's 12.8, so it fell back to the CPU with a thread per
  physical core, all spinning between chunks: 80% of a 32-thread CPU, even
  while Holly was silent. Now ~6% for the whole face server, with no late
  frames. On the CPU it costs no VRAM or GPU time.

  On the GPU (tried 2026-11-21, with onnxruntime's CUDA 13 libraries from
  `pip install onnxruntime-gpu[cuda,cudnn]`), HuBERT takes 8 ms a chunk
  against 61, but 1.5 GB of VRAM: its weights are 1.4 GB. It didn't let the
  lead shrink, since Ditto's motion model, not HuBERT, sets it. Four
  replies with the face at each setting:

  | HuBERT | Lead | VRAM free | Voice starts | Late frames |
  |---|---|---|---|---|
  | CPU | 600 ms | 3.5 GB | 1.82 s | 1 |
  | GPU | 600 ms | 1.9 GB | 1.64 s | 80 |
  | GPU | 450 ms | 2.1 GB | 1.56 s | 442 |
  | CPU | 450 ms | 3.4 GB | 1.44 s | 741 |
  | GPU | 350 ms | 1.9 GB | 1.50 s | 1,064 |
  | CPU | 350 ms | 3.4 GB | 1.27 s | 1,244 |

  So it stays on the CPU. With the GPU, keep TF32 off: on, its features
  differed from the CPU's by up to 0.07.

Every 10 s the server logs frames shown, late and how far ahead Ditto is.

## Results (2026-11-20)

Driven by Holly's real 20 s reference clip. Streaming feeds the audio in as
fast as Ditto will take it, so frames a second is its headroom.

| Run | Still | First frame | Frames a second | Extra VRAM |
|---|---|---|---|---|
| Offline, 50 steps | 1280x720 video frame | after all the audio | 27.5 | 2.4 GB |
| Streaming, as shipped, warmed up | 1280x720 video frame | 1.25 s | 17.3 | 2.4 GB |
| **Streaming, `--fast`** | 1820x1024 colour still | **0.37 s** | **68** | 2.5 GB |

As shipped, Ditto's stages queue for the GPU one after another, so their
times add up: per frame, motion from audio 24 ms, paste-back into the frame
(CPU) 24 ms at 720p and 68 ms at 1820x1024, decoding the face 15.5 ms,
warping 11 ms, HuBERT 8.5 ms. Mostly overhead, not arithmetic: half
precision made the motion model slower.

`--fast` applies `fast_ditto.py`:

- **Motion model** (an 8-layer transformer, run 20 times per 10-frame chunk
  for 10 diffusion steps with guidance): `torch.compile` plus a hand-recorded
  CUDA graph of the whole sampler.
- **Warp and decoder**: the same, in fp16 as Ditto already runs them, and
  frames no longer go to the CPU and back between them.

  Benchmarked alone with CUDA graphs, the motion model went from 237 to
  24 ms per chunk, the warp from 10.3 to 3.7 ms and the decoder from 14.9 to
  6.2 ms a frame.
- **Paste-back** on the GPU with `grid_sample`. The still never moves, so the
  sampling grid, mask and background are built once.
- All of it runs on one long-lived thread. torch.compile's own CUDA graphs
  (`mode="reduce-overhead"`) failed when the three models took turns, so the
  graphs are recorded by hand (`CudaGraph`).

The first call of each model compiles and records it: about 47 s of warm-up
on first use. These numbers are with Holly's own app closed; with the chat
model and VoxCPM also on the GPU, expect less headroom.

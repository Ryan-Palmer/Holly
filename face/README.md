# Holly's face

[Ditto](https://github.com/antgroup/ditto-talkinghead) (Apache-2.0) animates
a still of Holly's face from his voice, live, in a window. It runs in its own
venv, started by Holly's app with `--face`.

Ditto's code, its venv, the weights and everything under `face/data/` are
git-ignored: nothing made with Holly's likeness is to be published.

## Setup

```
git clone --depth 1 https://github.com/antgroup/ditto-talkinghead face/ditto-src
git -C face/ditto-src apply ../ditto-windows.patch
py -3.12 -m venv face/.venv
face\.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
face\.venv\Scripts\python -m pip install onnxruntime-gpu librosa tqdm filetype imageio imageio-ffmpeg opencv-python-headless scikit-image mediapipe colored "numpy>=2" einops huggingface_hub soundfile av pygame-ce moderngl
face\.venv\Scripts\python -c "from huggingface_hub import snapshot_download as d; d('digital-avatar/ditto-talkinghead', local_dir='face/checkpoints', allow_patterns=['ditto_pytorch/**', 'ditto_cfg/*'])"
```

The weights are 2.3 GB. `ditto-windows.patch` replaces a step that needs a
C compiler on Windows.

Then put a still of Holly's face at `face/data/holly_colour.jpg`: his face
on black, as on the ship's screen (1820x1024 works well).

## Use

```
.venv\Scripts\python -m app.listen --face
```

The face warms up for about a minute the first time. The window plays
Holly's voice too, so the two stay in step. It can be any size: F11 or a
double-click goes full screen, and Escape leaves. When Holly dozes off, so
does his face, until someone speaks. Its log is `face/data/server.log`.

Settings are under `[face]` in `holly.toml`: `enabled` (to have it on
without `--face`), how far the audio runs ahead of the face (`lead_ms`), the
height frames are drawn at, and how much the head moves while he's silent
(`idle_head`).

To render a clip without Holly, from any audio:

```
face\.venv\Scripts\python face\try_ditto.py --image face/data/holly_colour.jpg --audio voice/data/holly/reference.wav --online --fast
```

Clips go to `face/data/out/`.

## Files

- `server.py`: the face window. Plays Holly's audio and Ditto's frames on one clock.
- `fast_ditto.py`: compiles Ditto's models and keeps its frames on the GPU, so it keeps up live.
- `try_ditto.py`: renders a clip, for trying Ditto on its own.
- `ditto-windows.patch`: makes Ditto run on Windows.

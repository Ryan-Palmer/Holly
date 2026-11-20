"""Speeds up Ditto's PyTorch pipeline enough to stream in real time on the 5090.

Ditto as shipped spends most of its time on overheads rather than arithmetic:
its motion model launches thousands of tiny GPU kernels per call, frames go
from GPU to CPU and back between every stage, and the face is pasted back
into the full frame on the CPU. `speed_up(sdk)` changes a StreamSDK so that:

- the motion model, warp and decoder are compiled with torch.compile (kernel
  fusion, no CUDA graphs of its own), in half precision for the warp and
  decoder as Ditto already does, and each is then recorded as a CUDA graph
  and replayed, so a call costs one launch instead of thousands;
- frames stay on the GPU from the warp to the finished picture;
- the paste-back is done on the GPU, with its sampling grid and mask built
  once per source frame (a still never moves, so that's once);
- all of that runs on one long-lived thread. torch.compile's own CUDA graphs
  ("reduce-overhead") belong to the thread that recorded them and broke when
  three models took turns, which is why the graphs are recorded by hand.

The first call of each model compiles and records it, which takes a minute
or so; run a short warm-up clip before anything that's timed.
"""

import queue
import threading

import numpy as np
import torch
import torch.nn.functional as F

CROP = 512  # Ditto renders the face at 512x512, then pastes it into the frame


class CudaGraph:
    """Records fn as a CUDA graph on first call, then replays it.

    Every call must pass tensors of the same shapes. Inputs are copied into
    the graph's own buffers, and the output is copied out, because the next
    replay reuses its memory.
    """

    def __init__(self, fn):
        self.fn = fn
        self.graph = None

    def __call__(self, *inputs):
        if self.graph is None:
            self.inputs = [x.clone() for x in inputs]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):  # compiles, and settles cuDNN's choices, before recording
                    self.fn(*self.inputs)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.fn(*self.inputs)
        for buf, x in zip(self.inputs, inputs):
            buf.copy_(x)
        self.graph.replay()
        return self.output.clone()


class GpuThread:
    """Runs functions on one long-lived thread and waits for the result."""

    def __init__(self):
        self._jobs = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="ditto-gpu").start()

    def _run(self):
        while True:
            fn, args, done = self._jobs.get()
            try:
                with torch.no_grad():
                    done.result = fn(*args)
            except BaseException as e:  # handed back to the caller
                done.error = e
            done.set()

    def call(self, fn, *args):
        done = threading.Event()
        done.result = done.error = None
        self._jobs.put((fn, args, done))
        done.wait()
        if done.error is not None:
            raise done.error
        return done.result


class FastMotion:
    """Ditto's LMDM diffusion sampler (audio to head motion), compiled and graphed."""

    def __init__(self, gpu, lmdm):
        self.gpu, self.lmdm = gpu, lmdm  # Ditto's LMDM nn.Module
        lmdm.model.forward = torch.compile(lmdm.model.forward)
        self.sample = lmdm.ddim_sample
        self.graphs = {}  # one per number of diffusion steps

    def __call__(self, kp_cond, aud_cond, steps):
        return self.gpu.call(self._run, kp_cond, aud_cond, steps)

    def _run(self, kp_cond, aud_cond, steps):
        if steps not in self.graphs:
            self.lmdm.setup(steps)  # allocates its noise outside the graph
            self.graphs[steps] = CudaGraph(lambda kp, aud: self.sample(kp, aud, steps))
        return self.graphs[steps](kp_cond, aud_cond)


class FastWarp:
    def __init__(self, gpu, warp_f3d):
        self.gpu = gpu
        model = torch.compile(warp_f3d.warp_net.model)

        def warp(f_s, x_s, x_d):
            with torch.autocast("cuda", dtype=torch.float16):
                return model(f_s, x_s, x_d)

        self.graph = CudaGraph(warp)
        self._f_s = {}  # source features, uploaded once per source frame

    def __call__(self, f_s, x_s, x_d):
        return self.gpu.call(self._run, f_s, x_s, x_d)

    def _run(self, f_s, x_s, x_d):
        key = id(f_s)
        if key not in self._f_s:
            self._f_s[key] = (f_s, torch.from_numpy(f_s).cuda())  # holds f_s so its id stays unique
        return self.graph(self._f_s[key][1], torch.from_numpy(x_s).cuda(), torch.from_numpy(x_d).cuda())


class FastDecode:
    def __init__(self, gpu, decode_f3d):
        self.gpu = gpu
        model = torch.compile(decode_f3d.decoder.model)

        def decode(f_3d):
            with torch.autocast("cuda", dtype=torch.float16):
                return model(f_3d).float().clamp(0, 1)  # (1, 3, 512, 512), 0..1

        self.graph = CudaGraph(decode)

    def __call__(self, f_3d):
        return self.gpu.call(self.graph, f_3d)


class FastPutBack:
    """Ditto's paste-back (cv2.warpAffine of the face and a soft mask, then a blend), on the GPU."""

    def __init__(self, gpu, putback, height=None):
        self.gpu = gpu
        self.height = height  # scale the finished frame to this height, for display
        mask = putback.mask_ori_float  # (512, 512), 0..1
        self.mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).cuda()[None, None]
        self._frames = {}

    def __call__(self, frame_rgb, render_img, M_c2o):
        return self.gpu.call(self._run, frame_rgb, render_img, M_c2o)

    def _prepare(self, frame_rgb, M_c2o):
        h, w = frame_rgb.shape[:2]
        # For each output pixel, where it comes from in the 512x512 crop (cv2.warpAffine's inverse map).
        inv = np.linalg.inv(np.asarray(M_c2o, dtype=np.float64))
        ys, xs = np.mgrid[0:h, 0:w]
        pts = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3) @ inv.T
        u, v = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
        # Pixel centres in grid_sample's [-1, 1] coordinates (align_corners=False).
        grid = np.stack([(2 * u + 1) / CROP - 1, (2 * v + 1) / CROP - 1], -1).reshape(1, h, w, 2)
        grid = torch.from_numpy(grid.astype(np.float32)).cuda()
        mask = F.grid_sample(self.mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        mask = mask.clamp(0, 1)[0].permute(1, 2, 0)  # (h, w, 1)
        frame = torch.from_numpy(frame_rgb).cuda().float()  # (h, w, 3)
        return frame_rgb, grid, mask, frame * (1 - mask)

    def _run(self, frame_rgb, render_img, M_c2o):
        key = (id(frame_rgb), np.asarray(M_c2o).tobytes())
        if key not in self._frames:
            self._frames[key] = self._prepare(frame_rgb, M_c2o)
        _, grid, mask, background = self._frames[key]
        face = F.grid_sample(render_img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        face = face[0].permute(1, 2, 0) * 255  # (h, w, 3)
        out = face * mask + background
        if self.height and self.height != out.shape[0]:
            width = round(out.shape[1] * self.height / out.shape[0])
            out = F.interpolate(out.permute(2, 0, 1)[None], size=(self.height, width), mode="bilinear",
                                antialias=True, align_corners=False)[0].permute(1, 2, 0)
        return out.clamp(0, 255).to(torch.uint8).cpu().numpy()


def speed_up(sdk, height=None):
    """Swap a StreamSDK's slow stages for the compiled, GPU-resident ones. Call once, before setup().

    `height` scales the finished frames, e.g. to fit a window.
    """
    gpu = GpuThread()
    lmdm = sdk.audio2motion.lmdm  # Ditto's wrapper; its __call__ hands numpy to ddim_sample
    motion = FastMotion(gpu, lmdm.model)

    def ddim_sample(kp_cond, aud_cond, steps):
        return motion(kp_cond, aud_cond, steps)

    lmdm.model.ddim_sample = ddim_sample
    sdk.warp_f3d = FastWarp(gpu, sdk.warp_f3d)
    sdk.decode_f3d = FastDecode(gpu, sdk.decode_f3d)
    sdk.putback = FastPutBack(gpu, sdk.putback, height)
    return sdk

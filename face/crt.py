"""A CRT look for Holly's face window: the picture on the curved glass of an old monitor,
set in its plastic bezel, with scanlines, a vignette, phosphor glow, reflections on the
glass, and a soft bar rolling down now and then, bending the picture as it passes.

All on the GPU, through SDL2's renderer: the frame is drawn through a grid of quads bent
like the glass (Texture.draw_quad); the glow is the frame drawn small into a texture and
stretched back over it (the stretching blurs it); the rest are textures made once per
window size and blended over it. ~1 ms a frame at any window size.
"""

from __future__ import annotations

import numpy as np

GRID = (24, 40)  # quads across and down: rows fine enough for the rolling bar's bend
BLEND, ADD, MOD = 1, 2, 4  # SDL_BLENDMODE_BLEND, _ADD, _MOD (multiply)
OVERLAY_HEIGHT = 1080  # the overlays are made at most this tall, and scaled up: at 4K they took 0.7 s
SETTLE_S = 0.25  # while the window's being resized, the old overlays are stretched; remade after this
BEZEL = 0.065  # the bezel's width, of the window's height
ROLL_S = 1.5  # the rolling bar takes this long to cross the screen


class Crt:
    def __init__(self, renderer, curve: float = 0.07, scanlines: float = 0.22, vignette: float = 0.5,
                 every_s: float = 7.0, glow: float = 12.0, bloom: int = 30, bend: float = 0.006,
                 mask: float = 0.3):
        """`curve`: how far the glass's corners are pulled in (0, flat; 0.1, very curved).
        `scanlines`, `vignette`: how dark (0 to 1). `every_s`: seconds between rolling bars.
        `glow`: how far the glass's black is lifted (of 255). `bloom`: the phosphor glow's
        strength (of 255). `bend`: how far the rolling bar shifts the picture, of its width.
        `mask`: how strongly the phosphor stripes show (0 to 1)."""
        self.renderer, self.curve, self.glow, self.bend = renderer, curve, glow, bend
        self.scanline_dark, self.vignette_dark, self.every_s, self.bloom = scanlines, vignette, every_s, bloom
        self.mask_strength, self.mask, self.mask_size = mask, None, None
        self.box = None  # the box the layout is for
        self.resized = 0.0  # when the box last changed
        self.made = None  # the size the overlays were made at
        self.overlay = self.band = self.glass = self.shine = self.bezel = None
        self.small = self.tiny = None  # the glow's render targets

    # The layout ---------------------------------------------------------------------

    def _layout(self, box: tuple[int, int, int, int]):
        """The glass inside the bezel, and the mesh the frame is drawn through: each point
        where it lies on the glass, the corners pulled in and the edges bowed out."""
        x0, y0, w, h = box
        m = round(BEZEL * h)
        self.glass_box = (x0 + m, y0 + m, max(w - 2 * m, 1), max(h - 2 * m, 1))
        gx0, gy0, gw, gh = self.glass_box
        k = self.curve
        self.u = np.linspace(0, 1, GRID[0] + 1)
        self.v = np.linspace(0, 1, GRID[1] + 1)
        cx, cy = np.meshgrid(self.u * 2 - 1, self.v * 2 - 1)
        self.X = gx0 + (cx * (1 - k * cy * cy) + 1) / 2 * gw
        self.Y = gy0 + (cy * (1 - k * cx * cx) + 1) / 2 * gh
        self.flat = self._quads(self.X)
        self.box = box

    def _quads(self, X: np.ndarray) -> list[tuple]:
        Y, u, v = self.Y, self.u, self.v
        return [((X[j, i], Y[j, i]), (X[j, i + 1], Y[j, i + 1]), (X[j + 1, i + 1], Y[j + 1, i + 1]),
                 (X[j + 1, i], Y[j + 1, i]), (u[i], v[j]), (u[i + 1], v[j]), (u[i + 1], v[j + 1]), (u[i], v[j + 1]))
                for j in range(GRID[1]) for i in range(GRID[0])]

    # The textures ---------------------------------------------------------------------

    def _texture(self, rgba: np.ndarray, mode: int):
        import pygame
        from pygame._sdl2.video import Texture

        h, w = rgba.shape[:2]
        tex = Texture.from_surface(self.renderer, pygame.image.frombuffer(np.ascontiguousarray(rgba), (w, h), "RGBA"))
        tex.blend_mode = mode
        return tex

    def _overlays(self, W: int, H: int):
        """The textures laid over the frame, made `W` by `H` for the whole box and scaled
        to fit: scanlines and a vignette, the glass's glow and reflections, and the bezel."""
        import cv2

        k = self.curve
        m = BEZEL * H
        gw, gh = W - 2 * m, H - 2 * m
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = (xs - W / 2) / (gw / 2), (ys - H / 2) / (gh / 2)  # -1 to 1 across the glass
        # Where the glass is: its curved edges, and rounded corners, softened a pixel.
        ex, ey = np.abs(nx) / (1 - k * ny * ny), np.abs(ny) / (1 - k * nx * nx)
        corner = 0.1
        qx, qy = np.clip(ex - (1 - corner), 0, None), np.clip(ey - (1 - corner * gw / gh), 0, None)
        outside = np.maximum(np.hypot(qx / corner, qy / (corner * gw / gh)), np.maximum(ex, ey) > 1)
        glass = cv2.GaussianBlur((outside <= 1).astype(np.float32), (0, 0), max(H / 900, 0.6))

        # Scanlines and a vignette, on the glass.
        line = max(round(gh / 360), 2)  # about 360 lines down the screen, whatever its size
        dark = np.where((ys % line) >= line / 2, self.scanline_dark, 0.0).astype(np.float32)
        r2 = np.clip((nx * nx * 0.9 + ny * ny) / 1.9, 0, 1)
        dark = 1 - (1 - dark) * (1 - self.vignette_dark * r2 ** 1.6)
        over = np.zeros((H, W, 4), np.uint8)
        over[..., 3] = np.clip(dark * glass * 255, 0, 255).astype(np.uint8)
        self.overlay = self._texture(over, BLEND)

        def added(r, g, b):
            rgba = np.zeros((H, W, 4), np.uint8)
            for i, c in enumerate((r, g, b)):
                rgba[..., i] = np.clip(c, 0, 255).astype(np.uint8)
            rgba[..., 3] = 255
            return self._texture(rgba, ADD)

        # A tube is never quite black: the glass glows faintly, most in the middle.
        lift = self.glow * (1 - 0.6 * r2) * glass
        self.glass = added(lift * 0.9, lift, lift)
        # Reflections: a broad sheen up and to the left, and the glass's rim catching the
        # light along its top and left, as from a window behind and above. (A thin diagonal
        # glint as well looked like a scratch.)
        hx, hy = nx + 0.55, ny + 0.6
        sheen = np.exp(-(hx * hx * 2.2 + hy * hy * 5.5)) * 26
        rim = np.clip(glass - cv2.GaussianBlur(glass, (0, 0), H / 120), 0, 1)  # a thin band just inside the edge
        lit = np.clip(0.5 - 0.35 * nx - 0.65 * ny, 0, 1) ** 1.5  # top and left most
        shine = (sheen + rim * lit * 120) * glass
        self.shine = added(shine, shine, shine * 1.05)

        # The bezel: dark plastic around the glass, lit from above, with its outer corners
        # rounded, a shadow where it meets the glass and a highlight on the lip.
        radius = 0.09 * H
        cxr, cyr = np.clip(np.abs(xs - W / 2) - (W / 2 - radius), 0, None), np.clip(np.abs(ys - H / 2) - (H / 2 - radius), 0, None)
        case = np.clip(radius - np.hypot(cxr, cyr) + 0.5, 0, 1)
        plastic = 30 + 14 * (1 - ys / H)  # lighter at the top
        near = cv2.GaussianBlur(glass, (0, 0), H / 70)  # high close to the glass
        shade = 1 - 0.75 * np.clip(near * 1.6, 0, 1)  # its lip in shadow
        lip = np.clip(cv2.GaussianBlur(glass, (0, 0), H / 300) - glass, 0, 1)  # just outside the glass
        edge = np.clip(case - cv2.GaussianBlur(case, (0, 0), H / 250), 0, 1)  # the outer edge
        tone = plastic * shade + lip * 60 * (0.4 + 0.6 * lit) + edge * 30 * np.clip(1 - ys / H * 1.6, 0, 1)
        bezel = np.zeros((H, W, 4), np.uint8)
        bezel[..., 0] = np.clip(tone * 0.97, 0, 255).astype(np.uint8)
        bezel[..., 1] = np.clip(tone * 0.98, 0, 255).astype(np.uint8)
        bezel[..., 2] = np.clip(tone * 1.02, 0, 255).astype(np.uint8)
        bezel[..., 3] = np.clip(case * (1 - glass) * 255, 0, 255).astype(np.uint8)
        self.bezel = self._texture(bezel, BLEND)

        # The rolling bar: a soft band of light, added.
        t = np.linspace(-1, 1, 64, dtype=np.float32)
        band = np.zeros((64, 1, 4), np.uint8)
        band[..., :3] = ((np.cos(t * np.pi) + 1) / 2 * 18)[:, None, None].astype(np.uint8)  # 18 of 255 at most
        band[..., 3] = 255
        self.band = self._texture(band, ADD)
        self.made = (W, H)

    def _mask(self, gw: int, gh: int):
        """The phosphor stripes: red, green and blue columns, a pixel each, as on an aperture
        grille, multiplied over the picture. Made at the glass's own size, as they must be
        sharp: scaled, they blur to grey."""
        a = self.mask_strength
        stripes = np.full((3, 3), 255 * (1 - a), np.float32)
        np.fill_diagonal(stripes, 255)  # one column each: red, green, blue
        row = np.tile(stripes, ((gw + 2) // 3, 1))[:gw]  # (gw, 3)
        rgba = np.empty((1, gw, 4), np.uint8)
        rgba[0, :, :3] = row.astype(np.uint8)
        rgba[..., 3] = 255
        self.mask = self._texture(rgba, MOD)  # one row, stretched down the glass: columns stay sharp
        self.mask_size = (gw, gh)

    def _glow_targets(self, gw: int, gh: int):
        from pygame._sdl2.video import Texture

        size = lambda f: (max(gw // f, 4), max(gh // f, 4))
        # Shrunk less, the stretching blurs less: a tight bleed round bright parts, not a halo
        # (a twenty-fourth, then stretched back, made an angel's aura).
        if self.small is None or self.small.width != size(3)[0]:
            self.small = Texture(self.renderer, size(3), target=True, scale_quality=1)
            self.tiny = Texture(self.renderer, size(8), target=True, scale_quality=1)
            self.tiny.blend_mode = ADD

    # Drawing --------------------------------------------------------------------------

    def rolling(self, now: float) -> bool:
        """Is the bar on its way down? Then it's worth drawing smoothly, frame or no frame."""
        return now % self.every_s < ROLL_S

    def draw(self, texture, box: tuple[int, int, int, int], now: float):
        """Draw `texture` as the CRT shows it, in `box` (x, y, w, h), at time `now` (s)."""
        import pygame

        x0, y0, w, h = box
        if box != self.box:
            self._layout(box)
            self.resized = now
        scale = min(1.0, OVERLAY_HEIGHT / h)
        size = (max(round(w * scale), 1), max(round(h * scale), 1))
        if self.made is None or (size != self.made and now - self.resized > SETTLE_S):
            self._overlays(*size)
        gx0, gy0, gw, gh = self.glass_box
        renderer = self.renderer
        if self.bloom:
            # Phosphor glow, first, off screen (switching targets part-way through a frame
            # can lose what's drawn, on some backends): the frame drawn small, then smaller.
            self._glow_targets(gw, gh)
            renderer.target = self.small
            renderer.draw_color = (0, 0, 0, 255)
            renderer.clear()
            texture.draw(dstrect=pygame.Rect(0, 0, self.small.width, self.small.height))
            renderer.target = self.tiny
            renderer.clear()
            self.small.draw(dstrect=pygame.Rect(0, 0, self.tiny.width, self.tiny.height))
            renderer.target = None
            renderer.clear()  # the window, black again round the set
        phase = (now % self.every_s) / ROLL_S
        bh = max(gh // 5, 8)
        bar_y = gy0 - bh + phase * (gh + bh)  # the bar's top
        # The frame, through the bent mesh; where the bar is, its rows pushed sideways a
        # little, one way then the other, like a picture losing its hold for a moment.
        if phase < 1 and self.bend:
            d = (self.Y - (bar_y + bh / 2)) / (bh / 2)
            push = np.where(np.abs(d) < 1, np.sin(d * np.pi) * (1 - d * d), 0.0) * self.bend * gw
            quads = self._quads(self.X + push)
        else:
            quads = self.flat
        for quad in quads:
            texture.draw_quad(*quad)
        whole = pygame.Rect(x0, y0, w, h)
        if self.mask_strength:
            if self.mask is None or (self.mask_size != (gw, gh) and now - self.resized > SETTLE_S):
                self._mask(gw, gh)
            self.mask.draw(dstrect=pygame.Rect(gx0, gy0, gw, gh))
        self.glass.draw(dstrect=whole)
        self.overlay.draw(dstrect=whole)
        if phase < 1:
            self.band.draw(dstrect=pygame.Rect(gx0, round(bar_y), gw, bh))
        self.shine.draw(dstrect=whole)
        self.bezel.draw(dstrect=whole)
        # The glow, stretched back over the glass and a little past it, onto the bezel's
        # lip (the stretching blurs it), and added.
        if self.bloom:
            self.tiny.alpha = self.bloom
            spill_x, spill_y = round(gw * 0.01), round(gh * 0.01)
            self.tiny.draw(dstrect=pygame.Rect(gx0 - spill_x, gy0 - spill_y, gw + 2 * spill_x, gh + 2 * spill_y))

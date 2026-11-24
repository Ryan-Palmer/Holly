"""Holly's face on screen, through OpenGL: the frame scaled to fit the window on black, or,
with `crt`, on an old monitor. One fragment shader draws it all, per pixel, on the GPU:

- the picture on curved glass, set in a plastic bezel, lit from above, its lip in shadow;
- scanlines drawn as an electron beam would, wider where the picture is bright (after
  Timothy Lottes' CRT shader, which he put in the public domain);
- a phosphor mask: red, green and blue stripes, a screen pixel each, as on an aperture grille;
- halation, a tight glow round bright parts, from the frame's mipmaps (the GPU's own
  shrunken copies, made as each frame arrives);
- a vignette, reflections (a soft sheen, the glass's rim catching the light), and a soft
  bar rolling down every few seconds, bending the picture as it passes.

SDL's 2D renderer, used before, can't run a shader: its CRT look was layers blended over
the picture, and its glow, a blur, read as a halo.
"""

from __future__ import annotations

import numpy as np

ROLL_S = 4.0  # the rolling bar takes this long to drift down the screen (in 1.5 s, it raced)

VERTEX = """
#version 330
void main() {  // one triangle covering the window
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FRAGMENT = """
#version 330
uniform sampler2D frame;
uniform vec2 res;        // the window, in pixels
uniform vec4 tv;         // the box the frame fits, x y w h, from the top left
uniform int crt;
uniform float bezel;     // the bezel's width, of the box's height
uniform float curve;     // how far the glass's corners are pulled in
uniform float lines;     // scanlines down the glass
uniform float mask;      // how strongly the phosphor stripes show
uniform float halation;  // the glow round bright parts
uniform float vignette;
uniform float lift;      // the glass's black, lifted
uniform float bar;       // the rolling bar, 0 to 1 on its way down, or -1
uniform float bend;      // how far it pushes the picture sideways, of the glass's width
out vec4 color;

float box_dist(vec2 p, vec2 half_size, float r) {  // < 0 inside a rounded box, in its units
    vec2 q = abs(p) - half_size + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

vec3 at(vec2 uv, float lod) { return textureLod(frame, uv, lod).rgb; }

vec3 beam(float dist, vec3 c) {  // a scanline's light at `dist` lines from its centre
    vec3 width = 0.30 + 0.32 * sqrt(c);  // brighter, wider
    return exp(-dist * dist / (width * width) * 1.3);
}

void main() {
    vec2 p = vec2(gl_FragCoord.x, res.y - gl_FragCoord.y);
    if (crt == 0) {
        vec2 uv = (p - tv.xy) / tv.zw;
        bool inside = all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
        color = vec4(inside ? at(uv, 0.0) : vec3(0.0), 1.0);
        return;
    }
    float m = bezel * tv.w;
    vec4 glass = vec4(tv.xy + m, tv.zw - 2.0 * m);
    vec2 half_glass = glass.zw * 0.5;
    vec2 c = (p - glass.xy - half_glass) / half_glass;              // -1 to 1 across the glass
    vec2 s = vec2(c.x / (1.0 - curve * c.y * c.y), c.y / (1.0 - curve * c.x * c.x));  // the picture there
    float dg = box_dist(s * half_glass, half_glass, min(half_glass.x, half_glass.y) * 0.16);  // px from the glass's edge
    float band = 0.4;                                                // the bar's height, in the same units
    float bar_y = -1.0 - band * 0.5 + bar * (2.0 + band);            // its centre
    if (bar >= 0.0) {  // the picture, not the glass
        float d = (s.y - bar_y) / (band * 0.5);
        if (abs(d) < 1.0) s.x -= sin(d * 3.14159) * (1.0 - d * d) * bend * 2.0;
    }
    float r2 = clamp((c.x * c.x * 0.9 + c.y * c.y) / 1.9, 0.0, 1.0);
    float lit = pow(clamp(0.5 - 0.35 * c.x - 0.65 * c.y, 0.0, 1.0), 1.5);  // light from above, to the left
    vec2 uv = clamp(s * 0.5 + 0.5, 0.0, 1.0);

    // The glass.
    float y = uv.y * lines - 0.5;
    float f = fract(y);
    float x_lod = 1.2;  // a little horizontal softness, as a beam has
    vec3 a = mix(at(vec2(uv.x, (floor(y) + 0.5) / lines), 0.0), at(vec2(uv.x, (floor(y) + 0.5) / lines), x_lod), 0.5);
    vec3 b = mix(at(vec2(uv.x, (floor(y) + 1.5) / lines), 0.0), at(vec2(uv.x, (floor(y) + 1.5) / lines), x_lod), 0.5);
    vec3 pic = a * beam(f, a) + b * beam(1.0 - f, b);
    pic *= 1.25;  // the beam's gaps darken it: brought back up
    pic += (at(uv, 4.0) * 0.7 + at(uv, 6.0) * 0.3) * halation;
    int k = int(mod(gl_FragCoord.x, 3.0));
    vec3 stripes = vec3(1.0 - mask);
    stripes[k] = 1.0 + mask * 0.6;
    pic *= stripes;
    pic *= 1.0 - vignette * pow(r2, 1.6);
    pic += vec3(0.9, 1.0, 1.0) * lift * (1.0 - 0.6 * r2);
    if (bar >= 0.0) {
        float d = (c.y - bar_y) / (band * 0.5);
        if (abs(d) < 1.0) pic += 0.07 * (cos(d * 3.14159) + 1.0) * 0.5;
    }
    vec2 h = c + vec2(0.55, 0.6);
    pic += exp(-(h.x * h.x * 2.2 + h.y * h.y * 5.5)) * 0.10;               // a soft sheen
    pic += exp(dg / (tv.w * 0.010)) * lit * 0.45;                          // the rim catching the light

    // The bezel.
    vec2 rel = (p - tv.xy) / tv.zw;
    float tone = (30.0 + 14.0 * (1.0 - rel.y)) / 255.0;                   // lighter at the top
    tone *= 1.0 - 0.75 * exp(-max(dg, 0.0) / (tv.w * 0.02));              // its lip in shadow
    tone += exp(-pow((dg - tv.w * 0.004) / (tv.w * 0.003), 2.0)) * (0.4 + 0.6 * lit) * 40.0 / 255.0;  // the lip's edge
    float dc = box_dist(p - tv.xy - tv.zw * 0.5, tv.zw * 0.5, tv.w * 0.09);
    tone += exp(-pow((dc + tv.w * 0.006) / (tv.w * 0.005), 2.0)) * clamp(1.0 - rel.y * 1.6, 0.0, 1.0) * 16.0 / 255.0;
    vec3 case_col = vec3(0.97, 0.98, 1.02) * tone;
    case_col += at(uv, 6.0) * halation * 0.8 * exp(-max(dg, 0.0) / (tv.w * 0.008));  // the glow, spilling onto it

    vec3 col = mix(pic, case_col, smoothstep(-1.0, 1.0, dg));
    col *= 1.0 - smoothstep(-1.0, 1.0, dc);                                // black past the case
    color = vec4(col, 1.0);
}
"""


class Screen:
    """The face window's picture. `frame_size`: Ditto's frames, (w, h)."""

    def __init__(self, ctx, frame_size: tuple[int, int], crt: bool = False, curve: float = 0.07,
                 lines: float = 360, mask: float = 0.22, halation: float = 0.12, vignette: float = 0.5,
                 lift: float = 0.045, bezel: float = 0.065, every_s: float = 12.0, bend: float = 0.006):
        import moderngl

        self.ctx, self.frame_size, self.crt, self.every_s = ctx, frame_size, crt, every_s
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.vao = ctx.vertex_array(self.program, [])
        self.texture = ctx.texture(frame_size, 3)
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = self.texture.repeat_y = False
        self.texture.build_mipmaps()
        settings = dict(crt=int(crt), curve=curve, lines=lines, mask=mask, halation=halation,
                        vignette=vignette, lift=lift, bezel=bezel, bend=bend)
        for name, value in settings.items():
            self.program[name].value = value

    def update(self, img: np.ndarray):
        """A new frame: (h, w, 3) RGB bytes, top row first."""
        self.texture.write(np.ascontiguousarray(img, dtype=np.uint8))
        if self.crt:
            self.texture.build_mipmaps()  # for the glow and the beam's softness

    def rolling(self, now: float) -> bool:
        """Is the bar on its way down? Then it's worth drawing smoothly, frame or no frame."""
        return self.crt and now % self.every_s < ROLL_S

    def draw(self, size: tuple[int, int], now: float):
        """Draw the frame into a window `size` (w, h), as large as fits, centred."""
        w, h = size
        fw, fh = self.frame_size
        scale = min(w / fw, h / fh)
        dw, dh = fw * scale, fh * scale
        self.ctx.viewport = (0, 0, w, h)
        self.program["res"].value = (w, h)
        self.program["tv"].value = ((w - dw) / 2, (h - dh) / 2, dw, dh)
        phase = (now % self.every_s) / ROLL_S
        self.program["bar"].value = phase if self.crt and phase < 1 else -1.0
        self.texture.use(0)
        self.vao.render(vertices=3)

"""GPU tissue point cloud (moderngl) for GenomeShell 3D view.

Renders proc / soma / grown points into an offscreen RGB buffer matching
``project_3d_batch`` orthographic math, then returns an HxWx3 uint8 ndarray
(top-left origin) for ``pygame.surfarray.blit_array``.
Falls back gracefully when moderngl / GL context is unavailable.

WARNING: GenomeShell must construct this renderer only when ``FLYWIRE_GL=1``
(and not when ``FLYWIRE_NO_GL`` is set). Auto-enabling GL previously coincided
with a full-black pygame window including the left menu; never create a GL
context unless the user opted in.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

VERT = """
#version 330
uniform float u_ax, u_ay;
uniform float u_cx, u_cy, u_scale;
uniform float u_vw, u_vh;
uniform float u_bri;
uniform float u_point_size;
in vec3 in_pos;
in vec3 in_rgb;
in float in_act;
out vec3 v_rgb;
void main() {
    float cos_y = cos(u_ay);
    float sin_y = sin(u_ay);
    float x_rot = in_pos.x * cos_y - in_pos.z * sin_y;
    float z_rot = in_pos.x * sin_y + in_pos.z * cos_y;
    float cos_x = cos(u_ax);
    float sin_x = sin(u_ax);
    float y_rot = in_pos.y * cos_x - z_rot * sin_x;
    float sx = u_cx + x_rot * u_scale;
    float sy = u_cy - y_rot * u_scale;
    float ndc_x = (sx / u_vw) * 2.0 - 1.0;
    float ndc_y = 1.0 - (sy / u_vh) * 2.0;
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    float a = clamp(in_act, 0.0, 1.0);
    vec3 c = in_rgb * u_bri;
    if (a > 0.05) {
        c = c * (0.72 + 0.28 * a) + vec3(a * 0.35);
    }
    v_rgb = clamp(c, 0.0, 1.0);
    gl_PointSize = u_point_size + step(0.08, a) * 1.5;
}
"""

FRAG = """
#version 330
in vec3 v_rgb;
out vec4 f_color;
void main() {
    // Square points — skip PointCoord discard. Tiny gl_PointSize + hard circle
    // tests made tissue invisible on some drivers (black view).
    f_color = vec4(v_rgb, 1.0);
}
"""


class GlTissueRenderer:
    """Offscreen point-cloud renderer for CNS Brain mass (somata / processes)."""

    def __init__(self) -> None:
        self.ok = False
        self._ctx: Any = None
        self._prog: Any = None
        self._fbo: Any = None
        self._tex: Any = None
        self._w = 0
        self._h = 0
        self._layers: dict[str, dict[str, Any]] = {}
        self._geom_key: tuple | None = None
        try:
            import moderngl

            # Standalone context (works with pygame software display).
            self._ctx = moderngl.create_standalone_context()
            self._ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            # Also set fixed size as fallback if vertex gl_PointSize ignored.
            self._ctx.point_size = 3.0
            self._prog = self._ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
            self.ok = True
        except Exception:
            self.ok = False
            self._ctx = None

    def close(self) -> None:
        self._layers.clear()
        self._fbo = None
        self._tex = None
        self._prog = None
        if self._ctx is not None:
            try:
                self._ctx.release()
            except Exception:
                pass
        self._ctx = None
        self.ok = False

    def _ensure_fbo(self, w: int, h: int) -> None:
        if self._tex is not None and self._w == w and self._h == h:
            return
        assert self._ctx is not None
        self._tex = self._ctx.texture((w, h), 3)
        self._fbo = self._ctx.framebuffer(color_attachments=[self._tex])
        self._w, self._h = w, h

    def invalidate(self) -> None:
        self._layers.clear()
        self._geom_key = None

    def set_layer(
        self,
        name: str,
        xyz: np.ndarray,
        rgb: np.ndarray,
        *,
        act: np.ndarray | None = None,
        point_size: float = 1.5,
        brightness: float = 1.0,
    ) -> None:
        if not self.ok or self._ctx is None or self._prog is None:
            return
        if xyz is None or len(xyz) == 0:
            self._layers.pop(name, None)
            return
        pos = np.ascontiguousarray(xyz, dtype=np.float32)
        col = np.ascontiguousarray(rgb, dtype=np.float32)
        if col.max() > 1.5:
            col = col / 255.0
        if act is None or len(act) != len(pos):
            a = np.zeros((len(pos),), dtype=np.float32)
        else:
            a = np.ascontiguousarray(act, dtype=np.float32)
        vbo_pos = self._ctx.buffer(pos.tobytes())
        vbo_rgb = self._ctx.buffer(col.tobytes())
        vbo_act = self._ctx.buffer(a.tobytes())
        vao = self._ctx.vertex_array(
            self._prog,
            [
                (vbo_pos, "3f", "in_pos"),
                (vbo_rgb, "3f", "in_rgb"),
                (vbo_act, "1f", "in_act"),
            ],
        )
        self._layers[name] = {
            "vao": vao,
            "n": len(pos),
            "point_size": float(point_size),
            "brightness": float(brightness),
            "vbo_act": vbo_act,
            "act_len": len(pos),
        }

    def update_activity(self, name: str, act: np.ndarray | None) -> None:
        layer = self._layers.get(name)
        if not layer or act is None:
            return
        a = np.ascontiguousarray(act, dtype=np.float32)
        if len(a) != int(layer["act_len"]):
            return
        layer["vbo_act"].write(a.tobytes())

    def render(
        self,
        *,
        view_w: int,
        view_h: int,
        view_lim_y: int,
        view_angle_x: float,
        view_angle_y: float,
        pan_x: float,
        pan_y: float,
        scale_3d: float,
        bg_rgb: tuple[int, int, int] = (18, 20, 28),
        layer_order: tuple[str, ...] = ("proc", "soma", "grown"),
    ) -> np.ndarray | None:
        """Return HxWx3 uint8 RGB (top-left origin, pygame-like)."""
        if not self.ok or self._ctx is None or self._prog is None:
            return None
        w, h = int(view_w), int(view_h)
        if w < 2 or h < 2:
            return None
        self._ensure_fbo(w, h)
        assert self._fbo is not None and self._tex is not None
        self._fbo.use()
        self._ctx.viewport = (0, 0, w, h)
        self._ctx.clear(
            bg_rgb[0] / 255.0,
            bg_rgb[1] / 255.0,
            bg_rgb[2] / 255.0,
            1.0,
        )
        cx = view_w * 0.5 + pan_x
        cy = (view_lim_y) * 0.5 + pan_y
        self._prog["u_ax"].value = float(view_angle_x)
        self._prog["u_ay"].value = float(view_angle_y)
        self._prog["u_cx"].value = float(cx)
        self._prog["u_cy"].value = float(cy)
        self._prog["u_scale"].value = float(scale_3d)
        self._prog["u_vw"].value = float(view_w)
        self._prog["u_vh"].value = float(view_h)
        # Ensure vertex gl_PointSize is honored (else drivers may emit 1px / 0px).
        self._ctx.enable(self._ctx.PROGRAM_POINT_SIZE)
        for name in layer_order:
            layer = self._layers.get(name)
            if not layer:
                continue
            ps = max(1.0, float(layer["point_size"]))
            self._prog["u_bri"].value = float(layer["brightness"])
            self._prog["u_point_size"].value = ps
            self._ctx.point_size = ps
            layer["vao"].render(mode=self._ctx.POINTS, vertices=int(layer["n"]))
        # Moderngl read is bottom-up.
        raw = self._fbo.read(components=3, alignment=1)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(np.flipud(img))


def try_create_renderer() -> GlTissueRenderer | None:
    r = GlTissueRenderer()
    return r if r.ok else None

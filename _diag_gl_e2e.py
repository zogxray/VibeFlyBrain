#!/usr/bin/env python3
"""End-to-end GenomeShell GL tissue draw diagnostic (DISPLAY=:1)."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _surf_stats(surf: pygame.Surface, label: str) -> dict:
    arr = pygame.surfarray.array3d(surf)  # (w,h,3)
    flat = arr.reshape(-1, 3).astype(np.float64)
    bg = np.array([12, 12, 16], dtype=np.float64)
    near_bg = np.all(np.abs(flat - bg) <= 2, axis=1)
    near_black = np.all(flat <= 2, axis=1)
    out = {
        "label": label,
        "size": surf.get_size(),
        "bitsize": surf.get_bitsize(),
        "masks": surf.get_masks(),
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
        "mean": flat.mean(axis=0).tolist(),
        "non_near_bg": int((~near_bg).sum()),
        "non_near_black": int((~near_black).sum()),
        "frac_non_bg": float((~near_bg).mean()),
    }
    print(
        f"[{label}] size={out['size']} bits={out['bitsize']} masks={out['masks']}\n"
        f"  min={out['min']} max={out['max']} mean={[round(x,2) for x in out['mean']]}\n"
        f"  non_near_bg={out['non_near_bg']:,} ({out['frac_non_bg']*100:.2f}%) "
        f"non_near_black={out['non_near_black']:,}"
    )
    return out


def main() -> None:
    from fly_model import load_fly
    from genome_shell import BG_VIEW, NAV_PANEL_H, GenomeShell

    print(f"DISPLAY={os.environ.get('DISPLAY')} cwd={ROOT}")
    print(f"PIDs note: this diagnostic is pid={os.getpid()}")

    fly_path = Path("/home/vpavlov/work/G2/FlyWire/data/x.fly")
    if not fly_path.is_file():
        fly_path = ROOT / "data" / "flies" / "x.fly"
    print(f"fly={fly_path} exists={fly_path.is_file()}")

    # Temporarily log exceptions inside _try_draw_tissue_gl
    import genome_shell as gs

    _orig = GenomeShell._try_draw_tissue_gl

    def _logged(self, cx, cy, view_lim_y, act):
        if not self.use_gl_tissue or self._gl_tissue is None or not self._gl_tissue.ok:
            print("[try_gl] early False: use_gl / gl missing")
            return False
        if self._view_surf is None:
            print("[try_gl] early False: no view_surf")
            return False
        try:
            ok_geom = self._ensure_gl_tissue_geom()
            print(f"[try_gl] ensure_geom={ok_geom}")
            if not ok_geom:
                return False
            layers = getattr(self._gl_tissue, "_layers", {})
            for name in ("proc", "soma", "grown"):
                layer = layers.get(name)
                n = 0 if not layer else int(layer.get("n", 0))
                print(f"[try_gl] layer {name}: n={n:,}")
            soma_idx = getattr(self, "_gl_soma_idx", None)
            if (
                act is not None
                and soma_idx is not None
                and len(soma_idx)
                and len(act) == self.n_shown
            ):
                self._gl_tissue.update_activity("soma", act[soma_idx])
            img = self._gl_tissue.render(
                view_w=int(self.view_w),
                view_h=int(self.view_h),
                view_lim_y=int(view_lim_y),
                view_angle_x=float(self.view_angle_x),
                view_angle_y=float(self.view_angle_y),
                pan_x=float(self.pan_x),
                pan_y=float(self.pan_y),
                scale_3d=float(self.scale_3d),
                bg_rgb=tuple(int(c) for c in BG_VIEW),
            )
            if img is None:
                print("[try_gl] render returned None")
                return False
            print(
                f"[try_gl] img shape={img.shape} dtype={img.dtype} "
                f"min={img.min()} max={img.max()} mean={img.mean():.3f}"
            )
            bg = np.array(BG_VIEW, dtype=np.uint8)
            nonbg = int(np.any(img != bg, axis=2).sum())
            print(f"[try_gl] img non_exact_bg={nonbg:,}")
            blit = np.transpose(img, (1, 0, 2))
            print(
                f"[try_gl] blit_array arr shape={blit.shape} "
                f"view_surf bits={self._view_surf.get_bitsize()} "
                f"size={self._view_surf.get_size()}"
            )
            pygame.surfarray.blit_array(self._view_surf, blit)
            _surf_stats(self._view_surf, "view_surf AFTER blit_array")
        except Exception as exc:
            print(f"[try_gl] EXCEPTION: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._gl_geom_key = None
            return False
        if getattr(self, "ground_on", False) and int(view_lim_y) > 1 and int(self.view_w) > 1:
            try:
                self._draw_ground(cx, cy, view_lim_y)
            except Exception as exc:
                print(f"[try_gl] ground EXCEPTION: {exc}")
        return True

    GenomeShell._try_draw_tissue_gl = _logged  # type: ignore[method-assign]

    t0 = __import__("time").perf_counter()
    shell = GenomeShell()
    print(
        f"boot {( __import__('time').perf_counter()-t0)*1000:.0f} ms  "
        f"use_gl_tissue={shell.use_gl_tissue} "
        f"gl_ok={shell._gl_tissue.ok if shell._gl_tissue else None} "
        f"driver={pygame.display.get_driver()} "
        f"screen={shell.screen.get_size()} bits={shell.screen.get_bitsize()}"
    )
    print(
        f"view={shell.view_w}x{shell.view_h} "
        f"view_surf bits={shell._view_surf.get_bitsize() if shell._view_surf else None} "
        f"angles=({shell.view_angle_x:.4f},{shell.view_angle_y:.4f}) "
        f"scale={shell.scale_3d:.3f} pan=({shell.pan_x:.1f},{shell.pan_y:.1f})"
    )
    print(f"io_status={getattr(shell, '_fly_io_status', None)}")
    print(
        f"after_restore n_soma={len(shell.points_xyz):,} "
        f"n_proc={len(shell.proc_xyz):,} "
        f"last_model={getattr(shell, '_last_model', None)}"
    )

    # Ensure x.fly loaded the same way as open / restore
    if fly_path.is_file():
        loaded = load_fly(fly_path)
        shell._adopt_fly(
            loaded,
            f"Loaded {fly_path.name}",
            refit=False,
            model_path=fly_path,
            remember=False,
        )
        print(
            f"after_adopt n_soma={len(shell.points_xyz):,} "
            f"n_proc={len(shell.proc_xyz):,} adult={shell.fly.is_adult()}"
        )

    # Match restored camera already applied by _restore; print them again
    print(
        f"cam angles=({shell.view_angle_x:.4f},{shell.view_angle_y:.4f}) "
        f"scale={shell.scale_3d:.3f} pan=({shell.pan_x:.1f},{shell.pan_y:.1f})"
    )

    cx = shell.view_w * 0.5 + shell.pan_x
    cy = (shell.view_h - NAV_PANEL_H) * 0.5 + shell.pan_y
    view_lim_y = shell.view_h - NAV_PANEL_H
    act = getattr(shell, "soma_activity", None)

    print("\n=== direct _ensure + _try_draw_tissue_gl ===")
    ok = shell._try_draw_tissue_gl(cx, cy, view_lim_y, act)
    print(f"_try_draw_tissue_gl -> {ok}")

    print("\n=== full _draw() once ===")
    # Reset geom so GL path runs again with logging
    shell._gl_geom_key = None
    shell._view_surf.fill((0, 0, 0))
    shell._draw()
    _surf_stats(shell._view_surf, "view_surf AFTER _draw")
    # Sample screen region corresponding to view
    screen_arr = pygame.surfarray.array3d(shell.screen)
    vx, vy = shell.view_x, shell.view_y
    vw, vh = shell.view_w, shell.view_h
    region = screen_arr[vx : vx + vw, vy : vy + vh]
    flat = region.reshape(-1, 3).astype(np.float64)
    bg = np.array(BG_VIEW, dtype=np.float64)
    nonbg = int((~np.all(np.abs(flat - bg) <= 2, axis=1)).sum())
    print(
        f"[screen view region] min={flat.min(axis=0).tolist()} "
        f"max={flat.max(axis=0).tolist()} mean={[round(x,2) for x in flat.mean(axis=0).tolist()]} "
        f"non_near_bg={nonbg:,}"
    )

    # CPU project check: fraction of somata on screen
    from genome_shell import project_3d_batch

    if len(shell.points_xyz):
        sx, sy = project_3d_batch(
            shell.points_xyz,
            shell.view_angle_x,
            shell.view_angle_y,
            cx,
            cy,
            shell.scale_3d,
        )
        m = (sx >= 0) & (sx < shell.view_w) & (sy >= 0) & (sy < view_lim_y)
        print(
            f"[cpu project] n={len(sx):,} onscreen={int(m.sum()):,} "
            f"({100.0*m.mean():.1f}%) sx=[{sx.min():.1f},{sx.max():.1f}] "
            f"sy=[{sy.min():.1f},{sy.max():.1f}]"
        )

    print(
        "\n=== _draw blit order ===\n"
        "1) screen.fill(BG_VIEW)\n"
        "2) panel rect + _draw_control_panel\n"
        "3) _draw_particles -> (GL) blit_array(_view_surf) + overlays\n"
        "4) screen.blit(_view_surf, (view_x, view_y))\n"
        "5) _draw_legend + modal + flip\n"
        "=> no fill AFTER tissue"
    )

    GenomeShell._try_draw_tissue_gl = _orig  # type: ignore[method-assign]
    pygame.quit()
    print("DONE")


if __name__ == "__main__":
    main()

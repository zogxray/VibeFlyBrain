#!/usr/bin/env python3
"""Microbench adult _draw after soma LUT / funnel blit cache patches.

Reports ms for: full _draw, _draw_learn_flash, _draw_device_funnels 1st vs 2nd,
_soma_indices_for_ids on hot set, and sim+draw frame estimate.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ms(fn, n=5, warm=1) -> float:
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def once_ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    import pygame
    from fly_model import load_fly
    from genome_shell import NAV_PANEL_H, GenomeShell

    fly_path = ROOT / "data" / "flies" / "x.fly"
    if not fly_path.exists():
        fly_path = ROOT / "data" / "x.fly"
    if not fly_path.exists():
        raise SystemExit(f"missing x.fly under {ROOT / 'data'}")

    print(f"fly={fly_path}")
    t_boot = time.perf_counter()
    shell = GenomeShell()
    print(
        f"GenomeShell boot {(time.perf_counter() - t_boot) * 1000:.0f} ms  "
        f"driver={pygame.display.get_driver()} size={shell.screen.get_size()}"
    )

    loaded = load_fly(fly_path)
    shell._adopt_fly(loaded, f"Loaded {fly_path.name}", model_path=fly_path)
    fly = shell.fly
    print(
        f"adult={fly.is_adult()} n_soma={len(fly.soma_ids):,} "
        f"contiguous={getattr(fly, '_activity_ids_contiguous', lambda: '?')()}"
    )

    # Adult closed-loop + devices + funnels + high learn amp.
    shell.learn_on = True
    shell.ocelli_on = True
    shell.turret_on = True
    shell.show_ocelli_funnel = True
    shell.show_turret_funnel = True
    if fly.is_adult():
        shell._activity_running = True
        shell.sim_running = True
        for _ in range(6):
            try:
                shell._tick_adult_closed_loop(1.0 / 60.0)
            except Exception as exc:
                print(f"[tick] {exc}")
                break

    # Force learn flash path hot.
    pre = getattr(shell, "_learn_flash_pre", None)
    if pre is None or len(pre) == 0:
        # Synthetic hot set if ticks didn't produce flash pairs.
        ids = fly.soma_ids
        n = min(400, len(ids))
        hot = ids[:n].astype(object) if hasattr(ids, "astype") else list(ids)[:n]
        shell._learn_flash_pre = fly.soma_ids[:n].copy()
        shell._learn_flash_post = fly.soma_ids[1 : n + 1].copy() if n + 1 <= len(ids) else fly.soma_ids[:n].copy()
        shell._learn_flash_n = n
    shell._learn_flash_amp = 0.95
    print(
        f"learn amp={shell._learn_flash_amp:.2f} "
        f"pre={len(getattr(shell, '_learn_flash_pre', []))} "
        f"funnels oc={shell.show_ocelli_funnel} tu={shell.show_turret_funnel}"
    )

    view_lim = shell.view_h - NAV_PANEL_H
    cx = shell.view_w * 0.5 + shell.pan_x
    cy = (shell.view_h - NAV_PANEL_H) * 0.5 + shell.pan_y

    # Warm ground/sky/compass caches (not under test).
    for _ in range(2):
        shell._draw()

    # Hot set for soma LUT bench.
    pre = shell._learn_flash_pre
    post = shell._learn_flash_post
    engaged = set(int(a) for a in pre.tolist()) | set(int(b) for b in post.tolist())
    if len(engaged) > 900:
        engaged = set(list(engaged)[:900])
    hot_ids = list(engaged)

    print("\n=== microbench (ms) ===")
    t_soma = ms(lambda: shell._soma_indices_for_ids(hot_ids), n=20, warm=3)
    print(f"_soma_indices_for_ids (hot n={len(hot_ids)})  {t_soma:7.3f} ms")

    amp_save = float(shell._learn_flash_amp)

    def flash():
        shell._learn_flash_amp = amp_save
        shell._draw_learn_flash(cx, cy, view_lim)
        shell._learn_flash_amp = amp_save

    t_flash = ms(flash, n=5, warm=1)
    print(f"_draw_learn_flash                    {t_flash:7.2f} ms")

    # Invalidate funnel blit cache, then time cold vs warm (camera still).
    shell._device_funnel_blit_key = None
    shell._device_funnel_blit_surf = None
    t_funnel_1 = once_ms(lambda: shell._draw_device_funnels(cx, cy, view_lim))
    t_funnel_2 = once_ms(lambda: shell._draw_device_funnels(cx, cy, view_lim))
    # Extra warm hits for stability.
    t_funnel_hit = ms(
        lambda: shell._draw_device_funnels(cx, cy, view_lim), n=5, warm=0
    )
    print(f"_draw_device_funnels 1st (cold)       {t_funnel_1:7.2f} ms")
    print(f"_draw_device_funnels 2nd (cache)      {t_funnel_2:7.2f} ms")
    print(f"_draw_device_funnels cache avg        {t_funnel_hit:7.2f} ms")
    cached = (
        shell._device_funnel_blit_key is not None
        and shell._device_funnel_blit_surf is not None
    )
    print(f"  blit_cache_present={cached}")

    # Full _draw with learn amp high + funnels on.
    shell._learn_flash_amp = amp_save
    shell.show_ocelli_funnel = True
    shell.show_turret_funnel = True

    def full_draw():
        shell._learn_flash_amp = amp_save
        shell._draw()
        shell._learn_flash_amp = amp_save

    # One cold-ish after funnel invalidate to include funnel rebuild once.
    shell._device_funnel_blit_key = None
    shell._device_funnel_blit_surf = None
    t_full_cold = once_ms(full_draw)
    t_full = ms(full_draw, n=4, warm=1)
    print(f"full _draw (funnels cold once)       {t_full_cold:7.2f} ms")
    print(f"full _draw (amp high, funnels on)    {t_full:7.2f} ms")

    # Sim + draw frame estimate.
    dt = 1.0 / 60.0
    if fly.is_adult():
        t_sim = ms(lambda: shell._tick_adult_closed_loop(dt), n=4, warm=1)
    else:
        t_sim = float("nan")
        print("[warn] not adult — sim skipped")
    t_frame = t_sim + t_full if t_sim == t_sim else t_full
    print(f"sim closed_loop.step                 {t_sim:7.2f} ms")
    print(f"sim+draw frame estimate              {t_frame:7.2f} ms  (~{1000.0 / max(t_frame, 1e-6):.1f} FPS)")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Microbench after hybrid patches: hop weight cache + active frontier,
tissue blit cache on stable camera, ClosedLoop.step learn_every=8.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ms(fn, n=8, warm=2) -> float:
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
    from blueprint import Blueprint
    from cns_runtime import CNSRuntime
    from fly_model import load_fly
    from genome_shell import GenomeShell
    import pygame

    fly_path = ROOT / "data" / "flies" / "x.fly"
    if not fly_path.exists():
        fly_path = ROOT / "data" / "x.fly"
    if not fly_path.exists():
        raise SystemExit(f"missing x.fly under {ROOT / 'data'}")

    print(f"fly={fly_path}")

    # ---- Headless sim path (hop + ClosedLoop) ----
    bp = Blueprint.from_fly_meta(fly_path, yaw_offset=2.2, scope_on=True, hop_clip=True)
    rt = CNSRuntime.open(fly_path, blueprint=bp, keep_gains=False)
    rt.instant_fuse()
    rt.apply_training_scope()
    rt.prepare_activity()
    rt.learn_on = True
    rt.turret_on = True
    rt.ocelli_on = True
    rt.loop_hops = 1
    rt.learn_every = 8
    loop = rt._closed_loop
    loop.trainer = rt.trainer()
    dt = 1.0 / 60.0

    for _ in range(8):
        loop.step(dt)

    fly = rt.fly
    # Warm hop weight cache.
    for _ in range(2):
        fly.propagate_activity(gain=0.9)

    hop_cached = (
        getattr(fly, "_hop_edge_key", None) is not None
        and len(getattr(fly, "_hop_w", ())) > 0
    )
    alive = int((fly.soma_activity > 1e-4).sum()) if len(fly.soma_activity) else 0
    n_edges = len(fly.activity_graph_edges()[0])
    print(
        f"n={len(fly.neurons):,} hop_edges={n_edges:,} alive={alive:,} "
        f"contiguous={fly._activity_ids_contiguous()} "
        f"hop_w_cache={hop_cached} gains={len(fly.contact_gain):,}"
    )
    print(f"learn_every={rt.learn_every} loop_hops={rt.loop_hops}")

    rows: list[tuple[str, float]] = []
    rows.append(("propagate_activity", ms(lambda: fly.propagate_activity(gain=0.9), n=8, warm=2)))
    rows.append(
        (
            "tick_activity",
            ms(lambda: fly.tick_activity(hop_gain=0.85, decay=0.90), n=8, warm=2),
        )
    )

    rt.learn_every = 8
    for _ in range(2):
        loop.step(dt)
    rows.append(("ClosedLoop.step le=8", ms(lambda: loop.step(dt), n=8, warm=2)))

    # ---- GenomeShell draw path (tissue blit) ----
    shell = GenomeShell()
    loaded = load_fly(fly_path)
    shell._adopt_fly(loaded, f"Loaded {fly_path.name}", model_path=fly_path)
    shell.learn_on = True
    shell.ocelli_on = True
    shell.turret_on = True
    # Keep amp low so learn flash doesn't dominate tissue-cache comparison.
    shell._learn_flash_amp = 0.0
    shell.show_ocelli_funnel = False
    shell.show_turret_funnel = False

    if shell.fly.is_adult():
        shell._activity_running = True
        shell.sim_running = True
        for _ in range(4):
            try:
                shell._tick_adult_closed_loop(dt)
            except Exception as exc:
                print(f"[tick] {exc}")
                break
    shell._learn_flash_amp = 0.0

    # Invalidate tissue blit, then cold vs warm with same camera.
    shell._tissue_blit_key = None
    shell._tissue_blit_surf = None
    t_draw_1 = once_ms(shell._draw)
    hit1 = (
        shell._tissue_blit_key is not None and shell._tissue_blit_surf is not None
    )
    t_draw_2 = once_ms(shell._draw)
    hit2 = (
        shell._tissue_blit_key is not None and shell._tissue_blit_surf is not None
    )
    # Stable avg after cache fill.
    t_draw_hit = ms(shell._draw, n=4, warm=0)

    rows.append(("_draw 1st (tissue cold)", t_draw_1))
    rows.append(("_draw 2nd (tissue cache)", t_draw_2))
    rows.append(("_draw tissue-cache avg", t_draw_hit))

    print(
        f"tissue_blit after 1st={hit1} after 2nd={hit2} "
        f"driver={pygame.display.get_driver()} "
        f"size={shell.screen.get_size()}"
    )

    print("\n=== hybrid microbench (ms) ===")
    print(f"{'#':>2}  {'op':40s}  {'ms':>8}")
    for i, (name, val) in enumerate(rows, 1):
        print(f"{i:2d}  {name:40s}  {val:8.2f}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

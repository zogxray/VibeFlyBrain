#!/usr/bin/env python3
"""Profile GenomeShell adult _draw path (SDL dummy) + pure-Python microbenches.

Does not modify product code. Run:
  SDL_VIDEODRIVER=dummy python _profile_draw_path.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ms(fn, n=5, warm=1) -> float:
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def main() -> None:
    from genome_shell import (
        DRAW_GROWN_PTS_MAX,
        DRAW_PROC_MAX,
        DRAW_SOMA_MAX,
        GenomeShell,
        project_3d_batch,
        _density_take,
        _stratified_density_take,
    )

    fly_path = ROOT / "data" / "flies" / "x.fly"
    if not fly_path.exists():
        flies = sorted((ROOT / "data" / "flies").glob("*.fly"))
        if not flies:
            raise SystemExit("no .fly files")
        fly_path = flies[0]
    print(f"fly={fly_path}")

    t_boot = time.perf_counter()
    shell = GenomeShell()
    print(f"boot_shell {(time.perf_counter() - t_boot)*1000:.0f} ms  "
          f"size={shell.screen.get_size()} driver={pygame.display.get_driver()}")

    # Load adult fly like UI File→Open.
    from fly_model import load_fly

    t0 = time.perf_counter()
    loaded = load_fly(fly_path)
    shell._adopt_fly(loaded, f"Loaded {fly_path.name}", model_path=fly_path)
    print(f"load_fly {(time.perf_counter() - t0)*1000:.0f} ms")

    fly = shell.fly
    n_soma = len(getattr(shell, "points_xyz", [])) or len(fly.soma_ids)
    n_proc = len(getattr(shell, "proc_xyz", []))
    _gx = getattr(shell, "grown_xyz", None)
    n_grown = 0 if _gx is None else len(_gx)
    print(
        f"n_soma={n_soma:,} n_proc={n_proc:,} n_grown={n_grown:,} "
        f"adult={fly.is_adult()} thin={shell.draw_thin_mode}"
    )
    print(
        f"layers neurons={shell.show_layer_neurons} axons={shell.show_layer_axons} "
        f"dend={shell.show_layer_dendrites} graph={shell.show_layer_graph} "
        f"basins={shell.show_layer_basins}"
    )
    print(
        f"funnels oc={shell.show_ocelli_funnel} tu={shell.show_turret_funnel} "
        f"learn_on={shell.learn_on} amp={shell._learn_flash_amp:.3f}"
    )

    # Ensure adult closed-loop flash state exists for flash benches.
    shell.learn_on = True
    shell.ocelli_on = True
    shell.turret_on = True
    if fly.is_adult():
        shell._activity_running = True
        shell.sim_running = True
        for _ in range(6):
            try:
                shell._tick_adult_closed_loop(1.0 / 60.0)
            except Exception as exc:
                print(f"[tick] {exc}")
                break
        print(
            f"after_ticks amp={shell._learn_flash_amp:.3f} "
            f"flash_n={shell._learn_flash_n} "
            f"pre={len(getattr(shell, '_learn_flash_pre', []))}"
        )

    # Force amp so learn flash path is timed even if learn didn't fire.
    if float(shell._learn_flash_amp) < 0.1 and len(shell._learn_flash_pre):
        shell._learn_flash_amp = 0.85

    cx = shell.view_w * 0.5 + shell.pan_x
    cy = (shell.view_h - 80) * 0.5 + shell.pan_y
    view_lim_y = shell.view_h - 80  # approx NAV

    # ---------- Pure Python / numpy microbenches ----------
    print("\n=== Pure microbenches (no blit) ===")
    xyz_all = np.asarray(shell.points_xyz if n_soma else fly.soma_xyz, dtype=np.float32)
    if len(xyz_all) == 0:
        xyz_all = np.random.randn(140_000, 3).astype(np.float32)

    print(f"project_3d 140k   {ms(lambda: project_3d_batch(xyz_all[:140000] if len(xyz_all)>=140000 else xyz_all, shell.view_angle_x, shell.view_angle_y, cx, cy, shell.scale_3d), n=8):6.1f} ms  n={min(140000,len(xyz_all)):,}")
    for n in (16_000, 18_000, 52_000, len(xyz_all)):
        if n > len(xyz_all):
            continue
        sub = xyz_all[:n]
        print(f"project_3d {n:6d}  {ms(lambda: project_3d_batch(sub, shell.view_angle_x, shell.view_angle_y, cx, cy, shell.scale_3d), n=10):6.1f} ms")

    if n_proc and len(shell.proc_cls):
        print(
            f"thin proc→{DRAW_PROC_MAX}  "
            f"{ms(lambda: shell._apply_draw_thin(shell.proc_cls, DRAW_PROC_MAX, boost_sparse_soma=True), n=8):6.1f} ms"
        )
    if n_soma and len(shell.points_cls):
        print(
            f"thin soma→{DRAW_SOMA_MAX}  "
            f"{ms(lambda: shell._apply_draw_thin(shell.points_cls, DRAW_SOMA_MAX), n=8):6.1f} ms"
        )
        print(
            f"density_take soma  "
            f"{ms(lambda: _density_take(n_soma, DRAW_SOMA_MAX), n=20):6.1f} ms"
        )
        print(
            f"stratified soma    "
            f"{ms(lambda: _stratified_density_take(shell.points_cls, DRAW_SOMA_MAX), n=8):6.1f} ms"
        )

    # Learn flash pair Python loops (no pygame circles).
    pre = getattr(shell, "_learn_flash_pre", None)
    post = getattr(shell, "_learn_flash_post", None)
    if pre is not None and len(pre):
        ids = fly.soma_ids
        xyz = fly.soma_xyz

        def flash_prep():
            lut = {int(i): k for k, i in enumerate(ids.tolist())}
            engaged: set[int] = set()
            pairs = []
            for a, b in zip(pre.tolist(), post.tolist()):
                a, b = int(a), int(b)
                engaged.add(a)
                engaged.add(b)
                if a != b:
                    pairs.append((a, b))
            if len(engaged) > 900:
                engaged = set(list(engaged)[:900])
                pairs = [(a, b) for a, b in pairs if a in engaged and b in engaged]
            eng_idx = [lut[i] for i in engaged if i in lut]
            if eng_idx:
                e_xyz = xyz[np.asarray(eng_idx, dtype=np.int32)]
                project_3d_batch(
                    e_xyz, shell.view_angle_x, shell.view_angle_y, cx, cy, shell.scale_3d
                )
            return len(engaged), len(pairs)

        en, npairs = flash_prep()
        print(
            f"flash_prep+proj    {ms(flash_prep, n=5):6.1f} ms  "
            f"engaged={en} pairs={npairs} raw_pre={len(pre)}"
        )

        def topo():
            device = set(shell._turret_flat_ids()) | set(shell._ocelli_flat_ids())
            engaged = set(int(a) for a in pre.tolist()) | set(int(b) for b in post.tolist())
            if len(engaged) > 900:
                engaged = set(list(engaged)[:900])
            pairs = [
                (int(a), int(b))
                for a, b in zip(pre.tolist(), post.tolist())
                if int(a) != int(b) and int(a) in engaged and int(b) in engaged
            ]
            return shell._learn_topo_polylines(pairs, engaged, device=device)

        polys = topo()
        print(
            f"learn_topo_polys   {ms(topo, n=3, warm=1):6.1f} ms  n_polys={len(polys)}"
        )
    else:
        print("flash_prep          — (no flash pairs)")

    # Well / funnel geometry (CPU, may hit cache).
    def wells():
        return shell._device_funnel_pair()

    t_w = ms(wells, n=3, warm=1)
    specs = wells()
    print(f"device_funnel_pair  {t_w:6.1f} ms  specs={0 if specs is None else len(specs)}")

    # ---------- Stamp microbench (surfarray) ----------
    print("\n=== Stamp / surfarray microbenches ===")
    assert shell._view_surf is not None
    surf = shell._view_surf
    vw, vh = shell.view_w, shell.view_h

    def stamp_n(n: int, size: int = 1, fat: bool = False):
        if n_soma:
            idx = np.linspace(0, n_soma - 1, num=min(n, n_soma), dtype=np.int32)
            xyz = shell.points_xyz[idx]
            rgb = shell.point_rgb[idx]
        else:
            xyz = xyz_all[:n]
            rgb = np.random.randint(40, 220, (len(xyz), 3), dtype=np.uint8)
        sx, sy = project_3d_batch(
            xyz, shell.view_angle_x, shell.view_angle_y, cx, cy, shell.scale_3d
        )
        xi = np.rint(sx).astype(np.int32)
        yi = np.rint(sy).astype(np.int32)
        m = (xi >= 0) & (xi < vw) & (yi >= 0) & (yi < vh)
        if not np.any(m):
            return
        px = pygame.surfarray.pixels3d(surf)
        if size >= 3 or (fat and size >= 2):
            offsets = (
                (0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1),
                (1, -1), (-1, 1), (-1, -1),
            )
        elif fat or size >= 2:
            offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
        else:
            offsets = ((0, 0),)
        for dx, dy in offsets:
            xx = xi[m] + dx
            yy = yi[m] + dy
            ok = (xx >= 0) & (xx < vw) & (yy >= 0) & (yy < vh)
            cur = px[xx[ok], yy[ok]]
            nxt = rgb[m][ok]
            px[xx[ok], yy[ok]] = np.maximum(cur, nxt)
        del px

    for n, size, fat, label in (
        (16_000, 1, False, "proc 16k size1"),
        (16_000, 2, True, "soma 16k size2 fat"),
        (18_000, 1, False, "grown 18k size1 add~"),
        (800, 3, True, "hot 800 size3"),
    ):
        print(f"stamp {label:22s} {ms(lambda: stamp_n(n, size, fat), n=4):6.1f} ms")

    # Python line loop proxy for growth overlay segments.
    try:
        (
            _tip,
            _code,
            ray_a,
            ray_b,
            ray_cls,
            dend_a,
            dend_b,
            dend_cls,
        ) = fly.growth_overlay_arrays(
            tip_max=400, ray_max=2200, seg_max=3200, cls_filter=None
        )
    except TypeError:
        (
            _tip,
            _code,
            ray_a,
            ray_b,
            ray_cls,
            dend_a,
            dend_b,
            dend_cls,
        ) = fly.growth_overlay_arrays(tip_max=400, ray_max=2200, cls_filter=None)
    n_ray = len(ray_a) if ray_a is not None else 0
    n_dend = len(dend_a) if dend_a is not None else 0
    print(f"growth_overlay_arrays rays={n_ray} dend={n_dend}")

    def draw_seg_lines(a, b, cap=3200):
        if a is None or len(a) == 0:
            return
        if len(a) > cap:
            step = max(1, (len(a) + cap - 1) // cap)
            a = a[::step][:cap]
            b = b[::step][:cap]
        sa, ta = project_3d_batch(
            a, shell.view_angle_x, shell.view_angle_y, cx, cy, shell.scale_3d
        )
        sb, tb = project_3d_batch(
            b, shell.view_angle_x, shell.view_angle_y, cx, cy, shell.scale_3d
        )
        for i in range(len(a)):
            x0, y0 = int(sa[i]), int(ta[i])
            x1, y1 = int(sb[i]), int(tb[i])
            if (0 <= x0 < vw and 0 <= y0 < vh) or (0 <= x1 < vw and 0 <= y1 < vh):
                pygame.draw.line(surf, (120, 160, 200), (x0, y0), (x1, y1), 1)

    print(
        f"seg lines axons~{min(n_ray,3200):4d} "
        f"{ms(lambda: draw_seg_lines(ray_a, ray_b, 3200), n=3):6.1f} ms"
    )
    print(
        f"seg lines dend~{min(n_dend,3200):4d} "
        f"{ms(lambda: draw_seg_lines(dend_a, dend_b, 3200), n=3):6.1f} ms"
    )

    # ---------- Instrumented _draw / _draw_particles ----------
    print("\n=== Instrumented GenomeShell draw (dummy SDL) ===")

    def time_call(name, fn, acc):
        t0 = time.perf_counter()
        fn()
        acc[name] = (time.perf_counter() - t0) * 1000.0

    # Warm caches (ground, sky, compass).
    for _ in range(2):
        shell._draw()

    # Patch _draw_particles pieces by wrapping methods temporarily.
    results = []
    for trial in range(4):
        parts: dict[str, float] = {}
        # Full _draw
        t0 = time.perf_counter()
        shell._draw()
        parts["FULL__draw"] = (time.perf_counter() - t0) * 1000.0

        # Break down particles path manually (mirror call order).
        shell.screen.fill((12, 12, 16))
        t0 = time.perf_counter()
        shell._draw_control_panel()
        parts["control_panel"] = (time.perf_counter() - t0) * 1000.0

        # Re-run particles with internal timing via monkeypatch of key methods.
        view_lim = shell.view_h - 80
        # Approximate NAV from module
        from genome_shell import NAV_PANEL_H

        view_lim = shell.view_h - NAV_PANEL_H
        cx2 = shell.view_w * 0.5 + shell.pan_x
        cy2 = (shell.view_h - NAV_PANEL_H) * 0.5 + shell.pan_y

        t0 = time.perf_counter()
        shell._draw_sky_fill(view_lim)
        parts["sky"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell._draw_ground(cx2, cy2, view_lim)
        parts["ground"] = (time.perf_counter() - t0) * 1000.0

        # Stamp body layers by calling _draw_particles but we need finer grain —
        # time whole _draw_particles minus known overlays by sequencing.
        # Capture stamp costs by temporarily wrapping.
        stamp_acc = {"n": 0, "ms": 0.0}

        # Directly invoke the stamp-heavy portion: call _draw_particles and
        # time overlays separately after a clean fill.
        shell._view_surf.fill((12, 12, 16))
        t0 = time.perf_counter()
        shell._draw_particles()
        parts["particles_ALL"] = (time.perf_counter() - t0) * 1000.0

        # Overlay pieces alone (after particles already painted once).
        t0 = time.perf_counter()
        if shell.show_layer_basins:
            shell._draw_basins_layer(cx2, cy2, view_lim, shell.focus_basin_indices)
        parts["basins"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell._draw_device_funnels(cx2, cy2, view_lim)
        parts["device_funnels"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell._draw_channel_bed(cx2, cy2, view_lim)
        parts["channel_bed"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell._draw_orbit_compass(cx2, cy2, view_lim)
        parts["compass"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell._draw_point_lamp(cx2, cy2, view_lim)
        parts["lamp"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        from device_render import draw_devices

        draw_devices(
            shell,
            list(
                getattr(shell, "device_list", None)
                or getattr(getattr(shell, "blueprint", None), "device_list", [])
                or []
            ),
            cx2,
            cy2,
            view_lim,
        )
        parts["devices"] = (time.perf_counter() - t0) * 1000.0

        # Preserve amp across decay in flash draw.
        amp_save = float(shell._learn_flash_amp)
        t0 = time.perf_counter()
        shell._draw_learn_flash(cx2, cy2, view_lim)
        parts["learn_flash"] = (time.perf_counter() - t0) * 1000.0
        shell._learn_flash_amp = amp_save

        t0 = time.perf_counter()
        shell._draw_nav()
        parts["nav"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell._draw_fps()
        parts["fps_hud"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell._draw_legend()
        parts["legend"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        shell.screen.blit(shell._view_surf, (shell.view_x, shell.view_y))
        pygame.display.flip()
        parts["blit_flip"] = (time.perf_counter() - t0) * 1000.0

        # Growth overlay alone
        t0 = time.perf_counter()
        if shell.show_layer_axons or shell.show_layer_dendrites:
            shell._draw_growth_overlay(
                cx2,
                cy2,
                view_lim,
                axon_f=shell.focus_axon_indices,
                dend_f=shell.focus_dendrite_indices,
            )
        parts["growth_overlay"] = (time.perf_counter() - t0) * 1000.0

        # Recipe graph
        t0 = time.perf_counter()
        if shell.show_layer_graph:
            from genome_shell import DRAW_GRAPH_EDGES_MAX

            shell._draw_recipe_graph_overlay(
                cx2, cy2, view_lim, shell.focus_graph_indices, max_edges=DRAW_GRAPH_EDGES_MAX
            )
        parts["graph"] = (time.perf_counter() - t0) * 1000.0

        results.append(parts)

    # Average
    keys = sorted(results[0].keys())
    avg = {k: sum(r[k] for r in results) / len(results) for k in keys}
    print(f"(avg of {len(results)} trials after warm)")
    for k in sorted(avg, key=lambda x: -avg[x]):
        print(f"  {k:22s} {avg[k]:7.1f} ms")

    # Finer stamp breakdown: monkeypatch-free by timing thin+project+surfarray
    # for each layer budget as used in default adult view.
    print("\n=== Layer stamp estimates (project+surfarray max-blend) ===")
    if n_proc:
        pi = shell._apply_draw_thin(shell.proc_cls, DRAW_PROC_MAX, boost_sparse_soma=True)
        n_pi = n_proc if isinstance(pi, slice) else len(pi)
        print(
            f"proc stamp n={n_pi:,}  "
            f"{ms(lambda: stamp_n(n_pi, 1, False), n=3):6.1f} ms"
        )
    if n_soma:
        si = shell._apply_draw_thin(shell.points_cls, DRAW_SOMA_MAX)
        n_si = n_soma if isinstance(si, slice) else len(si)
        print(
            f"soma stamp n={n_si:,} size2  "
            f"{ms(lambda: stamp_n(n_si, 2, True), n=3):6.1f} ms"
        )
    if n_grown:
        from genome_shell import DRAW_GROWN_PTS_MAX as GMAX

        gi = _density_take(n_grown, GMAX)
        n_gi = n_grown if isinstance(gi, slice) else len(gi)
        print(
            f"grown stamp n={n_gi:,}  "
            f"{ms(lambda: stamp_n(min(n_gi, GMAX), 1, False), n=3):6.1f} ms"
        )

    # Sim cost reference
    print("\n=== Sim reference (same process) ===")
    if fly.is_adult():
        dt = 1.0 / 60.0
        print(f"closed_loop.step  {ms(lambda: shell._tick_adult_closed_loop(dt), n=4, warm=1):6.1f} ms")
    print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile a realistic GenomeShell.run()-like frame sequence (CPU tissue path).

Loads x.fly via GenomeShell boot (UI state restored), enables adult closed-loop
+ learn + devices like post-load defaults, then runs N frames of the same
sequence as run() (events skipped). Reports per-piece avg ms and tissue_hit rate.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["FLYWIRE_NO_GL"] = "1"  # force CPU path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

N_FRAMES = 30
FPS = 60
DT = 1.0 / float(FPS)


def main() -> int:
    import pygame
    import genome_shell as gs
    from genome_shell import DRAW_SOMA_MAX, GenomeShell, NAV_PANEL_H

    t_boot0 = time.perf_counter()
    shell = GenomeShell()
    t_boot = (time.perf_counter() - t_boot0) * 1000.0

    # Mirror defaults after load / Run: adult closed-loop + learn + devices.
    shell.use_gl_tissue = False
    shell.learn_on = True
    shell.ocelli_on = True
    shell.turret_on = True
    shell.sim_running = True
    shell._activity_running = True
    shell._sim_accum = 0.0
    shell._last_frame_wall = time.perf_counter()

    print("=== UI / boot state ===")
    print(f"boot_ms={t_boot:.0f}")
    print(f"driver={pygame.display.get_driver()} size={shell.screen.get_size()}")
    print(f"view={shell.view_w}x{shell.view_h} panel_w={shell.panel_w}")
    print(
        f"use_gl={shell.use_gl_tissue} adult_sim_hz={shell.adult_sim_hz} "
        f"learn_every={shell.learn_every} loop_hops={shell.loop_hops}"
    )
    print(
        f"funnels oc={shell.show_ocelli_funnel} tu={shell.show_turret_funnel} "
        f"scope_on={shell.scope_on}"
    )
    print(
        f"layers n={shell.show_layer_neurons} g={shell.show_layer_graph} "
        f"ax={shell.show_layer_axons} den={shell.show_layer_dendrites} "
        f"basins={shell.show_layer_basins}"
    )
    print(
        f"draw_thin={shell.draw_thin_mode} ground={shell.ground_on} "
        f"lamp={shell.lamp_on} learn={shell.learn_on} "
        f"adult={shell.fly.is_adult()} n_soma={len(shell.points_xyz)}"
    )
    print(
        f"focus soma={len(shell.focus_soma_indices)} axon={len(shell.focus_axon_indices)} "
        f"dend={len(shell.focus_dendrite_indices)} basin={len(shell.focus_basin_indices)}"
    )
    print(
        f"camera ax={shell.view_angle_x:.3f} ay={shell.view_angle_y:.3f} "
        f"scale={shell.scale_3d:.2f} pan=({shell.pan_x:.1f},{shell.pan_y:.1f})"
    )

    # Warm: one closed-loop + one cold tissue fill so rate reflects steady state.
    try:
        shell._tick_adult_closed_loop(DT)
    except Exception as exc:
        print(f"[warm tick] {exc}")
    shell._tissue_blit_key = None
    shell._tissue_blit_surf = None
    t_cold0 = time.perf_counter()
    shell._draw()
    t_cold = (time.perf_counter() - t_cold0) * 1000.0
    print(f"cold_draw_ms={t_cold:.1f}")

    # Instrument pieces.
    times: dict[str, list[float]] = defaultdict(list)
    tissue_hits = 0
    tissue_checks = 0
    amp_samples: list[float] = []
    ground_calls = 0
    well_rebuilds = 0

    restores: list[tuple[object, str, object]] = []

    def wrap(obj: object, name: str, key: str) -> None:
        orig = getattr(obj, name)

        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                times[key].append((time.perf_counter() - t0) * 1000.0)

        setattr(obj, name, wrapped)
        restores.append((obj, name, orig))

    wrap(shell, "_begin_frame_caches", "begin_caches")
    wrap(shell, "_tick_lamp", "tick_lamp")
    wrap(shell, "_update_ocelli_sensors", "ocelli_sensors")
    wrap(shell, "_update_turret_sensors", "turret_sensors")
    wrap(shell, "_update_turret_aim_metric", "turret_aim")
    wrap(shell, "_sync_device_funnel_gravity", "funnel_gravity")
    wrap(shell, "_tick_adult_closed_loop", "closed_loop")
    wrap(shell, "_draw_ground", "draw_ground")
    wrap(shell, "_draw_learn_flash", "learn_flash")
    wrap(shell, "_draw_orbit_compass", "compass")
    wrap(shell, "_draw_point_lamp", "lamp_sprite")
    wrap(shell, "_draw_device_funnels", "draw_funnels")
    wrap(shell, "_draw_channel_bed", "channel_bed")
    wrap(shell, "_draw_basins_layer", "basins")
    wrap(shell, "_draw_growth_overlay", "growth_overlay")
    wrap(shell, "_draw_control_panel", "control_panel")
    wrap(shell, "_draw_legend", "legend")
    wrap(shell, "_draw_nav", "nav")
    wrap(shell, "_draw_fps", "fps")
    wrap(shell, "_learn_topo_polylines", "learn_topo")

    # WellField rebuild if funnels path ever builds.
    if hasattr(shell, "_device_funnel_pair"):
        orig_pair = shell._device_funnel_pair

        def wrapped_pair(*a, **kw):
            nonlocal well_rebuilds
            t0 = time.perf_counter()
            try:
                return orig_pair(*a, **kw)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                times["funnel_pair"].append(dt_ms)
                if dt_ms > 5.0:
                    well_rebuilds += 1

        shell._device_funnel_pair = wrapped_pair  # type: ignore[method-assign]
        restores.append((shell, "_device_funnel_pair", orig_pair))

    orig_ground = None
    for obj, name, orig in list(restores):
        if name == "_draw_ground":
            orig_ground = orig
            break

    # Re-wrap ground to count calls (tissue miss path).
    if orig_ground is not None:

        def wrapped_ground(*a, **kw):
            nonlocal ground_calls
            ground_calls += 1
            t0 = time.perf_counter()
            try:
                return orig_ground(*a, **kw)
            finally:
                times["draw_ground"].append((time.perf_counter() - t0) * 1000.0)

        shell._draw_ground = wrapped_ground  # type: ignore[method-assign]

    orig_dd = gs.draw_devices

    def wrapped_dd(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_dd(*a, **kw)
        finally:
            times["devices"].append((time.perf_counter() - t0) * 1000.0)

    gs.draw_devices = wrapped_dd

    orig_particles = shell._draw_particles

    def wrapped_particles(*a, **kw):
        # Peek tissue hit the same way product does.
        nonlocal tissue_hits, tissue_checks
        cx = shell.view_w * 0.5 + shell.pan_x
        cy = (shell.view_h - NAV_PANEL_H) * 0.5 + shell.pan_y
        view_lim_y = shell.view_h - NAV_PANEL_H
        soma_f = shell.focus_soma_indices
        graph_f = shell.focus_graph_indices
        axon_f = shell.focus_axon_indices
        dend_f = shell.focus_dendrite_indices
        tissue_key = (
            round(float(shell.view_angle_x), 4),
            round(float(shell.view_angle_y), 4),
            round(float(shell.scale_3d), 3),
            round(float(shell.pan_x), 1),
            round(float(shell.pan_y), 1),
            int(shell.view_w),
            int(view_lim_y),
            bool(shell.show_layer_neurons),
            bool(shell.show_layer_graph),
            bool(shell.show_layer_axons),
            bool(shell.show_layer_dendrites),
            str(getattr(shell, "draw_thin_mode", "auto")),
            int(getattr(shell, "draw_soma_budget", DRAW_SOMA_MAX)),
            tuple(
                sorted(
                    (k, round(float(v), 3))
                    for k, v in (shell.draw_density or {}).items()
                )
            ),
            tuple(sorted(int(i) for i in soma_f)),
            tuple(sorted(int(i) for i in graph_f)),
            tuple(sorted(int(i) for i in axon_f)),
            tuple(sorted(int(i) for i in dend_f)),
            bool(getattr(shell, "scope_on", False)),
            int(id(getattr(shell, "_scope_mask", None))),
            bool(getattr(shell, "ground_on", False)),
            round(float(getattr(shell, "sky_fill", 0.0)), 3),
            bool(getattr(shell, "lamp_on", False)),
            round(float(getattr(shell, "lamp_intensity", 1.0)), 2),
        )
        hit = (
            tissue_key == shell._tissue_blit_key
            and shell._tissue_blit_surf is not None
            and shell._tissue_blit_surf.get_size() == (shell.view_w, shell.view_h)
        )
        tissue_checks += 1
        if hit:
            tissue_hits += 1
        t0 = time.perf_counter()
        orig_particles(*a, **kw)
        times["particles"].append((time.perf_counter() - t0) * 1000.0)
        times["tissue_hit_flag"].append(1.0 if hit else 0.0)

    shell._draw_particles = wrapped_particles  # type: ignore[method-assign]

    orig_draw = shell._draw

    def wrapped_draw():
        t0 = time.perf_counter()
        orig_draw()
        times["draw_total"].append((time.perf_counter() - t0) * 1000.0)

    shell._draw = wrapped_draw  # type: ignore[method-assign]

    frame_totals: list[float] = []
    for i in range(N_FRAMES):
        t_frame0 = time.perf_counter()
        # Same sequence as run() minus events.
        now = time.perf_counter()
        wall_dt = min(0.1, max(1e-4, now - float(shell._last_frame_wall)))
        shell._last_frame_wall = now

        shell._begin_frame_caches()
        shell._tick_lamp(DT)
        if shell.ocelli_on:
            shell._update_ocelli_sensors()
        if shell.turret_on:
            shell._update_turret_sensors()
            shell._update_turret_aim_metric()
        if getattr(shell, "show_ocelli_funnel", False) or getattr(
            shell, "show_turret_funnel", False
        ):
            shell._sync_device_funnel_gravity()
        if shell.sim_running:
            shell.fly.sim_tick()
            if shell.fly.is_adult() and shell._activity_running:
                sim_hz = float(getattr(shell, "adult_sim_hz", 20.0) or 20.0)
                sim_hz = float(__import__("numpy").clip(sim_hz, 5.0, float(FPS)))
                step = 1.0 / sim_hz
                shell._sim_accum = float(getattr(shell, "_sim_accum", 0.0)) + wall_dt
                if shell._sim_accum >= step:
                    shell._tick_adult_closed_loop(step)
                    shell._sim_accum -= step
                    if shell._sim_accum > 2.0 * step:
                        shell._sim_accum = step
        shell._draw()
        amp_samples.append(float(getattr(shell, "_learn_flash_amp", 0.0) or 0.0))
        frame_totals.append((time.perf_counter() - t_frame0) * 1000.0)
        if i < 3 or i == N_FRAMES - 1:
            print(
                f"frame[{i}] total={frame_totals[-1]:.1f}ms "
                f"amp={amp_samples[-1]:.3f} "
                f"lamp_ang={shell.lamp_angle:.3f}"
            )

    # Restore
    for obj, name, orig in restores:
        setattr(obj, name, orig)
    shell._draw_particles = orig_particles
    shell._draw = orig_draw
    gs.draw_devices = orig_dd

    def avg(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    print("\n=== timing table (avg ms over %d frames) ===" % N_FRAMES)
    print(f"{'piece':28s}  {'avg_ms':>8}  {'max_ms':>8}  {'n':>4}")
    rows = [
        ("frame_total", frame_totals),
        ("draw_total", times.get("draw_total", [])),
        ("particles", times.get("particles", [])),
        ("closed_loop", times.get("closed_loop", [])),
        ("begin_caches", times.get("begin_caches", [])),
        ("tick_lamp", times.get("tick_lamp", [])),
        ("ocelli_sensors", times.get("ocelli_sensors", [])),
        ("turret_sensors", times.get("turret_sensors", [])),
        ("turret_aim", times.get("turret_aim", [])),
        ("funnel_gravity", times.get("funnel_gravity", [])),
        ("funnel_pair", times.get("funnel_pair", [])),
        ("draw_ground", times.get("draw_ground", [])),
        ("growth_overlay", times.get("growth_overlay", [])),
        ("learn_flash", times.get("learn_flash", [])),
        ("learn_topo", times.get("learn_topo", [])),
        ("compass", times.get("compass", [])),
        ("lamp_sprite", times.get("lamp_sprite", [])),
        ("devices", times.get("devices", [])),
        ("draw_funnels", times.get("draw_funnels", [])),
        ("channel_bed", times.get("channel_bed", [])),
        ("basins", times.get("basins", [])),
        ("control_panel", times.get("control_panel", [])),
        ("legend", times.get("legend", [])),
        ("nav", times.get("nav", [])),
        ("fps", times.get("fps", [])),
    ]
    for name, xs in rows:
        if not xs:
            print(f"{name:28s}  {'—':>8}  {'—':>8}  {0:4d}")
            continue
        print(f"{name:28s}  {avg(xs):8.2f}  {max(xs):8.2f}  {len(xs):4d}")

    hit_rate = tissue_hits / max(1, tissue_checks)
    print(
        f"\ntissue_hit={tissue_hits}/{tissue_checks} ({100.0 * hit_rate:.1f}%)"
    )
    print(f"ground_calls={ground_calls} (should be 0 if tissue always hits)")
    print(f"well_rebuild_hot={well_rebuilds}")
    print(
        f"learn_flash_amp min/avg/max="
        f"{min(amp_samples):.3f}/{avg(amp_samples):.3f}/{max(amp_samples):.3f}"
    )
    print(
        f"implied_fps≈{1000.0 / max(1e-3, avg(frame_totals)):.1f} "
        f"(avg frame {avg(frame_totals):.1f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

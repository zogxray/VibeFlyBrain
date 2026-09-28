#!/usr/bin/env python3
"""Instrument ONE warm GenomeShell._draw after tissue blit cache fill.

Times: tissue blit, activity glow, compass, lamp, devices, learn_flash,
control_panel, legend, flip. Does not modify product code.
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


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    import pygame
    import genome_shell as gs
    from fly_model import load_fly
    from genome_shell import DRAW_SOMA_MAX, GenomeShell, NAV_PANEL_H

    fly_path = ROOT / "data" / "flies" / "x.fly"
    if not fly_path.exists():
        fly_path = ROOT / "data" / "x.fly"
    if not fly_path.exists():
        raise SystemExit(f"missing x.fly under {ROOT / 'data'}")

    shell = GenomeShell()
    loaded = load_fly(fly_path)
    shell._adopt_fly(loaded, f"Loaded {fly_path.name}", model_path=fly_path)
    shell.learn_on = True
    shell.ocelli_on = True
    shell.turret_on = True
    shell._learn_flash_amp = 0.0
    shell.show_ocelli_funnel = False
    shell.show_turret_funnel = False

    print(
        f"after_boot use_gl_tissue={shell.use_gl_tissue} "
        f"adult_sim_hz={shell.adult_sim_hz} learn_every={shell.learn_every}"
    )
    funnels_off = (
        not shell.show_ocelli_funnel
        and not shell.show_turret_funnel
        and not getattr(shell, "scope_on", False)
    )
    print(
        f"funnels_off={funnels_off} "
        f"(oc={shell.show_ocelli_funnel} tu={shell.show_turret_funnel} "
        f"scope_on={getattr(shell, 'scope_on', False)})"
    )

    dt = 1.0 / 60.0
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

    # Fill tissue cache (2 draws).
    shell._tissue_blit_key = None
    shell._tissue_blit_surf = None
    shell._draw()
    shell._draw()
    key_present = (
        shell._tissue_blit_key is not None and shell._tissue_blit_surf is not None
    )
    print(f"tissue_blit_key_present_after_2_draws={key_present}")

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
            sorted((k, round(float(v), 3)) for k, v in (shell.draw_density or {}).items())
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
    tissue_hit = (
        tissue_key == shell._tissue_blit_key
        and shell._tissue_blit_surf is not None
        and shell._tissue_blit_surf.get_size() == (shell.view_w, shell.view_h)
    )
    print(f"tissue_hit={tissue_hit}")

    # Isolated tissue blit cost (same call product uses on hit).
    assert shell._view_surf is not None and shell._tissue_blit_surf is not None
    t0 = time.perf_counter()
    shell._view_surf.blit(shell._tissue_blit_surf, (0, 0))
    blit_tissue_ms = _ms(t0)

    times: dict[str, float] = {"blit_tissue": blit_tissue_ms}
    restores: list[tuple[object, str, object]] = []

    def wrap(obj: object, name: str, key: str) -> None:
        orig = getattr(obj, name)

        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                times[key] = times.get(key, 0.0) + _ms(t0)

        setattr(obj, name, wrapped)
        restores.append((obj, name, orig))

    wrap(shell, "_draw_control_panel", "control_panel")
    wrap(shell, "_draw_orbit_compass", "compass")
    wrap(shell, "_draw_point_lamp", "lamp")
    wrap(shell, "_draw_learn_flash", "learn_flash")
    wrap(shell, "_draw_legend", "legend")
    wrap(shell, "_draw_device_funnels", "funnels")
    wrap(shell, "_draw_channel_bed", "channel_bed")
    wrap(shell, "_draw_basins_layer", "basins")
    wrap(shell, "_draw_nav", "nav")
    wrap(shell, "_draw_fps", "fps")

    orig_dd = gs.draw_devices

    def wrapped_dd(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_dd(*a, **kw)
        finally:
            times["devices"] = times.get("devices", 0.0) + _ms(t0)

    gs.draw_devices = wrapped_dd

    orig_flip = pygame.display.flip

    def wrapped_flip(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_flip(*a, **kw)
        finally:
            times["flip"] = times.get("flip", 0.0) + _ms(t0)

    pygame.display.flip = wrapped_flip

    # Wrap particles: measure whole call; glow = particles - blit - overlays.
    orig_particles = shell._draw_particles

    def wrapped_particles(*a, **kw):
        t0 = time.perf_counter()
        orig_particles(*a, **kw)
        particles_ms = _ms(t0)
        times["particles"] = particles_ms
        overlay = sum(
            times.get(k, 0.0)
            for k in (
                "compass",
                "lamp",
                "devices",
                "learn_flash",
                "funnels",
                "channel_bed",
                "basins",
                "nav",
                "fps",
            )
        )
        times["activity_glow"] = max(0.0, particles_ms - blit_tissue_ms - overlay)

    shell._draw_particles = wrapped_particles

    t_all0 = time.perf_counter()
    shell._draw()
    times["_draw_total"] = _ms(t_all0)

    for obj, name, orig in restores:
        setattr(obj, name, orig)
    shell._draw_particles = orig_particles
    gs.draw_devices = orig_dd
    pygame.display.flip = orig_flip

    print("\n=== warm _draw pieces (ms) ===")
    print(f"{'#':>2}  {'piece':28s}  {'ms':>8}")
    order = [
        "control_panel",
        "blit_tissue",
        "activity_glow",
        "compass",
        "lamp",
        "devices",
        "learn_flash",
        "funnels",
        "channel_bed",
        "basins",
        "nav",
        "fps",
        "legend",
        "flip",
        "particles",
        "_draw_total",
    ]
    i = 0
    for name in order:
        if name not in times:
            continue
        i += 1
        print(f"{i:2d}  {name:28s}  {times[name]:8.2f}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Frozen plant: do the left and right eye waves land differently on the motors?

Seeds one eye at a time (unit drive on its clamps), hops the wave, and reads the
four wing-site motor levels. If both eyes produce the same site pattern the
eye→motor projection is common-mode and no credit rule can turn it into a steer
signal — plasticity would have to invent the lateralisation from nothing.

Usage:
  python scripts/probe_eye_motor_split.py --hops 14
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from blueprint import Blueprint  # noqa: E402
from cns_runtime import CNSRuntime  # noqa: E402
from fly_model import load_fly  # noqa: E402


def _wired(dev: dict) -> bool:
    groups = (dev.get("ocelli_groups") or []) + (dev.get("turret_motor_groups") or [])
    return any(isinstance(g, list) and len(g) > 0 for g in groups)


def _site_levels(rt: CNSRuntime) -> list[float]:
    return [float(rt._turret_site_motor_level_raw(si)) for si in range(4)]


def _wave(rt: CNSRuntime, seeds: dict[int, float], *, hops: int) -> list[float]:
    rt.fly.clear_activity()
    rt.fly.stimulate_neurons(seeds)
    peak = [0.0, 0.0, 0.0, 0.0]
    for _ in range(max(1, int(hops))):
        rt.fly.tick_activity(hop_gain=0.9, decay=0.97)
        rt.fly.rebuild_activity_cache()
        lv = _site_levels(rt)
        peak = [max(p, x) for p, x in zip(peak, lv)]
    return peak


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eye L/R → motor site split")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--hops", type=int, default=14)
    p.add_argument("--yaw-offset", type=float, default=2.2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ui_path = ROOT / "data" / "processed" / "shell_ui.json"
    ui = json.loads(ui_path.read_text()) if ui_path.is_file() else {}
    path = Path(args.model or ui.get("last_model") or (ROOT / "data" / "flies" / "x.fly"))
    if not path.is_file():
        print(f"[probe] missing model: {path}", flush=True)
        return 2
    fly_meta = getattr(load_fly(path), "shell_meta", {}) or {}
    fly_dev = fly_meta.get("shell_devices") or {}
    ui_dev = ui.get("shell_devices") or {}
    shell_dev = fly_dev if _wired(fly_dev) else (ui_dev if _wired(ui_dev) else None)

    bp = Blueprint.from_fly_meta(
        path, shell_devices=shell_dev, yaw_offset=float(args.yaw_offset), scope_on=True
    )
    rt = CNSRuntime.open(path, blueprint=bp, shell_devices=shell_dev, keep_gains=False)
    if shell_dev and hasattr(rt, "_apply_device_snapshot"):
        rt._apply_device_snapshot(shell_dev)
        rt._sync_shell_fields_to_devices()
        rt._devices_dirty = True
    for attr in ("eye_left_on", "eye_right_on", "jo_left_on", "jo_right_on", "ocelli_on"):
        setattr(rt, attr, True)
    for d in getattr(rt, "device_list", []) or []:
        if getattr(d, "kind", "") in ("eye", "jo", "ocelli"):
            d.enabled = True
    nf = rt.instant_fuse()
    rt.fly.contact_gain = {}
    rt.sync_training_scope()
    print(f"[probe] fuse={nf} hops={args.hops} gains=1 (frozen)", flush=True)

    eyes: dict[str, list[int]] = {}
    for d in getattr(rt, "device_list", []) or []:
        if getattr(d, "kind", "") == "eye":
            eyes[str(d.name)] = [int(x) for x in d.flat_ids()]
    if len(eyes) < 2:
        print("[probe] eyes not wired", flush=True)
        return 2

    out: dict[str, list[float]] = {}
    for name, ids in sorted(eyes.items()):
        peak = _wave(rt, {i: 1.0 for i in ids}, hops=int(args.hops))
        out[name] = peak
        left = 0.5 * (peak[0] + peak[1])
        right = 0.5 * (peak[2] + peak[3])
        print(
            f"[probe] {name:<10} sites=[{peak[0]:.3f} {peak[1]:.3f} "
            f"{peak[2]:.3f} {peak[3]:.3f}]  L={left:.3f} R={right:.3f} "
            f"yaw_cmd={left - right:+.4f}",
            flush=True,
        )

    a = np.asarray(out["eye_left"], dtype=np.float64)
    b = np.asarray(out["eye_right"], dtype=np.float64)
    ya = 0.5 * (a[0] + a[1]) - 0.5 * (a[2] + a[3])
    yb = 0.5 * (b[0] + b[1]) - 0.5 * (b[2] + b[3])
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-9)
    print("\n=== eye lateralisation at the motors ===", flush=True)
    print(
        f"yaw_cmd(eyeL)={ya:+.4f}  yaw_cmd(eyeR)={yb:+.4f}  "
        f"split={ya - yb:+.4f}  (peak site drive {scale:.3f})",
        flush=True,
    )
    rel = abs(ya - yb) / scale
    print(f"relative L/R split = {rel:.1%} of peak motor drive", flush=True)
    if rel < 0.05:
        print(
            "COMMON-MODE — both eyes drive the wing sites the same way; the "
            "substrate carries no side signal for plasticity to amplify",
            flush=True,
        )
        return 1
    print("LATERALISED — eye side reaches the motor sides differently", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

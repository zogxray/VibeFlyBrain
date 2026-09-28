#!/usr/bin/env python3
"""Which wing pads must each eye drive for the loop to null its own bearing?

Device names do not settle this: the pads are indexed by the turret's own yaw
convention, not by anatomy. This puts the lamp where one eye sees it, reads the
eye population bearing the trainer uses, and prints the ``yaw_cmd`` sign that
bearing demands. Since ``yaw_cmd = mean(pad0,pad1) - mean(pad2,pad3)``, that
sign names the pad pair — and it comes out crossed, each eye driving the pads
on the far side of the body.

Usage:
    python scripts/probe_eye_yaw_sign.py
"""

from __future__ import annotations

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
from compound_eyes import (  # noqa: E402
    eye_lamp_azimuth,
    eye_probe_light_angle,
    eye_rig_pose,
    update_eye_sensors,
)
from fly_model import load_fly  # noqa: E402


def _wired(dev: dict) -> bool:
    groups = (dev.get("ocelli_groups") or []) + (dev.get("turret_motor_groups") or [])
    return any(isinstance(g, list) and len(g) > 0 for g in groups)


def _wrap(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi


def main() -> int:
    ui_path = ROOT / "data" / "processed" / "shell_ui.json"
    ui = json.loads(ui_path.read_text()) if ui_path.is_file() else {}
    path = Path(ui.get("last_model") or (ROOT / "data" / "flies" / "x.fly"))
    if not path.is_file():
        print(f"[sign] missing model: {path}", flush=True)
        return 2
    fly_meta = getattr(load_fly(path), "shell_meta", {}) or {}
    fly_dev = fly_meta.get("shell_devices") or {}
    ui_dev = ui.get("shell_devices") or {}
    shell_dev = fly_dev if _wired(fly_dev) else (ui_dev if _wired(ui_dev) else None)

    bp = Blueprint.from_fly_meta(path, shell_devices=shell_dev, scope_on=True)
    rt = CNSRuntime.open(path, blueprint=bp, shell_devices=shell_dev, keep_gains=False)
    if shell_dev and hasattr(rt, "_apply_device_snapshot"):
        rt._apply_device_snapshot(shell_dev)
        rt._sync_shell_fields_to_devices()
    for attr in ("eye_left_on", "eye_right_on"):
        setattr(rt, attr, True)
    for d in getattr(rt, "device_list", []) or []:
        if getattr(d, "kind", "") == "eye":
            d.enabled = True
    rt.instant_fuse()

    rt.turret_yaw = 0.0
    rt._frame_turret_pose = None
    pose = rt._turret_rig_pose()
    if pose is None:
        print("[sign] turret not wired", flush=True)
        return 2
    sites = np.asarray(pose["sites"], dtype=np.float64)
    print("pads (index → soma x): " + "  ".join(f"{i}:{s[0]:+.1f}" for i, s in enumerate(sites)))
    print("yaw_cmd = mean(pad0,pad1) − mean(pad2,pad3);  +yaw aims the barrel toward +X\n")

    for d in getattr(rt, "device_list", []) or []:
        if getattr(d, "kind", "") != "eye":
            continue
        rig = eye_rig_pose(rt, d.side)
        if not rig:
            continue
        outward = np.asarray(rig[0]["outward"], dtype=np.float64)
        az_orbit = eye_probe_light_angle(rt, d)
        if az_orbit is None:
            continue
        rt.lamp_angle = float(az_orbit)
        rt._update_lamp_xyz()
        update_eye_sensors(rt)
        out = eye_lamp_azimuth(rt)
        if out is None:
            print(f"{d.name}: no bearing (eye dark)", flush=True)
            continue
        az, conf = out
        steer = _wrap(float(az) - float(rt.turret_yaw))
        pads = "0,1" if steer > 0 else "2,3"
        print(
            f"{d.name}: faces x={outward[0]:+.2f}  bearing={az:+.2f} rad (conf {conf:.2f})  "
            f"steer={steer:+.2f} ⇒ needs {'+' if steer > 0 else '−'}yaw_cmd ⇒ pads {pads}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Headless AI Brain Spin test — CNS Brain frozen, sensor→turret MLP only.

Train under Spin with replay, then Freeze and verify Track holds.

  python scripts/probe_port_nn.py --steps 2400 --lamp-speed 0.08
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cns_runtime import CNSRuntime  # noqa: E402
from fly_model import load_fly  # noqa: E402
from sensor_port_nn import SensorPortNN  # noqa: E402


def _enable_sensors(rt) -> None:
    rt.ocelli_on = True
    rt.turret_on = True
    for name in ("eye_left", "eye_right", "jo_left", "jo_right"):
        setattr(rt, f"{name}_on", True)
    for d in getattr(rt, "device_list", []) or []:
        d.enabled = True
    if hasattr(rt, "blueprint") and rt.blueprint is not None:
        rt.blueprint.ocelli_on = True
        rt.blueprint.turret_on = True
        if hasattr(rt.blueprint, "sync_bundle_from_list"):
            rt.blueprint.sync_bundle_from_list()
    rt._devices_dirty = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--freeze-steps", type=int, default=1200)
    ap.add_argument("--lamp-speed", type=float, default=0.08)
    ap.add_argument("--log-every", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--model", type=str, default="")
    args = ap.parse_args()

    dt = 1.0 / 60.0
    ui_path = ROOT / "data" / "processed" / "shell_ui.json"
    ui = json.loads(ui_path.read_text()) if ui_path.is_file() else {}
    path = Path(
        args.model or ui.get("last_model") or (ROOT / "data" / "flies" / "x.fly")
    )
    if not path.is_file():
        print(f"[port] missing model: {path}", flush=True)
        return 2

    fly_meta = getattr(load_fly(path), "shell_meta", {}) or {}
    fly_dev = fly_meta.get("shell_devices") or {}
    ui_dev = ui.get("shell_devices") or {}
    shell_dev = fly_dev if fly_dev else ui_dev

    print(f"[port] load {path}", flush=True)
    rt = CNSRuntime.open(
        path,
        shell_devices=shell_dev or None,
        lamp_speed=float(args.lamp_speed),
        keep_gains=False,
    )
    if shell_dev and hasattr(rt, "_apply_device_snapshot"):
        rt._apply_device_snapshot(shell_dev)
        rt._sync_shell_fields_to_devices()
        rt._devices_dirty = True
    _enable_sensors(rt)
    try:
        rt.fly.enter_adult(abandon_open=True)
        rt.fly.rebuild_caches()
        nf = rt.instant_fuse()
        print(f"[port] fuse pairs={nf}", flush=True)
    except Exception as exc:
        print(f"[port] WARN boot: {exc}", flush=True)

    rt.sim_running = True
    rt._activity_running = True
    rt.learn_on = False
    rt.motor_socket_on = False
    rt.fly.contact_gain = {}
    rt.sensor_port_nn = SensorPortNN(lr=float(args.lr))
    rt.sensor_port_on = True
    rt.sensor_port_train = True
    if hasattr(rt, "_track_reset"):
        rt._track_reset()
    if hasattr(rt, "prepare_activity"):
        rt.prepare_activity()

    print(
        f"[port] TRAIN steps={args.steps} spin={args.lamp_speed} lr={args.lr} "
        f"replay={rt.sensor_port_nn.replay_k} phantom={rt.sensor_port_nn.phantom_k}",
        flush=True,
    )
    t0 = time.time()
    hist: list[tuple[int, float, float, float]] = []

    for i in range(int(args.steps)):
        rt.fly.sim_time = int(i)
        rt._frame_dt = dt
        if float(rt.lamp_speed) > 0.0:
            rt._tick_lamp(dt)
        info = rt.tick(dt)
        port = rt.sensor_port_nn
        track = float(getattr(rt, "_track_score", float("nan")))
        err = getattr(rt, "turret_aim_err", float("nan"))
        eps = (
            float(math.degrees(err))
            if isinstance(err, (int, float)) and math.isfinite(float(err))
            else float("nan")
        )
        if i % max(1, int(args.log_every)) == 0 or i + 1 == int(args.steps):
            print(
                f"[port] train t={i:5d} ε={eps:6.1f}° Track={track:5.1%} "
                f"loss={port.loss_ema:.4f} n={port.step_n} "
                f"buf={len(port._buf_x)} gains={len(rt.fly.contact_gain or {})}",
                flush=True,
            )
            hist.append((i, eps, track, float(port.loss_ema)))

    # Freeze + RUN
    port = rt.sensor_port_nn
    port.freeze()
    rt.sensor_port_train = False
    if hasattr(rt, "_track_reset"):
        rt._track_reset()
    print(f"[port] FREEZE RUN steps={args.freeze_steps}", flush=True)
    fhist: list[tuple[int, float, float]] = []
    for i in range(int(args.freeze_steps)):
        rt.fly.sim_time = int(args.steps) + i
        rt._frame_dt = dt
        rt._tick_lamp(dt)
        rt.tick(dt)
        track = float(getattr(rt, "_track_score", float("nan")))
        err = getattr(rt, "turret_aim_err", float("nan"))
        eps = (
            float(math.degrees(err))
            if isinstance(err, (int, float)) and math.isfinite(float(err))
            else float("nan")
        )
        if i % max(1, int(args.log_every)) == 0 or i + 1 == int(args.freeze_steps):
            print(
                f"[port] freeze t={i:5d} ε={eps:6.1f}° Track={track:5.1%} "
                f"gains={len(rt.fly.contact_gain or {})}",
                flush=True,
            )
            fhist.append((i, eps, track))

    def _avg(rows, idx):
        vals = [r[idx] for r in rows if math.isfinite(r[idx])]
        return float(np.mean(vals)) if vals else float("nan")

    early = hist[: max(1, len(hist) // 3)]
    late_tr = hist[-max(1, len(hist) // 3) :]
    late_fz = fhist[-max(1, len(fhist) // 2) :]
    e_trk, e_loss = _avg(early, 2), _avg(early, 3)
    l_trk, l_loss = _avg(late_tr, 2), _avg(late_tr, 3)
    f_trk, f_eps = _avg(late_fz, 2), _avg(late_fz, 1)
    ok_learn = port.step_n > 100 and (
        (math.isfinite(l_loss) and math.isfinite(e_loss) and l_loss < max(0.02, e_loss * 0.7))
        or (math.isfinite(l_trk) and l_trk > e_trk + 0.15)
    )
    ok_freeze = math.isfinite(f_trk) and f_trk >= 0.70
    ok_tissue = len(rt.fly.contact_gain or {}) == 0
    elapsed = time.time() - t0
    print(
        f"\n=== AI BRAIN RESULT === {elapsed:.1f}s train_steps={port.step_n}\n"
        f"  train early: Track={e_trk:.1%} loss={e_loss:.4f}\n"
        f"  train late:  Track={l_trk:.1%} loss={l_loss:.4f}\n"
        f"  freeze late: Track={f_trk:.1%} ε={f_eps:.1f}°\n"
        f"  tissue_gains=0 ok={ok_tissue}\n"
        f"  LEARN={'YES' if ok_learn else 'NO'}  "
        f"FROZEN_HOLD={'YES' if ok_freeze else 'NO'}  "
        f"verdict={'PASS' if (ok_learn and ok_freeze and ok_tissue) else 'FAIL'}",
        flush=True,
    )
    return 0 if (ok_learn and ok_freeze and ok_tissue) else 1


if __name__ == "__main__":
    raise SystemExit(main())

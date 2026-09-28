#!/usr/bin/env python3
"""Frozen plant: does the eye population-vector azimuth recover the yaw lag?

No plasticity, turret pose held on a yaw grid. Compares
``wrap(eye_azimuth − turret_yaw)`` against the geometric AimTask lag.

Usage:
  python scripts/probe_eye_bearing.py --lamp-speed 0.08 --steps 1600 --yaw-grid 6
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
from compound_eyes import eye_lamp_azimuth  # noqa: E402
from fly_model import load_fly  # noqa: E402


def _wrap(a: float) -> float:
    return float((a + math.pi) % math.tau - math.pi)


def _wired(dev: dict) -> bool:
    groups = (dev.get("ocelli_groups") or []) + (dev.get("turret_motor_groups") or [])
    return any(isinstance(g, list) and len(g) > 0 for g in groups)


def _alt_azimuths(rt) -> dict[str, float]:
    """Candidate decodes, for picking the sharpest site→direction map."""
    from compound_eyes import eye_rig_pose

    axes: list[np.ndarray] = []
    vals: list[float] = []
    for side, attr in (("left", "eye_left_I"), ("right", "eye_right_I")):
        pose = eye_rig_pose(rt, side)
        if pose is None:
            continue
        I = np.asarray(getattr(rt, attr, np.zeros(6)), dtype=np.float64)
        for i, oc in enumerate(pose):
            axes.append(np.asarray(oc["axis"], dtype=np.float64))
            vals.append(float(I[i]) if i < len(I) else 0.0)
    A = np.stack(axes, axis=0)
    v = np.asarray(vals, dtype=np.float64)
    out: dict[str, float] = {}

    def _az(vec: np.ndarray) -> float:
        return float(math.atan2(float(vec[0]), float(vec[2])))

    # 1. plain population vector over mean-subtracted sites
    c = v - float(np.mean(v))
    out["pv_mean"] = _az((c[:, None] * A).sum(axis=0))
    # 2. rectified population vector (lit sites only)
    r = np.maximum(0.0, v - float(np.median(v)))
    out["pv_relu"] = _az((r[:, None] * A).sum(axis=0))
    # 3. sharpened rectified (contrast^2)
    r2 = r**2
    out["pv_relu2"] = _az((r2[:, None] * A).sum(axis=0))
    # 4. ridge least squares on centred axes
    Ac = A - A.mean(axis=0, keepdims=True)
    out["ridge"] = _az(np.linalg.solve(Ac.T @ Ac + 1e-3 * np.eye(3), Ac.T @ c))
    # 5. winner-take-all: brightest site axis
    out["argmax"] = _az(A[int(np.argmax(v))])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eye population vector → yaw lag")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--lamp-speed", type=float, default=0.08)
    p.add_argument("--lamp-angle", type=float, default=0.8)
    p.add_argument("--yaw-offset", type=float, default=2.2)
    p.add_argument("--yaw-grid", type=int, default=6)
    p.add_argument("--no-swing", action="store_true")
    p.add_argument("--variants", action="store_true", help="Compare candidate decodes")
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
        path,
        shell_devices=shell_dev,
        yaw_offset=float(args.yaw_offset),
        lamp_speed=float(args.lamp_speed),
        lamp_angle=float(args.lamp_angle),
        scope_on=False,
    )
    rt = CNSRuntime.open(path, blueprint=bp, shell_devices=shell_dev, keep_gains=False)
    if shell_dev and hasattr(rt, "_apply_device_snapshot"):
        rt._apply_device_snapshot(shell_dev)
        rt._sync_shell_fields_to_devices()
        rt._devices_dirty = True
    for attr in ("eye_left_on", "eye_right_on", "jo_left_on", "jo_right_on", "ocelli_on"):
        setattr(rt, attr, True)
    for d in getattr(rt, "device_list", []) or []:
        kind = getattr(d, "kind", "")
        if kind in ("eye", "jo", "ocelli"):
            d.enabled = True

    swing_on = not bool(args.no_swing)
    rt.world_swing_on = swing_on
    rt.world_swing_amp = 0.20 if swing_on else 0.0
    rt.world_swing_hz = 0.40 if swing_on else 0.0
    rt.world_swing_phase = float(getattr(rt, "world_swing_phase", 0.0) or 0.0)

    rt.learn_on = False
    rt.motor_socket_on = False
    rt.sim_running = True
    rt._activity_running = True

    base_yaw = float(getattr(rt, "turret_yaw", 0.0) or 0.0) + float(args.yaw_offset)
    grid = max(1, int(args.yaw_grid))
    yaws = [_wrap(base_yaw + k * (math.tau / grid)) for k in range(grid)]

    dt = 1.0 / 60.0
    est: list[float] = []
    true: list[float] = []
    conf: list[float] = []
    alt: list[dict[str, float]] = []
    yaws_seen: list[float] = []
    steps_per = max(1, int(args.steps) // len(yaws))
    for yi, yaw in enumerate(yaws):
        rt.turret_yaw = float(yaw) % math.tau
        for i in range(steps_per):
            rt.fly.sim_time = int(yi * steps_per + i)
            if float(rt.lamp_speed) > 0.0:
                rt._tick_lamp(dt)
            else:
                rt._update_lamp_xyz()
            if swing_on:
                rt.world_swing_phase = (
                    float(rt.world_swing_phase) + float(rt.world_swing_hz) * dt * math.tau
                ) % math.tau
                rt._update_lamp_xyz()
            rt._update_eye_sensors()
            rt._update_jo_sensors()
            rt._update_turret_aim_metric()
            az = eye_lamp_azimuth(rt)
            des = rt._turret_desired_aim()
            if az is None or des is None:
                continue
            est.append(_wrap(float(az[0]) - float(rt.turret_yaw)))
            conf.append(float(az[1]))
            true.append(_wrap(float(des[0]) - float(rt.turret_yaw)))
            if args.variants:
                alt.append(_alt_azimuths(rt))
                yaws_seen.append(float(rt.turret_yaw))

    e = np.asarray(est, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    c = np.asarray(conf, dtype=np.float64)
    if len(t) < 50:
        print("[probe] not enough samples", flush=True)
        return 2
    d = np.asarray([_wrap(float(x)) for x in (e - t)], dtype=np.float64)
    ok = (np.abs(t) >= math.radians(5.0))
    agree = float(np.mean((e[ok] > 0) == (t[ok] > 0))) if int(np.sum(ok)) else float("nan")
    near = np.abs(t) < math.radians(30.0)
    agree_n = float("nan")
    m_near = near & (np.abs(t) >= math.radians(2.0))
    if int(np.sum(m_near)) >= 20:
        agree_n = float(np.mean((e[m_near] > 0) == (t[m_near] > 0)))
    print("\n=== eye population vector → yaw lag ===", flush=True)
    print(
        f"samples={len(t)} yaws={len(yaws)} spin={args.lamp_speed} swing={swing_on}",
        flush=True,
    )
    print(
        f"sign agree (|lag|≥5°) = {agree:.0%} (n={int(np.sum(ok))})   "
        f"near cone (2°≤|lag|<30°) = {agree_n:.0%} (n={int(np.sum(m_near))})",
        flush=True,
    )
    print(
        f"bias={math.degrees(float(np.mean(d))):+.1f}°  "
        f"|err| med={math.degrees(float(np.median(np.abs(d)))):.1f}°  "
        f"p90={math.degrees(float(np.percentile(np.abs(d), 90))):.1f}°  "
        f"conf mean={float(np.mean(c)):.2f} min={float(np.min(c)):.2f}",
        flush=True,
    )
    corr = float(np.corrcoef(np.sin(e), np.sin(t))[0, 1])
    print(f"corr(sin est, sin true) = {corr:+.3f}", flush=True)

    if alt:
        yv = np.asarray(yaws_seen, dtype=np.float64)
        print("\ncandidate decodes (lag_est = az − turret_yaw):", flush=True)
        for key in alt[0]:
            a = np.asarray([x[key] for x in alt], dtype=np.float64)
            lag_a = np.asarray([_wrap(float(z)) for z in (a - yv)], dtype=np.float64)
            da = np.asarray([_wrap(float(z)) for z in (lag_a - t)], dtype=np.float64)
            ag = (
                float(np.mean((lag_a[ok] > 0) == (t[ok] > 0)))
                if int(np.sum(ok))
                else float("nan")
            )
            ag_n = (
                float(np.mean((lag_a[m_near] > 0) == (t[m_near] > 0)))
                if int(np.sum(m_near)) >= 20
                else float("nan")
            )
            print(
                f"  {key:9s} sign={ag:.0%} near={ag_n:.0%} "
                f"bias={math.degrees(float(np.mean(da))):+5.1f}° "
                f"|err|med={math.degrees(float(np.median(np.abs(da)))):5.1f}°",
                flush=True,
            )
    good = math.isfinite(agree) and agree >= 0.80
    print("VERDICT:", "OK — eye azimuth is a usable steer signal" if good else "WEAK", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Frozen plant: does head sensing (eye + JO + ocelli) predict yaw lag under Spin?

No plasticity. Turret pose fixed. Collects runtime device vectors only.
Reports per-modality and joint linear decode R² / sign agreement.

Usage:
  python scripts/probe_head_lag.py
  python scripts/probe_head_lag.py --lamp-speed 0.08 --steps 1800 --yaw-grid 4
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


def _wrap(a: float) -> float:
    return float((a + math.pi) % math.tau - math.pi)


def _wired(dev: dict) -> bool:
    groups = (dev.get("ocelli_groups") or []) + (dev.get("turret_motor_groups") or [])
    return any(isinstance(g, list) and len(g) > 0 for g in groups)


def _yaw_lag(rt: CNSRuntime) -> float:
    des = rt._turret_desired_aim() if hasattr(rt, "_turret_desired_aim") else None
    if des is None:
        return float("nan")
    return _wrap(float(des[0]) - float(getattr(rt, "turret_yaw", 0.0) or 0.0))


def _vec(rt: CNSRuntime, attr: str, n: int) -> np.ndarray:
    v = np.asarray(getattr(rt, attr, np.zeros(n)), dtype=np.float64).ravel()
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)))
    return v[:n].astype(np.float64)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sign_agree(a: np.ndarray, b: np.ndarray, *, a_min: float, b_min: float) -> tuple[int, int]:
    ok = (np.abs(a) >= a_min) & (np.abs(b) >= b_min) & np.isfinite(a) & np.isfinite(b)
    n = int(np.sum(ok))
    if n == 0:
        return 0, 0
    return int(np.sum((a[ok] > 0) == (b[ok] > 0))), n


def _lin_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    if len(y) < X.shape[1] + 2:
        return float("nan"), np.zeros(X.shape[1])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    return float(r2), np.asarray(w, dtype=np.float64)


def _pred_sign_rate(X: np.ndarray, y: np.ndarray, *, y_min: float) -> tuple[float, int]:
    """Leave-in linear decode: sign(pred) vs sign(y) on |y|≥y_min."""
    r2, w = _lin_r2(X, y)
    if not math.isfinite(r2):
        return float("nan"), 0
    pred = X @ w
    ok = (np.abs(y) >= y_min) & np.isfinite(pred) & np.isfinite(y)
    n = int(np.sum(ok))
    if n == 0:
        return float("nan"), 0
    return float(np.mean((pred[ok] > 0) == (y[ok] > 0))), n


def _report_block(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    Xb = np.column_stack([np.ones(len(y)), X]) if X.ndim == 2 else np.column_stack(
        [np.ones(len(y)), X.reshape(-1, 1)]
    )
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        Xb = np.column_stack([np.ones(len(y)), X])
    r2, w = _lin_r2(Xb, y)
    rate, n = _pred_sign_rate(Xb, y, y_min=math.radians(8.0))
    # Near-cone subset.
    near = np.abs(y) < math.radians(30.0)
    r2_n = float("nan")
    rate_n, n_n = float("nan"), 0
    if int(np.sum(near)) >= 40:
        r2_n, _ = _lin_r2(Xb[near], y[near])
        rate_n, n_n = _pred_sign_rate(Xb[near], y[near], y_min=math.radians(3.0))
    stds = [float(np.std(X[:, j])) for j in range(X.shape[1])]
    print(
        f"  {name:12s} dim={X.shape[1]:2d}  "
        f"R²={r2:.3f}  sign(pred)↔lag={rate:.0%} (n={n})  "
        f"near R²={r2_n:.3f} sign={rate_n if math.isfinite(rate_n) else float('nan'):.0%} "
        f"(n={n_n})  "
        f"σ=[{', '.join(f'{s:.3f}' for s in stds[:6])}{'…' if len(stds) > 6 else ''}]",
        flush=True,
    )
    return {
        "name": name,
        "dim": int(X.shape[1]),
        "r2": r2,
        "sign_rate": rate,
        "sign_n": n,
        "r2_near": r2_n,
        "sign_rate_near": rate_n,
        "w": w,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe head sensors → yaw lag under Spin")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--lamp-speed", type=float, default=0.08)
    p.add_argument("--lamp-angle", type=float, default=0.8)
    p.add_argument("--yaw-offset", type=float, default=2.2)
    p.add_argument("--yaw-grid", type=int, default=4)
    p.add_argument(
        "--swing",
        action="store_true",
        default=True,
        help="Enable world swing so JO tilt is non-constant (default on)",
    )
    p.add_argument("--no-swing", action="store_true", help="Disable world swing")
    p.add_argument("--log-every", type=int, default=300)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ui_path = ROOT / "data" / "processed" / "shell_ui.json"
    ui = json.loads(ui_path.read_text()) if ui_path.is_file() else {}
    path = Path(
        args.model
        or ui.get("last_model")
        or (ROOT / "data" / "flies" / "x.fly")
    )
    if not path.is_file():
        print(f"[probe] missing model: {path}", flush=True)
        return 2

    fly_meta = getattr(load_fly(path), "shell_meta", {}) or {}
    fly_dev = fly_meta.get("shell_devices") or {}
    ui_dev = ui.get("shell_devices") or {}
    shell_dev = fly_dev if _wired(fly_dev) else (ui_dev if _wired(ui_dev) else None)

    print(f"[probe] load {path}", flush=True)
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

    # Force head sensors on (match .fly flags if present).
    for attr in ("eye_left_on", "eye_right_on", "jo_left_on", "jo_right_on", "ocelli_on"):
        setattr(rt, attr, True)
    if shell_dev:
        for attr in ("eye_left_on", "eye_right_on", "jo_left_on", "jo_right_on"):
            if isinstance(shell_dev.get(attr), bool):
                setattr(rt, attr, bool(shell_dev[attr]))
    # Mirror onto typed devices in device_list.
    for d in getattr(rt, "device_list", []) or []:
        kind = getattr(d, "kind", "")
        name = getattr(d, "name", "")
        if kind == "ocelli":
            d.enabled = bool(rt.ocelli_on)
        elif kind == "eye":
            d.enabled = bool(getattr(rt, f"{name}_on", True))
        elif kind == "jo":
            d.enabled = bool(getattr(rt, f"{name}_on", True))

    swing_on = bool(args.swing) and not bool(args.no_swing)
    rt.world_swing_on = swing_on
    if not hasattr(rt, "world_swing_phase"):
        rt.world_swing_phase = 0.0
    if swing_on:
        rt.world_swing_amp = float(getattr(rt, "world_swing_amp", 0.0) or 0.20)
        if rt.world_swing_amp < 0.05:
            rt.world_swing_amp = 0.20
        rt.world_swing_hz = float(getattr(rt, "world_swing_hz", 0.0) or 0.40)
        if rt.world_swing_hz < 0.05:
            rt.world_swing_hz = 0.40
    else:
        rt.world_swing_amp = 0.0
        rt.world_swing_hz = 0.0

    nf = rt.instant_fuse()
    rt.fly.contact_gain = {}
    rt.learn_on = False
    rt.motor_socket_on = False
    rt.rule.teacher_mix = 0.0
    rt.rule.clone_teacher = False
    rt.sim_running = True
    rt._activity_running = True

    print(
        f"[probe] fuse={nf} "
        f"ocelli={rt._ocelli_connected_mask()} "
        f"eyeL={rt._eye_connected_mask('eye_left') if hasattr(rt, '_eye_connected_mask') else '?'} "
        f"joL={rt._jo_connected_mask('jo_left') if hasattr(rt, '_jo_connected_mask') else '?'} "
        f"learn=OFF gains=0 spin={args.lamp_speed} swing={swing_on}",
        flush=True,
    )

    base_yaw = float(getattr(rt, "turret_yaw", 0.0) or 0.0) + float(args.yaw_offset)
    if int(args.yaw_grid) > 1:
        yaws = [
            _wrap(base_yaw + k * (math.tau / int(args.yaw_grid)))
            for k in range(int(args.yaw_grid))
        ]
    else:
        yaws = [base_yaw]

    dt = 1.0 / 60.0
    # Columns: oc(3) | eyeL(6) | eyeR(6) | joL(1) | joR(1) | lag | yaw
    feats: list[np.ndarray] = []
    lags: list[float] = []

    steps_per = max(1, int(args.steps) // len(yaws))
    for yi, yaw in enumerate(yaws):
        rt.turret_yaw = float(yaw)
        rt.turret_pitch = float(getattr(rt, "turret_pitch", 0.0) or 0.0)
        for i in range(steps_per):
            rt.fly.sim_time = int(yi * steps_per + i)
            if float(rt.lamp_speed) > 0.0:
                rt._tick_lamp(dt)
            else:
                rt._update_lamp_xyz()
            # Advance swing phase if host ticks it with lamp / frame.
            if swing_on and hasattr(rt, "_tick_world_swing"):
                rt._tick_world_swing(dt)
            elif swing_on and hasattr(rt, "world_swing_phase"):
                hz = float(getattr(rt, "world_swing_hz", 0.0) or 0.0)
                rt.world_swing_phase = float(rt.world_swing_phase) + float(hz) * dt * math.tau

            rt._update_ocelli_sensors()
            if hasattr(rt, "_update_eye_sensors"):
                rt._update_eye_sensors()
            if hasattr(rt, "_update_jo_sensors"):
                rt._update_jo_sensors()
            rt._update_turret_aim_metric()

            oc = _vec(rt, "ocelli_I", 3)
            el = _vec(rt, "eye_left_I", 6)
            er = _vec(rt, "eye_right_I", 6)
            jl = float(getattr(rt, "jo_left_tilt", 0.0) or 0.0)
            jr = float(getattr(rt, "jo_right_tilt", 0.0) or 0.0)
            lag = _yaw_lag(rt)
            feats.append(np.concatenate([oc, el, er, [jl, jr]]))
            lags.append(float(lag))

            if i % max(1, int(args.log_every)) == 0:
                di = float(oc[1] - oc[2])
                print(
                    f"[probe] yaw={yaw:+.2f} t={i:4d} "
                    f"ocΔI={di:+.3f} "
                    f"eyeL=[{el[0]:.2f},{el[1]:.2f}…] "
                    f"JO={jl:+.2f} lag={lag * 180 / math.pi:+.1f}°",
                    flush=True,
                )

    F = np.asarray(feats, dtype=np.float64)
    y = np.asarray(lags, dtype=np.float64)
    oc, el, er = F[:, 0:3], F[:, 3:9], F[:, 9:15]
    jo = F[:, 15:17]
    dI = oc[:, 1] - oc[:, 2]
    # Eye L/R azimuth contrast (NE−NW style): mean left-hemisphere sites vs right.
    # Sites: NE SE SW NW U D — use (NE+SE) − (NW+SW) on each eye, then L−R.
    eye_az_l = (el[:, 0] + el[:, 1]) - (el[:, 2] + el[:, 3])
    eye_az_r = (er[:, 0] + er[:, 1]) - (er[:, 2] + er[:, 3])
    eye_lr = eye_az_l - eye_az_r

    print("\n=== head sensors → lag (frozen, turret fixed) ===", flush=True)
    print(
        f"samples={len(y)} yaws={len(yaws)} spin={args.lamp_speed} swing={swing_on}",
        flush=True,
    )
    print(
        f"Pearson: ΔI_oc↔lag={_pearson(dI, y):+.3f}  "
        f"eye_lr↔lag={_pearson(eye_lr, y):+.3f}  "
        f"joL↔lag={_pearson(jo[:, 0], y):+.3f}",
        flush=True,
    )

    results = []
    print("per modality:", flush=True)
    results.append(_report_block("ocelli", oc, y))
    results.append(_report_block("ocelli_ΔI", dI, y))
    results.append(_report_block("eye_L", el, y))
    results.append(_report_block("eye_R", er, y))
    results.append(_report_block("eye_L+R", np.concatenate([el, er], axis=1), y))
    results.append(_report_block("eye_lr", eye_lr, y))
    results.append(_report_block("JO", jo, y))
    results.append(_report_block("head_all", F, y))

    # Best non-trivial block for verdict (skip if σ≈0).
    usable = [
        r
        for r in results
        if math.isfinite(r["r2"]) and r["sign_n"] >= 50
    ]
    best = max(usable, key=lambda r: (r["r2"], r["sign_rate"] or 0.0)) if usable else None

    print("\n=== VERDICT ===", flush=True)
    if best is None:
        print("INCONCLUSIVE — not enough informative samples", flush=True)
        return 2
    print(
        f"best block: {best['name']}  R²={best['r2']:.3f}  "
        f"sign={best['sign_rate']:.0%}  near_R²={best['r2_near']:.3f}",
        flush=True,
    )
    head = next(r for r in results if r["name"] == "head_all")
    if head["r2"] < 0.15 and (head["sign_rate"] or 0.0) < 0.60:
        print(
            "FAIL — head vector (eye+JO+ocelli) does not carry yaw lag; "
            "fix device geometry/mount before Trainer",
            flush=True,
        )
        code = 1
    elif head["r2"] < 0.30 or (head["sign_rate"] or 0.0) < 0.70:
        print(
            "WEAK — head sensing poorly predicts lag; learning from these "
            "devices will stay brittle until geometry improves",
            flush=True,
        )
        code = 0
    else:
        print(
            "OK — head sensors predict lag enough to train a sensory→motor policy",
            flush=True,
        )
        code = 0
    print(
        f"lag_deg mean={float(np.mean(y) * 180 / math.pi):.1f} "
        f"std={float(np.std(y) * 180 / math.pi):.1f} "
        f"|lag|med={float(np.median(np.abs(y)) * 180 / math.pi):.1f}",
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())

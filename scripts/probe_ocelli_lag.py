#!/usr/bin/env python3
"""Frozen plant: does ocelli I predict yaw lag under Spin?

No plasticity. Turret pose fixed (sensing geometry only).
Reports sign agreement and linear decode R² of lag from I / ΔI.

Usage:
  python scripts/probe_ocelli_lag.py
  python scripts/probe_ocelli_lag.py --lamp-speed 0.08 --steps 1800
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
    agree = int(np.sum((a[ok] > 0) == (b[ok] > 0)))
    return agree, n


def _lin_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares y ≈ X @ w (with bias col). Returns R² and weights."""
    if len(y) < X.shape[1] + 2:
        return float("nan"), np.zeros(X.shape[1])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    return float(r2), w


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe ocelli I → yaw lag under Spin")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--lamp-speed", type=float, default=0.08)
    p.add_argument("--lamp-angle", type=float, default=0.8)
    p.add_argument(
        "--yaw-offset",
        type=float,
        default=2.2,
        help="Fixed barrel yaw offset (rad); turret does not slew",
    )
    p.add_argument(
        "--yaw-grid",
        type=int,
        default=4,
        help="Also sample this many fixed barrel yaws (0 = only --yaw-offset)",
    )
    p.add_argument("--log-every", type=int, default=200)
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

    nf = rt.instant_fuse()
    rt.fly.contact_gain = {}
    rt.learn_on = False
    rt.motor_socket_on = False
    rt.rule.teacher_mix = 0.0
    rt.rule.clone_teacher = False
    rt.sim_running = True
    rt._activity_running = True
    print(
        f"[probe] fuse={nf} ocelli={rt._ocelli_connected_mask()} "
        f"turret={rt._turret_connected_mask()} "
        f"learn=OFF socket=OFF gains=0 spin={args.lamp_speed}",
        flush=True,
    )

    # Fixed barrel poses (no actuate — pure sensor vs geometry).
    base_yaw = float(getattr(rt, "turret_yaw", 0.0) or 0.0) + float(args.yaw_offset)
    if int(args.yaw_grid) > 1:
        yaws = [
            _wrap(base_yaw + k * (math.tau / int(args.yaw_grid)))
            for k in range(int(args.yaw_grid))
        ]
    else:
        yaws = [base_yaw]

    dt = 1.0 / 60.0
    rows: list[tuple[float, float, float, float, float, float]] = []
    # (Im, Il, Ir, dI, lag, yaw)

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
            rt._update_ocelli_sensors()
            rt._update_turret_aim_metric()
            I = np.asarray(rt.ocelli_I, dtype=np.float64)
            if len(I) < 3:
                I = np.pad(I, (0, 3 - len(I)))
            di = float(I[1] - I[2])
            lag = _yaw_lag(rt)
            rows.append(
                (float(I[0]), float(I[1]), float(I[2]), di, float(lag), float(yaw))
            )
            if i % max(1, int(args.log_every)) == 0:
                print(
                    f"[probe] yaw={yaw:+.2f} t={i:4d} "
                    f"I=[{I[0]:.2f},{I[1]:.2f},{I[2]:.2f}] "
                    f"ΔI={di:+.3f} lag={lag * 180 / math.pi:+.1f}°",
                    flush=True,
                )

    arr = np.asarray(rows, dtype=np.float64)
    Im, Il, Ir, dI, lag, yaw_a = arr.T
    lag_deg = lag * 180.0 / math.pi

    print("\n=== I → lag (frozen, turret fixed, Spin) ===", flush=True)
    print(f"samples={len(rows)} yaws={len(yaws)} spin={args.lamp_speed}", flush=True)

    # Chance baseline for sign: 50%.
    a_di, n_di = _sign_agree(dI, lag, a_min=0.05, b_min=math.radians(8.0))
    print(
        f"sign ΔI ↔ lag: {a_di}/{n_di} = {a_di / max(n_di, 1):.0%}  "
        f"(chance ~50%)",
        flush=True,
    )
    # Median-cup vs mean of L/R as another simple feature.
    d_med = Im - 0.5 * (Il + Ir)
    a_m, n_m = _sign_agree(d_med, np.abs(lag), a_min=0.02, b_min=math.radians(8.0))
    # |lag| vs median excess is not directional — skip; use L−R only for sign.

    r_di = _pearson(dI, lag)
    r_il = _pearson(Il, lag)
    r_ir = _pearson(Ir, lag)
    r_im = _pearson(Im, lag)
    print(
        f"Pearson: ΔI↔lag={r_di:+.3f}  Il↔lag={r_il:+.3f}  "
        f"Ir↔lag={r_ir:+.3f}  Im↔lag={r_im:+.3f}",
        flush=True,
    )

    # Linear decode lag from [1, Im, Il, Ir] and from [1, ΔI].
    y = lag
    X3 = np.column_stack([np.ones(len(y)), Im, Il, Ir])
    X1 = np.column_stack([np.ones(len(y)), dI])
    r2_3, w3 = _lin_r2(X3, y)
    r2_1, w1 = _lin_r2(X1, y)
    print(
        f"linear R² lag←[Im,Il,Ir]={r2_3:.3f}  w={np.round(w3, 3).tolist()}",
        flush=True,
    )
    print(
        f"linear R² lag←ΔI={r2_1:.3f}  w={np.round(w1, 3).tolist()}",
        flush=True,
    )

    # Near-cone subset (|lag|<30°) — where Track learning needs the signal.
    near = np.abs(lag) < math.radians(30.0)
    if int(np.sum(near)) >= 40:
        a_n, n_n = _sign_agree(dI[near], lag[near], a_min=0.03, b_min=math.radians(3.0))
        r2_n, _ = _lin_r2(X3[near], y[near])
        print(
            f"near |lag|<30°: n={int(np.sum(near))}  "
            f"sign ΔI↔lag={a_n}/{n_n}={a_n / max(n_n, 1):.0%}  "
            f"R²[I]={r2_n:.3f}",
            flush=True,
        )
    else:
        print(f"near |lag|<30°: n={int(np.sum(near))} (too few)", flush=True)

    # Verdict.
    rate = a_di / max(n_di, 1)
    if n_di < 50:
        verdict = "INCONCLUSIVE (too few informative samples)"
    elif rate < 0.60 and (not math.isfinite(r2_3) or r2_3 < 0.15):
        verdict = "FAIL — I does not carry yaw lag; fix sensing/geometry before Trainer"
    elif rate < 0.70 or (math.isfinite(r2_3) and r2_3 < 0.30):
        verdict = "WEAK — I poorly predicts lag; learning from ΔI will stay brittle"
    else:
        verdict = "OK — I predicts lag enough to train a sensory→motor policy"
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(
        f"lag_deg mean={float(np.mean(lag_deg)):.1f} "
        f"std={float(np.std(lag_deg)):.1f} "
        f"|lag|med={float(np.median(np.abs(lag_deg))):.1f}",
        flush=True,
    )
    return 0 if "OK" in verdict or "WEAK" in verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

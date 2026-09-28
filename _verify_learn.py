#!/usr/bin/env python3
"""Verify eligibility Learn: reachability + stable Δw + ε trend (CNSRuntime)."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from blueprint import Blueprint
from cns_runtime import CNSRuntime

ROOT = Path("/home/vpavlov/work/G2/FlyWire")
N = 300


def main() -> None:
    t0 = time.perf_counter()
    path = ROOT / "data/flies/x.fly"
    bp = Blueprint.from_fly_meta(path, yaw_offset=2.2, scope_on=True)
    rt = CNSRuntime.open(path, blueprint=bp, keep_gains=False)
    print(f"[learn] load {path.name} n={len(rt.fly.neurons)}")
    nf = rt.instant_fuse()
    print(
        f"[learn] fuse pairs={nf} ocelli={rt._ocelli_connected_mask()} "
        f"turret={rt._turret_connected_mask()} recipe={len(rt.fly.recipe_pre):,} "
        f"hop={len(rt.fly.activity_graph_edges()[0]):,}"
    )
    rt.sync_training_scope()
    reach = rt._probe_ocelli_motor_reach(hops=10)
    print(
        f"[learn] REACH hit={reach['hit']}/{reach['motors']} "
        f"best={reach['best']:.3f} ok={reach['ok']}"
    )
    rt.prepare_activity()
    dt = 1.0 / 60.0
    rows = []
    for i in range(N):
        if i == 0:
            rt._update_lamp_xyz()
        info = rt.tick(dt)
        if i % 30 == 0 or i == N - 1:
            err = float(rt.turret_aim_err)
            ed = err * 180 / math.pi if math.isfinite(err) else float("nan")
            gains = rt.fly.contact_gain or {}
            nan_g = sum(1 for w in gains.values() if not math.isfinite(float(w)))
            mean_abs = 0.0
            if gains:
                vals = [
                    abs(float(w) - 1.0) for w in gains.values() if math.isfinite(float(w))
                ]
                mean_abs = float(np.mean(vals)) if vals else 0.0
            rows.append((i, ed, float(rt.learn_dw_ema), len(gains), nan_g, mean_abs))
            print(
                f"[learn] t={i:4d} ε={ed:6.1f}° Δw={rt.learn_dw_ema:.4f} "
                f"gains={len(gains)} nan={nan_g} |g-1|={mean_abs:.4f} "
                f"plast={rt._learn_flash_n} touch={info.get('touched')}",
                flush=True,
            )
    early = [r[1] for r in rows if r[0] < 60 and math.isfinite(r[1])]
    late = [r[1] for r in rows if r[0] >= N - 60 and math.isfinite(r[1])]
    e0 = float(np.mean(early)) if early else float("nan")
    e1 = float(np.mean(late)) if late else float("nan")
    stable = all(r[4] == 0 for r in rows) and all(math.isfinite(r[2]) for r in rows)
    print(
        f"\n=== VERDICT === reach_ok={reach['ok']} stable={stable} "
        f"ε {e0:.1f}°→{e1:.1f}° ({e1 - e0:+.1f}°) "
        f"elapsed={time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()

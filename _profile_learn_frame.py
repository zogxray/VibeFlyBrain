#!/usr/bin/env python3
"""Short learn-frame microbench after dense-elig / rev-adj / merge-flash / graph-cache patches."""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from blueprint import Blueprint
from cns_runtime import CNSRuntime
from loop import ClosedLoop


def ms(fn, n=8, warm=2) -> float:
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def main() -> int:
    path = Path("data/flies/x.fly")
    bp = Blueprint.from_fly_meta(path, yaw_offset=2.2, scope_on=True, hop_clip=True)
    rt = CNSRuntime.open(path, blueprint=bp, keep_gains=False)
    rt.instant_fuse()
    rt.apply_training_scope()
    rt.prepare_activity()
    rt.learn_on = True
    rt.turret_on = True
    rt.ocelli_on = True
    rt.loop_hops = 1
    rt.learn_every = 1
    loop: ClosedLoop = rt._closed_loop
    loop.trainer = rt.trainer()
    dt = 1.0 / 60.0

    for _ in range(8):
        loop.step(dt)

    fly = rt.fly
    print(f"n={len(fly.neurons):,} hop_edges={len(fly.activity_graph_edges()[0]):,}")
    print(
        f"alive={(fly.soma_activity > 0).sum():,} "
        f"gains={len(fly.contact_gain):,} "
        f"scope_on={rt.scope_on} hop_clip={rt.hop_clip}"
    )
    print(f"loop_hops={rt.loop_hops}")

    rows: list[tuple[str, float]] = []

    rows.append(
        (
            "tick_activity",
            ms(lambda: fly.tick_activity(hop_gain=0.85, decay=0.90), n=8, warm=2),
        )
    )

    def trainer_once():
        prev = float(getattr(rt, "turret_aim_err", float("nan")))
        loop.trainer.step(prev)

    rows.append(("Trainer.step", ms(trainer_once, n=4, warm=1)))

    def merge_once():
        rt._merge_learn_chain_flash(hop_pre=None, hop_post=None)

    rows.append(("_merge_learn_chain_flash", ms(merge_once, n=8, warm=2)))

    # ClosedLoop.step avg for learn_every=1 and =4
    rt.learn_every = 1
    for _ in range(2):
        loop.step(dt)
    rows.append(("ClosedLoop.step le=1", ms(lambda: loop.step(dt), n=4, warm=1)))

    rt.learn_every = 4
    for _ in range(2):
        loop.step(dt)
    rows.append(("ClosedLoop.step le=4", ms(lambda: loop.step(dt), n=8, warm=2)))

    # motor_eligibility_upstream: 1st cold-ish, 2nd should hit reverse-adj cache
    targets = {}
    if hasattr(rt, "_turret_motor_neuron_targets"):
        targets = rt._turret_motor_neuron_targets()
    motor_ids = set(int(k) for k in targets.keys())
    depth = int(getattr(getattr(rt, "rule", None), "elig_depth", 8) or 8)

    # Invalidate reverse adj so first call rebuilds; second hits cache.
    if hasattr(fly, "invalidate_activity_graph_cache"):
        # Keep graph cache; only clear reverse adj if exposed.
        fly._elig_rev_adj = None
        fly._elig_rev_key = None
    elif hasattr(fly, "_elig_rev_adj"):
        fly._elig_rev_adj = None
        fly._elig_rev_key = None

    t0 = time.perf_counter()
    up1 = rt._motor_eligibility_upstream(motor_ids, depth=depth)
    t_cold = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    up2 = rt._motor_eligibility_upstream(motor_ids, depth=depth)
    t_hot = (time.perf_counter() - t0) * 1000.0
    rows.append(("motor_elig_upstream 1st", t_cold))
    rows.append(("motor_elig_upstream 2nd (cached)", t_hot))

    print(f"\nupstream_n={len(up1)} (2nd={len(up2)})")
    print("\n=== learn-frame microbench (ms) ===")
    print(f"{'#':>2}  {'op':40s}  {'ms':>8}")
    for i, (name, val) in enumerate(rows, 1):
        print(f"{i:2d}  {name:40s}  {val:8.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""One-tick plasticity autopsy: graph TrainingScope corridor / PATH_CREDIT."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blueprint import Blueprint
from cns_runtime import CNSRuntime


def main() -> int:
    path = ROOT / "data" / "flies" / "x.fly"
    bp = Blueprint.from_fly_meta(path, yaw_offset=2.2, scope_on=True, hop_clip=True)
    rt = CNSRuntime.open(path, blueprint=bp, keep_gains=False)
    rt.instant_fuse()
    rt.apply_training_scope()
    sc = rt._training_scope
    print(f"[cns] {sc.readout()}", flush=True)
    print(
        f"[cns] hop_allow={len(rt.fly.hop_allow_ids)} "
        f"hop_clip={rt.hop_clip} scope_on={rt.scope_on}",
        flush=True,
    )
    rt.prepare_activity()
    for _ in range(int(rt.rule.hops)):
        rt.fly.tick_activity(hop_gain=0.9, decay=0.97)
    rt.fly.rebuild_activity_cache()
    rt._plasticity_aim_step(float(rt.turret_aim_err))
    print(
        f"[cns] after plast gains={len(rt.fly.contact_gain or {})} "
        f"dw_cor={rt.count_dw_corridor_edges()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

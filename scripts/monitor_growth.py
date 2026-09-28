#!/usr/bin/env python3
"""Headless growth run with console progress log.

Usage:
  python scripts/monitor_growth.py
  python scripts/monitor_growth.py --sources ascending,descending --max-edges 4000
  python scripts/monitor_growth.py --ticks 2000 --log-every 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_model import (  # noqa: E402
    STATUS_CONNECTED,
    STATUS_FAILED,
    STATUS_GROWING,
    STATUS_SEEKING,
    load_fly_from_malecns,
)

SOMAS = ROOT / "data" / "processed" / "malecns_somas.npz"
PROCS = ROOT / "data" / "processed" / "malecns_processes.npz"
EDGES = ROOT / "data" / "processed" / "malecns_edges.npz"


def _neck_stats(fly) -> dict:
    """How many polylines span the gap between the two tissue modes."""
    if not fly._ensure_tissue_modes() or len(fly.growth_owner) == 0:
        return {"cross_armed": 0, "neck_poly": 0, "modes": 0}
    centers = fly._tissue_mode_centers
    axis = centers[1] - centers[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return {"cross_armed": 0, "neck_poly": 0, "modes": len(centers)}
    axis = axis / norm
    labels = fly._tissue_mode_labels
    pts = fly._process_tree_xyz
    m0 = float((pts[labels == 0] @ axis).mean())
    m1 = float((pts[labels == 1] @ axis).mean())
    mid = 0.5 * (m0 + m1)
    gap = 0.08 * fly._soma_span()
    soma_m = fly._point_modes(fly.growth_soma)
    goal_m = fly._point_modes(fly.growth_goal_xyz)
    cross_armed = int(np.sum((soma_m >= 0) & (goal_m >= 0) & (soma_m != goal_m)))
    neck = 0
    for i in range(len(fly.growth_owner)):
        k = int(fly.growth_poly_n[i])
        if k < 2:
            continue
        pr = fly.growth_poly[i, :k] @ axis
        if float(pr.min()) < mid - gap and float(pr.max()) > mid + gap:
            neck += 1
    return {"cross_armed": cross_armed, "neck_poly": neck, "modes": len(centers)}


def _mode_breakdown(fly) -> dict:
    """Connected / failed / open split by same-mode vs cross-mode tips."""
    empty = {
        "same": {"n": 0, "ok": 0, "fail": 0, "open": 0},
        "cross": {"n": 0, "ok": 0, "fail": 0, "open": 0},
    }
    if len(fly.growth_owner) == 0 or not fly._ensure_tissue_modes():
        return empty
    soma_m = fly._point_modes(fly.growth_soma)
    goal_m = fly._point_modes(fly.growth_goal_xyz)
    cross = (soma_m >= 0) & (goal_m >= 0) & (soma_m != goal_m)
    same = (soma_m >= 0) & (goal_m >= 0) & (soma_m == goal_m)
    st = fly.growth_status
    out = {}
    for key, mask in (("same", same), ("cross", cross)):
        if not np.any(mask):
            out[key] = {"n": 0, "ok": 0, "fail": 0, "open": 0}
            continue
        sub = st[mask]
        out[key] = {
            "n": int(np.sum(mask)),
            "ok": int(np.sum(sub == STATUS_CONNECTED)),
            "fail": int(np.sum(sub == STATUS_FAILED)),
            "open": int(
                np.sum((sub == STATUS_GROWING) | (sub == STATUS_SEEKING))
            ),
        }
    return out


def _log_line(fly, *, t0: float, prefix: str = "") -> None:
    st = fly.growth_status
    n = len(st)
    conn = int(np.sum(st == STATUS_CONNECTED)) if n else 0
    fail = int(np.sum(st == STATUS_FAILED)) if n else 0
    act = (
        int(np.sum((st == STATUS_GROWING) | (st == STATUS_SEEKING))) if n else 0
    )
    yld = fly.recipe_yield()
    neck = _neck_stats(fly)
    modes = _mode_breakdown(fly)
    multi_ax = sum(1 for n in fly.neurons.values() if len(n.axons) > 1)
    multi_den = sum(1 for n in fly.neurons.values() if len(n.dendrites) > 1)
    elapsed = time.perf_counter() - t0
    q = int(fly.genome_quota.sum()) if fly.genome_quota.size else 0
    sm, cr = modes["same"], modes["cross"]
    sm_pct = 100.0 * sm["ok"] / sm["n"] if sm["n"] else 0.0
    cr_pct = 100.0 * cr["ok"] / cr["n"] if cr["n"] else 0.0
    print(
        f"{prefix}"
        f"t={fly.sim_time:5d}  "
        f"conn={conn:5d}  act={act:5d}  fail={fail:5d}  "
        f"yield={yld['pct']:5.1f}% {yld['verdict']:<8}  "
        f"same={sm['ok']}/{sm['n']} ({sm_pct:4.1f}%)  "
        f"cross={cr['ok']}/{cr['n']} ({cr_pct:4.1f}%)  "
        f"neck={neck['neck_poly']:4d}  "
        f"trails={fly.growth_trail_point_count():7d}  "
        f"multiA={multi_ax} multiD={multi_den}  "
        f"quota={q}  {elapsed:6.1f}s",
        flush=True,
    )


def _print_class_yield(fly) -> None:
    yld = fly.recipe_yield()
    print("---- per-class Out% (axons from group) / In% (aimed at group) ----", flush=True)
    print(
        f"{'class':22} {'out_ok':>7} {'out_arm':>7} {'out%':>6}  "
        f"{'in_ok':>6} {'in_arm':>7} {'in%':>6}  open_out",
        flush=True,
    )
    rows = sorted(
        yld.get("by_class", []),
        key=lambda r: (-r.get("out_armed", 0), r.get("name", "")),
    )
    for row in rows:
        if not row.get("out_armed") and not row.get("in_armed"):
            continue
        name = row.get("name", "?")
        print(
            f"{name:22} {row.get('out_ok', 0):7d} {row.get('out_armed', 0):7d} "
            f"{row.get('out_pct', 0):5.1f}%  "
            f"{row.get('in_ok', 0):6d} {row.get('in_armed', 0):7d} "
            f"{row.get('in_pct', 0):5.1f}%  "
            f"{row.get('out_open', 0):5d}",
            flush=True,
        )


def _print_mode_final(fly) -> None:
    modes = _mode_breakdown(fly)
    print("---- same-mode vs cross-mode ----", flush=True)
    for key in ("same", "cross"):
        m = modes[key]
        pct = 100.0 * m["ok"] / m["n"] if m["n"] else 0.0
        fail_pct = 100.0 * m["fail"] / m["n"] if m["n"] else 0.0
        print(
            f"  {key:5}  n={m['n']:5d}  ok={m['ok']:5d} ({pct:5.1f}%)  "
            f"fail={m['fail']:5d} ({fail_pct:5.1f}%)  open={m['open']:5d}",
            flush=True,
        )
    neck = _neck_stats(fly)
    print(
        f"  neck_poly spanning gap: {neck['neck_poly']} / {neck['cross_armed']} cross-armed",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        default="",
        help="Comma-separated source sets (default: all)",
    )
    ap.add_argument(
        "--targets",
        default="",
        help="Comma-separated target sets (default: all)",
    )
    ap.add_argument("--max-edges", type=int, default=20000)
    ap.add_argument("--ticks", type=int, default=3000, help="Max sim ticks")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    print("=== MaleCNS growth monitor ===", flush=True)
    print(f"somas={SOMAS.name} procs={PROCS.name} edges={EDGES.name}", flush=True)
    t_load = time.perf_counter()
    fly = load_fly_from_malecns(SOMAS, PROCS, EDGES)
    print(
        f"loaded neurons={len(fly.neurons):,}  "
        f"recipe={len(fly.recipe_pre):,}  "
        f"processes={len(fly.atlas_process_xyz):,}  "
        f"({time.perf_counter() - t_load:.1f}s)",
        flush=True,
    )
    if fly.genome_quota.size == 0 or int(fly.genome_quota.sum()) == 0:
        nq = int(fly.rebuild_genome_from_atlas().sum())
        print(f"genome rebuilt quota={nq:,}", flush=True)
    else:
        print(f"genome quota={int(fly.genome_quota.sum()):,}", flush=True)

    sources = (
        [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.sources
        else list(fly.class_names)
    )
    targets = (
        [s.strip() for s in args.targets.split(",") if s.strip()]
        if args.targets
        else list(fly.class_names)
    )
    print(f"arming sources={sources} targets={targets} max={args.max_edges}", flush=True)
    t0 = time.perf_counter()
    armed = fly.ensure_growth_tips(sources, targets, max_edges=args.max_edges)
    print(f"armed tips={armed:,}", flush=True)
    if armed == 0:
        print("ERROR: nothing armed — check sets / recipe", flush=True)
        return 1

    fly._ensure_tissue_modes()
    neck0 = _neck_stats(fly)
    print(
        f"tissue modes={neck0['modes']}  "
        f"cross-mode tips={neck0['cross_armed']}  "
        f"centers=\n{fly._tissue_mode_centers}",
        flush=True,
    )
    print(
        "---- log: t  connected  active  fail  yield  neck_poly/cross  trails  multi ----",
        flush=True,
    )
    _log_line(fly, t0=t0, prefix="START ")

    for step in range(1, args.ticks + 1):
        fly.sim_tick()
        done = (
            int(
                np.sum(
                    (fly.growth_status == STATUS_GROWING)
                    | (fly.growth_status == STATUS_SEEKING)
                )
            )
            == 0
        )
        if step % args.log_every == 0 or done:
            _log_line(fly, t0=t0)
        if done:
            print("=== settled ===", flush=True)
            break
    else:
        print("=== tick budget exhausted ===", flush=True)

    yld = fly.recipe_yield()
    neck = _neck_stats(fly)
    print(
        f"FINAL yield={yld['pct']:.1f}% {yld['verdict']}  "
        f"ok={yld['ok']} fail={yld['fail']} open={yld['open']}  "
        f"neck_poly={neck['neck_poly']}  "
        f"trails={fly.growth_trail_point_count()}  "
        f"elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    _print_mode_final(fly)
    _print_class_yield(fly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Would real synapse counts lateralise the eye→motor wave?

The hop rule is ``post = max(post, gain * w * pre)`` with ``w = 1`` for every
atlas edge, so the level at a motor clamp is a pure function of BFS depth — both
eyes sit at the same depth and the sides cancel. This script replays the same
rule with ``w`` taken from the connectome synapse counts (max-product Dijkstra)
and prints the predicted per-site levels, before touching the engine.
"""

from __future__ import annotations

import heapq
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from diag_eye_lateral_common import boot  # noqa: E402


def best_product(adj_w, srcs, gain=0.9):
    """max over paths of prod(gain*w) — the level max-prop would settle on."""
    best: dict[int, float] = {int(s): 1.0 for s in srcs}
    heap = [(-1.0, int(s)) for s in srcs]
    heapq.heapify(heap)
    while heap:
        negv, u = heapq.heappop(heap)
        v = -negv
        if v < best.get(u, 0.0) - 1e-15:
            continue
        for w, t in adj_w.get(u, ()):
            nv = v * gain * w
            if nv < 1e-6:
                continue
            if nv > best.get(t, 0.0) + 1e-15:
                best[t] = nv
                heapq.heappush(heap, (-nv, t))
    return best


def main() -> int:
    rt = boot(scope=True)
    fly = rt.fly
    pre, post = fly.activity_graph_edges()
    n_atlas = len(fly.soma_cls)

    # Recipe synapse counts keyed by (pre, post) for the atlas edges in play.
    rp = np.asarray(fly.recipe_pre, dtype=np.int64)
    rq = np.asarray(fly.recipe_post, dtype=np.int64)
    rw = np.asarray(fly.recipe_weight, dtype=np.float64)
    key = (rp << 32) | (rq & 0xFFFFFFFF)
    order = np.argsort(key)
    key_s, rw_s = key[order], rw[order]

    xyz = np.asarray(fly.soma_xyz, dtype=np.float64)
    mid = float(np.median(xyz[:, 0]))

    p64 = pre.astype(np.int64)
    q64 = post.astype(np.int64)
    k = (p64 << 32) | (q64 & 0xFFFFFFFF)
    j = np.searchsorted(key_s, k)
    j = np.clip(j, 0, len(key_s) - 1)
    hit = key_s[j] == k
    syn = np.where(hit, rw_s[j], 0.0)
    print(f"edges={len(pre):,} with synapse count={int(hit.sum()):,} "
          f"(rest = grown fuse bridges)")
    s = syn[hit]
    print(f"syn: min={s.min():.0f} med={np.median(s):.0f} p90={np.percentile(s,90):.0f} max={s.max():.0f}")

    both = hit & (p64 < n_atlas) & (q64 < n_atlas)
    cross = (xyz[p64[both], 0] < mid) != (xyz[q64[both], 0] < mid)
    print(f"crossing edges {cross.mean():.1%}  syn med cross={np.median(syn[both][cross]):.0f} "
          f"same={np.median(syn[both][~cross]):.0f}")

    m_terms = [int(t) for t in (getattr(rt, "turret_motor_terminals", []) or [])]
    eyes = {}
    for d in getattr(rt, "device_list", []) or []:
        if getattr(d, "kind", "") == "eye":
            eyes[str(d.name)] = [int(t) for t in d.terminals]

    for label, norm in (
        ("w=1 (today)", None),
        ("w=clip(syn/p90,0.25,1)", ("p90", 0.25)),
        ("w=clip(syn/p90,0.05,1)", ("p90", 0.05)),
        ("w=clip(syn/med,0.05,1)", ("med", 0.05)),
    ):
        if norm is None:
            w = np.ones(len(pre), dtype=np.float64)
        else:
            ref = np.percentile(s, 90) if norm[0] == "p90" else np.median(s)
            w = np.clip(syn / float(ref), norm[1], 1.0)
            w[~hit] = 1.0  # grown device bridges stay unit
        adj_w: dict[int, list[tuple[float, int]]] = {}
        for a, b, ww in zip(pre.tolist(), post.tolist(), w.tolist()):
            adj_w.setdefault(int(a), []).append((float(ww), int(b)))
        rows = {}
        for name, terms in sorted(eyes.items()):
            best = best_product(adj_w, terms)
            rows[name] = [best.get(t, 0.0) for t in m_terms]
        a = np.asarray(rows["eye_left"])
        b = np.asarray(rows["eye_right"])
        ya = 0.5 * (a[0] + a[1]) - 0.5 * (a[2] + a[3])
        yb = 0.5 * (b[0] + b[1]) - 0.5 * (b[2] + b[3])
        scale = max(a.max(), b.max(), 1e-9)
        print(f"\n{label}")
        print(f"   eyeL {np.round(a,4)}  yaw={ya:+.4f}")
        print(f"   eyeR {np.round(b,4)}  yaw={yb:+.4f}")
        print(f"   split={ya-yb:+.4f} = {abs(ya-yb)/scale:.1%} of peak {scale:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Granular profile of UI closed-loop + Trainer.step after P0 hot-path patches."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from blueprint import Blueprint
from cns_runtime import CNSRuntime
from loop import ClosedLoop


def ms(fn, n=8, warm=2):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def _trainer_parts(rt, loop, prev_err: float) -> dict[str, float]:
    """Instrument Trainer.step sections without changing production code permanently."""
    h = rt
    tr = loop.trainer
    out_t: dict[str, float] = {}

    def mark(name: str, t0: float) -> float:
        t1 = time.perf_counter()
        out_t[name] = (t1 - t0) * 1000.0
        return t1

    t = time.perf_counter()
    if not getattr(h, "learn_on", False) or not getattr(h, "turret_on", False):
        return {"early_exit": 0.0}
    targets = tr.task.motor_neuron_targets(h)
    if not targets and hasattr(h, "_turret_motor_neuron_targets"):
        targets = h._turret_motor_neuron_targets()
    if not targets:
        return {"early_exit": 0.0}
    if hasattr(h, "_ensure_atlas_recipe"):
        h._ensure_atlas_recipe()
    pre, post = h.fly.activity_graph_edges()
    if len(pre) == 0:
        return {"early_exit": 0.0}
    ids = h.fly.soma_ids
    act = h.fly.soma_activity
    rebuilt = False
    if len(ids) == 0 or len(act) != len(ids):
        h.fly.rebuild_activity_cache()
        rebuilt = True
        ids = h.fly.soma_ids
        act = h.fly.soma_activity
        if len(ids) == 0 or len(act) != len(ids):
            return {"early_exit": 0.0}
    err = float(getattr(h, "turret_aim_err", float("nan")))
    if not math.isfinite(err):
        if hasattr(h, "_update_turret_aim_metric"):
            h._update_turret_aim_metric()
        err = float(getattr(h, "turret_aim_err", float("nan")))
    if not math.isfinite(err):
        return {"early_exit": 0.0}
    rule = tr.rule()
    rpe = tr.rpe(prev_err, err)
    rpe_dead = 0.002
    rpe_sign = float(np.sign(rpe)) if abs(rpe) > rpe_dead else 0.0
    rpe_mag = float(np.clip(abs(rpe) / (0.15 * math.pi), 0.0, 1.0))
    if rpe_sign == 0.0:
        rpe_mag = 0.0
    t = mark("rebuild_cache_err_rpe", t)
    out_t["rebuild_did"] = 1.0 if rebuilt else 0.0

    elig_tau = float(getattr(rule, "elig_tau", 0.88))
    tr._decay_traces(ids, act, tau=elig_tau)
    t = mark("elig_decay", t)

    device = set()
    if hasattr(h, "_turret_flat_ids"):
        device |= set(h._turret_flat_ids())
    if hasattr(h, "_ocelli_flat_ids"):
        device |= set(h._ocelli_flat_ids())
    motor_ids = set(int(k) for k in targets.keys())
    upstream = h._motor_eligibility_upstream(
        motor_ids, depth=int(rule.elig_depth)
    )
    if not upstream:
        upstream = set(motor_ids)
    for nid, e in tr.elig_trace.items():
        if e >= 0.08 and (nid in upstream or nid in motor_ids):
            upstream.add(int(nid))
    corridor = h._corridor_neuron_ids() if hasattr(h, "_corridor_neuron_ids") else set()
    t = mark("motor_eligibility_upstream_bfs", t)
    out_t["upstream_n"] = float(len(upstream))
    out_t["corridor_n"] = float(len(corridor))
    out_t["trace_n"] = float(len(tr.elig_trace))
    out_t["n_gains"] = float(len(getattr(h.fly, "contact_gain", {}) or {}))
    out_t["n_edges"] = float(len(pre))

    decay = float(rule.w_decay)
    off_decay = float(rule.offpath_decay)
    if (decay > 0 or off_decay > 0) and h.fly.contact_gain:
        for key in list(h.fly.contact_gain.keys()):
            cur = float(h.fly.contact_gain[key])
            if not math.isfinite(cur):
                h.fly.contact_gain.pop(key, None)
                continue
            a, b = int(key[0]), int(key[1])
            on_bed = (a in corridor) or (b in corridor)
            d = decay if on_bed else max(decay, off_decay)
            nxt = cur + d * (1.0 - cur)
            h.fly.set_contact_gain(
                a, b, nxt, lo=float(rule.w_min), hi=float(rule.w_max)
            )
    t = mark("contact_gain_decay_loop", t)

    # ---- Hebb / teacher edge loop (mirrors Trainer.step body) ----
    lr = float(rule.lr)
    hebb = float(rule.hebb)
    dw_cap = float(rule.dw_cap)
    dw_abs = 0.0
    n_touch = 0
    flash_a: list[int] = []
    flash_b: list[int] = []
    n_max = len(act)
    contiguous = (
        int(ids[0]) == 0
        and int(ids[-1]) == len(ids) - 1
        and len(ids) == len(h.fly.neurons)
    )
    lut = None
    if not contiguous:
        lut = {int(i): k for k, i in enumerate(ids.tolist())}
    motor_arr = np.fromiter(motor_ids, dtype=np.int32, count=len(motor_ids))
    up_arr = np.fromiter(upstream, dtype=np.int32, count=len(upstream))
    cor_arr = (
        np.fromiter(corridor, dtype=np.int32, count=len(corridor))
        if corridor
        else np.zeros((0,), dtype=np.int32)
    )
    path_on = float(rule.path_credit) > 0.0
    neck_gate = bool(corridor) and path_on
    left_motors, right_motors = tr._lr_motor_sets(h, targets)

    if contiguous:
        valid = (pre >= 0) & (post >= 0) & (pre < n_max) & (post < n_max)
        pre_i = pre[valid]
        post_i = post[valid]
        tr_ok = np.fromiter(
            (tr._trace_eligible(int(x)) for x in pre_i.tolist()),
            dtype=bool,
            count=len(pre_i),
        )
        elig = (
            (np.isin(pre_i, up_arr) | tr_ok)
            & np.isin(post_i, up_arr)
            & (act[pre_i] > 0.02)
        )
        into_motor = np.isin(post_i, motor_arr)
        touch_cor = (
            np.isin(pre_i, cor_arr) | np.isin(post_i, cor_arr)
            if len(cor_arr)
            else np.zeros(len(pre_i), dtype=bool)
        )
        if path_on:
            keep = elig & (into_motor | touch_cor) if neck_gate else elig
        else:
            keep = elig & into_motor
        pre_h = pre_i[keep]
        post_h = post_i[keep]
        a_pre_h = act[pre_h]
        a_post_h = act[post_h]
        if len(pre_h) > 2500:
            score = a_pre_h.copy()
            score = score + 2.0 * np.isin(post_h, motor_arr).astype(np.float32)
            if len(cor_arr):
                score = score + 1.0 * (
                    np.isin(pre_h, cor_arr) | np.isin(post_h, cor_arr)
                ).astype(np.float32)
            order = np.argsort(-score)[:2500]
            pre_h, post_h = pre_h[order], post_h[order]
            a_pre_h, a_post_h = a_pre_h[order], a_post_h[order]
        iter_rows = zip(
            pre_h.tolist(),
            post_h.tolist(),
            a_pre_h.tolist(),
            a_post_h.tolist(),
        )
    else:
        assert lut is not None
        rows = []
        for a, b in zip(pre.tolist(), post.tolist()):
            a, b = int(a), int(b)
            if b not in upstream and not tr._trace_eligible(a):
                continue
            if a not in upstream and not tr._trace_eligible(a):
                continue
            ia, ib = lut.get(a), lut.get(b)
            if ia is None or ib is None:
                continue
            ap = float(act[ia])
            if ap < 0.02:
                continue
            ao = float(act[ib])
            if b in motor_ids:
                rows.append((a, b, ap, ao))
                continue
            if not path_on:
                continue
            if neck_gate and a not in corridor and b not in corridor:
                continue
            rows.append((a, b, ap, ao))
        iter_rows = rows
    t = mark("hebb_filter_mask", t)

    for a, b, a_pre, a_post in iter_rows:
        a, b = int(a), int(b)
        if a in device and b in device:
            continue
        dw = 0.0
        tgt = targets.get(b)
        if tgt is not None:
            dw = lr * float(a_pre) * (float(tgt) - float(a_post))
            if corridor and (a in corridor or b in corridor):
                dw *= 1.35
            dw = tr._lr_compete_scale(
                dw, b, rpe_sign, left_motors, right_motors, targets
            )
        elif path_on and (b in upstream or tr._trace_eligible(a)):
            if neck_gate and a not in corridor and b not in corridor:
                continue
            err_n = float(np.clip(err / (0.5 * math.pi), 0.0, 1.0))
            dw = (
                float(rule.path_credit)
                * lr
                * float(a_pre)
                * err_n
                * (float(a_pre) - float(a_post))
            )
            if corridor and (a in corridor or b in corridor):
                dw *= 1.5
            if rpe_sign < 0:
                dw *= 0.15
            elif rpe_sign > 0:
                dw *= 0.5 + 0.5 * rpe_mag
            else:
                dw *= 0.08
        else:
            continue
        if hebb > 0.0 and rpe_sign != 0.0:
            dw += hebb * float(a_pre) * float(a_post) * rpe_sign
        if abs(dw) < 1e-7:
            continue
        dw = float(np.clip(dw, -dw_cap, dw_cap))
        if not math.isfinite(dw):
            continue
        h.fly.adjust_contact_gain(
            a, b, dw, lo=float(rule.w_min), hi=float(rule.w_max)
        )
        dw_abs += abs(dw)
        n_touch += 1
        flash_a.append(a)
        flash_b.append(b)
    t = mark("hebb_edge_loop", t)
    out_t["n_touch_hebb"] = float(n_touch)

    # Carve (same as Trainer.step)
    carve = float(rule.carve_lr)
    if (
        carve > 0.0
        and corridor
        and len(pre) > 0
        and rpe_sign != 0.0
        and rpe_mag > 0.02
    ):
        err_n = float(np.clip(err / (0.5 * math.pi), 0.0, 1.0))
        act_lo = float(rule.channel_act_min)
        cor_arr = np.fromiter(corridor, dtype=np.int32, count=len(corridor))
        carve_scale = carve * rpe_sign * rpe_mag * (0.35 + 0.65 * err_n)
        if contiguous:
            valid = (pre >= 0) & (post >= 0) & (pre < n_max) & (post < n_max)
            pa = pre[valid]
            pb = post[valid]
            touch_cor = np.isin(pa, cor_arr) | np.isin(pb, cor_arr)
            in_up = np.isin(pa, up_arr) & np.isin(pb, up_arr)
            lit = (act[pa] > act_lo) & (act[pb] > act_lo * 0.5)
            keep_c = touch_cor & in_up & lit
            if np.any(keep_c):
                pa_c = pa[keep_c]
                pb_c = pb[keep_c]
                ap_c = act[pa_c]
                ao_c = act[pb_c]
                if len(pa_c) > 1800:
                    order = np.argsort(-(ap_c * ao_c))[:1800]
                    pa_c, pb_c = pa_c[order], pb_c[order]
                    ap_c, ao_c = ap_c[order], ao_c[order]
                carve_rows = zip(
                    pa_c.tolist(), pb_c.tolist(), ap_c.tolist(), ao_c.tolist()
                )
            else:
                carve_rows = ()
        else:
            assert lut is not None
            carve_list = []
            for a, b in zip(pre.tolist(), post.tolist()):
                a, b = int(a), int(b)
                if a not in corridor and b not in corridor:
                    continue
                if a not in upstream or b not in upstream:
                    continue
                if a in device and b in device:
                    continue
                ia, ib = lut.get(a), lut.get(b)
                if ia is None or ib is None:
                    continue
                ap = float(act[ia])
                ao = float(act[ib])
                if ap < act_lo or ao < act_lo * 0.5:
                    continue
                carve_list.append((a, b, ap, ao))
            if len(carve_list) > 1800:
                carve_list.sort(key=lambda t: -(t[2] * t[3]))
                carve_list = carve_list[:1800]
            carve_rows = carve_list
        for a, b, a_pre, a_post in carve_rows:
            a, b = int(a), int(b)
            if a in device and b in device:
                continue
            dw = carve_scale * float(a_pre) * float(a_post)
            if abs(dw) < 1e-7:
                continue
            dw = float(np.clip(dw, -dw_cap, dw_cap))
            if not math.isfinite(dw):
                continue
            h.fly.adjust_contact_gain(
                a, b, dw, lo=float(rule.w_min), hi=float(rule.w_max)
            )
            dw_abs += abs(dw)
            n_touch += 1
            if len(flash_a) < 400:
                flash_a.append(a)
                flash_b.append(b)
    t = mark("carve_loop", t)

    h.learn_dw_ema = 0.85 * float(getattr(h, "learn_dw_ema", 0.0)) + 0.15 * dw_abs
    if not math.isfinite(h.learn_dw_ema):
        h.learn_dw_ema = 0.0
    h.learn_n_edges = int(len(getattr(h.fly, "contact_gain", {}) or {}))
    h._learn_flash_n = int(n_touch)
    if n_touch > 0:
        if len(flash_a) > 400:
            flash_a = flash_a[:400]
            flash_b = flash_b[:400]
        h._learn_flash_pre = np.asarray(flash_a, dtype=np.int32)
        h._learn_flash_post = np.asarray(flash_b, dtype=np.int32)
        h._learn_flash_amp = float(
            np.clip(0.55 + 0.45 * min(1.0, dw_abs * 8.0), 0.55, 1.0)
        )
    else:
        h._learn_flash_amp = float(getattr(h, "_learn_flash_amp", 0.0)) * 0.88
    if hasattr(h, "_refresh_channel_bed"):
        h._refresh_channel_bed()
    t = mark("flash_refresh_bed", t)
    out_t["n_touch"] = float(n_touch)
    out_t["total"] = sum(
        v
        for k, v in out_t.items()
        if k
        not in (
            "rebuild_did",
            "upstream_n",
            "corridor_n",
            "trace_n",
            "n_gains",
            "n_edges",
            "n_touch",
            "n_touch_hebb",
        )
    )
    return out_t


def _avg_parts(rt, loop, n=4, warm=1) -> dict[str, float]:
    for _ in range(warm):
        prev = float(getattr(rt, "turret_aim_err", float("nan")))
        loop.step(1.0 / 60.0)
        _trainer_parts(rt, loop, prev)
    acc: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _ in range(n):
        prev = float(getattr(rt, "turret_aim_err", float("nan")))
        # Keep activity warm via a hop, then profile trainer alone.
        loop.sense_all()
        rt.fly.tick_activity(hop_gain=0.85, decay=0.90)
        if hasattr(rt, "_sync_activity_view"):
            rt._sync_activity_view()
        loop.actuate_all(1.0 / 60.0)
        loop.sense_turret_proprio()
        parts = _trainer_parts(rt, loop, prev)
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: acc[k] / counts[k] for k in acc}


def _closed_loop_pieces(rt, loop, *, force_learn: bool) -> dict[str, float]:
    h = rt
    dt = 1.0 / 60.0
    out: dict[str, float] = {}

    def mark(name: str, t0: float) -> float:
        t1 = time.perf_counter()
        out[name] = (t1 - t0) * 1000.0
        return t1

    t = time.perf_counter()
    loop._frame_i = int(loop._frame_i) + 1
    if force_learn:
        # Make next frame a learn beat regardless of learn_every.
        le = max(1, int(getattr(h, "learn_every", 1) or 1))
        loop._frame_i = le  # % le == 0
    if getattr(h, "_devices_dirty", True):
        if hasattr(h, "_sync_shell_fields_to_devices"):
            h._sync_shell_fields_to_devices()
        h._devices_dirty = False
    prev_err = float(getattr(h, "turret_aim_err", float("nan")))
    loop.sense_all()
    t = mark("sense_all", t)
    if len(h.fly.recipe_pre) == 0 and hasattr(h, "_ensure_atlas_recipe"):
        h._ensure_atlas_recipe()
    hops = loop._resolve_hops()
    info: dict = {"edges": 0, "lit": 0, "touched": 0, "max": 0.0}
    for _ in range(hops):
        info = h.fly.tick_activity(hop_gain=0.85, decay=0.90)
    info["hops"] = hops
    t = mark("tick_activity", t)
    if hasattr(h, "_sync_activity_view"):
        h._sync_activity_view()
    t = mark("sync_activity_view", t)
    learn_on = bool(getattr(h, "learn_on", False) and getattr(h, "turret_on", False))
    learn_every = loop._resolve_learn_every()
    do_learn = learn_on and (loop._frame_i % learn_every == 0)
    info["learned"] = bool(do_learn)
    if learn_on and do_learn and hasattr(h, "_apply_aim_teacher_mix"):
        h._apply_aim_teacher_mix()
        if hasattr(h, "_sync_activity_view"):
            h._sync_activity_view()
    t = mark("teacher_mix", t)
    loop.actuate_all(float(dt))
    t = mark("actuate", t)
    loop.sense_turret_proprio()
    t = mark("sense_proprio", t)
    if do_learn:
        tr = loop.trainer
        if tr is None:
            from trainer import Trainer

            tr = Trainer(h, task=loop.task)
            loop.trainer = tr
            h._trainer = tr
        tr.step(prev_err)
        t = mark("trainer_step", t)
        h._learn_prev_err = float(getattr(h, "turret_aim_err", float("nan")))
        if hasattr(h, "_merge_learn_chain_flash"):
            h._merge_learn_chain_flash(
                hop_pre=info.get("flash_pre"),
                hop_post=info.get("flash_post"),
            )
        t = mark("merge_learn_flash", t)
    else:
        amp = float(getattr(h, "_learn_flash_amp", 0.0) or 0.0)
        h._learn_flash_amp = amp * 0.92
        out["trainer_step"] = 0.0
        out["merge_learn_flash"] = 0.0
        t = mark("flash_decay_skip", t)
    out["do_learn"] = 1.0 if do_learn else 0.0
    out["total"] = sum(
        v for k, v in out.items() if k not in ("do_learn",) and isinstance(v, float)
    )
    return out


def _avg_loop_pieces(rt, loop, *, force_learn: bool, n=4, warm=1) -> dict[str, float]:
    for _ in range(warm):
        _closed_loop_pieces(rt, loop, force_learn=force_learn)
    acc: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _ in range(n):
        parts = _closed_loop_pieces(rt, loop, force_learn=force_learn)
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: acc[k] / counts[k] for k in acc}


def _estimate_draw(rt) -> dict[str, float]:
    """Best-effort draw cost with dummy SDL (no real GPU blit)."""
    out: dict[str, float] = {}
    # Prefer GenomeShell if importable; else skip.
    try:
        # Avoid constructing full shell (needs display). Time flash merge helpers
        # and channel bed / flash draw prep that run on learn frames.
        t0 = time.perf_counter()
        if hasattr(rt, "_merge_learn_chain_flash"):
            info = {"flash_pre": None, "flash_post": None}
            # Use current flash arrays if present.
            rt._merge_learn_chain_flash(
                hop_pre=info.get("flash_pre"),
                hop_post=info.get("flash_post"),
            )
        out["merge_flash_alone"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if hasattr(rt, "_refresh_channel_bed"):
            rt._refresh_channel_bed()
        out["refresh_channel_bed"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if hasattr(rt, "_corridor_neuron_ids"):
            _ = rt._corridor_neuron_ids()
        out["corridor_ids"] = (time.perf_counter() - t0) * 1000.0

        # Motor BFS alone (often rebuilt every learn beat).
        targets = {}
        if hasattr(rt, "_turret_motor_neuron_targets"):
            targets = rt._turret_motor_neuron_targets()
        motor_ids = set(int(k) for k in targets.keys())
        depth = int(getattr(getattr(rt, "rule", None), "elig_depth", 8) or 8)
        t0 = time.perf_counter()
        if hasattr(rt, "_motor_eligibility_upstream"):
            up = rt._motor_eligibility_upstream(motor_ids, depth=depth)
            out["upstream_n"] = float(len(up))
        out["motor_bfs_alone"] = (time.perf_counter() - t0) * 1000.0

        # Proxy for learn flash draw: build engaged set + project somata.
        pre = getattr(rt, "_learn_flash_pre", None)
        post = getattr(rt, "_learn_flash_post", None)
        if pre is not None and post is not None and len(pre):
            ids = rt.fly.soma_ids
            xyz = rt.fly.soma_xyz
            lut = {int(i): k for k, i in enumerate(ids.tolist())}
            engaged = set(int(a) for a in pre.tolist()) | set(
                int(b) for b in post.tolist()
            )
            eng_idx = [lut[i] for i in engaged if i in lut][:900]
            t0 = time.perf_counter()
            if eng_idx and len(xyz) == len(ids):
                e_xyz = xyz[np.asarray(eng_idx, dtype=np.int32)]
                # Cheap project proxy (no pygame).
                _ = e_xyz[:, 0] * 0.1 + e_xyz[:, 1] * 0.1
            out["flash_project_proxy"] = (time.perf_counter() - t0) * 1000.0
            out["flash_engaged_n"] = float(len(engaged))
        else:
            out["flash_project_proxy"] = 0.0
            out["flash_engaged_n"] = 0.0
    except Exception as exc:
        out["draw_err"] = -1.0
        out["draw_err_msg"] = 0.0
        print(f"[draw estimate] {exc}")
    return out


def _print_section(title: str, parts: dict[str, float], keys: list[str]) -> None:
    print(f"\n=== {title} ===")
    for i, k in enumerate(keys, 1):
        if k not in parts:
            print(f"{i:2d}. {k:36s}  —")
            continue
        print(f"{i:2d}. {k:36s}  {parts[k]:8.2f} ms")
    meta = [k for k in sorted(parts) if k not in keys and not k.startswith("draw_err")]
    if meta:
        print("  -- meta --")
        for k in meta:
            v = parts[k]
            if abs(v - round(v)) < 1e-9 and abs(v) < 1e7:
                print(f"      {k:34s}  {int(round(v))}")
            else:
                print(f"      {k:34s}  {v:8.2f}")


def main() -> None:
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
    rt.learn_every = 4
    loop: ClosedLoop = rt._closed_loop
    loop.trainer = rt.trainer()
    dt = 1.0 / 60.0

    for _ in range(8):
        loop.step(dt)

    fly = rt.fly
    print(f"n={len(fly.neurons):,} hop_edges={len(fly.activity_graph_edges()[0]):,}")
    print(
        f"alive={(fly.soma_activity > 0).sum():,} "
        f"contiguous={fly._activity_ids_contiguous()} "
        f"gains={len(fly.contact_gain):,}"
    )
    print(f"learn_every={rt.learn_every} loop_hops={rt.loop_hops} scope_on={rt.scope_on}")

    print("\n--- baseline primitives ---")
    print(f"propagate         {ms(lambda: fly.propagate_activity(gain=0.9)):6.1f} ms")
    print(f"decay_activity    {ms(lambda: fly.decay_activity(0.97)):6.1f} ms")
    print(
        f"tick_activity     {ms(lambda: fly.tick_activity(hop_gain=0.9, decay=0.97)):6.1f} ms"
    )
    print(f"sense_all         {ms(lambda: loop.sense_all()):6.1f} ms")
    print(f"sense_proprio     {ms(lambda: loop.sense_turret_proprio()):6.1f} ms")
    print(
        f"Trainer.step      {ms(lambda: loop.trainer.step(float(getattr(rt, 'turret_aim_err', float('nan')))), n=4, warm=1):6.1f} ms"
    )

    # ---- Trainer.step breakdown ----
    parts = _avg_parts(rt, loop, n=4, warm=1)
    _print_section(
        "Trainer.step breakdown (avg ms)",
        parts,
        [
            "rebuild_cache_err_rpe",
            "elig_decay",
            "motor_eligibility_upstream_bfs",
            "contact_gain_decay_loop",
            "hebb_filter_mask",
            "hebb_edge_loop",
            "carve_loop",
            "flash_refresh_bed",
            "total",
        ],
    )

    # ---- ClosedLoop learn_every=1 vs 4 ----
    rt.learn_every = 1
    le1 = _avg_loop_pieces(rt, loop, force_learn=True, n=4, warm=1)
    _print_section(
        "ClosedLoop pieces learn_every=1 (forced learn)",
        le1,
        [
            "sense_all",
            "tick_activity",
            "sync_activity_view",
            "teacher_mix",
            "actuate",
            "sense_proprio",
            "trainer_step",
            "merge_learn_flash",
            "total",
        ],
    )

    rt.learn_every = 4
    # Forced learn beat
    le4_learn = _avg_loop_pieces(rt, loop, force_learn=True, n=4, warm=1)
    _print_section(
        "ClosedLoop pieces learn_every=4 LEARN beat",
        le4_learn,
        [
            "sense_all",
            "tick_activity",
            "sync_activity_view",
            "teacher_mix",
            "actuate",
            "sense_proprio",
            "trainer_step",
            "merge_learn_flash",
            "total",
        ],
    )
    # Skip beat: set frame so % 4 != 0
    def skip_once():
        loop._frame_i = 1  # will become 2 inside
        return _closed_loop_pieces(rt, loop, force_learn=False)

    for _ in range(1):
        skip_once()
    acc = {}
    for _ in range(4):
        # Force non-learn: call pieces with learn_every=4 and frame not multiple.
        # _closed_loop_pieces increments then checks; start at 0 → frame 1.
        loop._frame_i = 0
        p = _closed_loop_pieces(rt, loop, force_learn=False)
        for k, v in p.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    le4_skip = {k: v / 4.0 for k, v in acc.items()}
    _print_section(
        "ClosedLoop pieces learn_every=4 SKIP beat",
        le4_skip,
        [
            "sense_all",
            "tick_activity",
            "sync_activity_view",
            "teacher_mix",
            "actuate",
            "sense_proprio",
            "trainer_step",
            "merge_learn_flash",
            "flash_decay_skip",
            "total",
        ],
    )

    # Full step averages
    rt.learn_every = 4
    print(f"\nClosedLoop.step le=4  {ms(lambda: loop.step(dt), n=8):6.1f} ms")
    rt.learn_every = 1
    print(f"ClosedLoop.step le=1  {ms(lambda: loop.step(dt), n=4, warm=1):6.1f} ms")
    rt.learn_every = 4

    draw = _estimate_draw(rt)
    _print_section(
        "Draw / learn-frame side costs (CPU proxy, no real blit)",
        draw,
        [
            "merge_flash_alone",
            "refresh_channel_bed",
            "corridor_ids",
            "motor_bfs_alone",
            "flash_project_proxy",
        ],
    )

    # Note adult tick draw policy for the report.
    print(
        "\n[genome_shell adult] run() always calls self._draw() every frame "
        "(after closed-loop tick). Learn flash amp decays on skip beats; "
        "full _draw_learn_flash runs whenever amp>0.04."
    )


if __name__ == "__main__":
    main()

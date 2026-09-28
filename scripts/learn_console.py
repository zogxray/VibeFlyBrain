#!/usr/bin/env python3
"""Headless turret Learn via CNSRuntime + Blueprint (no pygame window).

Usage:
  python scripts/learn_console.py --lamp-speed 0.08
  python scripts/learn_console.py --steps 2000 --lamp-speed 0.05
  python scripts/learn_console.py --model data/flies/x.fly --save data/flies/x_learned.fly
  python scripts/learn_console.py --socket   # legacy motor-socket bypass

Stops early when:
  - SUCCESS: under Spin, Track ≥ --track-ready for --settle log windows
             (static lamp: ε ≤ --target-eps for --settle windows)
  - STALL:   ε not improving AND Δw below --min-dw for --patience windows
  - DEAD:    no plastic edges for --patience windows while ε still high
  - UNSTABLE: non-finite gains / Δw
  - REACH:   ocelli→motors probe fails at start
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Headless before pygame import via genome_shell (CNSRuntime.open → GenomeShell).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from blueprint import Blueprint  # noqa: E402
from cns_runtime import CNSRuntime  # noqa: E402
from fly_model import save_fly  # noqa: E402


def _print_path_debug(rt: CNSRuntime) -> dict:
    rep = rt._debug_device_path_report(max_hops=16)
    print(
        f"[path] graph_edges={rep['edges']:,} "
        f"oc_src={rep['ocelli_src']} sens_src={rep['sensory_src']} "
        f"motors={rep['motors']}",
        flush=True,
    )
    oc = rep["ocelli_to_motor"]
    se = rep["sensory_to_motor"]
    print(
        f"[path] ocelli→motor hit={oc['hit']}/{oc['motors']} "
        f"best_hops={oc['best_hops']} median={oc['median_hops']}",
        flush=True,
    )
    print(
        f"[path] sensory→motor hit={se['hit']}/{se['motors']} "
        f"best_hops={se['best_hops']} median={se['median_hops']}",
        flush=True,
    )
    if rep.get("example_path"):
        print(
            f"[path] example ocelli→motor len={rep['example_path_len']} "
            f"ids={rep['example_path'][:12]}...",
            flush=True,
        )
    else:
        print("[path] example ocelli→motor: NONE", flush=True)
    fun = rep.get("wells") or rep.get("funnel")
    if isinstance(fun, dict) and "error" not in fun:
        print(
            f"[path] wells A={fun.get('A')} B={fun.get('B')} ∩={fun.get('N')} "
            f"AN/DN={fun.get('neck_class', '—')}",
            flush=True,
        )
    elif isinstance(fun, dict):
        print(f"[path] wells error: {fun.get('error')}", flush=True)
    sc = rep.get("scope")
    if isinstance(sc, dict):
        print(
            f"[path] scope ok={sc.get('ok')} paths={sc.get('paths')} "
            f"n={sc.get('n')} e={sc.get('e')} "
            f"hops_floor={sc.get('hops_floor')} near={sc.get('best_hops')} "
            f"groups={sc.get('group_hops')}",
            flush=True,
        )
    return rep


def _gain_stats(rt: CNSRuntime) -> tuple[int, int, float, int, int, int, int]:
    gains = getattr(rt.fly, "contact_gain", {}) or {}
    device = set(rt._turret_flat_ids()) | set(rt._ocelli_flat_ids())
    corridor = rt._corridor_neuron_ids()
    nan_g = 0
    abs_vals: list[float] = []
    n_cns = n_br = n_dev = n_cor = 0
    for (a, b), w in gains.items():
        wf = float(w)
        if not math.isfinite(wf):
            nan_g += 1
            continue
        abs_vals.append(abs(wf - 1.0))
        a_i, b_i = int(a), int(b)
        a_d, b_d = a_i in device, b_i in device
        if a_d and b_d:
            n_dev += 1
        elif a_d != b_d:
            n_br += 1
        else:
            n_cns += 1
        if a_i in corridor or b_i in corridor:
            n_cor += 1
    mean_abs = float(np.mean(abs_vals)) if abs_vals else 0.0
    return len(gains), nan_g, mean_abs, n_cns, n_br, n_dev, n_cor


def _eps_deg(rt: CNSRuntime) -> float:
    err = float(rt.turret_aim_err)
    if not math.isfinite(err):
        return float("nan")
    return err * 180.0 / math.pi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless FlyWire turret Learn")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .fly (default: shell_ui last_model)",
    )
    p.add_argument("--steps", type=int, default=3000, help="Max closed-loop ticks")
    p.add_argument("--log-every", type=int, default=30, help="Console log period")
    p.add_argument(
        "--lamp-speed",
        type=float,
        default=0.0,
        help="Lamp rev/s during Run (0 = static curriculum)",
    )
    p.add_argument("--lamp-angle", type=float, default=0.8, help="Initial lamp azimuth")
    p.add_argument(
        "--yaw-offset",
        type=float,
        default=2.2,
        help="Add to saved turret yaw at start (rad)",
    )
    p.add_argument(
        "--keep-gains",
        action="store_true",
        help="Do not clear contact_gain before training",
    )
    p.add_argument(
        "--no-fuse",
        action="store_true",
        help="Skip instant Fuse (use edges already in the .fly)",
    )
    p.add_argument(
        "--target-eps",
        type=float,
        default=5.0,
        help="Static-lamp SUCCESS if ε≤this (deg) for --settle windows",
    )
    p.add_argument(
        "--track-ready",
        type=float,
        default=0.80,
        help="Spin SUCCESS if Track≥this for --settle windows (CNS hold)",
    )
    p.add_argument(
        "--settle",
        type=int,
        default=3,
        help="Consecutive good log windows for SUCCESS",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Bad log windows before STALL/DEAD stop",
    )
    p.add_argument(
        "--min-improve",
        type=float,
        default=1.0,
        help="ε must drop by this many degrees vs best to count as progress",
    )
    p.add_argument(
        "--min-dw",
        type=float,
        default=1e-4,
        help="Δw below this + no ε progress → STALL",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Log windows ignored before stall checks",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write .fly with learned gains (+ shell_devices)",
    )
    p.add_argument(
        "--save-always",
        action="store_true",
        help="Save even on STALL/DEAD (not on REACH/UNSTABLE)",
    )
    p.add_argument(
        "--debug-path",
        action="store_true",
        help="Print BFS path + funnel gate diagnostics (also before train)",
    )
    p.add_argument(
        "--path-only",
        action="store_true",
        help="Only run path diagnostics (no training loop)",
    )
    p.add_argument(
        "--no-scope",
        action="store_true",
        help="Skip TrainingScope (full hop; no graph corridor)",
    )
    p.add_argument(
        "--no-hop-clip",
        action="store_true",
        help="Keep scope corridor for Learn but do not clip hop_allow",
    )
    p.add_argument(
        "--cns",
        action="store_true",
        default=True,
        help="Force CNS mode: motor_socket OFF, teacher→slew 0 (default)",
    )
    p.add_argument(
        "--socket",
        action="store_true",
        help="Legacy motor-socket bypass (overrides --cns)",
    )
    p.add_argument(
        "--teacher-mix",
        type=float,
        default=None,
        help="Override rule.teacher_mix (0 = no AimTask clamp shaping)",
    )
    p.add_argument(
        "--no-stereo",
        action="store_true",
        help="Fuse eyes/wing pads by nearest soma (legacy common-mode projection)",
    )
    p.add_argument(
        "--no-eyes",
        action="store_true",
        help="Keep compound eyes off (default: eye_left+eye_right on)",
    )
    p.add_argument(
        "--no-jo",
        action="store_true",
        help="Keep Johnston organs off (default: jo_left+jo_right on)",
    )
    p.add_argument(
        "--no-ocelli",
        action="store_true",
        help="Keep ocelli off — isolates the eye→motor wave",
    )
    p.add_argument(
        "--reach-hops",
        type=int,
        default=24,
        help="Beats for the per-device sensor→motor reach probe",
    )
    return p.parse_args()


def _enable_head_sensors(rt, *, eyes: bool, jo: bool, ocelli: bool) -> dict[str, bool]:
    """Flip host flags + typed ``d.enabled`` before Fuse (Fuse skips disabled)."""
    want = {
        "ocelli": bool(ocelli),
        "eye_left": bool(eyes),
        "eye_right": bool(eyes),
        "jo_left": bool(jo),
        "jo_right": bool(jo),
    }
    rt.ocelli_on = bool(ocelli)
    for name in ("eye_left", "eye_right", "jo_left", "jo_right"):
        setattr(rt, f"{name}_on", bool(want[name]))
    for d in getattr(rt, "device_list", []) or []:
        kind = getattr(d, "kind", "")
        name = str(getattr(d, "name", "") or kind)
        if kind == "ocelli":
            d.enabled = bool(ocelli)
        elif kind in ("eye", "jo") and name in want:
            d.enabled = bool(want[name])
    rt.blueprint.ocelli_on = bool(ocelli)
    rt.blueprint.sync_bundle_from_list()
    rt._devices_dirty = True
    return want


def _print_device_reach(rt, *, hops: int) -> list[dict]:
    """Eye→motor wave reported apart from ocelli→motor (honest FAIL if dark)."""
    rows = rt.probe_sensor_reach_by_device(hops=int(hops))
    scope = getattr(rt, "_training_scope", None)
    geo = dict(getattr(scope, "group_best_hops", {}) or {}) if scope else {}
    for r in rows:
        state = "on " if r.get("enabled") else "off"
        first = r.get("first_hop")
        need = "—" if first is None else f"{first}"
        allh = r.get("all_hop")
        all_s = "—" if allh is None else f"{allh}"
        g = geo.get(r["device"])
        verdict = "OK  " if r["ok"] else ("WEAK" if r["hit"] else "FAIL")
        az = r.get("lit_az")
        lit = "world" if az is None else f"az={float(az):+.2f}"
        print(
            f"[reach] {verdict} {r['device']:<10} {state} "
            f"seed={r['seed']:<4d} hit={r['hit']}/{r['motors']} "
            f"best={r['best']:.3f} first_hop={need} all_hop={all_s} "
            f"drive={float(r.get('drive', 0.0)):.3f}→{float(r.get('drive_ref', 0.0)):.3f} "
            f"lit={lit} "
            f"geodesic={'—' if g is None else g} (probe hops={r['hops']})",
            flush=True,
        )
    for r in rows:
        if not r.get("enabled") or r["hit"]:
            continue
        g = geo.get(r["device"])
        if g is None:
            why = (
                f"no graph path to motors within scope_max_hops — re-Fuse or "
                f"raise blueprint.scope_max_hops (no shortcut wired)"
            )
        elif float(r.get("drive_ref", 0.0)) <= 0.12:
            why = (
                f"graph path exists at {g} hops but the sensor itself is dark "
                f"(drive {float(r.get('drive_ref', 0.0)):.3f}) — light/pose "
                f"problem, not the corridor"
            )
        else:
            why = (
                f"graph path exists at {g} hops but the wave decays below the "
                f"lit threshold (peak {r['best']:.3f}) — needs drive/gain, "
                f"not more beats"
            )
        print(
            f"[reach] FAIL {r['device']} → motors dark after {r['hops']} beats "
            f"(seed={r['seed']}): {why}",
            flush=True,
        )
    return rows


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    ui_path = ROOT / "data" / "processed" / "shell_ui.json"
    ui = json.loads(ui_path.read_text()) if ui_path.is_file() else {}
    path = Path(
        args.model
        or ui.get("last_model")
        or (ROOT / "data" / "flies" / "x.fly")
    )
    if not path.is_file():
        print(f"[learn] missing model: {path}", flush=True)
        return 2

    print(f"[learn] load {path}", flush=True)
    # Prefer .fly shell_devices; UI only if it actually has wired groups.
    from fly_model import load_fly

    fly_meta = getattr(load_fly(path), "shell_meta", {}) or {}
    fly_dev = fly_meta.get("shell_devices") or {}
    ui_dev = ui.get("shell_devices") or {}

    def _wired(dev: dict) -> bool:
        groups = (dev.get("ocelli_groups") or []) + (
            dev.get("turret_motor_groups") or []
        )
        return any(isinstance(g, list) and len(g) > 0 for g in groups)

    shell_dev = fly_dev if _wired(fly_dev) else (ui_dev if _wired(ui_dev) else None)
    bp = Blueprint.from_fly_meta(
        path,
        shell_devices=shell_dev,
        yaw_offset=float(args.yaw_offset),
        lamp_speed=float(args.lamp_speed),
        lamp_angle=float(args.lamp_angle),
        scope_on=not bool(args.no_scope),
        hop_clip=not bool(args.no_hop_clip),
    )
    bp.stereo_channels = not bool(args.no_stereo)
    rt = CNSRuntime.open(
        path,
        blueprint=bp,
        shell_devices=shell_dev,
        keep_gains=bool(args.keep_gains),
    )
    # Ensure flat shell fields match snapshot (open may skip empty-looking bundle).
    if shell_dev and hasattr(rt, "_apply_device_snapshot"):
        rt._apply_device_snapshot(shell_dev)
        rt._sync_shell_fields_to_devices()
        rt._devices_dirty = True
    print(f"[learn] n={len(rt.fly.neurons):,}", flush=True)

    # Head sensors must be live *before* Fuse — instant_fuse skips disabled.
    want = _enable_head_sensors(
        rt,
        eyes=not bool(args.no_eyes),
        jo=not bool(args.no_jo),
        ocelli=not bool(args.no_ocelli),
    )
    print(
        "[learn] sensors "
        + " ".join(f"{k}={'on' if v else 'off'}" for k, v in want.items()),
        flush=True,
    )

    if args.no_fuse:
        rt.fly.enter_adult(abandon_open=True)
        rt.fly.rebuild_caches()
        rt._ensure_atlas_recipe()
        nf = 0
    else:
        nf = rt.instant_fuse()
    print(
        f"[learn] fuse pairs={nf} ocelli={rt._ocelli_connected_mask()} "
        f"turret={rt._turret_connected_mask()} recipe={len(rt.fly.recipe_pre):,} "
        f"hop={len(rt.fly.activity_graph_edges()[0]):,}",
        flush=True,
    )
    for name in ("eye_left", "eye_right"):
        if want.get(name) and hasattr(rt, "_eye_connected_mask"):
            print(f"[learn] fuse {name}={rt._eye_connected_mask(name)}", flush=True)
    for name in ("jo_left", "jo_right"):
        if want.get(name) and hasattr(rt, "_jo_connected_mask"):
            print(f"[learn] fuse {name}={rt._jo_connected_mask(name)}", flush=True)

    if rt.blueprint.scope_on:
        if rt.sync_training_scope():
            sc = rt._training_scope
            print(
                f"[learn] scope {sc.readout(dw_edges=0)} "
                f"hop={len(rt.fly.activity_graph_edges()[0]):,} "
                f"hop_clip={rt.hop_clip}",
                flush=True,
            )
            print(
                f"[learn] scope per-sensor hops {sc.metrics().groups_readout()} "
                f"· motors_hit "
                + " ".join(
                    f"{k}:{v}" for k, v in sorted(sc.group_motors_hit.items())
                ),
                flush=True,
            )
        else:
            print(
                "[learn] WARN scope refused — PATH_CREDIT may spray CNS",
                flush=True,
            )

    if args.debug_path or args.path_only:
        _print_path_debug(rt)

    reach = rt.trainer().probe_ocelli_reach(hops=10)
    print(
        f"[learn] REACH ocelli→motors hit={reach['hit']}/{reach['motors']} "
        f"best={reach['best']:.3f} ok={reach['ok']}",
        flush=True,
    )
    sreach = rt.trainer().probe_sensory_reach(hops=12)
    print(
        f"[learn] REACH sensory→motors hit={sreach['hit']}/{sreach['motors']} "
        f"best={sreach['best']:.3f} lr_gap={sreach['lr_gap']:+.3f} ok={sreach['ok']}",
        flush=True,
    )
    dev_reach = _print_device_reach(rt, hops=int(args.reach_hops))
    eye_ok = any(r["ok"] for r in dev_reach if r["device"].startswith("eye"))
    if args.path_only:
        print("=== PATH-ONLY done ===", flush=True)
        return 0 if (reach["ok"] or sreach["ok"] or eye_ok) else 3
    if not reach["ok"] and not sreach["ok"] and not eye_ok:
        print("=== STOP REACH — no path to motors; fix Fuse first ===", flush=True)
        if not args.debug_path:
            _print_path_debug(rt)
        return 3
    if not sreach["ok"]:
        print(
            "[learn] WARN sensory→neck→motor weak — learning may stay common-mode",
            flush=True,
        )

    if not args.keep_gains:
        rt.fly.contact_gain = {}
    rt.prepare_activity()

    # Mode: CNS (default) vs legacy socket bypass.
    use_socket = bool(args.socket)
    rt.motor_socket_on = use_socket
    rt.rule.motor_socket_on = use_socket
    rt.blueprint.plasticity.motor_socket_on = use_socket
    rt.rule.socket_teacher_drive = 0.0
    rt.blueprint.plasticity.socket_teacher_drive = 0.0
    if not use_socket:
        # Closed loop: tissue only — no AimTask clamp shaping / clone, and no
        # geometric-lag oracle in the credit (steer must come from the head).
        for tgt in (rt.rule, rt.blueprint.plasticity):
            tgt.teacher_mix = 0.0
            tgt.clone_teacher = False
            tgt.bearing_credit = 0.0
    if args.teacher_mix is not None:
        tm = float(np.clip(float(args.teacher_mix), 0.0, 1.0))
        rt.rule.teacher_mix = tm
        rt.blueprint.plasticity.teacher_mix = tm
        print(f"[learn] override teacher_mix={tm} (cli)", flush=True)
    # Ensure live flags so Track / yaw_lag sample under headless Run.
    rt.sim_running = True
    rt._activity_running = True
    if hasattr(rt, "_track_reset"):
        rt._track_reset()
        rt._learn_progress_reset()

    rule = rt.rule
    dt = 1.0 / 60.0
    log_every = max(1, int(args.log_every))
    best_eps = float("inf")
    stall_bad = 0
    dead_bad = 0
    good_streak = 0
    windows = 0
    reason = "MAX_STEPS"
    last_info: dict = {}
    rt._learn_status_i = 1
    mode = "socket" if use_socket else "CNS"
    spin_on = abs(float(args.lamp_speed)) >= 0.02
    track_ready = float(np.clip(float(args.track_ready), 0.05, 1.0))

    print(
        f"[learn] mode={mode} steps≤{args.steps} lamp_speed={args.lamp_speed} "
        f"{'Track-hold≥' + f'{track_ready:.0%}' if spin_on else 'target_ε=' + f'{args.target_eps}°'} "
        f"patience={args.patience} "
        f"teacher_mix={rule.teacher_mix} "
        f"clone_teacher={bool(getattr(rule, 'clone_teacher', False))} "
        f"bearing_credit={float(getattr(rule, 'bearing_credit', 0.0)):.2f} "
        f"sense_align={float(getattr(rule, 'sense_align', 0.0)):.2f} "
        f"track_credit={float(getattr(rule, 'track_credit', 0.0)):.2f} "
        f"hebb={rule.hebb} path_credit={rule.path_credit} "
        f"hops={rt._closed_loop._resolve_hops()} (base {rt.loop_hops}) "
        f"learn_every={rt.learn_every} socket={rt.motor_socket_on}",
        flush=True,
    )

    for i in range(int(args.steps)):
        rt.fly.sim_time = int(i)
        if float(rt.lamp_speed) > 0.0:
            rt._tick_lamp(dt)
        else:
            rt._update_lamp_xyz()
        if hasattr(rt, "_update_ocelli_sensors"):
            rt._update_ocelli_sensors()
        if hasattr(rt, "_update_turret_aim_metric"):
            rt._update_turret_aim_metric()
        rt._learn_status_i = 1
        last_info = rt.tick(dt)

        if i % log_every != 0 and i != int(args.steps) - 1:
            continue

        windows += 1
        rep = rt.trainer().report(last_info)
        ed = float(rep["err_deg"])
        dw = float(rep["dw_ema"])
        n_gains = int(rep["n_gains"])
        nan_g = int(rep["nan_gains"])
        mean_abs = float(rep["mean_abs_g"])
        n_cns = int(rep["n_cns"])
        n_br = int(rep["n_bridge"])
        n_dev = int(rep["n_device"])
        n_cor = int(rep["n_corridor"])
        plast = int(rep["plast"])
        trk = float(rep.get("track", float("nan")))
        trk_s = f"{trk:.0%}" if math.isfinite(trk) else "—"
        lag = float(rep.get("yaw_lag", float("nan")))
        lag_s = f"{lag * 180.0 / math.pi:+.1f}°" if math.isfinite(lag) else "—"
        eye = float(rep.get("eye_lag", float("nan")))
        eye_s = f"{eye * 180.0 / math.pi:+6.1f}°" if math.isfinite(eye) else "     —"
        I = rep.get("ocelli_I") or [0, 0, 0]
        teach = "clone" if rep.get("clone_teacher") else "eye/Track"
        print(
            f"[learn/{mode}/{teach}] t={i:5d} ε={ed:6.1f}° Track={trk_s} lag={lag_s}  "
            f"eye={eye_s}(c={float(rep.get('eye_conf', 0)):.2f}) "
            f"I=[{I[0]:.2f},{I[1]:.2f},{I[2]:.2f}] ΔI={float(rep.get('I_lr', 0)):+.3f}  "
            f"cmd=({float(rep.get('yaw_cmd', 0)):+.2f},{float(rep.get('pitch_cmd', 0)):+.2f})  "
            f"align={float(rep.get('align', 0)):+.2f} "
            f"brake={float(rep.get('brake', 0)):.2f} "
            f"R={float(rep.get('reward', 0)):+.2f}  "
            f"Δw={dw:.4f} rpe={rep['rpe']:+.4f}  "
            f"gains={n_gains} (cns={n_cns} br={n_br} cor={n_cor} dev={n_dev} nan={nan_g}) "
            f"|g-1|={mean_abs:.4f} carve={rep.get('carve_n', 0)} "
            f"plast={plast} lit={rep.get('lit', 0)} touch={last_info.get('touched')}",
            flush=True,
        )

        if nan_g or not math.isfinite(dw) or not math.isfinite(ed) or dw > 5.0:
            reason = "UNSTABLE"
            break

        # Hold metric: under Spin — Track≥ready; static — ε≤target.
        if spin_on:
            good = bool(math.isfinite(trk) and trk >= track_ready)
        else:
            good = bool(ed <= float(args.target_eps))
        if good:
            good_streak += 1
            if good_streak >= int(args.settle):
                reason = "SUCCESS"
                break
        else:
            good_streak = 0

        if windows <= int(args.warmup):
            if math.isfinite(ed):
                best_eps = min(best_eps, ed)
            continue

        improved = math.isfinite(ed) and ed < (best_eps - float(args.min_improve))
        if improved:
            best_eps = ed
            stall_bad = 0
            dead_bad = 0
            continue

        if math.isfinite(ed):
            best_eps = min(best_eps, ed)

        if plast <= 0 and ed > float(args.target_eps) * 2.0:
            dead_bad += 1
        else:
            dead_bad = 0

        if dw < float(args.min_dw) and ed > float(args.target_eps):
            stall_bad += 1
        else:
            stall_bad = 0

        if dead_bad >= int(args.patience):
            reason = "DEAD"
            break
        if stall_bad >= int(args.patience):
            reason = "STALL"
            break

    elapsed = time.perf_counter() - t0
    ed = _eps_deg(rt)
    n_gains, nan_g, mean_abs, n_cns, n_br, n_dev, n_cor = _gain_stats(rt)
    trk_final = float(getattr(rt, "_track_score", float("nan")))
    trk_final_s = f"{trk_final:.0%}" if math.isfinite(trk_final) else "—"
    print(
        f"\n=== STOP {reason} === ε={ed:.1f}° Track={trk_final_s} best_ε={best_eps:.1f}° "
        f"gains={n_gains} (cns={n_cns} br={n_br} cor={n_cor} dev={n_dev} nan={nan_g}) "
        f"|g-1|={mean_abs:.4f} Δw={float(rt.learn_dw_ema):.4f} "
        f"elapsed={elapsed:.1f}s "
        f"({(i + 1) / max(elapsed, 1e-6):.1f} tick/s)",
        flush=True,
    )
    do_save = args.save is not None and reason not in ("REACH", "UNSTABLE")
    if do_save or (args.save_always and args.save is not None and reason not in ("REACH", "UNSTABLE")):
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        snap = rt.device_snapshot()
        save_fly(rt.fly, out, extra_meta={"shell_devices": snap})
        print(f"[learn] saved {out} gains={n_gains}", flush=True)

    if reason == "SUCCESS":
        return 0
    if reason == "MAX_STEPS":
        if spin_on:
            return 0 if (math.isfinite(trk_final) and trk_final >= track_ready * 0.5) else 1
        return 0 if ed <= float(args.target_eps) * 2.0 else 1
    if reason in ("STALL", "DEAD"):
        return 1
    if reason == "UNSTABLE":
        return 4
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

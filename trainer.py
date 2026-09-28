"""Aim plasticity: RPE-gated eligibility + river carve. Consumed by ClosedLoop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aim_task import AimTask
from blueprint import PlasticityRule

_UNSET = object()


@dataclass
class Trainer:
    """Closed-loop learning object. Owns RPE, temporal elig traces, carve sign."""

    host: Any
    task: AimTask = field(default_factory=AimTask)
    elig_trace: dict[int, float] = field(default_factory=dict)  # legacy mirror
    _elig: np.ndarray | None = field(default=None, repr=False)
    last_rpe: float = 0.0
    last_err: float = float("nan")
    last_reward: float = 0.0
    last_align: float = 0.0
    _prev_track: float = float("nan")
    _last_brake: float = 0.0
    _last_yaw_cmd: float = 0.0
    _last_carve_n: int = 0
    _last_n_touch: int = 0
    _last_eye_lag: float = float("nan")
    _last_eye_conf: float = 0.0
    _last_steer: float = 0.0
    _eye_cor_key: Any = field(default=_UNSET, repr=False)
    _eye_cor: set[int] = field(default_factory=set, repr=False)

    @staticmethod
    def _device_id_set(host: Any) -> set[int]:
        """All device neurons (ocelli, eyes, JO, turret) — internal mesh is not plastic."""
        out: set[int] = set()
        fn = getattr(host, "all_device_neuron_ids", None)
        if callable(fn):
            try:
                out |= {int(x) for x in fn()}
            except Exception:
                pass
        for name in ("_turret_flat_ids", "_ocelli_flat_ids"):
            ids = getattr(host, name, None)
            if callable(ids):
                out |= {int(x) for x in ids()}
        for d in getattr(host, "device_list", []) or []:
            if getattr(d, "kind", "") in ("eye", "jo"):
                out |= {int(x) for x in d.flat_ids()}
        return out

    @staticmethod
    def _motor_id_set(host: Any) -> set[int]:
        """Turret motor neuron ids from device wiring (no AimTask levels)."""
        out: set[int] = set()
        m_terms = getattr(host, "turret_motor_terminals", []) or []
        groups = getattr(host, "turret_motor_groups", []) or []
        for si in range(4):
            if si < len(m_terms):
                out.add(int(m_terms[si]))
            if si < len(groups):
                for nid in groups[si]:
                    out.add(int(nid))
        return out

    @staticmethod
    def _yaw_lag(host: Any) -> float:
        """Signed yaw lag (desired − actual), rad. Alive inside the aim cone."""
        lag = float(getattr(host, "_yaw_lag", float("nan")))
        if math.isfinite(lag):
            return lag
        des = None
        if callable(getattr(host, "_turret_desired_aim", None)):
            des = host._turret_desired_aim()
        else:
            task = getattr(host, "_aim_task", None)
            if task is not None and hasattr(task, "desired_aim"):
                des = task.desired_aim(host)
        if des is None:
            return float("nan")
        dyaw = float(des[0])
        yaw = float(getattr(host, "turret_yaw", 0.0) or 0.0)
        return float((dyaw - yaw + math.pi) % math.tau - math.pi)

    @staticmethod
    def _motor_yaw_cmd(host: Any) -> tuple[float, list[float]]:
        """Hop-driven yaw command (L−R over the four wing sites)."""
        raw_lvls: list[float] = []
        for si in range(4):
            fn = getattr(host, "_turret_site_motor_level_raw", None)
            raw_lvls.append(float(fn(si)) if callable(fn) else 0.0)
        while len(raw_lvls) < 4:
            raw_lvls.append(0.0)
        left = 0.5 * (raw_lvls[0] + raw_lvls[1])
        right = 0.5 * (raw_lvls[2] + raw_lvls[3])
        return float(np.clip(left - right, -1.0, 1.0)), raw_lvls

    @staticmethod
    def _ocelli_di(host: Any) -> float:
        """Ocellar L−R contrast. Telemetry / weak fallback only."""
        I = np.asarray(getattr(host, "ocelli_I", np.zeros(3)), dtype=np.float64)
        if len(I) < 3:
            I = np.pad(I, (0, 3 - len(I)))
        return float(I[1] - I[2])

    @staticmethod
    def _jo_trust(host: Any, weight: float) -> float:
        """Johnston organ: world swing smears the apparent bearing → trust less."""
        w = float(np.clip(weight, 0.0, 1.0))
        if w <= 0.0:
            return 1.0
        on = bool(getattr(host, "jo_left_on", False)) or bool(
            getattr(host, "jo_right_on", False)
        )
        if not on:
            return 1.0
        tilt = 0.5 * (
            float(getattr(host, "jo_left_tilt", 0.0) or 0.0)
            + float(getattr(host, "jo_right_tilt", 0.0) or 0.0)
        )
        prev = float(getattr(host, "_trainer_jo_prev", tilt))
        host._trainer_jo_prev = float(tilt)
        rate = abs(float(tilt) - prev)
        return float(np.clip(1.0 - w * (6.0 * rate + 0.20 * abs(tilt)), 0.5, 1.0))

    @staticmethod
    def _eye_bearing(host: Any, *, jo_weight: float = 0.0) -> tuple[float, float]:
        """Eye + proprio steer: (signed lag estimate rad, confidence 0..1).

        Eyes give the lamp azimuth as a population vector over the 12 look
        sites; turret proprioception gives the current barrel yaw. Their
        difference is what the motors must null. No lamp XYZ, no AimTask.
        """
        try:
            from compound_eyes import eye_lamp_azimuth
        except Exception:
            return float("nan"), 0.0
        out = eye_lamp_azimuth(host)
        if out is None:
            return float("nan"), 0.0
        az, conf = out
        if not math.isfinite(az):
            return float("nan"), 0.0
        yaw = float(getattr(host, "turret_yaw", 0.0) or 0.0)
        lag = float((float(az) - yaw + math.pi) % math.tau - math.pi)
        conf = float(conf) * Trainer._jo_trust(host, jo_weight)
        return lag, float(np.clip(conf, 0.0, 1.0))

    @staticmethod
    def _steer_side_scale(
        motor_id: int,
        steer: float,
        left: set[int],
        right: set[int],
    ) -> float:
        """Boost the side the sensory steer asks for; damp the other."""
        if abs(steer) < 0.03 or not left or not right:
            return 1.0
        mid = int(motor_id)
        on_left = mid in left
        on_right = mid in right
        if not on_left and not on_right:
            return 1.0
        prefer_left = steer > 0.0
        on_pref = (prefer_left and on_left) or ((not prefer_left) and on_right)
        return 1.3 if on_pref else 0.55

    def _eye_corridor(self, host: Any) -> set[int]:
        """Neurons downstream of the eye clamps inside the hop domain.

        Used to prefer eye→motor edges when the plastic row budget is tight,
        so credit does not collapse onto the short ocellar path.
        """
        key = getattr(host.fly, "_activity_graph_key", None)
        if getattr(self, "_eye_cor_key", None) == key:
            return self._eye_cor  # type: ignore[attr-defined]
        seeds: set[int] = set()
        for d in getattr(host, "device_list", []) or []:
            if getattr(d, "kind", "") == "eye" and getattr(d, "enabled", False):
                seeds.update(int(x) for x in d.flat_ids())
        out: set[int] = set()
        if seeds:
            pre, post = host.fly.activity_graph_edges()
            if len(pre):
                adj: dict[int, list[int]] = {}
                for a, b in zip(pre.tolist(), post.tolist()):
                    adj.setdefault(int(a), []).append(int(b))
                out = set(seeds)
                frontier = list(seeds)
                for _ in range(int(getattr(self.rule(), "elig_depth", 10)) + 6):
                    nxt: list[int] = []
                    for node in frontier:
                        for dst in adj.get(node, ()):
                            if dst not in out:
                                out.add(dst)
                                nxt.append(dst)
                    if not nxt:
                        break
                    frontier = nxt
        self._eye_cor_key = key
        self._eye_cor = out
        return out

    def rule(self) -> PlasticityRule:
        return getattr(self.host, "rule", None) or PlasticityRule()

    def err(self) -> float:
        return float(self.task.err(self.host))

    def rpe(self, prev_err: float, err: float | None = None) -> float:
        """Reward prediction error: improvement = prev_err − err (>0 when better)."""
        e = float(err) if err is not None else self.err()
        if not math.isfinite(prev_err) or not math.isfinite(e):
            return 0.0
        return float(prev_err) - float(e)

    def _ensure_elig(self, n: int) -> np.ndarray:
        elig = self._elig
        if elig is None or len(elig) != n:
            elig = np.zeros(n, dtype=np.float32)
            # Migrate sparse dict once if present.
            if self.elig_trace:
                for nid, e in self.elig_trace.items():
                    i = int(nid)
                    if 0 <= i < n:
                        elig[i] = float(e)
                self.elig_trace.clear()
            self._elig = elig
        return elig

    def _decay_traces(self, act_ids: np.ndarray, act: np.ndarray, *, tau: float) -> None:
        """Temporal eligibility: e ← τ·e + (1−τ)·a (dense float32)."""
        tau = float(np.clip(tau, 0.0, 0.999))
        keep = 1.0 - tau
        n = len(act)
        if n == 0:
            return
        elig = self._ensure_elig(n)
        contiguous = (
            len(act_ids) == n
            and int(act_ids[0]) == 0
            and int(act_ids[-1]) == n - 1
        )
        if contiguous:
            elig *= tau
            lit = act >= 0.02
            if np.any(lit):
                elig[lit] += np.float32(keep) * act[lit]
            elig[elig < 1e-4] = 0.0
            return
        # Sparse id fallback.
        elig *= tau
        for i, nid in enumerate(act_ids.tolist()):
            a = float(act[i])
            if a < 0.02:
                continue
            i_n = int(nid)
            if 0 <= i_n < n:
                elig[i_n] += keep * a
            else:
                # Non-index id: keep dict sidecar.
                prev = float(self.elig_trace.get(i_n, 0.0))
                self.elig_trace[i_n] = tau * prev + keep * a
        elig[elig < 1e-4] = 0.0
        dead = [nid for nid, e in self.elig_trace.items() if float(e) < 1e-4]
        for nid in dead:
            self.elig_trace.pop(nid, None)

    def _trace_eligible(self, nid: int, *, floor: float = 0.05) -> bool:
        i = int(nid)
        elig = self._elig
        if elig is not None and 0 <= i < len(elig):
            return float(elig[i]) >= float(floor)
        return float(self.elig_trace.get(i, 0.0)) >= float(floor)

    def _trace_n(self) -> int:
        n = 0
        if self._elig is not None:
            n += int(np.count_nonzero(self._elig >= 1e-4))
        n += len(self.elig_trace)
        return n

    def step(self, prev_err: float) -> dict:
        """One plasticity update. Track/ΔI reward by default; optional AimTask clone."""
        h = self.host
        out = {"rpe": 0.0, "err": float("nan"), "n_touch": 0, "dw_abs": 0.0}
        if not getattr(h, "learn_on", False) or not getattr(h, "turret_on", False):
            h._learn_flash_amp = float(getattr(h, "_learn_flash_amp", 0.0)) * 0.85
            return out
        rule = self.rule()
        clone = bool(getattr(rule, "clone_teacher", False))
        targets: dict[int, float] = {}
        if clone:
            targets = self.task.motor_neuron_targets(h)
            if not targets and hasattr(h, "_turret_motor_neuron_targets"):
                targets = h._turret_motor_neuron_targets()
        motor_ids = (
            set(int(k) for k in targets.keys())
            if targets
            else self._motor_id_set(h)
        )
        if not motor_ids:
            h._learn_flash_amp = float(getattr(h, "_learn_flash_amp", 0.0)) * 0.85
            return out
        if hasattr(h, "_ensure_atlas_recipe"):
            h._ensure_atlas_recipe()
        pre, post = h.fly.activity_graph_edges()
        if len(pre) == 0:
            h._learn_flash_amp = float(getattr(h, "_learn_flash_amp", 0.0)) * 0.85
            return out
        # soma_activity is source of truth after sparse hop/decay writeback.
        ids = h.fly.soma_ids
        act = h.fly.soma_activity
        if len(ids) == 0 or len(act) != len(ids):
            h.fly.rebuild_activity_cache()
            ids = h.fly.soma_ids
            act = h.fly.soma_activity
            if len(ids) == 0 or len(act) != len(ids):
                return out
        err = float(getattr(h, "turret_aim_err", float("nan")))
        if not math.isfinite(err):
            if hasattr(h, "_update_turret_aim_metric"):
                h._update_turret_aim_metric()
            err = float(getattr(h, "turret_aim_err", float("nan")))
        if not math.isfinite(err):
            return out

        rpe = self.rpe(prev_err, err)
        self.last_rpe = float(rpe)
        self.last_err = float(err)
        out["rpe"] = float(rpe)
        out["err"] = float(err)
        # Scale gates to learn cadence: plasticity fires every N frames, so the
        # meaningful Δε is across that window — not a single ~50 ms actuate.
        learn_every = max(
            1,
            int(
                getattr(h, "learn_every", None)
                or getattr(rule, "learn_every", 1)
                or 1
            ),
        )
        # ~0.1° × N → ~0.9° dead-zone at learn_every=8.
        rpe_dead = 0.002 * float(learn_every)
        rpe_sign = float(np.sign(rpe)) if abs(rpe) > rpe_dead else 0.0
        # Full-scale RPE ≈ tens of degrees over a learn window.
        rpe_norm = 0.15 * math.pi * math.sqrt(float(learn_every))
        rpe_mag = float(np.clip(abs(rpe) / max(rpe_norm, 1e-6), 0.0, 1.0))
        if rpe_sign == 0.0:
            rpe_mag = 0.0

        # Steer comes from the head, not from the lamp: eye population vector
        # (where the lamp is) minus turret proprioception (where we point).
        yaw_cmd, _raw_lvls = self._motor_yaw_cmd(h)
        di_lr = self._ocelli_di(h)
        eye_lag, eye_conf = self._eye_bearing(
            h, jo_weight=float(getattr(rule, "jo_credit", 0.0))
        )
        eye_alive = math.isfinite(eye_lag) and eye_conf >= 0.12
        if eye_alive:
            steer_n = float(
                np.clip(eye_lag / (0.5 * math.pi), -1.0, 1.0)
            ) * float(eye_conf)
            steer_rad = float(eye_lag)
        else:
            # Eyes dark: fall back to the (weak) ocellar contrast, never to
            # the geometric lag — that would be the oracle again.
            steer_n = 0.35 * float(np.tanh(di_lr * 4.0))
            steer_rad = float(steer_n) * 0.5 * math.pi
        align = float(np.clip(steer_n * float(yaw_cmd), -1.0, 1.0))
        self._last_eye_lag = float(eye_lag)
        self._last_eye_conf = float(eye_conf)
        self._last_steer = float(steer_n)
        # Geometric lag: telemetry, and an *opt-in* reward term only.
        yaw_lag = self._yaw_lag(h)
        bc_w_raw = float(np.clip(getattr(rule, "bearing_credit", 0.0), 0.0, 1.0))
        if bc_w_raw > 0.0 and math.isfinite(yaw_lag):
            lag_n = float(np.clip(yaw_lag / (0.5 * math.pi), -1.0, 1.0))
            bearing_align = float(np.clip(lag_n * float(yaw_cmd), -1.0, 1.0))
        else:
            bearing_align = 0.0

        track = float(getattr(h, "_track_score", float("nan")))
        prev_track = float(self._prev_track)
        d_track = 0.0
        if math.isfinite(track) and math.isfinite(prev_track):
            d_track = float(track) - float(prev_track)
        if math.isfinite(track):
            self._prev_track = float(track)
        sa_w = float(np.clip(getattr(rule, "sense_align", 0.0), 0.0, 1.0))
        bc_w = bc_w_raw
        tc_w = float(np.clip(getattr(rule, "track_credit", 0.0), 0.0, 1.0))
        w_sum = sa_w + bc_w + tc_w
        if w_sum > 1.0:
            sa_w, bc_w, tc_w = sa_w / w_sum, bc_w / w_sum, tc_w / w_sum
        rpe_w = max(0.0, 1.0 - sa_w - bc_w - tc_w)
        rpe_comp = float(np.clip(rpe / max(rpe_norm, 1e-6), -1.0, 1.0))
        track_comp = float(np.clip(d_track / 0.08, -1.0, 1.0))
        reward = (
            rpe_w * rpe_comp
            + tc_w * track_comp
            + bc_w * bearing_align
            + sa_w * float(align)
        )

        # Hold / overshoot from the *sensory* steer + cmd (learning only).
        aim_half = math.radians(15.0)
        # Cone comes from what the head can see, not from the lamp metric.
        in_cone = bool(abs(steer_rad) <= aim_half)
        sense_amp = abs(float(np.tanh(di_lr * 4.0)))
        lag_amp = abs(float(steer_n))
        # Need turn toward nulling the seen bearing: steer>0 → need +yaw_cmd.
        steer_dead = math.radians(3.0)
        need_left = bool(steer_rad > steer_dead)
        need_right = bool(steer_rad < -steer_dead)
        disagree = bool(
            (need_left and yaw_cmd < -0.03)
            or (need_right and yaw_cmd > 0.03)
        )
        hold_brake_w = float(np.clip(getattr(rule, "hold_brake", 0.55), 0.0, 1.5))
        wrong_w = float(np.clip(getattr(rule, "wrong_decay", 0.65), 0.0, 1.5))
        brake = 0.0
        if rpe_sign < 0.0 and abs(yaw_cmd) > 0.05:
            brake = max(brake, hold_brake_w * (0.45 + 0.55 * rpe_mag))
        # Budget scales with remaining |lag| (not ΔI — ΔI dies in cone).
        if in_cone or lag_amp < 0.35:
            cone_budget = 0.05 + 0.55 * lag_amp + 0.15 * sense_amp
            excess_c = max(0.0, abs(yaw_cmd) - cone_budget)
            brake = max(
                brake, hold_brake_w * float(np.clip(excess_c / 0.10, 0.0, 1.0))
            )
        if disagree:
            brake = max(
                brake,
                wrong_w * (0.50 + 0.50 * min(1.0, lag_amp + 0.2)),
            )
        brake = float(np.clip(brake, 0.0, 1.0))

        hold_comp = 0.0
        if rpe_sign < 0.0 and abs(yaw_cmd) > 0.05:
            hold_comp -= 0.45 * rpe_mag
        if in_cone and align > 0.0:
            hold_comp += 0.50 * (1.0 - abs(steer_rad) / max(aim_half, 1e-6))
        if disagree:
            hold_comp -= 0.40 * min(1.0, lag_amp + 0.15)
        hb_w = float(np.clip(getattr(rule, "hold_credit", 0.20), 0.0, 0.6))
        reward = (1.0 - hb_w) * float(reward) + hb_w * float(hold_comp)

        self.last_reward = float(reward)
        self.last_align = float(align)
        self._last_brake = float(brake)
        self._last_yaw_cmd = float(yaw_cmd)
        out["reward"] = float(reward)
        out["align"] = float(align)
        out["bearing_align"] = float(bearing_align)
        out["eye_lag"] = float(eye_lag)
        out["eye_conf"] = float(eye_conf)
        out["steer"] = float(steer_n)
        out["d_track"] = float(d_track)
        out["di_lr"] = float(di_lr)
        out["yaw_lag"] = float(yaw_lag) if math.isfinite(yaw_lag) else float("nan")
        out["yaw_cmd"] = float(yaw_cmd)
        out["brake"] = float(brake)
        out["in_cone"] = bool(in_cone)
        out["clone_teacher"] = bool(clone)
        rew_dead = 0.02
        rew_sign = float(np.sign(reward)) if abs(reward) > rew_dead else 0.0
        rew_mag = float(np.clip(abs(reward), 0.0, 1.0))
        gate_sign = rpe_sign if clone else rew_sign
        gate_mag = rpe_mag if clone else rew_mag
        carve_mag_min = 0.015
        self._last_carve_n = 0
        # Temporal eligibility (not only BFS depth).
        elig_tau = float(getattr(rule, "elig_tau", 0.88))
        self._decay_traces(ids, act, tau=elig_tau)

        device = self._device_id_set(h)
        eye_cor = self._eye_corridor(h)
        upstream = h._motor_eligibility_upstream(
            motor_ids, depth=int(rule.elig_depth)
        )
        if not upstream:
            upstream = set(motor_ids)
        # Expand BFS upstream with temporal-trace neurons that touch motors/path.
        elig = self._elig
        if elig is not None and len(elig):
            for nid in np.flatnonzero(elig >= 0.08).tolist():
                nid = int(nid)
                if nid in upstream or nid in motor_ids:
                    upstream.add(nid)
        else:
            for nid, e in self.elig_trace.items():
                if e >= 0.08 and (nid in upstream or nid in motor_ids):
                    upstream.add(int(nid))
        corridor = h._corridor_neuron_ids() if hasattr(h, "_corridor_neuron_ids") else set()

        decay = float(rule.w_decay)
        off_decay = float(rule.offpath_decay)
        if (decay > 0 or off_decay > 0) and h.fly.contact_gain:
            for key in list(h.fly.contact_gain.keys()):
                cur = float(h.fly.contact_gain[key])
                if not math.isfinite(cur):
                    h.fly.contact_gain.pop(key, None)
                    continue
                a, b = int(key[0]), int(key[1])
                # Empty corridor ≠ "everything is off-path" — that was erasing
                # the whole gain table while scope was off.
                if corridor:
                    on_bed = (a in corridor) or (b in corridor)
                    d = decay if on_bed else max(decay, off_decay)
                else:
                    d = decay
                nxt = cur + d * (1.0 - cur)
                h.fly.set_contact_gain(
                    a, b, nxt, lo=float(rule.w_min), hi=float(rule.w_max)
                )

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
        eye_arr = (
            np.fromiter(eye_cor, dtype=np.int32, count=len(eye_cor))
            if eye_cor
            else np.zeros((0,), dtype=np.int32)
        )
        path_on = float(rule.path_credit) > 0.0
        neck_gate = bool(corridor) and path_on

        # L/R motor competition sets (sites 0,1 = left; 2,3 = right).
        left_motors, right_motors = self._lr_motor_sets(h, targets)

        if contiguous:
            valid = (pre >= 0) & (post >= 0) & (pre < n_max) & (post < n_max)
            pre_i = pre[valid]
            post_i = post[valid]
            # BFS upstream OR temporal elig on pre (dense array lookup).
            elig_arr = self._ensure_elig(n_max)
            tr_ok = elig_arr[pre_i] >= 0.05
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
                if eye_arr.size:
                    # Keep the eye→motor corridor when the budget is tight.
                    score = score + 1.5 * np.isin(pre_h, eye_arr).astype(np.float32)
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
                if b not in upstream and not self._trace_eligible(a):
                    continue
                if a not in upstream and not self._trace_eligible(a):
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

        for a, b, a_pre, a_post in iter_rows:
            a, b = int(a), int(b)
            if a in device and b in device:
                continue
            dw = 0.0
            tgt = targets.get(b) if clone else None
            if tgt is not None:
                # Legacy: clone AimTask motor targets into contact_gain.
                dw = lr * float(a_pre) * (float(tgt) - float(a_post))
                if corridor and (a in corridor or b in corridor):
                    dw *= 1.35
                dw = self._lr_compete_scale(
                    dw, b, rpe_sign, left_motors, right_motors, targets
                )
            elif (not clone) and b in motor_ids:
                # Acquire toward the side the *eyes* ask for; scrub the cmd
                # that fights them. No geometric lag anywhere in this branch.
                prefer_left = (
                    need_left if (need_left or need_right) else (steer_n > 0.0)
                )
                side = self._steer_side_scale(
                    b, float(steer_n), left_motors, right_motors
                )
                hebb_r = float(a_pre) * float(a_post) * float(reward)
                drive = 0.0
                approach = abs(steer_rad) / max(aim_half, 1e-6)
                taper = float(np.clip(0.22 + 0.78 * min(approach, 1.0), 0.22, 1.0))
                on_pref = (prefer_left and b in left_motors) or (
                    (not prefer_left) and b in right_motors
                )
                cmd_left = yaw_cmd >= 0.0
                on_cmd = (cmd_left and b in left_motors) or (
                    (not cmd_left) and b in right_motors
                )

                acq = 0.0
                drive_amp = max(lag_amp, 0.35 * sense_amp)
                if drive_amp > 0.04:
                    amp = max(float(reward), 0.65 * drive_amp, 0.08) * taper
                    if on_pref:
                        acq = float(a_pre) * (0.95 - float(a_post)) * amp
                    else:
                        acq = -0.55 * float(a_pre) * float(a_post) * amp
                    if rpe_sign < 0.0 and on_cmd and lag_amp < 0.08:
                        acq *= 0.25
                if not (need_left or need_right):
                    # Eyes say centred: no side to acquire, only a cmd to null.
                    acq *= 0.15

                scrub = 0.0
                if brake > 0.08 and abs(yaw_cmd) > 0.03:
                    kill = brake * (0.90 + 0.55 * min(1.0, abs(yaw_cmd)))
                    if on_cmd:
                        # ``brake`` already means "cmd exceeds what the eyes
                        # ask for" — that includes the centred case, where
                        # there is no side to disagree with and ε may still be
                        # falling. Gating on disagree/RPE let the turret keep
                        # a full-scale cmd through the cone and fly past.
                        scrub = -float(a_pre) * float(a_post) * kill
                    elif disagree and on_pref:
                        scrub = float(a_pre) * (0.9 - float(a_post)) * brake * 0.85

                if brake > 0.08 and scrub != 0.0:
                    # Overshoot / wrong-side: scrub dominates.
                    drive = 0.80 * scrub + 0.20 * acq
                else:
                    drive = acq
                # Extra reverse push when cmd fights lag.
                if disagree and on_pref and drive_amp > 0.04:
                    drive += (
                        float(a_pre)
                        * (0.95 - float(a_post))
                        * max(0.12, 0.55 * drive_amp)
                        * taper
                    )
                dw = lr * (0.20 * hebb_r + 0.80 * drive) * side
                if corridor and (a in corridor or b in corridor):
                    dw *= 1.25
            elif path_on and (b in upstream or self._trace_eligible(a)):
                if neck_gate and a not in corridor and b not in corridor:
                    continue
                if brake > 0.35:
                    # Overshoot cleanup is motor-local — don't flood path Δw.
                    continue
                err_n = float(np.clip(abs(steer_rad) / (0.5 * math.pi), 0.0, 1.0))
                dw = (
                    float(rule.path_credit)
                    * lr
                    * float(a_pre)
                    * err_n
                    * (float(a_pre) - float(a_post))
                )
                if corridor and (a in corridor or b in corridor):
                    dw *= 1.5
                if eye_cor and a in eye_cor:
                    dw *= 1.35
                # Gate path credit by reward (Track/ΔI) or ε-RPE (clone).
                if gate_sign < 0:
                    dw *= 0.15
                elif gate_sign > 0:
                    dw *= 0.5 + 0.5 * gate_mag
                else:
                    dw *= 0.08
            else:
                continue
            if hebb > 0.0 and gate_sign != 0.0:
                dw += hebb * float(a_pre) * float(a_post) * gate_sign
            if abs(dw) < 1e-7:
                continue
            dw = float(np.clip(dw, -dw_cap, dw_cap))
            if not math.isfinite(dw):
                continue
            # Hard stop: runaway gains (hop overflow risk).
            if dw_abs > 3.5:
                break
            h.fly.adjust_contact_gain(
                a, b, dw, lo=float(rule.w_min), hi=float(rule.w_max)
            )
            dw_abs += abs(dw)
            n_touch += 1
            flash_a.append(a)
            flash_b.append(b)

        # Carve river × gate_sign. Mute while braking / far from hold —
        # cleanup is motor-local; carving the whole corridor at |lag|≫0 runs away.
        carve = float(rule.carve_lr)
        carve_n = 0
        dw_ema_now = float(getattr(h, "learn_dw_ema", 0.0) or 0.0)
        carve_ok = (
            carve > 0.0
            and corridor
            and len(pre) > 0
            and gate_sign != 0.0
            and gate_mag > carve_mag_min
            and brake < 0.35
            and not disagree
            and lag_amp < 0.55
            and abs(steer_rad) < math.radians(55.0)
            and dw_ema_now < 1.8
        )
        if carve_ok:
            err_n = float(np.clip(abs(steer_rad) / (0.5 * math.pi), 0.0, 1.0))
            act_lo = float(rule.channel_act_min)
            cor_arr = np.fromiter(corridor, dtype=np.int32, count=len(corridor))
            carve_scale = carve * gate_sign * gate_mag * (0.35 + 0.65 * err_n)
            # Soften when already near cone — motor plasticity owns the hold.
            if in_cone or lag_amp < 0.20:
                carve_scale *= 0.35
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
                carve_n += 1
                if len(flash_a) < 400:
                    flash_a.append(a)
                    flash_b.append(b)

        self._last_carve_n = int(carve_n)
        self._last_n_touch = int(n_touch)
        out["n_touch"] = int(n_touch)
        out["carve_n"] = int(carve_n)
        out["dw_abs"] = float(dw_abs)
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
        return out

    @staticmethod
    def _lr_motor_sets(
        host: Any, targets: dict[int, float]
    ) -> tuple[set[int], set[int]]:
        left: set[int] = set()
        right: set[int] = set()
        m_terms = getattr(host, "turret_motor_terminals", []) or []
        groups = getattr(host, "turret_motor_groups", []) or []
        for si in range(4):
            bucket = left if si in (0, 1) else right
            if si < len(m_terms):
                bucket.add(int(m_terms[si]))
            if si < len(groups):
                for nid in groups[si]:
                    bucket.add(int(nid))
        # Fallback: split targets keys by index order.
        if not left and not right and targets:
            keys = sorted(int(k) for k in targets.keys())
            mid = max(1, len(keys) // 2)
            left = set(keys[:mid])
            right = set(keys[mid:])
        return left, right

    @staticmethod
    def _lr_compete_scale(
        dw: float,
        motor_id: int,
        rpe_sign: float,
        left: set[int],
        right: set[int],
        targets: dict[int, float],
    ) -> float:
        """When improving, boost the side the teacher prefers; damp the other."""
        if rpe_sign == 0.0 or not left or not right:
            return dw
        mid = int(motor_id)
        tgt = float(targets.get(mid, 0.5))
        on_left = mid in left
        on_right = mid in right
        if not on_left and not on_right:
            return dw
        # Teacher wants this side high when tgt > 0.5.
        prefer_this = (tgt >= 0.5 and on_left) or (tgt < 0.5 and on_right)
        if rpe_sign > 0:
            return dw * (1.25 if prefer_this else 0.55)
        # Worsening: pull back the preferred side slightly.
        return dw * (0.7 if prefer_this else 1.0)

    # --- public probes / report (learn_console; no _probe_* required) ----------

    def probe_ocelli_reach(self, *, hops: int = 10) -> dict:
        fn = getattr(self.host, "_probe_ocelli_motor_reach", None)
        if callable(fn):
            return dict(fn(hops=hops))
        return {"ok": False, "hit": 0, "motors": 0, "best": 0.0}

    def probe_sensory_reach(self, *, hops: int = 12) -> dict:
        fn = getattr(self.host, "_probe_turret_sensory_motor_reach", None)
        if callable(fn):
            return dict(fn(hops=hops))
        return {"ok": False, "hit": 0, "motors": 0, "best": 0.0, "lr_gap": 0.0}

    def report(self, info: dict | None = None) -> dict:
        """Telemetry snapshot for learn_console / status plaque."""
        h = self.host
        info = info or {}
        err = float(getattr(h, "turret_aim_err", float("nan")))
        device = self._device_id_set(h)
        corridor = (
            h._corridor_neuron_ids() if hasattr(h, "_corridor_neuron_ids") else set()
        )
        gains = getattr(h.fly, "contact_gain", {}) or {}
        n_cns = n_br = n_dev = n_cor = 0
        abs_sum = 0.0
        abs_n = 0
        nan_g = 0
        for (a, b), w in gains.items():
            wf = float(w)
            if not math.isfinite(wf):
                nan_g += 1
                continue
            abs_sum += abs(wf - 1.0)
            abs_n += 1
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
        rule = self.rule()
        # CNS drive snapshot (hop motors, not socket W).
        I = np.asarray(getattr(h, "ocelli_I", np.zeros(3)), dtype=np.float64)
        if len(I) < 3:
            I = np.pad(I, (0, 3 - len(I)))
        raw_lvls = []
        for si in range(4):
            fn = getattr(h, "_turret_site_motor_level_raw", None)
            raw_lvls.append(float(fn(si)) if callable(fn) else 0.0)
        left = 0.5 * (raw_lvls[0] + raw_lvls[1])
        right = 0.5 * (raw_lvls[2] + raw_lvls[3])
        lo = 0.5 * (raw_lvls[0] + raw_lvls[2])
        hi = 0.5 * (raw_lvls[1] + raw_lvls[3])
        yaw_cmd = float(np.clip(left - right, -1.0, 1.0))
        pitch_cmd = float(np.clip(lo - hi, -1.0, 1.0))
        track = float(getattr(h, "_track_score", float("nan")))
        sock_on = bool(getattr(h, "motor_socket_on", False))
        return {
            "err": err,
            "err_deg": (err * 180.0 / math.pi) if math.isfinite(err) else float("nan"),
            "rpe": float(self.last_rpe),
            "dw_ema": float(getattr(h, "learn_dw_ema", 0.0)),
            "n_gains": len(gains),
            "n_touch": int(getattr(self, "_last_n_touch", 0) or info.get("n_touch", 0) or 0),
            "carve_n": int(getattr(self, "_last_carve_n", 0) or 0),
            "n_cns": n_cns,
            "n_bridge": n_br,
            "n_device": n_dev,
            "n_corridor": n_cor,
            "corridor_n": len(corridor),
            "nan_gains": nan_g,
            "mean_abs_g": (abs_sum / abs_n) if abs_n else 0.0,
            "plast": int(getattr(h, "_learn_flash_n", 0)),
            "hop_edges": int(info.get("edges", 0)),
            "lit": int(info.get("lit", 0)),
            "touched": int(info.get("touched", 0)),
            "teacher_mix": float(rule.teacher_mix),
            "clone_teacher": bool(getattr(rule, "clone_teacher", False)),
            "sense_align": float(getattr(rule, "sense_align", 0.0)),
            "track_credit": float(getattr(rule, "track_credit", 0.0)),
            "hebb": float(rule.hebb),
            "path_credit": float(rule.path_credit),
            "carve_lr": float(rule.carve_lr),
            "elig_trace_n": self._trace_n(),
            "sock_on": sock_on,
            "ocelli_I": [float(I[0]), float(I[1]), float(I[2])],
            "I_lr": float(I[1] - I[2]),
            "motor_raw": [float(x) for x in raw_lvls],
            "yaw_cmd": yaw_cmd,
            "pitch_cmd": pitch_cmd,
            "track": track,
            "yaw_lag": float(self._yaw_lag(h)),
            "eye_lag": float(self._last_eye_lag),
            "eye_conf": float(self._last_eye_conf),
            "steer": float(self._last_steer),
            "bearing_credit": float(getattr(rule, "bearing_credit", 0.0)),
            "reward": float(self.last_reward),
            "align": float(self.last_align),
            "brake": float(self._last_brake),
            "yaw": float(getattr(h, "turret_yaw", 0.0) or 0.0),
            "pitch": float(getattr(h, "turret_pitch", 0.0) or 0.0),
        }

    def format_report_line(self, info: dict | None = None) -> str:
        r = self.report(info)
        ed = r["err_deg"]
        aim_s = f"{ed:.1f}°" if math.isfinite(ed) else "nan"
        dw = r["dw_ema"]
        dw_s = f"{dw:.4f}" if math.isfinite(dw) else "nan/inf"
        warn = ""
        if r["nan_gains"] or not math.isfinite(dw) or not math.isfinite(ed):
            warn = "  WARN=unstable"
        mode = "socket" if r["sock_on"] else "CNS"
        teach = "clone" if r.get("clone_teacher") else "eye/Track"
        trk = r["track"]
        trk_s = f"{trk:.0%}" if math.isfinite(float(trk)) else "—"
        I = r["ocelli_I"]
        lag = float(r.get("yaw_lag", float("nan")))
        lag_s = f"{lag * 180.0 / math.pi:+.1f}°" if math.isfinite(lag) else "—"
        eye = float(r.get("eye_lag", float("nan")))
        eye_s = f"{eye * 180.0 / math.pi:+.1f}°" if math.isfinite(eye) else "—"
        return (
            f"[learn/{mode}/{teach}] ε={aim_s} Track={trk_s} lag={lag_s}  "
            f"eye={eye_s}(c={float(r.get('eye_conf', 0.0)):.2f}) "
            f"I=[{I[0]:.2f},{I[1]:.2f},{I[2]:.2f}] ΔI_lr={r['I_lr']:+.3f}  "
            f"yaw_cmd={r['yaw_cmd']:+.3f} align={r['align']:+.2f} "
            f"brake={r['brake']:.2f} R={r['reward']:+.2f}  "
            f"Δw={dw_s} rpe={r['rpe']:+.4f}  "
            f"touch={r['n_touch']} carve={r['carve_n']}  "
            f"gains={r['n_gains']} (cns={r['n_cns']} br={r['n_bridge']} "
            f"cor={r['n_corridor']}/{r['corridor_n']} dev={r['n_device']}"
            + (f" nan={r['nan_gains']}" if r["nan_gains"] else "")
            + f")  |g-1|={r['mean_abs_g']:.4f}  "
            f"plast={r['plast']} hop={r['hop_edges']} lit={r['lit']}  "
            f"trace={r['elig_trace_n']}{warn}"
        )

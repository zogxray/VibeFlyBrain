"""Closed-loop adult step: sense → hop → actuate over blueprint.device_list.

Interactive shell prefers ``host.loop_hops`` (``hops_ui``) for draw FPS;
when Learn is on, hops are floored to ``max(8, scope.hops_floor)`` — the
*slowest* sensor group's geodesic — so every sensor→motor wave can actually
arrive, not just the nearest modality. Sensors are re-clamped every frame
(activity decays; a stale sense-cache was letting the plant go dark).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aim_task import AimTask
from device import OcelliDevice, TurretDevice
import numpy as np


@dataclass
class ClosedLoop:
    """Device-agnostic closed loop. Shell/runtime is a thin host."""

    host: Any
    task: AimTask = field(default_factory=AimTask)
    trainer: Any | None = None  # Trainer; lazy-bound
    _frame_i: int = 0
    _last_sense: dict[int, dict[int, float]] = field(default_factory=dict)

    def _devices(self) -> list[Any]:
        h = self.host
        if hasattr(h, "iter_devices"):
            return [d for d in h.iter_devices()]
        return list(getattr(getattr(h, "blueprint", None), "device_list", []) or [])

    def _sync_turret_device_pose(self) -> None:
        """Pull host yaw/pitch into typed TurretDevice before actuate."""
        h = self.host
        for d in self._devices():
            if isinstance(d, TurretDevice) and d.enabled:
                d.yaw = float(getattr(h, "turret_yaw", d.yaw))
                d.pitch = float(getattr(h, "turret_pitch", d.pitch))

    def sense_all(self, *, sensors_only: bool = False) -> int:
        """Stimulate CNS from every enabled device sense().

        ``sensors_only``: skip turret proprio (post-actuate path can be turret-only).
        """
        h = self.host
        n = 0
        for d in self._devices():
            if not getattr(d, "enabled", True):
                continue
            if isinstance(d, OcelliDevice):
                if not getattr(h, "ocelli_on", True):
                    continue
            elif isinstance(d, TurretDevice):
                if sensors_only:
                    continue
                if not getattr(h, "turret_on", True):
                    continue
            else:
                if sensors_only and getattr(d, "kind", "") != "ocelli":
                    # unknown device: still sense unless sensors_only for non-ocelli
                    pass
            levels = d.sense(h)
            if not levels:
                continue
            # Always re-clamp: hop decay clears activity each beat; skipping
            # when lamp/pose look "unchanged" lets motors go dark mid-run.
            self._last_sense[id(d)] = {int(k): float(v) for k, v in levels.items()}
            n += int(h.fly.stimulate_neurons(levels))
        return n

    def sense_turret_proprio(self) -> int:
        """Post-move proprio only (avoids double ocelli sense)."""
        h = self.host
        if not getattr(h, "turret_on", True):
            return 0
        n = 0
        for d in self._devices():
            if not isinstance(d, TurretDevice) or not getattr(d, "enabled", True):
                continue
            levels = d.sense(h)
            if not levels:
                continue
            self._last_sense[id(d)] = {int(k): float(v) for k, v in levels.items()}
            n += int(h.fly.stimulate_neurons(levels))
        return n

    def actuate_all(self, dt: float) -> None:
        h = self.host
        self._sync_turret_device_pose()
        for d in self._devices():
            if not getattr(d, "enabled", True):
                continue
            if isinstance(d, TurretDevice):
                if not getattr(h, "turret_on", True):
                    continue
                d.actuate(None, dt=float(dt), world=h)
            else:
                d.actuate(None, dt=float(dt), world=h)

    def _resolve_hops(self) -> int:
        h = self.host
        rule = getattr(h, "rule", None)
        base = getattr(h, "loop_hops", None)
        if base is None:
            base = getattr(rule, "hops", 8) if rule else 8
        base = max(1, int(base))
        # Learning needs the full sensor→motor geodesic; UI hops_ui=1 is only
        # for draw when Learn is off. View Detail can thin hops when not learning.
        learn = bool(
            getattr(h, "learn_on", False) and getattr(h, "turret_on", False)
        )
        if not learn:
            kn = (
                h._view_detail_knobs()
                if hasattr(h, "_view_detail_knobs")
                else None
            )
            if kn is not None:
                return max(1, int(kn.get("hops", base)))
            return base
        floor = 8
        scope = getattr(h, "_training_scope", None)
        # Slowest sensor group, not the nearest one: the eye geodesic is far
        # longer than turret proprio, and hopping the proprio count leaves the
        # optic wave stuck in the lobe.
        need = getattr(scope, "hops_floor", None) if scope is not None else None
        if need is None and scope is not None:
            need = getattr(scope, "best_hops", None)
        if need is not None:
            try:
                floor = max(floor, int(need))
            except (TypeError, ValueError):
                pass
        return max(base, floor)

    def _resolve_learn_every(self) -> int:
        h = self.host
        if hasattr(h, "_view_detail_knobs"):
            return max(1, int(h._view_detail_knobs()["learn_every"]))
        override = getattr(h, "learn_every", None)
        if override is not None:
            return max(1, int(override))
        rule = getattr(h, "rule", None)
        return max(1, int(getattr(rule, "learn_every", 1) if rule else 1))

    def step(self, dt: float) -> dict:
        """One adult frame: sense → hop → actuate → (learn every N)."""
        import time as _time

        h = self.host
        self._frame_i = int(self._frame_i) + 1
        # Sync device wiring only when UI/Fuse dirtied the shell fields.
        if getattr(h, "_devices_dirty", True):
            if hasattr(h, "_sync_shell_fields_to_devices"):
                h._sync_shell_fields_to_devices()
            h._devices_dirty = False
        prev_err = float(getattr(h, "turret_aim_err", float("nan")))
        port_on = bool(getattr(h, "sensor_port_on", False))

        # AI Brain mode: sensors → MLP → turret. No tissue inject / hop / carve
        # (otherwise axons blink while "fly graph frozen").
        if port_on:
            t0 = _time.perf_counter()
            if hasattr(h, "_update_ocelli_sensors"):
                h._update_ocelli_sensors()
            if hasattr(h, "_update_eye_sensors"):
                h._update_eye_sensors()
            if hasattr(h, "_update_jo_sensors"):
                h._update_jo_sensors()
            if hasattr(h, "_update_turret_sensors"):
                h._update_turret_sensors()
            t_sense = (_time.perf_counter() - t0) * 1000.0
            # Quench residual hop glow once / while driving from port.
            fly = getattr(h, "fly", None)
            if fly is not None and hasattr(fly, "clear_activity"):
                fly.clear_activity()
            if hasattr(h, "_sync_activity_view"):
                h._sync_activity_view()
            info: dict = {
                "edges": 0,
                "lit": 0,
                "touched": 0,
                "max": 0.0,
                "hops": 0,
                "learned": False,
                "port_nn": True,
            }
            t0 = _time.perf_counter()
            self.actuate_all(float(dt))
            t_actuate = (_time.perf_counter() - t0) * 1000.0
            info["t_sense_ms"] = float(t_sense)
            info["t_hop_ms"] = 0.0
            info["t_actuate_ms"] = float(t_actuate)
            info["t_learn_ms"] = 0.0
            # Periodic Port-NN progress (same role as CNS learn console).
            h._learn_status_i = int(getattr(h, "_learn_status_i", 0)) + 1
            if h._learn_status_i % 20 == 0:
                port = getattr(h, "sensor_port_nn", None)
                if port is not None and hasattr(h, "_log_learn_console"):
                    h._log_learn_console(info)
                if port is not None and hasattr(h, "_set_io_status"):
                    try:
                        badge, _, _ = port.format_ui_badge(h)
                        h._set_io_status(badge)
                    except Exception:
                        pass
            return info

        t0 = _time.perf_counter()
        self.sense_all()
        t_sense = (_time.perf_counter() - t0) * 1000.0
        if len(h.fly.recipe_pre) == 0 and hasattr(h, "_ensure_atlas_recipe"):
            h._ensure_atlas_recipe()
        hops = self._resolve_hops()
        info = {"edges": 0, "lit": 0, "touched": 0, "max": 0.0}
        t0 = _time.perf_counter()
        for _ in range(hops):
            info = h.fly.tick_activity(hop_gain=0.85, decay=0.90)
        t_hop = (_time.perf_counter() - t0) * 1000.0
        info["hops"] = hops
        if hasattr(h, "_sync_activity_view"):
            h._sync_activity_view()
        # Corridor ∩ hop engagement every frame (real hop only; scope bed is separate).
        if hasattr(h, "_update_engagement_from_hop"):
            h._update_engagement_from_hop(info)
        learn_on = bool(
            getattr(h, "learn_on", False) and getattr(h, "turret_on", False)
        )
        # AI Brain owns the effector whenever it drives — never carve tissue
        # in parallel (train or frozen RUN).
        if bool(getattr(h, "sensor_port_on", False)):
            learn_on = False
        learn_every = self._resolve_learn_every()
        do_learn = learn_on and (self._frame_i % learn_every == 0)
        info["learned"] = bool(do_learn)
        t_teacher = 0.0
        if learn_on and do_learn and hasattr(h, "_apply_aim_teacher_mix"):
            # Teacher only on learn beats (cheap skip otherwise).
            t0 = _time.perf_counter()
            h._apply_aim_teacher_mix()
            if hasattr(h, "_sync_activity_view"):
                h._sync_activity_view()
            t_teacher = (_time.perf_counter() - t0) * 1000.0
        t0 = _time.perf_counter()
        self.actuate_all(float(dt))
        # Proprio after move — not a full second ocelli pass.
        self.sense_turret_proprio()
        t_actuate = (_time.perf_counter() - t0) * 1000.0
        t_learn = 0.0
        if do_learn:
            t0 = _time.perf_counter()
            import math as _math

            baseline = float(getattr(h, "_learn_prev_err", float("nan")))
            if not _math.isfinite(baseline):
                baseline = prev_err
            socket_on = bool(getattr(h, "motor_socket_on", False)) and getattr(
                h, "motor_socket", None
            ) is not None
            if socket_on:
                # Corridor weights frozen — only port readout W learns.
                tgts = self.task.teacher_site_levels(h)
                if tgts is None:
                    tgts = [0.5, 0.5, 0.5, 0.5]
                err_now = float(getattr(h, "turret_aim_err", float("nan")))
                # Teacher-driven Train already holds the cone — ε-gate would
                # kill cloning lr (ΔW→~0). Clone at full lr; gate only on pure W.
                alpha = float(getattr(h, "_socket_drive_alpha", 0.0) or 0.0)
                dw = float(
                    h.motor_socket.step(
                        h,
                        tgts,
                        err_rad=None if alpha >= 0.5 else err_now,
                    )
                )
                # Freeze is manual (Learn panel · Freeze now) — no beat timeout.
                h.motor_socket_dw_ema = (
                    0.85 * float(getattr(h, "motor_socket_dw_ema", 0.0)) + 0.15 * dw
                )
                # Honest meters: no graph Hebb. Flash = hop wave only (not path_edges).
                h.learn_n_edges = int(
                    len(getattr(getattr(h, "fly", None), "contact_gain", {}) or {})
                )
            else:
                tr = self.trainer
                if tr is None:
                    from trainer import Trainer

                    tr = Trainer(h, task=self.task)
                    self.trainer = tr
                    h._trainer = tr
                # RPE baseline = error at previous *learn* beat (not this frame start).
                tr.step(baseline)
            h._learn_prev_err = float(getattr(h, "turret_aim_err", float("nan")))
            if hasattr(h, "_merge_learn_chain_flash"):
                h._merge_learn_chain_flash(
                    hop_pre=info.get("flash_pre"),
                    hop_post=info.get("flash_post"),
                )
            h._learn_status_i = int(getattr(h, "_learn_status_i", 0)) + 1
            if h._learn_status_i % max(1, 20 // learn_every) == 0:
                if hasattr(h, "_log_learn_console"):
                    h._log_learn_console(info)
                aim = float(getattr(h, "turret_aim_err", float("nan")))
                aim_s = (
                    f"ε {aim * 180.0 / _math.pi:.0f}°"
                    if _math.isfinite(aim)
                    else "ε —"
                )
                if socket_on:
                    g = h.motor_socket.gains
                    tag = "FROZEN" if h.motor_socket.frozen else "train"
                    msg = (
                        f"Socket {tag} · g {[round(float(x), 2) for x in g]} · "
                        f"ΔW {float(getattr(h, 'motor_socket_dw_ema', 0.0)):.3f} · "
                        f"hop×{hops} · {aim_s}"
                    )
                else:
                    msg = (
                        f"Learn · Δw {float(getattr(h, 'learn_dw_ema', 0.0)):.3f} · "
                        f"{int(getattr(h, 'learn_n_edges', 0))} gains · "
                        f"chain {int(getattr(h, '_learn_flash_n', 0))} · "
                        f"hop×{hops} every{learn_every} · "
                        f"hop {int(info.get('edges', 0)):,} · {aim_s}"
                    )
                if hasattr(h, "_set_io_status"):
                    h._set_io_status(msg)
            t_learn = (_time.perf_counter() - t0) * 1000.0
        elif learn_on:
            # Keep flash decay so HUD doesn't freeze between learn beats.
            amp = float(getattr(h, "_learn_flash_amp", 0.0) or 0.0)
            h._learn_flash_amp = amp * 0.92
        info["t_sense_ms"] = float(t_sense)
        info["t_hop_ms"] = float(t_hop)
        info["t_actuate_ms"] = float(t_actuate)
        info["t_learn_ms"] = float(t_learn + t_teacher)
        return info

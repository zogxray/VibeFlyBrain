"""Aim task: lamp-relative teacher signal, separate from device afferents.

Teacher soft-land + track (not freeze-on-lock):
  • Far — scheduled P chase + brake when closing (no overshoot slam).
  • In cone — keep following: feedforward from lamp ω + weak residual P.
  • Lead — aim at lamp future azimuth (ω·τ) for the P error target.
Hold means "stay on the lamp", not "zero motors".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Turret pitch limits (match genome_shell defaults).
_PITCH_MIN = -0.35
_PITCH_MAX = 0.85

# Soft-land / track geometry.
_HOLD_DEG = 15.0  # aim cone — lock band (matches TURRET_AIM_HALF_DEG)
_FULL_DEG = 30.0  # full chase drive beyond this
_YAW_SCALE = 0.55  # rad → |cmd|=1 (far-field P)
_PITCH_SCALE = 0.40
_LOCK_P = 0.28  # residual P gain inside the cone (on top of feedforward)
_BRAKE_REV = 0.35
_BRAKE_CLOSE_DEG = 3.0
_LEAD_TAU_MAX = 0.25  # s
# Slew floors / margin — mirror genome_shell.turret_slew_rates (avoid import cycle).
_RATE_YAW = 2.0
_RATE_PITCH = 1.6
_TRACK_MARGIN = 2.4


@dataclass
class AimTask:
    """Supervised aim: desired barrel pose + ε. Not injected into sensory clamps."""

    pitch_min: float = _PITCH_MIN
    pitch_max: float = _PITCH_MAX
    hold_deg: float = _HOLD_DEG
    full_deg: float = _FULL_DEG
    # Lead aims ahead of the lamp — looks like "упреждение"; off by default.
    lead: bool = False
    # Per-axis previous signed error (rad) for brake/D.
    _prev_err_yaw: float = field(default=float("nan"), repr=False)
    _prev_err_pitch: float = field(default=float("nan"), repr=False)
    _prev_err_mag: float = field(default=float("nan"), repr=False)
    # Smoothed pursuit cmds for train-assist slew (kill jerk).
    _smooth_yaw_cmd: float = field(default=0.0, repr=False)
    _smooth_pitch_cmd: float = field(default=0.0, repr=False)

    def _hold_rad(self) -> float:
        return math.radians(float(self.hold_deg))

    def _full_rad(self) -> float:
        return max(self._hold_rad() + 1e-3, math.radians(float(self.full_deg)))

    def _slew_rates(self, lamp_speed: float) -> tuple[float, float]:
        omega = abs(float(lamp_speed)) * math.tau
        rate_yaw = max(float(_RATE_YAW), omega * float(_TRACK_MARGIN))
        pitch_ratio = float(_RATE_PITCH) / max(float(_RATE_YAW), 1e-6)
        rate_pitch = max(float(_RATE_PITCH), rate_yaw * pitch_ratio)
        return float(rate_yaw), float(rate_pitch)

    def _lead_tau(self, world: Any) -> float:
        """Latency estimate (s): one sim beat + hop slack."""
        if not self.lead:
            return 0.0
        hz = float(getattr(world, "adult_sim_hz", 20.0) or 20.0)
        kn = getattr(world, "_view_detail_knobs", None)
        if callable(kn):
            try:
                hz = float(kn().get("sim_hz", hz) or hz)
            except Exception:
                pass
        hops = float(getattr(world, "loop_hops", 1) or 1)
        tau = hops / max(hz, 1.0) + 0.03
        return float(np.clip(tau, 0.0, float(_LEAD_TAU_MAX)))

    def _mech_mount(self, world: Any) -> np.ndarray | None:
        pose = None
        mech = getattr(world, "_turret_mech_pose", None)
        if callable(mech):
            pose = mech()
        if pose is None:
            pose = world._turret_rig_pose() if hasattr(world, "_turret_rig_pose") else None
        if pose is None:
            return None
        return np.asarray(pose["mount"], dtype=np.float64)

    def _bearing_to(
        self, mount: np.ndarray, lamp: np.ndarray
    ) -> tuple[float, float] | None:
        to_lamp = np.asarray(lamp, dtype=np.float64) - np.asarray(mount, dtype=np.float64)
        n = float(np.linalg.norm(to_lamp))
        if n < 1e-9:
            return None
        to_lamp = to_lamp / n
        pitch_d = float(
            np.clip(
                math.asin(float(np.clip(to_lamp[1], -1.0, 1.0))),
                self.pitch_min,
                self.pitch_max,
            )
        )
        cp = math.cos(pitch_d)
        if abs(cp) < 1e-6:
            yaw_d = 0.0
        else:
            yaw_d = float(math.atan2(to_lamp[0] / cp, to_lamp[2] / cp))
        return yaw_d, pitch_d

    def _lamp_xyz_now(self, world: Any) -> np.ndarray:
        world._update_lamp_xyz()
        return np.asarray(world._lamp_xyz, dtype=np.float64).copy()

    def _lamp_target_xyz(self, world: Any) -> np.ndarray:
        """Lamp XYZ, optionally advanced by ω·τ (P-error target / lead)."""
        lamp = self._lamp_xyz_now(world)
        tau = self._lead_tau(world)
        speed = float(getattr(world, "lamp_speed", 0.0) or 0.0)
        eff = getattr(world, "effective_lamp_speed", None)
        if callable(eff):
            try:
                speed = float(eff())
            except Exception:
                pass
        if tau <= 1e-6 or abs(speed) <= 1e-9:
            return lamp
        ang0 = float(getattr(world, "lamp_angle", 0.0))
        world.lamp_angle = (ang0 + float(speed) * math.tau * tau) % math.tau
        world._update_lamp_xyz()
        out = np.asarray(world._lamp_xyz, dtype=np.float64).copy()
        world.lamp_angle = ang0
        world._update_lamp_xyz()
        return out

    def _track_feedforward(self, world: Any) -> tuple[float, float]:
        """Cmd that matches d(desired_bearing)/dt under current slew rates.

        Finite-diff lamp azimuth by one display tick; convert ω → cmd via slew.
        """
        mount = self._mech_mount(world)
        if mount is None:
            return 0.0, 0.0
        speed = float(getattr(world, "lamp_speed", 0.0) or 0.0)
        eff = getattr(world, "effective_lamp_speed", None)
        if callable(eff):
            try:
                speed = float(eff())
            except Exception:
                pass
        if abs(speed) <= 1e-9:
            return 0.0, 0.0
        lamp0 = self._lamp_xyz_now(world)
        b0 = self._bearing_to(mount, lamp0)
        if b0 is None:
            return 0.0, 0.0
        eps = 1.0 / 60.0
        ang0 = float(getattr(world, "lamp_angle", 0.0))
        world.lamp_angle = (ang0 + float(speed) * math.tau * eps) % math.tau
        world._update_lamp_xyz()
        lamp1 = np.asarray(world._lamp_xyz, dtype=np.float64).copy()
        world.lamp_angle = ang0
        world._update_lamp_xyz()
        b1 = self._bearing_to(mount, lamp1)
        if b1 is None:
            return 0.0, 0.0
        d_yaw = (b1[0] - b0[0] + math.pi) % math.tau - math.pi
        d_pitch = float(b1[1] - b0[1])
        rate_yaw, rate_pitch = self._slew_rates(speed)
        ff_yaw = float(np.clip(d_yaw / eps / max(rate_yaw, 1e-6), -1.0, 1.0))
        ff_pitch = float(np.clip(d_pitch / eps / max(rate_pitch, 1e-6), -1.0, 1.0))
        return ff_yaw, ff_pitch

    def desired_aim(self, world: Any) -> tuple[float, float] | None:
        """Lamp-relative (yaw, pitch) the barrel should hold (lead-aware).

        Origin must be the *visual* pedestal (mech mount), not the tall neuro
        Fuse mount — otherwise teacher pitch is systematically low and the
        drawn barrel aims under the lamp.
        """
        if not getattr(world, "turret_on", False):
            return None
        if not getattr(world, "lamp_on", True) or not getattr(world, "light_enabled", True):
            return None
        mount = self._mech_mount(world)
        if mount is None:
            return None
        lamp = self._lamp_target_xyz(world)
        bearing = self._bearing_to(mount, lamp)
        if bearing is None:
            yaw = float(getattr(world, "turret_yaw", 0.0))
            pitch = float(getattr(world, "turret_pitch", 0.4))
            return yaw, pitch
        return bearing

    def err(self, world: Any) -> float:
        """Cached aim error (rad) or NaN."""
        err = float(getattr(world, "turret_aim_err", float("nan")))
        if math.isfinite(err):
            return err
        fn = getattr(world, "_turret_aim_error", None)
        if callable(fn):
            e = fn()
            return float(e) if e is not None else float("nan")
        return float("nan")

    def _soft_land_cmds(
        self, world: Any, err_yaw: float, err_pitch: float
    ) -> tuple[float, float]:
        """Map pose error + lamp motion → yaw/pitch cmd ∈ [−1, 1]."""
        hold = self._hold_rad()
        full = self._full_rad()
        err_mag = float(math.hypot(err_yaw, err_pitch))
        ff_yaw, ff_pitch = self._track_feedforward(world)

        if err_mag <= hold:
            # Lock: follow lamp (ff) + weak residual P — do NOT freeze.
            yaw_cmd = float(
                np.clip(
                    ff_yaw
                    + float(_LOCK_P)
                    * float(np.clip(err_yaw / float(_YAW_SCALE), -1.0, 1.0)),
                    -1.0,
                    1.0,
                )
            )
            pitch_cmd = float(
                np.clip(
                    ff_pitch
                    + float(_LOCK_P)
                    * float(np.clip(err_pitch / float(_PITCH_SCALE), -1.0, 1.0)),
                    -1.0,
                    1.0,
                )
            )
        else:
            # Chase: scheduled P, brake only the P term when closing; keep ff.
            yaw_p = float(np.clip(err_yaw / float(_YAW_SCALE), -1.0, 1.0))
            pitch_p = float(np.clip(err_pitch / float(_PITCH_SCALE), -1.0, 1.0))
            g = float(np.clip((err_mag - hold) / (full - hold), 0.0, 1.0))
            g = g * g
            yaw_p *= g
            pitch_p *= g

            prev_mag = float(self._prev_err_mag)
            if math.isfinite(prev_mag) and err_mag < full:
                closing = err_mag < prev_mag - 1e-6
                if closing:
                    close = prev_mag - err_mag
                    fast = close >= math.radians(float(_BRAKE_CLOSE_DEG))
                    if fast:
                        # Soft brake — keep most of P so we still acquire.
                        br = float(_BRAKE_REV)
                        yaw_p *= 1.0 - 0.5 * br
                        pitch_p *= 1.0 - 0.5 * br
                    else:
                        # Slow close: keep residual P (old code zeroed → lead-only lag).
                        yaw_p *= 0.55
                        pitch_p *= 0.55

            yaw_cmd = float(np.clip(yaw_p + ff_yaw, -1.0, 1.0))
            pitch_cmd = float(np.clip(pitch_p + ff_pitch, -1.0, 1.0))

        self._prev_err_yaw = float(err_yaw)
        self._prev_err_pitch = float(err_pitch)
        self._prev_err_mag = float(err_mag)
        return yaw_cmd, pitch_cmd

    def _raw_pursuit(
        self, world: Any, *, with_ff: bool = False
    ) -> tuple[float, float] | None:
        """Unsmoothed P(+optional ff) on lamp-now. None if unavailable."""
        if not getattr(world, "turret_on", False):
            return None
        if not getattr(world, "lamp_on", True) or not getattr(
            world, "light_enabled", True
        ):
            return None
        mount = self._mech_mount(world)
        if mount is None:
            return None
        lamp = self._lamp_xyz_now(world)
        bearing = self._bearing_to(mount, lamp)
        if bearing is None:
            return None
        dyaw, dpitch = bearing
        yaw = float(getattr(world, "turret_yaw", 0.0))
        pitch = float(getattr(world, "turret_pitch", 0.4))
        err_yaw = (dyaw - yaw + math.pi) % math.tau - math.pi
        err_pitch = float(dpitch) - pitch
        raw_yaw = float(np.clip(err_yaw / 0.40, -1.0, 1.0))
        raw_pitch = float(np.clip(err_pitch / 0.30, -1.0, 1.0))
        if with_ff:
            ff_yaw, ff_pitch = self._track_feedforward(world)
            raw_yaw = float(np.clip(raw_yaw + ff_yaw, -1.0, 1.0))
            raw_pitch = float(np.clip(raw_pitch + ff_pitch, -1.0, 1.0))
        return raw_yaw, raw_pitch

    def pursuit_cmds(
        self, world: Any, *, with_ff: bool = False
    ) -> tuple[float, float] | None:
        """EMA-smoothed aim-at-lamp-now for Train slew (no lead)."""
        raw = self._raw_pursuit(world, with_ff=with_ff)
        if raw is None:
            return None
        raw_yaw, raw_pitch = raw
        a = 0.35
        self._smooth_yaw_cmd = (1.0 - a) * float(self._smooth_yaw_cmd) + a * raw_yaw
        self._smooth_pitch_cmd = (
            (1.0 - a) * float(self._smooth_pitch_cmd) + a * raw_pitch
        )
        return (
            float(np.clip(self._smooth_yaw_cmd, -1.0, 1.0)),
            float(np.clip(self._smooth_pitch_cmd, -1.0, 1.0)),
        )

    def teacher_site_levels(self, world: Any) -> list[float] | None:
        """Socket W targets: P+ff (in-cone ≠ idle). Does not touch slew EMA."""
        cmds = self._raw_pursuit(world, with_ff=True)
        if cmds is None:
            desired = self.desired_aim(world)
            if desired is None:
                return None
            dyaw, dpitch = desired
            yaw = float(getattr(world, "turret_yaw", 0.0))
            pitch = float(getattr(world, "turret_pitch", 0.4))
            err_yaw = (dyaw - yaw + math.pi) % math.tau - math.pi
            err_pitch = float(dpitch) - pitch
            yaw_cmd, pitch_cmd = self._soft_land_cmds(world, err_yaw, err_pitch)
        else:
            yaw_cmd, pitch_cmd = cmds
        y = float(np.clip(yaw_cmd, -1.0, 1.0))
        p = float(np.clip(pitch_cmd, -1.0, 1.0))
        return [
            float(np.clip(0.5 + 0.5 * y + 0.5 * p, 0.0, 1.0)),
            float(np.clip(0.5 + 0.5 * y - 0.5 * p, 0.0, 1.0)),
            float(np.clip(0.5 - 0.5 * y + 0.5 * p, 0.0, 1.0)),
            float(np.clip(0.5 - 0.5 * y - 0.5 * p, 0.0, 1.0)),
        ]

    def motor_neuron_targets(self, world: Any) -> dict[int, float]:
        levels = self.teacher_site_levels(world)
        if levels is None:
            return {}
        out: dict[int, float] = {}
        m_terms = getattr(world, "turret_motor_terminals", []) or []
        groups = getattr(world, "turret_motor_groups", []) or []
        for si, tgt in enumerate(levels):
            if si < len(m_terms):
                out[int(m_terms[si])] = tgt
            if si < len(groups):
                for nid in groups[si]:
                    out[int(nid)] = tgt
        return out

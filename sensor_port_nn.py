"""Sensor-port NN: ordinary MLP on device I/O — fly graph untouched.

Features: eyes + JO + proprio + eye population-vector bearing (anatomical
readout — not a world oracle).
Labels: pursuit cmds from geometric bearing error (imitation teacher).
CNS Brain / contact_gain never updated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# eye_L(6) + eye_R(6) + jo_L + jo_R + S(2)
# + sin_yaw + cos_yaw + pitch
# + ehat_yaw + ehat_pitch + eye_conf
N_IN = 6 + 6 + 1 + 1 + 2 + 3 + 3
N_HIDDEN = 64
N_OUT = 2  # yaw_cmd, pitch_cmd

# Live imitation MSE: ~0.5 early → ~0.01 when Track holds.
_LOSS_BAD = 0.20
_LOSS_GOOD = 0.025


def loss_quality(loss: float) -> float:
    """1 = good (low loss), 0 = bad (high)."""
    if not math.isfinite(float(loss)):
        return 0.0
    return float(np.clip(1.0 - (float(loss) - _LOSS_GOOD) / max(1e-6, _LOSS_BAD - _LOSS_GOOD), 0.0, 1.0))


def loss_rgb(loss: float) -> tuple[int, int, int]:
    """Red (bad) → amber → green (good) for UI meters / badges."""
    q = loss_quality(loss)
    bad = (220, 85, 75)
    mid = (220, 175, 85)
    good = (110, 205, 130)
    if q < 0.5:
        t = q * 2.0
        a, b = bad, mid
    else:
        t = (q - 0.5) * 2.0
        a, b = mid, good
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def loss_band(loss: float) -> str:
    if not math.isfinite(float(loss)):
        return "—"
    q = loss_quality(loss)
    if q >= 0.75:
        return "good"
    if q >= 0.35:
        return "ok"
    return "high"


def _wrap(a: float) -> float:
    return float((a + math.pi) % math.tau - math.pi)


def _vec(host: Any, attr: str, n: int) -> np.ndarray:
    v = np.asarray(getattr(host, attr, np.zeros(n)), dtype=np.float64).ravel()
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)))
    return v[:n].astype(np.float64)


def refresh_sensors(host: Any) -> None:
    for name in (
        "_update_ocelli_sensors",
        "_update_eye_sensors",
        "_update_jo_sensors",
        "_update_turret_sensors",
    ):
        fn = getattr(host, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def features(host: Any) -> np.ndarray:
    """Sensor vector: raw eyes/JO/proprio + anatomical eye bearing estimate."""
    refresh_sensors(host)
    el = _vec(host, "eye_left_I", 6)
    er = _vec(host, "eye_right_I", 6)
    jl = float(getattr(host, "jo_left_tilt", 0.0) or 0.0)
    jr = float(getattr(host, "jo_right_tilt", 0.0) or 0.0)
    S = _vec(host, "turret_S", 2)
    yaw = float(getattr(host, "turret_yaw", 0.0) or 0.0)
    pitch = float(getattr(host, "turret_pitch", 0.0) or 0.0)
    ehat_y = 0.0
    ehat_p = 0.0
    eye_conf = 0.0
    try:
        from compound_eyes import eye_lamp_azimuth, eye_light_vector

        az = eye_lamp_azimuth(host)
        if az is not None:
            eye_az, eye_conf = float(az[0]), float(az[1])
            # Scale so ~45° → unit feature.
            ehat_y = float(np.clip(_wrap(eye_az - yaw) / math.radians(45.0), -1.5, 1.5))
        vec = eye_light_vector(host)
        if vec is not None:
            v = np.asarray(vec[0], dtype=np.float64)
            elev = float(math.asin(float(np.clip(v[1], -1.0, 1.0))))
            ehat_p = float(np.clip((elev - pitch) / math.radians(35.0), -1.5, 1.5))
            eye_conf = max(eye_conf, float(vec[1]))
    except Exception:
        pass
    return np.concatenate(
        [
            el,
            er,
            [jl, jr],
            S,
            [math.sin(yaw), math.cos(yaw), pitch],
            [ehat_y, ehat_p, eye_conf],
        ]
    ).astype(np.float64)


def _desired_aim(host: Any) -> tuple[float, float] | None:
    fn = getattr(host, "_turret_desired_aim", None)
    if callable(fn):
        try:
            des = fn()
            if des is not None:
                return float(des[0]), float(des[1])
        except Exception:
            pass
    task = getattr(host, "_aim_task", None)
    if task is not None and hasattr(task, "desired_aim"):
        try:
            des = task.desired_aim(host)
            if des is not None:
                return float(des[0]), float(des[1])
        except Exception:
            pass
    return None


def training_targets(host: Any) -> tuple[np.ndarray, float] | None:
    """Expert slew cmds from bearing error (labels). Features stay sensory."""
    des = _desired_aim(host)
    if des is None:
        return None
    dyaw, dpitch = des
    yaw = float(getattr(host, "turret_yaw", 0.0) or 0.0)
    pitch = float(getattr(host, "turret_pitch", 0.0) or 0.0)
    err_yaw = _wrap(dyaw - yaw)
    err_pitch = float(dpitch - pitch)
    scale = math.radians(18.0)
    ty = float(np.clip(err_yaw / scale, -1.0, 1.0))
    tp = float(np.clip(err_pitch / scale, -1.0, 1.0))
    conf = 1.0
    try:
        from compound_eyes import eye_light_vector

        out = eye_light_vector(host)
        if out is not None:
            conf = max(0.15, min(1.0, float(out[1])))
    except Exception:
        pass
    return np.asarray([ty, tp], dtype=np.float64), conf


sensor_targets = training_targets  # back-compat


@dataclass
class SensorPortNN:
    """2-layer ReLU→tanh MLP + momentum SGD. CNS Brain never modified."""

    W1: np.ndarray = field(default_factory=lambda: np.zeros((N_HIDDEN, N_IN)))
    b1: np.ndarray = field(default_factory=lambda: np.zeros(N_HIDDEN))
    W2: np.ndarray = field(default_factory=lambda: np.zeros((N_OUT, N_HIDDEN)))
    b2: np.ndarray = field(default_factory=lambda: np.zeros(N_OUT))
    vW1: np.ndarray = field(default_factory=lambda: np.zeros((N_HIDDEN, N_IN)))
    vb1: np.ndarray = field(default_factory=lambda: np.zeros(N_HIDDEN))
    vW2: np.ndarray = field(default_factory=lambda: np.zeros((N_OUT, N_HIDDEN)))
    vb2: np.ndarray = field(default_factory=lambda: np.zeros(N_OUT))
    lr: float = 0.02
    momentum: float = 0.85
    weight_decay: float = 1e-4
    frozen: bool = False
    last_loss: float = float("nan")
    last_cmd: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_h: np.ndarray = field(default_factory=lambda: np.zeros(N_HIDDEN))
    last_x: np.ndarray = field(default_factory=lambda: np.zeros(N_IN))
    step_n: int = 0
    loss_ema: float = 0.0
    loss_start: float = float("nan")
    loss_best: float = float("nan")
    phantom_k: int = 3
    kick_every: int = 0  # 0 = no live pose scramble (was breaking Train tracking)
    replay_k: int = 8  # SGD draws from ring buffer (stationary policy)
    _buf_x: list = field(default_factory=list, repr=False)
    _buf_t: list = field(default_factory=list, repr=False)
    _buf_max: int = 2048
    _pending_kick: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.W1.shape != (N_HIDDEN, N_IN) or float(np.max(np.abs(self.W1))) < 1e-12:
            self.reset_weights()

    def reset_weights(self) -> None:
        rng = np.random.default_rng(7)
        self.W1 = (
            rng.normal(0.0, 1.0, (N_HIDDEN, N_IN)) * math.sqrt(2.0 / N_IN)
        ).astype(np.float64)
        self.b1 = np.zeros(N_HIDDEN, dtype=np.float64)
        self.W2 = (
            rng.normal(0.0, 1.0, (N_OUT, N_HIDDEN)) * math.sqrt(1.0 / N_HIDDEN)
        ).astype(np.float64)
        self.b2 = np.zeros(N_OUT, dtype=np.float64)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.frozen = False
        self.last_loss = float("nan")
        self.loss_ema = 0.0
        self.loss_start = float("nan")
        self.loss_best = float("nan")
        self.step_n = 0
        self._buf_x = []
        self._buf_t = []
        self._pending_kick = False

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).ravel()
        if len(x) != N_IN:
            x = np.pad(x, (0, max(0, N_IN - len(x))))[:N_IN]
        z1 = self.W1 @ x + self.b1
        h = np.maximum(z1, 0.0)
        y = np.tanh(self.W2 @ h + self.b2)
        self.last_x = x.copy()
        self.last_h = h.copy()
        self.last_cmd = y.copy()
        return y, h, x

    def commands(self, host: Any) -> tuple[float, float]:
        y, _, _ = self.forward(features(host))
        self.last_cmd = y.copy()
        return float(y[0]), float(y[1])

    def _sgd(self, x: np.ndarray, t: np.ndarray) -> float:
        y, h, x = self.forward(x)
        err = y - t
        loss = float(0.5 * np.mean(err * err))
        dy = err * (1.0 - y * y) / float(N_OUT)
        dW2 = np.outer(dy, h)
        db2 = dy.copy()
        dh = self.W2.T @ dy
        dz1 = dh * (h > 0.0)
        dW1 = np.outer(dz1, x)
        db1 = dz1.copy()
        mu = float(self.momentum)
        lr = float(self.lr)
        self.vW2 = mu * self.vW2 + dW2
        self.vb2 = mu * self.vb2 + db2
        self.vW1 = mu * self.vW1 + dW1
        self.vb1 = mu * self.vb1 + db1
        self.W2 -= lr * self.vW2
        self.b2 -= lr * self.vb2
        self.W1 -= lr * self.vW1
        self.b1 -= lr * self.vb1
        wd = float(self.weight_decay)
        if wd > 0.0:
            self.W2 *= 1.0 - lr * wd
            self.W1 *= 1.0 - lr * wd
        return loss

    def _set_pose(self, host: Any, yaw: float, pitch: float) -> None:
        host.turret_yaw = float(yaw) % math.tau
        host.turret_pitch = float(np.clip(pitch, -0.35, 0.85))
        host._frame_turret_pose = None
        host._frame_eye_pose = None
        for d in getattr(host, "device_list", []) or []:
            if getattr(d, "kind", "") == "turret":
                d.yaw = float(host.turret_yaw)
                d.pitch = float(host.turret_pitch)

    def _remember(self, x: np.ndarray, t: np.ndarray) -> None:
        self._buf_x.append(np.asarray(x, dtype=np.float64).copy())
        self._buf_t.append(np.asarray(t, dtype=np.float64).copy())
        cap = max(64, int(self._buf_max))
        if len(self._buf_x) > cap:
            overflow = len(self._buf_x) - cap
            del self._buf_x[:overflow]
            del self._buf_t[:overflow]

    def train_step(self, host: Any) -> float:
        """Imitation SGD with phantoms + replay (converges to frozen-stable policy).

        Actuates with pre-update live prediction (matches frozen RUN).
        Phantoms teleport only for SGD samples, then restore pose — live kick
        never leaves the plant on a scrambled pose with a stale cmd.
        """
        if self.frozen:
            return float(self.last_loss) if math.isfinite(self.last_loss) else 0.0

        # Apply deferred exploration kick *before* sensing this frame.
        des_early = _desired_aim(host)
        if self._pending_kick and des_early is not None:
            self._pending_kick = False
            rng = np.random.default_rng(self.step_n * 17 + 3)
            self._set_pose(
                host,
                float(des_early[0]) + float(rng.uniform(-0.9, 0.9)),
                float(des_early[1]) + float(rng.uniform(-0.2, 0.2)),
            )
            refresh_sensors(host)

        tgt = training_targets(host)
        x_live = features(host)
        y, _, _ = self.forward(x_live)
        self.last_cmd = y.copy()
        if tgt is None:
            return float(self.last_loss) if math.isfinite(self.last_loss) else 0.0
        t, conf = tgt
        live_loss = 0.0
        if conf >= 0.08:
            self._remember(x_live, t)
            live_loss = self._sgd(x_live, t)

        des = _desired_aim(host)
        yaw0 = float(getattr(host, "turret_yaw", 0.0) or 0.0)
        pitch0 = float(getattr(host, "turret_pitch", 0.0) or 0.0)
        # Sync device pose so phantoms don't fight TurretDevice.yaw.
        self._set_pose(host, yaw0, pitch0)
        k = max(0, int(self.phantom_k))
        if des is not None and k > 0:
            rng = np.random.default_rng(self.step_n * 9973 + 11)
            dyaw0, dpitch0 = des
            for j in range(k):
                if j == 0:
                    d_yaw = float(rng.uniform(-0.35, 0.35))
                    d_pit = float(rng.uniform(-0.12, 0.12))
                else:
                    d_yaw = float(rng.uniform(-1.6, 1.6))
                    d_pit = float(rng.uniform(-0.3, 0.3))
                self._set_pose(host, dyaw0 + d_yaw, dpitch0 + d_pit)
                pt = training_targets(host)
                if pt is None:
                    continue
                t_p, c_p = pt
                if c_p < 0.05:
                    continue
                x_p = features(host)
                self._remember(x_p, t_p)
                self._sgd(x_p, t_p)
            self._set_pose(host, yaw0, pitch0)
            refresh_sensors(host)

        # Replay: re-fit past (sensor, expert) pairs → stationary policy.
        rk = max(0, int(self.replay_k))
        nbuf = len(self._buf_x)
        if rk > 0 and nbuf > 8:
            rng = np.random.default_rng(self.step_n * 13 + 7)
            for _ in range(rk):
                j = int(rng.integers(0, nbuf))
                self._sgd(self._buf_x[j], self._buf_t[j])

        # Critical: _sgd/forward overwrite last_cmd with phantom/replay draws.
        # Actuate must use the live-pose policy — same path as Freeze/commands().
        y_act, _, _ = self.forward(features(host))
        self.last_cmd = y_act.copy()

        # Meter: live closed-loop loss (not last phantom/replay draw).
        self.last_loss = float(live_loss)
        if not math.isfinite(self.loss_start):
            self.loss_start = float(live_loss)
        if not math.isfinite(self.loss_best) or float(live_loss) < float(self.loss_best):
            self.loss_best = float(live_loss)
        self.loss_ema = (
            live_loss if self.step_n == 0 else (0.92 * self.loss_ema + 0.08 * live_loss)
        )
        self.step_n += 1
        ke = int(self.kick_every)
        if ke > 0 and self.step_n % ke == 0 and des is not None:
            # Defer to next frame so this frame's cmd matches the pose we slew from.
            self._pending_kick = True
        return float(live_loss)

    def progress(self, host: Any | None = None) -> dict[str, Any]:
        """UI / console snapshot of AI Brain learning progress."""
        loss = float(self.loss_ema)
        cmd = np.asarray(self.last_cmd, dtype=np.float64).ravel()
        track = float("nan")
        eps = float("nan")
        if host is not None:
            track = float(getattr(host, "_track_score", float("nan")))
            err = getattr(host, "turret_aim_err", float("nan"))
            if isinstance(err, (int, float)) and math.isfinite(float(err)):
                eps = float(math.degrees(float(err)))
        mode = "off"
        if host is not None and bool(getattr(host, "sensor_port_on", False)):
            if self.frozen:
                mode = "frozen"
            elif bool(getattr(host, "sensor_port_train", False)):
                mode = "train"
            else:
                mode = "run"
        elif self.frozen:
            mode = "frozen"
        start = float(self.loss_start)
        best = float(self.loss_best)
        trend = "·"
        if math.isfinite(loss) and math.isfinite(start) and abs(loss - start) > 0.01:
            trend = "↓" if loss < start * 0.85 else ("↑" if loss > start * 1.15 else "·")
        return {
            "mode": mode,
            "step_n": int(self.step_n),
            "loss": loss,
            "loss_start": start,
            "loss_best": best,
            "loss_q": loss_quality(loss),
            "loss_band": loss_band(loss),
            "loss_rgb": loss_rgb(loss),
            "trend": trend,
            "buf": len(self._buf_x),
            "cmd_yaw": float(cmd[0]) if len(cmd) else 0.0,
            "cmd_pitch": float(cmd[1]) if len(cmd) > 1 else 0.0,
            "track": track,
            "eps_deg": eps,
            "frozen": bool(self.frozen),
            "lr": float(self.lr),
        }

    def format_console_line(self, host: Any | None = None) -> str:
        p = self.progress(host)
        tr = p["track"]
        tr_s = f"{tr:.0%}" if math.isfinite(tr) else "—"
        eps = p["eps_deg"]
        eps_s = f"{eps:.1f}°" if math.isfinite(eps) else "—"
        loss = p["loss"]
        loss_s = f"{loss:.4f}" if math.isfinite(loss) else "—"
        best = p["loss_best"]
        best_s = f"{best:.4f}" if math.isfinite(best) else "—"
        start = p["loss_start"]
        start_s = f"{start:.3f}" if math.isfinite(start) else "—"
        return (
            f"[ai-brain] {p['mode']}  t={p['step_n']}  "
            f"ε={eps_s}  Track={tr_s}  "
            f"loss={loss_s}{p['trend']} [{p['loss_band']}] (start {start_s} best {best_s})  "
            f"buf={p['buf']}  "
            f"cmd=({p['cmd_yaw']:+.2f},{p['cmd_pitch']:+.2f})  "
            f"lr={p['lr']:.3f}"
        )

    def format_ui_badge(self, host: Any | None = None) -> tuple[str, float, tuple[int, int, int]]:
        """Short meter label + fill + accent (accent tracks loss quality)."""
        p = self.progress(host)
        loss = p["loss"]
        fill = loss_quality(loss) if math.isfinite(loss) else 0.0
        accent = loss_rgb(loss) if math.isfinite(loss) else (110, 110, 100)
        tr = p["track"]
        tr_s = f"{tr:.0%}" if math.isfinite(tr) else "—"
        eps = p["eps_deg"]
        eps_s = f"{eps:.0f}°" if math.isfinite(eps) else "—"
        loss_s = f"{loss:.3f}" if math.isfinite(loss) else "—"
        band = str(p.get("loss_band", "—"))
        if p["mode"] == "frozen":
            return (
                f"AI Brain FROZEN · L {loss_s} {band} · T {tr_s} · ε {eps_s} · n={p['step_n']}",
                max(fill, 0.4),
                accent,
            )
        if p["mode"] == "train":
            return (
                f"AI Brain TRAIN · L {loss_s}{p['trend']} {band} · T {tr_s} · ε {eps_s} · n={p['step_n']}",
                fill,
                accent,
            )
        if p["mode"] == "run":
            return (
                f"AI Brain RUN · L {loss_s} {band} · T {tr_s} · ε {eps_s} · n={p['step_n']}",
                max(fill, 0.25),
                accent,
            )
        return "AI Brain off", 0.0, (110, 110, 100)

    def format_corner_badge(self, host: Any | None = None) -> tuple[str, tuple[int, int, int]]:
        """Compact top-left badge (replaces CNS 'Learn off' while AI Brain owns drive)."""
        p = self.progress(host)
        tr = p["track"]
        tr_s = f"{tr:.0%}" if math.isfinite(tr) else "—"
        eps = p["eps_deg"]
        eps_s = f"{eps:.0f}°" if math.isfinite(eps) else "—"
        loss = p["loss"]
        loss_s = f"{loss:.3f}" if math.isfinite(loss) else "—"
        accent = loss_rgb(loss) if math.isfinite(loss) else (110, 110, 100)
        band = str(p.get("loss_band", "—"))
        if p["mode"] == "frozen":
            return f"AI Brain FROZEN · L {loss_s} {band} · T {tr_s}", accent
        if p["mode"] == "train":
            return (
                f"AI Brain TRAIN · L {loss_s}{p['trend']} {band} · T {tr_s}",
                accent,
            )
        if p["mode"] == "run":
            return f"AI Brain RUN · L {loss_s} {band} · T {tr_s} · ε {eps_s}", accent
        # Train flag without drive still counts as AI Brain learning session.
        if host is not None and bool(getattr(host, "sensor_port_train", False)):
            return (
                f"AI Brain TRAIN · L {loss_s}{p['trend']} {band} · T {tr_s}",
                accent,
            )
        return "AI Brain", (160, 150, 120)

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "lr": float(self.lr),
            "momentum": float(self.momentum),
            "frozen": bool(self.frozen),
            "loss_ema": float(self.loss_ema),
            "step_n": int(self.step_n),
            "n_in": N_IN,
            "n_hidden": N_HIDDEN,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, *, lr: float = 0.02) -> SensorPortNN:
        nn = cls(lr=float(lr))
        if not isinstance(raw, dict):
            return nn
        try:
            W1 = np.asarray(raw.get("W1"), dtype=np.float64)
            W2 = np.asarray(raw.get("W2"), dtype=np.float64)
            b1 = np.asarray(raw.get("b1"), dtype=np.float64)
            b2 = np.asarray(raw.get("b2"), dtype=np.float64)
            if W1.shape == (N_HIDDEN, N_IN) and W2.shape == (N_OUT, N_HIDDEN):
                nn.W1, nn.W2 = W1, W2
                nn.b1 = b1.ravel()[:N_HIDDEN]
                nn.b2 = b2.ravel()[:N_OUT]
                nn.vW1 = np.zeros_like(nn.W1)
                nn.vb1 = np.zeros_like(nn.b1)
                nn.vW2 = np.zeros_like(nn.W2)
                nn.vb2 = np.zeros_like(nn.b2)
            else:
                nn.reset_weights()
            nn.frozen = bool(raw.get("frozen", False))
            nn.lr = float(raw.get("lr", lr) or lr)
            nn.momentum = float(raw.get("momentum", 0.85) or 0.85)
            nn.loss_ema = float(raw.get("loss_ema", 0.0) or 0.0)
            nn.step_n = int(raw.get("step_n", 0) or 0)
        except Exception:
            nn.reset_weights()
        return nn

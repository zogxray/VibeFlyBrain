"""Motor socket: learned linear readout at the effector port.

CNS hop to motors is weak and common-mode; ocelli cups *do* carry lamp side.
This port maps port features → 4 site drives, trains toward AimTask targets,
then freezes (Learn panel button) so the frozen map (not the teacher) steers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Feature layout: I_m, I_l, I_r, S_yaw, S_pitch, raw0..raw3, bias
N_FEAT = 3 + 2 + 4 + 1
N_SITE = 4


def _ocelli_I(host: Any) -> np.ndarray:
    I = np.asarray(getattr(host, "ocelli_I", np.zeros(3)), dtype=np.float64)
    if len(I) < 3:
        I = np.pad(I, (0, 3 - len(I)))
    return I[:3]


def _turret_S(host: Any) -> np.ndarray:
    S = np.asarray(getattr(host, "turret_S", np.array([0.5, 0.5])), dtype=np.float64)
    if len(S) < 2:
        S = np.pad(S, (0, 2 - len(S)), constant_values=0.5)
    return S[:2]


def _motor_raw(host: Any) -> np.ndarray:
    raw = np.zeros(N_SITE, dtype=np.float64)
    fn = getattr(host, "_turret_site_motor_level_raw", None)
    if callable(fn):
        for si in range(N_SITE):
            raw[si] = float(fn(si))
    return raw


def lateralize_raw(raw: np.ndarray, I: np.ndarray, *, mix: float = 0.40) -> np.ndarray:
    """Fixed assembly wire: ipsilateral ocelli cup bleeds into motor sites.

    Cups: 0=medial, 1=left, 2=right. Sites: 0,1=left; 2,3=right.
    Makes lamp-side visible at the port even when hop washed out.
    """
    mix = float(np.clip(mix, 0.0, 1.0))
    out = np.asarray(raw, dtype=np.float64).copy()
    Il = float(I[1]) if len(I) > 1 else 0.0
    Ir = float(I[2]) if len(I) > 2 else 0.0
    Im = float(I[0]) if len(I) else 0.0
    out[0] = (1.0 - mix) * out[0] + mix * (0.7 * Il + 0.3 * Im)
    out[1] = (1.0 - mix) * out[1] + mix * (0.7 * Il + 0.3 * Im)
    out[2] = (1.0 - mix) * out[2] + mix * (0.7 * Ir + 0.3 * Im)
    out[3] = (1.0 - mix) * out[3] + mix * (0.7 * Ir + 0.3 * Im)
    return np.clip(out, 0.0, 1.0)


@dataclass
class MotorSocket:
    """Port readout: features → 4 site levels. Train then freeze."""

    W: np.ndarray = field(default_factory=lambda: np.zeros((N_SITE, N_FEAT)))
    lr: float = 0.35
    lateral_mix: float = 0.40
    frozen: bool = False
    settle_err: float = 0.30
    # Legacy 4-gain view for badges (mean |row| scale).
    gains: np.ndarray = field(default_factory=lambda: np.ones(N_SITE))

    def __post_init__(self) -> None:
        W = np.asarray(self.W, dtype=np.float64)
        if W.shape != (N_SITE, N_FEAT):
            W = np.zeros((N_SITE, N_FEAT), dtype=np.float64)
            # Weak prior: ipsilateral ocelli → same-side motors, proprio mild.
            # feat: Im Il Ir Sy Sp r0 r1 r2 r3 bias
            W[0, 1] = 0.8  # Il → Llo
            W[1, 1] = 0.8
            W[2, 2] = 0.8  # Ir → Rlo/Rhi
            W[3, 2] = 0.8
            W[0, 3] = 0.3  # yaw proprio
            W[1, 3] = 0.3
            W[2, 3] = -0.3
            W[3, 3] = -0.3
            W[:, -1] = 0.05
        self.W = W
        self.gains = np.ones(N_SITE, dtype=np.float64)
        self._sync_gains_badge()

    def _sync_gains_badge(self) -> None:
        # Rough per-site drive strength for UI.
        self.gains = np.clip(np.sum(np.abs(self.W[:, :3]), axis=1) + 0.5, 0.2, 10.0)

    def features(self, host: Any) -> np.ndarray:
        I = _ocelli_I(host)
        S = _turret_S(host)
        raw = lateralize_raw(_motor_raw(host), I, mix=self.lateral_mix)
        return np.concatenate([I, S, raw, np.array([1.0])])

    def levels(self, host: Any) -> list[float]:
        x = self.features(host)
        y = self.W @ x
        return [float(np.clip(v, 0.0, 1.0)) for v in y.tolist()]

    def scale(self, si: int, raw: float) -> float:
        """Compat: prefer full ``levels(host)``; single-site falls back to raw."""
        del si
        return float(np.clip(raw, 0.0, 1.0))

    def scale_levels(self, raw_levels: list[float] | np.ndarray) -> list[float]:
        """Unused when host passed to actuate via ``levels(host)``."""
        return [float(np.clip(x, 0.0, 1.0)) for x in list(raw_levels)[:4]]

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize learned port map for .fly / UI state."""
        W = np.asarray(self.W, dtype=np.float64)
        return {
            "W": [[float(x) for x in row] for row in W.tolist()],
            "lr": float(self.lr),
            "lateral_mix": float(self.lateral_mix),
            "frozen": bool(self.frozen),
            "settle_err": float(self.settle_err),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, *, lr: float | None = None) -> MotorSocket:
        """Restore socket; missing/invalid W falls back to ipsilateral prior."""
        raw = raw if isinstance(raw, dict) else {}
        use_lr = float(raw["lr"]) if isinstance(raw.get("lr"), (int, float)) else (
            float(lr) if lr is not None else 0.35
        )
        sock = cls(lr=use_lr)
        if isinstance(raw.get("lateral_mix"), (int, float)):
            sock.lateral_mix = float(np.clip(float(raw["lateral_mix"]), 0.0, 1.0))
        if isinstance(raw.get("settle_err"), (int, float)):
            sock.settle_err = float(max(1e-3, float(raw["settle_err"])))
        W_raw = raw.get("W")
        if isinstance(W_raw, list) and W_raw:
            try:
                W = np.asarray(W_raw, dtype=np.float64)
                if W.shape == (N_SITE, N_FEAT):
                    sock.W = np.clip(W, -4.0, 4.0)
                    sock._sync_gains_badge()
            except (TypeError, ValueError):
                pass
        sock.frozen = bool(raw.get("frozen", False))
        return sock

    def apply_dict(self, raw: dict[str, Any] | None) -> bool:
        """In-place restore. True if a full W matrix was applied."""
        if not isinstance(raw, dict):
            return False
        if isinstance(raw.get("lr"), (int, float)):
            self.lr = float(raw["lr"])
        if isinstance(raw.get("lateral_mix"), (int, float)):
            self.lateral_mix = float(np.clip(float(raw["lateral_mix"]), 0.0, 1.0))
        if isinstance(raw.get("settle_err"), (int, float)):
            self.settle_err = float(max(1e-3, float(raw["settle_err"])))
        ok = False
        W_raw = raw.get("W")
        if isinstance(W_raw, list) and W_raw:
            try:
                W = np.asarray(W_raw, dtype=np.float64)
                if W.shape == (N_SITE, N_FEAT):
                    self.W = np.clip(W, -4.0, 4.0)
                    self._sync_gains_badge()
                    ok = True
            except (TypeError, ValueError):
                pass
        if "frozen" in raw:
            self.frozen = bool(raw["frozen"])
        return ok

    def step(
        self,
        host: Any,
        targets: list[float] | np.ndarray,
        *,
        err_rad: float | None = None,
    ) -> float:
        """One supervised update toward teacher site targets. No-op if frozen."""
        if self.frozen:
            return 0.0
        tgt = np.asarray(list(targets)[:4], dtype=np.float64)
        if len(tgt) < 4:
            tgt = np.pad(tgt, (0, 4 - len(tgt)), constant_values=0.5)
        x = self.features(host)
        y = np.clip(self.W @ x, 0.0, 1.0)
        err = tgt - y
        lr = float(self.lr)
        if err_rad is not None and math.isfinite(float(err_rad)):
            e = abs(float(err_rad))
            gate = float(np.clip(e / max(self.settle_err, 1e-3), 0.0, 1.0))
            lr *= 0.2 + 0.8 * gate
        # dW = lr * err ⊗ x
        dW = lr * np.outer(err, x)
        dW = np.clip(dW, -0.5, 0.5)
        self.W = np.clip(self.W + dW, -4.0, 4.0)
        self._sync_gains_badge()
        return float(np.mean(np.abs(dW)))

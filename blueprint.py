"""Declarative CNS Blueprint: devices + plasticity rule + training scope flags.

Headless training and the pygame shell both consume the same Blueprint.
LEARN_* / CHANNEL_* live on PlasticityRule — not as GUI module constants.
Devices are a typed list (see ``device.py``), not a fixed ocelli/turret pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from device import (
    Device,
    OcelliDevice,
    TurretDevice,
    devices_from_shell,
    devices_to_shell,
)


@dataclass
class PlasticityRule:
    """Aim-learning / river-carve parameters (was LEARN_* + CHANNEL_*)."""

    lr: float = 0.08
    hebb: float = 0.0
    # Soft shaping into motor *clamps* (via stimulate), not into slew.
    # 0 = pure sensory→CNS→motor (recommended with clone_teacher=False).
    teacher_mix: float = 0.0
    w_min: float = 0.05
    w_max: float = 2.5
    dw_cap: float = 0.10
    w_decay: float = 0.001
    device_gain_scale: float = 0.0
    elig_depth: int = 10
    elig_tau: float = 0.88  # temporal eligibility decay (Trainer traces)
    path_credit: float = 0.22
    hops: int = 8  # headless / full train beat (≥ typical sensor→motor geodesic)
    hops_ui: int = 1  # interactive shell draw cadence; Learn floors via ClosedLoop
    learn_every: int = 8  # UI: plasticity every N frames (1 = every frame)
    carve_lr: float = 0.045
    offpath_decay: float = 0.004
    channel_show_dw: float = 0.06
    channel_max_edges: int = 220
    channel_act_min: float = 0.04
    # Clone AimTask motor targets into contact_gain (legacy). False = learn
    # from Track / signed yaw lag / ΔI (no tgt clone).
    clone_teacher: bool = False
    # Primary steer: eye population vector − turret proprio, × yaw_cmd.
    sense_align: float = 0.55
    # Geometric AimTask lag × yaw_cmd. 0 in the CNS recipe — that term is the
    # oracle that taught contact_gain to grow without learning to follow.
    bearing_credit: float = 0.0
    # Weight of Track score Δ in the plasticity reward (scalar success only).
    track_credit: float = 0.30
    # Johnston organ: how hard world swing discounts the eye steer (0 = ignore).
    jo_credit: float = 0.25
    # Brake acquisition when cmd fights lag / overshoots.
    hold_brake: float = 0.55
    # Extra decay when yaw_cmd fights needed bearing side.
    wrong_decay: float = 0.65
    # Reward weight for being in cone with cmd matching lag.
    hold_credit: float = 0.20
    # Default OFF: steer from hop motor activity (sensor→CNS→turret).
    # ON = legacy port bypass W@[I,…] (not through tissue).
    motor_socket_on: bool = False
    motor_socket_lr: float = 0.35
    # Legacy DAgger slew blend. Keep 0 in CNS mode (never cmd→yaw directly).
    socket_teacher_drive: float = 0.0
    # Legacy (ignored): freeze is manual via Learn panel button.
    motor_socket_freeze_after: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PlasticityRule:
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: raw[k] for k in known if k in raw})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunnelGate:
    """Deprecated: well-grow size hints only. Training uses TrainingScope.ok.

    Prefer ``wells.WellGrowParams``. ``min_paths`` is ignored for wells.
    """

    min_a: int = 30
    min_b: int = 30
    min_n: int = 20
    min_neck_class: int = 6
    min_paths: int = 3  # legacy; unused by WellField / TrainingScope

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> FunnelGate:
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: int(raw[k]) for k in known if k in raw})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceBundle:
    """Legacy flat ``shell_devices`` blob — prefer ``Blueprint.device_list``."""

    ocelli_groups: list[list[int]] = field(default_factory=lambda: [[], [], []])
    ocelli_terminals: list[int] = field(default_factory=list)
    turret_motor_groups: list[list[int]] = field(
        default_factory=lambda: [[], [], [], []]
    )
    turret_sensory_groups: list[list[int]] = field(
        default_factory=lambda: [[], [], [], []]
    )
    turret_motor_terminals: list[int] = field(default_factory=list)
    turret_sensory_terminals: list[int] = field(default_factory=list)
    turret_bus_terminal: int | None = None
    turret_yaw: float = 0.0
    turret_pitch: float = 0.40
    # Pass-through for newer sockets (eyes, …) so sync doesn't drop them.
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> DeviceBundle:
        if not raw:
            return cls()
        bus = raw.get("turret_bus_terminal")
        known = {
            "ocelli_groups",
            "ocelli_terminals",
            "turret_motor_groups",
            "turret_sensory_groups",
            "turret_motor_terminals",
            "turret_sensory_terminals",
            "turret_bus_terminal",
            "turret_yaw",
            "turret_pitch",
            "v",
        }
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(
            ocelli_groups=[
                [int(x) for x in g] for g in (raw.get("ocelli_groups") or [[], [], []])
            ],
            ocelli_terminals=[int(x) for x in (raw.get("ocelli_terminals") or [])],
            turret_motor_groups=[
                [int(x) for x in g]
                for g in (raw.get("turret_motor_groups") or [[], [], [], []])
            ],
            turret_sensory_groups=[
                [int(x) for x in g]
                for g in (raw.get("turret_sensory_groups") or [[], [], [], []])
            ],
            turret_motor_terminals=[
                int(x) for x in (raw.get("turret_motor_terminals") or [])
            ],
            turret_sensory_terminals=[
                int(x) for x in (raw.get("turret_sensory_terminals") or [])
            ],
            turret_bus_terminal=int(bus) if bus is not None else None,
            turret_yaw=float(raw.get("turret_yaw", 0.0)),
            turret_pitch=float(raw.get("turret_pitch", 0.40)),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "ocelli_groups": [list(g) for g in self.ocelli_groups],
            "ocelli_terminals": list(self.ocelli_terminals),
            "turret_motor_groups": [list(g) for g in self.turret_motor_groups],
            "turret_sensory_groups": [list(g) for g in self.turret_sensory_groups],
            "turret_motor_terminals": list(self.turret_motor_terminals),
            "turret_sensory_terminals": list(self.turret_sensory_terminals),
            "turret_bus_terminal": self.turret_bus_terminal,
            "turret_yaw": float(self.turret_yaw),
            "turret_pitch": float(self.turret_pitch),
        }
        out.update(self.extra or {})
        return out

    def has_ocelli_turret(self) -> bool:
        return bool(self.ocelli_groups) and bool(self.turret_motor_terminals)

    def to_device_list(
        self, *, ocelli_on: bool = True, turret_on: bool = True
    ) -> list[Device]:
        return devices_from_shell(
            self.to_dict(), ocelli_on=ocelli_on, turret_on=turret_on
        )


@dataclass
class Blueprint:
    """CNS + device list + plasticity + training flags."""

    fly_path: str | None = None
    device_list: list[Any] = field(default_factory=list)
    # Legacy accessor — kept in sync with device_list for .fly shell_devices.
    devices: DeviceBundle = field(default_factory=DeviceBundle)
    plasticity: PlasticityRule = field(default_factory=PlasticityRule)
    funnel_gate: FunnelGate = field(default_factory=FunnelGate)
    scope_on: bool = True  # graph TrainingScope active
    hop_clip: bool = True  # optional: hop_allow = scope.corridor
    # Extra hops beyond each motor's shortest sensor→motor path (geodesic slack).
    scope_path_margin: int = 2
    # BFS cap per sensor group. Optic → DN is far longer than ocelli → DN,
    # so 16 silently dropped the eye corridor.
    scope_max_hops: int = 24
    # Fuse picks eye entries / wing-pad partners by reach so the two eyes drive
    # opposite pads. Off = legacy nearest-soma pick (common-mode projection).
    stereo_channels: bool = True
    ocelli_on: bool = True
    turret_on: bool = True
    learn_on: bool = True
    yaw_offset: float = 0.0
    lamp_speed: float = 0.0
    lamp_angle: float = 0.8

    def __post_init__(self) -> None:
        if not self.device_list:
            self.device_list = self.devices.to_device_list(
                ocelli_on=self.ocelli_on, turret_on=self.turret_on
            )
        else:
            self.sync_bundle_from_list()


    def sync_bundle_from_list(self) -> None:
        self.devices = DeviceBundle.from_dict(devices_to_shell(self.device_list))
        for d in self.device_list:
            if isinstance(d, OcelliDevice):
                self.ocelli_on = bool(d.enabled)
            elif isinstance(d, TurretDevice):
                self.turret_on = bool(d.enabled)

    def device(self, name: str) -> Any | None:
        for d in self.device_list:
            if getattr(d, "name", None) == name or getattr(d, "kind", None) == name:
                return d
        return None

    def get_ocelli(self) -> OcelliDevice | None:
        d = self.device("ocelli")
        return d if isinstance(d, OcelliDevice) else None

    def get_turret(self) -> TurretDevice | None:
        d = self.device("turret")
        return d if isinstance(d, TurretDevice) else None

    @classmethod
    def default(cls) -> Blueprint:
        return cls()

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Blueprint:
        if not raw:
            return cls()
        bundle = DeviceBundle.from_dict(raw.get("devices") or raw.get("shell_devices"))
        ocelli_on = bool(raw.get("ocelli_on", True))
        turret_on = bool(raw.get("turret_on", True))
        # One door: scope_on. Migrate older funnel_model / funnel_model_on on load only.
        if "scope_on" in raw:
            scope_on = bool(raw["scope_on"])
        elif "funnel_model_on" in raw:
            scope_on = bool(raw["funnel_model_on"])
        elif "funnel_model" in raw:
            scope_on = bool(raw["funnel_model"])
        else:
            scope_on = True
        return cls(
            fly_path=raw.get("fly_path"),
            devices=bundle,
            device_list=bundle.to_device_list(
                ocelli_on=ocelli_on, turret_on=turret_on
            ),
            plasticity=PlasticityRule.from_dict(raw.get("plasticity")),
            funnel_gate=FunnelGate.from_dict(raw.get("funnel_gate")),
            scope_on=scope_on,
            hop_clip=bool(raw.get("hop_clip", True)),
            scope_path_margin=int(raw.get("scope_path_margin", 2)),
            scope_max_hops=int(raw.get("scope_max_hops", 24)),
            stereo_channels=bool(raw.get("stereo_channels", True)),
            ocelli_on=ocelli_on,
            turret_on=turret_on,
            learn_on=bool(raw.get("learn_on", True)),
            yaw_offset=float(raw.get("yaw_offset", 0.0)),
            lamp_speed=float(raw.get("lamp_speed", 0.0)),
            lamp_angle=float(raw.get("lamp_angle", 0.8)),
        )

    def to_dict(self) -> dict[str, Any]:
        self.sync_bundle_from_list()
        return {
            "fly_path": self.fly_path,
            "devices": self.devices.to_dict(),
            "device_kinds": [getattr(d, "kind", "?") for d in self.device_list],
            "plasticity": self.plasticity.to_dict(),
            "funnel_gate": self.funnel_gate.to_dict(),
            "scope_on": bool(self.scope_on),
            "hop_clip": bool(self.hop_clip),
            "scope_path_margin": int(self.scope_path_margin),
            "scope_max_hops": int(self.scope_max_hops),
            "stereo_channels": bool(self.stereo_channels),
            "ocelli_on": bool(self.ocelli_on),
            "turret_on": bool(self.turret_on),
            "learn_on": bool(self.learn_on),
            "yaw_offset": float(self.yaw_offset),
            "lamp_speed": float(self.lamp_speed),
            "lamp_angle": float(self.lamp_angle),
        }

    @classmethod
    def from_fly_meta(
        cls,
        fly_path: str | Path,
        *,
        shell_devices: dict[str, Any] | None = None,
        plasticity: PlasticityRule | None = None,
        scope_on: bool = True,
        hop_clip: bool = True,
        yaw_offset: float = 0.0,
        lamp_speed: float = 0.0,
        lamp_angle: float = 0.8,
        ocelli_on: bool = True,
        turret_on: bool = True,
    ) -> Blueprint:
        bundle = DeviceBundle.from_dict(shell_devices)
        return cls(
            fly_path=str(fly_path),
            devices=bundle,
            device_list=bundle.to_device_list(
                ocelli_on=ocelli_on, turret_on=turret_on
            ),
            plasticity=plasticity or PlasticityRule(),
            scope_on=bool(scope_on),
            hop_clip=bool(hop_clip),
            ocelli_on=ocelli_on,
            turret_on=turret_on,
            yaw_offset=float(yaw_offset),
            lamp_speed=float(lamp_speed),
            lamp_angle=float(lamp_angle),
        )

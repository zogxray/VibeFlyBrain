"""Device interface for Blueprint CNS sessions.

Devices are typed objects (not parallel ``_ocelli_*`` / ``_turret_*`` fields).
Fuse, panels, and training scope iterate ``blueprint.device_list``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

ChannelKind = Literal["sensor", "motor"]
FuseDirection = Literal["to_cns", "from_cns"]  # afferent / efferent


@dataclass(frozen=True)
class AttachPolicy:
    """Which CNS classes Fuse should prefer when wiring a channel."""

    prefer: tuple[str, ...]
    fallback: tuple[str, ...] = ("central", "sensory", "motor")
    direction: FuseDirection = "to_cns"
    tips_per_site: int = 3


# Default policies (was FUSE_* module constants).
POLICY_OCELLI = AttachPolicy(
    prefer=("descending", "ascending"),
    fallback=("central", "sensory", "motor"),
    direction="to_cns",
    tips_per_site=3,
)
POLICY_TURRET_MOTOR = AttachPolicy(
    prefer=("descending", "motor"),
    fallback=("central", "sensory", "motor"),
    direction="from_cns",
    tips_per_site=2,
)
POLICY_TURRET_SENSORY = AttachPolicy(
    prefer=("ascending", "sensory", "central"),
    fallback=("central", "sensory", "motor"),
    direction="to_cns",
    tips_per_site=2,
)
# Compound eyes → optic lobe / VPN (lamp bearing), not ocelli neck tracts.
POLICY_EYE = AttachPolicy(
    prefer=("optic", "visual_projection"),
    fallback=("central", "sensory", "visual_centrifugal"),
    direction="to_cns",
    tips_per_site=2,
)
# Johnston's organ (paired antennae) → sensory / ascending tracts.
POLICY_JO = AttachPolicy(
    prefer=("sensory", "ascending"),
    fallback=("central", "sensory", "motor"),
    direction="to_cns",
    tips_per_site=2,
)

# Per-eye photoreceptor layout: 4 azimuth corners + up + down.
EYE_SITE_LABELS: tuple[str, ...] = ("NE", "SE", "SW", "NW", "U", "D")
EYE_N_SITES = len(EYE_SITE_LABELS)


@dataclass
class DeviceSite:
    """One physical/logical attach site (cup, wing pad, …)."""

    index: int
    label: str
    group_ids: list[int] = field(default_factory=list)
    terminal_id: int | None = None


@dataclass
class Channel:
    """One I/O stream on a device (sensor readout or motor drive)."""

    name: str
    kind: ChannelKind
    sites: list[DeviceSite] = field(default_factory=list)
    attach: AttachPolicy = field(default_factory=lambda: POLICY_OCELLI)
    bus_terminal: int | None = None

    def terminal_ids(self) -> list[int]:
        out: list[int] = []
        for s in self.sites:
            if s.terminal_id is not None:
                out.append(int(s.terminal_id))
        if self.bus_terminal is not None:
            out.append(int(self.bus_terminal))
        return out

    def flat_ids(self) -> list[int]:
        out: list[int] = []
        for s in self.sites:
            out.extend(int(x) for x in s.group_ids)
            if s.terminal_id is not None:
                out.append(int(s.terminal_id))
        if self.bus_terminal is not None:
            out.append(int(self.bus_terminal))
        return out


@dataclass
class FuseLink:
    """One Fuse edge batch: owner terminal(s) ↔ CNS goal ids."""

    device: str
    channel: str
    direction: FuseDirection
    owners: list[int]  # growth tip owners / connect_pair sources
    goals: list[int]  # CNS partners
    policy: AttachPolicy


@runtime_checkable
class Device(Protocol):
    """Heterogeneous overlay: sites, channels, sense/actuate, attach-policy."""

    name: str
    enabled: bool
    kind: str  # registry key: "ocelli" | "turret" | …

    def channels(self) -> list[Channel]: ...

    def flat_ids(self) -> list[int]: ...

    def sense(self, world: Any) -> dict[int, float]: ...

    def actuate(self, drive: Any = None, *, dt: float = 0.0, world: Any = None) -> Any: ...

    def panel_rows(self) -> list[Any]: ...

    def draw(self, host: Any, cx: float, cy: float, view_lim_y: int) -> None: ...

    def to_shell_fragment(self) -> dict[str, Any]: ...


@dataclass
class OcelliDevice:
    """Three-cup light sensor → CNS (afferent Fuse)."""

    name: str = "ocelli"
    kind: str = "ocelli"
    enabled: bool = True
    groups: list[list[int]] = field(default_factory=lambda: [[], [], []])
    terminals: list[int] = field(default_factory=list)
    intensity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    attach: AttachPolicy = field(default_factory=lambda: POLICY_OCELLI)
    funnel_flag: str = "show_ocelli_funnel"
    panel_accent: tuple[int, int, int] = (140, 170, 200)
    panel_heading: str = "Ocelli"

    def channels(self) -> list[Channel]:
        sites = [
            DeviceSite(
                index=i,
                label=("M", "L", "R")[i] if i < 3 else str(i),
                group_ids=list(self.groups[i]) if i < len(self.groups) else [],
                terminal_id=(
                    int(self.terminals[i]) if i < len(self.terminals) else None
                ),
            )
            for i in range(max(len(self.groups), len(self.terminals), 3))
        ]
        return [
            Channel(
                name="light",
                kind="sensor",
                sites=sites[:3],
                attach=self.attach,
            )
        ]

    def flat_ids(self) -> list[int]:
        out: list[int] = []
        for g in self.groups:
            out.extend(int(x) for x in g)
        out.extend(int(x) for x in self.terminals)
        return out

    def sense(self, world: Any) -> dict[int, float]:
        """Photoreceptor I → cup clamps/packs. Uses world._update_ocelli_sensors."""
        if not self.enabled:
            return {}
        if hasattr(world, "_update_ocelli_sensors"):
            world._update_ocelli_sensors()
        I = np.asarray(getattr(world, "ocelli_I", self.intensity), dtype=np.float64)
        self.intensity = I.copy()
        levels: dict[int, float] = {}
        for gi, gids in enumerate(self.groups):
            if gi >= len(I):
                continue
            val = float(I[gi])
            if gi < len(self.terminals):
                levels[int(self.terminals[gi])] = val
            for nid in gids:
                levels[int(nid)] = val
        return levels

    def actuate(self, drive: Any = None, *, dt: float = 0.0, world: Any = None) -> None:
        del drive, dt, world  # pure sensor

    def panel_rows(self) -> list[Any]:
        from device_render import Bars, Hint, Toggle

        return [
            Toggle("enabled", "Device", accent=(120, 220, 140)),
            Toggle(
                "funnel",
                "Funnel",
                accent=(120, 190, 230),
                font="hint",
            ),
            Bars(
                key="readout",
                track_w=90,
                accent_on=(120, 210, 140),
                accent_off=(140, 150, 140),
                track_bg=(35, 40, 38),
                track_border=(80, 100, 85),
            ),
            Hint("Mercedes axes · cups on crown · Fuse → CNS"),
        ]

    def bar_levels(self, host: Any) -> list[tuple[str, float, bool]]:
        if hasattr(host, "_update_ocelli_sensors"):
            host._update_ocelli_sensors()
        I = np.asarray(getattr(host, "ocelli_I", self.intensity), dtype=np.float64)
        conn = (
            host._ocelli_connected_mask()
            if hasattr(host, "_ocelli_connected_mask")
            else (False, False, False)
        )
        labels = ("M", "L", "R")
        out: list[tuple[str, float, bool]] = []
        for i, lab in enumerate(labels):
            val = float(I[i]) if i < len(I) else 0.0
            ok = bool(conn[i]) if i < len(conn) else False
            out.append((lab, val, ok))
        return out

    def draw(self, host: Any, cx: float, cy: float, view_lim_y: int) -> None:
        if not self.enabled:
            return
        fn = getattr(host, "_draw_ocelli", None)
        if callable(fn):
            fn(cx, cy, view_lim_y)

    def panel_keys(self) -> list[tuple[str, str]]:
        """Deprecated — prefer panel_rows()."""
        return [("ocelli_on", "ocelli"), ("ocelli_readout", "ocelli"), ("show_ocelli_funnel", "ocelli")]

    def to_shell_fragment(self) -> dict[str, Any]:
        return {
            "ocelli_groups": [list(g) for g in self.groups],
            "ocelli_terminals": list(self.terminals),
        }

    @classmethod
    def from_shell(cls, snap: dict[str, Any], *, enabled: bool = True) -> OcelliDevice:
        return cls(
            enabled=enabled,
            groups=[
                [int(x) for x in g]
                for g in (snap.get("ocelli_groups") or [[], [], []])
            ],
            terminals=[int(x) for x in (snap.get("ocelli_terminals") or [])],
        )


@dataclass
class CompoundEyeDevice:
    """One compound eye: corner-azimuth + up/down photodiodes → optic Fuse.

    Two instances (eye_left / eye_right). Azimuth corners cover a full circle;
    U/D cover elevation. Runtime sense only — no lamp XYZ into motors.
    """

    name: str = "eye_left"
    kind: str = "eye"
    side: str = "left"  # "left" | "right"
    enabled: bool = False
    groups: list[list[int]] = field(default_factory=list)
    terminals: list[int] = field(default_factory=list)
    intensity: np.ndarray = field(
        default_factory=lambda: np.zeros(EYE_N_SITES, dtype=np.float64)
    )
    attach: AttachPolicy = field(default_factory=lambda: POLICY_EYE)
    funnel_flag: str = "show_eye_funnel"
    panel_accent: tuple[int, int, int] = (100, 190, 210)
    panel_heading: str = "Eye L"

    def __post_init__(self) -> None:
        if not self.groups:
            self.groups = [[] for _ in range(EYE_N_SITES)]
        while len(self.groups) < EYE_N_SITES:
            self.groups.append([])
        side = str(self.side).lower()
        if side.startswith("r"):
            self.side = "right"
            self.name = "eye_right"
            self.panel_heading = "Eye R"
            self.panel_accent = (90, 170, 200)
        else:
            self.side = "left"
            self.name = "eye_left"
            self.panel_heading = "Eye L"
            self.panel_accent = (100, 190, 210)

    def channels(self) -> list[Channel]:
        sites = [
            DeviceSite(
                index=i,
                label=EYE_SITE_LABELS[i] if i < EYE_N_SITES else str(i),
                group_ids=list(self.groups[i]) if i < len(self.groups) else [],
                terminal_id=(
                    int(self.terminals[i]) if i < len(self.terminals) else None
                ),
            )
            for i in range(EYE_N_SITES)
        ]
        return [
            Channel(
                name="light",
                kind="sensor",
                sites=sites,
                attach=self.attach,
            )
        ]

    def flat_ids(self) -> list[int]:
        out: list[int] = []
        for g in self.groups:
            out.extend(int(x) for x in g)
        out.extend(int(x) for x in self.terminals)
        return out

    def sense(self, world: Any) -> dict[int, float]:
        if not self.enabled:
            return {}
        if hasattr(world, "_update_eye_sensors"):
            world._update_eye_sensors()
        key = f"{self.name}_I"
        I = np.asarray(getattr(world, key, self.intensity), dtype=np.float64)
        self.intensity = I.copy() if len(I) else self.intensity
        levels: dict[int, float] = {}
        for gi, gids in enumerate(self.groups):
            if gi >= len(I):
                continue
            val = float(I[gi])
            if gi < len(self.terminals):
                levels[int(self.terminals[gi])] = val
            for nid in gids:
                levels[int(nid)] = val
        return levels

    def actuate(self, drive: Any = None, *, dt: float = 0.0, world: Any = None) -> None:
        del drive, dt, world

    def probe_light_angle(self, world: Any) -> float | None:
        """Lamp azimuth where this eye faces the lamp (reach probe reference)."""
        from compound_eyes import eye_probe_light_angle

        return eye_probe_light_angle(world, self)

    def panel_rows(self) -> list[Any]:
        from device_render import Bars, Hint, Toggle

        return [
            Toggle("enabled", "Device", accent=(100, 220, 180)),
            Toggle(
                "funnel",
                "Funnel",
                accent=(90, 180, 210),
                font="hint",
            ),
            Bars(
                key="readout",
                track_w=72,
                accent_on=(100, 210, 190),
                accent_off=(100, 130, 130),
                track_bg=(32, 40, 42),
                track_border=(70, 110, 110),
            ),
            Hint("Optic hemisphere · corners+U/D · Fuse → optic/VPN"),
        ]

    def bar_levels(self, host: Any) -> list[tuple[str, float, bool]]:
        if hasattr(host, "_update_eye_sensors"):
            host._update_eye_sensors()
        key = f"{self.name}_I"
        I = np.asarray(getattr(host, key, self.intensity), dtype=np.float64)
        conn = (
            host._eye_connected_mask(self.name)
            if hasattr(host, "_eye_connected_mask")
            else (False,) * EYE_N_SITES
        )
        out: list[tuple[str, float, bool]] = []
        for i, lab in enumerate(EYE_SITE_LABELS):
            val = float(I[i]) if i < len(I) else 0.0
            ok = bool(conn[i]) if i < len(conn) else False
            out.append((lab, val, ok))
        return out

    def draw(self, host: Any, cx: float, cy: float, view_lim_y: int) -> None:
        if not self.enabled:
            return
        fn = getattr(host, "_draw_compound_eye", None)
        if callable(fn):
            fn(self, cx, cy, view_lim_y)

    def to_shell_fragment(self) -> dict[str, Any]:
        prefix = self.name  # eye_left / eye_right
        return {
            f"{prefix}_groups": [list(g) for g in self.groups],
            f"{prefix}_terminals": list(self.terminals),
            f"{prefix}_on": bool(self.enabled),
        }

    @classmethod
    def from_shell(
        cls,
        snap: dict[str, Any],
        *,
        side: str,
        enabled: bool | None = None,
    ) -> CompoundEyeDevice:
        side_l = "right" if str(side).lower().startswith("r") else "left"
        name = f"eye_{side_l}"
        on_key = f"{name}_on"
        en = bool(snap.get(on_key, False)) if enabled is None else bool(enabled)
        raw_g = snap.get(f"{name}_groups") or []
        groups = [[int(x) for x in g] for g in raw_g] if raw_g else []
        terms = [int(x) for x in (snap.get(f"{name}_terminals") or [])]
        return cls(
            side=side_l,
            enabled=en,
            groups=groups,
            terminals=terms,
        )


@dataclass
class JohnstonOrganDevice:
    """One antennal Johnston organ: Matrix tilt sensor → sensory Fuse.

    Two instances (jo_left / jo_right). Signal is signed tilt ∈ [−1, 1]
    (degrees / ref → bipolar); CNS activity uses mid-encoded 0.5·(tilt+1).
    """

    name: str = "jo_left"
    kind: str = "jo"
    side: str = "left"  # "left" | "right"
    enabled: bool = False
    groups: list[list[int]] = field(default_factory=list)
    terminals: list[int] = field(default_factory=list)
    tilt: float = 0.0  # signed ∈ [−1, 1]
    attach: AttachPolicy = field(default_factory=lambda: POLICY_JO)
    funnel_flag: str = "show_jo_funnel"
    panel_accent: tuple[int, int, int] = (200, 120, 120)
    panel_heading: str = "JO L"

    def __post_init__(self) -> None:
        if not self.groups:
            self.groups = [[]]
        side = str(self.side).lower()
        if side.startswith("r"):
            self.side = "right"
            self.name = "jo_right"
            self.panel_heading = "JO R"
            self.panel_accent = (190, 110, 110)
        else:
            self.side = "left"
            self.name = "jo_left"
            self.panel_heading = "JO L"
            self.panel_accent = (200, 120, 120)

    def channels(self) -> list[Channel]:
        gids = list(self.groups[0]) if self.groups else []
        term = int(self.terminals[0]) if self.terminals else None
        return [
            Channel(
                name="tilt",
                kind="sensor",
                sites=[
                    DeviceSite(
                        index=0,
                        label="tilt",
                        group_ids=gids,
                        terminal_id=term,
                    )
                ],
                attach=self.attach,
            )
        ]

    def flat_ids(self) -> list[int]:
        out: list[int] = []
        for g in self.groups:
            out.extend(int(x) for x in g)
        out.extend(int(x) for x in self.terminals)
        return out

    def sense(self, world: Any) -> dict[int, float]:
        if not self.enabled:
            return {}
        if hasattr(world, "_update_jo_sensors"):
            world._update_jo_sensors()
        key = f"{self.name}_tilt"
        raw = getattr(world, key, self.tilt)
        tilt = float(np.clip(float(raw), -1.0, 1.0))
        self.tilt = tilt
        # Non-negative activity: mid-encode signed tilt.
        level = float(0.5 * (tilt + 1.0))
        levels: dict[int, float] = {}
        if self.terminals:
            levels[int(self.terminals[0])] = level
        for gids in self.groups:
            for nid in gids:
                levels[int(nid)] = level
        return levels

    def actuate(self, drive: Any = None, *, dt: float = 0.0, world: Any = None) -> None:
        del drive, dt, world

    def panel_rows(self) -> list[Any]:
        from device_render import Bars, Hint, Toggle

        return [
            Toggle("enabled", "Device", accent=(220, 140, 140)),
            Toggle(
                "funnel",
                "Funnel",
                accent=(200, 120, 120),
                font="hint",
            ),
            Bars(
                key="readout",
                track_w=90,
                accent_on=(220, 130, 130),
                accent_off=(140, 90, 90),
                track_bg=(40, 28, 28),
                track_border=(120, 70, 70),
            ),
            Hint("Antenna JO · Matrix tilt · Fuse → sensory"),
        ]

    def bar_levels(self, host: Any) -> list[tuple[str, float, bool]]:
        if hasattr(host, "_update_jo_sensors"):
            host._update_jo_sensors()
        key = f"{self.name}_tilt"
        tilt = float(np.clip(float(getattr(host, key, self.tilt)), -1.0, 1.0))
        self.tilt = tilt
        conn = (
            host._jo_connected_mask(self.name)
            if hasattr(host, "_jo_connected_mask")
            else [False]
        )
        ok = bool(conn[0]) if conn else False
        from johnston_organ import JO_TILT_REF_DEG

        deg = tilt * float(JO_TILT_REF_DEG)
        return [(f"{deg:+.0f}°", float(abs(tilt)), ok)]

    def draw(self, host: Any, cx: float, cy: float, view_lim_y: int) -> None:
        if not self.enabled:
            return
        fn = getattr(host, "_draw_johnston_organ", None)
        if callable(fn):
            fn(self, cx, cy, view_lim_y)

    def to_shell_fragment(self) -> dict[str, Any]:
        prefix = self.name
        return {
            f"{prefix}_groups": [list(g) for g in self.groups],
            f"{prefix}_terminals": list(self.terminals),
            f"{prefix}_on": bool(self.enabled),
        }

    @classmethod
    def from_shell(
        cls,
        snap: dict[str, Any],
        *,
        side: str,
        enabled: bool | None = None,
    ) -> JohnstonOrganDevice:
        side_l = "right" if str(side).lower().startswith("r") else "left"
        name = f"jo_{side_l}"
        on_key = f"{name}_on"
        en = bool(snap.get(on_key, False)) if enabled is None else bool(enabled)
        raw_g = snap.get(f"{name}_groups") or []
        groups = [[int(x) for x in g] for g in raw_g] if raw_g else []
        terms = [int(x) for x in (snap.get(f"{name}_terminals") or [])]
        return cls(
            side=side_l,
            enabled=en,
            groups=groups,
            terminals=terms,
        )


@dataclass
class TurretDevice:
    """Aimable wing-pad effector: motor efferent + sensory afferent."""

    name: str = "turret"
    kind: str = "turret"
    enabled: bool = True
    motor_groups: list[list[int]] = field(
        default_factory=lambda: [[], [], [], []]
    )
    sensory_groups: list[list[int]] = field(
        default_factory=lambda: [[], [], [], []]
    )
    motor_terminals: list[int] = field(default_factory=list)
    sensory_terminals: list[int] = field(default_factory=list)
    bus_terminal: int | None = None
    yaw: float = 0.0
    pitch: float = 0.40
    S: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    M: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    motor_attach: AttachPolicy = field(default_factory=lambda: POLICY_TURRET_MOTOR)
    sensory_attach: AttachPolicy = field(
        default_factory=lambda: POLICY_TURRET_SENSORY
    )
    funnel_flag: str = "show_turret_funnel"
    panel_accent: tuple[int, int, int] = (200, 150, 120)
    panel_heading: str = "Turret"

    def channels(self) -> list[Channel]:
        n = max(
            len(self.motor_groups),
            len(self.sensory_groups),
            len(self.motor_terminals),
            len(self.sensory_terminals),
            4,
        )
        motor_sites = [
            DeviceSite(
                index=i,
                label=f"M{i}",
                group_ids=list(self.motor_groups[i]) if i < len(self.motor_groups) else [],
                terminal_id=(
                    int(self.motor_terminals[i])
                    if i < len(self.motor_terminals)
                    else None
                ),
            )
            for i in range(n)
        ]
        sens_sites = [
            DeviceSite(
                index=i,
                label=f"S{i}",
                group_ids=(
                    list(self.sensory_groups[i]) if i < len(self.sensory_groups) else []
                ),
                terminal_id=(
                    int(self.sensory_terminals[i])
                    if i < len(self.sensory_terminals)
                    else None
                ),
            )
            for i in range(n)
        ]
        return [
            Channel(
                name="aim_motor",
                kind="motor",
                sites=motor_sites[:4],
                attach=self.motor_attach,
                bus_terminal=self.bus_terminal,
            ),
            Channel(
                name="aim_sense",
                kind="sensor",
                sites=sens_sites[:4],
                attach=self.sensory_attach,
            ),
        ]

    def flat_ids(self) -> list[int]:
        out: list[int] = []
        for g in self.motor_groups:
            out.extend(int(x) for x in g)
        for g in self.sensory_groups:
            out.extend(int(x) for x in g)
        out.extend(int(x) for x in self.motor_terminals)
        out.extend(int(x) for x in self.sensory_terminals)
        if self.bus_terminal is not None:
            out.append(int(self.bus_terminal))
        return out

    def sense(self, world: Any) -> dict[int, float]:
        """Proprioceptive afferent only — no AimTask bearing (teacher stays separate)."""
        if not self.enabled:
            return {}
        # Sync pose sensors from world yaw/pitch.
        yaw = float(getattr(world, "turret_yaw", self.yaw))
        pitch = float(getattr(world, "turret_pitch", self.pitch))
        self.yaw, self.pitch = yaw, pitch
        s_yaw = 0.5 + 0.5 * math.sin(yaw)
        span = max(1e-6, 0.85 - (-0.35))
        s_pitch = (pitch - (-0.35)) / span
        self.S = np.asarray(
            [float(np.clip(s_yaw, 0.0, 1.0)), float(np.clip(s_pitch, 0.0, 1.0))],
            dtype=np.float64,
        )
        if hasattr(world, "turret_S"):
            world.turret_S = self.S.copy()
        levels: dict[int, float] = {}
        for si in range(4):
            yaw_side = float(self.S[0]) if si in (0, 1) else (1.0 - float(self.S[0]))
            pitch_side = float(self.S[1]) if si in (0, 2) else (1.0 - float(self.S[1]))
            val = float(np.clip(0.5 * (yaw_side + pitch_side), 0.0, 1.0))
            if si < len(self.sensory_terminals):
                levels[int(self.sensory_terminals[si])] = val
            if si < len(self.sensory_groups):
                for nid in self.sensory_groups[si]:
                    levels[int(nid)] = val
        if self.bus_terminal is not None:
            levels[int(self.bus_terminal)] = float(
                0.5 * (float(self.S[0]) + float(self.S[1]))
            )
        return levels

    def actuate(self, drive: Any = None, *, dt: float = 1.0 / 60.0, world: Any = None) -> np.ndarray:
        """Decode motor clamp activity → yaw/pitch. ``drive`` unused (reads fly)."""
        del drive
        w = world
        if w is None:
            return self.M
        n = 4
        site_levels: list[float] = []
        for si in range(n):
            vals: list[float] = []
            weights: list[float] = []
            if si < len(self.motor_terminals):
                neu = w.fly.neurons.get(int(self.motor_terminals[si]))
                if neu is not None:
                    vals.append(float(neu.activity))
                    weights.append(0.7)
            if si < len(self.motor_groups):
                g = self.motor_groups[si]
                for nid in g:
                    neu = w.fly.neurons.get(int(nid))
                    if neu is None:
                        continue
                    vals.append(float(neu.activity))
                    weights.append(0.3 / max(1, len(g)))
            if not vals:
                site_levels.append(0.0)
            else:
                ww = np.asarray(weights, dtype=np.float64)
                aa = np.asarray(vals, dtype=np.float64)
                site_levels.append(
                    float(np.clip(np.sum(ww * aa) / max(1e-9, float(np.sum(ww))), 0.0, 1.0))
                )
        while len(site_levels) < 4:
            site_levels.append(0.0)
        # Motor socket readout (learned port map), else raw hop decode.
        # Sensor-port NN (separate MLP on eyes/JO/proprio) takes priority —
        # fly contact_gain / hop are not used for slew when it is on.
        port = getattr(w, "sensor_port_nn", None)
        port_on = port is not None and bool(getattr(w, "sensor_port_on", False))
        if port_on:
            if bool(getattr(w, "sensor_port_train", False)) and not bool(
                getattr(port, "frozen", False)
            ):
                port.train_step(w)
                yaw_cmd = float(port.last_cmd[0])
                pitch_cmd = float(port.last_cmd[1])
            else:
                yaw_cmd, pitch_cmd = port.commands(w)
            setattr(w, "_socket_drive_alpha", 0.0)
            try:
                from genome_shell import (
                    TURRET_CMD_DEADZONE,
                    TURRET_PITCH_MAX,
                    TURRET_PITCH_MIN,
                    turret_slew_rates,
                )

                speed = getattr(w, "lamp_speed", 0.0)
                eff = getattr(w, "effective_lamp_speed", None)
                if callable(eff):
                    try:
                        speed = eff()
                    except Exception:
                        pass
                rate_yaw, rate_pitch = turret_slew_rates(speed)
                dz = float(TURRET_CMD_DEADZONE)
                pmin, pmax = float(TURRET_PITCH_MIN), float(TURRET_PITCH_MAX)
            except Exception:
                rate_yaw = float(getattr(w, "TURRET_RATE_YAW", 0.55))
                rate_pitch = 0.50
                dz = 0.04
                pmin, pmax = -0.35, 0.85
            if abs(yaw_cmd) < dz:
                yaw_cmd = 0.0
            if abs(pitch_cmd) < dz:
                pitch_cmd = 0.0
            self.yaw = (float(self.yaw) + float(dt) * rate_yaw * yaw_cmd) % math.tau
            self.pitch = float(
                np.clip(
                    float(self.pitch) + float(dt) * rate_pitch * pitch_cmd, pmin, pmax
                )
            )
            self.M = np.asarray(
                [
                    float(np.clip(0.5 + 0.5 * yaw_cmd, 0.0, 1.0)),
                    float(np.clip(0.5 + 0.5 * pitch_cmd, 0.0, 1.0)),
                ],
                dtype=np.float64,
            )
            w.turret_yaw = float(self.yaw)
            w.turret_pitch = float(self.pitch)
            w.turret_M = self.M.copy()
            w._frame_turret_pose = None
            if hasattr(w, "_update_turret_aim_metric"):
                w._update_turret_aim_metric()
            if hasattr(w, "blueprint"):
                td = w.blueprint.get_turret()
                if td is not None and td is not self:
                    td.yaw, td.pitch = self.yaw, self.pitch
                    td.M = self.M.copy()
            return self.M

        sock = getattr(w, "motor_socket", None)
        sock_on = sock is not None and bool(getattr(w, "motor_socket_on", False))
        teach_yaw = teach_pitch = None
        alpha = 0.0
        if sock_on:
            site_levels = list(sock.levels(w))
            # Train assist: smooth pursuit on lamp-now (no lead/ff).
            # Frozen / Learn off → pure W (honest closed-loop test).
            frozen = bool(getattr(sock, "frozen", False))
            learn_on = bool(getattr(w, "learn_on", False))
            if learn_on and not frozen:
                alpha = 0.90
                rule = getattr(w, "rule", None)
                if rule is not None:
                    alpha = float(
                        np.clip(
                            getattr(rule, "socket_teacher_drive", alpha), 0.0, 1.0
                        )
                    )
                try:
                    from aim_task import AimTask

                    task = getattr(w, "_aim_task", None)
                    if task is None:
                        task = AimTask()
                        w._aim_task = task
                    cmds = task.pursuit_cmds(w)
                    if cmds is not None:
                        teach_yaw, teach_pitch = float(cmds[0]), float(cmds[1])
                except Exception:
                    teach_yaw = teach_pitch = None
                # Keep site blend as fallback if pursuit unavailable.
                if teach_yaw is None:
                    tgts = None
                    fn = getattr(w, "_turret_teacher_site_levels", None)
                    if callable(fn):
                        try:
                            tgts = fn()
                        except Exception:
                            tgts = None
                    if tgts is not None and alpha > 1e-6:
                        t = [float(x) for x in list(tgts)[:4]]
                        while len(t) < 4:
                            t.append(0.5)
                        site_levels = [
                            float(
                                np.clip(
                                    (1.0 - alpha) * float(site_levels[i])
                                    + alpha * t[i],
                                    0.0,
                                    1.0,
                                )
                            )
                            for i in range(4)
                        ]
            setattr(w, "_socket_drive_alpha", float(alpha) if teach_yaw is not None or alpha > 0 else 0.0)
        if teach_yaw is not None and teach_pitch is not None and alpha > 1e-6:
            # Direct cmd blend — skip site encode (no gain tricks).
            w_left = 0.5 * (site_levels[0] + site_levels[1])
            w_right = 0.5 * (site_levels[2] + site_levels[3])
            w_lo = 0.5 * (site_levels[0] + site_levels[2])
            w_hi = 0.5 * (site_levels[1] + site_levels[3])
            w_yaw = float(np.clip(w_left - w_right, -1.0, 1.0))
            w_pitch = float(np.clip(w_lo - w_hi, -1.0, 1.0))
            yaw_cmd = float(
                np.clip((1.0 - alpha) * w_yaw + alpha * teach_yaw, -1.0, 1.0)
            )
            pitch_cmd = float(
                np.clip((1.0 - alpha) * w_pitch + alpha * teach_pitch, -1.0, 1.0)
            )
            setattr(w, "_socket_drive_alpha", float(alpha))
        else:
            left = 0.5 * (site_levels[0] + site_levels[1])
            right = 0.5 * (site_levels[2] + site_levels[3])
            low_z = 0.5 * (site_levels[0] + site_levels[2])
            high_z = 0.5 * (site_levels[1] + site_levels[3])
            yaw_cmd = float(np.clip(left - right, -1.0, 1.0))
            pitch_cmd = float(np.clip(low_z - high_z, -1.0, 1.0))
        try:
            from genome_shell import (
                TURRET_CMD_DEADZONE,
                TURRET_PITCH_MAX,
                TURRET_PITCH_MIN,
                turret_slew_rates,
            )

            speed = getattr(w, "lamp_speed", 0.0)
            eff = getattr(w, "effective_lamp_speed", None)
            if callable(eff):
                try:
                    speed = eff()
                except Exception:
                    pass
            rate_yaw, rate_pitch = turret_slew_rates(speed)
            dz = float(TURRET_CMD_DEADZONE)
            pmin, pmax = float(TURRET_PITCH_MIN), float(TURRET_PITCH_MAX)
        except Exception:
            rate_yaw = float(getattr(w, "TURRET_RATE_YAW", 0.55))
            rate_pitch = 0.50
            dz = 0.04
            pmin, pmax = -0.35, 0.85
        # Teacher-heavy train assist: don't deadzone away real chase cmds.
        if float(getattr(w, "_socket_drive_alpha", 0.0) or 0.0) >= 0.5:
            dz = 0.0
        if abs(yaw_cmd) < dz:
            yaw_cmd = 0.0
        if abs(pitch_cmd) < dz:
            pitch_cmd = 0.0
        self.yaw = (float(self.yaw) + float(dt) * rate_yaw * yaw_cmd) % math.tau
        self.pitch = float(
            np.clip(float(self.pitch) + float(dt) * rate_pitch * pitch_cmd, pmin, pmax)
        )
        self.M = np.asarray(
            [
                float(np.clip(0.5 + 0.5 * yaw_cmd, 0.0, 1.0)),
                float(np.clip(0.5 + 0.5 * pitch_cmd, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )
        w.turret_yaw = float(self.yaw)
        w.turret_pitch = float(self.pitch)
        w.turret_M = self.M.copy()
        w._frame_turret_pose = None
        if hasattr(w, "_update_turret_aim_metric"):
            w._update_turret_aim_metric()
        # Keep typed device in blueprint in sync.
        if hasattr(w, "blueprint"):
            td = w.blueprint.get_turret()
            if td is not None and td is not self:
                td.yaw, td.pitch = self.yaw, self.pitch
                td.M = self.M.copy()
        return self.M

    def panel_rows(self) -> list[Any]:
        from device_render import Bars, Hint, Meter, Toggle

        return [
            Toggle("enabled", "Device", accent=(230, 160, 120)),
            Toggle(
                "funnel",
                "Funnel",
                accent=(230, 160, 110),
                font="hint",
            ),
            Toggle(
                "show_turret_aim",
                "Aim cone",
                accent=(220, 180, 120),
                font="hint",
            ),
            Bars(
                key="readout",
                track_w=52,
                accent_on=(230, 150, 120),
                accent_off=(160, 100, 90),
                track_bg=(40, 35, 32),
                track_border=(90, 80, 70),
            ),
            Meter("aim"),
            Hint("Aim / ports — Learn lives in Learn tab"),
        ]

    def bar_levels(self, host: Any) -> list[tuple[str, float, bool]]:
        if hasattr(host, "_update_turret_sensors"):
            host._update_turret_sensors()
        conn = (
            host._turret_connected_mask()
            if hasattr(host, "_turret_connected_mask")
            else (False, False, False, False)
        )
        labels = ("LA", "LP", "RA", "RP")
        out: list[tuple[str, float, bool]] = []
        for i, lab in enumerate(labels):
            if hasattr(host, "_turret_site_sensory_level"):
                val = float(host._turret_site_sensory_level(i))
            else:
                val = 0.5
            ok = bool(conn[i]) if i < len(conn) else False
            out.append((lab, val, ok))
        return out

    def draw(self, host: Any, cx: float, cy: float, view_lim_y: int) -> None:
        if not self.enabled:
            return
        fn = getattr(host, "_draw_turret", None)
        if callable(fn):
            fn(cx, cy, view_lim_y)

    def panel_keys(self) -> list[tuple[str, str]]:
        """Deprecated — prefer panel_rows()."""
        return [
            ("turret_on", "turret"),
            ("turret_readout", "turret"),
            ("turret_aim", "turret"),
            ("show_turret_funnel", "turret"),
            ("show_turret_aim", "turret"),
        ]

    def to_shell_fragment(self) -> dict[str, Any]:
        return {
            "turret_motor_groups": [list(g) for g in self.motor_groups],
            "turret_sensory_groups": [list(g) for g in self.sensory_groups],
            "turret_motor_terminals": list(self.motor_terminals),
            "turret_sensory_terminals": list(self.sensory_terminals),
            "turret_bus_terminal": self.bus_terminal,
            "turret_yaw": float(self.yaw),
            "turret_pitch": float(self.pitch),
        }

    @classmethod
    def from_shell(cls, snap: dict[str, Any], *, enabled: bool = True) -> TurretDevice:
        bus = snap.get("turret_bus_terminal")
        return cls(
            enabled=enabled,
            motor_groups=[
                [int(x) for x in g]
                for g in (snap.get("turret_motor_groups") or [[], [], [], []])
            ],
            sensory_groups=[
                [int(x) for x in g]
                for g in (snap.get("turret_sensory_groups") or [[], [], [], []])
            ],
            motor_terminals=[
                int(x) for x in (snap.get("turret_motor_terminals") or [])
            ],
            sensory_terminals=[
                int(x) for x in (snap.get("turret_sensory_terminals") or [])
            ],
            bus_terminal=int(bus) if bus is not None else None,
            yaw=float(snap.get("turret_yaw", 0.0)),
            pitch=float(snap.get("turret_pitch", 0.40)),
        )


@dataclass
class StubBeaconDevice:
    """Third-device stub: panel + draw without genome_shell kind branches."""

    name: str = "beacon"
    kind: str = "beacon"
    enabled: bool = True
    funnel_flag: str = ""
    panel_accent: tuple[int, int, int] = (180, 140, 220)
    panel_heading: str = "Beacon"
    level: float = 0.35

    def channels(self) -> list[Channel]:
        return []

    def flat_ids(self) -> list[int]:
        return []

    def sense(self, world: Any) -> dict[int, float]:
        del world
        return {}

    def actuate(self, drive: Any = None, *, dt: float = 0.0, world: Any = None) -> None:
        del drive, dt, world

    def panel_rows(self) -> list[Any]:
        from device_render import Bars, Hint, Toggle

        return [
            Toggle("enabled", "Device", accent=(180, 140, 220)),
            Bars(
                key="readout",
                track_w=80,
                accent_on=(200, 160, 240),
                accent_off=(120, 100, 140),
            ),
            Hint("Stub third device — panel via descriptors only"),
        ]

    def bar_levels(self, host: Any) -> list[tuple[str, float, bool]]:
        del host
        v = float(np.clip(self.level, 0.0, 1.0))
        return [("A", v, self.enabled), ("B", 1.0 - v, False)]

    def draw(self, host: Any, cx: float, cy: float, view_lim_y: int) -> None:
        if not self.enabled:
            return
        fn = getattr(host, "_draw_beacon", None)
        if callable(fn):
            fn(cx, cy, view_lim_y)

    def panel_keys(self) -> list[tuple[str, str]]:
        return [("beacon_on", "beacon")]

    def to_shell_fragment(self) -> dict[str, Any]:
        return {"beacon_level": float(self.level)}


_REGISTRY: dict[str, type] = {
    "ocelli": OcelliDevice,
    "turret": TurretDevice,
    "eye": CompoundEyeDevice,
    "jo": JohnstonOrganDevice,
    "beacon": StubBeaconDevice,
}


def register_device(kind: str, cls: type) -> None:
    _REGISTRY[str(kind)] = cls


def devices_from_shell(
    snap: dict[str, Any] | None,
    *,
    ocelli_on: bool = True,
    turret_on: bool = True,
) -> list[Any]:
    """Build typed device list from legacy ``shell_devices`` blob."""
    raw = snap or {}
    if "ocelli_groups" not in raw and isinstance(raw.get("shell_devices"), dict):
        raw = raw["shell_devices"]
    if "ocelli_groups" not in raw and isinstance(raw.get("devices"), dict):
        inner = raw["devices"]
        if "ocelli_groups" in inner:
            raw = inner
    out: list[Any] = [
        OcelliDevice.from_shell(raw, enabled=ocelli_on),
        CompoundEyeDevice.from_shell(raw, side="left"),
        CompoundEyeDevice.from_shell(raw, side="right"),
        JohnstonOrganDevice.from_shell(raw, side="left"),
        JohnstonOrganDevice.from_shell(raw, side="right"),
        TurretDevice.from_shell(raw, enabled=turret_on),
    ]
    return out


def devices_to_shell(devices: list[Any]) -> dict[str, Any]:
    """Merge device fragments into one ``shell_devices`` dict."""
    merged: dict[str, Any] = {}
    for d in devices:
        frag = d.to_shell_fragment()
        merged.update(frag)
    return merged


def all_device_ids(devices: list[Any]) -> set[int]:
    ids: set[int] = set()
    for d in devices:
        if getattr(d, "enabled", True):
            ids.update(int(x) for x in d.flat_ids())
    return ids


def panel_entries(devices: list[Any]) -> list[tuple[str, str]]:
    """Legacy key list derived from panel_rows (Toggle keys + kind)."""
    from device_render import Toggle

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for d in devices:
        group = getattr(d, "kind", "device")
        rows = d.panel_rows() if hasattr(d, "panel_rows") else []
        for row in rows:
            if not isinstance(row, Toggle):
                continue
            key = row.key
            if key == "enabled":
                key = f"{group}_on"
            elif key == "funnel":
                key = getattr(d, "funnel_flag", "") or f"show_{group}_funnel"
            if key in seen or not key:
                continue
            seen.add(key)
            out.append((key, group))
    return out

"""Panel row descriptors + device draw/click — no ``if kind == ocelli/turret`` in legend.

GenomeShell builds rows from ``device.panel_rows()`` / ``blueprint_panel_rows()``
and dispatches draw/click through this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Row descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Toggle:
    """Checkbox bound to host attr, device.enabled, or device funnel flag."""

    key: str  # "enabled" | "funnel" | "learn_on" | "scope_on" | "hop_clip" | …
    label: str
    accent: tuple[int, int, int] = (120, 220, 140)
    font: Literal["small", "hint"] = "small"


@dataclass(frozen=True)
class Bars:
    """Shared multi-channel level bars (ocelli / turret / stub readouts)."""

    key: str = "readout"
    track_w: int = 70
    accent_on: tuple[int, int, int] = (120, 210, 140)
    accent_off: tuple[int, int, int] = (160, 100, 90)
    track_bg: tuple[int, int, int] = (35, 40, 38)
    track_border: tuple[int, int, int] = (80, 100, 85)


@dataclass(frozen=True)
class Meter:
    """Single fill meter + label (aim / learn / scope)."""

    key: str  # "aim" | "learn" | "scope"


@dataclass(frozen=True)
class Hint:
    text: str
    color: tuple[int, int, int] = (110, 115, 130)


@dataclass(frozen=True)
class Action:
    """One-shot panel button (freeze / reset / …)."""

    key: str
    label: str
    accent: tuple[int, int, int] = (160, 200, 180)


PanelRow = Union[Toggle, Bars, Meter, Hint, Action]

# Visual / click interaction for panel rows.
# active  = pressed / current mode (bright, filled)
# idle    = can click, not current
# disabled = muted, no hit
RowUI = Literal["active", "idle", "disabled"]

_MUTE = (95, 95, 105)
_MUTE_TEXT = (105, 105, 118)
_IDLE_BORDER = (120, 120, 140)


def _port_frozen(host: Any) -> bool:
    nn = getattr(host, "sensor_port_nn", None)
    return bool(nn is not None and getattr(nn, "frozen", False))


def _socket_frozen(host: Any) -> bool:
    sock = getattr(host, "motor_socket", None)
    return bool(sock is not None and getattr(sock, "frozen", False))


def row_ui_state(host: Any, row: PanelRow, device: Any | None = None) -> RowUI:
    """Which control is pressed, which can be pressed, which is muted."""
    port_on = bool(getattr(host, "sensor_port_on", False))
    port_train = bool(getattr(host, "sensor_port_train", False)) and not _port_frozen(host)
    sock_on = bool(getattr(host, "motor_socket_on", False))
    scope_on = bool(getattr(host, "scope_on", False))
    device_on = bool(getattr(device, "enabled", True)) if device is not None else True

    if isinstance(row, Toggle):
        key = row.key
        on = toggle_is_on(host, row, device)
        # Master device power — always clickable.
        if key == "enabled":
            return "active" if on else "idle"
        # Children of a powered-off device block.
        if device is not None and not device_on and key != "enabled":
            return "disabled"
        # Scope child.
        if key == "hop_clip" and not scope_on:
            return "disabled"
        # Exclusive drive: AI Brain vs CNS Learn vs Socket.
        if key == "learn_on" and port_on:
            return "disabled"
        if key == "motor_socket_on" and port_on:
            return "disabled"
        if key == "sensor_port_on" and sock_on:
            return "disabled"
        # AI Brain train lives inside AI Brain drive block.
        if key == "sensor_port_train":
            if not port_on:
                return "disabled"
            if _port_frozen(host):
                return "idle"  # click → unfreeze + train
            return "active" if on else "idle"
        return "active" if on else "idle"

    if isinstance(row, Action):
        key = row.key
        # Socket actions — only inside socket block.
        if key.startswith("socket_"):
            if not sock_on or port_on:
                return "disabled"
            frozen = _socket_frozen(host)
            if key == "socket_freeze":
                return "active" if frozen else "idle"
            if key == "socket_unfreeze":
                return "active" if not frozen else "idle"
            if key == "socket_reset":
                return "idle"
            return "disabled"
        # AI Brain actions — only inside AI Brain block.
        if key.startswith("port_"):
            if not port_on:
                return "disabled"
            frozen = _port_frozen(host)
            if key == "port_freeze":
                return "active" if frozen else "idle"
            if key == "port_unfreeze":
                if frozen:
                    return "idle"  # available: train again
                if port_train:
                    return "active"  # already training
                return "idle"
            if key == "port_reset":
                return "idle"
            return "disabled"
        return "idle"

    if isinstance(row, Bars):
        if device is not None and not device_on:
            return "disabled"
        return "active"

    if isinstance(row, Meter):
        if row.key == "port_nn":
            return "active" if port_on else "disabled"
        if row.key == "learn":
            if port_on:
                return "disabled"
            return "active" if bool(getattr(host, "learn_on", False)) else "idle"
        if row.key == "scope":
            return "active" if scope_on else "disabled"
        if row.key == "aim" and device is not None and not device_on:
            return "disabled"
        return "active"

    if isinstance(row, Hint):
        text = row.text
        if (
            text.startswith("— AI Brain")
            or text.startswith("AI Brain:")
            or "AI Brain exclusive" in text
            or text.startswith("— Sensor port")
            or text.startswith("Port:")
        ):
            return "active" if port_on else "disabled"
        if "Motor socket" in text:
            return "active" if sock_on else "disabled"
        if device is not None and not device_on:
            return "disabled"
        return "idle"

    return "idle"


def action_label(host: Any, row: Action) -> str:
    """Label shows pressed mode when active."""
    st = row_ui_state(host, row, None)
    if row.key == "port_freeze" and st == "active":
        return "● Frozen"
    if row.key == "port_unfreeze" and st == "active":
        return "● Training"
    if row.key == "socket_freeze" and st == "active":
        return "● Socket frozen"
    if row.key == "socket_unfreeze" and st == "active":
        return "● Socket training"
    return row.label


def blueprint_panel_rows() -> list[PanelRow]:
    """Training-scope block (not per-device)."""
    return [
        Toggle(
            "scope_on",
            "Training scope",
            accent=(180, 220, 140),
            font="hint",
        ),
        Toggle(
            "hop_clip",
            "Clip hop to scope",
            accent=(160, 200, 220),
            font="hint",
        ),
        Meter("scope"),
        Hint("Scope = sensor→motor paths · hop clip optional"),
    ]


def learn_panel_rows() -> list[PanelRow]:
    """Global learn + motor-socket + sensor-port NN controls."""
    return [
        Toggle("learn_on", "Learn (CNS Brain)", accent=(160, 210, 255)),
        Meter("learn"),
        Toggle(
            "motor_socket_on",
            "Motor socket",
            accent=(180, 220, 160),
            font="hint",
        ),
        Action("socket_freeze", "Freeze socket", accent=(180, 220, 150)),
        Action("socket_unfreeze", "Train socket", accent=(160, 190, 210)),
        Action("socket_reset", "Reset socket W", accent=(220, 170, 140)),
        Hint("— AI Brain (MLP · CNS Brain frozen) —", color=(140, 160, 150)),
        Toggle(
            "sensor_port_on",
            "AI Brain drive",
            accent=(200, 180, 120),
            font="hint",
        ),
        Toggle(
            "sensor_port_train",
            "AI Brain train",
            accent=(220, 190, 120),
            font="hint",
        ),
        Meter("port_nn"),
        Hint("AI Brain: loss↓ + Track↑ under Spin · then Freeze", color=(150, 145, 130)),
        Action("port_freeze", "Freeze AI Brain", accent=(200, 190, 130)),
        Action("port_unfreeze", "Train AI Brain again", accent=(180, 170, 140)),
        Action("port_reset", "Reset AI Brain", accent=(220, 160, 120)),
        Hint("AI Brain: eyes+JO+proprio → yaw/pitch · no hop / CNS Brain dark"),
        Hint("CNS Learn = contact_gain carve (CNS Brain, separate from AI Brain)"),
        Hint("Motor socket = legacy W@[I] bypass"),
    ]


# ---------------------------------------------------------------------------
# State resolve / click
# ---------------------------------------------------------------------------


def toggle_is_on(host: Any, row: Toggle, device: Any | None) -> bool:
    if row.key == "enabled" and device is not None:
        return bool(getattr(device, "enabled", True))
    if row.key == "funnel" and device is not None:
        attr = getattr(device, "funnel_flag", "") or ""
        return bool(getattr(host, attr, False)) if attr else False
    if row.key == "scope_on":
        return bool(getattr(host, "scope_on", False))
    if row.key == "socket_frozen":
        sock = getattr(host, "motor_socket", None)
        return bool(sock is not None and getattr(sock, "frozen", False))
    if row.key == "motor_socket_on":
        return bool(getattr(host, "motor_socket_on", False))
    if row.key == "sensor_port_on":
        return bool(getattr(host, "sensor_port_on", False))
    if row.key == "sensor_port_train":
        return bool(getattr(host, "sensor_port_train", False))
    return bool(getattr(host, row.key, False))


def _ensure_motor_socket(host: Any) -> Any:
    sock = getattr(host, "motor_socket", None)
    if sock is not None:
        return sock
    from motor_socket import MotorSocket

    rule = getattr(host, "rule", None)
    lr = float(getattr(rule, "motor_socket_lr", 0.35) or 0.35)
    sock = MotorSocket(lr=lr)
    host.motor_socket = sock
    host.motor_socket_on = bool(getattr(host, "motor_socket_on", False))
    return sock


def _ensure_sensor_port(host: Any) -> Any:
    nn = getattr(host, "sensor_port_nn", None)
    if nn is not None:
        return nn
    from sensor_port_nn import SensorPortNN

    nn = SensorPortNN(lr=0.02)
    host.sensor_port_nn = nn
    host.sensor_port_on = bool(getattr(host, "sensor_port_on", False))
    host.sensor_port_train = bool(getattr(host, "sensor_port_train", False))
    return nn


def handle_row_click(
    host: Any, row: PanelRow, device: Any | None = None
) -> bool:
    """Apply one panel click. True if consumed."""
    if row_ui_state(host, row, device) == "disabled":
        return False
    if isinstance(row, Action):
        return bool(handle_action_click(host, row))
    if not isinstance(row, Toggle):
        return False
    if row.key == "enabled" and device is not None:
        device.enabled = not bool(getattr(device, "enabled", True))
        sync_enabled_to_host(host, device)
        kind = getattr(device, "kind", "")
        if device.enabled:
            if kind == "ocelli" and hasattr(host, "_ensure_ocelli_neurons"):
                host._ensure_ocelli_neurons()
            elif kind == "turret" and hasattr(host, "_ensure_turret_neurons"):
                host._ensure_turret_neurons()
            elif kind == "eye" and hasattr(host, "_ensure_eye_neurons"):
                host._ensure_eye_neurons(device)
            elif kind == "jo" and hasattr(host, "_ensure_jo_neurons"):
                host._ensure_jo_neurons(device)
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    if row.key == "funnel" and device is not None:
        attr = getattr(device, "funnel_flag", "") or ""
        if not attr:
            return False
        setattr(host, attr, not bool(getattr(host, attr, False)))
        if hasattr(host, "_sync_device_funnel_gravity"):
            host._sync_device_funnel_gravity()
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    if row.key == "learn_on":
        host.learn_on = not bool(getattr(host, "learn_on", False))
        if hasattr(host, "blueprint"):
            host.blueprint.learn_on = bool(host.learn_on)
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    if row.key == "motor_socket_on":
        host.motor_socket_on = not bool(getattr(host, "motor_socket_on", False))
        if host.motor_socket_on:
            _ensure_motor_socket(host)
            # Mutual exclusion with port NN drive.
            host.sensor_port_on = False
        if hasattr(host, "rule"):
            host.rule.motor_socket_on = bool(host.motor_socket_on)
        if hasattr(host, "blueprint") and hasattr(host.blueprint, "plasticity"):
            host.blueprint.plasticity.motor_socket_on = bool(host.motor_socket_on)
        if hasattr(host, "_set_io_status"):
            host._set_io_status(
                "Motor socket ON" if host.motor_socket_on else "Motor socket off"
            )
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    if row.key == "sensor_port_on":
        host.sensor_port_on = not bool(getattr(host, "sensor_port_on", False))
        if host.sensor_port_on:
            _ensure_sensor_port(host)
            host.motor_socket_on = False
            # AI Brain exclusive: tissue Learn stays visually toggled but inactive
            # while drive is on (see loop.py). Clear flag so UI is honest.
            host.learn_on = False
            if hasattr(host, "blueprint"):
                host.blueprint.learn_on = False
            for attr in (
                "eye_left_on",
                "eye_right_on",
                "jo_left_on",
                "jo_right_on",
                "turret_on",
            ):
                if hasattr(host, attr):
                    setattr(host, attr, True)
        if hasattr(host, "_set_io_status"):
            host._set_io_status(
                "AI Brain drive ON · CNS Learn off · CNS Brain unused for slew"
                if host.sensor_port_on
                else "AI Brain drive off"
            )
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    if row.key == "sensor_port_train":
        host.sensor_port_train = not bool(getattr(host, "sensor_port_train", False))
        nn = _ensure_sensor_port(host)
        if host.sensor_port_train:
            host.sensor_port_on = True
            host.motor_socket_on = False
            host.learn_on = False
            if hasattr(host, "blueprint"):
                host.blueprint.learn_on = False
            nn.unfreeze()
            for attr in (
                "eye_left_on",
                "eye_right_on",
                "jo_left_on",
                "jo_right_on",
                "turret_on",
            ):
                if hasattr(host, attr):
                    setattr(host, attr, True)
        if hasattr(host, "_set_io_status"):
            try:
                badge, _, _ = nn.format_ui_badge(host)
                host._set_io_status(badge)
            except Exception:
                tag = "TRAIN" if host.sensor_port_train and not nn.frozen else "idle"
                host._set_io_status(f"AI Brain {tag} · graph frozen")
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        print(nn.format_console_line(host), flush=True)
        return True
    if row.key == "socket_frozen":
        sock = _ensure_motor_socket(host)
        if sock.frozen:
            sock.unfreeze()
            if hasattr(host, "_learn_on_train_again"):
                host._learn_on_train_again()
            if hasattr(host, "_set_io_status"):
                host._set_io_status("TRAIN · teacher can update W")
            print("[socket] TRAIN (unfrozen via UI)", flush=True)
        else:
            sock.freeze()
            if hasattr(host, "_learn_on_freeze"):
                host._learn_on_freeze()
            if hasattr(host, "_set_io_status"):
                host._set_io_status("FROZEN · W locked · teacher off · hop+sense live")
            print("[socket] FROZEN via UI — W locked, teacher off", flush=True)
        host.motor_socket_on = True
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    if row.key == "scope_on":
        if not getattr(host, "scope_on", False):
            ok = host.apply_training_scope() if hasattr(host, "apply_training_scope") else False
            if not ok and hasattr(host, "_save_ui_state"):
                host._save_ui_state()
                return True
        else:
            if hasattr(host, "clear_training_scope"):
                host.clear_training_scope()
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    if row.key == "hop_clip":
        host.hop_clip = not bool(getattr(host, "hop_clip", True))
        if hasattr(host, "blueprint"):
            host.blueprint.hop_clip = bool(host.hop_clip)
        if getattr(host, "scope_on", False) and hasattr(host, "apply_training_scope"):
            host.apply_training_scope()
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    # Generic bool attr toggle.
    if hasattr(host, row.key):
        setattr(host, row.key, not bool(getattr(host, row.key)))
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    return False


def handle_action_click(host: Any, row: Action) -> bool:
    st = row_ui_state(host, row, None)
    if st == "disabled":
        return False
    # Already in this mode — keep visual pressed, no further work.
    if st == "active" and row.key in (
        "port_freeze",
        "socket_freeze",
        "port_unfreeze",
        "socket_unfreeze",
    ):
        return True
    sock = _ensure_motor_socket(host)
    if row.key == "socket_freeze":
        sock.freeze()
        host.motor_socket_on = True
        if hasattr(host, "_learn_on_freeze"):
            host._learn_on_freeze()
        if hasattr(host, "_set_io_status"):
            host._set_io_status("FROZEN · W locked · teacher off · hop+sense live")
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        print("[socket] FROZEN via UI — W locked, teacher off (persisted)", flush=True)
        return True
    if row.key == "socket_unfreeze":
        sock.unfreeze()
        host.motor_socket_on = True
        if hasattr(host, "_learn_on_train_again"):
            host._learn_on_train_again()
        if hasattr(host, "_set_io_status"):
            host._set_io_status("TRAIN · teacher can update W")
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        print("[socket] TRAIN via UI", flush=True)
        return True
    if row.key == "socket_reset":
        from motor_socket import MotorSocket

        lr = float(getattr(sock, "lr", 0.35) or 0.35)
        fresh = MotorSocket(lr=lr)
        sock.W = fresh.W.copy()
        sock.unfreeze()
        sock._sync_gains_badge()
        host.motor_socket_on = True
        host.motor_socket_dw_ema = 0.0
        if hasattr(host, "_learn_on_reset"):
            host._learn_on_reset()
        if hasattr(host, "_set_io_status"):
            host._set_io_status("Socket · W reset · TRAIN")
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        print("[socket] RESET W → TRAIN", flush=True)
        return True
    if row.key in ("port_freeze", "port_unfreeze", "port_reset"):
        nn = _ensure_sensor_port(host)
        if row.key == "port_freeze":
            nn.freeze()
            host.sensor_port_on = True
            host.sensor_port_train = False
            host.motor_socket_on = False
            host.learn_on = False
            if hasattr(host, "blueprint"):
                host.blueprint.learn_on = False
            if hasattr(host, "_set_io_status"):
                try:
                    host._set_io_status(nn.format_ui_badge(host)[0])
                except Exception:
                    host._set_io_status("AI Brain FROZEN · drive on · CNS Learn off")
            print(nn.format_console_line(host), flush=True)
        elif row.key == "port_unfreeze":
            nn.unfreeze()
            host.sensor_port_on = True
            host.sensor_port_train = True
            host.motor_socket_on = False
            host.learn_on = False
            if hasattr(host, "blueprint"):
                host.blueprint.learn_on = False
            if hasattr(host, "_set_io_status"):
                try:
                    host._set_io_status(nn.format_ui_badge(host)[0])
                except Exception:
                    host._set_io_status("AI Brain TRAIN · eyes+JO+proprio → cmd")
            print(nn.format_console_line(host), flush=True)
        else:
            nn.reset_weights()
            host.sensor_port_on = True
            host.sensor_port_train = True
            host.motor_socket_on = False
            host.learn_on = False
            if hasattr(host, "blueprint"):
                host.blueprint.learn_on = False
            if hasattr(host, "_set_io_status"):
                try:
                    host._set_io_status(nn.format_ui_badge(host)[0])
                except Exception:
                    host._set_io_status("AI Brain reset · TRAIN")
            print(nn.format_console_line(host), flush=True)
        if hasattr(host, "_save_ui_state"):
            host._save_ui_state()
        return True
    return False


def sync_enabled_to_host(host: Any, device: Any) -> None:
    """Mirror ``d.enabled`` → legacy host/Blueprint flags."""
    kind = getattr(device, "kind", "")
    en = bool(getattr(device, "enabled", True))
    if kind == "ocelli":
        host.ocelli_on = en
        if hasattr(host, "blueprint"):
            host.blueprint.ocelli_on = en
    elif kind == "turret":
        host.turret_on = en
        if hasattr(host, "blueprint"):
            host.blueprint.turret_on = en
    elif kind == "eye":
        name = getattr(device, "name", "eye_left")
        setattr(host, f"{name}_on", en)
    elif kind == "jo":
        name = getattr(device, "name", "jo_left")
        setattr(host, f"{name}_on", en)
    # Keep typed list flag in sync for other kinds too.
    if hasattr(host, "blueprint") and hasattr(host.blueprint, "device_list"):
        pass  # device object already mutated


def sync_enabled_from_host(host: Any) -> None:
    """Pull legacy ocelli_on/turret_on into ``d.enabled`` (load / UI hydrate)."""
    devices = list(
        getattr(host, "device_list", None)
        or getattr(getattr(host, "blueprint", None), "device_list", None)
        or []
    )
    for d in devices:
        kind = getattr(d, "kind", "")
        if kind == "ocelli":
            d.enabled = bool(getattr(host, "ocelli_on", True))
        elif kind == "turret":
            d.enabled = bool(getattr(host, "turret_on", True))
        elif kind == "eye":
            name = getattr(d, "name", "eye_left")
            if hasattr(host, f"{name}_on"):
                d.enabled = bool(getattr(host, f"{name}_on"))
        elif kind == "jo":
            name = getattr(d, "name", "jo_left")
            if hasattr(host, f"{name}_on"):
                d.enabled = bool(getattr(host, f"{name}_on"))


# ---------------------------------------------------------------------------
# Value providers
# ---------------------------------------------------------------------------


def bars_levels(
    host: Any, device: Any | None, row: Bars
) -> list[tuple[str, float, bool]]:
    """(label, value∈[0,1], connected) for Bars renderer."""
    if device is not None and hasattr(device, "bar_levels"):
        return list(device.bar_levels(host))
    return []


def meter_state(
    host: Any, row: Meter, device: Any | None
) -> tuple[str, float, tuple[int, int, int]]:
    """(label, fill∈[0,1], rgb)."""
    key = row.key
    if key == "aim":
        if hasattr(host, "_update_turret_aim_metric"):
            host._update_turret_aim_metric()
        err = float(getattr(host, "turret_aim_err", float("nan")))
        if math.isfinite(err):
            score = (
                float(host._turret_aim_score())
                if hasattr(host, "_turret_aim_score")
                else float(np.clip(1.0 - err / (0.5 * math.pi), 0.0, 1.0))
            )
            deg = err * (180.0 / math.pi)
            if deg < 15.0:
                rgb = (120, 220, 150)
            elif deg < 45.0:
                rgb = (220, 200, 120)
            else:
                rgb = (230, 140, 120)
            return f"aim {deg:.0f}°", score, rgb
        return "aim —", 0.0, (140, 135, 130)
    if key == "learn":
        kind = (
            host._constructor_panel_kind()
            if hasattr(host, "_constructor_panel_kind")
            else "learn"
        )
        if kind == "growth" and hasattr(host, "_yield_phase"):
            st = host._yield_phase()
            label = str(st.get("badge", "Growth"))
            accent = tuple(st.get("accent", (170, 200, 230)))
            pct = float(st.get("pct", 0.0) or 0.0)
            fill = float(np.clip(pct / 100.0, 0.0, 1.0))
            if st.get("ready"):
                fill = max(fill, 0.85)
            return label, fill, accent  # type: ignore[return-value]
        if hasattr(host, "_learn_phase"):
            st = host._learn_phase()
            label = str(st.get("badge", "Learn"))
            accent = tuple(st.get("accent", (160, 210, 255)))
            dw = float(st.get("dw", 0.0) or 0.0)
            ready = bool(st.get("ready", False))
            frozen = bool(st.get("frozen", False))
            if frozen:
                fill = 0.35
            elif ready:
                fill = 0.85
            else:
                fill = float(np.clip(0.15 + 0.7 * dw, 0.0, 1.0))
            return label, fill, accent  # type: ignore[return-value]
        # Fallback if host lacks the new phase helper.
        sock = getattr(host, "motor_socket", None)
        frozen = bool(sock is not None and getattr(sock, "frozen", False))
        dw = float(getattr(host, "motor_socket_dw_ema", getattr(host, "learn_dw_ema", 0.0)) or 0.0)
        fill = float(np.clip(dw, 0.0, 1.0))
        if frozen:
            return "FROZEN · W locked", 0.35, (200, 230, 140)
        return f"Δw {dw:.3f}", fill, (160, 210, 255)
    if key == "port_nn":
        nn = getattr(host, "sensor_port_nn", None)
        on = bool(getattr(host, "sensor_port_on", False))
        if nn is None or not on:
            return "AI Brain off", 0.0, (110, 110, 100)
        try:
            return nn.format_ui_badge(host)
        except Exception:
            return "AI Brain —", 0.0, (120, 120, 110)
    if key == "scope":
        scope = getattr(host, "_training_scope", None)
        on = bool(
            getattr(host, "scope_on", False)
        )
        if on and scope is not None and getattr(scope, "path_neurons", None):
            dw_n = (
                int(host.count_dw_corridor_edges())
                if hasattr(host, "count_dw_corridor_edges")
                else 0
            )
            txt = scope.readout(dw_edges=dw_n)
            return txt, 1.0, (180, 230, 170)
        return "off · full atlas hop", 0.0, (120, 125, 140)
    return key, 0.0, (120, 120, 140)


# ---------------------------------------------------------------------------
# Renderers (pygame)
# ---------------------------------------------------------------------------


@dataclass
class PanelDrawCtx:
    """Geometry + fonts for one legend frame."""

    screen: Any
    host: Any
    content_x: int
    y_off: int
    line_h: int
    panel_width: int
    check_sz: int
    small_font: Any
    hint_font: Any
    draw_checkbox: Any  # (surf, x, y, sz, on, accent) -> Rect
    hits: list  # append (rect, "prow", idx)
    prow_targets: list  # parallel: (row, device)


def render_toggle(ctx: PanelDrawCtx, row: Toggle, device: Any | None) -> None:
    import pygame

    st = row_ui_state(ctx.host, row, device)
    on = st == "active"
    y = ctx.y_off
    cb_y = y + (ctx.line_h - ctx.check_sz) // 2
    if st == "disabled":
        accent = _MUTE
        col = _MUTE_TEXT
    elif st == "active":
        accent = row.accent
        col = tuple(min(255, c + 40) for c in row.accent)
    else:
        accent = _IDLE_BORDER
        col = (160, 160, 175)
    cb = ctx.draw_checkbox(
        ctx.screen, ctx.content_x, cb_y, ctx.check_sz, on, accent
    )
    font = ctx.small_font if row.font == "small" else ctx.hint_font
    lab = font.render(row.label, True, col)
    ctx.screen.blit(
        lab,
        (
            ctx.content_x + ctx.check_sz + 8,
            y + (ctx.line_h - lab.get_height()) // 2,
        ),
    )
    if st == "disabled":
        return
    hit = pygame.Rect(ctx.content_x - 2, y, ctx.panel_width - 16, ctx.line_h)
    idx = len(ctx.prow_targets)
    ctx.prow_targets.append((row, device))
    ctx.hits.append((hit, "prow", idx))
    ctx.hits.append((cb.inflate(4, 4), "prow", idx))


def render_bars(ctx: PanelDrawCtx, row: Bars, device: Any | None) -> None:
    import pygame

    levels = bars_levels(ctx.host, device, row)
    if not levels:
        return
    muted = row_ui_state(ctx.host, row, device) == "disabled"
    x0 = ctx.content_x + 2
    y = ctx.y_off
    tw = int(row.track_w)
    for lab, val, ok in levels:
        track = pygame.Rect(x0 + 18, y + 7, tw, 7)
        pygame.draw.rect(ctx.screen, (28, 28, 32) if muted else row.track_bg, track)
        fw = max(0, int(tw * float(np.clip(val, 0.0, 1.0))))
        if fw and not muted:
            col = row.accent_on if ok else row.accent_off
            pygame.draw.rect(ctx.screen, col, (track.x, track.y, fw, 7))
        pygame.draw.rect(
            ctx.screen,
            _MUTE if muted else row.track_border,
            track,
            1,
        )
        mark = "*" if ok and not muted else ""
        t = ctx.hint_font.render(
            f"{lab}{mark}",
            True,
            _MUTE_TEXT if muted else ((220, 200, 170) if ok else (140, 135, 130)),
        )
        ctx.screen.blit(t, (x0, y + (ctx.line_h - t.get_height()) // 2))
        x0 += tw + 26


def render_meter(ctx: PanelDrawCtx, row: Meter, device: Any | None) -> None:
    import pygame

    st = row_ui_state(ctx.host, row, device)
    label, fill, rgb = meter_state(ctx.host, row, device)
    if st == "disabled":
        rgb = _MUTE_TEXT
        fill = 0.0
    y = ctx.y_off
    t = ctx.hint_font.render(label, True, rgb)
    ctx.screen.blit(
        t, (ctx.content_x + 2, y + (ctx.line_h - t.get_height()) // 2)
    )
    if row.key == "scope":
        return
    if row.key == "learn":
        return
    if row.key == "port_nn":
        return
    tw = max(24, ctx.panel_width - 28 - t.get_width())
    track = pygame.Rect(
        ctx.content_x + 6 + t.get_width(),
        y + (ctx.line_h - 7) // 2,
        tw,
        7,
    )
    pygame.draw.rect(ctx.screen, (40, 35, 32), track)
    fw = max(0, int(tw * float(np.clip(fill, 0.0, 1.0))))
    if fw and st != "disabled":
        pygame.draw.rect(ctx.screen, rgb, (track.x, track.y, fw, 7))
    pygame.draw.rect(ctx.screen, (90, 80, 70) if st != "disabled" else _MUTE, track, 1)


def render_hint(ctx: PanelDrawCtx, row: Hint, device: Any | None = None) -> None:
    y = ctx.y_off
    st = row_ui_state(ctx.host, row, device)
    col = _MUTE_TEXT if st == "disabled" else row.color
    lab = ctx.hint_font.render(row.text, True, col)
    ctx.screen.blit(
        lab, (ctx.content_x + 4, y + (ctx.line_h - lab.get_height()) // 2)
    )


def render_action(ctx: PanelDrawCtx, row: Action, device: Any | None = None) -> None:
    import pygame

    del device
    st = row_ui_state(ctx.host, row, None)
    y = ctx.y_off
    pad_x = 4
    rect = pygame.Rect(
        ctx.content_x + pad_x,
        y + 2,
        max(80, ctx.panel_width - 24),
        ctx.line_h - 4,
    )
    label = action_label(ctx.host, row)
    if st == "disabled":
        fill = (32, 32, 38)
        border = _MUTE
        text_col = _MUTE_TEXT
    elif st == "active":
        fill = tuple(max(0, min(255, c - 40)) for c in row.accent)
        border = row.accent
        text_col = (245, 245, 235)
    else:
        fill = tuple(max(0, c - 100) for c in row.accent)
        border = row.accent
        text_col = row.accent
    pygame.draw.rect(ctx.screen, fill, rect, border_radius=3)
    pygame.draw.rect(ctx.screen, border, rect, 1, border_radius=3)
    lab = ctx.hint_font.render(label, True, text_col)
    ctx.screen.blit(
        lab,
        (
            rect.x + (rect.w - lab.get_width()) // 2,
            rect.y + (rect.h - lab.get_height()) // 2,
        ),
    )
    if st == "disabled":
        return
    idx = len(ctx.prow_targets)
    ctx.prow_targets.append((row, None))
    ctx.hits.append((rect.inflate(2, 2), "prow", idx))


def render_panel_row(
    ctx: PanelDrawCtx, row: PanelRow, device: Any | None = None
) -> None:
    if isinstance(row, Toggle):
        render_toggle(ctx, row, device)
    elif isinstance(row, Bars):
        render_bars(ctx, row, device)
    elif isinstance(row, Meter):
        render_meter(ctx, row, device)
    elif isinstance(row, Hint):
        render_hint(ctx, row, device)
    elif isinstance(row, Action):
        render_action(ctx, row, device)


# ---------------------------------------------------------------------------
# 3D draw dispatch via Device.draw
# ---------------------------------------------------------------------------


def draw_devices(
    host: Any,
    devices: Sequence[Any],
    cx: float,
    cy: float,
    view_lim_y: int,
) -> None:
    """Call ``d.draw(host, …)`` for every enabled device — no DRAW_DISPATCH."""
    sync_enabled_from_host(host)
    for d in devices:
        if not bool(getattr(d, "enabled", True)):
            continue
        draw = getattr(d, "draw", None)
        if callable(draw):
            draw(host, cx, cy, view_lim_y)


def iter_panel_rows(devices: Sequence[Any]) -> list[tuple[Any, PanelRow]]:
    """(device, row) pairs for legend builders / tests."""
    out: list[tuple[Any, PanelRow]] = []
    for d in devices:
        rows = d.panel_rows() if hasattr(d, "panel_rows") else []
        for row in rows:
            out.append((d, row))
    return out


# Back-compat alias used by older imports.
def iter_panel_keys(devices: list[Any]) -> list[tuple[str, str]]:
    from device import panel_entries

    return panel_entries(devices)


def panel_title(device: Any) -> str:
    kind = getattr(device, "kind", getattr(device, "name", "device"))
    custom = getattr(device, "panel_heading", None)
    if callable(custom):
        return str(custom())
    if isinstance(custom, str) and custom:
        return custom
    return str(kind)[:1].upper() + str(kind)[1:]


def panel_color(device: Any) -> tuple[int, int, int]:
    c = getattr(device, "panel_accent", None)
    if c is not None:
        return tuple(int(x) for x in c)  # type: ignore[return-value]
    kind = getattr(device, "kind", "")
    if kind == "turret":
        return (200, 150, 120)
    if kind == "beacon":
        return (180, 140, 220)
    return (140, 170, 200)

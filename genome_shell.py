#!/usr/bin/env python3
"""
Genome shell — MaleCNS whole-CNS cloud + synthesizer-style 3D viewport.

Left 1/3: control panel.
Right 2/3: orthographic particle projection (somata + process mass).
Bottom of viewport: X / Y / Z / Side — camera.
Top bar: Grow / Brain / AI / Map — schematics (CNS Growing / Brain / AI Brain / Research).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pygame

from blueprint import Blueprint, FunnelGate, PlasticityRule
from wells import WellField, WellGrowParams
from ai_brain_schematic import (
    AiBrainSchematic,
    draw_ai_brain,
    hit_ai_brain,
    layout_ai_brain,
    next_stage as ai_brain_next_stage,
)
from brain_schematic import (
    BrainSchematic,
    draw_brain,
    hit_brain,
    layout_brain,
    next_stage as brain_next_stage,
)
from grow_schematic import (
    GrowSchematic,
    draw_grow,
    hit_grow,
    layout_grow,
    next_stage as grow_next_stage,
)
from research_schematic import (
    ResearchSchematic,
    draw_research,
    hit_research,
    layout_research,
    next_stage as research_next_stage,
)
from cns_runtime import CNSRuntime
from device import (
    POLICY_OCELLI,
    POLICY_TURRET_MOTOR,
    POLICY_TURRET_SENSORY,
    POLICY_EYE,
    POLICY_JO,
    CompoundEyeDevice,
    JohnstonOrganDevice,
)
import compound_eyes
import johnston_organ
from device_render import (
    Action,
    Hint,
    PanelDrawCtx,
    blueprint_panel_rows,
    draw_devices,
    handle_action_click,
    handle_row_click,
    learn_panel_rows,
    panel_color,
    panel_title,
    render_hint,
    render_panel_row,
    sync_enabled_from_host,
)
from fly_model import (
    FLY_SECONDS_PER_REVOLUTION,
    GROW_TIP_DRAW_MAX,
    GROW_RAY_DRAW_MAX,
    GROW_SEG_DRAW_MAX,
    STATUS_CONNECTED,
    STATUS_GROWING,
    STATUS_SEEKING,
    Fly,
    load_fly,
    load_fly_from_malecns,
    save_fly,
)

ROOT = Path(__file__).resolve().parent
SOMAS_PATH = ROOT / "data" / "processed" / "malecns_somas.npz"
PROCESSES_PATH = ROOT / "data" / "processed" / "malecns_processes.npz"
UI_STATE_PATH = ROOT / "data" / "processed" / "shell_ui.json"
FLIES_DIR = ROOT / "data" / "flies"

PANEL_COLLAPSED_W = 28
PANEL_OPEN_FRAC = 6  # left panel width = window // this (was 3)

WINDOW_W = 1440
WINDOW_H = 900
BG_PANEL = (28, 28, 34)
BG_VIEW = (12, 12, 16)
# Environment lights (cyborg / ocelli prep): dim sky fill + orbiting point lamp.
SKY_FILL_RGB = (28, 36, 52)  # dull blue-gray zenith wash
SKY_GROUND_RGB = (10, 10, 14)  # slightly darker nadir
LAMP_RGB = (255, 220, 140)
LAMP_GLOW_RGB = (255, 200, 100)
LAMP_ORBIT_FRAC = 1.45  # radius vs body orbit frame
LAMP_ELEV = 0.42  # rad above horizon (world +Y)
LAMP_SPEED_MAX = 1.5  # rev/s — Env Spin slider top end
# Matrix world swing: fly fixed, gravity/lamp/ground/compass tilt about +X.
WORLD_SWING_AMP_MAX = math.radians(40.0)  # peak tilt
WORLD_SWING_HZ_MAX = 0.85  # cycles/s
WORLD_SWING_AMP_DEFAULT = math.radians(18.0)
WORLD_SWING_HZ_DEFAULT = 0.22
GRAV_ARROW_FRAC = 0.58  # shaft length vs body orbit r
GRAV_ARROW_RGB = (220, 70, 70)
GRAV_ARROW_ALPHA = 140
GRAV_ARROW_HEAD_ALPHA = 110
GRAV_BADGE_BG = (40, 18, 18, 160)
GRAV_BADGE_FG = (255, 170, 170)
OCELLI_GROUP_N = 6  # sensory neurons per ocellus (dense device pack)
OCELLI_SENSOR_CHANNELS = 2  # first N of each pack wired to the cup
OCELLI_TIPS_PER_GROUP = 4  # multi-tip Fuse into ascending/descending
OCELLI_CLUSTER_FRAC = 0.018  # tight pack vs body orbit r
# Ocelli FOV: soft lobe past 90° so seams between cups don't go dark.
OCELLI_FACING_BIAS = 0.22  # accept ~cos⁻¹(-bias) off-axis before hard zero
OCELLI_FACING_POW = 0.70  # <1 → flatter peripheral response
# Crown placement (fraction of body orbit r).
OCELLI_CROWN_ANT = 0.10
OCELLI_CROWN_UP = 0.24  # a bit lower, still clear of the crown
OCELLI_LATERAL = 0.07  # compact crown triangle (short fuse runs)
OCELLI_STAR_ELEV = 0.18  # +Y mix into Mercedes look axes
OCELLI_INSET_FRAC = 0.05  # clamp inset toward CNS for tip launch
OCELLI_FUSE_SEEK = 1200.0  # fly-s budget for device↔CNS tips
# Wing-mounted aiming turret: 4 wing-site packs (motor + sensory each).
TURRET_SITES = 4
TURRET_MOTOR_N = 4  # motor neurons per wing site
TURRET_SENSORY_N = 4  # sensory neurons per wing site
TURRET_MOTOR_CHANNELS = 1  # actuators wired to the cup per site
TURRET_SENSOR_CHANNELS = 1  # pose readout channels per site
TURRET_TIPS_PER_SITE = 2  # multi-tip Fuse into neck tracts
TURRET_CLUSTER_FRAC = 0.016
TURRET_MOUNT_UP = 0.30  # neuro/Fuse attach: clears tissue (unchanged for wiring)
TURRET_SITE_UP = 0.06  # lift each wing pack above tissue (× orbit r)
# Draw + funnel only: compact mech (neurons/Fuse keep full wing span + high mount).
TURRET_MECH_XZ = 0.40  # pad footprint scale in XZ (1=full wing span)
TURRET_MECH_MOUNT_UP = 0.05  # visual pedestal height above pads (× orbit r)
# Effector slew (rad/s). Floor covers slow lamp; scales up with lamp ω so the
# barrel can still track at Spin=max (see turret_slew_rates).
TURRET_RATE_YAW = 2.0  # floor yaw slew (rad/s)
TURRET_RATE_PITCH = 1.6  # floor pitch slew (rad/s)
# P-lag ≈ teacher_e0 / margin; keep ≤ aim cone (~15° ≈ 0.26 rad) → margin ≳ 2.2
TURRET_TRACK_MARGIN = 2.4
TURRET_CMD_DEADZONE = 0.04  # hold when network drive is noise-level
TURRET_PITCH_MIN = -0.35
TURRET_PITCH_MAX = 0.85
TURRET_AIM_HALF_DEG = 15.0  # aim cone half-angle (matches green ε band)
TURRET_AIM_CONE_LEN = 0.90  # cone length × orbit r
# Tracking competence (follow lamp, not one-shot aim).
TRACK_MIN_SPIN = 0.02  # rev/s — below: Track is N/A (still / hold-only)
TRACK_WINDOW_REV = 1.0  # score over last N lamp revolutions
TRACK_WINDOW_MIN_S = 3.0
TRACK_WINDOW_MAX_S = 18.0
TRACK_BUF_MAX = 480  # samples (~24 s at 20 Hz)
TRACK_READY = 0.80  # Freeze-ready when Track ≥ this under Spin
TRACK_SETTLE_SLIPS = 2  # max slips in window for SETTLING
# Learn-track stimulus jitter: vary Spin / swing ±20% around slider setpoints.
TRACK_STIM_JITTER = 0.20
TRACK_STIM_JITTER_HZ = 0.07  # ~14 s cycle (covers Track window)
# Learning progress gates.
TRACK_WARN_AFTER_S = 10.0  # live+moving before "not learning" warn
TRACK_WARN_MAX = 0.45  # Track still below this → stuck
TRACK_IMPROVE_MIN = 0.12  # early→now rise that counts as learned
TRACK_DW_SAW = 0.02  # peak Δw that counts as plastic activity
TRACK_HOLD_S = 6.0  # hold Track@ready this long under jitter → trained
# Aim learning defaults — source of truth is PlasticityRule (Blueprint.plasticity).
# Module aliases kept for scripts / diagnostics that import genome_shell.LEARN_*.
_DEFAULT_RULE = PlasticityRule()
LEARN_LR = _DEFAULT_RULE.lr
LEARN_HEBB = _DEFAULT_RULE.hebb
LEARN_TEACHER_MIX = _DEFAULT_RULE.teacher_mix
LEARN_W_MIN = _DEFAULT_RULE.w_min
LEARN_W_MAX = _DEFAULT_RULE.w_max
LEARN_DW_CAP = _DEFAULT_RULE.dw_cap
LEARN_W_DECAY = _DEFAULT_RULE.w_decay
LEARN_DEVICE_GAIN_SCALE = _DEFAULT_RULE.device_gain_scale
LEARN_ELIG_DEPTH = _DEFAULT_RULE.elig_depth
LEARN_PATH_CREDIT = _DEFAULT_RULE.path_credit
LEARN_HOPS = _DEFAULT_RULE.hops
# Fuse prefers classes from Device AttachPolicy (see device.py).
FUSE_NECK_CLASSES = POLICY_OCELLI.prefer
FUSE_NECK_FALLBACK = POLICY_OCELLI.fallback
FUSE_EYE_CLASSES = POLICY_EYE.prefer
FUSE_EYE_FALLBACK = POLICY_EYE.fallback
FUSE_JO_CLASSES = POLICY_JO.prefer
FUSE_JO_FALLBACK = POLICY_JO.fallback
FUSE_MOTOR_CLASSES = POLICY_TURRET_MOTOR.prefer
FUSE_SENSORY_CLASSES = POLICY_TURRET_SENSORY.prefer
GROUND_GRID = 48  # quads along each axis (larger disk needs a bit more mesh)
GROUND_EXTENT_FRAC = 7.5  # far horizon disk vs compass orbit radius
GROUND_GAP_FRAC = 0.10  # just below compass ring bottom (× orbit r)
GROUND_RGB = (48, 52, 44)  # dark but readable against sky void
GROUND_SHADOW_MUL = 0.42  # clear umbra without disappearing
EDGE = (48, 48, 56)
FPS = 60
MOUSE_ROTATION_SENSITIVITY = 0.01  # cosmos.config
MOUSE_WHEEL_SENSITIVITY = 0.15
# Default Side preset (nav button). Mouse-orbit while on Side is remembered.
SIDE_VIEW_ANGLE_X = 0.3
SIDE_VIEW_ANGLE_Y = 0.5
SIDE_VIEW_MATCH = 0.1
BUTTON_SIZE = 22
BUTTON_SPACING = 5
NAV_HINT_H = 22
NAV_PANEL_PAD = 10
# Projection no longer hosts chrome — keep 0 so view_lim_y == view_h.
NAV_PANEL_H = 0
# Reserved chrome (outside projection).
TOP_BAR_H = 36
TOOL_W = 40
STATUS_H = 28
DOCK_W = 312  # right panels column when expanded (push, not float)
CHROME_BG = (22, 24, 30)
CHROME_EDGE = (48, 52, 62)
TOOL_SLOT = 28

# Orbital compass / gyro around the body (world-fixed, outside the cloud).
COMPASS_RING_FRAC = 1.14  # radius vs farthest soma from center
COMPASS_SEGS = 96
COMPASS_TICK_MAJOR = 30  # degrees
COMPASS_TICK_MINOR = 10
# Distinct ring colors (RGBA) — horizon / sagittal / coronal (soft, not loud).
COMPASS_COL_HZ = (90, 180, 255, 70)  # cyan — azimuth N/E/S/W
COMPASS_COL_MER = (255, 170, 90, 60)  # amber — elevation UP/DN
COMPASS_COL_COR = (220, 120, 200, 55)  # magenta — roll L/R
COMPASS_COL_HZ_RGB = (110, 170, 210)
COMPASS_COL_MER_RGB = (200, 160, 110)
COMPASS_COL_COR_RGB = (190, 130, 175)
COMPASS_TICK_ALPHA = 90
COMPASS_LABEL_ALPHA = 140

MODAL_WIDTH = 320
MODAL_HEIGHT = 120
BTN_W = 80
BTN_H = 32

CLASS_COLORS = np.array(
    [
        (90, 170, 255),   # optic
        (200, 200, 210),  # central
        (80, 200, 180),   # vnc
        (120, 220, 140),  # sensory
        (100, 200, 230),  # visual_projection
        (160, 140, 255),  # visual_centrifugal
        (255, 160, 100),  # descending
        (255, 200, 120),  # ascending
        (255, 120, 140),  # motor
        (220, 180, 220),  # endocrine
    ],
    dtype=np.uint8,
)

# Soft focus contrast — selected pops; rest stays readable, just quieter.
FOCUS_DIM = 0.55
FOCUS_BOOST = 24
# Draw density caps — full CNS is already a sample of the connectome.
DRAW_SOMA_MAX = 10000
DRAW_FOCUS_SOMA_MAX = 6000  # highlight keeps points sparse — never denser fill
DRAW_PROC_MAX = 10000  # atlas process mass = CNS body silhouette
DRAW_GROWN_PTS_MAX = 11000  # ambient tissue dust (full CNS)
DRAW_GROWN_FOCUS_MAX = 32000  # dense trail stamp when ≤3 axon classes focused
DRAW_GRAPH_EDGES_MAX = 600
DRAW_TRACT_SAMPLE_MAX = 1200  # stratified recipe chords (focus / scope)
DRAW_SEG_MAX = 2000  # axon/dendrite segments after filter (full CNS)
DRAW_SEG_FOCUS_MAX = 6500  # more polyline segs for focused tracts
DRAW_FOCUS_GLOW_MAX = 100  # soft blink markers when a class is focused
DRAW_FOCUS_RAY_MAX = 1600  # axons kept when a class is focused
DRAW_FOCUS_POLY_POINTS = 12  # samples along each focused axon trail
FOCUS_LINE_WIDTH = 2  # thicker chords so tracts read as clumps
BODY_PROC_BRIGHT = 0.55  # atlas ghost under Super-class G (empty G = hide)
# Layer visibility is derived from Super-class focus (N/G/A/D/B) — no separate
# Layers toggles. show_layer_* attrs remain as mirrors for the draw path.
BASIN_TOP_N = 8  # largest classes to scan for agglomerations
BASIN_MIN_PTS = 500  # ignore tiny sets in the default overview
BASIN_MAX_MODES = 8  # elongated clouds may need many axial wells
BASIN_RING_STEPS = 4  # level curves (wireframe; was 10 + wall quads)
BASIN_RING_SEGS = 16  # points per ring (was 28)
BASIN_SPOKES = 6  # meridian lines (was 12)
BASIN_VIS_SCALE = 1.85  # enlarge wells vs raw cloud radius
BASIN_VIS_MIN_PX = 55  # never smaller than this on screen
BASIN_ALPHA = 0.70  # 30% more transparent than full well paint
# Device wells: seat in tissue at devices, grow until ∩ = neck with paths.
# Selection = ball around well centre (same r as drawn funnel rim).
DEVICE_FUNNEL_LOCAL_K = 140  # nearest somata to seat / size each well
DEVICE_FUNNEL_LOCAL_Q = 0.55  # seed radius percentile before grow
DEVICE_FUNNEL_R_MIN_FRAC = 0.04  # vs body span
DEVICE_FUNNEL_R_MAX_FRAC = 0.42  # vs body span (allow grow toward partner)
DEVICE_FUNNEL_R_MAX_GAP = 0.72  # also ≤ this × device gap
DEVICE_FUNNEL_GROW_STEPS = 28  # grow iterations toward partner
DEVICE_FUNNEL_GROW_FRAC = 0.035  # +frac×gap per step (each well)
DEVICE_FUNNEL_GRAVITY = 0.42  # tip chemotaxis blend inside the wells
DEVICE_FUNNEL_ALPHA = 0.55
DEVICE_FUNNEL_DEPTH = 1.05  # drawn cone depth × r
DEVICE_FUNNEL_SLACK = 1.06  # unused for balls; kept for cone helpers
# Well grow defaults (geometry) — not TrainingScope / path gate.
_DEFAULT_WELL_GROW = WellGrowParams()
_DEFAULT_GATE = FunnelGate()  # blueprint JSON compat only
CHANNEL_CARVE_LR = _DEFAULT_RULE.carve_lr
CHANNEL_OFFPATH_DECAY = _DEFAULT_RULE.offpath_decay
CHANNEL_SHOW_DW = _DEFAULT_RULE.channel_show_dw
CHANNEL_MAX_EDGES = _DEFAULT_RULE.channel_max_edges
CHANNEL_ACT_MIN = _DEFAULT_RULE.channel_act_min

# Right side panels (independent; was one MaleCNS tabbed legend).
SIDE_PANELS: tuple[str, ...] = ("filters", "env", "devices", "learn", "view")
SIDE_PANEL_LABELS: dict[str, str] = {
    "filters": "Filters",
    "env": "Env",
    "devices": "I/O",
    "learn": "Learn",
    "view": "View",
}
# View Detail → FPS tradeoff (1 = full quality, 0 = cheapest).
VIEW_DETAIL_SIM_HZ_LO = 8.0
VIEW_DETAIL_DRAW_LO = 0.22
VIEW_DETAIL_LEARN_EVERY_HI = 24  # at Detail=0, learn less often
# Legacy aliases (UI state / old saves).
LEGEND_TABS = SIDE_PANELS
LEGEND_TAB_LABELS = SIDE_PANEL_LABELS
_LEGACY_TAB_MAP = {"atlas": "filters", "MaleCNS": "filters"}

# Superclass blurbs — MaleCNS-aligned
CLASS_HELP = {
    "optic": (
        "Назначение: первичная и локальная обработка зрения. "
        "Устройство: intrinsic к optic lobes — сома и связи внутри долей."
    ),
    "central": (
        "Назначение: интеграция, память, решение в центральном мозге. "
        "Устройство: intrinsic к central brain (не optic / не чистый вход-выход)."
    ),
    "vnc": (
        "Назначение: локальные контуры ног, крыльев, позы в брюшной нервной цепи. "
        "Устройство: intrinsic к VNC (analogue of spinal cord) — ядро туловищной ЦНС."
    ),
    "sensory": (
        "Назначение: афферентный вход с периферии в ЦНС (голова и тело). "
        "Устройство: сенсорные аксоны по нервам в brain и/или VNC."
    ),
    "visual_projection": (
        "Назначение: зрительный выход optic lobes → central brain (VPN)."
    ),
    "visual_centrifugal": (
        "Назначение: обратная связь central brain → optic lobes."
    ),
    "descending": (
        "Назначение: команды мозг → VNC/тело через шею (DN). "
        "Устройство: сома в мозге, аксон в cervical connective вниз."
    ),
    "ascending": (
        "Назначение: сигналы VNC/тело → мозг (AN); пара к descending."
    ),
    "motor": (
        "Назначение: прямой моторный выход ЦНС на мышцы (голова и тело/VNC)."
    ),
    "endocrine": (
        "Назначение: нейроэндокринный выход (corpora / visceral)."
    ),
}

DEVICE_HELP = {
    "lamp": (
        "Лампа — стимул среды. Крутится над тканью; ocelli читают яркость, "
        "турьель целится. Выкл. — небо/земля без точечного источника."
    ),
    "ocelli": (
        "Ocelli — три дорсальных датчика света (входной socket). "
        "Чашки Medial / Left / Right → sensory; Fuse тянет tips к ЦНС."
    ),
    "eye_left": (
        "Eye L — полусфера вокруг левого optic lobe: контур + 4 угла/U/D. "
        "Фотодатчики на оболочке → Fuse в optic / visual_projection."
    ),
    "eye_right": (
        "Eye R — зеркало Eye L на правом optic lobe. "
        "Вместе с L дают bearing лампы; ocelli — небо/стабилизация."
    ),
    "jo_left": (
        "JO L — орган Джонстона на левой антенне. "
        "Виртуальный tilt от Matrix world-swing → sensory Fuse."
    ),
    "jo_right": (
        "JO R — парный орган Джонстона на правой антенне. "
        "Тот же tilt-сокет; сторону можно отключать отдельно."
    ),
    "turret": (
        "Turret — моторный socket (Llo/Lhi/Rlo/Rhi) + сенсорный feedback. "
        "Aim → Learn / Motor socket; Freeze фиксирует W. "
        "Скорость привода растёт с Spin лампы, чтобы успевать на любом Spin."
    ),
}


def turret_slew_rates(lamp_speed: float) -> tuple[float, float]:
    """Yaw/pitch slew caps (rad/s) that can still track the orbiting lamp.

    Floor rates cover a slow/stopped lamp; above that yaw scales with lamp ω
    so full ±1 motor command stays ahead of Spin=max (P-lag ≲ aim cone).
    """
    omega = abs(float(lamp_speed)) * math.tau
    rate_yaw = max(float(TURRET_RATE_YAW), omega * float(TURRET_TRACK_MARGIN))
    pitch_ratio = float(TURRET_RATE_PITCH) / max(float(TURRET_RATE_YAW), 1e-6)
    rate_pitch = max(float(TURRET_RATE_PITCH), rate_yaw * pitch_ratio)
    return float(rate_yaw), float(rate_pitch)


class ExitConfirmModal:
    """«Выйти из приложения?» — same behavior as Temporal Sound Synthesizer."""

    def __init__(self) -> None:
        self._visible = False

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def is_visible(self) -> bool:
        return self._visible

    def handle_click(
        self,
        mouse_pos: tuple[int, int],
        screen_width: int,
        screen_height: int,
    ) -> Optional[Literal["yes", "no", "dismiss"]]:
        if not self._visible:
            return None
        exit_modal_x = (screen_width - MODAL_WIDTH) // 2
        exit_modal_y = (screen_height - MODAL_HEIGHT) // 2
        btn_y = exit_modal_y + MODAL_HEIGHT - BTN_H - 20
        yes_x = exit_modal_x + (MODAL_WIDTH - BTN_W * 2 - 20) // 2
        no_x = yes_x + BTN_W + 20
        mx, my = mouse_pos
        if yes_x <= mx <= yes_x + BTN_W and btn_y <= my <= btn_y + BTN_H:
            return "yes"
        if no_x <= mx <= no_x + BTN_W and btn_y <= my <= btn_y + BTN_H:
            return "no"
        if not (
            exit_modal_x <= mx <= exit_modal_x + MODAL_WIDTH
            and exit_modal_y <= my <= exit_modal_y + MODAL_HEIGHT
        ):
            return "dismiss"
        return None

    def render(
        self,
        screen: pygame.Surface,
        width: int,
        height: int,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        if not self._visible:
            return
        modal_overlay = pygame.Surface((width, height))
        modal_overlay.set_alpha(200)
        modal_overlay.fill((0, 0, 0))
        screen.blit(modal_overlay, (0, 0))
        exit_modal_x = (width - MODAL_WIDTH) // 2
        exit_modal_y = (height - MODAL_HEIGHT) // 2
        pygame.draw.rect(
            screen,
            (40, 40, 50),
            (exit_modal_x, exit_modal_y, MODAL_WIDTH, MODAL_HEIGHT),
        )
        pygame.draw.rect(
            screen,
            (80, 80, 100),
            (exit_modal_x, exit_modal_y, MODAL_WIDTH, MODAL_HEIGHT),
            2,
        )
        title_text = font.render("Выйти из приложения?", True, (255, 255, 255))
        title_x = exit_modal_x + (MODAL_WIDTH - title_text.get_width()) // 2
        screen.blit(title_text, (title_x, exit_modal_y + 18))
        btn_y = exit_modal_y + MODAL_HEIGHT - BTN_H - 20
        yes_x = exit_modal_x + (MODAL_WIDTH - BTN_W * 2 - 20) // 2
        no_x = yes_x + BTN_W + 20
        pygame.draw.rect(screen, (180, 80, 80), (yes_x, btn_y, BTN_W, BTN_H))
        pygame.draw.rect(screen, (80, 120, 80), (no_x, btn_y, BTN_W, BTN_H))
        yes_label = small_font.render("Да", True, (255, 255, 255))
        no_label = small_font.render("Нет", True, (255, 255, 255))
        screen.blit(
            yes_label,
            (
                yes_x + (BTN_W - yes_label.get_width()) // 2,
                btn_y + (BTN_H - yes_label.get_height()) // 2,
            ),
        )
        screen.blit(
            no_label,
            (
                no_x + (BTN_W - no_label.get_width()) // 2,
                btn_y + (BTN_H - no_label.get_height()) // 2,
            ),
        )


def project_3d_batch(
    xyz: np.ndarray,
    view_angle_x: float,
    view_angle_y: float,
    center_x: float,
    center_y: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized orthographic projection (same math as cosmos.visualizer.project_3d)."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    cos_y = math.cos(view_angle_y)
    sin_y = math.sin(view_angle_y)
    x_rot = x * cos_y - z * sin_y
    z_rot = x * sin_y + z * cos_y
    cos_x = math.cos(view_angle_x)
    sin_x = math.sin(view_angle_x)
    y_rot = y * cos_x - z_rot * sin_x
    screen_x = center_x + x_rot * scale
    screen_y = center_y - y_rot * scale
    return screen_x, screen_y


# Human-readable labels for legend
CLASS_LABELS = {
    "optic": "Optic",
    "central": "Central",
    "vnc": "VNC",
    "sensory": "Sensory",
    "visual_projection": "Vis. projection",
    "visual_centrifugal": "Vis. centrifugal",
    "descending": "Descending",
    "ascending": "Ascending",
    "motor": "Motor",
    "endocrine": "Endocrine",
}


def _rgb_for_cls(cls: np.ndarray) -> np.ndarray:
    if len(cls) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return CLASS_COLORS[np.clip(cls, 0, len(CLASS_COLORS) - 1)]


def _axon_rgb_for_cls(cls: np.ndarray) -> np.ndarray:
    """Dim class tint for grown axons — secondary to somata."""
    base = _rgb_for_cls(cls).astype(np.float32)
    if len(base) == 0:
        return base.astype(np.uint8)
    return np.clip(base * 0.42, 0, 255).astype(np.uint8)


def _dendrite_rgb_for_cls(cls: np.ndarray) -> np.ndarray:
    """Dimmer class tint for dendrite stubs — quietest neurite layer."""
    base = _rgb_for_cls(cls).astype(np.float32)
    if len(base) == 0:
        return base.astype(np.uint8)
    return np.clip(base * 0.32, 0, 255).astype(np.uint8)


def _density_take(n: int, max_n: int) -> np.ndarray | slice:
    """Even stride subsample when a layer exceeds the draw budget."""
    if n <= max_n or max_n <= 0:
        return slice(None)
    step = max(1, (n + max_n - 1) // max_n)
    return np.arange(0, n, step, dtype=np.int32)[:max_n]


def _stratified_density_take(
    cls: np.ndarray,
    max_n: int,
    *,
    keep_small_below: int = 400,
    boost_classes: set[int] | None = None,
    boost_frac: float = 0.28,
    boost_budget_frac: float = 0.40,
) -> np.ndarray | slice:
    """Subsample dense classes; keep tiny classes intact.

    Global stride treats 60 sensory like 81k optic — rare types nearly vanish.
    Budget: small-in-cloud full → optional boost (sparse-soma / process-heavy
    classes keep a larger fraction) → equal share among the rest.
    """
    n = int(len(cls))
    if n <= max_n or max_n <= 0:
        return slice(None)
    boost = {int(c) for c in (boost_classes or ())}
    classes = np.unique(cls)
    small: list[np.ndarray] = []
    boosted: list[tuple[int, np.ndarray]] = []
    large: list[tuple[int, np.ndarray]] = []
    for c in classes.tolist():
        c = int(c)
        sub = np.flatnonzero(cls == c)
        if len(sub) <= int(keep_small_below):
            small.append(sub)
        elif c in boost:
            boosted.append((c, sub))
        else:
            large.append((c, sub))
    kept_small = (
        np.concatenate(small) if small else np.zeros((0,), dtype=np.int32)
    )
    remain = int(max_n) - int(len(kept_small))
    if remain <= 0:
        return _density_take(n, max_n)
    parts: list[np.ndarray] = [kept_small]
    used = 0

    def _take_share(sub: np.ndarray, share: int) -> np.ndarray:
        share = int(max(0, share))
        if share <= 0:
            return np.zeros((0,), dtype=np.int32)
        if len(sub) <= share:
            return sub
        step = max(1, (len(sub) + share - 1) // share)
        return sub[::step][:share]

    # Boost: process-heavy classes with few atlas somata (e.g. sensory).
    if boosted and remain > 0:
        boost_pool = max(1, int(remain * float(np.clip(boost_budget_frac, 0.05, 0.8))))
        wants: list[tuple[np.ndarray, int]] = []
        total_want = 0
        for _c, sub in boosted:
            want = min(
                len(sub),
                max(int(keep_small_below), int(len(sub) * float(boost_frac))),
            )
            wants.append((sub, want))
            total_want += want
        if total_want > boost_pool and total_want > 0:
            scale = boost_pool / total_want
            wants = [(sub, max(1, int(round(w * scale)))) for sub, w in wants]
        for sub, want in wants:
            share = min(want, remain - used)
            take = _take_share(sub, share)
            if len(take):
                parts.append(take)
                used += len(take)

    remain2 = remain - used
    if large and remain2 > 0:
        per = max(1, remain2 // len(large))
        for _c, sub in large:
            share = min(len(sub), per, max(0, remain - used))
            take = _take_share(sub, share)
            if len(take):
                parts.append(take)
                used += len(take)
    out = np.sort(np.concatenate(parts))
    return out[:max_n]


def _manual_density_take(
    cls: np.ndarray,
    max_n: int,
    dens: dict[int, float],
) -> np.ndarray | slice:
    """Per-class keep fraction (0..1), then hard-cap to max_n."""
    n = int(len(cls))
    if n == 0:
        return slice(None)
    parts: list[np.ndarray] = []
    for c in np.unique(cls).tolist():
        sub = np.flatnonzero(cls == int(c))
        frac = float(np.clip(dens.get(int(c), 1.0), 0.0, 1.0))
        want = int(round(len(sub) * frac))
        if want <= 0:
            continue
        if want >= len(sub):
            parts.append(sub)
        else:
            step = max(1, (len(sub) + want - 1) // want)
            parts.append(sub[::step][:want])
    if not parts:
        return np.zeros((0,), dtype=np.int32)
    out = np.sort(np.concatenate(parts))
    if len(out) <= max_n or max_n <= 0:
        return out
    # Proportionally trim while keeping at least one point per kept class if possible.
    step = max(1, (len(out) + max_n - 1) // max_n)
    return out[::step][:max_n]


class GenomeShell(CNSRuntime):
    """Pygame view/controller — the public API over the CNS plant base."""

    def _boot_headless_extras(self) -> None:
        """Attrs that interactive ``__init__`` would set; safe stubs for open()."""
        self.screen = None
        self._view_surf = None
        self.running = False
        self.clock = None
        self.font = None
        self.small_font = None
        self.nav_font = None
        self.hint_font = None
        self._maximized = False
        self._fps_smooth = 0.0
        self._perf_ms = {}
        self._perf_ema = {}
        self.use_gl_tissue = False
        self._gl_tissue = None
        self.legend_mini = True
        self.panel_collapsed = True
        self.legend_tab = "filters"
        self.show_ocelli_funnel = False
        self.show_turret_funnel = False
        self.show_turret_aim = True
        self.show_port_schema = False
        self.show_brain_schema = False
        self.show_grow_schema = False
        self.show_research_schema = False
        self._port_selected = None
        self._learn_panel_open = False
        self._fps_panel_open = False

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("MaleCNS Genome — whole CNS shell")
        self._maximized = True
        info = pygame.display.Info()
        boot_w = int(info.current_w) if info.current_w > 0 else WINDOW_W
        boot_h = int(info.current_h) if info.current_h > 0 else WINDOW_H
        self.screen = pygame.display.set_mode(
            (boot_w, boot_h), self._display_flags()
        )
        boot_w, boot_h = self.screen.get_size()
        self.clock = pygame.time.Clock()
        self._fps_smooth: float = float(FPS)
        self._fps_panel_open = False
        self._fps_badge_rect: pygame.Rect | None = None
        self._fps_panel_rect: pygame.Rect | None = None
        self._fps_corner_h = 0  # legend top offset (prev-frame badge/panel)
        self._learn_panel_open = False
        self._learn_badge_rect: pygame.Rect | None = None
        self._learn_panel_rect: pygame.Rect | None = None
        self._learn_topo_drawn = 0
        self._learn_topo_nodes = 0
        self._perf_ms: dict[str, float] = {}
        self._perf_ema: dict[str, float] = {}
        self._tissue_cache_hit = False
        self.running = True
        self.exit_confirm_modal = ExitConfirmModal()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.nav_font = pygame.font.Font(None, 20)
        self.hint_font = pygame.font.Font(None, 18)

        # Headless core state (devices / learn / funnels) — no display needed.
        CNSRuntime._init_runtime_state(self, Fly(), Blueprint.default())
        # Interactive cadence: 1 hop/frame, learn every N (headless keeps rule.hops).
        self.loop_hops = int(getattr(self.rule, "hops_ui", 1) or 1)
        self.learn_every = int(getattr(self.rule, "learn_every", 8) or 8)
        self._devices_dirty = True
        # GUI defaults: funnels off until user enables (Blueprint headless keeps on).
        self.show_ocelli_funnel = False
        self.show_turret_funnel = False
        self.show_turret_aim = True
        self.scope_on = False
        self.hop_clip = True
        self.blueprint.scope_on = False
        self.blueprint.hop_clip = True

        # GPU tissue is opt-in only (FLYWIRE_GL=1). Auto-creating a moderngl
        # context previously coincided with a full-black pygame window (menu
        # included) on this machine — never construct GL unless opted in.
        # FLYWIRE_NO_GL also forces off when both are set.
        self._gl_tissue = None
        self._gl_geom_key: tuple | None = None
        self._gl_soma_idx = np.zeros((0,), dtype=np.int32)
        self.use_gl_tissue = False
        self.adult_sim_hz: float = 20.0
        self._sim_accum: float = 0.0
        self._last_frame_wall: float = 0.0
        _gl_opt_in = os.environ.get("FLYWIRE_GL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        _gl_opt_out = os.environ.get("FLYWIRE_NO_GL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if _gl_opt_in and not _gl_opt_out:
            try:
                from gl_tissue import try_create_renderer

                self._gl_tissue = try_create_renderer()
                self.use_gl_tissue = bool(
                    self._gl_tissue is not None and self._gl_tissue.ok
                )
            except Exception:
                self._gl_tissue = None
                self.use_gl_tissue = False
        if self.use_gl_tissue:
            self.adult_sim_hz = 30.0

        # Start empty — MaleCNS is imported manually (File → Import)
        self._sync_from_fly()

        self.legend_mini = True  # dock collapsed by default — clear projection
        self.panel_collapsed = False
        self.legend_tab = "filters"  # legacy; panels are independent
        self.sim_running = False  # fly clock advances only while running
        self._activity_running = False  # adult Run: closed device loop (or bare wave)
        self._fuse_running = False  # grow only device↔fly tips (no atlas re-grow)
        self.panel_toggle_rect: pygame.Rect | None = None
        self.legend_toggle_rect: pygame.Rect | None = None
        self.legend_panel_rect: pygame.Rect | None = None
        self.legend_scroll = 0  # legacy alias → side_panel_scroll["filters"]
        self.side_panel_mini: dict[str, bool] = {p: False for p in SIDE_PANELS}
        self.side_panel_scroll: dict[str, int] = {p: 0 for p in SIDE_PANELS}
        self._side_panel_rects: dict[str, pygame.Rect] = {}
        self._side_panel_toggle_rects: dict[str, pygame.Rect] = {}
        self.legend_help_rect: pygame.Rect | None = None  # unused; kept for cache slots
        # Independent filters: somata / atlas graph / grown axons / grown dendrites
        self.focus_soma_indices: set[int] = set()
        self.focus_graph_indices: set[int] = set()  # atlas recipe edges by class
        self.focus_axon_indices: set[int] = set()
        self.focus_dendrite_indices: set[int] = set()
        self.focus_tract_indices: set[int] = set()  # recipe soma↔soma chords (T)
        self.focus_basin_indices: set[int] = set()  # attractor wells by class
        # Legacy alias used by older UI state (process ghost → graph layer).
        self.focus_proc_indices: set[int] = set()
        self.focus_hint_class: int | None = None  # last focused parent (stimulus)
        # Hover tip in 3D: ("class", idx) | ("device", key) | None
        self._hover_tip: tuple[str, int | str] | None = None
        self._hover_tip_pos: tuple[int, int] = (0, 0)
        self._class_cent_xyz: np.ndarray | None = None
        self._class_cent_key: tuple | None = None
        self._legend_hits: list[
            tuple[pygame.Rect, str, int]
        ] = []  # (rect, layer, idx)
        self._t0 = time.perf_counter()

        # Layer mirrors — source of truth is Super-class focus (N/G/A/D/B).
        # Non-empty focus_* → layer on; empty → off. N/A/D seeded at migrate.
        self.show_layer_neurons = True
        self.show_layer_graph = False  # dense atlas edges — opt-in via G
        self.show_layer_axons = True
        self.show_layer_dendrites = True
        self.show_layer_basins = False  # wells — opt-in via B
        self.show_port_schema = False  # AI Brain iterative block diagram
        self._port_selected: str | None = None
        self._port_hits: list[tuple[pygame.Rect, str]] = []
        self._port_stage: str = "sense"
        self.show_brain_schema = False  # CNS Brain hop/carve block diagram
        self._brain_selected: str | None = None
        self._brain_hits: list[tuple[pygame.Rect, str]] = []
        self._brain_stage: str = "sense"
        self.show_grow_schema = False  # CNS Growing tip/synapse policy diagram
        self._grow_selected: str | None = None
        self._grow_hits: list[tuple[pygame.Rect, str]] = []
        self._grow_stage: str = "recipe"
        self.show_research_schema = False  # Research map / project flowchart
        self._research_selected: str | None = None
        self._research_hits: list[tuple[pygame.Rect, str]] = []
        self._research_stage: str = "now"
        self._basin_cls: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._basin_xyz: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._basin_r: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._basin_n: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._basin_cache_key: tuple | None = None
        # Basin formation: 0 = few large funnels, 1 = many small wells.
        self.basin_form: float = 0.55
        self._basin_form_rect: pygame.Rect | None = None
        self._basin_form_drag: bool = False
        # Environment: dim sky fill + orbiting point lamp.
        self.light_enabled: bool = True
        self.lamp_on: bool = True
        self.sky_fill: float = 0.28  # 0..1 ambient sky strength
        self.lamp_intensity: float = 1.0
        self.lamp_speed: float = 0.11  # revolutions per second (was 0.22)
        self.lamp_angle: float = 0.0  # rad, azimuth about world +Y
        self._lamp_xyz: np.ndarray = np.zeros(3, dtype=np.float64)
        self._light_slider_rects: dict[str, pygame.Rect] = {}
        self.view_detail: float = 1.0  # 0=fast · 1=full (Hop/Activity + overlays)
        self._view_detail_rect: pygame.Rect | None = None
        self._view_detail_drag: bool = False
        self._light_drag: str | None = None  # "sky" | "speed" | "swing_amp" | "swing_hz"
        self._sky_grad_cache: tuple | None = None  # (w, h, sky_q, surf)
        self.ground_on: bool = True
        self.ground_relief: float = 0.55  # 0..1 bump amplitude
        self._ground_cache_key: tuple | None = None
        self._ground_x: np.ndarray | None = None  # (N,) world X samples
        self._ground_z: np.ndarray | None = None
        self._ground_y: np.ndarray | None = None  # (N,N) heights
        self._ground_center: np.ndarray | None = None
        self._ground_extent: float = 1.0
        self._ground_relief_rect: pygame.Rect | None = None
        self._ground_relief_drag: bool = False
        # Matrix: swing gravity / world frame about body +X (fly tissue stays put).
        self.world_swing_on: bool = False
        self.world_swing_amp: float = float(WORLD_SWING_AMP_DEFAULT)
        self.world_swing_hz: float = float(WORLD_SWING_HZ_DEFAULT)
        self.world_swing_phase: float = 0.0  # rad of sin oscillator
        # Ground draw cache: projected verts + last full blit when cam/lamp stable.
        self._ground_draw_cam_key: tuple | None = None
        self._ground_draw_cache: dict | None = None
        self._ground_blit_key: tuple | None = None
        self._ground_blit_surf: pygame.Surface | None = None
        # Basins layer blit cache (same camera → skip redraw).
        self._basin_blit_key: tuple | None = None
        self._basin_blit_surf: pygame.Surface | None = None
        self._overlay_buf: pygame.Surface | None = None  # reusable SRCALPHA scratch
        # Pose / orbit caches — avoid rescanning 1e5 somata every draw call.
        self._orbit_cache_key: tuple | None = None
        self._orbit_cache: tuple[np.ndarray, float] | None = None
        self._wing_anchor_key: tuple | None = None
        self._wing_anchor_cache: np.ndarray | None = None
        self._frame_ocelli_pose: list[dict] | None = None
        self._frame_eye_pose: dict = {}
        self._frame_turret_pose: dict | None = None
        self._frame_pose_stamp: tuple | None = None
        # soma_id → row index (invalidate when soma_ids array replaced).
        self._soma_lut: dict[int, int] | None = None
        self._soma_lut_key: tuple | None = None
        # Device funnel layer blit cache (camera + device pose stable).
        self._device_funnel_blit_key: tuple | None = None
        self._device_funnel_blit_surf: pygame.Surface | None = None
        # Static tissue (sky+ground+stamps without activity) — camera-keyed.
        self._tissue_blit_key: tuple | None = None
        self._tissue_blit_surf: pygame.Surface | None = None
        # Legend / compass blit caches (UI/camera stable → skip rebuild).
        self._legend_blit_key: tuple | None = None
        self._legend_blit_surf: pygame.Surface | None = None
        self._legend_blit_pos: tuple[int, int] = (0, 0)
        self._legend_hits_cached: list = []
        self._legend_panel_rect_cached: pygame.Rect | None = None
        self._legend_toggle_rect_cached: pygame.Rect | None = None
        self._legend_help_rect_cached: pygame.Rect | None = None
        self._side_panel_rects_cached: dict = {}
        self._side_panel_toggle_rects_cached: dict = {}
        self._compass_blit_key: tuple | None = None
        self._compass_blit_surf: pygame.Surface | None = None
        # Ocelli fields also set before first sync; keep defaults here if re-entered.
        if not hasattr(self, "ocelli_on"):
            self.ocelli_on = True
            self.ocelli_groups = [[], [], []]
            self.ocelli_terminals = []
            self.ocelli_ids = []
            self.ocelli_I = np.zeros(3, dtype=np.float64)
        if not hasattr(self, "turret_on"):
            self.turret_on = True
            self.turret_motor_groups = [[], [], [], []]
            self.turret_sensory_groups = [[], [], [], []]
            self.turret_motor_terminals = []
            self.turret_sensory_terminals = []
            self.turret_bus_terminal = None
            self.turret_yaw = 0.0
            self.turret_pitch = 0.40
            self.turret_S = np.zeros(2, dtype=np.float64)
            self.turret_M = np.zeros(2, dtype=np.float64)
            self.turret_aim_err = float("nan")
        if not hasattr(self, "eye_left_on"):
            self.eye_left_on = False
            self.eye_right_on = False
            self.eye_left_I = np.zeros(6, dtype=np.float64)
            self.eye_right_I = np.zeros(6, dtype=np.float64)
            self.show_eye_funnel = False
            self._frame_eye_pose = {}
        if not hasattr(self, "jo_left_on"):
            self.jo_left_on = False
            self.jo_right_on = False
            self.jo_left_tilt = 0.0
            self.jo_right_tilt = 0.0
            self.show_jo_funnel = False
            self._frame_jo_pose = {}
        if not hasattr(self, "learn_on"):
            self.learn_on = True
            self._learn_prev_err = float("nan")
            self.learn_dw_ema = 0.0
            self.learn_n_edges = 0
        else:
            self.learn_n_edges = int(
                len(getattr(self.fly, "contact_gain", {}) or {})
            )
        if not hasattr(self, "motor_socket_on"):
            rule = getattr(self, "rule", None) or _DEFAULT_RULE
            self.motor_socket_on = bool(getattr(rule, "motor_socket_on", False))
        if not hasattr(self, "motor_socket"):
            self.motor_socket = None
        if not hasattr(self, "motor_socket_dw_ema"):
            self.motor_socket_dw_ema = 0.0
        if not hasattr(self, "sensor_port_nn"):
            from sensor_port_nn import SensorPortNN

            self.sensor_port_nn = SensorPortNN(lr=0.02)
        if not hasattr(self, "sensor_port_on"):
            self.sensor_port_on = False
        if not hasattr(self, "sensor_port_train"):
            self.sensor_port_train = False
        if not hasattr(self, "_learn_err_start_deg"):
            self._learn_err_start_deg = float("nan")
        if not hasattr(self, "_learn_err_freeze_deg"):
            self._learn_err_freeze_deg = float("nan")
        if not hasattr(self, "_track_samples"):
            self._track_reset()
        if not hasattr(self, "_stim_jitter_phase"):
            self._stim_jitter_phase = 0.0
        if not hasattr(self, "_learn_live_s"):
            self._learn_progress_reset()
        if not hasattr(self, "_learn_rule_open"):
            self._learn_rule_open = False
        if not hasattr(self, "_learn_rule_hit"):
            self._learn_rule_hit: pygame.Rect | None = None
        if not hasattr(self, "show_ocelli_funnel"):
            # Migrate old combined toggle if present in memory.
            legacy = bool(getattr(self, "show_device_funnels", False))
            self.show_ocelli_funnel = legacy
            self.show_turret_funnel = legacy
        if not hasattr(self, "show_turret_funnel"):
            self.show_turret_funnel = False
        if not hasattr(self, "show_turret_aim"):
            self.show_turret_aim = True
        if not hasattr(self, "view_detail"):
            self.view_detail = 1.0
        if not hasattr(self, "_view_detail_rect"):
            self._view_detail_rect = None
            self._view_detail_drag = False
        for _pid in SIDE_PANELS:
            self.side_panel_mini.setdefault(_pid, False)
            self.side_panel_scroll.setdefault(_pid, 0)
        if not hasattr(self, "_scope_mask"):
            self._scope_mask = None
            self._funnel_pair_cache = None
            self._funnel_pair_cache_key = None
            self._funnel_probe_layers = None
        if not hasattr(self, "_well_field"):
            self._well_field = WellField()
        if not hasattr(self, "well_grow"):
            self.well_grow = WellGrowParams.from_funnel_gate(
                getattr(self, "funnel_gate", None)
            )
        if not hasattr(self, "_channel_pairs"):
            self._channel_pairs = []
            self._channel_ids = set()
            self._corridor_cache_key = None
            self._corridor_ids = set()
        if not hasattr(self, "_corridor_topo_pairs"):
            self._corridor_topo_pairs: list[tuple[int, int]] = []
            self._corridor_topo_ids: set[int] = set()
            self._corridor_topo_key = None
        # Plastic pathway flash (pre→post edges touched this beat).
        self._learn_flash_pre: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._learn_flash_post: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._learn_flash_amp: float = 0.0
        self._learn_flash_n: int = 0  # edges touched last step
        self._learn_hot_ids: set[int] = set()
        self._learn_status_i: int = 0
        # Backfill terminal fields on older sessions.
        if not hasattr(self, "ocelli_terminals"):
            self.ocelli_terminals = []
        if not hasattr(self, "turret_motor_terminals"):
            self.turret_motor_terminals = []
            self.turret_sensory_terminals = []
            self.turret_bus_terminal = None
        # Draw thinning: auto (stratified) / manual (per-class sliders) / off.
        self.draw_thin_mode: str = "auto"  # auto | manual | off
        self.draw_thin_open: bool = False  # filter dropdown expanded
        self.draw_density: dict[str, float] = {}  # class name -> keep frac 0..1
        self.draw_soma_budget: int = DRAW_SOMA_MAX
        self._density_drag: int | None = None  # class idx while dragging slider
        self._density_drag_budget: bool = False
        self._density_slider_rects: dict[int, pygame.Rect] = {}

        self._coord_span = 1.0
        self._panel_hits: list[tuple[pygame.Rect, str]] = []
        self._fly_io_status = "Empty — Import MaleCNS or Load .fly"
        self._last_model: str | None = None  # "malecns" or absolute .fly path
        FLIES_DIR.mkdir(parents=True, exist_ok=True)

        # Default Y (top) — same angles as synthesizer set_view_y
        self.view_angle_x = math.pi / 2
        self.view_angle_y = 0.0
        self.scale_3d = 48.0
        self.scale_3d_min = 8.0
        self.scale_3d_max = 800.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self._dragging_orbit = False
        self._dragging_pan = False
        self._last_mouse = (0, 0)
        self._view_surf: pygame.Surface | None = None
        self._nav_rects: dict[str, pygame.Rect] = {}
        self._chrome_rects: dict[str, pygame.Rect] = {}
        self._tool_rects: dict[str, pygame.Rect] = {}
        self._active_tool: str = "nav"  # arrow = projection orbit/pan/zoom
        # Side: remember mouse-adjusted angles; None → default Side on next click.
        self._nav_preset: str | None = "y"  # boot default is View Y
        self._side_view_memory: tuple[float, float] | None = None

        self._layout(boot_w, boot_h)
        # Load fly first (needs class_names for focus), but do not persist UI yet —
        # defaults would wipe camera / funnel checkboxes before _load_ui_state.
        self._restore_last_model()
        if not self._load_ui_state():
            if self.fly.neurons:
                self._fit_scale()
        else:
            # Apply persisted panel collapse to geometry
            self._layout(self.width, self.height)
        # Fold legacy Layers toggles into Super-class focus (one source of truth).
        self._migrate_layers_into_focus()
        self._prune_and_default_focus(seed_if_blank=False)
        # scope_on is restored after adopt — apply hop/draw lock now.
        if getattr(self, "scope_on", False):
            if not self.apply_training_scope():
                self.scope_on = False
        # Persist device map from adopt together with restored camera/funnels.
        self._save_ui_state()

    def _display_flags(self) -> int:
        flags = pygame.DOUBLEBUF | pygame.RESIZABLE
        if self._maximized:
            # Maximized window (decorated), not exclusive fullscreen.
            flags |= getattr(pygame, "WINDOWMAXIMIZED", 0)
        return flags

    def _reopen_display(self, w: int | None = None, h: int | None = None) -> None:
        if w is None or h is None:
            cw, ch = self.screen.get_size()
            w = cw if w is None else w
            h = ch if h is None else h
        self.screen = pygame.display.set_mode((int(w), int(h)), self._display_flags())
        self._layout(int(w), int(h))

    def set_view_x(self) -> None:
        self._stash_side_view_memory()
        self.show_port_schema = False
        self.show_brain_schema = False
        self.show_grow_schema = False
        self.show_research_schema = False
        self.view_angle_x = 0.0
        self.view_angle_y = 0.0
        self._nav_preset = "x"

    def set_view_y(self) -> None:
        self._stash_side_view_memory()
        self.show_port_schema = False
        self.show_brain_schema = False
        self.show_grow_schema = False
        self.show_research_schema = False
        self.view_angle_x = math.pi / 2
        self.view_angle_y = 0.0
        self._nav_preset = "y"

    def set_view_z(self) -> None:
        self._stash_side_view_memory()
        self.show_port_schema = False
        self.show_brain_schema = False
        self.show_grow_schema = False
        self.show_research_schema = False
        self.view_angle_x = 0.0
        self.view_angle_y = math.pi / 2
        self._nav_preset = "z"

    def set_view_side(self) -> None:
        """Restore last mouse-adjusted Side, else the default Side preset."""
        self.show_port_schema = False
        self.show_brain_schema = False
        self.show_grow_schema = False
        self.show_research_schema = False
        mem = getattr(self, "_side_view_memory", None)
        if mem is not None:
            self.view_angle_x = float(mem[0])
            self.view_angle_y = float(mem[1])
        else:
            self.view_angle_x = float(SIDE_VIEW_ANGLE_X)
            self.view_angle_y = float(SIDE_VIEW_ANGLE_Y)
        self._nav_preset = "side"

    def _is_default_side_angles(self, ax: float, ay: float) -> bool:
        return (
            abs(float(ax) - float(SIDE_VIEW_ANGLE_X)) < float(SIDE_VIEW_MATCH)
            and abs(float(ay) - float(SIDE_VIEW_ANGLE_Y)) < float(SIDE_VIEW_MATCH)
        )

    def _stash_side_view_memory(self) -> None:
        """If leaving a mouse-tweaked Side, keep those angles for the next Side click."""
        if getattr(self, "_nav_preset", None) != "side":
            return
        if self._schematic_active():
            return
        ax, ay = float(self.view_angle_x), float(self.view_angle_y)
        if self._is_default_side_angles(ax, ay):
            return
        self._side_view_memory = (ax, ay)

    def _note_side_mouse_edit(self) -> None:
        """Orbit/pan/zoom while on Side → remember this as the Side return state."""
        if getattr(self, "_nav_preset", None) != "side":
            return
        if self._schematic_active():
            return
        self._side_view_memory = (
            float(self.view_angle_x),
            float(self.view_angle_y),
        )

    def _infer_nav_preset(self) -> str | None:
        """Best-effort preset from angles (cold load without nav_preset key)."""
        if self._schematic_active():
            return getattr(self, "_nav_preset", None)
        ax, ay = float(self.view_angle_x), float(self.view_angle_y)
        if abs(ax) < 0.1 and abs(ay) < 0.1:
            return "x"
        if abs(ax - math.pi / 2) < 0.1:
            return "y"
        if abs(ay - math.pi / 2) < 0.1 and abs(ax) < 0.1:
            return "z"
        if self._is_default_side_angles(ax, ay):
            return "side"
        mem = getattr(self, "_side_view_memory", None)
        if (
            mem is not None
            and abs(ax - float(mem[0])) < float(SIDE_VIEW_MATCH)
            and abs(ay - float(mem[1])) < float(SIDE_VIEW_MATCH)
        ):
            return "side"
        return None

    def _name_to_idx(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.class_names)}

    def _names_to_indices(self, names: object) -> set[int]:
        if not isinstance(names, list):
            return set()
        lut = self._name_to_idx()
        out: set[int] = set()
        for n in names:
            i = lut.get(str(n))
            if i is not None:
                out.add(i)
        return out

    def _indices_to_names(self, indices: set[int]) -> list[str]:
        return sorted(
            self.class_names[i]
            for i in indices
            if 0 <= i < len(self.class_names)
        )

    def _load_ui_state(self) -> bool:
        """Restore filter + camera. Returns True if a state file was applied."""
        if not UI_STATE_PATH.is_file():
            return False
        try:
            raw = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(raw, dict):
            return False

        self.focus_soma_indices = self._names_to_indices(raw.get("focus_soma"))
        self.focus_graph_indices = self._names_to_indices(
            raw.get("focus_graph", raw.get("focus_proc"))
        )
        self.focus_axon_indices = self._names_to_indices(raw.get("focus_axon"))
        self.focus_dendrite_indices = self._names_to_indices(raw.get("focus_dendrite"))
        self.focus_tract_indices = self._names_to_indices(raw.get("focus_tract"))
        self.focus_basin_indices = self._names_to_indices(raw.get("focus_basin"))
        # Keep legacy proc mirror in sync with graph (atlas ghost).
        self.focus_proc_indices = set(self.focus_graph_indices)
        hint = raw.get("focus_hint")
        if isinstance(hint, str):
            self.focus_hint_class = self._name_to_idx().get(hint)
        elif self._any_focus():
            self.focus_hint_class = next(iter(self._focused_parents()), None)
        else:
            self.focus_hint_class = None

        for key, attr, default in (
            ("show_layer_neurons", "show_layer_neurons", True),
            ("show_layer_graph", "show_layer_graph", False),
            ("show_layer_axons", "show_layer_axons", True),
            ("show_layer_dendrites", "show_layer_dendrites", True),
            ("show_layer_basins", "show_layer_basins", False),
        ):
            if isinstance(raw.get(key), bool):
                setattr(self, attr, raw[key])
            else:
                setattr(self, attr, default)
        if isinstance(raw.get("show_port_schema"), bool):
            self.show_port_schema = raw["show_port_schema"]
        if isinstance(raw.get("show_brain_schema"), bool):
            self.show_brain_schema = raw["show_brain_schema"]
        elif isinstance(raw.get("show_tissue_schema"), bool):
            # Legacy key from before CNS tissue → CNS Brain rename.
            self.show_brain_schema = raw["show_tissue_schema"]
        if isinstance(raw.get("show_grow_schema"), bool):
            self.show_grow_schema = raw["show_grow_schema"]
        if isinstance(raw.get("show_research_schema"), bool):
            self.show_research_schema = raw["show_research_schema"]
        # At most one schematic replaces the 3D view.
        schem_flags = [
            ("show_port_schema", bool(getattr(self, "show_port_schema", False))),
            ("show_brain_schema", bool(getattr(self, "show_brain_schema", False))),
            ("show_grow_schema", bool(getattr(self, "show_grow_schema", False))),
            ("show_research_schema", bool(getattr(self, "show_research_schema", False))),
        ]
        on = [n for n, f in schem_flags if f]
        if len(on) > 1:
            prefer = next(
                (
                    n
                    for n in (
                        "show_research_schema",
                        "show_grow_schema",
                        "show_brain_schema",
                        "show_port_schema",
                    )
                    if raw.get(n)
                    or (n == "show_brain_schema" and raw.get("show_tissue_schema"))
                ),
                on[0],
            )
            self.show_port_schema = prefer == "show_port_schema"
            self.show_brain_schema = prefer == "show_brain_schema"
            self.show_grow_schema = prefer == "show_grow_schema"
            self.show_research_schema = prefer == "show_research_schema"
        # Ignore legacy Asm/Sample schematic flags from older UI saves.

        for key, attr, lo, hi in (
            ("view_angle_x", "view_angle_x", -math.pi / 2 + 0.05, math.pi / 2 - 0.05),
            ("view_angle_y", "view_angle_y", -math.tau, math.tau),
            ("scale_3d", "scale_3d", self.scale_3d_min, self.scale_3d_max),
            ("pan_x", "pan_x", -1e5, 1e5),
            ("pan_y", "pan_y", -1e5, 1e5),
        ):
            val = raw.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                setattr(self, attr, float(max(lo, min(hi, val))))

        mem = raw.get("side_view_memory")
        if (
            isinstance(mem, (list, tuple))
            and len(mem) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in mem)
        ):
            self._side_view_memory = (float(mem[0]), float(mem[1]))
        else:
            self._side_view_memory = None
        nav = raw.get("nav_preset")
        if isinstance(nav, str) and nav in ("x", "y", "z", "side"):
            self._nav_preset = nav
        else:
            self._nav_preset = self._infer_nav_preset()

        if isinstance(raw.get("legend_mini"), bool):
            self.legend_mini = raw["legend_mini"]
        if isinstance(raw.get("panel_collapsed"), bool):
            self.panel_collapsed = raw["panel_collapsed"]
        tab = raw.get("legend_tab")
        if isinstance(tab, str):
            tab = _LEGACY_TAB_MAP.get(tab, tab)
            if tab in SIDE_PANEL_LABELS:
                self.legend_tab = tab
        sp_mini = raw.get("side_panel_mini")
        if isinstance(sp_mini, dict):
            for k, v in sp_mini.items():
                kk = _LEGACY_TAB_MAP.get(str(k), str(k))
                if kk in SIDE_PANEL_LABELS and isinstance(v, bool):
                    self.side_panel_mini[kk] = v
        sp_scroll = raw.get("side_panel_scroll")
        if isinstance(sp_scroll, dict):
            for k, v in sp_scroll.items():
                kk = _LEGACY_TAB_MAP.get(str(k), str(k))
                if kk in SIDE_PANEL_LABELS and isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.side_panel_scroll[kk] = max(0, int(v))
        mode = raw.get("draw_thin_mode")
        if mode in ("auto", "manual", "off"):
            self.draw_thin_mode = mode
        dens = raw.get("draw_density")
        if isinstance(dens, dict):
            for k, v in dens.items():
                if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.draw_density[k] = float(max(0.0, min(1.0, float(v))))
        bf = raw.get("basin_form")
        if isinstance(bf, (int, float)) and not isinstance(bf, bool):
            self.basin_form = float(max(0.0, min(1.0, float(bf))))
        if isinstance(raw.get("light_enabled"), bool):
            self.light_enabled = raw["light_enabled"]
        if isinstance(raw.get("lamp_on"), bool):
            self.lamp_on = raw["lamp_on"]
        for key, attr, lo, hi in (
            ("sky_fill", "sky_fill", 0.0, 1.0),
            ("lamp_intensity", "lamp_intensity", 0.0, 2.0),
            ("lamp_speed", "lamp_speed", 0.0, LAMP_SPEED_MAX),
            ("lamp_angle", "lamp_angle", -math.tau * 4, math.tau * 4),
            ("ground_relief", "ground_relief", 0.0, 1.0),
        ):
            val = raw.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                setattr(self, attr, float(max(lo, min(hi, float(val)))))
        if isinstance(raw.get("ground_on"), bool):
            self.ground_on = raw["ground_on"]
        if isinstance(raw.get("world_swing_on"), bool):
            self.world_swing_on = raw["world_swing_on"]
        for key, attr, lo, hi in (
            ("world_swing_amp", "world_swing_amp", 0.0, WORLD_SWING_AMP_MAX),
            ("world_swing_hz", "world_swing_hz", 0.0, WORLD_SWING_HZ_MAX),
            ("world_swing_phase", "world_swing_phase", -math.tau * 8, math.tau * 8),
        ):
            val = raw.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                setattr(self, attr, float(max(lo, min(hi, float(val)))))
        if isinstance(raw.get("ocelli_on"), bool):
            self.ocelli_on = raw["ocelli_on"]
        if isinstance(raw.get("turret_on"), bool):
            self.turret_on = raw["turret_on"]
        if isinstance(raw.get("eye_left_on"), bool):
            self.eye_left_on = raw["eye_left_on"]
        if isinstance(raw.get("eye_right_on"), bool):
            self.eye_right_on = raw["eye_right_on"]
        if isinstance(raw.get("jo_left_on"), bool):
            self.jo_left_on = raw["jo_left_on"]
        if isinstance(raw.get("jo_right_on"), bool):
            self.jo_right_on = raw["jo_right_on"]
        if isinstance(raw.get("learn_on"), bool):
            self.learn_on = raw["learn_on"]
        if isinstance(raw.get("motor_socket_on"), bool):
            self.motor_socket_on = raw["motor_socket_on"]
            if hasattr(self, "rule"):
                self.rule.motor_socket_on = bool(self.motor_socket_on)
        if isinstance(raw.get("motor_socket"), dict):
            self._apply_motor_socket_snapshot(raw["motor_socket"])
        if isinstance(raw.get("sensor_port_on"), bool):
            self.sensor_port_on = bool(raw["sensor_port_on"])
        if isinstance(raw.get("sensor_port_train"), bool):
            self.sensor_port_train = bool(raw["sensor_port_train"])
        if isinstance(raw.get("sensor_port_nn"), dict):
            from sensor_port_nn import SensorPortNN

            self.sensor_port_nn = SensorPortNN.from_dict(raw["sensor_port_nn"])
        if isinstance(raw.get("show_ocelli_funnel"), bool):
            self.show_ocelli_funnel = raw["show_ocelli_funnel"]
        if isinstance(raw.get("show_eye_funnel"), bool):
            self.show_eye_funnel = raw["show_eye_funnel"]
        if isinstance(raw.get("show_jo_funnel"), bool):
            self.show_jo_funnel = raw["show_jo_funnel"]
        if isinstance(raw.get("show_turret_funnel"), bool):
            self.show_turret_funnel = raw["show_turret_funnel"]
        elif isinstance(raw.get("show_device_funnels"), bool):
            # Legacy single toggle → both.
            self.show_ocelli_funnel = bool(raw["show_device_funnels"])
            self.show_turret_funnel = bool(raw["show_device_funnels"])
        if isinstance(raw.get("show_turret_aim"), bool):
            self.show_turret_aim = raw["show_turret_aim"]
        if isinstance(raw.get("view_detail"), (int, float)) and not isinstance(
            raw.get("view_detail"), bool
        ):
            self.view_detail = float(max(0.0, min(1.0, float(raw["view_detail"]))))
            self.draw_soma_budget = self._draw_cap(DRAW_SOMA_MAX)
        # One door: scope_on. Older UI JSON may still say funnel_model_on.
        if isinstance(raw.get("scope_on"), bool):
            self.scope_on = raw["scope_on"]
        elif isinstance(raw.get("funnel_model_on"), bool):
            self.scope_on = raw["funnel_model_on"]
        if hasattr(self, "blueprint"):
            self.blueprint.scope_on = bool(getattr(self, "scope_on", False))
        if isinstance(raw.get("hop_clip"), bool):
            self.hop_clip = raw["hop_clip"]
            if hasattr(self, "blueprint"):
                self.blueprint.hop_clip = bool(raw["hop_clip"])
        last = raw.get("last_model")
        if isinstance(last, str) and last.strip():
            self._last_model = last.strip()
        sync_enabled_from_host(self)
        return True

    def _save_ui_state(self) -> None:
        payload = {
            "focus_soma": self._indices_to_names(self.focus_soma_indices),
            "focus_graph": self._indices_to_names(self.focus_graph_indices),
            "focus_axon": self._indices_to_names(self.focus_axon_indices),
            "focus_dendrite": self._indices_to_names(self.focus_dendrite_indices),
            "focus_tract": self._indices_to_names(self.focus_tract_indices),
            "focus_basin": self._indices_to_names(self.focus_basin_indices),
            "focus_proc": self._indices_to_names(self.focus_graph_indices),
            "show_layer_neurons": self.show_layer_neurons,
            "show_layer_graph": self.show_layer_graph,
            "show_layer_axons": self.show_layer_axons,
            "show_layer_dendrites": self.show_layer_dendrites,
            "show_layer_basins": self.show_layer_basins,
            "show_port_schema": bool(getattr(self, "show_port_schema", False)),
            "show_brain_schema": bool(getattr(self, "show_brain_schema", False)),
            "show_grow_schema": bool(getattr(self, "show_grow_schema", False)),
            "show_research_schema": bool(getattr(self, "show_research_schema", False)),
            "focus_hint": (
                self.class_names[self.focus_hint_class]
                if self.focus_hint_class is not None
                and 0 <= self.focus_hint_class < len(self.class_names)
                else None
            ),
            "view_angle_x": self.view_angle_x,
            "view_angle_y": self.view_angle_y,
            "scale_3d": self.scale_3d,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "nav_preset": getattr(self, "_nav_preset", None),
            "side_view_memory": (
                [float(self._side_view_memory[0]), float(self._side_view_memory[1])]
                if getattr(self, "_side_view_memory", None) is not None
                else None
            ),
            "legend_mini": self.legend_mini,
            "panel_collapsed": self.panel_collapsed,
            "legend_tab": str(getattr(self, "legend_tab", "filters")),
            "side_panel_mini": {
                p: bool(self.side_panel_mini.get(p, False)) for p in SIDE_PANELS
            },
            "side_panel_scroll": {
                p: int(self.side_panel_scroll.get(p, 0)) for p in SIDE_PANELS
            },
            "draw_thin_mode": self.draw_thin_mode,
            "draw_density": {
                k: float(max(0.0, min(1.0, float(v))))
                for k, v in self.draw_density.items()
                if isinstance(k, str)
            },
            "basin_form": float(max(0.0, min(1.0, float(self.basin_form)))),
            "light_enabled": bool(self.light_enabled),
            "lamp_on": bool(self.lamp_on),
            "sky_fill": float(self.sky_fill),
            "lamp_intensity": float(self.lamp_intensity),
            "lamp_speed": float(self.lamp_speed),
            "lamp_angle": float(self.lamp_angle),
            "ground_on": bool(self.ground_on),
            "ground_relief": float(self.ground_relief),
            "world_swing_on": bool(self.world_swing_on),
            "world_swing_amp": float(self.world_swing_amp),
            "world_swing_hz": float(self.world_swing_hz),
            "world_swing_phase": float(self.world_swing_phase),
            "ocelli_on": bool(self.ocelli_on),
            "turret_on": bool(self.turret_on),
            "eye_left_on": bool(getattr(self, "eye_left_on", False)),
            "eye_right_on": bool(getattr(self, "eye_right_on", False)),
            "jo_left_on": bool(getattr(self, "jo_left_on", False)),
            "jo_right_on": bool(getattr(self, "jo_right_on", False)),
            "learn_on": bool(self.learn_on),
            "motor_socket_on": bool(getattr(self, "motor_socket_on", False)),
            "motor_socket": self._motor_socket_snapshot(),
            "sensor_port_on": bool(getattr(self, "sensor_port_on", False)),
            "sensor_port_train": bool(getattr(self, "sensor_port_train", False)),
            "sensor_port_nn": (
                self.sensor_port_nn.to_dict()
                if getattr(self, "sensor_port_nn", None) is not None
                else None
            ),
            "show_ocelli_funnel": bool(self.show_ocelli_funnel),
            "show_eye_funnel": bool(getattr(self, "show_eye_funnel", False)),
            "show_jo_funnel": bool(getattr(self, "show_jo_funnel", False)),
            "show_turret_funnel": bool(self.show_turret_funnel),
            "show_turret_aim": bool(getattr(self, "show_turret_aim", True)),
            "view_detail": float(max(0.0, min(1.0, float(getattr(self, "view_detail", 1.0))))),
            "scope_on": bool(getattr(self, "scope_on", False)),
            "hop_clip": bool(getattr(self, "hop_clip", True)),
            "last_model": self._last_model,
            # Device neuron ids — backup when .fly was saved with shell_devices.
            "shell_devices": self._device_snapshot(),
        }
        try:
            UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            UI_STATE_PATH.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    def _device_snapshot(self) -> dict:
        """Serializable device neuron id map (survives Save/Load)."""
        from device import devices_to_shell

        base = {
            "v": 1,
            "ocelli_groups": [
                [int(x) for x in g] for g in (self.ocelli_groups or [])
            ],
            "ocelli_terminals": [
                int(x) for x in (getattr(self, "ocelli_terminals", []) or [])
            ],
            "turret_motor_groups": [
                [int(x) for x in g]
                for g in (getattr(self, "turret_motor_groups", []) or [])
            ],
            "turret_sensory_groups": [
                [int(x) for x in g]
                for g in (getattr(self, "turret_sensory_groups", []) or [])
            ],
            "turret_motor_terminals": [
                int(x)
                for x in (getattr(self, "turret_motor_terminals", []) or [])
            ],
            "turret_sensory_terminals": [
                int(x)
                for x in (getattr(self, "turret_sensory_terminals", []) or [])
            ],
            "turret_bus_terminal": (
                int(self.turret_bus_terminal)
                if getattr(self, "turret_bus_terminal", None) is not None
                else None
            ),
            "turret_yaw": float(getattr(self, "turret_yaw", 0.0)),
            "turret_pitch": float(getattr(self, "turret_pitch", 0.4)),
        }
        # Merge eye (and any other typed) fragments from device_list.
        try:
            base.update(devices_to_shell(list(self.iter_devices())))
        except Exception:
            pass
        return base

    def _motor_socket_snapshot(self) -> dict | None:
        """Learned port W (+ frozen) for .fly / UI persistence."""
        sock = getattr(self, "motor_socket", None)
        if sock is None:
            return None
        try:
            return sock.to_dict()
        except Exception:
            return None

    def _apply_motor_socket_snapshot(self, snap: dict | None) -> bool:
        """Restore motor socket W/frozen. True if a full W matrix landed."""
        if not isinstance(snap, dict):
            return False
        from motor_socket import MotorSocket

        rule = getattr(self, "rule", None)
        lr = float(getattr(rule, "motor_socket_lr", 0.35) or 0.35)
        sock = getattr(self, "motor_socket", None)
        if sock is None:
            self.motor_socket = MotorSocket.from_dict(snap, lr=lr)
            ok = True
        else:
            ok = bool(sock.apply_dict(snap))
        self.motor_socket_on = bool(
            getattr(self, "motor_socket_on", False)
        )
        # Never force socket ON just because a freeze snapshot exists —
        # CNS learn must stay socket-off unless the user enables it.
        if getattr(self.motor_socket, "frozen", False):
            # Keep freeze ε latch empty until live aim updates; mark as frozen recipe.
            if not math.isfinite(
                float(getattr(self, "_learn_err_freeze_deg", float("nan")))
            ):
                self._learn_err_freeze_deg = float("nan")
        return ok

    def _apply_device_snapshot(self, snap: dict | None) -> bool:
        """Bind shell device groups to saved neuron ids. Returns True if applied."""
        if not isinstance(snap, dict) or not self.fly.neurons:
            return False
        # Accept nested {"shell_devices": {...}} or flat snapshot.
        if "ocelli_groups" not in snap and isinstance(snap.get("shell_devices"), dict):
            snap = snap["shell_devices"]
        if "ocelli_groups" not in snap and isinstance(snap.get("devices"), dict):
            snap = snap["devices"]
        if "ocelli_groups" not in snap:
            return False

        def _alive(ids: list[int]) -> list[int]:
            return [int(i) for i in ids if int(i) in self.fly.neurons]

        def _flatten(groups: list) -> list[int]:
            out: list[int] = []
            for g in groups:
                if isinstance(g, list):
                    out.extend(int(x) for x in g)
            return out

        # Reject snapshots that belong to another fly (most ids missing).
        probe = _flatten(snap.get("ocelli_groups") or [])
        probe.extend(int(x) for x in (snap.get("ocelli_terminals") or []) if x is not None)
        probe.extend(_flatten(snap.get("turret_motor_groups") or []))
        probe.extend(_flatten(snap.get("turret_sensory_groups") or []))
        for key in (
            "turret_motor_terminals",
            "turret_sensory_terminals",
        ):
            for x in snap.get(key) or []:
                if x is not None:
                    probe.append(int(x))
        bus = snap.get("turret_bus_terminal")
        if bus is not None:
            probe.append(int(bus))
        if probe:
            n_alive = sum(1 for i in probe if int(i) in self.fly.neurons)
            if n_alive < max(1, int(0.5 * len(probe))):
                return False

        og = snap.get("ocelli_groups") or []
        ot = snap.get("ocelli_terminals") or []
        groups = [_alive(g if isinstance(g, list) else []) for g in og]
        terms = _alive(ot if isinstance(ot, list) else [])
        # Require at least one live ocellus id if ocelli_on, else allow empty.
        if self.ocelli_on and (len(groups) == 3 and any(groups) and len(terms) == 3):
            self.ocelli_groups = groups
            self.ocelli_terminals = terms
            self.ocelli_ids = self._ocelli_flat_ids()
        elif self.ocelli_on and any(groups):
            # Partial — still better than recreating.
            while len(groups) < 3:
                groups.append([])
            self.ocelli_groups = groups[:3]
            self.ocelli_terminals = terms
            self.ocelli_ids = self._ocelli_flat_ids()

        n_sites = int(TURRET_SITES)
        tmg = snap.get("turret_motor_groups") or []
        tsg = snap.get("turret_sensory_groups") or []
        tmt = snap.get("turret_motor_terminals") or []
        tst = snap.get("turret_sensory_terminals") or []
        bus = snap.get("turret_bus_terminal")
        if self.turret_on:
            self.turret_motor_groups = [
                _alive(g if isinstance(g, list) else [])
                for g in list(tmg)[:n_sites]
            ]
            self.turret_sensory_groups = [
                _alive(g if isinstance(g, list) else [])
                for g in list(tsg)[:n_sites]
            ]
            while len(self.turret_motor_groups) < n_sites:
                self.turret_motor_groups.append([])
            while len(self.turret_sensory_groups) < n_sites:
                self.turret_sensory_groups.append([])
            self.turret_motor_terminals = _alive(
                tmt if isinstance(tmt, list) else []
            )
            self.turret_sensory_terminals = _alive(
                tst if isinstance(tst, list) else []
            )
            if bus is not None and int(bus) in self.fly.neurons:
                self.turret_bus_terminal = int(bus)
            else:
                self.turret_bus_terminal = None
            if "turret_yaw" in snap:
                self.turret_yaw = float(snap["turret_yaw"])
            if "turret_pitch" in snap:
                self.turret_pitch = float(
                    np.clip(float(snap["turret_pitch"]), TURRET_PITCH_MIN, TURRET_PITCH_MAX)
                )
        # Compound eyes (typed devices).
        for d in compound_eyes.iter_eye_devices(self):
            prefix = d.name
            on = snap.get(f"{prefix}_on")
            if isinstance(on, bool):
                d.enabled = on
                setattr(self, f"{prefix}_on", on)
            raw_g = snap.get(f"{prefix}_groups") or []
            raw_t = snap.get(f"{prefix}_terminals") or []
            if raw_g and raw_t:
                groups = [_alive(g if isinstance(g, list) else []) for g in raw_g]
                terms = _alive(raw_t if isinstance(raw_t, list) else [])
                if any(groups) and terms:
                    d.groups = groups
                    d.terminals = terms
        # Johnston organs (paired antennae).
        for d in johnston_organ.iter_jo_devices(self):
            prefix = d.name
            on = snap.get(f"{prefix}_on")
            if isinstance(on, bool):
                d.enabled = on
                setattr(self, f"{prefix}_on", on)
            raw_g = snap.get(f"{prefix}_groups") or []
            raw_t = snap.get(f"{prefix}_terminals") or []
            if raw_g and raw_t:
                groups = [_alive(g if isinstance(g, list) else []) for g in raw_g]
                terms = _alive(raw_t if isinstance(raw_t, list) else [])
                if any(groups) and terms:
                    d.groups = groups
                    d.terminals = terms
        return bool(
            self._ocelli_flat_ids()
            or self._turret_flat_ids()
            or any(d.flat_ids() for d in compound_eyes.iter_eye_devices(self))
            or any(d.flat_ids() for d in johnston_organ.iter_jo_devices(self))
        )

    def _recover_devices_from_fly(self) -> bool:
        """Legacy .fly: rebind packs by soma proximity to mounts (no id meta)."""
        if not self.fly.neurons:
            return False
        ok = False
        if self.ocelli_on and not any(self.ocelli_groups):
            ok = self._recover_ocelli_by_pose() or ok
        if self.turret_on and not any(
            getattr(self, "turret_motor_groups", []) or []
        ):
            ok = self._recover_turret_by_pose() or ok
        return ok

    def _recover_ocelli_by_pose(self) -> bool:
        pose = self._ocelli_rig_pose()
        if pose is None:
            return False
        try:
            sens_cls = self.class_names.index("sensory")
        except ValueError:
            return False
        # Candidates: sensory somata near each cup (added device packs sit here).
        n_pack = int(OCELLI_GROUP_N)
        frame = self._body_orbit_frame()
        r = float(frame[1]) if frame is not None else 1.0
        rad = max(0.08 * r, 1e-3)
        groups: list[list[int]] = []
        terms: list[int] = []
        used: set[int] = set()
        for oc in pose:
            lens = np.asarray(oc["lens"], dtype=np.float64)
            near: list[tuple[float, int]] = []
            for nid, neu in self.fly.neurons.items():
                if int(nid) in used or int(neu.cls) != int(sens_cls):
                    continue
                d = float(np.linalg.norm(np.asarray(neu.soma, dtype=np.float64) - lens))
                if d <= rad * 1.8:
                    near.append((d, int(nid)))
            near.sort()
            if len(near) < n_pack + 1:
                return False
            # Closest ≈ clamp (inset), next n_pack ≈ pack.
            term = near[0][1]
            pack = [nid for _, nid in near[1 : 1 + n_pack]]
            if len(pack) < n_pack:
                return False
            terms.append(term)
            groups.append(pack)
            used.add(term)
            used.update(pack)
        if len(groups) != 3 or len(terms) != 3:
            return False
        self.ocelli_groups = groups
        self.ocelli_terminals = terms
        self.ocelli_ids = self._ocelli_flat_ids()
        return True

    def _recover_turret_by_pose(self) -> bool:
        pose = self._turret_rig_pose()
        if pose is None:
            return False
        try:
            motor_cls = self.class_names.index("motor")
        except ValueError:
            return False
        try:
            sens_cls = self.class_names.index("sensory")
        except ValueError:
            return False
        n_m = int(TURRET_MOTOR_N)
        n_s = int(TURRET_SENSORY_N)
        r = float(pose["r"])
        rad = max(0.06 * r, 1e-3)
        sites = pose["sites"]
        mount = np.asarray(pose["mount"], dtype=np.float64)
        used: set[int] = set(self._ocelli_flat_ids())
        m_groups: list[list[int]] = []
        s_groups: list[list[int]] = []
        m_terms: list[int] = []
        s_terms: list[int] = []

        def _near(origin: np.ndarray, cls: int, ban: set[int]) -> list[tuple[float, int]]:
            out: list[tuple[float, int]] = []
            for nid, neu in self.fly.neurons.items():
                if int(nid) in ban or int(neu.cls) != int(cls):
                    continue
                d = float(
                    np.linalg.norm(np.asarray(neu.soma, dtype=np.float64) - origin)
                )
                if d <= rad * 2.2:
                    out.append((d, int(nid)))
            out.sort()
            return out

        for site in sites:
            origin = np.asarray(site, dtype=np.float64)
            m_near = _near(origin, motor_cls, used)
            s_near = _near(origin, sens_cls, used)
            if len(m_near) < n_m + 1 or len(s_near) < n_s + 1:
                return False
            mt, st = m_near[0][1], s_near[0][1]
            mg = [nid for _, nid in m_near[1 : 1 + n_m]]
            sg = [nid for _, nid in s_near[1 : 1 + n_s]]
            m_terms.append(mt)
            s_terms.append(st)
            m_groups.append(mg)
            s_groups.append(sg)
            used.update([mt, st, *mg, *sg])
        bus_near = _near(mount, motor_cls, used)
        if not bus_near:
            return False
        self.turret_motor_groups = m_groups
        self.turret_sensory_groups = s_groups
        self.turret_motor_terminals = m_terms
        self.turret_sensory_terminals = s_terms
        self.turret_bus_terminal = int(bus_near[0][1])
        return True

    def _remember_model(self, key: str) -> None:
        self._last_model = key
        self._save_ui_state()

    def _restore_last_model(self) -> None:
        """Re-open the last Import/Load target from shell_ui.json (if any)."""
        if not UI_STATE_PATH.is_file():
            return
        try:
            raw = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        last = raw.get("last_model")
        if not isinstance(last, str) or not last.strip():
            return
        last = last.strip()
        self._last_model = last
        if last == "malecns":
            if not SOMAS_PATH.is_file():
                self._fly_io_status = "Last model MaleCNS missing — re-import"
                return
            try:
                fly = load_fly_from_malecns(SOMAS_PATH, PROCESSES_PATH)
            except OSError as exc:
                self._fly_io_status = f"Auto-load MaleCNS failed: {exc}"
                return
            self._adopt_fly(
                fly,
                f"Atlas MaleCNS ({len(fly.neurons):,} n, "
                f"{len(fly.recipe_pre):,} recipe edges)",
                refit=False,
                from_malecns=True,
                remember=False,
            )
            return
        path = Path(last)
        if not path.is_file():
            self._fly_io_status = f"Last model missing: {path.name}"
            return
        try:
            loaded = load_fly(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._fly_io_status = f"Auto-load failed: {exc}"
            return
        self._adopt_fly(
            loaded,
            f"Loaded {path.name}",
            refit=False,
                model_path=path,
                remember=False,
            )

    def _sync_from_fly(self) -> None:
        """Refresh render arrays from the Fly model."""
        self._orbit_cache_key = None
        self._orbit_cache = None
        self._wing_anchor_key = None
        self._wing_anchor_cache = None
        self._frame_ocelli_pose = None
        self._frame_turret_pose = None
        self._frame_eye_pose = {}
        self._frame_jo_pose = {}
        self._compass_blit_key = None
        self._legend_blit_key = None
        self.class_names = self.fly.class_names
        self.points_xyz = self.fly.soma_xyz
        self.points_cls = self.fly.soma_cls
        self.soma_activity = self.fly.soma_activity
        self.point_rgb = _rgb_for_cls(self.points_cls)
        self.class_counts = self.fly.class_counts()
        self.n_shown = len(self.points_xyz)
        self.n_source_points = self.n_shown
        self.proc_xyz = self.fly.atlas_process_xyz
        self.proc_cls = self.fly.atlas_process_cls
        self.proc_rgb = _rgb_for_cls(self.proc_cls)
        self.n_proc_shown = len(self.proc_xyz)
        self.proc_counts = (
            np.bincount(self.proc_cls, minlength=len(self.class_names)).astype(np.int32)
            if len(self.proc_cls)
            else np.zeros(len(self.class_names), dtype=np.int32)
        )
        # Developmental neurite trails (grown axons), separate from atlas ghost.
        self.grown_xyz = self.fly.neurite_xyz
        self.grown_cls = self.fly.neurite_cls
        self.grown_rgb = _axon_rgb_for_cls(self.grown_cls)
        self._basin_cache_key = None
        self._recipe_soma_cache = None
        self._gl_geom_key = None
        gl = getattr(self, "_gl_tissue", None)
        if gl is not None:
            gl.invalidate()
        self._ensure_draw_density()
        # Relocate device packs with the crown — do NOT rewire every sync
        # (strip+connect_pair + rebuild_caches is far too costly).
        if (
            getattr(self, "ocelli_on", False)
            and getattr(self, "ocelli_groups", None)
            and self.fly.neurons
        ):
            pose = self._ocelli_rig_pose()
            if pose is not None and len(self.ocelli_groups) == 3:
                self._place_ocelli_clusters(pose, move_only=True)
        if (
            getattr(self, "turret_on", False)
            and getattr(self, "turret_motor_groups", None)
            and self.fly.neurons
        ):
            if self._turret_rig_pose() is not None and any(self.turret_motor_groups):
                self._place_turret_clusters(move_only=True)

    def _ensure_draw_density(self) -> None:
        """Fill missing per-class keep fractions (small groups stay dense)."""
        for i, name in enumerate(self.class_names):
            if name in self.draw_density:
                continue
            cnt = int(self.class_counts[i]) if i < len(self.class_counts) else 0
            if cnt <= 400:
                self.draw_density[name] = 1.0
            elif cnt <= 5000:
                self.draw_density[name] = 0.45
            elif cnt <= 20000:
                self.draw_density[name] = 0.18
            else:
                self.draw_density[name] = 0.10

    def _density_by_idx(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for i, name in enumerate(self.class_names):
            out[i] = float(np.clip(self.draw_density.get(name, 1.0), 0.0, 1.0))
        return out

    def _sparse_soma_classes(self, *, below: int = 400) -> set[int]:
        """Classes with few atlas somata (process tissue is their visible mass)."""
        out: set[int] = set()
        for i in range(len(self.class_names)):
            cnt = int(self.class_counts[i]) if i < len(self.class_counts) else 0
            if cnt <= int(below):
                out.add(i)
        return out

    def _class_focus_union(self) -> set[int]:
        """Any layer focus → class is "selected" for tissue highlighting."""
        return set().union(
            self.focus_soma_indices,
            self.focus_graph_indices,
            self.focus_axon_indices,
            self.focus_dendrite_indices,
            self.focus_tract_indices,
            self.focus_basin_indices,
        )

    def _tract_focus_set(self) -> set[int]:
        """Recipe-chord filter (Super-class T). Empty T → hide chords (no scope bypass)."""
        return set(self.focus_tract_indices)

    def _focus_modulate_rgb(
        self,
        rgb: np.ndarray,
        cls: np.ndarray,
        focus_set: set[int],
        *,
        other_layer_focused: bool = False,
    ) -> np.ndarray:
        """Match CPU ``_stamp_points`` focus dim/boost (uint8 or float RGB)."""
        if len(rgb) == 0 or len(cls) != len(rgb):
            return rgb
        any_focus = bool(focus_set) or other_layer_focused
        if not any_focus:
            return rgb
        out = np.array(rgb, dtype=np.float32, copy=True)
        scale = 255.0 if float(np.max(out)) > 1.5 else 1.0
        if focus_set:
            is_f = np.isin(cls, list(focus_set))
        else:
            is_f = np.zeros(len(cls), dtype=bool)
        dim = ~is_f
        out[dim] *= float(FOCUS_DIM)
        if scale > 1.5:
            out[is_f] = np.clip(out[is_f] + float(FOCUS_BOOST), 0.0, 255.0)
        else:
            out[is_f] = np.clip(
                out[is_f] + float(FOCUS_BOOST) / 255.0, 0.0, 1.0
            )
        if rgb.dtype == np.uint8 or scale > 1.5:
            return np.clip(out, 0, 255).astype(np.uint8)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _apply_draw_thin(
        self,
        cls_arr: np.ndarray,
        budget: int,
        *,
        boost_sparse_soma: bool = False,
    ) -> np.ndarray | slice:
        """Subsample by current thinning mode (auto / manual / off)."""
        n = int(len(cls_arr))
        if n <= budget or budget <= 0:
            return slice(None)
        mode = self.draw_thin_mode
        if mode == "off":
            return _density_take(n, budget)
        if mode == "manual":
            return _manual_density_take(cls_arr, budget, self._density_by_idx())
        boost = self._sparse_soma_classes() if boost_sparse_soma else None
        return _stratified_density_take(cls_arr, budget, boost_classes=boost)

    def _sync_layers_from_focus(self) -> None:
        """Mirror Super-class focus → show_layer_* (draw path still reads layers)."""
        fp = (
            frozenset(self.focus_soma_indices),
            frozenset(self.focus_graph_indices),
            frozenset(self.focus_axon_indices),
            frozenset(self.focus_dendrite_indices),
            frozenset(self.focus_tract_indices),
            frozenset(self.focus_basin_indices),
        )
        changed = fp != getattr(self, "_focus_sync_fp", None)
        was_basins = bool(getattr(self, "show_layer_basins", False))
        self.show_layer_neurons = bool(self.focus_soma_indices)
        self.show_layer_graph = bool(self.focus_graph_indices)
        self.show_layer_axons = bool(self.focus_axon_indices)
        self.show_layer_dendrites = bool(self.focus_dendrite_indices)
        self.show_layer_basins = bool(self.focus_basin_indices)
        if was_basins and not self.show_layer_basins:
            self._basin_cache_key = None
        self.focus_proc_indices = set(self.focus_graph_indices)
        # Bust caches only when focus sets change — per-frame sync otherwise
        # rebuilt Super-class every tick (pulse blink / full-panel flicker).
        if changed:
            self._focus_sync_fp = fp
            self._legend_blit_key = None
            self._gl_geom_key = None
            self._tissue_blit_key = None
            self._tissue_blit_surf = None

    def _migrate_layers_into_focus(self) -> None:
        """One-shot: legacy show_layer_* + empty focus → seed or stay hidden.

        After this, focus sets are the only switch. N/A/D default to all active
        classes when the old layer was on; G/B stay empty unless layer was on.
        """
        active = self._active_class_indices()
        pairs = (
            ("show_layer_neurons", "focus_soma_indices"),
            ("show_layer_graph", "focus_graph_indices"),
            ("show_layer_axons", "focus_axon_indices"),
            ("show_layer_dendrites", "focus_dendrite_indices"),
            ("show_layer_basins", "focus_basin_indices"),
        )
        for layer_attr, focus_attr in pairs:
            focus: set[int] = getattr(self, focus_attr)
            layer_on = bool(getattr(self, layer_attr, False))
            if focus:
                continue
            if layer_on and active:
                setattr(self, focus_attr, set(active))
            # layer off + empty focus → remain empty (hidden)
        self._sync_layers_from_focus()

    def _prune_and_default_focus(self, *, seed_if_blank: bool = False) -> None:
        """Drop out-of-range focus idx; optionally seed N/A/D if tissue blank."""
        n = len(self.class_names)
        for attr in (
            "focus_soma_indices",
            "focus_graph_indices",
            "focus_axon_indices",
            "focus_dendrite_indices",
            "focus_tract_indices",
            "focus_basin_indices",
        ):
            s = getattr(self, attr)
            setattr(self, attr, {i for i in s if 0 <= int(i) < n})
        active = self._active_class_indices()
        if (
            seed_if_blank
            and active
            and not (
                self.focus_soma_indices
                or self.focus_axon_indices
                or self.focus_dendrite_indices
            )
        ):
            self.focus_soma_indices = set(active)
            self.focus_axon_indices = set(active)
            self.focus_dendrite_indices = set(active)
        self._sync_layers_from_focus()

    def _soma_drawn_count(self) -> int:
        """How many somata the current thin/scope path would stamp (Cloud meter)."""
        n = int(len(self.points_xyz))
        if not self.show_layer_neurons or n == 0:
            return 0
        # Match draw budgets: GL uploads denser; CPU stamps DRAW_SOMA_MAX.
        gl = getattr(self, "_gl_tissue", None)
        if bool(getattr(self, "use_gl_tissue", False)) and gl is not None and gl.ok:
            budget = min(DRAW_SOMA_MAX * 4, 60000)
        else:
            budget = int(getattr(self, "draw_soma_budget", DRAW_SOMA_MAX))
            budget = min(max(1, budget), DRAW_SOMA_MAX)
        take = self._apply_draw_thin(self.points_cls, budget)
        if isinstance(take, slice):
            idx = np.arange(n, dtype=np.int32)
        else:
            idx = np.asarray(take, dtype=np.int32)
        mask = getattr(self, "_scope_mask", None)
        if (
            getattr(self, "scope_on", False)
            and mask is not None
            and len(mask) == n
            and len(idx)
        ):
            return int(np.count_nonzero(mask[idx]))
        return int(len(idx))

    def _set_density_from_x(self, idx: int, x: int, track: pygame.Rect) -> None:
        if idx < 0 or idx >= len(self.class_names):
            return
        t = (float(x) - float(track.x)) / max(1.0, float(track.w))
        self.draw_density[self.class_names[idx]] = float(np.clip(t, 0.0, 1.0))
        self._legend_blit_key = None

    def _handle_thin_legend(self, kind: str, idx: int, pos: tuple[int, int]) -> bool:
        """Handle thinning / layer legend hits. True if consumed."""
        if kind == "side_dock_toggle":
            self._toggle_dock()
            return True
        if kind == "side_panel_toggle":
            if 0 <= idx < len(SIDE_PANELS):
                pid = SIDE_PANELS[idx]
                self.side_panel_mini[pid] = not bool(
                    self.side_panel_mini.get(pid, False)
                )
                self._legend_blit_key = None
            return True
        if kind == "legend_tab":
            # Legacy: map old tabs → expand that side panel.
            if 0 <= idx < len(SIDE_PANELS):
                pid = SIDE_PANELS[idx]
                self.side_panel_mini[pid] = False
                self.legend_tab = pid
                self._legend_blit_key = None
            return True
        if kind == "thin_toggle":
            self.draw_thin_open = not self.draw_thin_open
            return True
        if kind == "thin_mode":
            modes = ("auto", "manual", "off")
            if 0 <= idx < len(modes):
                self.draw_thin_mode = modes[idx]
                self.draw_thin_open = False
                if self.draw_thin_mode == "manual":
                    self._ensure_draw_density()
            return True
        if kind == "thin_slider":
            track = self._density_slider_rects.get(idx)
            if track is not None:
                self._density_drag = idx
                self._set_density_from_x(idx, pos[0], track)
            return True
        if kind == "basin_form":
            track = self._basin_form_rect
            if track is not None:
                self._basin_form_drag = True
                self._set_basin_form_from_x(pos[0], track)
            return True
        if kind == "light_lamp":
            self.lamp_on = not self.lamp_on
            self.light_enabled = bool(self.lamp_on)
            if self.lamp_on:
                self._update_lamp_xyz()
            self._save_ui_state()
            return True
        if kind == "light_sky":
            track = self._light_slider_rects.get("sky")
            if track is not None:
                self._light_drag = "sky"
                self._set_light_slider_from_x("sky", pos[0], track)
            return True
        if kind == "light_speed":
            track = self._light_slider_rects.get("speed")
            if track is not None:
                self._light_drag = "speed"
                self._set_light_slider_from_x("speed", pos[0], track)
                return True
        if kind == "view_detail":
            track = self._view_detail_rect
            if track is not None:
                self._view_detail_drag = True
                self._set_view_detail_from_x(pos[0], track)
                return True
        if kind == "ground_on":
            self.ground_on = not self.ground_on
            self._ground_cache_key = None
            self._save_ui_state()
            return True
        if kind == "world_swing":
            self.world_swing_on = not self.world_swing_on
            self._ground_cache_key = None
            self._ground_blit_key = None
            self._compass_blit_key = None
            self._update_lamp_xyz()
            self._save_ui_state()
            return True
        if kind == "world_swing_amp":
            track = self._light_slider_rects.get("swing_amp")
            if track is not None:
                self._light_drag = "swing_amp"
                self._set_light_slider_from_x("swing_amp", pos[0], track)
            return True
        if kind == "world_swing_hz":
            track = self._light_slider_rects.get("swing_hz")
            if track is not None:
                self._light_drag = "swing_hz"
                self._set_light_slider_from_x("swing_hz", pos[0], track)
            return True
        if kind == "ground_relief":
            track = self._ground_relief_rect
            if track is not None:
                self._ground_relief_drag = True
                self._set_ground_relief_from_x(pos[0], track)
                self._legend_blit_key = None
            return True
        if kind == "prow":
            targets = getattr(self, "_prow_targets", None) or []
            if 0 <= idx < len(targets):
                row, device = targets[idx]
                ok = bool(handle_row_click(self, row, device))
                self._devices_dirty = True
                # Toggle/action may change Port/Learn flags — bust legend cache.
                self._legend_blit_key = None
                return ok
            return False
        return False

    def _sync_activity_view(self) -> None:
        self.soma_activity = self.fly.soma_activity

    def _invalidate_soma_lut(self) -> None:
        self._soma_lut = None
        self._soma_lut_key = None

    def _soma_id_lut(self) -> dict[int, int]:
        """Cached neuron-id → soma row index (avoid rebuilding 140k dict/frame)."""
        ids = self.fly.soma_ids
        key = (id(ids), int(len(ids)))
        if self._soma_lut is not None and self._soma_lut_key == key:
            return self._soma_lut
        lut = {int(i): k for k, i in enumerate(ids.tolist())}
        self._soma_lut = lut
        self._soma_lut_key = key
        return lut

    def _soma_indices_for_ids(self, nids) -> list[int]:
        """Map neuron ids → row indices; contiguous ids use identity."""
        ids = self.fly.soma_ids
        n = len(ids)
        if n == 0:
            return []
        contiguous = (
            hasattr(self.fly, "_activity_ids_contiguous")
            and self.fly._activity_ids_contiguous()
        )
        out: list[int] = []
        if contiguous:
            for nid in nids:
                i = int(nid)
                if 0 <= i < n:
                    out.append(i)
            return out
        lut = self._soma_id_lut()
        for nid in nids:
            k = lut.get(int(nid))
            if k is not None:
                out.append(int(k))
        return out

    def _basin_form_knobs(self) -> dict:
        """Map basin_form∈[0,1] → agglomeration knobs (few large ↔ many small)."""
        t = float(np.clip(self.basin_form, 0.0, 1.0))
        return {
            "max_modes": int(round(1 + t * 11)),  # 1..12
            "min_frac": 0.12 - 0.07 * t,  # 0.12..0.05
            "min_frac_afferent": 0.09 - 0.06 * t,  # 0.09..0.03
            "sep_factor": 1.40 - 0.25 * t,  # 1.40..1.15
            "sep_factor_afferent": 1.35 - 0.22 * t,  # 1.35..1.13
            "vis_scale": float(BASIN_VIS_SCALE) * (1.15 - 0.35 * t),
        }

    def _set_basin_form_from_x(self, x: int, track: pygame.Rect) -> None:
        t = (float(x) - float(track.x)) / max(1.0, float(track.w))
        new_v = float(np.clip(t, 0.0, 1.0))
        if abs(new_v - self.basin_form) < 1e-4:
            return
        self.basin_form = new_v
        self._basin_cache_key = None
        # Shape only — visibility is Super-class B focus.

    def _set_light_slider_from_x(
        self, key: str, x: int, track: pygame.Rect
    ) -> None:
        t = float(np.clip((float(x) - float(track.x)) / max(1.0, float(track.w)), 0.0, 1.0))
        if key == "sky":
            self.sky_fill = t
            self.light_enabled = True
        elif key == "speed":
            self.lamp_speed = t * float(LAMP_SPEED_MAX)  # 0..max rev/s
            self.light_enabled = True
        elif key == "swing_amp":
            self.world_swing_amp = t * float(WORLD_SWING_AMP_MAX)
            self.world_swing_on = True
            self._ground_cache_key = None
            self._ground_blit_key = None
            self._compass_blit_key = None
            self._update_lamp_xyz()
        elif key == "swing_hz":
            self.world_swing_hz = t * float(WORLD_SWING_HZ_MAX)
            self.world_swing_on = True

    def _set_ground_relief_from_x(self, x: int, track: pygame.Rect) -> None:
        t = float(np.clip((float(x) - float(track.x)) / max(1.0, float(track.w)), 0.0, 1.0))
        if abs(t - self.ground_relief) < 1e-4:
            return
        self.ground_relief = t
        self.ground_on = True
        self.light_enabled = True
        self._ground_cache_key = None

    def _view_detail_knobs(self) -> dict:
        """Map View Detail∈[0,1] → sim cadence + draw/overlay budgets."""
        t = float(np.clip(getattr(self, "view_detail", 1.0), 0.0, 1.0))
        base_hz = float(np.clip(getattr(self, "adult_sim_hz", 20.0) or 20.0, 5.0, 60.0))
        lo_hz = float(VIEW_DETAIL_SIM_HZ_LO)
        sim_hz = float(lo_hz + (base_hz - lo_hz) * t)
        draw_mul = float(VIEW_DETAIL_DRAW_LO + (1.0 - VIEW_DETAIL_DRAW_LO) * t)
        hops_cap = max(1, int(getattr(self, "loop_hops", 1) or 1))
        hops = max(1, int(round(1 + (hops_cap - 1) * t)))
        learn_base = max(1, int(getattr(self, "learn_every", 8) or 8))
        learn_hi = max(learn_base, int(VIEW_DETAIL_LEARN_EVERY_HI))
        learn_every = max(1, int(round(learn_hi + (learn_base - learn_hi) * t)))
        return {
            "t": t,
            "sim_hz": sim_hz,
            "draw_mul": draw_mul,
            "hops": hops,
            "learn_every": learn_every,
            "overlay": t,
            "label": (
                f"sim {sim_hz:.0f}Hz · draw {int(round(draw_mul * 100))}% · "
                f"{'overlays full' if t >= 0.85 else ('overlays soft' if t >= 0.35 else 'overlays low')}"
            ),
        }

    def _draw_cap(self, cap: int) -> int:
        """Scale a draw budget by View Detail (floor so tissue never vanishes)."""
        mul = float(self._view_detail_knobs()["draw_mul"])
        return max(64, int(float(cap) * mul))

    def _set_view_detail_from_x(self, x: int, track: pygame.Rect) -> None:
        t = float(
            np.clip((float(x) - float(track.x)) / max(1.0, float(track.w)), 0.0, 1.0)
        )
        if abs(t - float(getattr(self, "view_detail", 1.0))) < 1e-4:
            return
        self.view_detail = t
        kn = self._view_detail_knobs()
        self.draw_soma_budget = self._draw_cap(DRAW_SOMA_MAX)
        # Bust tissue / legend caches so density change is immediate.
        self._gl_geom_key = None
        self._legend_blit_key = None
        self._tissue_blit_key = None
        if hasattr(self, "_set_io_status"):
            self._set_io_status(f"View detail · {kn['label']}")

    def _refresh_basins(self, *, focus: set[int] | None = None) -> None:
        """Recompute large-class attractor wells when fly or focus set changes."""
        n_proc = len(getattr(self.fly, "atlas_process_xyz", ()))
        n_soma = len(self.points_xyz)
        foc = frozenset(focus) if focus else frozenset()
        knobs = self._basin_form_knobs()
        form_q = round(float(self.basin_form), 3)
        # Pose of live devices must NOT invalidate this — class_basin_fields is
        # multi-second work. Device cups/turret have their own overlays; wells
        # track tissue topology, not per-frame aim.
        key = (
            id(self.fly),
            n_proc,
            n_soma,
            foc,
            len(self.class_names),
            form_q,
            "basin_v12",
        )
        if key == self._basin_cache_key:
            return
        common = dict(
            top_n=BASIN_TOP_N,
            max_modes=knobs["max_modes"],
            min_frac=knobs["min_frac"],
            min_frac_afferent=knobs["min_frac_afferent"],
            sep_factor=knobs["sep_factor"],
            sep_factor_afferent=knobs["sep_factor_afferent"],
        )
        cls_f, xyz, radii, n_pts = self.fly.class_basin_fields(
            min_count=BASIN_MIN_PTS,
            class_filter=set(foc) if foc else None,
            **common,
        )
        # With focus but no basins (tiny classes): still show focused if any points.
        if foc and len(cls_f) == 0:
            cls_f, xyz, radii, n_pts = self.fly.class_basin_fields(
                min_count=1,
                class_filter=set(foc),
                **common,
            )
        self._basin_cls = cls_f
        self._basin_xyz = xyz
        self._basin_r = radii
        self._basin_n = n_pts
        self._basin_cache_key = key
        self._basin_blit_key = None  # geometry changed → force layer redraw

    def _sync_soma_derived(self) -> None:
        self.fly.rebuild_caches()
        self._invalidate_soma_lut()
        self._sync_from_fly()

    def _ensure_atlas_recipe(self) -> bool:
        """Attach malecns_edges.npz recipe if the live fly has none."""
        if len(self.fly.recipe_pre) > 0:
            return True
        edges_path = Path(SOMAS_PATH).resolve().parent / "malecns_edges.npz"
        if not edges_path.is_file():
            return False
        # Need body ids aligned with soma_ids (atlas only; device ids stay −1).
        if len(self.fly.soma_body_id) != len(self.fly.soma_ids):
            if SOMAS_PATH.is_file():
                data = np.load(SOMAS_PATH, allow_pickle=True)
                if "body_id" in data.files:
                    body = data["body_id"].astype(np.int64)
                    ids = np.asarray(self.fly.soma_ids, dtype=np.int64)
                    out = np.full(len(ids), -1, dtype=np.int64)
                    if len(body):
                        # Contiguous MaleCNS: neuron id == atlas row for id < n_atlas.
                        # Device neurons (added packs) sit past len(body) → −1.
                        valid = (ids >= 0) & (ids < len(body))
                        out[valid] = body[ids[valid]]
                    self.fly.soma_body_id = out
        if len(self.fly.soma_body_id) != len(self.fly.soma_ids):
            return False
        if not np.any(self.fly.soma_body_id >= 0):
            return False
        try:
            ed = np.load(edges_path, allow_pickle=True)
            n = self.fly.set_recipe_from_body_edges(ed["pre"], ed["post"], ed["weight"])
        except OSError:
            return False
        return n > 0

    def _arm_growth_channel(self, *, restart_if_idle: bool = False) -> int:
        """Spawn tips for all atlas recipe set→set classes (automatic genome quota).

        Adult / settled wiring is never re-armed here — use Reset → rising.
        `restart_if_idle` is kept for call-site compat; idle tips settle to adult.
        """
        del restart_if_idle  # settled → adult via _settle_if_idle, not wipe+rearm
        if self.fly.is_adult():
            return 0
        if self._settle_if_idle():
            return 0
        if not self.class_names or not self.fly.neurons:
            return 0
        if len(self.fly.recipe_pre) == 0 and not self._ensure_atlas_recipe():
            self._set_io_status(
                "No atlas recipe — Import MaleCNS (needs malecns_edges.npz)"
            )
            return 0
        sources = list(self.class_names)
        targets = list(self.class_names)
        n = self.fly.ensure_growth_tips(sources, targets)
        n_oc = self._arm_ocelli_tips()
        n_eye = self._arm_eye_tips()
        n_jo = self._arm_jo_tips()
        n_tu = self._arm_turret_tips()
        n = int(n) + int(n_oc) + int(n_eye) + int(n_jo) + int(n_tu)
        if n == 0:
            q = (
                int(self.fly.genome_quota.sum())
                if self.fly.genome_quota.size
                else 0
            )
            self._set_io_status(
                f"No tips to arm (quota {q:,}, atlas {len(self.fly.recipe_pre):,} edges)"
            )
        return n

    def _devices_need_fuse(self) -> bool:
        """True if any enabled device site is not yet fused to the fly."""
        if self.ocelli_on and self.ocelli_groups:
            if not all(self._ocelli_connected_mask()):
                return True
        for d in compound_eyes.iter_eye_devices(self):
            if d.enabled and d.terminals:
                if not all(self._eye_connected_mask(d.name)):
                    return True
        for d in johnston_organ.iter_jo_devices(self):
            if d.enabled and d.terminals:
                if not all(self._jo_connected_mask(d.name)):
                    return True
        if self.turret_on and any(self.turret_motor_groups):
            if not all(self._turret_connected_mask()):
                return True
        return False

    def _device_owner_has_open_tip(self, owner_id: int) -> bool:
        if len(self.fly.growth_owner) == 0:
            return False
        m = self.fly.growth_owner == int(owner_id)
        if not np.any(m):
            return False
        st = self.fly.growth_status[m]
        return bool(
            np.any((st == STATUS_GROWING) | (st == STATUS_SEEKING))
        )

    def _arm_fuse_tips(self) -> int:
        """Arm only missing device↔CNS tips (no atlas recipe re-grow)."""
        if not self.class_names or not self.fly.neurons:
            return 0
        # Devices must exist; create packs/clamps if needed.
        if self.ocelli_on:
            self._ensure_ocelli_neurons()
        self._ensure_eye_neurons()
        self._ensure_jo_neurons()
        if self.turret_on:
            self._ensure_turret_neurons()
        n = 0
        n += int(self._arm_ocelli_tips(missing_only=True))
        n += int(self._arm_eye_tips(missing_only=True))
        n += int(self._arm_jo_tips(missing_only=True))
        n += int(self._arm_turret_tips(missing_only=True))
        return n

    def _clear_device_cns_fuse(self) -> dict:
        """Strip device↔CNS Fuse links; keep packs and device-local wiring.

        Old Fuse tips often landed on nearest soma (not AN/DN). Call before
        re-arming Fuse into ascending/descending tracts.
        """
        device = set(self._ocelli_flat_ids()) | set(self._turret_flat_ids())
        for d in compound_eyes.iter_eye_devices(self):
            device.update(d.flat_ids())
        for d in johnston_organ.iter_jo_devices(self):
            device.update(d.flat_ids())
        out = {"tips": 0, "axons": 0, "dendrites": 0, "gains": 0}
        if not device or not self.fly.neurons:
            return out

        # 1) Drop growth tips owned by device neurons.
        if len(self.fly.growth_owner):
            owners = np.asarray(sorted(device), dtype=np.int32)
            keep = ~np.isin(self.fly.growth_owner, owners)
            dropped = int((~keep).sum())
            if dropped:
                out["tips"] = dropped
                self.fly.growth_owner = self.fly.growth_owner[keep]
                self.fly.growth_soma = self.fly.growth_soma[keep]
                self.fly.growth_tip = self.fly.growth_tip[keep]
                self.fly.growth_status = self.fly.growth_status[keep]
                self.fly.growth_partner = self.fly.growth_partner[keep]
                self.fly.growth_goal = self.fly.growth_goal[keep]
                self.fly.growth_goal_xyz = self.fly.growth_goal_xyz[keep]
                self.fly.growth_basin_r = self.fly.growth_basin_r[keep]
                self.fly.growth_seek_t = self.fly.growth_seek_t[keep]
                self.fly.growth_seek_limit = self.fly.growth_seek_limit[keep]
                self.fly.growth_cls = self.fly.growth_cls[keep]
                self.fly.growth_poly = self.fly.growth_poly[keep]
                self.fly.growth_poly_n = self.fly.growth_poly_n[keep]
                if len(self.fly._growth_materialized) == len(keep):
                    self.fly._growth_materialized = self.fly._growth_materialized[keep]
                else:
                    self.fly._growth_materialized = np.zeros(
                        len(self.fly.growth_owner), dtype=bool
                    )
                if len(getattr(self.fly, "growth_bundle_id", ())) == len(keep):
                    self.fly.growth_bundle_id = self.fly.growth_bundle_id[keep]

        # 2) Axons/dendrites: keep device↔device; drop device↔CNS.
        for nid, neu in list(self.fly.neurons.items()):
            nid = int(nid)
            if nid in device:
                kept_ax = []
                for ax in neu.axons:
                    pid = ax.partner_id
                    if pid is not None and int(pid) in device:
                        kept_ax.append(ax)
                    else:
                        out["axons"] += 1
                neu.axons = kept_ax
                # Device dendrites that linked back from CNS materialize are rare;
                # keep only those whose partner is also a device.
                kept_den = []
                for den in neu.dendrites:
                    pid = den.partner_id
                    if pid is not None and int(pid) in device:
                        kept_den.append(den)
                    else:
                        out["dendrites"] += 1
                neu.dendrites = kept_den
            else:
                kept_ax = []
                for ax in neu.axons:
                    pid = ax.partner_id
                    if pid is not None and int(pid) in device:
                        out["axons"] += 1
                    else:
                        kept_ax.append(ax)
                neu.axons = kept_ax
                kept_den = []
                for den in neu.dendrites:
                    pid = den.partner_id
                    if pid is not None and int(pid) in device:
                        out["dendrites"] += 1
                    else:
                        kept_den.append(den)
                neu.dendrites = kept_den

        # 3) Plastic gains on any edge that touched a device.
        gains = getattr(self.fly, "contact_gain", None)
        if isinstance(gains, dict) and gains:
            drop_keys = [
                k
                for k in list(gains.keys())
                if int(k[0]) in device or int(k[1]) in device
            ]
            for k in drop_keys:
                gains.pop(k, None)
            out["gains"] = len(drop_keys)

        # 4) Re-apply pack-local wires (bus / mesh) after axon cull.
        if self.ocelli_on:
            self._wire_ocelli_device()
        if self.turret_on:
            self._wire_turret_device()
        self.fly.rebuild_caches()
        self._sync_from_fly()
        self._fuse_running = False
        return out

    def _on_fuse_start(self) -> None:
        """Enter Fuse: clear old device↔CNS links, grow into AN/DN tracts."""
        if not self.fly.neurons:
            self._set_io_status("No fly loaded")
            return
        if not (self.ocelli_on or self.turret_on):
            self._set_io_status("No devices enabled")
            return
        self._activity_running = False
        # Always drop prior Fuse (nearest-soma era) before arming neck tips.
        cleared = self._clear_device_cns_fuse()
        self._ensure_atlas_recipe()
        self.refresh_stereo_plan()
        self.fly.enter_rising()
        n = self._arm_fuse_tips()
        if n <= 0:
            # Maybe already open tips in flight.
            flat = set(self._ocelli_flat_ids()) | set(self._turret_flat_ids())
            active = False
            if len(self.fly.growth_owner) and flat:
                m = np.isin(self.fly.growth_owner, list(flat))
                if np.any(m):
                    st = self.fly.growth_status[m]
                    active = bool(
                        np.any((st == STATUS_GROWING) | (st == STATUS_SEEKING))
                    )
            if not active:
                self.fly.enter_adult(abandon_open=False)
                self._fuse_running = False
                self.sim_running = False
                self._set_io_status(
                    f"Fuse · nothing to grow "
                    f"(cleared tips {cleared['tips']} ax {cleared['axons']})"
                )
                return
        self._fuse_running = True
        self.sim_running = True
        oc = sum(1 for c in self._ocelli_connected_mask() if c) if self.ocelli_on else 0
        tu = sum(1 for c in self._turret_connected_mask() if c) if self.turret_on else 0
        self._set_io_status(
            f"Fuse · reset · armed {n:,} AN/DN tips · "
            f"ocelli {oc}/3 · turret {tu}/{int(TURRET_SITES)} · "
            f"cleared ax {cleared['axons']}"
        )

    def _settle_if_idle(self) -> bool:
        """If rising growth has settled (or loaded contacts exist) → adult.

        Returns True when the fly is adult after this call.
        Fuse mode only waits on device tips — atlas leftovers don't block.
        """
        if self.fly.is_adult():
            self._fuse_running = False
            return True
        if self._fuse_running:
            flat = set(self._ocelli_flat_ids()) | set(self._turret_flat_ids())
            if flat and len(self.fly.growth_owner):
                m = np.isin(
                    self.fly.growth_owner,
                    np.asarray(list(flat), dtype=np.int32),
                )
                if np.any(m):
                    st = self.fly.growth_status[m]
                    if np.any(
                        (st == STATUS_GROWING) | (st == STATUS_SEEKING)
                    ):
                        return False
            elif self._devices_need_fuse():
                # Armed nothing / tips missing — keep Fuse open.
                return False
        elif len(self.fly.growth_owner):
            st = self.fly.neurite_stats()
            if int(st.get("active", 0)) > 0:
                return False
        else:
            # Empty growth field: only auto-adult if contacts already exist (load).
            n_edges = len(self.fly.grown_contact_edges()[0])
            if n_edges == 0:
                return False
        was_fuse = bool(self._fuse_running)
        self.sim_running = False
        self._fuse_running = False
        self.fly.enter_adult(abandon_open=True)
        self._ensure_atlas_recipe()
        yld = self._recipe_yield()
        head, _ = self._yield_headline(yld)
        n_edges = len(self.fly.grown_contact_edges()[0])
        n_hop = len(self.fly.activity_graph_edges()[0])
        if was_fuse:
            oc = (
                sum(1 for c in self._ocelli_connected_mask() if c)
                if self.ocelli_on
                else 0
            )
            tu = (
                sum(1 for c in self._turret_connected_mask() if c)
                if self.turret_on
                else 0
            )
            self._set_io_status(
                f"Adult · fused ocelli {oc}/3 · turret {tu}/{int(TURRET_SITES)} · "
                f"{n_edges:,} Fuse · hop {n_hop:,}"
                + (
                    " · Save .fly to keep Fuse on reopen"
                    if (not self._last_model or self._last_model == "malecns")
                    else ""
                )
            )
            self._save_ui_state()
        else:
            self._set_io_status(
                f"Adult · Fuse {n_edges:,} · hop {n_hop:,} · {head}"
            )
        return True

    def _projection_name(self) -> str:
        if getattr(self, "show_port_schema", False):
            return "AI Brain"
        if getattr(self, "show_brain_schema", False):
            return "CNS Brain"
        if getattr(self, "show_grow_schema", False):
            return "CNS Growing"
        if getattr(self, "show_research_schema", False):
            return "Research map"
        if getattr(self, "_nav_preset", None) == "side":
            return "Side"
        if abs(self.view_angle_x) < 0.1 and abs(self.view_angle_y) < 0.1:
            return "View X"
        if abs(self.view_angle_x - math.pi / 2) < 0.1:
            return "View Y"
        if abs(self.view_angle_y - math.pi / 2) < 0.1 and abs(self.view_angle_x) < 0.1:
            return "View Z"
        if self._is_default_side_angles(self.view_angle_x, self.view_angle_y):
            return "Side"
        return "3D"

    def _schematic_active(self) -> bool:
        return bool(
            getattr(self, "show_port_schema", False)
            or getattr(self, "show_brain_schema", False)
            or getattr(self, "show_grow_schema", False)
            or getattr(self, "show_research_schema", False)
        )

    def _clear_schematic_selection(self) -> None:
        self._port_selected = None
        self._port_hits = []
        self._brain_selected = None
        self._brain_hits = []
        self._grow_selected = None
        self._grow_hits = []
        self._research_selected = None
        self._research_hits = []

    def set_view_port_schema(self, on: bool | None = None) -> None:
        """Toggle AI Brain iterative block diagram (replaces 3D while active)."""
        if on is None:
            self.show_port_schema = not bool(self.show_port_schema)
        else:
            self.show_port_schema = bool(on)
        if self.show_port_schema:
            self._stash_side_view_memory()
            self.show_brain_schema = False
            self.show_grow_schema = False
            self.show_research_schema = False
            self._clear_schematic_selection()
            self._port_stage = "sense"
            was_open = not bool(getattr(self, "legend_mini", True))
            self.legend_mini = True
            if was_open:
                self._layout(self.width, self.height)
            # Ensure AI Brain object exists so meters can refresh.
            if getattr(self, "sensor_port_nn", None) is None:
                try:
                    from sensor_port_nn import SensorPortNN

                    self.sensor_port_nn = SensorPortNN(lr=0.02)
                except Exception:
                    pass
        else:
            self._port_selected = None
            self._port_hits = []

    def set_view_brain_schema(self, on: bool | None = None) -> None:
        """Toggle CNS Brain hop/carve block diagram (replaces 3D while active)."""
        if on is None:
            self.show_brain_schema = not bool(self.show_brain_schema)
        else:
            self.show_brain_schema = bool(on)
        if self.show_brain_schema:
            self._stash_side_view_memory()
            self.show_port_schema = False
            self.show_grow_schema = False
            self.show_research_schema = False
            self._clear_schematic_selection()
            self._brain_stage = "sense"
            was_open = not bool(getattr(self, "legend_mini", True))
            self.legend_mini = True
            if was_open:
                self._layout(self.width, self.height)
        else:
            self._brain_selected = None
            self._brain_hits = []

    def set_view_grow_schema(self, on: bool | None = None) -> None:
        """Toggle CNS Growing tip/synapse policy diagram (replaces 3D while active)."""
        if on is None:
            self.show_grow_schema = not bool(self.show_grow_schema)
        else:
            self.show_grow_schema = bool(on)
        if self.show_grow_schema:
            self._stash_side_view_memory()
            self.show_port_schema = False
            self.show_brain_schema = False
            self.show_research_schema = False
            self._clear_schematic_selection()
            self._grow_stage = "recipe"
            was_open = not bool(getattr(self, "legend_mini", True))
            self.legend_mini = True
            if was_open:
                self._layout(self.width, self.height)
        else:
            self._grow_selected = None
            self._grow_hits = []

    def set_view_research_schema(self, on: bool | None = None) -> None:
        """Toggle research map flowchart (project state / branches)."""
        if on is None:
            self.show_research_schema = not bool(self.show_research_schema)
        else:
            self.show_research_schema = bool(on)
        if self.show_research_schema:
            self._stash_side_view_memory()
            self.show_port_schema = False
            self.show_brain_schema = False
            self.show_grow_schema = False
            self._clear_schematic_selection()
            self._research_stage = "now"
            was_open = not bool(getattr(self, "legend_mini", True))
            self.legend_mini = True
            if was_open:
                self._layout(self.width, self.height)
        else:
            self._research_selected = None
            self._research_hits = []

    def _layout(self, w: int, h: int) -> None:
        self.width = w
        self.height = h
        self.top_h = TOP_BAR_H
        self.tool_w = TOOL_W
        self.status_h = STATUS_H
        self.panel_w = (
            PANEL_COLLAPSED_W
            if self.panel_collapsed
            else max(PANEL_COLLAPSED_W, w // PANEL_OPEN_FRAC)
        )
        self.panel_x = int(self.tool_w)
        self.dock_w = 0 if bool(getattr(self, "legend_mini", True)) else DOCK_W
        self.dock_x = max(0, w - int(self.dock_w))
        self.view_x = int(self.panel_x + self.panel_w)
        self.view_y = int(self.top_h)
        self.view_w = max(1, w - self.view_x - int(self.dock_w))
        self.view_h = max(1, h - int(self.top_h) - int(self.status_h))
        # blit_array requires 24/32-bit; convert() matches the display format.
        self._view_surf = pygame.Surface((self.view_w, self.view_h)).convert()
        self._tissue_blit_key = None
        self._tissue_blit_surf = None
        self._legend_blit_key = None

    def _toggle_panel_collapsed(self) -> None:
        self.panel_collapsed = not self.panel_collapsed
        self._layout(self.width, self.height)

    def _toggle_dock(self) -> None:
        self.legend_mini = not bool(self.legend_mini)
        self._layout(self.width, self.height)

    def _fit_scale(self) -> None:
        if self.n_shown == 0:
            self.scale_3d = 48.0
            return
        sx, sy = project_3d_batch(
            self.points_xyz,
            self.view_angle_x,
            self.view_angle_y,
            0.0,
            0.0,
            1.0,
        )
        span = max(float(sx.max() - sx.min()), float(sy.max() - sy.min()), 1e-3)
        usable_h = max(1, self.view_h - NAV_PANEL_H)
        # Leave margin for compass; a bit more when ground square is on.
        fit_frac = 0.42 if getattr(self, "ground_on", False) else 0.58
        target = fit_frac * min(self.view_w, usable_h)
        self.scale_3d = float(np.clip(target / span, self.scale_3d_min, self.scale_3d_max))

    def _adopt_fly(
        self,
        fly: Fly,
        status: str,
        *,
        refit: bool = True,
        model_path: Path | None = None,
        from_malecns: bool = False,
        remember: bool = True,
    ) -> None:
        """Install working fly and refresh view caches."""
        self.fly = fly
        self._invalidate_soma_lut()
        self._device_funnel_blit_key = None
        self._device_funnel_blit_surf = None
        self._tissue_blit_key = None
        self._tissue_blit_surf = None
        self._fuse_mask_cache = None
        self._cg_partner_idx_cache = None
        self.ocelli_ids = []  # device neurons are per-fly
        self.ocelli_groups = [[], [], []]
        self.ocelli_terminals = []
        self.turret_motor_groups = [[], [], [], []]
        self.turret_sensory_groups = [[], [], [], []]
        self.turret_motor_terminals = []
        self.turret_sensory_terminals = []
        self.turret_bus_terminal = None
        # Restore device id map BEFORE ensure_* so Fuse axons are not orphaned.
        snap = getattr(fly, "shell_meta", None) or {}
        applied = False
        if isinstance(snap, dict) and snap:
            applied = self._apply_device_snapshot(snap)
            if isinstance(snap.get("motor_socket"), dict):
                self._apply_motor_socket_snapshot(snap["motor_socket"])
        if not applied and UI_STATE_PATH.is_file():
            # Fallback: last UI snapshot (same session / older .fly without meta).
            try:
                raw = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("shell_devices"), dict):
                    applied = self._apply_device_snapshot(raw["shell_devices"])
                if isinstance(raw, dict) and isinstance(raw.get("motor_socket"), dict):
                    self._apply_motor_socket_snapshot(raw["motor_socket"])
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        if not applied:
            self._recover_devices_from_fly()
        self._sync_from_fly()
        if self.ocelli_on:
            self._ensure_ocelli_neurons()
        self._ensure_eye_neurons()
        self._ensure_jo_neurons()
        if self.turret_on:
            self._ensure_turret_neurons()
        # Re-lock hop/draw to funnel wells if UI asked for funnel model.
        if getattr(self, "scope_on", False):
            if not self.apply_training_scope():
                self.scope_on = False
                self.fly.hop_allow_ids = np.zeros((0,), dtype=np.int32)
        # Attach atlas edge recipe when missing (e.g. old .fly + malecns_edges.npz).
        if self._ensure_atlas_recipe() and "recipe" not in status.lower():
            status = f"{status} · recipe {len(self.fly.recipe_pre):,}"
        if len(self.fly.recipe_pre) and (
            self.fly.genome_quota.size == 0 or int(self.fly.genome_quota.sum()) == 0
        ):
            self.fly.rebuild_genome_from_atlas()
        self._prune_and_default_focus(seed_if_blank=False)
        if self.n_shown:
            span = float(np.ptp(self.points_xyz, axis=0).max())
            self._coord_span = max(span, 1e-3)
        else:
            self._coord_span = 1.0
        if refit:
            self._fit_scale()
        self.sim_running = False
        self._activity_running = False
        # Loaded wired flies (or settled tip fields) must not stay re-armable rising.
        if self._settle_if_idle() and "adult" not in status.lower():
            status = f"{status} · adult"
        # Annotate fuse state after bind/ensure.
        if self.fly.is_adult() and (self.ocelli_on or self.turret_on):
            oc = (
                sum(1 for c in self._ocelli_connected_mask() if c)
                if self.ocelli_on
                else 0
            )
            tu = (
                sum(1 for c in self._turret_connected_mask() if c)
                if self.turret_on
                else 0
            )
            if oc or tu:
                status = (
                    f"{status} · fused ocelli {oc}/3 · "
                    f"turret {tu}/{int(TURRET_SITES)}"
                )
        self._fly_io_status = status
        if remember:
            if from_malecns:
                self._remember_model("malecns")
            elif model_path is not None:
                self._remember_model(str(model_path.resolve()))
        # Cold auto-restore must not save here: camera/funnels are still defaults
        # until _load_ui_state runs (see __init__).

    def _import_malecns(self) -> None:
        if not SOMAS_PATH.is_file():
            self._set_io_status("Missing malecns_somas.npz — run build script", redraw=True)
            return
        self._set_io_status("Importing MaleCNS…", redraw=True)
        try:
            fly = load_fly_from_malecns(SOMAS_PATH, PROCESSES_PATH)
        except OSError as exc:
            self._set_io_status(f"Import failed: {exc}", redraw=True)
            return
        self._adopt_fly(
            fly,
            f"Atlas MaleCNS ({len(fly.neurons):,} n, "
            f"{len(fly.recipe_pre):,} recipe edges)",
            from_malecns=True,
        )
        self._draw()

    def _in_view(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return (
            self.view_x <= x < self.view_x + self.view_w
            and self.view_y <= y < self.view_y + self.view_h
        )

    def _hit_nav_button(self, pos: tuple[int, int]) -> Optional[str]:
        for name, rect in self._nav_rects.items():
            if rect.collidepoint(pos):
                return name
        for name, rect in getattr(self, "_chrome_rects", {}).items():
            if rect.collidepoint(pos):
                return name
        return None

    def _hit_legend(self, pos: tuple[int, int]) -> tuple[str, int] | None:
        for rect, kind, idx in self._legend_hits:
            if rect.collidepoint(pos):
                return kind, idx
        return None

    def _active_class_indices(self) -> set[int]:
        return {
            i
            for i in range(len(self.class_names))
            if i < len(self.class_counts) and int(self.class_counts[i]) > 0
        }

    def _focus_set_for(self, kind: str) -> set[int]:
        return {
            "soma": self.focus_soma_indices,
            "neuron": self.focus_soma_indices,
            "graph": self.focus_graph_indices,
            "proc": self.focus_graph_indices,
            "axon": self.focus_axon_indices,
            "dendrite": self.focus_dendrite_indices,
            "tract": self.focus_tract_indices,
            "basin": self.focus_basin_indices,
        }.get(kind, self.focus_soma_indices)

    def _any_focus(self) -> bool:
        return bool(
            self.focus_soma_indices
            or self.focus_graph_indices
            or self.focus_axon_indices
            or self.focus_dendrite_indices
            or self.focus_tract_indices
            or self.focus_basin_indices
        )

    def _sync_focus_hint(self) -> None:
        if self.focus_hint_class is not None and (
            self.focus_hint_class not in self._focused_parents()
        ):
            self.focus_hint_class = next(iter(self._focused_parents()), None)

    def _toggle_focus(self, kind: str, idx: int) -> None:
        # Bulk / group actions (idx unused or class idx for "group").
        if kind.endswith("_col"):
            layer = kind[: -len("_col")]
            focus = self._focus_set_for(layer)
            active = self._active_class_indices()
            inter = focus & active
            if inter:
                # Any selection → clear column (hides that tissue layer).
                focus.clear()
            else:
                focus.update(active)
            if layer in ("graph", "proc"):
                self.focus_proc_indices = set(self.focus_graph_indices)
            self._sync_focus_hint()
            self._sync_layers_from_focus()
            return
        if kind.endswith("_all"):
            layer = kind[: -len("_all")]
            self._focus_set_for(layer).update(self._active_class_indices())
            if layer in ("graph", "proc"):
                self.focus_proc_indices = set(self.focus_graph_indices)
            self._sync_focus_hint()
            self._sync_layers_from_focus()
            return
        if kind.endswith("_none"):
            layer = kind[: -len("_none")]
            self._focus_set_for(layer).clear()
            if layer in ("graph", "proc"):
                self.focus_proc_indices.clear()
            self._sync_focus_hint()
            self._sync_layers_from_focus()
            return
        if kind == "group":
            # Name click: empty/partial → fill N+A+D; all three → clear.
            # Graph (G) and basins (B) stay explicit.
            layers = (
                self.focus_soma_indices,
                self.focus_axon_indices,
                self.focus_dendrite_indices,
            )
            all_on = all(idx in s for s in layers)
            if all_on:
                for s in layers:
                    s.discard(idx)
            else:
                for s in layers:
                    s.add(idx)
                self.focus_hint_class = idx
            self._sync_focus_hint()
            self._sync_layers_from_focus()
            return

        target = self._focus_set_for(kind)
        if idx in target:
            target.discard(idx)
        else:
            target.add(idx)
            self.focus_hint_class = idx
        if kind in ("graph", "proc"):
            self.focus_proc_indices = set(self.focus_graph_indices)
        self._sync_focus_hint()
        self._sync_layers_from_focus()

    def _focused_parents(self) -> set[int]:
        return (
            self.focus_soma_indices
            | self.focus_graph_indices
            | self.focus_axon_indices
            | self.focus_dendrite_indices
            | self.focus_tract_indices
            | self.focus_basin_indices
        )

    def _class_color(self, idx: int) -> tuple[int, int, int]:
        c = CLASS_COLORS[idx % len(CLASS_COLORS)]
        return int(c[0]), int(c[1]), int(c[2])

    def _derive_axon_color(self, idx: int) -> tuple[int, int, int]:
        """Dim class tint for axon legend swatches."""
        r, g, b = self._class_color(idx)
        return (int(r * 0.42), int(g * 0.42), int(b * 0.42))

    def _derive_dendrite_color(self, idx: int) -> tuple[int, int, int]:
        """Dimmer class tint for dendrite legend swatches."""
        r, g, b = self._class_color(idx)
        return (int(r * 0.32), int(g * 0.32), int(b * 0.32))

    def _tip_color_for_status(
        self, base: tuple[int, int, int], code: int
    ) -> tuple[int, int, int]:
        r, g, b = base
        if code == 1:  # growing — bright
            return (min(255, r + 40), min(255, g + 40), min(255, b + 40))
        if code == 2:  # seeking — warm shift
            return (min(255, r + 60), min(255, int(g * 0.9 + 30)), max(0, b - 20))
        if code == 3:  # connected — greenish lift
            return (max(0, r - 20), min(255, g + 50), max(0, b - 10))
        if code == 4:  # failed — reddish
            return (min(255, r + 80), max(0, g - 40), max(0, b - 40))
        return base

    def _recipe_yield(self) -> dict:
        """UI-facing Recipe Yield (global + per-class Out/In)."""
        return self.fly.recipe_yield()

    def _yield_verdict_color(self, verdict: str) -> tuple[int, int, int]:
        return {
            "CONNECTED": (120, 220, 150),
            "PARTIAL": (220, 190, 110),
            "FAILED": (220, 110, 110),
            "RUNNING": (170, 200, 230),
            "IDLE": (120, 120, 140),
        }.get(verdict, (160, 160, 180))

    def _yield_headline(self, yld: dict) -> tuple[str, tuple[int, int, int]]:
        """One-line Yield status for legend / panel / mini strip."""
        verdict = str(yld.get("verdict", "IDLE"))
        color = self._yield_verdict_color(verdict)
        armed = int(yld.get("armed", 0))
        if armed == 0:
            if len(self.fly.recipe_pre) == 0:
                return "Yield  —  no recipe", (180, 120, 100)
            return "Yield  —  Start to arm", (120, 120, 140)
        pct = float(yld.get("pct", 0.0))
        ok = int(yld.get("ok", 0))
        fail = int(yld.get("fail", 0))
        open_n = int(yld.get("open", 0))
        if verdict == "RUNNING":
            return (
                f"Yield  {pct:.0f}%  ·  {ok:,}/{armed:,}  ·  still {open_n:,}",
                color,
            )
        return (
            f"Yield  {verdict}  {pct:.0f}%  ·  {ok:,} ok / {fail:,} fail",
            color,
        )

    def _pulse(self) -> float:
        """Smooth 0..1 blink factor (~1.1 Hz)."""
        t = time.perf_counter() - self._t0
        return 0.5 + 0.5 * math.sin(t * math.tau * 1.1)

    def _request_exit(self) -> None:
        self.exit_confirm_modal.show()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_exit()
            elif event.type == pygame.VIDEORESIZE:
                self._reopen_display(event.w, event.h)
                self._fit_scale()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.exit_confirm_modal.is_visible():
                    self.exit_confirm_modal.hide()
                else:
                    self._request_exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if self.exit_confirm_modal.is_visible():
                    continue
                self._toggle_sim_run_pause()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.exit_confirm_modal.is_visible():
                    result = self.exit_confirm_modal.handle_click(
                        event.pos, self.width, self.height
                    )
                    if result == "yes":
                        self.exit_confirm_modal.hide()
                        self.running = False
                    elif result in ("no", "dismiss"):
                        self.exit_confirm_modal.hide()
                    continue

                if (
                    event.button == 1
                    and self.panel_toggle_rect is not None
                    and self.panel_toggle_rect.collidepoint(event.pos)
                ):
                    self._toggle_panel_collapsed()
                    continue

                if (
                    event.button == 1
                    and self._fps_badge_rect is not None
                    and self._fps_badge_rect.collidepoint(event.pos)
                ):
                    self._fps_panel_open = not self._fps_panel_open
                    if self._fps_panel_open:
                        self._learn_panel_open = False
                    continue

                if (
                    event.button == 1
                    and self._learn_badge_rect is not None
                    and self._learn_badge_rect.collidepoint(event.pos)
                ):
                    self._learn_panel_open = not self._learn_panel_open
                    if self._learn_panel_open:
                        self._fps_panel_open = False
                    continue

                if (
                    event.button == 1
                    and self._fps_panel_open
                    and self._fps_panel_rect is not None
                    and self._fps_panel_rect.collidepoint(event.pos)
                ):
                    continue

                if (
                    event.button == 1
                    and self._learn_panel_open
                    and self._learn_panel_rect is not None
                    and self._learn_panel_rect.collidepoint(event.pos)
                ):
                    hit = getattr(self, "_learn_rule_hit", None)
                    if hit is not None and hit.collidepoint(event.pos):
                        self._learn_rule_open = not bool(
                            getattr(self, "_learn_rule_open", False)
                        )
                    continue

                if (
                    event.button == 1
                    and self.legend_toggle_rect is not None
                    and self.legend_toggle_rect.collidepoint(event.pos)
                ):
                    self._toggle_dock()
                    continue

                if event.button == 1:
                    hit = self._hit_legend(event.pos)
                    if hit is not None:
                        kind, idx = hit
                        if self._handle_thin_legend(kind, idx, event.pos):
                            continue
                        self._toggle_focus(kind, idx)
                        continue

                if event.button == 1 and self._handle_control_panel_click(event.pos):
                    continue

                if not self._in_view(event.pos):
                    # Status / top chrome camera + schema hits still work.
                    if event.button == 1:
                        hit = self._hit_nav_button(event.pos)
                        if hit == "X":
                            self.set_view_x()
                            self._save_ui_state()
                        elif hit == "Y":
                            self.set_view_y()
                            self._save_ui_state()
                        elif hit == "Z":
                            self.set_view_z()
                            self._save_ui_state()
                        elif hit == "Side":
                            self.set_view_side()
                            self._save_ui_state()
                        elif hit == "Grow":
                            self.set_view_grow_schema()
                            self._save_ui_state()
                        elif hit == "AI":
                            self.set_view_port_schema()
                            self._save_ui_state()
                        elif hit == "Brain":
                            self.set_view_brain_schema()
                            self._save_ui_state()
                        elif hit == "Map":
                            self.set_view_research_schema()
                            self._save_ui_state()
                        elif hit == "Panels":
                            self._toggle_dock()
                    continue

                # Don't orbit/pan when interacting with any side panel
                side_rects = list(getattr(self, "_side_panel_rects", {}).values())
                if self.legend_panel_rect is not None:
                    side_rects.append(self.legend_panel_rect)
                if any(
                    r is not None and r.collidepoint(event.pos) for r in side_rects
                ) and event.button in (1, 2, 3):
                    continue

                if (
                    self._fps_panel_rect is not None
                    and self._fps_panel_rect.collidepoint(event.pos)
                    and event.button in (1, 2, 3)
                ):
                    continue

                if (
                    self._fps_badge_rect is not None
                    and self._fps_badge_rect.collidepoint(event.pos)
                    and event.button in (1, 2, 3)
                ):
                    continue

                if (
                    self._learn_panel_rect is not None
                    and self._learn_panel_rect.collidepoint(event.pos)
                    and event.button in (1, 2, 3)
                ):
                    continue

                if (
                    self._learn_badge_rect is not None
                    and self._learn_badge_rect.collidepoint(event.pos)
                    and event.button in (1, 2, 3)
                ):
                    continue

                if event.button == 1:
                    hit = self._hit_nav_button(event.pos)
                    if hit == "X":
                        self.set_view_x()
                        self._save_ui_state()
                    elif hit == "Y":
                        self.set_view_y()
                        self._save_ui_state()
                    elif hit == "Z":
                        self.set_view_z()
                        self._save_ui_state()
                    elif hit == "Side":
                        self.set_view_side()
                        self._save_ui_state()
                    elif hit == "Grow":
                        self.set_view_grow_schema()
                        self._save_ui_state()
                    elif hit == "AI":
                        self.set_view_port_schema()
                        self._save_ui_state()
                    elif hit == "Brain":
                        self.set_view_brain_schema()
                        self._save_ui_state()
                    elif hit == "Map":
                        self.set_view_research_schema()
                        self._save_ui_state()
                    elif getattr(self, "show_port_schema", False):
                        bid = hit_ai_brain(
                            event.pos,
                            self._port_hits,
                            origin=(self.view_x, self.view_y),
                        )
                        if bid is not None:
                            if bid.startswith("stage:"):
                                self._port_stage = bid.split(":", 1)[1]
                            elif self._port_selected == bid:
                                self._port_selected = None
                            else:
                                self._port_selected = bid
                        else:
                            # Click empty → advance stage (iterative walk)
                            self._port_stage = ai_brain_next_stage(
                                getattr(self, "_port_stage", "sense")
                            )
                    elif getattr(self, "show_brain_schema", False):
                        bid = hit_brain(
                            event.pos,
                            self._brain_hits,
                            origin=(self.view_x, self.view_y),
                        )
                        if bid is not None:
                            if bid.startswith("stage:"):
                                self._brain_stage = bid.split(":", 1)[1]
                            elif self._brain_selected == bid:
                                self._brain_selected = None
                            else:
                                self._brain_selected = bid
                        else:
                            self._brain_stage = brain_next_stage(
                                getattr(self, "_brain_stage", "sense")
                            )
                    elif getattr(self, "show_grow_schema", False):
                        bid = hit_grow(
                            event.pos,
                            self._grow_hits,
                            origin=(self.view_x, self.view_y),
                        )
                        if bid is not None:
                            if bid.startswith("stage:"):
                                self._grow_stage = bid.split(":", 1)[1]
                            elif self._grow_selected == bid:
                                self._grow_selected = None
                            else:
                                self._grow_selected = bid
                        else:
                            self._grow_stage = grow_next_stage(
                                getattr(self, "_grow_stage", "recipe")
                            )
                    elif getattr(self, "show_research_schema", False):
                        bid = hit_research(
                            event.pos,
                            self._research_hits,
                            origin=(self.view_x, self.view_y),
                        )
                        if bid is not None:
                            if bid.startswith("stage:"):
                                self._research_stage = bid.split(":", 1)[1]
                            elif self._research_selected == bid:
                                self._research_selected = None
                            else:
                                self._research_selected = bid
                        else:
                            self._research_stage = research_next_stage(
                                getattr(self, "_research_stage", "now")
                            )
                    continue

                if self._hit_nav_button(event.pos):
                    continue

                if self._schematic_active():
                    continue

                if event.button == 3:
                    self._dragging_orbit = True
                    self._last_mouse = event.pos
                elif event.button == 2:
                    self._dragging_pan = True
                    self._last_mouse = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._density_drag = None
                    self._density_drag_budget = False
                    self._basin_form_drag = False
                    if self._light_drag is not None:
                        self._light_drag = None
                        self._save_ui_state()
                    self._ground_relief_drag = False
                    if self._view_detail_drag:
                        self._view_detail_drag = False
                        self._save_ui_state()
                if event.button == 3:
                    if self._dragging_orbit:
                        self._dragging_orbit = False
                        self._save_ui_state()
                elif event.button == 2:
                    if self._dragging_pan:
                        self._dragging_pan = False
                        self._save_ui_state()
            elif event.type == pygame.MOUSEMOTION:
                if self.exit_confirm_modal.is_visible():
                    continue
                if self._density_drag is not None:
                    track = self._density_slider_rects.get(self._density_drag)
                    if track is not None:
                        self._set_density_from_x(
                            self._density_drag, event.pos[0], track
                        )
                    continue
                if self._basin_form_drag and self._basin_form_rect is not None:
                    self._set_basin_form_from_x(event.pos[0], self._basin_form_rect)
                    continue
                if self._light_drag is not None:
                    track = self._light_slider_rects.get(self._light_drag)
                    if track is not None:
                        self._set_light_slider_from_x(
                            self._light_drag, event.pos[0], track
                        )
                    continue
                if self._ground_relief_drag and self._ground_relief_rect is not None:
                    self._set_ground_relief_from_x(
                        event.pos[0], self._ground_relief_rect
                    )
                    continue
                if self._view_detail_drag and self._view_detail_rect is not None:
                    self._set_view_detail_from_x(
                        event.pos[0], self._view_detail_rect
                    )
                    continue
                if self._dragging_orbit:
                    self._hover_tip = None
                    if self._schematic_active():
                        continue
                    dx = event.pos[0] - self._last_mouse[0]
                    dy = event.pos[1] - self._last_mouse[1]
                    self.view_angle_y += dx * MOUSE_ROTATION_SENSITIVITY
                    self.view_angle_x += dy * MOUSE_ROTATION_SENSITIVITY
                    self.view_angle_x = max(
                        -math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self.view_angle_x)
                    )
                    self._note_side_mouse_edit()
                    self._last_mouse = event.pos
                elif self._dragging_pan:
                    self._hover_tip = None
                    if self._schematic_active():
                        continue
                    dx = event.pos[0] - self._last_mouse[0]
                    dy = event.pos[1] - self._last_mouse[1]
                    self.pan_x += dx
                    self.pan_y += dy
                    self._note_side_mouse_edit()
                    self._last_mouse = event.pos
                else:
                    self._update_hover_tip(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                if self.exit_confirm_modal.is_visible():
                    continue
                pos = pygame.mouse.get_pos()
                if pos[0] < int(getattr(self, "panel_x", 0)) + self.panel_w:
                    continue
                # Scroll the side panel under the cursor.
                scrolled = False
                if not self.legend_mini:
                    for pid, prect in list(
                        getattr(self, "_side_panel_rects", {}).items()
                    ):
                        if prect is None or not prect.collidepoint(pos):
                            continue
                        if self.side_panel_mini.get(pid, False):
                            continue
                        self.side_panel_scroll[pid] = max(
                            0, int(self.side_panel_scroll.get(pid, 0)) - event.y
                        )
                        if pid == "filters":
                            self.legend_scroll = self.side_panel_scroll[pid]
                        self._legend_blit_key = None
                        scrolled = True
                        break
                if scrolled:
                    continue
                if self._in_view(pos) and not self._hit_nav_button(pos):
                    if self._schematic_active():
                        continue
                    factor = 1.0 + event.y * MOUSE_WHEEL_SENSITIVITY
                    if factor <= 0:
                        factor = 0.85
                    self.scale_3d = float(
                        np.clip(
                            self.scale_3d * factor,
                            self.scale_3d_min,
                            self.scale_3d_max,
                        )
                    )
                    self._note_side_mouse_edit()
                    self._save_ui_state()

    def _draw_nav(self) -> None:
        """Status bar: camera presets + hint (outside projection)."""
        self._nav_rects.clear()
        y0 = self.height - self.status_h
        pygame.draw.rect(self.screen, CHROME_BG, (0, y0, self.width, self.status_h))
        pygame.draw.line(self.screen, CHROME_EDGE, (0, y0), (self.width, y0), 1)

        btn = BUTTON_SIZE
        sp = BUTTON_SPACING
        nav_y = y0 + (self.status_h - btn) // 2
        nav_x = 10

        x_active = abs(self.view_angle_x) < 0.1 and abs(self.view_angle_y) < 0.1
        rect = pygame.Rect(nav_x, nav_y, btn, btn)
        pygame.draw.rect(
            self.screen, (200, 100, 100) if x_active else (80, 80, 100), rect
        )
        pygame.draw.rect(self.screen, (120, 120, 140), rect, 1)
        t = self.nav_font.render("X", True, (255, 255, 255))
        self.screen.blit(t, (nav_x + (btn - t.get_width()) // 2, nav_y + 3))
        self._nav_rects["X"] = rect
        nav_x += btn + sp

        y_active = abs(self.view_angle_x - math.pi / 2) < 0.1
        rect = pygame.Rect(nav_x, nav_y, btn, btn)
        pygame.draw.rect(
            self.screen, (100, 200, 100) if y_active else (80, 80, 100), rect
        )
        pygame.draw.rect(self.screen, (120, 120, 140), rect, 1)
        t = self.nav_font.render("Y", True, (255, 255, 255))
        self.screen.blit(t, (nav_x + (btn - t.get_width()) // 2, nav_y + 3))
        self._nav_rects["Y"] = rect
        nav_x += btn + sp

        z_active = (
            abs(self.view_angle_y - math.pi / 2) < 0.1 and abs(self.view_angle_x) < 0.1
        )
        rect = pygame.Rect(nav_x, nav_y, btn, btn)
        pygame.draw.rect(
            self.screen, (100, 100, 200) if z_active else (80, 80, 100), rect
        )
        pygame.draw.rect(self.screen, (120, 120, 140), rect, 1)
        t = self.nav_font.render("Z", True, (255, 255, 255))
        self.screen.blit(t, (nav_x + (btn - t.get_width()) // 2, nav_y + 3))
        self._nav_rects["Z"] = rect
        nav_x += btn + sp

        side_active = (
            getattr(self, "_nav_preset", None) == "side"
            or self._is_default_side_angles(self.view_angle_x, self.view_angle_y)
        ) and not self._schematic_active()
        rect = pygame.Rect(nav_x, nav_y, btn * 2, btn)
        pygame.draw.rect(
            self.screen, (150, 150, 200) if side_active else (80, 80, 100), rect
        )
        pygame.draw.rect(self.screen, (120, 120, 140), rect, 1)
        t = self.nav_font.render("Side", True, (255, 255, 255))
        self.screen.blit(t, (nav_x + 2, nav_y + 3))
        self._nav_rects["Side"] = rect
        nav_x += btn * 2 + sp + 8

        proj = self._projection_name()
        if getattr(self, "show_port_schema", False):
            hint_txt = "AI Brain · click block · chips iterate · AI again → 3D"
            hint_col = (200, 180, 120)
        elif getattr(self, "show_brain_schema", False):
            hint_txt = "CNS Brain · hop/carve · chips iterate · Brain again → 3D"
            hint_col = (120, 190, 210)
        elif getattr(self, "show_grow_schema", False):
            hint_txt = "CNS Growing · policies · chips iterate · Grow again → 3D"
            hint_col = (210, 180, 110)
        elif getattr(self, "show_research_schema", False):
            hint_txt = "Research map · branches / pivots · chips filter · Map again → 3D"
            hint_col = (180, 190, 220)
        else:
            hint_txt = (
                f"{proj}  ·  RMB rotate · wheel zoom ({self.scale_3d:.2f}x) · MMB pan"
            )
            hint_col = (130, 135, 150)
        hint = self.hint_font.render(hint_txt, True, hint_col)
        self.screen.blit(
            hint, (nav_x, y0 + (self.status_h - hint.get_height()) // 2)
        )

    def _draw_toolbar(self) -> None:
        """Left tool rail (Photoshop-style). Only Navigate/arrow for now."""
        self._tool_rects = {}
        pygame.draw.rect(self.screen, CHROME_BG, (0, self.top_h, self.tool_w, self.view_h))
        pygame.draw.line(
            self.screen,
            CHROME_EDGE,
            (self.tool_w - 1, self.top_h),
            (self.tool_w - 1, self.top_h + self.view_h),
            1,
        )
        slot = TOOL_SLOT
        pad = (self.tool_w - slot) // 2
        tx = pad
        ty = self.top_h + pad
        on = getattr(self, "_active_tool", "nav") == "nav"
        rect = pygame.Rect(tx, ty, slot, slot)
        fill = (48, 52, 64) if on else (28, 30, 38)
        edge = (180, 190, 210) if on else (90, 95, 110)
        pygame.draw.rect(self.screen, fill, rect, border_radius=3)
        pygame.draw.rect(self.screen, edge, rect, 1, border_radius=3)
        # Arrow cursor glyph (pixel-ish).
        ax, ay = rect.centerx - 2, rect.centery - 5
        pts = [
            (ax, ay),
            (ax, ay + 14),
            (ax + 4, ay + 10),
            (ax + 7, ay + 16),
            (ax + 9, ay + 15),
            (ax + 6, ay + 9),
            (ax + 11, ay + 9),
        ]
        pygame.draw.polygon(self.screen, (220, 225, 235) if on else (150, 155, 170), pts)
        self._tool_rects["nav"] = rect

    def _draw_top_bar(self) -> None:
        """Top chrome: FPS / Learn chips, schema modes, Panels."""
        self._chrome_rects = {}
        pygame.draw.rect(self.screen, CHROME_BG, (0, 0, self.width, self.top_h))
        pygame.draw.line(
            self.screen, CHROME_EDGE, (0, self.top_h - 1), (self.width, self.top_h - 1), 1
        )

        x = 10
        y_mid = self.top_h // 2
        chip_h = 24

        # --- FPS chip (drawn here; panel later) ---
        fps = float(getattr(self, "_fps_smooth", FPS) or FPS)
        fps_txt = f"{fps:.0f} FPS"
        if getattr(self, "use_gl_tissue", False):
            fps_txt += " · GL"
        if self._fps_panel_open:
            fps_txt += " ▾"
        else:
            fps_txt += " ▸"
        fps_lab = self.hint_font.render(fps_txt, True, (200, 210, 230))
        pw = fps_lab.get_width() + 16
        ph = chip_h
        bx = x
        by = y_mid - ph // 2
        open_ = bool(self._fps_panel_open)
        plate = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(
            plate,
            (18, 22, 32, 200) if open_ else (14, 16, 24, 180),
            plate.get_rect(),
            border_radius=3,
        )
        pygame.draw.rect(
            plate,
            (140, 160, 190, 200) if open_ else (90, 100, 120, 160),
            plate.get_rect(),
            width=1,
            border_radius=3,
        )
        plate.blit(fps_lab, (8, (ph - fps_lab.get_height()) // 2))
        self.screen.blit(plate, (bx, by))
        self._fps_badge_rect = pygame.Rect(bx, by, pw, ph)
        self._chrome_rects["fps"] = self._fps_badge_rect
        x = bx + pw + 8

        # --- Learn chip ---
        txt, accent = self._learn_mode_label()
        txt = f"{txt} ▾" if self._learn_panel_open else f"{txt} ▸"
        lab = self.hint_font.render(txt, True, accent)
        lpw = lab.get_width() + 16
        lph = chip_h
        lbx, lby = x, y_mid - lph // 2
        open_l = bool(self._learn_panel_open)
        amp = float(getattr(self, "_learn_flash_amp", 0.0) or 0.0)
        glow = float(np.clip(amp, 0.0, 1.0))
        lplate = pygame.Surface((lpw, lph), pygame.SRCALPHA)
        fill_a = int(170 + 40 * glow) if open_l else int(150 + 40 * glow)
        pygame.draw.rect(
            lplate,
            (18, 28, 40, fill_a) if open_l else (12, 20, 28, fill_a),
            lplate.get_rect(),
            border_radius=3,
        )
        edge = (
            (accent[0], accent[1], accent[2], int(180 + 50 * glow))
            if open_l
            else (100, 150, 200, int(120 + 60 * glow))
        )
        pygame.draw.rect(lplate, edge, lplate.get_rect(), width=1, border_radius=3)
        lplate.blit(lab, (8, (lph - lab.get_height()) // 2))
        self.screen.blit(lplate, (lbx, lby))
        self._learn_badge_rect = pygame.Rect(lbx, lby, lpw, lph)
        self._chrome_rects["learn"] = self._learn_badge_rect

        # --- Mode toggles (Grow / Brain / AI) centered-right ---
        btn = BUTTON_SIZE
        sp = BUTTON_SPACING
        modes = [
            ("Grow", "Grow", bool(getattr(self, "show_grow_schema", False)), (170, 140, 80), (210, 180, 110)),
            ("Brain", "Brain", bool(getattr(self, "show_brain_schema", False)), (90, 150, 180), (120, 180, 210)),
            ("AI", "AI", bool(getattr(self, "show_port_schema", False)), (180, 150, 90), (200, 170, 110)),
            ("Map", "Map", bool(getattr(self, "show_research_schema", False)), (140, 150, 190), (180, 190, 220)),
        ]
        # Panels chip width estimate
        panels_w = 92
        mode_w = sum((btn * 2 + (8 if key == "Grow" else 4)) + sp for key, *_ in modes) - sp
        mx = self.width - 10 - panels_w - 12 - mode_w
        mx = max(mx, self._learn_badge_rect.right + 16)
        my = y_mid - btn // 2
        for key, label, on, fill_on, edge_on in modes:
            rw = btn * 2 + (8 if key == "Grow" else 4)
            rect = pygame.Rect(mx, my, rw, btn)
            pygame.draw.rect(
                self.screen, fill_on if on else (80, 80, 100), rect, border_radius=2
            )
            pygame.draw.rect(
                self.screen, edge_on if on else (120, 120, 140), rect, 1, border_radius=2
            )
            t = self.nav_font.render(label, True, (255, 255, 255))
            self.screen.blit(
                t, (mx + (rw - t.get_width()) // 2, my + (btn - t.get_height()) // 2)
            )
            self._chrome_rects[key] = rect
            mx += rw + sp

        # --- Panels toggle ---
        dock_open = not bool(getattr(self, "legend_mini", True))
        pr = pygame.Rect(self.width - 10 - panels_w, my, panels_w, btn)
        pygame.draw.rect(
            self.screen,
            (50, 55, 70) if dock_open else (32, 36, 48),
            pr,
            border_radius=2,
        )
        pygame.draw.rect(
            self.screen,
            (140, 160, 190) if dock_open else (110, 130, 160),
            pr,
            1,
            border_radius=2,
        )
        plab = self.small_font.render("Panels", True, (190, 205, 225))
        self.screen.blit(
            plab, (pr.x + 8, pr.y + (btn - plab.get_height()) // 2)
        )
        # Chevron: down when open (dock visible), left when closed
        cx = pr.right - 14
        cy = pr.centery
        if dock_open:
            pygame.draw.polygon(
                self.screen,
                (200, 210, 230),
                [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 5)],
            )
        else:
            pygame.draw.polygon(
                self.screen,
                (200, 210, 230),
                [(cx + 4, cy - 6), (cx + 4, cy + 6), (cx - 5, cy)],
            )
        self._chrome_rects["Panels"] = pr
        self.legend_toggle_rect = pr

    def _draw_learn_badge(self) -> None:
        """Learn detail panel only (chip lives in top bar)."""
        self._learn_panel_rect = None
        self._learn_rule_hit = None
        if not bool(self._learn_panel_open):
            return

        blocks = self._learn_detail_blocks()
        row_h = self.hint_font.get_height() + 2
        sec_h = self.hint_font.get_height() + 6
        bar_h = row_h + 2
        panel_kind = self._constructor_panel_kind()
        title_txt = "Growth / Yield" if panel_kind == "growth" else "Learn"
        title_col = (
            (200, 210, 180) if panel_kind == "growth" else (200, 225, 255)
        )
        title = self.small_font.render(title_txt, True, title_col)
        panel_w = 320
        inner_w = panel_w - 16

        def _wrap_h(text: str) -> int:
            words = str(text).split()
            if not words:
                return row_h
            lines = 1
            cur = words[0]
            for w in words[1:]:
                trial = f"{cur} {w}"
                if self.hint_font.size(trial)[0] <= inner_w:
                    cur = trial
                else:
                    lines += 1
                    cur = w
            return lines * row_h + 2

        h = 10 + title.get_height() + 10
        for row in blocks:
            kind = row[0]
            if kind == "head":
                h += self.small_font.get_height() + 6
            elif kind == "sec":
                h += sec_h
            elif kind == "bars":
                h += bar_h * max(1, len(row[1])) + 2
            elif kind in ("hint", "warn"):
                h += _wrap_h(str(row[1]))
            else:
                h += row_h
        h += 8

        # Prefer over blueprint column; else under chip at left of projection.
        anchor = self._learn_badge_rect
        py = int(self.top_h + 4)
        if self.panel_w >= panel_w + 16:
            px = int(self.panel_x + 8)
        elif anchor is not None:
            px = int(anchor.x)
        else:
            px = int(self.view_x + 8)
        max_bottom = self.view_y + self.view_h - 8
        panel_h = min(h, max(40, max_bottom - py))
        max_right = self.view_x + self.view_w - 8
        if px + panel_w > max_right:
            px = max(int(self.panel_x + 8), max_right - panel_w)

        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (12, 18, 28, 235), surf.get_rect(), border_radius=4)
        pygame.draw.rect(
            surf, (100, 160, 210, 200), surf.get_rect(), width=1, border_radius=4
        )
        y = 8
        surf.blit(title, (8, y))
        y += title.get_height() + 6
        pygame.draw.line(surf, (45, 60, 80), (8, y), (panel_w - 8, y), 1)
        y += 6

        for row in blocks:
            kind = row[0]
            if kind == "head":
                t = self.small_font.render(str(row[1]), True, (160, 210, 180))
                surf.blit(t, (8, y))
                y += self.small_font.get_height() + 6
            elif kind == "sec":
                t = self.hint_font.render(str(row[1]), True, (120, 170, 200))
                surf.blit(t, (8, y))
                y += sec_h
            elif kind == "rule":
                pygame.draw.line(
                    surf, (40, 55, 70), (8, y - 2), (panel_w - 8, y - 2), 1
                )
                surf.blit(
                    self.hint_font.render(str(row[1]), True, (140, 160, 180)),
                    (8, y),
                )
                self._learn_rule_hit = pygame.Rect(
                    px + 6, py + y - 1, panel_w - 12, row_h + 2
                )
                y += row_h
            elif kind == "kv":
                kk = self.hint_font.render(str(row[1]), True, (140, 150, 165))
                surf.blit(kk, (8, y))
                val_col = (210, 225, 245)
                if len(row) >= 4 and isinstance(row[3], tuple) and len(row[3]) == 3:
                    val_col = tuple(int(c) for c in row[3])  # type: ignore[misc]
                vv = self.hint_font.render(str(row[2]), True, val_col)
                surf.blit(vv, (panel_w - 8 - vv.get_width(), y))
                y += row_h
            elif kind == "hint":
                col = (170, 185, 160)
                words = str(row[1]).split()
                line = ""
                for w in words:
                    trial = w if not line else f"{line} {w}"
                    if self.hint_font.size(trial)[0] <= inner_w:
                        line = trial
                    else:
                        if line:
                            surf.blit(self.hint_font.render(line, True, col), (8, y))
                            y += row_h
                        line = w
                if line:
                    surf.blit(self.hint_font.render(line, True, col), (8, y))
                    y += row_h + 2
            elif kind == "warn":
                col = (240, 170, 100)
                words = str(row[1]).split()
                line = ""
                for w in words:
                    trial = w if not line else f"{line} {w}"
                    if self.hint_font.size(trial)[0] <= inner_w:
                        line = trial
                    else:
                        if line:
                            surf.blit(self.hint_font.render(line, True, col), (8, y))
                            y += row_h
                        line = w
                if line:
                    surf.blit(self.hint_font.render(line, True, col), (8, y))
                    y += row_h + 2
            elif kind == "bars":
                frozen_bars = bool(row[2]) if len(row) > 2 else False
                for lab, val in row[1]:
                    v = float(np.clip(val, 0.0, 1.0))
                    col = (200, 230, 140) if frozen_bars else (160, 210, 200)
                    tag = self.hint_font.render(str(lab), True, col)
                    surf.blit(tag, (8, y))
                    track = pygame.Rect(36, y + 5, panel_w - 36 - 44, 7)
                    pygame.draw.rect(surf, (35, 42, 48), track)
                    fw = max(0, int(track.w * v))
                    if fw:
                        pygame.draw.rect(surf, col, (track.x, track.y, fw, track.h))
                    pygame.draw.rect(surf, (70, 90, 100), track, 1)
                    num = self.hint_font.render(f"{v:.2f}", True, (200, 210, 220))
                    surf.blit(num, (panel_w - 8 - num.get_width(), y))
                    y += bar_h
                y += 2

        self.screen.blit(surf, (px, py))
        self._learn_panel_rect = pygame.Rect(px, py, panel_w, panel_h)

    def _draw_ai_brain_schema_view(self) -> None:
        """AI Brain iterative block diagram (replaces 3D while AI is on)."""
        assert self._view_surf is not None
        import time as _time

        t0 = _time.perf_counter()
        self._view_surf.fill((14, 16, 22))
        area = pygame.Rect(0, 0, self.view_w, max(1, self.view_h - NAV_PANEL_H))
        stage = str(getattr(self, "_port_stage", "sense") or "sense")
        schematic = AiBrainSchematic.from_host(self, stage=stage)
        rects = layout_ai_brain(schematic, area)
        local_hits = draw_ai_brain(
            self._view_surf,
            schematic,
            rects,
            fonts={
                "small": self.small_font,
                "hint": self.hint_font,
                "title": self.small_font,
            },
            selected=getattr(self, "_port_selected", None),
            stage=stage,
        )
        self._port_hits = list(local_hits)
        self._perf_ms["draw.tissue"] = (_time.perf_counter() - t0) * 1000.0
        self._perf_ms["draw.overlays"] = 0.0
        self._tissue_cache_hit = False

    def _draw_brain_schema_view(self) -> None:
        """CNS Brain hop/carve block diagram (replaces 3D while Brain is on)."""
        assert self._view_surf is not None
        import time as _time

        t0 = _time.perf_counter()
        self._view_surf.fill((12, 18, 24))
        area = pygame.Rect(0, 0, self.view_w, max(1, self.view_h - NAV_PANEL_H))
        stage = str(getattr(self, "_brain_stage", "sense") or "sense")
        schematic = BrainSchematic.from_host(self, stage=stage)
        rects = layout_brain(schematic, area)
        local_hits = draw_brain(
            self._view_surf,
            schematic,
            rects,
            fonts={
                "small": self.small_font,
                "hint": self.hint_font,
                "title": self.small_font,
            },
            selected=getattr(self, "_brain_selected", None),
            stage=stage,
        )
        self._brain_hits = list(local_hits)
        self._perf_ms["draw.tissue"] = (_time.perf_counter() - t0) * 1000.0
        self._perf_ms["draw.overlays"] = 0.0
        self._tissue_cache_hit = False

    def _draw_grow_schema_view(self) -> None:
        """CNS Growing tip/synapse policy diagram (replaces 3D while Grow is on)."""
        assert self._view_surf is not None
        import time as _time

        t0 = _time.perf_counter()
        self._view_surf.fill((18, 16, 12))
        area = pygame.Rect(0, 0, self.view_w, max(1, self.view_h - NAV_PANEL_H))
        stage = str(getattr(self, "_grow_stage", "recipe") or "recipe")
        schematic = GrowSchematic.from_host(self, stage=stage)
        rects = layout_grow(schematic, area)
        local_hits = draw_grow(
            self._view_surf,
            schematic,
            rects,
            fonts={
                "small": self.small_font,
                "hint": self.hint_font,
                "title": self.small_font,
            },
            selected=getattr(self, "_grow_selected", None),
            stage=stage,
        )
        self._grow_hits = list(local_hits)
        self._perf_ms["draw.tissue"] = (_time.perf_counter() - t0) * 1000.0
        self._perf_ms["draw.overlays"] = 0.0
        self._tissue_cache_hit = False

    def _draw_research_schema_view(self) -> None:
        """Research map flowchart (replaces 3D while Map is on)."""
        assert self._view_surf is not None
        import time as _time

        t0 = _time.perf_counter()
        self._view_surf.fill((14, 16, 22))
        area = pygame.Rect(0, 0, self.view_w, max(1, self.view_h - NAV_PANEL_H))
        stage = str(getattr(self, "_research_stage", "now") or "now")
        schematic = ResearchSchematic.from_host(self, stage=stage)
        rects = layout_research(schematic, area)
        local_hits = draw_research(
            self._view_surf,
            schematic,
            rects,
            fonts={
                "small": self.small_font,
                "hint": self.hint_font,
                "title": self.small_font,
            },
            selected=getattr(self, "_research_selected", None),
            stage=stage,
        )
        self._research_hits = list(local_hits)
        self._perf_ms["draw.tissue"] = (_time.perf_counter() - t0) * 1000.0
        self._perf_ms["draw.overlays"] = 0.0
        self._tissue_cache_hit = False

    def _ocelli_rig_pose(self) -> list[dict] | None:
        """Three ocelli: Mercedes look axes (120°), cups on a compact crown triangle.

        Sensing covers full azimuth; mounts stay near the head so Fuse tips reach CNS.
        """
        if self._frame_ocelli_pose is not None:
            return self._frame_ocelli_pose
        frame = self._body_orbit_frame()
        if frame is None:
            return None
        c, r = frame
        c = np.asarray(c, dtype=np.float64)
        r = float(r)
        # Compass: anterior = −Z, dorsal = +Y, right = +X
        ant = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        crown = c + ant * (float(OCELLI_CROWN_ANT) * r) + up * (
            float(OCELLI_CROWN_UP) * r
        )
        lat = float(OCELLI_LATERAL) * r
        elev = float(OCELLI_STAR_ELEV)
        # Physical mounts: tight anterior triangle (fuse-friendly).
        mounts = (
            crown + ant * (0.045 * r),
            crown - right * lat + ant * (0.015 * r),
            crown + right * lat + ant * (0.015 * r),
        )
        # Look axes: Mercedes star, one spoke forward, ±120° about +Y.
        names = ("medial", "left", "right")
        colors = ((120, 220, 140), (100, 200, 130), (100, 200, 130))
        angles = (0.0, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0)
        out: list[dict] = []
        for name, col, phi, lens in zip(names, colors, angles, mounts):
            cp, sp = math.cos(phi), math.sin(phi)
            horiz = np.array(
                [
                    ant[0] * cp + ant[2] * sp,
                    0.0,
                    -ant[0] * sp + ant[2] * cp,
                ],
                dtype=np.float64,
            )
            hn = float(np.linalg.norm(horiz))
            horiz = horiz / hn if hn > 1e-9 else ant.copy()
            axis = horiz * (1.0 - elev) + up * elev
            an = float(np.linalg.norm(axis))
            axis = axis / an if an > 1e-9 else horiz
            out.append(
                {
                    "name": name,
                    "lens": np.asarray(lens, dtype=np.float64),
                    "axis": axis.astype(np.float64),
                    "color": col,
                }
            )
        self._frame_ocelli_pose = out
        return out

    def _ocelli_facing(self, axis: np.ndarray, direction: np.ndarray) -> float:
        """Soft wide-angle acceptance (no hard blind seam between cups)."""
        c = float(np.dot(axis, direction))
        bias = float(OCELLI_FACING_BIAS)
        raw = max(0.0, (c + bias) / (1.0 + bias))
        return float(raw ** float(OCELLI_FACING_POW))

    def _update_ocelli_sensors(self) -> None:
        """Read sky + point lamp into I_m, I_l, I_r (virtual photoreceptors)."""
        pose = self._ocelli_rig_pose()
        if pose is None:
            self.ocelli_I = np.zeros(3, dtype=np.float64)
            return
        sky_a = float(np.clip(self.sky_fill, 0.0, 1.0)) if self.light_enabled else 0.15
        up = self._world_up()
        self._update_lamp_xyz()
        lamp = self._lamp_xyz.astype(np.float64)
        inten = float(np.clip(self.lamp_intensity, 0.0, 2.0)) if self.lamp_on else 0.0
        frame = self._body_orbit_frame()
        body_r = float(frame[1]) if frame is not None else 1.0
        vals = []
        for oc in pose:
            axis = oc["axis"]
            lens = oc["lens"]
            # Sky: soft facing to zenith (dull fill) — Matrix world up.
            sky = sky_a * self._ocelli_facing(axis, up)
            # Lamp: facing * falloff.
            to_l = lamp - lens
            dist = float(np.linalg.norm(to_l))
            if dist < 1e-6 or inten <= 0.0 or not self.light_enabled:
                lamp_t = 0.0
            else:
                L = to_l / dist
                facing = self._ocelli_facing(axis, L)
                atten = (body_r * body_r) / (dist * dist + (0.4 * body_r) ** 2)
                lamp_t = inten * facing * atten
            vals.append(float(np.clip(0.12 * sky_a + 0.55 * sky + 0.85 * lamp_t, 0.0, 1.0)))
        self.ocelli_I = np.asarray(vals, dtype=np.float64)

    def _ocelli_flat_ids(self) -> list[int]:
        ids = [int(i) for g in self.ocelli_groups for i in g]
        ids.extend(int(i) for i in getattr(self, "ocelli_terminals", []) or [])
        return ids

    def _device_wire_mesh(self, gids: list[int]) -> None:
        """Ring + dense local axon/dendrite mesh inside a pack."""
        if len(gids) < 2:
            return
        done: set[tuple[int, int]] = set()

        def _link(a: int, b: int) -> None:
            if a == b or (a, b) in done:
                return
            if self.fly.connect_pair(a, b):
                done.add((a, b))

        for i in range(len(gids)):
            _link(int(gids[i]), int(gids[(i + 1) % len(gids)]))
        # Extra chords so the pack reads as one solid component.
        for i in range(len(gids)):
            _link(int(gids[i]), int(gids[(i + 2) % len(gids)]))

    def _device_wire_star(self, hub: int, members: list[int]) -> None:
        """Every member ↔ virtual clamp (both directions = axon + dendrite stubs)."""
        h = int(hub)
        for mid in members:
            m = int(mid)
            if m == h:
                continue
            self.fly.connect_pair(m, h)
            self.fly.connect_pair(h, m)

    def _place_ocelli_clusters(
        self, pose: list[dict], *, move_only: bool = False
    ) -> list[np.ndarray]:
        """World positions for each group's somata (tight pack at each lens)."""
        frame = self._body_orbit_frame()
        if frame is None:
            c = np.zeros(3, dtype=np.float64)
            r = 1.0
        else:
            c = np.asarray(frame[0], dtype=np.float64)
            r = float(frame[1])
        cl_r = float(OCELLI_CLUSTER_FRAC) * r
        inset = float(OCELLI_INSET_FRAC) * r
        all_pos: list[np.ndarray] = []
        for gi, oc in enumerate(pose):
            lens = np.asarray(oc["lens"], dtype=np.float64)
            axis = oc["axis"]
            to_c = c - lens
            tn = float(np.linalg.norm(to_c))
            origin = lens + (to_c / tn) * inset if tn > 1e-8 else lens
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, axis))) > 0.9:
                ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = np.cross(axis, ref)
            u /= max(float(np.linalg.norm(u)), 1e-8)
            v = np.cross(axis, u)
            v /= max(float(np.linalg.norm(v)), 1e-8)
            rng = np.random.default_rng(1000 + gi)
            n = int(OCELLI_GROUP_N)
            pos = np.zeros((n, 3), dtype=np.float64)
            # Pack around the virtual cup clamp (lens), not on it.
            ang = np.linspace(0.0, math.tau, n, endpoint=False)
            ang = ang + float(rng.uniform(0, 0.15))
            rad = cl_r * np.sqrt(rng.uniform(0.15, 1.0, size=n))
            depth = cl_r * rng.uniform(0.08, 0.55, size=n)
            pos[:] = (
                origin[None, :]
                + rad[:, None] * np.cos(ang)[:, None] * u[None, :]
                + rad[:, None] * np.sin(ang)[:, None] * v[None, :]
                - depth[:, None] * axis[None, :]
            )
            all_pos.append(pos.astype(np.float32))
            if move_only and gi < len(self.ocelli_groups):
                for j, nid in enumerate(self.ocelli_groups[gi]):
                    neu = self.fly.neurons.get(int(nid))
                    if neu is not None and j < len(pos):
                        neu.soma = pos[j]
            # Virtual cup terminal sits inset toward CNS (shorter Fuse axon).
            if move_only and gi < len(getattr(self, "ocelli_terminals", [])):
                tid = int(self.ocelli_terminals[gi])
                tneu = self.fly.neurons.get(tid)
                if tneu is not None:
                    tneu.soma = origin.astype(np.float32)
        return all_pos

    def _strip_ocelli_device_wires(self, ids: set[int]) -> None:
        """Drop structural neurites whose both ends live in the device set."""
        for nid in ids:
            neu = self.fly.neurons.get(int(nid))
            if neu is None:
                continue
            neu.axons = [
                ax
                for ax in neu.axons
                if ax.partner_id is None or int(ax.partner_id) not in ids
            ]
            neu.dendrites = [
                dn
                for dn in neu.dendrites
                if dn.partner_id is None or int(dn.partner_id) not in ids
            ]

    def _wire_ocelli_device(self) -> None:
        """Pack mesh + star to cup clamps + bus across the three cups."""
        flat = set(self._ocelli_flat_ids())
        if len(flat) < 2:
            return
        self._strip_ocelli_device_wires(flat)
        terms = [int(t) for t in getattr(self, "ocelli_terminals", []) or []]
        for gi, gids in enumerate(self.ocelli_groups):
            self._device_wire_mesh([int(x) for x in gids])
            if gi < len(terms):
                self._device_wire_star(terms[gi], [int(x) for x in gids])
        # Three cup clamps form one ocelli instrument.
        for i in range(len(terms)):
            self.fly.connect_pair(terms[i], terms[(i + 1) % len(terms)])
            self.fly.connect_pair(terms[(i + 1) % len(terms)], terms[i])

    def _ensure_ocelli_neurons(self) -> bool:
        """Create/update 3 sensory packs + virtual cup clamps."""
        if not self.ocelli_on or not self.fly.neurons or not self.class_names:
            return False
        pose = self._ocelli_rig_pose()
        if pose is None:
            return False
        frame = self._body_orbit_frame()
        if frame is None:
            return False
        c = np.asarray(frame[0], dtype=np.float64)
        r = float(frame[1])
        try:
            sens_cls = self.class_names.index("sensory")
        except ValueError:
            sens_cls = 0
        need_pack = 3 * int(OCELLI_GROUP_N)
        need_term = 3
        flat = self._ocelli_flat_ids()
        alive_pack = sum(
            1 for g in self.ocelli_groups for i in g if i in self.fly.neurons
        )
        alive_term = sum(
            1
            for i in getattr(self, "ocelli_terminals", []) or []
            if i in self.fly.neurons
        )
        if (
            alive_pack != need_pack
            or alive_term != need_term
            or len(self.ocelli_groups) != 3
        ):
            for i in flat:
                self.fly.neurons.pop(i, None)
            for s in self.fly.sets.values():
                s.members.difference_update(flat)
            self.ocelli_groups = [[], [], []]
            self.ocelli_terminals = []
            self.ocelli_ids = []
            positions = self._place_ocelli_clusters(pose, move_only=False)
            for gi, pos in enumerate(positions):
                gids: list[int] = []
                for j in range(len(pos)):
                    n = self.fly.add_neuron(
                        pos[j], sens_cls, set_name="sensory"
                    )
                    gids.append(int(n.id))
                self.ocelli_groups[gi] = gids
                # Virtual cup clamp inset toward CNS (not on the outer lens).
                lens = np.asarray(pose[gi]["lens"], dtype=np.float64)
                to_c = c - lens
                tn = float(np.linalg.norm(to_c))
                inset = float(OCELLI_INSET_FRAC) * r
                clamp = (
                    lens + (to_c / tn) * inset if tn > 1e-8 else lens
                ).astype(np.float32)
                t = self.fly.add_neuron(clamp, sens_cls, set_name="sensory")
                self.ocelli_terminals.append(int(t.id))
            self.ocelli_ids = self._ocelli_flat_ids()
            self._wire_ocelli_device()
            self.fly.rebuild_caches()
            self._sync_from_fly()
            try:
                si = self.class_names.index("sensory")
                self.focus_basin_indices.add(si)
                self.focus_soma_indices.add(si)
                self._sync_layers_from_focus()
                self._basin_cache_key = None
            except ValueError:
                pass
            return True
        self._place_ocelli_clusters(pose, move_only=True)
        self.ocelli_ids = self._ocelli_flat_ids()
        self._wire_ocelli_device()
        self.fly.rebuild_caches()
        self._basin_cache_key = None
        return True

    def _class_indices(self, names: tuple[str, ...] | list[str]) -> set[int]:
        out: set[int] = set()
        for n in names:
            try:
                out.add(int(self.class_names.index(n)))
            except ValueError:
                continue
        return out

    def _pick_cns_near(
        self,
        probe: np.ndarray,
        n: int,
        *,
        prefer: tuple[str, ...] | list[str],
        fallback: tuple[str, ...] | list[str] = (),
        ban: set[int] | None = None,
        used: set[int] | None = None,
    ) -> list[int]:
        """Nearest CNS posts in prefer classes, then fallback — never device ids."""
        ban = set(ban or ())
        used = set(used or ())
        prefer_i = self._class_indices(prefer)
        fall_i = self._class_indices(fallback)
        probe = np.asarray(probe, dtype=np.float64).reshape(3)

        def _collect(want: set[int]) -> tuple[np.ndarray, np.ndarray]:
            ids: list[int] = []
            xyz: list[np.ndarray] = []
            if not want:
                return (
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0, 3), dtype=np.float64),
                )
            for nid, neu in self.fly.neurons.items():
                i = int(nid)
                if i in ban or i in used:
                    continue
                if int(neu.cls) not in want:
                    continue
                ids.append(i)
                xyz.append(neu.soma)
            if not ids:
                return (
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0, 3), dtype=np.float64),
                )
            return (
                np.asarray(ids, dtype=np.int32),
                np.stack(xyz, axis=0).astype(np.float64),
            )

        picks: list[int] = []
        for want in (prefer_i, fall_i):
            if len(picks) >= n:
                break
            ids_a, xyz_a = _collect(want)
            if len(ids_a) == 0:
                continue
            d = np.linalg.norm(xyz_a - probe[None, :], axis=1)
            for j in np.argsort(d).tolist():
                cand = int(ids_a[j])
                if cand in used or cand in picks:
                    continue
                picks.append(cand)
                used.add(cand)
                if len(picks) >= n:
                    break
        # Last resort: any non-device soma near probe.
        if len(picks) < n:
            ids_a, xyz_a = _collect(
                set(range(len(self.class_names)))  # all classes
            )
            if len(ids_a):
                d = np.linalg.norm(xyz_a - probe[None, :], axis=1)
                for j in np.argsort(d).tolist():
                    cand = int(ids_a[j])
                    if cand in used or cand in picks:
                        continue
                    picks.append(cand)
                    used.add(cand)
                    if len(picks) >= n:
                        break
        return picks

    def _ocelli_pick_targets(self) -> list[list[int]] | None:
        """Per ocellus: multi CNS posts in ascending/descending (neck tracts)."""
        pose = self._ocelli_rig_pose()
        if pose is None or not self.ocelli_groups:
            return None
        frame = self._body_orbit_frame()
        if frame is None:
            return None
        c = np.asarray(frame[0], dtype=np.float64)
        ban = set(self._ocelli_flat_ids()) | set(self._turret_flat_ids())
        used: set[int] = set()
        per = max(1, int(OCELLI_TIPS_PER_GROUP))
        groups: list[list[int]] = []
        for oc in pose:
            probe = oc["lens"] + 0.72 * (c - oc["lens"])
            picks = self._pick_cns_near(
                probe,
                per,
                prefer=FUSE_NECK_CLASSES,
                fallback=FUSE_NECK_FALLBACK,
                ban=ban,
                used=used,
            )
            if not picks:
                return None
            # Pad with last pick if atlas is sparse for this ROI.
            while len(picks) < per:
                picks.append(picks[-1])
            groups.append(picks)
        return groups

    def _arm_ocelli_tips(self, *, missing_only: bool = False) -> int:
        """Tips leave from virtual cup clamps → ascending/descending CNS.

        ``missing_only``: arm only cups that are not yet fused (Fuse mode).
        """
        if not self.ocelli_on:
            return 0
        if self.fly.is_adult() and not missing_only:
            return 0
        if not self._ensure_ocelli_neurons():
            return 0
        flat = np.asarray(self._ocelli_flat_ids(), dtype=np.int32)
        if (
            not missing_only
            and len(self.fly.growth_owner)
            and len(flat)
            and np.any(np.isin(self.fly.growth_owner, flat))
        ):
            return 0
        goals_g = self._ocelli_pick_targets()
        terms = getattr(self, "ocelli_terminals", []) or []
        if goals_g is None or len(terms) != 3 or len(goals_g) != 3:
            return 0
        conn = self._ocelli_connected_mask() if missing_only else [False, False, False]
        owners: list[int] = []
        goals: list[int] = []
        for gi, tid in enumerate(terms):
            if missing_only and gi < len(conn) and conn[gi]:
                continue
            if missing_only and self._device_owner_has_open_tip(int(tid)):
                continue
            for g in goals_g[gi]:
                owners.append(int(tid))
                goals.append(int(g))
        if not owners:
            return 0
        return int(
            self.fly.append_growth_tips(
                np.asarray(owners, dtype=np.int32),
                np.asarray(goals, dtype=np.int32),
                seek_limit=float(OCELLI_FUSE_SEEK),
            )
        )

    def _seed_ocelli_activity(self) -> int:
        """Drive cup clamps + packs by photoreceptor I (one solid instrument)."""
        if not self.ocelli_on or not self.ocelli_groups:
            return 0
        self._update_ocelli_sensors()
        levels: dict[int, float] = {}
        terms = getattr(self, "ocelli_terminals", []) or []
        for gi, gids in enumerate(self.ocelli_groups):
            if gi >= len(self.ocelli_I):
                continue
            I = float(self.ocelli_I[gi])
            if gi < len(terms):
                levels[int(terms[gi])] = I
            for nid in gids:
                levels[int(nid)] = I
        if not levels:
            return 0
        return int(self.fly.stimulate_neurons(levels))

    def _fuse_mask_cache_key(self) -> tuple:
        """Fingerprint for device↔CNS fuse masks (stable across adult Run frames)."""
        fly = self.fly
        go = getattr(fly, "growth_owner", None)
        n_g = int(len(go)) if go is not None else 0
        n_conn = (
            int(np.sum(fly.growth_status == STATUS_CONNECTED)) if n_g else 0
        )
        return (
            id(fly),
            int(len(fly.neurons)),
            n_g,
            n_conn,
            tuple(self._ocelli_flat_ids()),
            tuple(self._turret_flat_ids()),
        )

    def _contact_gain_partner_index(self) -> dict[int, set[int]]:
        """nid → partners from contact_gain; rebuilt when gains version/len change.

        Avoids O(|gains|) full scans per fuse site (~2ms×N_sites on miss).
        """
        fly = self.fly
        gains = getattr(fly, "contact_gain", None) or {}
        ver = int(getattr(fly, "_contact_gain_version", 0))
        stamp = (id(fly), ver, int(len(gains)))
        cached = getattr(self, "_cg_partner_idx_cache", None)
        if isinstance(cached, tuple) and cached[0] == stamp:
            return cached[1]
        idx: dict[int, set[int]] = {}
        for key in gains.keys():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            a, b = int(key[0]), int(key[1])
            idx.setdefault(a, set()).add(b)
            idx.setdefault(b, set()).add(a)
        self._cg_partner_idx_cache = (stamp, idx)
        return idx

    def _owners_fused_to_cns(self, owner_ids: list[int]) -> bool:
        """True if device site is fused: afferent (device→CNS) or efferent (CNS→device).

        Never scans the full neuron dict — that was ~30ms×N_sites/frame on miss
        and dominated CPU draw after GL off (~5 FPS). Efferent links are read
        from device dendrites / growth tips / contact_gain only.
        """
        ids = [int(i) for i in owner_ids if i is not None]
        if not ids:
            return False
        ban = set(self._ocelli_flat_ids()) | set(self._turret_flat_ids())
        try:
            import compound_eyes as _ce

            for d in _ce.iter_eye_devices(self):
                ban.update(d.flat_ids())
        except Exception:
            pass
        id_set = set(ids)
        if len(self.fly.growth_owner):
            m = np.isin(self.fly.growth_owner, np.asarray(ids, dtype=np.int32))
            if np.any(m) and np.any(self.fly.growth_status[m] == STATUS_CONNECTED):
                return True
            # Efferent tips: CNS owner → device goal.
            if len(self.fly.growth_goal) == len(self.fly.growth_owner):
                m2 = np.isin(self.fly.growth_goal, np.asarray(ids, dtype=np.int32))
                if np.any(m2) and np.any(
                    self.fly.growth_status[m2] == STATUS_CONNECTED
                ):
                    return True
        for nid in ids:
            neu = self.fly.neurons.get(int(nid))
            if neu is None:
                continue
            # Afferent: device → CNS axon.
            for ax in neu.axons:
                if ax.status != "connected" or ax.partner_id is None:
                    continue
                pid = int(ax.partner_id)
                if pid in ban:
                    continue  # device-local bus wire
                if pid in self.fly.neurons:
                    return True
            # Efferent: CNS → device (dendrite partner on the device soma).
            for den in neu.dendrites:
                if den.partner_id is None:
                    continue
                pid = int(den.partner_id)
                if pid in ban:
                    continue
                if pid in self.fly.neurons:
                    return True
        # Gain / contact edges touching this site (indexed; no full-dict scan).
        gains = getattr(self.fly, "contact_gain", None) or {}
        if gains:
            idx = self._contact_gain_partner_index()
            for nid in id_set:
                for pid in idx.get(nid, ()):
                    if pid not in ban:
                        return True
        return False

    def _ocelli_connected_mask(self) -> list[bool]:
        key = self._fuse_mask_cache_key()
        cached = getattr(self, "_fuse_mask_cache", None)
        if (
            isinstance(cached, dict)
            and cached.get("key") == key
            and "ocelli" in cached
        ):
            return list(cached["ocelli"])
        out = [False, False, False]
        terms = getattr(self, "ocelli_terminals", []) or []
        for i in range(3):
            gids = list(self.ocelli_groups[i]) if i < len(self.ocelli_groups) else []
            if i < len(terms):
                gids.append(int(terms[i]))
            out[i] = self._owners_fused_to_cns(gids)
        if not isinstance(cached, dict):
            cached = {}
            self._fuse_mask_cache = cached
        if cached.get("key") != key:
            cached.clear()
            cached["key"] = key
        cached["ocelli"] = list(out)
        return out

    def _draw_ocelli(self, cx: float, cy: float, view_lim_y: int) -> None:
        """Cups + lenses; cluster funnels come from the normal Basins layer."""
        assert self._view_surf is not None
        if not self.ocelli_on:
            return
        pose = self._ocelli_rig_pose()
        if pose is None:
            return
        connected = self._ocelli_connected_mask()
        frame = self._body_orbit_frame()
        rad = 0.035 * float(frame[1]) if frame is not None else 0.08
        # Soft screen-space group box around cups + sensory clusters.
        self._draw_ocelli_group_box(pose, connected, cx, cy, view_lim_y)
        overlay = self._overlay_surface(view_lim_y)
        for i, oc in enumerate(pose):
            I = float(self.ocelli_I[i]) if i < len(self.ocelli_I) else 0.0
            col = oc["color"]
            bri = 0.35 + 0.65 * I
            rgb = tuple(int(np.clip(c * bri, 0, 255)) for c in col)
            lens = oc["lens"]
            axis = oc["axis"]
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, axis))) > 0.9:
                ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = np.cross(axis, ref)
            u /= max(float(np.linalg.norm(u)), 1e-8)
            v = np.cross(axis, u)
            v /= max(float(np.linalg.norm(v)), 1e-8)
            depth = rad * 2.2
            for t in (0.0, 0.45, 0.85):
                rr = rad * (1.0 - 0.55 * t)
                center = lens - axis * (depth * t)
                ang = np.linspace(0.0, math.tau, 16, endpoint=False)
                ring = (
                    center[None, :]
                    + rr * np.cos(ang)[:, None] * u[None, :]
                    + rr * np.sin(ang)[:, None] * v[None, :]
                ).astype(np.float32)
                sx, sy = project_3d_batch(
                    ring,
                    self.view_angle_x,
                    self.view_angle_y,
                    cx,
                    cy,
                    self.scale_3d,
                )
                pts = [
                    (int(round(float(x))), int(round(float(y))))
                    for x, y in zip(sx, sy)
                    if 0 <= x < self.view_w and 0 <= y < view_lim_y
                ]
                if len(pts) >= 3:
                    a = int(40 + 100 * I * (1.0 - 0.3 * t))
                    pygame.draw.polygon(overlay, (*rgb, a), pts, 0)
                    pygame.draw.polygon(overlay, (*rgb, min(255, a + 40)), pts, 1)
            sxl, syl = project_3d_batch(
                lens.reshape(1, 3).astype(np.float32),
                self.view_angle_x,
                self.view_angle_y,
                cx,
                cy,
                self.scale_3d,
            )
            lx, ly = int(round(float(sxl[0]))), int(round(float(syl[0])))
            if 0 <= lx < self.view_w and 0 <= ly < view_lim_y:
                core = max(2, int(3 + 4 * I))
                pygame.draw.circle(overlay, (*rgb, 220), (lx, ly), core)
                pygame.draw.circle(
                    overlay, (230, 255, 230, 200), (lx, ly), max(1, core // 2)
                )
                tag = (
                    "M"
                    if oc["name"] == "medial"
                    else ("L" if oc["name"] == "left" else "R")
                )
                if connected[i]:
                    tag = tag + "*"
                # Per-sensor value plate (original layout).
                txt = f"{tag} {I:.2f}"
                lab = self.hint_font.render(txt, True, (220, 245, 220))
                pad_x, pad_y = 5, 2
                bw = lab.get_width() + pad_x * 2
                bh = lab.get_height() + pad_y * 2
                bx = lx + core + 4
                by = ly - bh // 2
                bx = int(np.clip(bx, 2, max(2, self.view_w - bw - 2)))
                by = int(np.clip(by, 2, max(2, view_lim_y - bh - 2)))
                plate = pygame.Surface((bw, bh), pygame.SRCALPHA)
                pygame.draw.rect(
                    plate, (18, 28, 22, 200), plate.get_rect(), border_radius=3
                )
                pygame.draw.rect(
                    plate, (*rgb, 210), plate.get_rect(), width=1, border_radius=3
                )
                bar_w = max(1, int((bw - 4) * np.clip(I, 0.0, 1.0)))
                pygame.draw.rect(
                    plate, (*rgb, 160), pygame.Rect(2, bh - 3, bar_w, 2)
                )
                plate.blit(lab, (pad_x, pad_y))
                overlay.blit(plate, (bx, by))
        self._view_surf.blit(overlay, (0, 0))

    def _draw_ocelli_group_box(
        self,
        pose: list[dict],
        connected: list[bool],
        cx: float,
        cy: float,
        view_lim_y: int,
    ) -> None:
        """Nearly transparent screen box unifying the three ocelli clusters."""
        assert self._view_surf is not None
        pts3: list[np.ndarray] = []
        for oc in pose:
            pts3.append(np.asarray(oc["lens"], dtype=np.float64))
        for gids in self.ocelli_groups:
            for nid in gids:
                neu = self.fly.neurons.get(int(nid))
                if neu is not None:
                    pts3.append(np.asarray(neu.soma, dtype=np.float64))
        if len(pts3) < 2:
            return
        arr = np.stack(pts3, axis=0).astype(np.float32)
        sx, sy = project_3d_batch(
            arr,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        xs = [float(x) for x in sx if np.isfinite(x)]
        ys = [float(y) for y in sy if np.isfinite(y)]
        if len(xs) < 2:
            return
        pad = 14
        left = int(min(xs) - pad)
        right = int(max(xs) + pad)
        top = int(min(ys) - pad)
        bot = int(max(ys) + pad)
        left = max(0, left)
        top = max(0, top)
        right = min(self.view_w - 1, right)
        bot = min(view_lim_y - 1, bot)
        if right - left < 8 or bot - top < 8:
            return
        bw, bh = right - left, bot - top
        overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
        rect = pygame.Rect(0, 0, bw, bh)
        n_ok = sum(1 for c in connected if c)
        n_tot = max(1, len(connected))
        if n_ok == n_tot:
            fill, edge, tick_c, title_c = (
                (40, 70, 55, 28),
                (120, 180, 140, 55),
                (160, 220, 180, 90),
                (170, 210, 180),
            )
        elif n_ok > 0:
            fill, edge, tick_c, title_c = (
                (70, 60, 30, 32),
                (200, 180, 90, 70),
                (220, 200, 120, 95),
                (210, 200, 140),
            )
        else:
            # Not fused with the fly — alarm red box.
            fill, edge, tick_c, title_c = (
                (70, 28, 28, 36),
                (210, 90, 90, 90),
                (230, 120, 120, 110),
                (220, 150, 150),
            )
        pygame.draw.rect(overlay, fill, rect, border_radius=8)
        pygame.draw.rect(overlay, edge, rect, width=1, border_radius=8)
        tick = 10
        for (x0, y0, dx, dy) in (
            (0, 0, 1, 1),
            (bw - 1, 0, -1, 1),
            (0, bh - 1, 1, -1),
            (bw - 1, bh - 1, -1, -1),
        ):
            pygame.draw.line(
                overlay, tick_c, (x0, y0), (x0 + dx * tick, y0), 1
            )
            pygame.draw.line(
                overlay, tick_c, (x0, y0), (x0, y0 + dy * tick), 1
            )
        title = self.hint_font.render("ocelli", True, title_c)
        title.set_alpha(160)
        # Title sits just above the box on the parent view.
        self._view_surf.blit(
            title, (left + 6, max(2, top - title.get_height() - 1))
        )

        # Device↔fly fuse plate — a bit below the top edge of the box.
        if n_ok == n_tot:
            st_txt = "fused"
            st_rgb = (120, 220, 150)
            st_bg = (16, 36, 24, 200)
        elif n_ok > 0:
            st_txt = f"fusing {n_ok}/{n_tot}"
            st_rgb = (200, 200, 120)
            st_bg = (36, 34, 16, 200)
        else:
            st_txt = "not fused"
            st_rgb = (230, 140, 140)
            st_bg = (40, 18, 18, 200)
        st_lab = self.hint_font.render(st_txt, True, st_rgb)
        pad_x, pad_y = 6, 3
        dot_gap = 12
        pw = st_lab.get_width() + pad_x * 2 + dot_gap
        ph = st_lab.get_height() + pad_y * 2
        px = 8
        py = 10
        px = int(np.clip(px, 2, max(2, bw - pw - 2)))
        py = int(np.clip(py, 2, max(2, bh - ph - 2)))
        plate = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(plate, st_bg, plate.get_rect(), border_radius=3)
        pygame.draw.rect(plate, (*st_rgb, 200), plate.get_rect(), width=1, border_radius=3)
        pygame.draw.circle(plate, (*st_rgb, 230), (pad_x + 3, ph // 2), 3)
        plate.blit(st_lab, (pad_x + dot_gap, pad_y))
        overlay.blit(plate, (px, py))
        self._view_surf.blit(overlay, (left, top))

    # ------------------------------------------------------------------
    # Compound eyes (Left/Right × corners + U/D) — lamp bearing sockets
    # ------------------------------------------------------------------

    def _update_eye_sensors(self) -> None:
        compound_eyes.update_eye_sensors(self)

    def _ensure_eye_neurons(self, device: CompoundEyeDevice | None = None) -> bool:
        ok = False
        if device is not None:
            return bool(compound_eyes.ensure_eye_neurons(self, device))
        for d in compound_eyes.iter_eye_devices(self):
            if d.enabled:
                ok = compound_eyes.ensure_eye_neurons(self, d) or ok
        return ok

    def _eye_connected_mask(self, eye_name: str) -> list[bool]:
        return compound_eyes.eye_connected_mask(self, eye_name)

    def _arm_eye_tips(self, *, missing_only: bool = False) -> int:
        n = 0
        for d in compound_eyes.iter_eye_devices(self):
            n += int(compound_eyes.arm_eye_tips(self, d, missing_only=missing_only))
        return n

    def _seed_eye_activity(self) -> int:
        return int(compound_eyes.seed_eye_activity(self))

    def _draw_compound_eye(
        self, device: CompoundEyeDevice, cx: float, cy: float, view_lim_y: int
    ) -> None:
        compound_eyes.draw_compound_eye(self, device, cx, cy, view_lim_y)

    # ------------------------------------------------------------------
    # Johnston's organ (paired antennae) — Matrix tilt → sensory Fuse
    # ------------------------------------------------------------------

    def _update_jo_sensors(self) -> None:
        johnston_organ.update_jo_sensors(self)

    def _ensure_jo_neurons(self, device: JohnstonOrganDevice | None = None) -> bool:
        ok = False
        if device is not None:
            return bool(johnston_organ.ensure_jo_neurons(self, device))
        for d in johnston_organ.iter_jo_devices(self):
            if d.enabled:
                ok = johnston_organ.ensure_jo_neurons(self, d) or ok
        return ok

    def _jo_connected_mask(self, jo_name: str) -> list[bool]:
        return johnston_organ.jo_connected_mask(self, jo_name)

    def _arm_jo_tips(self, *, missing_only: bool = False) -> int:
        n = 0
        for d in johnston_organ.iter_jo_devices(self):
            n += int(johnston_organ.arm_jo_tips(self, d, missing_only=missing_only))
        return n

    def _seed_jo_activity(self) -> int:
        return int(johnston_organ.seed_jo_activity(self))

    def _draw_johnston_organ(
        self, device: JohnstonOrganDevice, cx: float, cy: float, view_lim_y: int
    ) -> None:
        johnston_organ.draw_johnston_organ(self, device, cx, cy, view_lim_y)

    # ------------------------------------------------------------------
    # Wing turret device (motor + sensory packs on thoracic wing sites)
    # ------------------------------------------------------------------

    def _turret_flat_ids(self) -> list[int]:
        out: list[int] = []
        for g in getattr(self, "turret_motor_groups", []) or []:
            out.extend(int(i) for i in g)
        for g in getattr(self, "turret_sensory_groups", []) or []:
            out.extend(int(i) for i in g)
        out.extend(int(i) for i in getattr(self, "turret_motor_terminals", []) or [])
        out.extend(int(i) for i in getattr(self, "turret_sensory_terminals", []) or [])
        bus = getattr(self, "turret_bus_terminal", None)
        if bus is not None:
            out.append(int(bus))
        return out

    def _turret_wing_anchors(self) -> np.ndarray | None:
        """Four dorsal thoracic motor centroids ≈ L/R × A/P wing sites."""
        try:
            mi = self.class_names.index("motor")
        except ValueError:
            return None
        xyz_all = self.fly.soma_xyz
        cls_all = self.fly.soma_cls
        ids_all = getattr(self.fly, "soma_ids", None)
        if len(xyz_all) == 0 or len(cls_all) != len(xyz_all):
            return None
        ban = tuple(
            sorted(
                set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
            )
        )
        key = (id(xyz_all), int(len(xyz_all)), int(mi), ban)
        if key == self._wing_anchor_key and self._wing_anchor_cache is not None:
            return self._wing_anchor_cache
        mask = cls_all.astype(np.int32) == int(mi)
        if ban and ids_all is not None and len(ids_all) == len(xyz_all):
            ban_arr = np.asarray(ban, dtype=np.int64)
            mask = mask & ~np.isin(np.asarray(ids_all, dtype=np.int64), ban_arr)
        if not np.any(mask):
            return None
        xyz = np.asarray(xyz_all[mask], dtype=np.float64)
        if len(xyz) < 8:
            return None
        z = xyz[:, 2]
        lo, hi = np.percentile(z, [25, 75])
        y_med = float(np.median(xyz[:, 1]))
        sel = (z >= lo) & (z <= hi) & (xyz[:, 1] >= y_med)
        sub = xyz[sel]
        if len(sub) < 8:
            sub = xyz
        xm = float(np.median(sub[:, 0]))
        centers: list[np.ndarray] = []
        for x_mask in (sub[:, 0] < xm, sub[:, 0] >= xm):
            half = sub[x_mask]
            if len(half) < 2:
                continue
            zm = float(np.median(half[:, 2]))
            for z_mask in (half[:, 2] < zm, half[:, 2] >= zm):
                chunk = half[z_mask]
                if len(chunk) == 0:
                    continue
                centers.append(chunk.mean(axis=0))
        if len(centers) < 4:
            xs = np.quantile(sub[:, 0], [0.25, 0.75])
            zs = np.quantile(sub[:, 2], [0.25, 0.75])
            centers = []
            for xq in xs:
                for zq in zs:
                    d = (sub[:, 0] - xq) ** 2 + (sub[:, 2] - zq) ** 2
                    centers.append(sub[int(np.argmin(d))])
        out = np.stack(centers[:4], axis=0).astype(np.float64)
        self._wing_anchor_key = key
        self._wing_anchor_cache = out
        return out

    def _turret_rig_pose(self) -> dict | None:
        """Central mount + aim + 4 wing-site pads."""
        if self._frame_turret_pose is not None:
            return self._frame_turret_pose
        frame = self._body_orbit_frame()
        if frame is None:
            return None
        _c, r = frame
        r = float(r)
        anchors = self._turret_wing_anchors()
        if anchors is None or len(anchors) < int(TURRET_SITES):
            return None
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        site_lift = up * (float(TURRET_SITE_UP) * r)
        sites = anchors[: int(TURRET_SITES)] + site_lift[None, :]
        mount = sites.mean(axis=0) + up * (float(TURRET_MOUNT_UP) * r)
        yaw = float(self.turret_yaw)
        pitch = float(
            np.clip(self.turret_pitch, TURRET_PITCH_MIN, TURRET_PITCH_MAX)
        )
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
        nrm = float(np.linalg.norm(fwd))
        if nrm < 1e-8:
            fwd = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            fwd = fwd / nrm
        right = np.cross(fwd, up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / rn
        up_aim = np.cross(right, fwd)
        un = float(np.linalg.norm(up_aim))
        if un > 1e-8:
            up_aim = up_aim / un
        out = {
            "mount": mount,
            "fwd": fwd,
            "right": right,
            "up": up_aim,
            "anchors": anchors[: int(TURRET_SITES)],
            "sites": sites.astype(np.float64),
            "barrel": mount + fwd * (0.12 * r),
            "r": r,
        }
        self._frame_turret_pose = out
        return out

    def _turret_mech_pose(self) -> dict | None:
        """Visual hardware: pads squashed in XZ, short struts to a low pedestal.

        Neurons / Fuse keep ``_turret_rig_pose``. Drawn pads + mount + funnel use
        this compact frame — long neuro ``MOUNT_UP`` stays for wiring clearance.
        """
        pose = self._turret_rig_pose()
        if pose is None:
            return None
        sites = np.asarray(pose["sites"], dtype=np.float64).copy()
        s = float(np.clip(TURRET_MECH_XZ, 0.05, 1.0))
        c = sites.mean(axis=0)
        sites[:, 0] = c[0] + (sites[:, 0] - c[0]) * s
        sites[:, 2] = c[2] + (sites[:, 2] - c[2]) * s
        r = float(pose["r"])
        # Low pedestal over the pads → short legs (not the tall neuro mount).
        mount = np.array(
            [
                float(sites[:, 0].mean()),
                float(sites[:, 1].mean()) + float(TURRET_MECH_MOUNT_UP) * r,
                float(sites[:, 2].mean()),
            ],
            dtype=np.float64,
        )
        fwd = np.asarray(pose["fwd"], dtype=np.float64)
        out = dict(pose)
        out["sites"] = sites
        out["mount"] = mount
        out["barrel"] = mount + fwd * (0.12 * r)
        if pose.get("anchors") is not None:
            anc = np.asarray(pose["anchors"], dtype=np.float64).copy()
            ac = anc.mean(axis=0)
            anc[:, 0] = ac[0] + (anc[:, 0] - ac[0]) * s
            anc[:, 2] = ac[2] + (anc[:, 2] - ac[2]) * s
            out["anchors"] = anc
        return out

    def _turret_site_motor_level_raw(self, si: int) -> float:
        """Clamp-weighted mean activity at one wing motor site (0..1), pre-socket."""
        vals: list[float] = []
        weights: list[float] = []
        m_terms = getattr(self, "turret_motor_terminals", []) or []
        if si < len(m_terms):
            neu = self.fly.neurons.get(int(m_terms[si]))
            if neu is not None:
                vals.append(float(neu.activity))
                weights.append(0.7)
        groups = getattr(self, "turret_motor_groups", []) or []
        if si < len(groups):
            for nid in groups[si]:
                neu = self.fly.neurons.get(int(nid))
                if neu is None:
                    continue
                vals.append(float(neu.activity))
                weights.append(0.3 / max(1, len(groups[si])))
        if not vals:
            return 0.0
        w = np.asarray(weights, dtype=np.float64)
        a = np.asarray(vals, dtype=np.float64)
        return float(np.clip(np.sum(w * a) / max(1e-9, float(np.sum(w))), 0.0, 1.0))

    def _turret_site_motor_level(self, si: int) -> float:
        """Motor site level: socket readout when on, else raw hop."""
        sock = getattr(self, "motor_socket", None)
        if sock is not None and getattr(self, "motor_socket_on", False):
            levels = sock.levels(self)
            if 0 <= int(si) < len(levels):
                return float(levels[int(si)])
        return self._turret_site_motor_level_raw(si)

    def _turret_motor_commands(self) -> tuple[float, float]:
        """L−R → yaw, low-z−high-z → pitch. Site order: Llo, Lhi, Rlo, Rhi."""
        n = int(TURRET_SITES)
        levels = [self._turret_site_motor_level(si) for si in range(n)]
        while len(levels) < 4:
            levels.append(0.0)
        left = 0.5 * (levels[0] + levels[1])
        right = 0.5 * (levels[2] + levels[3])
        low_z = 0.5 * (levels[0] + levels[2])
        high_z = 0.5 * (levels[1] + levels[3])
        yaw_cmd = float(np.clip(left - right, -1.0, 1.0))
        pitch_cmd = float(np.clip(low_z - high_z, -1.0, 1.0))
        return yaw_cmd, pitch_cmd

    def _tick_turret(self, dt: float) -> None:
        """Effector: motor clamp/pack activity → yaw/pitch (hold if quiet)."""
        if not getattr(self, "turret_on", False):
            return
        # Prefer typed TurretDevice.actuate (handles AI Brain / socket).
        for d in getattr(self, "device_list", []) or []:
            if getattr(d, "kind", "") == "turret" and getattr(d, "enabled", True):
                d.yaw = float(self.turret_yaw)
                d.pitch = float(self.turret_pitch)
                d.actuate(None, dt=float(dt), world=self)
                return
        # Fallback: AI Brain direct, else hop decode.
        port = getattr(self, "sensor_port_nn", None)
        if port is not None and bool(getattr(self, "sensor_port_on", False)):
            if bool(getattr(self, "sensor_port_train", False)) and not port.frozen:
                port.train_step(self)
                yaw_cmd = float(port.last_cmd[0])
                pitch_cmd = float(port.last_cmd[1])
            else:
                yaw_cmd, pitch_cmd = port.commands(self)
        else:
            yaw_cmd, pitch_cmd = self._turret_motor_commands()
        dz = float(TURRET_CMD_DEADZONE)
        if abs(yaw_cmd) < dz:
            yaw_cmd = 0.0
        if abs(pitch_cmd) < dz:
            pitch_cmd = 0.0
        rate_yaw, rate_pitch = turret_slew_rates(self.effective_lamp_speed())
        self.turret_yaw = (
            float(self.turret_yaw) + float(dt) * float(rate_yaw) * yaw_cmd
        ) % math.tau
        self.turret_pitch = float(
            np.clip(
                float(self.turret_pitch)
                + float(dt) * float(rate_pitch) * pitch_cmd,
                TURRET_PITCH_MIN,
                TURRET_PITCH_MAX,
            )
        )
        self._frame_turret_pose = None  # yaw/pitch moved — recompute barrel
        # Plate readout of signed drive (0.5 = neutral).
        self.turret_M = np.asarray(
            [
                float(np.clip(0.5 + 0.5 * yaw_cmd, 0.0, 1.0)),
                float(np.clip(0.5 + 0.5 * pitch_cmd, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )
        self._update_turret_sensors()
        self._update_turret_aim_metric()

    def _update_turret_sensors(self) -> None:
        """Map turret pose → shared yaw/pitch channels in [0, 1]."""
        yaw = float(self.turret_yaw)
        pitch = float(
            np.clip(self.turret_pitch, TURRET_PITCH_MIN, TURRET_PITCH_MAX)
        )
        s_yaw = 0.5 + 0.5 * math.sin(yaw)
        span = max(1e-6, float(TURRET_PITCH_MAX - TURRET_PITCH_MIN))
        s_pitch = (pitch - float(TURRET_PITCH_MIN)) / span
        self.turret_S = np.asarray(
            [np.clip(s_yaw, 0.0, 1.0), np.clip(s_pitch, 0.0, 1.0)],
            dtype=np.float64,
        )

    def _turret_site_sensory_level(self, si: int) -> float:
        """Site sensory: proprioception only (AimTask bearing is teacher, not afferent)."""
        s_yaw = float(self.turret_S[0]) if len(self.turret_S) else 0.5
        s_pitch = float(self.turret_S[1]) if len(self.turret_S) > 1 else 0.5
        yaw_side = s_yaw if si in (0, 1) else (1.0 - s_yaw)
        pitch_side = s_pitch if si in (0, 2) else (1.0 - s_pitch)
        return float(np.clip(0.5 * (yaw_side + pitch_side), 0.0, 1.0))

    def _apply_turret_feedback(self) -> int:
        """Feedback: pose → sensory clamps/packs (and into the graph via seed)."""
        if not getattr(self, "turret_on", False):
            return 0
        # Prefer typed TurretDevice.sense (proprio-only).
        tu = None
        if hasattr(self, "blueprint"):
            tu = self.blueprint.get_turret()
        if tu is not None and tu.enabled:
            levels = tu.sense(self)
            if levels:
                return int(self.fly.stimulate_neurons(levels))
        groups = getattr(self, "turret_sensory_groups", []) or []
        s_terms = getattr(self, "turret_sensory_terminals", []) or []
        if not any(groups) and not s_terms:
            return 0
        self._update_turret_sensors()
        levels: dict[int, float] = {}
        n_sites = int(TURRET_SITES)
        for si in range(n_sites):
            val = self._turret_site_sensory_level(si)
            if si < len(s_terms):
                levels[int(s_terms[si])] = val
            if si < len(groups):
                for nid in groups[si]:
                    levels[int(nid)] = val
        bus = getattr(self, "turret_bus_terminal", None)
        if bus is not None and len(self.turret_S) >= 2:
            levels[int(bus)] = float(
                0.5 * (float(self.turret_S[0]) + float(self.turret_S[1]))
            )
        if not levels:
            return 0
        return int(self.fly.stimulate_neurons(levels))

    def _seed_turret_activity(self) -> int:
        """Adult seed hook: proprioceptive feedback into the network."""
        return self._apply_turret_feedback()

    def _turret_aim_error(self) -> float | None:
        """Angle (rad) between barrel forward and mech-mount→lamp. None if N/A.

        Uses the visual pedestal (same origin as AimTask / aim cone), not the
        tall neuro Fuse mount — otherwise ε reads green while the barrel
        points under the lamp.
        """
        if not getattr(self, "turret_on", False):
            return None
        if not getattr(self, "light_enabled", True) or not getattr(
            self, "lamp_on", False
        ):
            return None
        pose = self._turret_mech_pose()
        if pose is None:
            return None
        self._update_lamp_xyz()
        to_lamp = self._lamp_xyz.astype(np.float64) - np.asarray(
            pose["mount"], dtype=np.float64
        )
        n = float(np.linalg.norm(to_lamp))
        if n < 1e-9:
            return 0.0
        to_lamp = to_lamp / n
        fwd = np.asarray(pose["fwd"], dtype=np.float64)
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            return None
        fwd = fwd / fn
        dot = float(np.clip(np.dot(fwd, to_lamp), -1.0, 1.0))
        return float(math.acos(dot))

    def _update_turret_aim_metric(self) -> None:
        """Cache aim error for plates / legend (bus health readout)."""
        err = self._turret_aim_error()
        self.turret_aim_err = float(err) if err is not None else float("nan")
        self._track_metric_sample()
        # Learning progress once per metric tick (not per badge redraw).
        trk = self._track_snapshot()
        sock_on = bool(getattr(self, "motor_socket_on", False)) and getattr(
            self, "motor_socket", None
        ) is not None
        if sock_on:
            dw = float(getattr(self, "motor_socket_dw_ema", 0.0) or 0.0)
        else:
            dw = float(getattr(self, "learn_dw_ema", 0.0) or 0.0)
        dt = float(getattr(self, "_frame_dt", 1.0 / float(FPS)) or (1.0 / float(FPS)))
        self._learn_progress_note(
            float(trk["score"]), bool(trk["moving"]), dw, dt
        )

    def _track_reset(self) -> None:
        """Clear tracking competence window (new Run / reset / train again)."""
        self._track_samples: deque[tuple[bool, float, float, bool]] = deque(
            maxlen=int(TRACK_BUF_MAX)
        )
        # (in_cone, err_rad, yaw_lag_rad, moving)
        self._track_was_in = False
        self._track_score = float("nan")
        self._track_rms_deg = float("nan")
        self._track_lag_deg = float("nan")
        self._track_slips = 0
        self._track_moving = False
        self._track_n = 0

    def _learn_progress_reset(self) -> None:
        """Clear session learning-progress meters (warn / trained)."""
        self._learn_live_s = 0.0
        self._track_early = float("nan")
        self._track_peak = float("nan")
        self._learn_dw_peak = 0.0
        self._track_ready_hold_s = 0.0
        self._learn_not_learning = False
        self._stim_jitter_phase = 0.0

    def _stim_jitter_active(self) -> bool:
        """±20% Spin/swing wander only during live adult learn (not frozen)."""
        if not bool(getattr(self, "learn_on", False)):
            return False
        if not bool(getattr(self, "turret_on", False)):
            return False
        sock = getattr(self, "motor_socket", None)
        if sock is not None and bool(getattr(sock, "frozen", False)):
            return False
        # Teacher-driven Train: jitter + pursuit = visible jerk / fake lead.
        # Resume jitter after Freeze (pure W competence test).
        rule = getattr(self, "rule", None)
        drive = float(getattr(rule, "socket_teacher_drive", 0.0) or 0.0) if rule else 0.0
        if drive >= 0.5 and sock is not None and not bool(getattr(sock, "frozen", False)):
            return False
        return bool(
            getattr(self, "sim_running", False)
            and getattr(self, "_activity_running", False)
            and self.fly.is_adult()
        )

    def _stim_jitter_factor(self, phase_off: float = 0.0) -> float:
        """1 ± TRACK_STIM_JITTER from slow sine (setpoint stays on slider)."""
        if not self._stim_jitter_active():
            return 1.0
        ph = float(getattr(self, "_stim_jitter_phase", 0.0) or 0.0) + float(phase_off)
        return 1.0 + float(TRACK_STIM_JITTER) * math.sin(ph)

    def _tick_stim_jitter(self, dt: float) -> None:
        if not self._stim_jitter_active():
            return
        self._stim_jitter_phase = (
            float(getattr(self, "_stim_jitter_phase", 0.0) or 0.0)
            + float(dt) * float(TRACK_STIM_JITTER_HZ) * math.tau
        ) % math.tau

    def effective_lamp_speed(self) -> float:
        """Live Spin used by orbit / teacher / Track (setpoint × jitter)."""
        base = float(getattr(self, "lamp_speed", 0.0) or 0.0)
        v = base * self._stim_jitter_factor(0.0)
        return float(np.clip(v, 0.0, float(LAMP_SPEED_MAX)))

    def effective_world_swing_amp(self) -> float:
        base = float(getattr(self, "world_swing_amp", 0.0) or 0.0)
        v = base * self._stim_jitter_factor(2.1)  # out of phase vs Spin
        return float(np.clip(v, 0.0, float(WORLD_SWING_AMP_MAX)))

    def effective_world_swing_hz(self) -> float:
        base = float(getattr(self, "world_swing_hz", 0.0) or 0.0)
        v = base * self._stim_jitter_factor(4.0)
        return float(np.clip(v, 0.0, float(WORLD_SWING_HZ_MAX)))

    def _track_stimulus_moving(self) -> tuple[bool, float]:
        """True when lamp Spin or world swing drives the stimulus."""
        spin = abs(float(self.effective_lamp_speed()))
        if spin >= float(TRACK_MIN_SPIN):
            return True, spin
        swing_on = bool(getattr(self, "world_swing_on", False))
        amp = abs(float(self.effective_world_swing_amp()))
        hz = abs(float(self.effective_world_swing_hz()))
        # Treat non-trivial swing as motion (≥~2° amp, ≥0.05 Hz).
        if swing_on and amp >= math.radians(2.0) and hz >= 0.05:
            return True, max(spin, hz)
        return False, spin

    def _track_window_secs(self) -> float:
        _moving, spin = self._track_stimulus_moving()
        if spin < float(TRACK_MIN_SPIN) and not _moving:
            return float(TRACK_WINDOW_MIN_S)
        rate = max(spin, float(TRACK_MIN_SPIN))
        secs = float(TRACK_WINDOW_REV) / max(rate, 1e-6)
        return float(
            np.clip(secs, float(TRACK_WINDOW_MIN_S), float(TRACK_WINDOW_MAX_S))
        )

    def _track_metric_sample(self) -> None:
        """Append one tracking sample; recompute Track / RMS / lag / slips."""
        if not hasattr(self, "_track_samples"):
            self._track_reset()
        err = float(getattr(self, "turret_aim_err", float("nan")))
        if not math.isfinite(err):
            return
        # Only accumulate during live adult closed-loop (same as learn live).
        if not (
            getattr(self, "turret_on", False)
            and getattr(self, "sim_running", False)
            and getattr(self, "_activity_running", False)
            and self.fly.is_adult()
        ):
            return
        cone = math.radians(float(TURRET_AIM_HALF_DEG))
        in_cone = bool(err <= cone)
        moving, _ = self._track_stimulus_moving()
        # Yaw lag vs teacher desired (lead-aware AimTask).
        lag = float("nan")
        try:
            des = self._turret_desired_aim()
            if des is not None:
                dyaw = float(des[0])
                yaw = float(getattr(self, "turret_yaw", 0.0))
                lag = (dyaw - yaw + math.pi) % math.tau - math.pi
        except Exception:
            lag = float("nan")
        self._track_samples.append(
            (in_cone, float(err), float(lag) if math.isfinite(lag) else 0.0, moving)
        )
        self._track_was_in = in_cone
        # Live signed yaw lag for Trainer (desired − actual); alive in the cone.
        self._yaw_lag = float(lag) if math.isfinite(lag) else float("nan")
        self._track_recompute()


    def _track_recompute(self) -> None:
        buf = getattr(self, "_track_samples", None)
        if not buf:
            self._track_score = float("nan")
            self._track_rms_deg = float("nan")
            self._track_lag_deg = float("nan")
            self._track_slips = 0
            self._track_moving = False
            self._track_n = 0
            return
        # How many recent samples cover the spin window.
        hz = float(getattr(self, "adult_sim_hz", 20.0) or 20.0)
        kn = getattr(self, "_view_detail_knobs", None)
        if callable(kn):
            try:
                hz = float(kn().get("sim_hz", hz) or hz)
            except Exception:
                pass
        n_want = int(
            np.clip(
                self._track_window_secs() * max(hz, 1.0),
                20,
                int(TRACK_BUF_MAX),
            )
        )
        samples = list(buf)[-n_want:]
        moving = [s for s in samples if s[3]]
        self._track_moving = len(moving) >= max(10, n_want // 4)
        use = moving if self._track_moving else samples
        self._track_n = len(use)
        if not use:
            self._track_score = float("nan")
            self._track_rms_deg = float("nan")
            self._track_lag_deg = float("nan")
            self._track_slips = 0
            return
        in_flags = [1.0 if s[0] else 0.0 for s in use]
        errs = np.asarray([s[1] for s in use], dtype=np.float64)
        lags = np.asarray([s[2] for s in use], dtype=np.float64)
        self._track_score = float(np.mean(in_flags))
        self._track_rms_deg = float(np.sqrt(np.mean(errs * errs)) * (180.0 / math.pi))
        self._track_lag_deg = float(np.mean(np.abs(lags)) * (180.0 / math.pi))
        slips = 0
        prev = use[0][0]
        for s in use[1:]:
            if prev and not s[0]:
                slips += 1
            prev = s[0]
        self._track_slips = int(slips)

    def _track_snapshot(self) -> dict:
        """Read-only tracking competence for Learn badge / detail."""
        if not hasattr(self, "_track_samples"):
            self._track_reset()
        _moving, spin = self._track_stimulus_moving()
        return {
            "score": float(getattr(self, "_track_score", float("nan"))),
            "rms_deg": float(getattr(self, "_track_rms_deg", float("nan"))),
            "lag_deg": float(getattr(self, "_track_lag_deg", float("nan"))),
            "slips": int(getattr(self, "_track_slips", 0) or 0),
            "n": int(getattr(self, "_track_n", 0) or 0),
            "moving": bool(getattr(self, "_track_moving", False)),
            "spin": float(spin),
            "window_s": float(self._track_window_secs()),
        }

    def _turret_aim_score(self) -> float:
        """1 = on lamp, 0 = ≥90° off (or unavailable)."""
        err = float(self.turret_aim_err)
        if not math.isfinite(err):
            return 0.0
        return float(np.clip(1.0 - err / (0.5 * math.pi), 0.0, 1.0))

    def _ensure_aim_task(self):
        task = getattr(self, "_aim_task", None)
        if task is None:
            from aim_task import AimTask

            task = AimTask()
            self._aim_task = task
        return task

    def _turret_desired_aim(self) -> tuple[float, float] | None:
        """Lamp-relative yaw/pitch the barrel should hold (AimTask teacher)."""
        return self._ensure_aim_task().desired_aim(self)

    def _turret_teacher_site_levels(self) -> list[float] | None:
        """Desired motor activity per wing site (= AimTask bearing coding)."""
        return self._ensure_aim_task().teacher_site_levels(self)

    def _turret_motor_neuron_targets(self) -> dict[int, float]:
        """Map motor clamp/pack ids → teacher target activity."""
        return self._ensure_aim_task().motor_neuron_targets(self)

    def _turret_bearing_site_levels(self) -> list[float] | None:
        """Legacy alias: teacher bearing coding (not injected as sensory afferent)."""
        return self._turret_teacher_site_levels()

    def _apply_aim_teacher_mix(self) -> int:
        """Soft demonstrator: blend motor clamps toward lamp targets (fades as ε→0)."""
        if not self.learn_on or not self.turret_on:
            return 0
        targets = self._turret_motor_neuron_targets()
        if not targets:
            return 0
        err = float(self.turret_aim_err)
        if not math.isfinite(err):
            self._update_turret_aim_metric()
            err = float(self.turret_aim_err)
        if not math.isfinite(err):
            return 0
        mix = float((getattr(self, "rule", None) or _DEFAULT_RULE).teacher_mix) * float(
            np.clip(err / (0.5 * math.pi), 0.0, 1.0)
        )
        if mix < 0.02:
            return 0
        levels: dict[int, float] = {}
        for nid, tgt in targets.items():
            neu = self.fly.neurons.get(int(nid))
            if neu is None:
                continue
            cur = float(neu.activity)
            levels[int(nid)] = float((1.0 - mix) * cur + mix * tgt)
        if not levels:
            return 0
        n = int(self.fly.stimulate_neurons(levels))
        if n > 0:
            # Teacher pulse counts as a visible learn flash (motor chain head).
            ids = np.fromiter(levels.keys(), dtype=np.int32, count=len(levels))
            self._learn_flash_pre = ids
            self._learn_flash_post = ids
            self._learn_flash_amp = max(float(self._learn_flash_amp), 0.55 + 0.45 * mix)
            self._learn_flash_n = max(int(self._learn_flash_n), int(n))
        return n

    def _plasticity_aim_step(self, prev_err: float) -> None:
        """Delegate to Trainer (RPE, carve×sign(rpe), temporal elig traces)."""
        tr = getattr(self, "_trainer", None)
        if tr is None:
            from trainer import Trainer

            tr = Trainer(self, task=self._ensure_aim_task())
            self._trainer = tr
            loop = getattr(self, "_closed_loop", None)
            if loop is not None:
                loop.trainer = tr
        tr.step(float(prev_err))

    def _motor_eligibility_upstream(
        self, motor_ids: set[int], *, depth: int = 8
    ) -> set[int]:
        """Neurons that can reach motor posts within `depth` reverse hops.

        When ``hop_allow_ids`` is set (Sample/scope), stay inside that set so Learn
        does not credit CNS outside the funnel ∪.
        """
        if not motor_ids:
            return set()
        pre, post = self.fly.activity_graph_edges()
        if len(pre) == 0:
            return set(motor_ids)
        depth_i = max(1, int(depth))
        motors_key = frozenset(int(m) for m in motor_ids)
        graph_key = getattr(self.fly, "_activity_graph_key", None)
        allow_raw = getattr(self.fly, "hop_allow_ids", None)
        allow_id = id(allow_raw) if allow_raw is not None and len(allow_raw) else 0
        up_key = (graph_key, allow_id, depth_i, motors_key)
        cached_up = getattr(self, "_elig_upstream_cache", None)
        if cached_up is not None and getattr(self, "_elig_upstream_key", None) == up_key:
            return set(cached_up)

        allow: set[int] | None = None
        allow_cache = getattr(self, "_elig_allow_set", None)
        allow_cache_id = getattr(self, "_elig_allow_id", None)
        if allow_raw is not None and len(allow_raw) > 0:
            raw_id = id(allow_raw)
            if allow_cache is not None and allow_cache_id == raw_id:
                allow = allow_cache
            else:
                allow = {int(x) for x in np.asarray(allow_raw, dtype=np.int64).tolist()}
                self._elig_allow_set = allow
                self._elig_allow_id = raw_id
        rev = self.fly.reverse_activity_adj(allow)
        upstream: set[int] = set(motors_key)
        frontier = list(upstream)
        for _ in range(depth_i):
            nxt: list[int] = []
            for node in frontier:
                for src in rev.get(int(node), ()):
                    if src in upstream:
                        continue
                    if allow is not None and src not in allow:
                        continue
                    upstream.add(int(src))
                    nxt.append(int(src))
            frontier = nxt
            if not frontier:
                break
            if len(upstream) > 8000:
                break
        self._elig_upstream_key = up_key
        self._elig_upstream_cache = frozenset(upstream)
        return upstream

    def _debug_device_path_report(self, *, max_hops: int = 16) -> dict:
        """BFS diagnostics: ocelli → motors / sensory → motors + funnel A/B/∩ gate."""
        self._ensure_atlas_recipe()
        pre, post = self.fly.activity_graph_edges()
        adj: dict[int, list[int]] = {}
        for a, b in zip(pre.tolist(), post.tolist()):
            adj.setdefault(int(a), []).append(int(b))

        def _bfs_dist(starts: list[int]) -> dict[int, int]:
            dist = {int(s): 0 for s in starts}
            q = list(dist.keys())
            i = 0
            while i < len(q):
                u = q[i]
                i += 1
                if dist[u] >= int(max_hops):
                    continue
                for v in adj.get(u, ()):
                    if v in dist:
                        continue
                    dist[v] = dist[u] + 1
                    q.append(v)
            return dist

        oc_src = [int(x) for x in (getattr(self, "ocelli_terminals", []) or [])]
        if not oc_src:
            oc_src = list(self._ocelli_flat_ids())
        sens_src = [
            int(x) for x in (getattr(self, "turret_sensory_terminals", []) or [])
        ]
        motors = list(self._turret_motor_flat_ids())
        d_oc = _bfs_dist(oc_src) if oc_src else {}
        d_sens = _bfs_dist(sens_src) if sens_src else {}

        def _motor_hits(dist: dict[int, int]) -> dict:
            hits = []
            for m in motors:
                if int(m) in dist:
                    hits.append((int(m), int(dist[int(m)])))
            hits.sort(key=lambda t: t[1])
            return {
                "hit": len(hits),
                "motors": len(motors),
                "best_hops": int(hits[0][1]) if hits else None,
                "median_hops": (
                    float(np.median([h for _, h in hits])) if hits else None
                ),
                "sample": hits[:8],
            }

        wells = None
        try:
            layers = self._funnel_layer_sets()
            wf = getattr(self, "_well_field", None)
            if layers or (wf is not None and wf.specs):
                met = (wf.metrics_dict() if wf is not None else {})
                inter = (layers or {}).get("inter", set())
                wells = {
                    "A": int(met.get("A", len((layers or {}).get("ocelli", ())))),
                    "B": int(met.get("B", len((layers or {}).get("turret", ())))),
                    "N": int(met.get("N", len(inter))),
                    "neck_class": len(self._neck_class_ids_in(inter)) if inter else 0,
                    "names": list(met.get("names", [])),
                }
            # Graph training scope meters (not geometric gate).
            scope = getattr(self, "_training_scope", None)
            if scope is None or not getattr(scope, "path_neurons", None):
                try:
                    scope = self.build_training_scope(with_wells=False)
                except Exception:
                    scope = None
            scope_met = scope.metrics() if scope is not None else None
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            wells = {"error": f"{type(exc).__name__}: {exc}"}
            scope_met = None

        # Shortest reconstructed path ocelli→first motor hit (ids only).
        path_ids: list[int] = []
        oc_hit = _motor_hits(d_oc)
        if oc_hit["hit"] and oc_src:
            target = int(oc_hit["sample"][0][0])
            # Reconstruct via parent map.
            parent: dict[int, int | None] = {int(s): None for s in oc_src}
            q = list(parent.keys())
            i = 0
            found = False
            while i < len(q) and not found:
                u = q[i]
                i += 1
                for v in adj.get(u, ()):
                    if v in parent:
                        continue
                    parent[v] = u
                    if int(v) == target:
                        found = True
                        break
                    q.append(v)
            if found:
                cur: int | None = target
                while cur is not None:
                    path_ids.append(int(cur))
                    cur = parent.get(cur)
                path_ids.reverse()

        return {
            "edges": int(len(pre)),
            "ocelli_src": len(oc_src),
            "sensory_src": len(sens_src),
            "motors": len(motors),
            "ocelli_to_motor": oc_hit,
            "sensory_to_motor": _motor_hits(d_sens),
            "example_path": path_ids[:24],
            "example_path_len": len(path_ids),
            "wells": wells,
            "scope": (
                {
                    "ok": bool(scope_met.ok),
                    "paths": f"{scope_met.motors_hit}/{scope_met.n_motors}",
                    "n": int(scope_met.path_neurons),
                    "e": int(scope_met.path_edges),
                    "best_hops": scope_met.best_hops,
                    "hops_floor": scope_met.hops_floor,
                    "group_hops": dict(scope_met.group_best_hops),
                    "readout": scope_met.readout(),
                }
                if scope_met is not None
                else None
            ),
            # Legacy key for learn_console path printer.
            "funnel": wells,
        }

    def _probe_ocelli_motor_reach(
        self, *, hops: int = 12, hop_gain: float = 0.9
    ) -> dict:
        """Frozen-weight check: does ocelli activity reach turret motors?"""
        motors = list(self._turret_motor_flat_ids())
        if not motors:
            m_terms = getattr(self, "turret_motor_terminals", []) or []
            groups = getattr(self, "turret_motor_groups", []) or []
            motors = [int(x) for x in m_terms]
            for g in groups:
                motors.extend(int(x) for x in g)
        motors = sorted(set(motors))
        saved_gains = dict(getattr(self.fly, "contact_gain", {}) or {})
        self.fly.contact_gain = {}
        self.fly.clear_activity()
        n_seed = int(self._seed_ocelli_activity()) if self.ocelli_on else 0
        reached = {m: 0.0 for m in motors}
        for _ in range(max(1, int(hops))):
            self.fly.tick_activity(hop_gain=float(hop_gain), decay=0.97)
            self.fly.rebuild_activity_cache()
            for m in motors:
                neu = self.fly.neurons.get(int(m))
                if neu is not None:
                    reached[m] = max(reached[m], float(neu.activity))
        self.fly.contact_gain = saved_gains
        hit = {m: a for m, a in reached.items() if a > 0.02}
        return {
            "seed": n_seed,
            "motors": len(motors),
            "hit": len(hit),
            "best": float(max(reached.values()) if reached else 0.0),
            "mean_hit": float(sum(hit.values()) / len(hit)) if hit else 0.0,
            "ok": len(hit) >= max(1, len(motors) // 4),
        }

    def _probe_turret_sensory_motor_reach(
        self, *, hops: int = 12, hop_gain: float = 0.9
    ) -> dict:
        """Frozen-weight: turret sensory (optical) → neck → motors (no ocelli)."""
        motors = list(self._turret_motor_flat_ids())
        motors = sorted(set(motors))
        saved_gains = dict(getattr(self.fly, "contact_gain", {}) or {})
        self.fly.contact_gain = {}
        self.fly.clear_activity()
        self._update_lamp_xyz()
        self._update_turret_aim_metric()
        n_seed = int(self._apply_turret_feedback()) if self.turret_on else 0
        reached = {m: 0.0 for m in motors}
        for _ in range(max(1, int(hops))):
            self.fly.tick_activity(hop_gain=float(hop_gain), decay=0.97)
            self.fly.rebuild_activity_cache()
            for m in motors:
                neu = self.fly.neurons.get(int(m))
                if neu is not None:
                    reached[m] = max(reached[m], float(neu.activity))
        self.fly.contact_gain = saved_gains
        hit = {m: a for m, a in reached.items() if a > 0.02}
        # Lateralization: L vs R site mean activity gap.
        m_terms = getattr(self, "turret_motor_terminals", []) or []
        left = right = 0.0
        if len(m_terms) >= 4:
            left = 0.5 * (
                float(reached.get(int(m_terms[0]), 0.0))
                + float(reached.get(int(m_terms[1]), 0.0))
            )
            right = 0.5 * (
                float(reached.get(int(m_terms[2]), 0.0))
                + float(reached.get(int(m_terms[3]), 0.0))
            )
        return {
            "seed": n_seed,
            "motors": len(motors),
            "hit": len(hit),
            "best": float(max(reached.values()) if reached else 0.0),
            "mean_hit": float(sum(hit.values()) / len(hit)) if hit else 0.0,
            "lr_gap": float(left - right),
            "ok": len(hit) >= max(1, len(motors) // 4),
        }

    def _turret_motor_flat_ids(self) -> list[int]:
        out: list[int] = []
        for g in getattr(self, "turret_motor_groups", []) or []:
            out.extend(int(x) for x in g)
        out.extend(int(x) for x in (getattr(self, "turret_motor_terminals", []) or []))
        return out

    def _place_turret_clusters(
        self, *, move_only: bool = False
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Per wing site: motor/sensory disks + virtual clamps on the pads."""
        pose = self._turret_rig_pose()
        if pose is None:
            return None
        r = float(pose["r"])
        cl_r = float(TURRET_CLUSTER_FRAC) * r
        right = pose["right"]
        up = pose["up"]
        fwd = pose["fwd"]
        sites = pose["sites"]
        mount = pose["mount"]

        def _disk(origin: np.ndarray, n: int, seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            pos = np.zeros((n, 3), dtype=np.float64)
            ang = np.linspace(0.0, math.tau, n, endpoint=False)
            ang = ang + float(rng.uniform(0, 0.12))
            rad = cl_r * np.sqrt(rng.uniform(0.15, 1.0, size=n))
            depth = cl_r * rng.uniform(0.05, 0.35, size=n)
            pos[:] = (
                origin[None, :]
                + rad[:, None] * np.cos(ang)[:, None] * right[None, :]
                + rad[:, None] * np.sin(ang)[:, None] * up[None, :]
                - depth[:, None] * fwd[None, :]
            )
            return pos.astype(np.float32)

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for si, site in enumerate(sites):
            motor_origin = site - fwd * (0.01 * r) - right * (0.01 * r)
            sens_origin = site + up * (0.012 * r) + right * (
                0.012 * r * (1 if si % 2 == 0 else -1)
            )
            motor_pos = _disk(motor_origin, int(TURRET_MOTOR_N), 40 + si)
            sens_pos = _disk(sens_origin, int(TURRET_SENSORY_N), 80 + si)
            out.append((motor_pos, sens_pos))
            if move_only:
                if si < len(self.turret_motor_groups):
                    for j, nid in enumerate(self.turret_motor_groups[si]):
                        neu = self.fly.neurons.get(int(nid))
                        if neu is not None and j < len(motor_pos):
                            neu.soma = motor_pos[j]
                if si < len(self.turret_sensory_groups):
                    for j, nid in enumerate(self.turret_sensory_groups[si]):
                        neu = self.fly.neurons.get(int(nid))
                        if neu is not None and j < len(sens_pos):
                            neu.soma = sens_pos[j]
                # Virtual clamps sit on the pad centres.
                if si < len(getattr(self, "turret_motor_terminals", [])):
                    neu = self.fly.neurons.get(int(self.turret_motor_terminals[si]))
                    if neu is not None:
                        neu.soma = site.astype(np.float32)
                if si < len(getattr(self, "turret_sensory_terminals", [])):
                    neu = self.fly.neurons.get(int(self.turret_sensory_terminals[si]))
                    if neu is not None:
                        neu.soma = (site + up * (0.02 * r)).astype(np.float32)
        bus = getattr(self, "turret_bus_terminal", None)
        if move_only and bus is not None:
            neu = self.fly.neurons.get(int(bus))
            if neu is not None:
                neu.soma = mount.astype(np.float32)
        return out

    def _strip_turret_device_wires(self, ids: set[int]) -> None:
        for nid in ids:
            neu = self.fly.neurons.get(int(nid))
            if neu is None:
                continue
            neu.axons = [
                ax
                for ax in neu.axons
                if ax.partner_id is None or int(ax.partner_id) not in ids
            ]
            neu.dendrites = [
                dn
                for dn in neu.dendrites
                if dn.partner_id is None or int(dn.partner_id) not in ids
            ]

    def _wire_turret_device(self) -> None:
        """Pack meshes ↔ site clamps; no cross-site bus (keeps L/R differentials)."""
        flat = set(self._turret_flat_ids())
        if len(flat) < 2:
            return
        self._strip_turret_device_wires(flat)
        n_sites = int(TURRET_SITES)
        m_terms = [int(t) for t in getattr(self, "turret_motor_terminals", []) or []]
        s_terms = [int(t) for t in getattr(self, "turret_sensory_terminals", []) or []]
        for si in range(n_sites):
            m = (
                [int(x) for x in self.turret_motor_groups[si]]
                if si < len(self.turret_motor_groups)
                else []
            )
            s = (
                [int(x) for x in self.turret_sensory_groups[si]]
                if si < len(self.turret_sensory_groups)
                else []
            )
            self._device_wire_mesh(m)
            self._device_wire_mesh(s)
            if si < len(m_terms):
                self._device_wire_star(m_terms[si], m)
            if si < len(s_terms):
                self._device_wire_star(s_terms[si], s)
            # Pad: motor→sensory efference only — no sensory→motor shortcut.
            # Aim drive must hop sensory → Fuse/CNS → motor Fuse (neck path).
            if si < len(m_terms) and si < len(s_terms):
                self.fly.connect_pair(m_terms[si], s_terms[si])
        # bus clamp stays on the device for mount/UI; not a cross-site equalizer.

    def _ensure_turret_neurons(self) -> bool:
        """Create/update 4 wing-site packs + virtual clamps + bus."""
        if not self.turret_on or not self.fly.neurons or not self.class_names:
            return False
        pose = self._turret_rig_pose()
        if pose is None:
            return False
        try:
            motor_cls = self.class_names.index("motor")
        except ValueError:
            motor_cls = 0
        try:
            sens_cls = self.class_names.index("sensory")
        except ValueError:
            sens_cls = 0
        n_sites = int(TURRET_SITES)
        need_m = n_sites * int(TURRET_MOTOR_N)
        need_s = n_sites * int(TURRET_SENSORY_N)
        flat = self._turret_flat_ids()
        alive_m = sum(
            1
            for g in self.turret_motor_groups
            for i in g
            if i in self.fly.neurons
        )
        alive_s = sum(
            1
            for g in self.turret_sensory_groups
            for i in g
            if i in self.fly.neurons
        )
        alive_mt = sum(
            1
            for i in getattr(self, "turret_motor_terminals", []) or []
            if i in self.fly.neurons
        )
        alive_st = sum(
            1
            for i in getattr(self, "turret_sensory_terminals", []) or []
            if i in self.fly.neurons
        )
        bus = getattr(self, "turret_bus_terminal", None)
        alive_bus = 1 if bus is not None and bus in self.fly.neurons else 0
        bad = (
            alive_m != need_m
            or alive_s != need_s
            or alive_mt != n_sites
            or alive_st != n_sites
            or alive_bus != 1
            or len(self.turret_motor_groups) != n_sites
            or len(self.turret_sensory_groups) != n_sites
        )
        if bad:
            for i in flat:
                self.fly.neurons.pop(i, None)
            for s in self.fly.sets.values():
                s.members.difference_update(flat)
            self.turret_motor_groups = [[] for _ in range(n_sites)]
            self.turret_sensory_groups = [[] for _ in range(n_sites)]
            self.turret_motor_terminals = []
            self.turret_sensory_terminals = []
            self.turret_bus_terminal = None
            placed = self._place_turret_clusters(move_only=False)
            if placed is None or len(placed) != n_sites:
                return False
            sites = pose["sites"]
            mount = pose["mount"]
            up = pose["up"]
            r = float(pose["r"])
            for si, (motor_pos, sens_pos) in enumerate(placed):
                mg: list[int] = []
                sg: list[int] = []
                for j in range(len(motor_pos)):
                    n = self.fly.add_neuron(
                        motor_pos[j], motor_cls, set_name="motor"
                    )
                    mg.append(int(n.id))
                for j in range(len(sens_pos)):
                    n = self.fly.add_neuron(
                        sens_pos[j], sens_cls, set_name="sensory"
                    )
                    sg.append(int(n.id))
                self.turret_motor_groups[si] = mg
                self.turret_sensory_groups[si] = sg
                mt = self.fly.add_neuron(
                    sites[si].astype(np.float32), motor_cls, set_name="motor"
                )
                st = self.fly.add_neuron(
                    (sites[si] + up * (0.02 * r)).astype(np.float32),
                    sens_cls,
                    set_name="sensory",
                )
                self.turret_motor_terminals.append(int(mt.id))
                self.turret_sensory_terminals.append(int(st.id))
            bus_n = self.fly.add_neuron(
                mount.astype(np.float32), motor_cls, set_name="motor"
            )
            self.turret_bus_terminal = int(bus_n.id)
            self._wire_turret_device()
            self.fly.rebuild_caches()
            self._sync_from_fly()
            try:
                self.focus_basin_indices.add(motor_cls)
                self.focus_basin_indices.add(sens_cls)
                self.focus_soma_indices.add(motor_cls)
                self.focus_soma_indices.add(sens_cls)
                self._sync_layers_from_focus()
                self._basin_cache_key = None
            except Exception:
                pass
            return True
        self._place_turret_clusters(move_only=True)
        self._wire_turret_device()
        self.fly.rebuild_caches()
        self._basin_cache_key = None
        return True

    def _turret_pick_targets(self) -> list[tuple[list[int], list[int]]] | None:
        """Per site: CNS DN/motor → motor clamp (efferent); sensory → AN (afferent)."""
        pose = self._turret_rig_pose()
        if pose is None:
            return None
        sites = pose["sites"]
        ban = set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
        used: set[int] = set()
        per = max(1, int(TURRET_TIPS_PER_SITE))
        plan = self.stereo_plan()
        pairs: list[tuple[list[int], list[int]]] = []
        for si, site in enumerate(sites):
            probe = np.asarray(site, dtype=np.float64)
            motor_sources = plan.sources_for(si) if plan is not None else None
            if motor_sources:
                used.update(int(g) for g in motor_sources)
            else:
                motor_sources = self._pick_cns_near(
                    probe,
                    per,
                    prefer=FUSE_MOTOR_CLASSES,
                    fallback=FUSE_NECK_FALLBACK,
                    ban=ban,
                    used=used,
                )
            sens_goals = self._pick_cns_near(
                probe,
                per,
                prefer=FUSE_SENSORY_CLASSES,
                fallback=FUSE_NECK_FALLBACK,
                ban=ban,
                used=used,
            )
            if not motor_sources or not sens_goals:
                return None
            while len(motor_sources) < per:
                motor_sources.append(motor_sources[-1])
            while len(sens_goals) < per:
                sens_goals.append(sens_goals[-1])
            pairs.append((motor_sources, sens_goals))
        return pairs

    def _arm_turret_tips(self, *, missing_only: bool = False) -> int:
        """Efferent CNS→motor clamps + afferent sensory→AN/DN.

        ``missing_only``: arm only wing sites that are not yet fused (Fuse mode).
        """
        if not self.turret_on:
            return 0
        if self.fly.is_adult() and not missing_only:
            return 0
        if not self._ensure_turret_neurons():
            return 0
        flat = np.asarray(self._turret_flat_ids(), dtype=np.int32)
        m_terms_arr = np.asarray(
            getattr(self, "turret_motor_terminals", []) or [], dtype=np.int32
        )
        if not missing_only and len(self.fly.growth_owner):
            # Already growing from packs or into motor clamps.
            if len(flat) and np.any(np.isin(self.fly.growth_owner, flat)):
                return 0
            if len(m_terms_arr) and len(self.fly.growth_goal):
                if np.any(np.isin(self.fly.growth_goal, m_terms_arr)):
                    return 0
                if np.any(np.isin(self.fly.growth_owner, m_terms_arr)):
                    return 0
        pairs = self._turret_pick_targets()
        m_terms = getattr(self, "turret_motor_terminals", []) or []
        s_terms = getattr(self, "turret_sensory_terminals", []) or []
        if (
            pairs is None
            or len(pairs) != int(TURRET_SITES)
            or len(m_terms) != int(TURRET_SITES)
            or len(s_terms) != int(TURRET_SITES)
        ):
            return 0
        conn = (
            self._turret_connected_mask()
            if missing_only
            else [False] * int(TURRET_SITES)
        )
        owners: list[int] = []
        goals: list[int] = []
        for si, (src_m, goal_s) in enumerate(pairs):
            if missing_only and si < len(conn) and conn[si]:
                continue
            if missing_only and (
                self._device_owner_has_open_tip(int(m_terms[si]))
                or self._device_owner_has_open_tip(int(s_terms[si]))
                or any(self._device_owner_has_open_tip(int(g)) for g in src_m)
            ):
                continue
            # Efferent: CNS DN/motor → turret motor clamp.
            for g in src_m:
                owners.append(int(g))
                goals.append(int(m_terms[si]))
            # Afferent: sensory clamp → ascending/central.
            for g in goal_s:
                owners.append(int(s_terms[si]))
                goals.append(int(g))
        if not owners:
            return 0
        return int(
            self.fly.append_growth_tips(
                np.asarray(owners, dtype=np.int32),
                np.asarray(goals, dtype=np.int32),
                seek_limit=float(OCELLI_FUSE_SEEK),
            )
        )

    def _turret_connected_mask(self) -> list[bool]:
        """One flag per wing site (clamp tip fused into CNS)."""
        key = self._fuse_mask_cache_key()
        cached = getattr(self, "_fuse_mask_cache", None)
        if (
            isinstance(cached, dict)
            and cached.get("key") == key
            and "turret" in cached
        ):
            return list(cached["turret"])
        n_sites = int(TURRET_SITES)
        out = [False] * n_sites
        m_terms = getattr(self, "turret_motor_terminals", []) or []
        s_terms = getattr(self, "turret_sensory_terminals", []) or []
        for si in range(n_sites):
            gids = []
            if si < len(self.turret_motor_groups):
                gids.extend(self.turret_motor_groups[si])
            if si < len(self.turret_sensory_groups):
                gids.extend(self.turret_sensory_groups[si])
            if si < len(m_terms):
                gids.append(int(m_terms[si]))
            if si < len(s_terms):
                gids.append(int(s_terms[si]))
            out[si] = self._owners_fused_to_cns(gids)
        if not isinstance(cached, dict):
            cached = {}
            self._fuse_mask_cache = cached
        if cached.get("key") != key:
            cached.clear()
            cached["key"] = key
        cached["turret"] = list(out)
        return out

    def _draw_turret(self, cx: float, cy: float, view_lim_y: int) -> None:
        """Pedestal + barrel + 4 wing pads; packs use normal Basins layer."""
        assert self._view_surf is not None
        if not self.turret_on:
            return
        pose = self._turret_mech_pose()
        if pose is None:
            return
        connected = self._turret_connected_mask()
        self._draw_turret_group_box(pose, connected, cx, cy, view_lim_y)
        overlay = self._overlay_surface(view_lim_y)
        r = float(pose["r"])
        mount = pose["mount"]
        barrel = pose["barrel"]
        fwd = pose["fwd"]
        right = pose["right"]
        up = pose["up"]
        sites = pose["sites"]
        labels = ("LA", "LP", "RA", "RP")
        for si, site in enumerate(sites):
            sxa, sya = project_3d_batch(
                site.reshape(1, 3).astype(np.float32),
                self.view_angle_x,
                self.view_angle_y,
                cx,
                cy,
                self.scale_3d,
            )
            ax, ay = int(round(float(sxa[0]))), int(round(float(sya[0])))
            if not (0 <= ax < self.view_w and 0 <= ay < view_lim_y):
                continue
            ok = connected[si] if si < len(connected) else False
            col = (120, 220, 150, 200) if ok else (230, 120, 120, 200)
            pygame.draw.circle(overlay, col, (ax, ay), 4)
            pygame.draw.circle(overlay, (255, 200, 160, 140), (ax, ay), 7, 1)
            tag = labels[si] if si < len(labels) else str(si)
            lab = self.hint_font.render(tag + ("*" if ok else ""), True, col[:3])
            overlay.blit(lab, (ax + 6, ay - lab.get_height() // 2))
            # Strut from site pad up to the turret mount.
            shaft = np.stack([site, mount], axis=0).astype(np.float32)
            sxs, sys_ = project_3d_batch(
                shaft, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
            )
            sp = [
                (int(round(float(x))), int(round(float(y))))
                for x, y in zip(sxs, sys_)
            ]
            if len(sp) >= 2:
                pygame.draw.lines(overlay, (200, 140, 110, 120), False, sp, 1)
        rad = 0.035 * r
        ang = np.linspace(0.0, math.tau, 20, endpoint=False)
        ring = (
            mount[None, :]
            + rad * np.cos(ang)[:, None] * right[None, :]
            + rad * np.sin(ang)[:, None] * up[None, :]
        ).astype(np.float32)
        sx, sy = project_3d_batch(
            ring, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
        )
        pts = [
            (int(round(float(x))), int(round(float(y))))
            for x, y in zip(sx, sy)
            if 0 <= x < self.view_w and 0 <= y < view_lim_y
        ]
        if len(pts) >= 3:
            pygame.draw.polygon(overlay, (180, 110, 90, 90), pts, 0)
            pygame.draw.polygon(overlay, (230, 160, 120, 180), pts, 1)
        tip = barrel + fwd * (0.06 * r)
        shaft = np.stack([mount, barrel, tip], axis=0).astype(np.float32)
        sxs, sys_ = project_3d_batch(
            shaft, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
        )
        sp = [
            (int(round(float(x))), int(round(float(y))))
            for x, y in zip(sxs, sys_)
        ]
        if len(sp) >= 2:
            pygame.draw.lines(overlay, (240, 180, 120, 220), False, sp, 3)
            pygame.draw.circle(overlay, (255, 220, 160, 230), sp[-1], 4)
        self._draw_turret_aim(
            pose,
            tip.astype(np.float64),
            overlay,
            cx,
            cy,
            view_lim_y,
        )
        sxb, syb = project_3d_batch(
            barrel.reshape(1, 3).astype(np.float32),
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        bx = int(round(float(sxb[0])))
        by = int(round(float(syb[0])))
        rows: list[tuple[str, float, tuple[int, int, int]]] = []
        aim = float(self.turret_aim_err)
        if math.isfinite(aim):
            aim_deg = aim * (180.0 / math.pi)
            score = self._turret_aim_score()
            if aim_deg < 15.0:
                aim_rgb = (120, 220, 150)
            elif aim_deg < 45.0:
                aim_rgb = (220, 200, 120)
            else:
                aim_rgb = (230, 140, 120)
            rows.append((f"ε {aim_deg:.0f}°", score, aim_rgb))
        rows.extend(
            (
                (f"Y {float(self.turret_S[0]):.2f}", float(self.turret_S[0]), (220, 200, 140)),
                (f"P {float(self.turret_S[1]):.2f}", float(self.turret_S[1]), (200, 180, 220)),
                (f"My {float(self.turret_M[0]):.2f}", float(self.turret_M[0]), (230, 140, 120)),
                (f"Mp {float(self.turret_M[1]):.2f}", float(self.turret_M[1]), (230, 160, 140)),
            )
        )
        y_off = -28
        for txt, val, rgb in rows:
            lab = self.hint_font.render(txt, True, (235, 230, 220))
            pad_x, pad_y = 5, 2
            pw = lab.get_width() + pad_x * 2
            ph = lab.get_height() + pad_y * 2
            px = int(np.clip(bx + 10, 2, max(2, self.view_w - pw - 2)))
            py = int(np.clip(by + y_off, 2, max(2, view_lim_y - ph - 2)))
            plate = pygame.Surface((pw, ph), pygame.SRCALPHA)
            pygame.draw.rect(
                plate, (22, 18, 16, 200), plate.get_rect(), border_radius=3
            )
            pygame.draw.rect(
                plate, (*rgb, 210), plate.get_rect(), width=1, border_radius=3
            )
            bar_w = max(1, int((pw - 4) * np.clip(val, 0.0, 1.0)))
            pygame.draw.rect(plate, (*rgb, 160), pygame.Rect(2, ph - 3, bar_w, 2))
            plate.blit(lab, (pad_x, pad_y))
            overlay.blit(plate, (px, py))
            y_off += ph + 3
        self._view_surf.blit(overlay, (0, 0))

    def _draw_turret_aim(
        self,
        pose: dict,
        tip: np.ndarray,
        overlay: pygame.Surface,
        cx: float,
        cy: float,
        view_lim_y: int,
    ) -> None:
        """Barrel aim cone + target / hit points (Turret panel · Aim cone)."""
        if not bool(getattr(self, "show_turret_aim", False)):
            return
        if float(self._view_detail_knobs()["overlay"]) < 0.35:
            return
        fwd = np.asarray(pose["fwd"], dtype=np.float64)
        right = np.asarray(pose["right"], dtype=np.float64)
        up = np.asarray(pose["up"], dtype=np.float64)
        r = float(pose["r"])
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9 or r < 1e-9:
            return
        fwd = fwd / fn
        half = math.radians(float(TURRET_AIM_HALF_DEG))
        cone_len = float(TURRET_AIM_CONE_LEN) * r
        aim = float(self.turret_aim_err)
        if math.isfinite(aim):
            if aim < half:
                rgb = (120, 220, 150)
            elif aim < 3.0 * half:
                rgb = (220, 200, 120)
            else:
                rgb = (230, 140, 120)
        else:
            rgb = (200, 170, 130)

        n_ring = 16
        n_steps = 4
        ang = np.linspace(0.0, math.tau, n_ring, endpoint=False)
        cos_a = np.cos(ang)[:, None]
        sin_a = np.sin(ang)[:, None]
        rings_xyz: list[np.ndarray] = []
        for s in range(1, n_steps + 1):
            t = s / float(n_steps)
            d = cone_len * t
            rr = d * math.tan(half)
            c = tip + fwd * d
            rings_xyz.append(
                (
                    c[None, :]
                    + rr * cos_a * right[None, :]
                    + rr * sin_a * up[None, :]
                ).astype(np.float32)
            )
        rim = rings_xyz[-1]
        # Apex + rings + 4 spokes (cardinal) + axis tip.
        spoke_idx = (0, n_ring // 4, n_ring // 2, 3 * n_ring // 4)
        spoke_xyz = np.stack(
            [tip] + [rim[i] for i in spoke_idx] + [tip + fwd * cone_len],
            axis=0,
        ).astype(np.float32)
        all_xyz = np.concatenate(rings_xyz + [spoke_xyz], axis=0)
        sx, sy = project_3d_batch(
            all_xyz,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )

        def _pts(xs, ys) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            for x, y in zip(xs, ys):
                if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                    continue
                out.append((int(round(float(x))), int(round(float(y)))))
            return out

        off = 0
        for s in range(n_steps):
            poly = _pts(sx[off : off + n_ring], sy[off : off + n_ring])
            off += n_ring
            if len(poly) < 3:
                continue
            t = (s + 1) / float(n_steps)
            fill_a = int(18 + 28 * t)
            edge_a = int(70 + 50 * t)
            pygame.draw.polygon(overlay, (*rgb, fill_a), poly, 0)
            pygame.draw.polygon(overlay, (*rgb, edge_a), poly, 1)
        tip_sc = _pts(sx[off : off + 1], sy[off : off + 1])
        off += 1
        spoke_sc = _pts(sx[off : off + 4], sy[off : off + 4])
        off += 4
        axis_sc = _pts(sx[off : off + 1], sy[off : off + 1])
        if tip_sc:
            apex = tip_sc[0]
            for sp in spoke_sc:
                pygame.draw.line(overlay, (*rgb, 110), apex, sp, 1)
            if axis_sc:
                pygame.draw.line(overlay, (*rgb, 160), apex, axis_sc[0], 1)

        # Aim points: barrel tip, lamp target, barrel-axis hit at lamp range.
        markers: list[tuple[np.ndarray, str, tuple[int, int, int]]] = [
            (tip, "tip", (255, 220, 160)),
        ]
        lamp_ok = bool(
            getattr(self, "light_enabled", True)
            and getattr(self, "lamp_on", False)
        )
        if lamp_ok:
            self._update_lamp_xyz()
            lamp = self._lamp_xyz.astype(np.float64)
            markers.append((lamp, "tgt", (255, 230, 140)))
            to_lamp = lamp - tip
            dist = float(np.dot(to_lamp, fwd))
            if dist > 1e-6:
                hit = tip + fwd * dist
                markers.append((hit, "hit", rgb))
                # Dotted tip → lamp.
                n_dot = 18
                dots = np.linspace(0.0, 1.0, n_dot)
                chain = (tip[None, :] + dots[:, None] * to_lamp[None, :]).astype(
                    np.float32
                )
                dx, dy = project_3d_batch(
                    chain,
                    self.view_angle_x,
                    self.view_angle_y,
                    cx,
                    cy,
                    self.scale_3d,
                )
                dpts = _pts(dx, dy)
                for i in range(0, len(dpts) - 1, 2):
                    pygame.draw.line(
                        overlay, (*rgb, 140), dpts[i], dpts[i + 1], 1
                    )

        mxyz = np.stack([m[0] for m in markers], axis=0).astype(np.float32)
        mx, my = project_3d_batch(
            mxyz,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        for i, (_p, tag, col) in enumerate(markers):
            if i >= len(mx):
                break
            x, y = float(mx[i]), float(my[i])
            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and 0 <= x < self.view_w
                and 0 <= y < view_lim_y
            ):
                continue
            ax, ay = int(round(x)), int(round(y))
            pygame.draw.circle(overlay, (*col, 220), (ax, ay), 5)
            pygame.draw.circle(overlay, (255, 255, 240, 200), (ax, ay), 5, 1)
            # Crosshair ticks
            pygame.draw.line(overlay, (*col, 200), (ax - 8, ay), (ax - 4, ay), 1)
            pygame.draw.line(overlay, (*col, 200), (ax + 4, ay), (ax + 8, ay), 1)
            pygame.draw.line(overlay, (*col, 200), (ax, ay - 8), (ax, ay - 4), 1)
            pygame.draw.line(overlay, (*col, 200), (ax, ay + 4), (ax, ay + 8), 1)
            lab = self.hint_font.render(tag, True, col)
            overlay.blit(lab, (ax + 7, ay - lab.get_height() // 2))

    def _draw_turret_group_box(
        self,
        pose: dict,
        connected: list[bool],
        cx: float,
        cy: float,
        view_lim_y: int,
    ) -> None:
        assert self._view_surf is not None
        pts3: list[np.ndarray] = [
            np.asarray(pose["mount"], dtype=np.float64),
            np.asarray(pose["barrel"], dtype=np.float64),
        ]
        for site in pose["sites"]:
            pts3.append(np.asarray(site, dtype=np.float64))
        for nid in self._turret_flat_ids():
            neu = self.fly.neurons.get(int(nid))
            if neu is not None:
                pts3.append(np.asarray(neu.soma, dtype=np.float64))
        if len(pts3) < 2:
            return
        arr = np.stack(pts3, axis=0).astype(np.float32)
        sx, sy = project_3d_batch(
            arr,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        xs = [float(x) for x in sx if np.isfinite(x)]
        ys = [float(y) for y in sy if np.isfinite(y)]
        if len(xs) < 2:
            return
        pad = 16
        left = max(0, int(min(xs) - pad))
        right = min(self.view_w - 1, int(max(xs) + pad))
        top = max(0, int(min(ys) - pad))
        bot = min(view_lim_y - 1, int(max(ys) + pad))
        if right - left < 8 or bot - top < 8:
            return
        n_ok = sum(1 for c in connected if c)
        n_tot = max(1, len(connected))
        if n_ok == n_tot:
            fill, edge, tick_c, title_c = (
                (55, 45, 40, 30),
                (200, 150, 110, 60),
                (230, 180, 130, 95),
                (220, 180, 140),
            )
        elif n_ok > 0:
            fill, edge, tick_c, title_c = (
                (70, 55, 30, 32),
                (200, 170, 90, 70),
                (220, 190, 120, 95),
                (210, 190, 140),
            )
        else:
            fill, edge, tick_c, title_c = (
                (70, 28, 28, 36),
                (210, 90, 90, 90),
                (230, 120, 120, 110),
                (220, 150, 150),
            )
        bw, bh = right - left, bot - top
        overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
        rect = pygame.Rect(0, 0, bw, bh)
        pygame.draw.rect(overlay, fill, rect, border_radius=8)
        pygame.draw.rect(overlay, edge, rect, width=1, border_radius=8)
        tick = 10
        for (x0, y0, dx, dy) in (
            (0, 0, 1, 1),
            (bw - 1, 0, -1, 1),
            (0, bh - 1, 1, -1),
            (bw - 1, bh - 1, -1, -1),
        ):
            pygame.draw.line(overlay, tick_c, (x0, y0), (x0 + dx * tick, y0), 1)
            pygame.draw.line(overlay, tick_c, (x0, y0), (x0, y0 + dy * tick), 1)
        title = self.hint_font.render("turret", True, title_c)
        title.set_alpha(160)
        self._view_surf.blit(
            title, (left + 6, max(2, top - title.get_height() - 1))
        )
        if n_ok == n_tot:
            st_txt, st_rgb, st_bg = "fused", (120, 220, 150), (16, 36, 24, 200)
        elif n_ok > 0:
            st_txt, st_rgb, st_bg = (
                f"fusing {n_ok}/{n_tot}",
                (200, 200, 120),
                (36, 34, 16, 200),
            )
        else:
            st_txt, st_rgb, st_bg = "not fused", (230, 140, 140), (40, 18, 18, 200)
        st_lab = self.hint_font.render(st_txt, True, st_rgb)
        pad_x, pad_y, dot_gap = 6, 3, 12
        pw = st_lab.get_width() + pad_x * 2 + dot_gap
        ph = st_lab.get_height() + pad_y * 2
        px = int(np.clip(8, 2, max(2, bw - pw - 2)))
        py = int(np.clip(10, 2, max(2, bh - ph - 2)))
        plate = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(plate, st_bg, plate.get_rect(), border_radius=3)
        pygame.draw.rect(plate, (*st_rgb, 200), plate.get_rect(), width=1, border_radius=3)
        pygame.draw.circle(plate, (*st_rgb, 230), (pad_x + 3, ph // 2), 3)
        plate.blit(st_lab, (pad_x + dot_gap, pad_y))
        overlay.blit(plate, (px, py))
        self._view_surf.blit(overlay, (left, top))

    def _tick_lamp(self, dt: float) -> None:
        """Orbit lamp only during adult Run (closed loop); freeze otherwise."""
        if not self.light_enabled or not self.lamp_on:
            self._update_lamp_xyz()
            return
        if not (self.fly.is_adult() and getattr(self, "_activity_running", False)):
            self._update_lamp_xyz()
            return
        self.lamp_angle = (
            self.lamp_angle + float(dt) * float(self.effective_lamp_speed()) * math.tau
        ) % math.tau
        self._update_lamp_xyz()

    def _tick_world_swing(self, dt: float) -> None:
        """Advance Matrix gravity only during adult Run; freeze otherwise."""
        if not getattr(self, "world_swing_on", False):
            return
        if not (self.fly.is_adult() and getattr(self, "_activity_running", False)):
            self._update_lamp_xyz()
            return
        hz = float(self.effective_world_swing_hz())
        if hz <= 1e-9:
            self._update_lamp_xyz()
            return
        self.world_swing_phase = (
            float(self.world_swing_phase) + float(dt) * hz * math.tau
        ) % math.tau
        # World frame moved → invalidate env caches tied to tilt.
        self._ground_blit_key = None
        self._compass_blit_key = None
        self._update_lamp_xyz()

    def _world_swing_angle(self) -> float:
        """Current tilt rad about body +X (0 = anatomical up)."""
        if not getattr(self, "world_swing_on", False):
            return 0.0
        amp = float(self.effective_world_swing_amp())
        if amp <= 1e-9:
            return 0.0
        return amp * math.sin(float(getattr(self, "world_swing_phase", 0.0) or 0.0))

    def _world_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Matrix world basis (up, right, forward). Fly tissue stays anatomical."""
        th = self._world_swing_angle()
        ct, st = math.cos(th), math.sin(th)
        up = np.array([0.0, ct, st], dtype=np.float64)
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        forward = np.array([0.0, -st, ct], dtype=np.float64)  # R_x(θ) @ +Z
        return up, right, forward

    def _world_up(self) -> np.ndarray:
        return self._world_frame()[0]

    def _world_down(self) -> np.ndarray:
        return -self._world_frame()[0]

    def _update_lamp_xyz(self) -> None:
        frame = self._body_orbit_frame()
        if frame is None:
            # Empty fly — orbit a unit sphere at origin.
            c = np.zeros(3, dtype=np.float64)
            r = 3.0
        else:
            c, r0 = frame
            c = np.asarray(c, dtype=np.float64)
            r = float(r0) * float(LAMP_ORBIT_FRAC)
        elev = float(LAMP_ELEV)
        az = float(self.lamp_angle)
        ce, se = math.cos(elev), math.sin(elev)
        up, right, forward = self._world_frame()
        # Azimuth about Matrix world-up: lamp stays in the "sky" band.
        self._lamp_xyz = (
            c
            + r * ce * math.cos(az) * right
            + r * se * up
            + r * ce * math.sin(az) * forward
        ).astype(np.float64)

    def _draw_sky_fill(self, view_lim_y: int) -> None:
        """Dim sky wash + slightly darker ground — fill light for the scene."""
        assert self._view_surf is not None
        if not self.light_enabled or view_lim_y <= 1:
            self._view_surf.fill(BG_VIEW)
            return
        sky_a = float(np.clip(self.sky_fill, 0.0, 1.0))
        h = max(1, int(view_lim_y))
        w = int(self.view_w)
        sky_q = round(sky_a, 3)
        cache = self._sky_grad_cache
        if cache is None or cache[0] != w or cache[1] != h or cache[2] != sky_q:
            top = tuple(
                int(BG_VIEW[i] * (1.0 - sky_a) + SKY_FILL_RGB[i] * sky_a)
                for i in range(3)
            )
            bot = tuple(
                int(
                    BG_VIEW[i] * (1.0 - 0.55 * sky_a)
                    + SKY_GROUND_RGB[i] * (0.55 * sky_a)
                )
                for i in range(3)
            )
            # Coarse column then scale — cheap per-frame when cache misses.
            steps = min(h, 64)
            col_surf = pygame.Surface((1, steps))
            for y in range(steps):
                t = y / max(1, steps - 1)
                col = tuple(int(top[i] * (1.0 - t) + bot[i] * t) for i in range(3))
                col_surf.set_at((0, y), col)
            grad = pygame.transform.smoothscale(col_surf, (w, h))
            self._sky_grad_cache = (w, h, sky_q, grad)
        else:
            grad = cache[3]
        self._view_surf.blit(grad, (0, 0))
        if self.view_h > view_lim_y:
            pygame.draw.rect(
                self._view_surf,
                BG_VIEW,
                (0, view_lim_y, self.view_w, self.view_h - view_lim_y),
            )


    def _ensure_ground_field(self) -> bool:
        """Build / refresh heightfield under the body (world −Y = down)."""
        frame = self._body_orbit_frame()
        if frame is None:
            c = np.zeros(3, dtype=np.float64)
            r = 3.0
        else:
            c, r = frame
            c = np.asarray(c, dtype=np.float64)
            r = float(r)
        relief_q = round(float(self.ground_relief), 3)
        key = (
            round(float(c[0]), 3),
            round(float(c[1]), 3),
            round(float(c[2]), 3),
            round(r, 3),
            relief_q,
            GROUND_GRID,
            "g5",
        )
        if key == self._ground_cache_key and self._ground_y is not None:
            return True
        n = int(GROUND_GRID) + 1
        extent = float(r) * float(GROUND_EXTENT_FRAC)
        # Just under the coordinate rings (compass radius ≈ r; DN = world −Y).
        y_base = float(c[1]) - float(r) * (1.0 + float(GROUND_GAP_FRAC))
        # Relief scale vs body, not vs huge ground span.
        amp = float(np.clip(self.ground_relief, 0.0, 1.0)) * 0.16 * float(r)
        xs = np.linspace(c[0] - extent, c[0] + extent, n, dtype=np.float64)
        zs = np.linspace(c[2] - extent, c[2] + extent, n, dtype=np.float64)
        xx, zz = np.meshgrid(xs, zs, indexing="xy")
        # Normalized coords for procedural hills — low-frequency only (less busy).
        u = (xx - c[0]) / max(extent, 1e-6)
        v = (zz - c[2]) / max(extent, 1e-6)
        h = (
            0.62 * np.sin(1.6 * u + 0.4) * np.cos(1.3 * v - 0.2)
            + 0.28 * np.sin(2.8 * u - 0.9 * v)
            + 0.12 * np.sin(4.2 * v + 0.3 * u)
        )
        # Flatten a soft pad under the fly so it doesn't sink into spikes.
        pad = np.exp(-((u * u + v * v) / 0.22))
        h = h * (1.0 - 0.80 * pad)
        yy = y_base + amp * h
        self._ground_x = xs
        self._ground_z = zs
        self._ground_y = yy.astype(np.float64)
        self._ground_center = c
        self._ground_extent = extent
        self._ground_cache_key = key
        return True

    def _ground_height_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Bilinear sample of the heightfield."""
        xs = self._ground_x
        zs = self._ground_z
        yy = self._ground_y
        assert xs is not None and zs is not None and yy is not None
        # xs is length N along X; yy indexed [iz, ix] with meshgrid indexing xy
        # Actually meshgrid(xs, zs, indexing="xy") → xx shape (len(zs), len(xs))?
        # indexing="xy": xx.shape = (len(zs), len(xs)) if... 
        # np.meshgrid(xs, zs, indexing="xy") → xx, zz both (len(zs), len(xs))?
        # For indexing='xy': xx.shape = (Ny, Nx) when zs has Ny and xs has Nx.
        # We used meshgrid(xs, zs, indexing="xy") so xx.shape = (len(zs), len(xs)).
        # And yy = y_base + amp*h with h from xx,zz → yy[iz, ix].
        nx = len(xs)
        nz = len(zs)
        fx = (x - xs[0]) / max(xs[-1] - xs[0], 1e-9) * (nx - 1)
        fz = (z - zs[0]) / max(zs[-1] - zs[0], 1e-9) * (nz - 1)
        fx = np.clip(fx, 0, nx - 1.001)
        fz = np.clip(fz, 0, nz - 1.001)
        ix0 = np.floor(fx).astype(np.int32)
        iz0 = np.floor(fz).astype(np.int32)
        ix1 = ix0 + 1
        iz1 = iz0 + 1
        tx = fx - ix0
        tz = fz - iz0
        y00 = yy[iz0, ix0]
        y10 = yy[iz0, ix1]
        y01 = yy[iz1, ix0]
        y11 = yy[iz1, ix1]
        return (
            y00 * (1 - tx) * (1 - tz)
            + y10 * tx * (1 - tz)
            + y01 * (1 - tx) * tz
            + y11 * tx * tz
        )

    def _ground_lit_shadow(
        self, pts: np.ndarray, normals: np.ndarray
    ) -> np.ndarray:
        """Per-point brightness 0..1 with lamp diffuse + heightfield shadow + body umbra."""
        n = len(pts)
        # Readable stage: muted, but not lost in the void.
        sky = 0.28 + 0.22 * float(np.clip(self.sky_fill, 0.0, 1.0))
        lit = np.full(n, sky, dtype=np.float64)
        if not self.lamp_on:
            return np.clip(lit, 0.22, 0.55)
        lamp = self._lamp_xyz.astype(np.float64)
        to_l = lamp[None, :] - pts
        dist = np.linalg.norm(to_l, axis=1)
        dist = np.maximum(dist, 1e-6)
        L = to_l / dist[:, None]
        ndotl = np.einsum("ij,ij->i", normals, L)
        ndotl = np.clip(0.25 + 0.75 * ndotl, 0.0, 1.0)
        # Falloff vs body scale (not the huge ground span).
        frame = self._body_orbit_frame()
        body_r = float(frame[1]) if frame is not None else max(float(self._ground_extent) * 0.3, 1.0)
        atten = (body_r * body_r) / (dist * dist + (0.45 * body_r) ** 2)
        inten = float(np.clip(self.lamp_intensity, 0.0, 2.0))
        direct = inten * 0.70 * ndotl * atten
        ext = max(float(self._ground_extent), 1e-3)

        # Heightfield self-shadow: march toward lamp in XZ (skip when Matrix-tilted).
        steps = 7
        shadowed = np.zeros(n, dtype=bool)
        if abs(float(self._world_swing_angle())) < 1e-4:
            for s in range(1, steps + 1):
                t = s / float(steps + 1)
                # Stay below lamp; sample terrain along segment.
                qx = pts[:, 0] + t * (lamp[0] - pts[:, 0])
                qz = pts[:, 2] + t * (lamp[2] - pts[:, 2])
                qy_ray = pts[:, 1] + t * (lamp[1] - pts[:, 1])
                qy_ter = self._ground_height_at(qx, qz)
                shadowed |= qy_ter > qy_ray + 0.01 * ext

        # Soft body umbra: occlude if lamp→point ray skims body sphere.
        if frame is not None:
            c, br = frame
            c = np.asarray(c, dtype=np.float64)
            rad = float(br) * 0.48
            w = pts - c[None, :]
            d = lamp[None, :] - pts
            dd = np.einsum("ij,ij->i", d, d)
            dd = np.maximum(dd, 1e-9)
            t_c = -np.einsum("ij,ij->i", w, d) / dd
            t_c = np.clip(t_c, 0.05, 0.95)
            closest = pts + t_c[:, None] * d
            body_hit = np.linalg.norm(closest - c[None, :], axis=1) < rad
            up = self._world_up()
            below = np.einsum("ij,j->i", pts - c[None, :], up) < 0.0
            shadowed |= body_hit & below

        direct = np.where(shadowed, direct * float(GROUND_SHADOW_MUL), direct)
        return np.clip(lit + direct, 0.18, 0.92)

    def _draw_ground(self, cx: float, cy: float, view_lim_y: int) -> None:
        """Relief ground under the fly with lamp shadows."""
        assert self._view_surf is not None
        if not self.light_enabled or not self.ground_on:
            return
        if not self._ensure_ground_field():
            return
        self._update_lamp_xyz()
        xs = self._ground_x
        zs = self._ground_z
        yy = self._ground_y
        assert xs is not None and zs is not None and yy is not None
        nx = len(xs) - 1
        nz = len(zs) - 1
        if nx < 1 or nz < 1:
            return

        lamp = self._lamp_xyz.astype(np.float64)
        tilt_q = round(float(self._world_swing_angle()), 4)
        blit_key = (
            round(float(self.view_angle_x), 4),
            round(float(self.view_angle_y), 4),
            round(float(self.scale_3d), 3),
            round(float(self.pan_x), 1),
            round(float(self.pan_y), 1),
            round(float(lamp[0]), 2),
            round(float(lamp[1]), 2),
            round(float(lamp[2]), 2),
            round(float(self.lamp_intensity), 2),
            bool(self.lamp_on),
            int(self.view_w),
            int(view_lim_y),
            id(yy),
            tilt_q,
        )
        if (
            blit_key == self._ground_blit_key
            and self._ground_blit_surf is not None
            and self._ground_blit_surf.get_size() == (self.view_w, view_lim_y)
        ):
            self._view_surf.blit(self._ground_blit_surf, (0, 0))
            return

        cam_key = blit_key[:5] + (int(self.view_w), int(view_lim_y), id(yy), tilt_q)
        cache = self._ground_draw_cache
        if cam_key != self._ground_draw_cam_key or cache is None:
            xx, zz = np.meshgrid(xs, zs, indexing="xy")
            pts = np.stack([xx, yy, zz], axis=-1)  # (nz+1, nx+1, 3)
            dy_dx = np.zeros_like(yy)
            dy_dz = np.zeros_like(yy)
            dx = float(xs[1] - xs[0]) if nx > 0 else 1.0
            dz = float(zs[1] - zs[0]) if nz > 0 else 1.0
            dy_dx[:, 1:-1] = (yy[:, 2:] - yy[:, :-2]) / (2 * dx)
            dy_dx[:, 0] = (yy[:, 1] - yy[:, 0]) / dx
            dy_dx[:, -1] = (yy[:, -1] - yy[:, -2]) / dx
            dy_dz[1:-1, :] = (yy[2:, :] - yy[:-2, :]) / (2 * dz)
            dy_dz[0, :] = (yy[1, :] - yy[0, :]) / dz
            dy_dz[-1, :] = (yy[-1, :] - yy[-2, :]) / dz
            normals = np.stack([-dy_dx, np.ones_like(yy), -dy_dz], axis=-1)
            nn = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = normals / np.maximum(nn, 1e-9)

            # Matrix tilt: rotate ground about body center (fly fixed).
            th = float(self._world_swing_angle())
            if abs(th) > 1e-6:
                body = self._body_orbit_frame()
                pivot = (
                    np.asarray(body[0], dtype=np.float64)
                    if body is not None
                    else (
                        np.asarray(self._ground_center, dtype=np.float64)
                        if self._ground_center is not None
                        else np.zeros(3, dtype=np.float64)
                    )
                )
                ct, st = math.cos(th), math.sin(th)
                flat = pts.reshape(-1, 3)
                rel = flat - pivot[None, :]
                ry = rel[:, 1] * ct - rel[:, 2] * st
                rz = rel[:, 1] * st + rel[:, 2] * ct
                flat = np.stack(
                    [rel[:, 0] + pivot[0], ry + pivot[1], rz + pivot[2]],
                    axis=1,
                )
                pts = flat.reshape(yy.shape + (3,))
                nflat = normals.reshape(-1, 3)
                ny = nflat[:, 1] * ct - nflat[:, 2] * st
                nz = nflat[:, 1] * st + nflat[:, 2] * ct
                normals = np.stack([nflat[:, 0], ny, nz], axis=1).reshape(
                    normals.shape
                )

            flat_pts = pts.reshape(-1, 3)
            sx, sy = project_3d_batch(
                flat_pts.astype(np.float32),
                self.view_angle_x,
                self.view_angle_y,
                cx,
                cy,
                self.scale_3d,
            )
            sx = sx.reshape(yy.shape)
            sy = sy.reshape(yy.shape)
            x = flat_pts[:, 0]
            y = flat_pts[:, 1]
            z = flat_pts[:, 2]
            cos_y = math.cos(self.view_angle_y)
            sin_y = math.sin(self.view_angle_y)
            z_rot = x * sin_y + z * cos_y
            cos_x = math.cos(self.view_angle_x)
            sin_x = math.sin(self.view_angle_x)
            y_rot = (y * cos_x - z_rot * sin_x).reshape(yy.shape)

            gc = self._ground_center
            if gc is None:
                gc = np.zeros(3, dtype=np.float64)
            rad = float(self._ground_extent)
            rad2 = rad * rad
            rim0 = 0.88 * rad

            # Vectorized quad centers / depths / disk mask.
            mx = 0.25 * (
                xx[:-1, :-1] + xx[:-1, 1:] + xx[1:, :-1] + xx[1:, 1:]
            )
            mz = 0.25 * (
                zz[:-1, :-1] + zz[:-1, 1:] + zz[1:, :-1] + zz[1:, 1:]
            )
            d2 = (mx - gc[0]) ** 2 + (mz - gc[2]) ** 2
            inside = d2 <= rad2
            corners_out = (
                ((xx[:-1, :-1] - gc[0]) ** 2 + (zz[:-1, :-1] - gc[2]) ** 2 > rad2).astype(
                    np.int8
                )
                + ((xx[:-1, 1:] - gc[0]) ** 2 + (zz[:-1, 1:] - gc[2]) ** 2 > rad2).astype(
                    np.int8
                )
                + ((xx[1:, 1:] - gc[0]) ** 2 + (zz[1:, 1:] - gc[2]) ** 2 > rad2).astype(
                    np.int8
                )
                + ((xx[1:, :-1] - gc[0]) ** 2 + (zz[1:, :-1] - gc[2]) ** 2 > rad2).astype(
                    np.int8
                )
            )
            keep = inside & (corners_out < 3)
            depth = 0.25 * (
                y_rot[:-1, :-1]
                + y_rot[:-1, 1:]
                + y_rot[1:, :-1]
                + y_rot[1:, 1:]
            )
            # Corner screen coords for kept quads.
            iz, ix = np.nonzero(keep)
            if len(iz) == 0:
                self._ground_draw_cam_key = cam_key
                self._ground_draw_cache = {"empty": True}
                return
            polys = np.stack(
                [
                    np.stack([sx[iz, ix], sy[iz, ix]], axis=1),
                    np.stack([sx[iz, ix + 1], sy[iz, ix + 1]], axis=1),
                    np.stack([sx[iz + 1, ix + 1], sy[iz + 1, ix + 1]], axis=1),
                    np.stack([sx[iz + 1, ix], sy[iz + 1, ix]], axis=1),
                ],
                axis=1,
            )  # (Q, 4, 2)
            # Offscreen cull
            xs_p = polys[:, :, 0]
            ys_p = polys[:, :, 1]
            on = ~(
                (xs_p.max(axis=1) < 0)
                | (xs_p.min(axis=1) >= self.view_w)
                | (ys_p.max(axis=1) < 0)
                | (ys_p.min(axis=1) >= view_lim_y)
            )
            iz, ix = iz[on], ix[on]
            polys = polys[on]
            depth = depth[keep][on]
            d2_q = d2[keep][on]
            order = np.argsort(depth)
            cache = {
                "empty": False,
                "iz": iz[order],
                "ix": ix[order],
                "polys": np.rint(polys[order]).astype(np.int32),
                "d2": d2_q[order],
                "normals": normals,
                "pts": pts,
                "yy": yy,
                "gc": gc,
                "rad": rad,
                "rim0": rim0,
            }
            self._ground_draw_cam_key = cam_key
            self._ground_draw_cache = cache

        assert cache is not None
        if cache.get("empty"):
            return

        iz = cache["iz"]
        ix = cache["ix"]
        polys = cache["polys"]
        d2_q = cache["d2"]
        normals = cache["normals"]
        pts = cache["pts"]
        yy = cache["yy"]
        gc = cache["gc"]
        rad = float(cache["rad"])
        rim0 = float(cache["rim0"])

        flat_pts = pts.reshape(-1, 3)
        flat_n = normals.reshape(-1, 3)
        bright = self._ground_lit_shadow(flat_pts, flat_n).reshape(yy.shape)
        bq = 0.25 * (
            bright[iz, ix]
            + bright[iz, ix + 1]
            + bright[iz + 1, ix]
            + bright[iz + 1, ix + 1]
        )
        dist_xz = np.sqrt(d2_q)
        fade = np.ones_like(bq)
        outer = dist_xz > rim0
        if np.any(outer):
            fade[outer] = 0.35 + 0.65 * np.clip(
                1.0 - (dist_xz[outer] - rim0) / max(rad - rim0, 1e-6), 0.0, 1.0
            )
        bq = bq * fade
        base = np.asarray(GROUND_RGB, dtype=np.float64)
        cols = np.clip(base[None, :] * bq[:, None], 0, 255).astype(np.uint8)

        layer = pygame.Surface((self.view_w, view_lim_y), pygame.SRCALPHA)
        layer.fill((0, 0, 0, 0))
        # Draw into layer then blit — lets us cache the finished ground.
        target = layer
        for i in range(len(polys)):
            poly = [
                (int(polys[i, 0, 0]), int(polys[i, 0, 1])),
                (int(polys[i, 1, 0]), int(polys[i, 1, 1])),
                (int(polys[i, 2, 0]), int(polys[i, 2, 1])),
                (int(polys[i, 3, 0]), int(polys[i, 3, 1])),
            ]
            col = (int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2]))
            pygame.draw.polygon(target, col, poly)
        self._view_surf.blit(layer, (0, 0))
        self._ground_blit_key = blit_key
        self._ground_blit_surf = layer

    def _draw_point_lamp(self, cx: float, cy: float, view_lim_y: int) -> None:
        """Draw the orbiting point lamp (bulb + soft glow)."""
        assert self._view_surf is not None
        if not self.light_enabled or not self.lamp_on:
            return
        self._update_lamp_xyz()
        xyz = self._lamp_xyz.reshape(1, 3).astype(np.float32)
        sx, sy = project_3d_batch(
            xyz,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        x, y = int(round(float(sx[0]))), int(round(float(sy[0])))
        if not (0 <= x < self.view_w and 0 <= y < view_lim_y):
            return
        inten = float(np.clip(self.lamp_intensity, 0.0, 2.0))
        glow_r = max(10, int(18 + 14 * inten))
        core_r = max(3, int(4 + 3 * inten))
        glow = pygame.Surface((glow_r * 2 + 4, glow_r * 2 + 4), pygame.SRCALPHA)
        gcx, gcy = glow_r + 2, glow_r + 2
        # Soft radial glow rings
        for i, (rad, a) in enumerate(
            (
                (glow_r, int(28 * inten)),
                (int(glow_r * 0.65), int(55 * inten)),
                (int(glow_r * 0.35), int(90 * inten)),
            )
        ):
            if rad < 1 or a < 1:
                continue
            col = (*LAMP_GLOW_RGB, min(255, a))
            pygame.draw.circle(glow, col, (gcx, gcy), rad)
        self._view_surf.blit(glow, (x - gcx, y - gcy))
        # Core bulb
        pygame.draw.circle(self._view_surf, LAMP_RGB, (x, y), core_r)
        pygame.draw.circle(
            self._view_surf,
            (255, 255, 230),
            (x, y),
            max(1, core_r // 2),
        )
        # Tiny label
        tag = self.hint_font.render("lamp", True, (220, 200, 140))
        self._view_surf.blit(tag, (x + core_r + 4, y - tag.get_height() // 2))

    def _ensure_gl_tissue_geom(self) -> bool:
        """Upload proc/soma/grown point clouds to GPU (thinned for density caps)."""
        gl = self._gl_tissue
        if gl is None or not gl.ok:
            return False
        grown_xyz = getattr(self, "grown_xyz", None)
        soma_f = set(self.focus_soma_indices)
        graph_f = set(self.focus_graph_indices)
        axon_f = set(self.focus_axon_indices)
        dend_f = set(self.focus_dendrite_indices)
        neur_f = axon_f | dend_f
        class_focus = self._class_focus_union()
        key = (
            id(self.points_xyz),
            int(len(self.points_xyz)),
            id(self.proc_xyz),
            int(len(self.proc_xyz)),
            id(grown_xyz) if grown_xyz is not None else 0,
            int(len(grown_xyz) if grown_xyz is not None else 0),
            bool(self.show_layer_neurons),
            bool(self.show_layer_graph),
            bool(self.show_layer_axons),
            bool(self.show_layer_dendrites),
            str(getattr(self, "draw_thin_mode", "auto")),
            int(getattr(self, "draw_soma_budget", DRAW_SOMA_MAX)),
            round(float(getattr(self, "view_detail", 1.0)), 2),
            tuple(
                sorted(
                    (k, round(float(v), 3))
                    for k, v in (self.draw_density or {}).items()
                )
            ),
            bool(getattr(self, "scope_on", False)),
            int(id(getattr(self, "_scope_mask", None))),
            frozenset(int(i) for i in soma_f),
            frozenset(int(i) for i in graph_f),
            frozenset(int(i) for i in axon_f),
            frozenset(int(i) for i in dend_f),
            frozenset(int(i) for i in self.focus_basin_indices),
        )
        if key == self._gl_geom_key:
            return True
        # Process ghost (atlas tissue) — Super-class G only. Empty G = hide
        # (same contract as N/A/D/T/B: empty column turns the layer off).
        if self.show_layer_graph and len(self.proc_xyz):
            pi = self._apply_draw_thin(
                self.proc_cls,
                self._draw_cap(min(DRAW_PROC_MAX * 4, 80000)),
                boost_sparse_soma=True,
            )
            proc_rgb = self._focus_modulate_rgb(
                self.proc_rgb[pi], self.proc_cls[pi], class_focus
            )
            gl.set_layer(
                "proc",
                self.proc_xyz[pi],
                proc_rgb,
                point_size=2.5,
                brightness=float(BODY_PROC_BRIGHT),
            )
        else:
            gl.set_layer("proc", np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
        if self.show_layer_neurons and len(self.points_xyz):
            si = self._apply_draw_thin(self.points_cls, min(DRAW_SOMA_MAX * 4, 60000))
            # _apply_draw_thin may return slice(None) when under budget — never
            # np.asarray(slice), that TypeErrors and leaves a black _view_surf.
            if isinstance(si, slice):
                idx = np.arange(len(self.points_xyz), dtype=np.int32)
            else:
                idx = np.asarray(si, dtype=np.int32)
            mask = getattr(self, "_scope_mask", None)
            if (
                getattr(self, "scope_on", False)
                and mask is not None
                and len(mask) == len(self.points_xyz)
            ):
                keep = mask[idx]
                idx = idx[keep] if np.any(keep) else np.zeros((0,), dtype=np.int32)
            self._gl_soma_idx = idx
            soma_rgb = self._focus_modulate_rgb(
                self.point_rgb[self._gl_soma_idx],
                self.points_cls[self._gl_soma_idx],
                soma_f,
                other_layer_focused=bool(graph_f or axon_f or dend_f),
            )
            gl.set_layer(
                "soma",
                self.points_xyz[self._gl_soma_idx],
                soma_rgb,
                point_size=3.0,
                brightness=1.05,
            )
        else:
            self._gl_soma_idx = np.zeros((0,), dtype=np.int32)
            gl.set_layer("soma", np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
        if (
            (self.show_layer_axons or self.show_layer_dendrites)
            and bool(neur_f)
            and grown_xyz is not None
            and len(grown_xyz)
        ):
            gi = _density_take(len(grown_xyz), min(DRAW_GROWN_PTS_MAX * 2, 40000))
            grown_rgb = self._focus_modulate_rgb(
                self.grown_rgb[gi],
                self.grown_cls[gi],
                neur_f,
                other_layer_focused=bool(soma_f or graph_f),
            )
            gl.set_layer(
                "grown",
                grown_xyz[gi],
                grown_rgb,
                point_size=2.0,
                brightness=0.7,
            )
        else:
            gl.set_layer(
                "grown", np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
            )
        self._gl_geom_key = key
        return True

    def _try_draw_tissue_gl(
        self,
        cx: float,
        cy: float,
        view_lim_y: int,
        act: np.ndarray | None,
    ) -> bool:
        """Render tissue via OpenGL into ``_view_surf``. Returns False → CPU path."""
        if not self.use_gl_tissue or self._gl_tissue is None or not self._gl_tissue.ok:
            return False
        if self._view_surf is None:
            return False
        # Match CPU order: sky + ground under points (never paint ground over tissue).
        try:
            self._draw_sky_fill(view_lim_y)
            if getattr(self, "ground_on", False) and int(view_lim_y) > 1 and int(self.view_w) > 1:
                self._draw_ground(cx, cy, view_lim_y)
        except Exception:
            pass
        try:
            if not self._ensure_gl_tissue_geom():
                return False
            layers = getattr(self._gl_tissue, "_layers", {}) or {}
            n_up = sum(int(L.get("n", 0) or 0) for L in layers.values())
            tissue_wanted = bool(
                self.show_layer_neurons
                or self.show_layer_graph
                or self.show_layer_axons
                or self.show_layer_dendrites
            )
            if n_up <= 0:
                # Empty Super-class → intentional blank tissue (do not fall back to
                # CPU, which used to paint wave tracts over an "empty" filter).
                if not tissue_wanted:
                    return True
                if len(self.points_xyz) > 0 or len(self.proc_xyz) > 0:
                    return False
                return True
            soma_idx = getattr(self, "_gl_soma_idx", None)
            if (
                act is not None
                and soma_idx is not None
                and len(soma_idx)
                and len(act) == self.n_shown
            ):
                self._gl_tissue.update_activity("soma", act[soma_idx])
            # Clear to 0 so we can composite lit fragments over sky/ground.
            img = self._gl_tissue.render(
                view_w=int(self.view_w),
                view_h=int(self.view_h),
                view_lim_y=int(view_lim_y),
                view_angle_x=float(self.view_angle_x),
                view_angle_y=float(self.view_angle_y),
                pan_x=float(self.pan_x),
                pan_y=float(self.pan_y),
                scale_3d=float(self.scale_3d),
                bg_rgb=(0, 0, 0),
            )
            if img is None or int(img.max()) == 0:
                return False
            if self._view_surf.get_bitsize() < 24:
                self._view_surf = self._view_surf.convert()
            # pygame surfarray wants contiguous (w, h, 3).
            src = np.ascontiguousarray(np.transpose(img, (1, 0, 2)))
            lit = np.any(src != 0, axis=2)
            if not np.any(lit):
                return False
            px = pygame.surfarray.pixels3d(self._view_surf)
            px[lit] = src[lit]
            del px
        except Exception as exc:
            # Upload/render/blit failure → CPU path (avoid stuck black frame).
            print(f"[gl_tissue] draw failed → CPU: {type(exc).__name__}: {exc}", flush=True)
            self._gl_geom_key = None
            return False
        return True

    def _draw_particles(self) -> None:
        assert self._view_surf is not None
        # Focus sets are source of truth — keep show_layer_* mirrors honest every frame.
        self._sync_layers_from_focus()
        if getattr(self, "show_port_schema", False):
            self._draw_ai_brain_schema_view()
            return
        if getattr(self, "show_brain_schema", False):
            self._draw_brain_schema_view()
            return
        if getattr(self, "show_grow_schema", False):
            self._draw_grow_schema_view()
            return
        if getattr(self, "show_research_schema", False):
            self._draw_research_schema_view()
            return
        import time as _time

        cx = self.view_w * 0.5 + self.pan_x
        cy = (self.view_h - NAV_PANEL_H) * 0.5 + self.pan_y
        view_lim_y = self.view_h - NAV_PANEL_H

        soma_f = self.focus_soma_indices
        graph_f = self.focus_graph_indices
        axon_f = self.focus_axon_indices
        dend_f = self.focus_dendrite_indices
        act = getattr(self, "soma_activity", None)

        # GPU tissue path — skip CPU stamp/cache when moderngl is up.
        t_tissue0 = _time.perf_counter()
        if self._try_draw_tissue_gl(cx, cy, view_lim_y, act):
            self._tissue_cache_hit = False
            self._perf_ms["draw.tissue"] = (_time.perf_counter() - t_tissue0) * 1000.0
            t_ov0 = _time.perf_counter()
            # Overlays only (funnels / flash / devices / nav).
            hot_ids = getattr(self, "_learn_hot_ids", None) or set()
            if (
                hot_ids
                and float(getattr(self, "_learn_flash_amp", 0.0) or 0.0) > 0.05
                and len(self.points_xyz)
                and len(self.fly.soma_ids) == len(self.points_xyz)
            ):
                # Lightweight CPU blooms for learn chain (already tiny set).
                pass  # learn flash draws topo below
            ov = float(self._view_detail_knobs()["overlay"])
            if self.show_layer_graph and ov >= 0.25:
                self._draw_recipe_graph_overlay(
                    cx,
                    cy,
                    view_lim_y,
                    graph_f,
                    max_edges=self._draw_cap(DRAW_GRAPH_EDGES_MAX),
                )
            if ov >= 0.40:
                self._draw_tract_sample(cx, cy, view_lim_y)
            # Rising trails — only with Super-class A/D focus (empty = hide).
            if axon_f or dend_f:
                self._draw_growth_overlay(
                    cx, cy, view_lim_y, axon_f=axon_f, dend_f=dend_f
                )
            if self.show_layer_basins and ov >= 0.30:
                self._draw_basins_layer(cx, cy, view_lim_y, self.focus_basin_indices)
            if ov >= 0.20:
                self._draw_device_funnels(cx, cy, view_lim_y)
            if ov >= 0.35:
                self._draw_corridor_topology(cx, cy, view_lim_y)
                self._draw_channel_bed(cx, cy, view_lim_y)
            if ov >= 0.20:
                self._draw_orbit_compass(cx, cy, view_lim_y)
                self._draw_gravity_pendulum(cx, cy, view_lim_y)
            self._draw_point_lamp(cx, cy, view_lim_y)
            draw_devices(
                self,
                list(
                    getattr(self, "device_list", None)
                    or getattr(getattr(self, "blueprint", None), "device_list", [])
                    or []
                ),
                cx,
                cy,
                view_lim_y,
            )
            if ov >= 0.30:
                self._draw_learn_flash(cx, cy, view_lim_y)
            self._perf_ms["draw.overlays"] = (_time.perf_counter() - t_ov0) * 1000.0
            return

        tissue_key = (
            round(float(self.view_angle_x), 4),
            round(float(self.view_angle_y), 4),
            round(float(self.scale_3d), 3),
            round(float(self.pan_x), 1),
            round(float(self.pan_y), 1),
            int(self.view_w),
            int(view_lim_y),
            bool(self.show_layer_neurons),
            bool(self.show_layer_graph),
            bool(self.show_layer_axons),
            bool(self.show_layer_dendrites),
            str(getattr(self, "draw_thin_mode", "auto")),
            int(getattr(self, "draw_soma_budget", DRAW_SOMA_MAX)),
            round(float(getattr(self, "view_detail", 1.0)), 2),
            tuple(sorted((k, round(float(v), 3)) for k, v in (self.draw_density or {}).items())),
            tuple(sorted(int(i) for i in soma_f)),
            tuple(sorted(int(i) for i in graph_f)),
            tuple(sorted(int(i) for i in axon_f)),
            tuple(sorted(int(i) for i in dend_f)),
            bool(getattr(self, "scope_on", False)),
            int(id(getattr(self, "_scope_mask", None))),
            bool(getattr(self, "ground_on", False)),
            round(float(getattr(self, "sky_fill", 0.0)), 3),
            bool(getattr(self, "lamp_on", False)),
            round(float(getattr(self, "lamp_intensity", 1.0)), 2),
        )
        tissue_hit = (
            tissue_key == self._tissue_blit_key
            and self._tissue_blit_surf is not None
            and self._tissue_blit_surf.get_size() == (self.view_w, self.view_h)
        )
        self._tissue_cache_hit = bool(tissue_hit)
        if tissue_hit:
            self._view_surf.blit(self._tissue_blit_surf, (0, 0))
        else:
            self._draw_sky_fill(view_lim_y)
            self._draw_ground(cx, cy, view_lim_y)

        def _stamp_points(
            xyz: np.ndarray,
            cls: np.ndarray,
            base_rgb: np.ndarray,
            *,
            brightness: float,
            fat: bool,
            focus_set: set[int],
            other_layer_focused: bool,
            activity: np.ndarray | None = None,
            size: int = 1,
            blend: str = "max",
        ) -> None:
            if len(xyz) == 0:
                return
            sx, sy = project_3d_batch(
                xyz,
                self.view_angle_x,
                self.view_angle_y,
                cx,
                cy,
                self.scale_3d,
            )
            xi = np.rint(sx).astype(np.int32)
            yi = np.rint(sy).astype(np.int32)
            m = (xi >= 0) & (xi < self.view_w) & (yi >= 0) & (yi < view_lim_y)
            if not np.any(m):
                return
            rgb = np.clip(base_rgb.astype(np.float32) * brightness, 0, 255)
            if activity is not None and len(activity) == len(cls):
                # Only modulate when something is actually firing — resting CNS
                # must keep full class colors (not a flat 45% grey-out).
                if float(np.max(activity)) > 0.05:
                    factor = 0.72 + 0.28 * np.clip(activity, 0.0, 1.0)
                    rgb = rgb * factor[:, None]
                    hot = activity > 0.05
                    if np.any(hot):
                        rgb[hot] = np.clip(
                            rgb[hot] + activity[hot, None] * 90.0, 0, 255
                        )
            rgb = rgb.astype(np.uint8)
            any_focus = bool(focus_set) or other_layer_focused
            if any_focus:
                if focus_set:
                    is_f = np.isin(cls, list(focus_set))
                else:
                    # Other layer filtered, this layer fully dimmed
                    is_f = np.zeros(len(cls), dtype=bool)
                dim = ~is_f
                rgb[dim] = (rgb[dim].astype(np.float32) * FOCUS_DIM).astype(np.uint8)
                rgb[is_f] = np.clip(
                    rgb[is_f].astype(np.int16) + FOCUS_BOOST, 0, 255
                ).astype(np.uint8)
            px = pygame.surfarray.pixels3d(self._view_surf)
            # size 1 = 1px, 2 = 2×2, 3 = 3×3 (somata read above neurite clutter).
            if size >= 3 or (fat and size >= 2):
                offsets = (
                    (0, 0),
                    (1, 0),
                    (0, 1),
                    (1, 1),
                    (-1, 0),
                    (0, -1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                )
            elif fat or size >= 2:
                offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
            else:
                offsets = ((0, 0),)
            for dx, dy in offsets:
                xx = xi[m] + dx
                yy = yi[m] + dy
                ok = (xx >= 0) & (xx < self.view_w) & (yy >= 0) & (yy < view_lim_y)
                cur = px[xx[ok], yy[ok]]
                nxt = rgb[m][ok]
                if blend == "add":
                    # Soft add: sparse color rides on top of denser underlay.
                    px[xx[ok], yy[ok]] = np.clip(
                        cur.astype(np.int16) + (nxt.astype(np.int16) * 3 // 4),
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    px[xx[ok], yy[ok]] = np.maximum(cur, nxt)
            del px

        def _take(
            n: int,
            budget: int,
            focus: set[int],
            cls_arr: np.ndarray | None = None,
        ) -> np.ndarray | slice:
            # Density-capped full-CNS sample (body silhouette).
            del focus  # brightness handled via focus_set, not drop
            if cls_arr is not None and len(cls_arr) == n:
                return self._apply_draw_thin(cls_arr, budget)
            return _density_take(n, budget)

        def _take_focus_sparse(
            n: int,
            budget: int,
            focus: set[int],
            cls_arr: np.ndarray,
            xyz: np.ndarray,
        ) -> np.ndarray:
            """Focused-class subsample: equal share per enabled class, still sparse."""
            if not focus or len(cls_arr) != n:
                return np.zeros((0,), dtype=np.int32)
            classes = sorted(focus)
            per = max(1, min(budget, DRAW_FOCUS_SOMA_MAX) // max(1, len(classes)))
            parts: list[np.ndarray] = []
            for c in classes:
                sub = np.flatnonzero(cls_arr == int(c))
                if len(sub) == 0:
                    continue
                if len(sub) > per:
                    step = max(1, (len(sub) + per - 1) // per)
                    sub = sub[::step][:per]
                if len(sub) <= 1:
                    parts.append(sub)
                    continue
                # Screen-thin per class so one dense type cannot erase another.
                sx, sy = project_3d_batch(
                    xyz[sub],
                    self.view_angle_x,
                    self.view_angle_y,
                    cx,
                    cy,
                    self.scale_3d,
                )
                cell = 3
                qx = np.floor(sx / cell).astype(np.int32)
                qy = np.floor(sy / cell).astype(np.int32)
                keys = qx.astype(np.int64) * 1_000_003 + qy.astype(np.int64)
                _, keep = np.unique(keys, return_index=True)
                keep.sort()
                parts.append(sub[keep])
            if not parts:
                return np.zeros((0,), dtype=np.int32)
            return np.concatenate(parts)

        # Atlas process ghost = CNS body outline under Super-class G.
        # Empty G → hide (Graph edges + tissue share the G column).
        if not tissue_hit:
            class_focus = self._class_focus_union()
            if self.show_layer_graph and len(self.proc_xyz):
                pi = self._apply_draw_thin(
                    self.proc_cls, DRAW_PROC_MAX, boost_sparse_soma=True
                )
                _stamp_points(
                    self.proc_xyz[pi],
                    self.proc_cls[pi],
                    self.proc_rgb[pi],
                    brightness=BODY_PROC_BRIGHT,
                    fat=False,
                    focus_set=class_focus,
                    other_layer_focused=False,
                    size=1,
                )
                if class_focus:
                    fi = _take_focus_sparse(
                        len(self.proc_xyz),
                        DRAW_FOCUS_SOMA_MAX,
                        class_focus,
                        self.proc_cls,
                        self.proc_xyz,
                    )
                    if len(fi):
                        _stamp_points(
                            self.proc_xyz[fi],
                            self.proc_cls[fi],
                            self.proc_rgb[fi],
                            brightness=0.95,
                            fat=False,
                            focus_set=class_focus,
                            other_layer_focused=False,
                            size=1,
                            blend="add",
                        )
            # Recipe graph (G) + tracts (T) are overlays — never bake into tissue
            # cache (focus_tract was missing from tissue_key → T-off left ghosts).

            if self.show_layer_neurons:
                si = _take(len(self.points_xyz), DRAW_SOMA_MAX, soma_f, self.points_cls)
                mask = getattr(self, "_scope_mask", None)
                if (
                    getattr(self, "scope_on", False)
                    and mask is not None
                    and len(mask) == len(self.points_xyz)
                ):
                    if isinstance(si, slice):
                        idx = np.arange(len(self.points_xyz), dtype=np.int32)
                    else:
                        idx = np.asarray(si, dtype=np.int32)
                    keep = mask[idx]
                    if np.any(keep):
                        si = idx[keep]
                    else:
                        si = np.zeros((0,), dtype=np.int32)
                axons_on = self.show_layer_axons or self.show_layer_dendrites
                # Same brightness whether one class or the whole column is on.
                if soma_f:
                    soma_bri = 1.15
                else:
                    soma_bri = 1.15 if not axons_on else 1.05
                body_size = 2
                _stamp_points(
                    self.points_xyz[si],
                    self.points_cls[si],
                    self.point_rgb[si],
                    brightness=soma_bri,
                    fat=True,
                    focus_set=soma_f,
                    other_layer_focused=bool(graph_f or axon_f or dend_f),
                    activity=None,
                    size=body_size,
                )
                if soma_f:
                    fi = _take_focus_sparse(
                        len(self.points_xyz),
                        DRAW_FOCUS_SOMA_MAX,
                        soma_f,
                        self.points_cls,
                        self.points_xyz,
                    )
                    if len(fi):
                        n_per = len(fi) // max(1, len(soma_f))
                        focus_size = 2 if n_per <= 2500 else 1
                        _stamp_points(
                            self.points_xyz[fi],
                            self.points_cls[fi],
                            self.point_rgb[fi],
                            brightness=1.2,
                            fat=True,
                            focus_set=soma_f,
                            other_layer_focused=False,
                            activity=None,
                            size=focus_size,
                        )

            grown_xyz = getattr(self, "grown_xyz", None)
            neur_f = axon_f | dend_f
            neur_focus_on = bool(neur_f)
            grown_budget = DRAW_GROWN_FOCUS_MAX if neur_focus_on else DRAW_GROWN_PTS_MAX
            if (
                grown_xyz is not None
                and len(grown_xyz)
                and neur_focus_on
                and (self.show_layer_axons or self.show_layer_dendrites)
            ):
                gi = _take(
                    len(grown_xyz), grown_budget, neur_f, getattr(self, "grown_cls", None)
                )
                _stamp_points(
                    grown_xyz[gi],
                    self.grown_cls[gi],
                    self.grown_rgb[gi],
                    brightness=0.85 if neur_focus_on else 0.75,
                    fat=False,
                    focus_set=neur_f,
                    other_layer_focused=bool(soma_f or graph_f),
                    size=1,
                    blend="add",
                )
                if neur_focus_on:
                    gi = _take_focus_sparse(
                        len(grown_xyz),
                        min(DRAW_GROWN_FOCUS_MAX, 20000),
                        neur_f,
                        self.grown_cls,
                        grown_xyz,
                    )
                    if len(gi):
                        _stamp_points(
                            grown_xyz[gi],
                            self.grown_cls[gi],
                            self.grown_rgb[gi],
                            brightness=0.95,
                            fat=False,
                            focus_set=neur_f,
                            other_layer_focused=False,
                            size=1,
                            blend="add",
                        )

            # Live growth trails are overlays (gated by A/D) — never bake into
            # tissue cache (Rising would keep showing after A is cleared).

            self._tissue_blit_surf = self._view_surf.copy()
            self._tissue_blit_key = tissue_key

        # Live activity glow on top of cached tissue (sparse lit somata only).
        if (
            self.show_layer_neurons
            and act is not None
            and len(act) == self.n_shown
            and len(self.points_xyz)
        ):
            lit = np.flatnonzero(act > 0.08)
            if len(lit) > 1200:
                step = max(1, (len(lit) + 1199) // 1200)
                lit = lit[::step][:1200]
            if len(lit):
                _stamp_points(
                    self.points_xyz[lit],
                    self.points_cls[lit],
                    self.point_rgb[lit],
                    brightness=1.1,
                    fat=True,
                    focus_set=soma_f,
                    other_layer_focused=False,
                    activity=act[lit],
                    size=2,
                    blend="add",
                )

        # Learn chain somata — force-bright only when Super-class N includes them.
        hot_ids = getattr(self, "_learn_hot_ids", None) or set()
        soma_focus = set(self.focus_soma_indices)
        if (
            hot_ids
            and soma_focus
            and self.show_layer_neurons
            and float(getattr(self, "_learn_flash_amp", 0.0) or 0.0) > 0.05
            and len(self.points_xyz)
            and len(self.fly.soma_ids) == len(self.points_xyz)
        ):
            hi = [
                i
                for i in self._soma_indices_for_ids(hot_ids)
                if int(self.points_cls[i]) in soma_focus
            ]
            if hi:
                hi_a = np.asarray(hi[:800], dtype=np.int32)
                pulse = self._pulse()
                _stamp_points(
                    self.points_xyz[hi_a],
                    self.points_cls[hi_a],
                    self.point_rgb[hi_a],
                    brightness=1.25 + 0.35 * pulse,
                    fat=True,
                    focus_set=soma_focus,
                    other_layer_focused=False,
                    activity=np.ones(len(hi_a), dtype=np.float32),
                    size=3,
                    blend="add",
                )

        # Glitter only when few somata are on screen (not by class-count threshold).
        if self.show_layer_neurons and soma_f and self.n_shown > 0:
            glow_idx = np.flatnonzero(np.isin(self.points_cls, list(soma_f)))
            if 0 < len(glow_idx) <= 2500:
                if len(glow_idx) > DRAW_FOCUS_GLOW_MAX:
                    step = max(
                        1,
                        (len(glow_idx) + DRAW_FOCUS_GLOW_MAX - 1)
                        // DRAW_FOCUS_GLOW_MAX,
                    )
                    glow_idx = glow_idx[::step][:DRAW_FOCUS_GLOW_MAX]
                sx, sy = project_3d_batch(
                    self.points_xyz[glow_idx],
                    self.view_angle_x,
                    self.view_angle_y,
                    cx,
                    cy,
                    self.scale_3d,
                )
                xi = np.rint(sx).astype(np.int32)
                yi = np.rint(sy).astype(np.int32)
                m = (xi >= 0) & (xi < self.view_w) & (yi >= 0) & (yi < view_lim_y)
                pulse = self._pulse()
                radius = max(1, int(1 + 1.6 * pulse))
                glow_alpha = int(40 + 90 * pulse)
                fx = xi[m]
                fy = yi[m]
                fcol = self.point_rgb[glow_idx][m]
                glow = pygame.Surface((self.view_w, self.view_h), pygame.SRCALPHA)
                for x, y, col in zip(fx, fy, fcol):
                    c = (int(col[0]), int(col[1]), int(col[2]), glow_alpha)
                    pygame.draw.circle(glow, c, (int(x), int(y)), radius)
                self._view_surf.blit(glow, (0, 0))

        self._perf_ms["draw.tissue"] = (_time.perf_counter() - t_tissue0) * 1000.0
        t_ov0 = _time.perf_counter()
        ov = float(self._view_detail_knobs()["overlay"])
        # Blueprint overlays (not in tissue cache — must follow Super-class live).
        if self.show_layer_graph and ov >= 0.25:
            self._draw_recipe_graph_overlay(
                cx,
                cy,
                view_lim_y,
                graph_f,
                max_edges=self._draw_cap(DRAW_GRAPH_EDGES_MAX),
            )
        if ov >= 0.40:
            self._draw_tract_sample(cx, cy, view_lim_y)
        # Rising / Fuse trails — Super-class A/D only (never without focus).
        if axon_f or dend_f:
            self._draw_growth_overlay(
                cx, cy, view_lim_y, axon_f=axon_f, dend_f=dend_f
            )
        if self.show_layer_basins and ov >= 0.30:
            # On top of tissue so cosmology wells stay readable.
            self._draw_basins_layer(cx, cy, view_lim_y, self.focus_basin_indices)
        if ov >= 0.20:
            self._draw_device_funnels(cx, cy, view_lim_y)
        if ov >= 0.35:
            self._draw_corridor_topology(cx, cy, view_lim_y)
            self._draw_channel_bed(cx, cy, view_lim_y)
        if ov >= 0.20:
            self._draw_orbit_compass(cx, cy, view_lim_y)
            self._draw_gravity_pendulum(cx, cy, view_lim_y)
        self._draw_point_lamp(cx, cy, view_lim_y)
        draw_devices(
            self,
            list(
                getattr(self, "device_list", None)
                or getattr(getattr(self, "blueprint", None), "device_list", [])
                or []
            ),
            cx,
            cy,
            view_lim_y,
        )
        if ov >= 0.30:
            self._draw_learn_flash(cx, cy, view_lim_y)
        self._perf_ms["draw.overlays"] = (_time.perf_counter() - t_ov0) * 1000.0

    def _draw_learn_flash(
        self, cx: float, cy: float, view_lim_y: int
    ) -> None:
        """Glow engaged tissue on real topography (neurites + somata — no chords).

        Super-class gates: soma blooms → N; neurite polys → A/D. Empty = hide.
        """
        assert self._view_surf is not None
        amp = float(getattr(self, "_learn_flash_amp", 0.0) or 0.0)
        scope_live = bool(
            getattr(self, "scope_on", False)
            or getattr(self, "motor_socket_on", False)
        )
        wave_live = bool(self.learn_on) or (
            scope_live and bool(getattr(self, "_activity_running", False))
        )
        if amp < 0.04 or not wave_live:
            return
        soma_f = set(self.focus_soma_indices)
        neur_f = set(self.focus_axon_indices) | set(self.focus_dendrite_indices)
        if not soma_f and not neur_f:
            return
        pre = getattr(self, "_learn_flash_pre", None)
        post = getattr(self, "_learn_flash_post", None)
        if pre is None or post is None or len(pre) == 0 or len(pre) != len(post):
            return
        ids = self.fly.soma_ids
        xyz = self.fly.soma_xyz
        if len(ids) == 0 or len(xyz) != len(ids):
            return
        engaged: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for a, b in zip(pre.tolist(), post.tolist()):
            a, b = int(a), int(b)
            engaged.add(a)
            engaged.add(b)
            if a != b:
                pairs.append((a, b))
        # Cap engaged set for FPS (prefer motors / flash order already ranked).
        if len(engaged) > 900:
            engaged = set(list(engaged)[:900])
            pairs = [(a, b) for a, b in pairs if a in engaged and b in engaged]

        overlay = self._overlay_surface(view_lim_y)
        pulse = float(self._pulse())
        glow = float(np.clip(0.55 + 0.45 * amp, 0.55, 1.0))

        # 1) Soma blooms for engaged neurons — class hue, not flat cyan.
        node_xy: dict[int, tuple[int, int]] = {}
        eng_idx = self._soma_indices_for_ids(engaged) if soma_f else []
        if eng_idx:
            e_xyz = xyz[np.asarray(eng_idx, dtype=np.int32)]
            e_ids = [int(ids[i]) for i in eng_idx]
            sx, sy = project_3d_batch(
                e_xyz, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
            )
            for nid, x, y in zip(e_ids, sx.tolist(), sy.tolist()):
                if not self._nid_in_focus(int(nid), soma_f):
                    continue
                xi, yi = int(x), int(y)
                if 0 <= xi < self.view_w and 0 <= yi < view_lim_y:
                    node_xy[int(nid)] = (xi, yi)
                    rgb = self._soma_rgb_for_nid(int(nid))
                    r_halo = max(5, int(7 + 6 * glow + 2 * pulse))
                    pygame.draw.circle(
                        overlay,
                        self._rgba_class(rgb, int(40 + 45 * glow), lift=0.25),
                        (xi, yi),
                        r_halo,
                    )
                    pygame.draw.circle(
                        overlay,
                        self._rgba_class(rgb, int(70 + 70 * glow), lift=0.55),
                        (xi, yi),
                        max(3, r_halo // 2),
                    )

        # 2) Real neurite polylines — skip device↔device local mesh (same as CNS: soma bloom only).
        device = set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
        polys = (
            self._learn_topo_polylines(pairs, engaged, device=device) if neur_f else []
        )
        n_edges_drawn = 0
        for pts3, owner in polys:
            if len(pts3) < 2:
                continue
            if not self._nid_in_focus(int(owner), neur_f):
                continue
            if len(pts3) > 28:
                step = max(1, (len(pts3) + 27) // 28)
                pts3 = pts3[::step]
            sx, sy = project_3d_batch(
                pts3, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
            )
            pts = [
                (int(x), int(y))
                for x, y in zip(sx, sy)
                if np.isfinite(x) and np.isfinite(y)
            ]
            if len(pts) < 2:
                continue
            # Keep segments that clip into the view.
            if not any(
                0 <= x < self.view_w and 0 <= y < view_lim_y for x, y in pts
            ):
                continue
            rgb = self._soma_rgb_for_nid(int(owner))
            pygame.draw.lines(
                overlay,
                self._rgba_class(rgb, int(60 + 55 * glow), lift=0.15),
                False,
                pts,
                max(3, int(4 + 3 * glow)),
            )
            pygame.draw.lines(
                overlay,
                self._rgba_class(rgb, int(150 + 90 * glow), lift=0.50),
                False,
                pts,
                max(2, int(2 + 2 * glow)),
            )
            pygame.draw.circle(
                overlay,
                self._rgba_class(rgb, int(160 + 70 * glow), lift=0.35),
                pts[0],
                3,
            )
            pygame.draw.circle(
                overlay,
                self._rgba_class(rgb, int(180 + 60 * glow), lift=0.65),
                pts[-1],
                4,
            )
            n_edges_drawn += 1
            if n_edges_drawn >= 220:
                break

        if n_edges_drawn <= 0 and not node_xy:
            return
        self._learn_topo_drawn = int(n_edges_drawn)
        self._learn_topo_nodes = int(len(node_xy))
        self._view_surf.blit(overlay, (0, 0))
        self._learn_flash_amp = float(amp) * 0.92

    def _contact_has_topo(self, pre_id: int, post_id: int) -> bool:
        """True if a materialized axon (or matching dendrite) exists for the pair."""
        pre_id, post_id = int(pre_id), int(post_id)
        neu = self.fly.neurons.get(pre_id)
        if neu is not None:
            for ax in neu.axons:
                if (
                    ax.partner_id is not None
                    and int(ax.partner_id) == post_id
                    and ax.status == "connected"
                    and len(ax.points) >= 2
                ):
                    return True
        post = self.fly.neurons.get(post_id)
        if post is not None:
            for den in post.dendrites:
                if (
                    den.partner_id is not None
                    and int(den.partner_id) == pre_id
                    and len(den.points) >= 2
                ):
                    return True
        return False

    def _soma_rgb_for_nid(self, nid: int) -> tuple[int, int, int]:
        """Class RGB for a neuron id (same palette as tissue stamps)."""
        idxs = self._soma_indices_for_ids((int(nid),))
        if idxs:
            i = int(idxs[0])
            pr = getattr(self, "point_rgb", None)
            if pr is not None and i < len(pr):
                c = pr[i]
                return int(c[0]), int(c[1]), int(c[2])
            pc = getattr(self, "points_cls", None)
            if pc is not None and i < len(pc):
                return self._class_color(int(pc[i]))
        neu = self.fly.neurons.get(int(nid))
        if neu is not None:
            ci = getattr(neu, "cls", None)
            if ci is None:
                ci = getattr(neu, "class_idx", None)
            if ci is not None:
                return self._class_color(int(ci))
        return (180, 180, 200)

    def _cls_of_nid(self, nid: int) -> int | None:
        """Super-class index for neuron id, or None if unknown."""
        idxs = self._soma_indices_for_ids((int(nid),))
        if idxs:
            i = int(idxs[0])
            pc = getattr(self, "points_cls", None)
            if pc is not None and i < len(pc):
                return int(pc[i])
        neu = self.fly.neurons.get(int(nid))
        if neu is not None:
            ci = getattr(neu, "cls", None)
            if ci is None:
                ci = getattr(neu, "class_idx", None)
            if ci is not None:
                return int(ci)
        return None

    def _nid_in_focus(self, nid: int, focus: set[int]) -> bool:
        if not focus:
            return False
        ci = self._cls_of_nid(int(nid))
        return ci is not None and int(ci) in focus

    @staticmethod
    def _rgba_class(
        rgb: tuple[int, int, int],
        alpha: int,
        *,
        lift: float = 0.40,
    ) -> tuple[int, int, int, int]:
        """Keep hue; lift toward white for glow readability."""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        t = float(np.clip(lift, 0.0, 1.0))
        r = min(255, int(r + (255 - r) * t))
        g = min(255, int(g + (255 - g) * t))
        b = min(255, int(b + (255 - b) * t))
        return (r, g, b, int(np.clip(alpha, 0, 255)))

    def _learn_topo_polylines(
        self,
        pairs: list[tuple[int, int]],
        engaged: set[int],
        *,
        device: set[int] | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """Collect real neurite polylines for engaged contacts (no invented chords).

        Returns (xyz Nx3, owner_nid) so overlays can tint by superclass.
        Device↔device pack wires are omitted — those stay on the normal axon layer.
        Learn glow only traces Fuse bridges and CNS topography.
        """
        out: list[tuple[np.ndarray, int]] = []
        seen: set[tuple[str, int]] = set()
        device = set(device or ())

        def _is_device_local(a: int, b: int) -> bool:
            return int(a) in device and int(b) in device

        def _add(pts: np.ndarray, kind: str, key_id: int, owner_nid: int) -> None:
            key = (kind, int(key_id))
            if key in seen:
                return
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            if len(arr) < 2:
                return
            seen.add(key)
            out.append((arr, int(owner_nid)))

        # Explicit flash pairs first (bridges / CNS only).
        for a, b in pairs:
            if _is_device_local(a, b):
                continue
            neu = self.fly.neurons.get(int(a))
            if neu is not None:
                for ax in neu.axons:
                    if (
                        ax.partner_id is not None
                        and int(ax.partner_id) == int(b)
                        and ax.status == "connected"
                    ):
                        _add(ax.points, "ax", int(ax.id), int(a))
            post = self.fly.neurons.get(int(b))
            if post is not None:
                for den in post.dendrites:
                    if den.partner_id is not None and int(den.partner_id) == int(a):
                        _add(den.points, "den", int(den.id), int(b))

        # Connected neurites between engaged neurons — never device-local mesh.
        for nid in engaged:
            neu = self.fly.neurons.get(int(nid))
            if neu is None:
                continue
            for ax in neu.axons:
                if ax.status != "connected" or ax.partner_id is None:
                    continue
                pid = int(ax.partner_id)
                if pid not in engaged:
                    continue
                if _is_device_local(int(nid), pid):
                    continue
                _add(ax.points, "ax", int(ax.id), int(nid))
            for den in neu.dendrites:
                if den.partner_id is None:
                    continue
                pid = int(den.partner_id)
                if pid not in engaged:
                    continue
                if _is_device_local(int(nid), pid):
                    continue
                _add(den.points, "den", int(den.id), int(nid))

        # Live growth-field trails for engaged owners (Fuse into CNS, not pack mesh).
        if len(self.fly.growth_owner) and len(self.fly.growth_poly_n):
            for i in range(len(self.fly.growth_owner)):
                own = int(self.fly.growth_owner[i])
                if own not in engaged:
                    continue
                if int(self.fly.growth_status[i]) != int(STATUS_CONNECTED):
                    continue
                k = int(self.fly.growth_poly_n[i])
                if k < 2:
                    continue
                goal = (
                    int(self.fly.growth_goal[i])
                    if len(self.fly.growth_goal) > i
                    else -1
                )
                if goal >= 0 and _is_device_local(own, goal):
                    continue
                if goal >= 0 and goal not in engaged:
                    if not any(a == own for a, _ in pairs):
                        continue
                _add(self.fly.growth_poly[i, :k], "g", i, own)
        return out

    def _learn_edge_screen_poly(
        self,
        pre_id: int,
        post_id: int,
        cx: float,
        cy: float,
        view_lim_y: int,
    ) -> list[tuple[int, int]] | None:
        """Project the real axon polyline for a contact, if materialized."""
        neu = self.fly.neurons.get(int(pre_id))
        if neu is None:
            return None
        best = None
        for ax in neu.axons:
            if ax.partner_id is None or int(ax.partner_id) != int(post_id):
                continue
            if ax.status != "connected":
                continue
            pts = np.asarray(ax.points, dtype=np.float32).reshape(-1, 3)
            if len(pts) < 2:
                continue
            best = pts
            break
        if best is None:
            return None
        if len(best) > 24:
            step = max(1, (len(best) + 23) // 24)
            best = best[::step]
        sx, sy = project_3d_batch(
            best,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        out = [
            (int(x), int(y))
            for x, y in zip(sx, sy)
            if np.isfinite(x) and np.isfinite(y)
        ]
        return out if len(out) >= 2 else None

    def _learn_err_deg(self) -> float:
        err = float(getattr(self, "turret_aim_err", float("nan")))
        if not math.isfinite(err) and hasattr(self, "_update_turret_aim_metric"):
            self._update_turret_aim_metric()
            err = float(getattr(self, "turret_aim_err", float("nan")))
        if not math.isfinite(err):
            return float("nan")
        return float(err) * (180.0 / math.pi)

    def _learn_is_live(self) -> bool:
        return bool(
            self.learn_on
            and self.turret_on
            and self.sim_running
            and getattr(self, "_activity_running", False)
            and self.fly.is_adult()
        )

    def _learn_note_err(self, err_deg: float) -> None:
        """Latch session-start ε the first time we see a live finite aim."""
        if not math.isfinite(err_deg):
            return
        if not hasattr(self, "_learn_err_start_deg"):
            self._learn_err_start_deg = float("nan")
        if self._learn_is_live() and not math.isfinite(
            float(self._learn_err_start_deg)
        ):
            self._learn_err_start_deg = float(err_deg)

    def _learn_on_freeze(self) -> None:
        deg = self._learn_err_deg()
        self._learn_err_freeze_deg = (
            float(deg) if math.isfinite(deg) else float("nan")
        )

    def _learn_on_train_again(self) -> None:
        deg = self._learn_err_deg()
        self._learn_err_start_deg = (
            float(deg) if math.isfinite(deg) else float("nan")
        )
        self._learn_err_freeze_deg = float("nan")
        self._track_reset()
        self._learn_progress_reset()

    def _learn_on_reset(self) -> None:
        self._learn_err_start_deg = float("nan")
        self._learn_err_freeze_deg = float("nan")
        self._track_reset()
        self._learn_progress_reset()

    def _learn_progress_note(self, track: float, moving: bool, dw: float, dt: float) -> None:
        """Update live-time / early Track / Δw peak for trained + warn gates."""
        if not hasattr(self, "_learn_live_s"):
            self._learn_progress_reset()
        live = self._learn_is_live()
        frozen = bool(
            getattr(self, "motor_socket", None) is not None
            and getattr(self.motor_socket, "frozen", False)
        )
        if not live or frozen:
            return
        if moving:
            self._learn_live_s = float(self._learn_live_s) + float(dt)
        self._learn_dw_peak = max(float(self._learn_dw_peak), float(dw))
        if math.isfinite(track):
            if not math.isfinite(float(self._track_peak)):
                self._track_peak = float(track)
            else:
                self._track_peak = max(float(self._track_peak), float(track))
            # Latch early Track after a short warmup under motion.
            if moving and float(self._learn_live_s) >= 2.0:
                if not math.isfinite(float(self._track_early)):
                    self._track_early = float(track)
            if moving and track >= float(TRACK_READY):
                self._track_ready_hold_s = float(self._track_ready_hold_s) + float(dt)
            elif moving:
                self._track_ready_hold_s = max(
                    0.0, float(self._track_ready_hold_s) - 0.5 * float(dt)
                )

        early = float(self._track_early)
        improved = (
            math.isfinite(track)
            and math.isfinite(early)
            and (track - early) >= float(TRACK_IMPROVE_MIN)
        )
        plastic = float(self._learn_dw_peak) >= float(TRACK_DW_SAW)
        stuck = bool(
            moving
            and float(self._learn_live_s) >= float(TRACK_WARN_AFTER_S)
            and math.isfinite(track)
            and track < float(TRACK_WARN_MAX)
            and not improved
            and not plastic
        )
        self._learn_not_learning = stuck

    def _learn_is_trained(self, track: float, track_ok: bool, dw: float, rpe: float) -> bool:
        """True when competence held under ±20% stim and plasticity settled."""
        if not track_ok:
            return False
        if abs(float(rpe)) >= 0.02 or float(dw) >= 0.03:
            return False
        early = float(getattr(self, "_track_early", float("nan")))
        improved = (
            math.isfinite(track)
            and math.isfinite(early)
            and (track - early) >= float(TRACK_IMPROVE_MIN)
        )
        plastic = float(getattr(self, "_learn_dw_peak", 0.0) or 0.0) >= float(
            TRACK_DW_SAW
        )
        held = float(getattr(self, "_track_ready_hold_s", 0.0) or 0.0) >= float(
            TRACK_HOLD_S
        )
        # Trained = high Track under jitter, and either learned or held long enough.
        return bool(improved or plastic or held)

    def _learn_socket_levels(self) -> list[float] | None:
        sock = getattr(self, "motor_socket", None)
        if sock is None or not getattr(self, "motor_socket_on", False):
            return None
        try:
            return [float(x) for x in sock.levels(self)[:4]]
        except Exception:
            g = getattr(sock, "gains", None)
            if g is None:
                return None
            arr = np.asarray(g, dtype=np.float64).ravel()
            if len(arr) < 4:
                return None
            m = float(np.max(arr)) if len(arr) else 1.0
            m = max(m, 1e-6)
            return [float(np.clip(v / m, 0.0, 1.0)) for v in arr[:4]]

    def _learn_decode_socket(self, levels: list[float]) -> str:
        if len(levels) < 4:
            return "—"
        left = 0.5 * (levels[0] + levels[1])
        right = 0.5 * (levels[2] + levels[3])
        lo = 0.5 * (levels[0] + levels[2])
        hi = 0.5 * (levels[1] + levels[3])
        yaw = float(left - right)
        pitch = float(lo - hi)
        parts: list[str] = []
        if abs(yaw) < 0.08:
            parts.append("yaw·")
        else:
            parts.append("yaw←" if yaw > 0 else "yaw→")
        if abs(pitch) < 0.08:
            parts.append("pitch·")
        else:
            parts.append("pitch↑" if pitch > 0 else "pitch↓")
        return "  ".join(parts)

    def _constructor_panel_kind(self) -> str:
        """Which story the Learn badge tells: growth (Yield) or learn (plasticity).

        Can't train the socket until connections are grown → adult.
        """
        fly = getattr(self, "fly", None)
        if fly is None or not getattr(fly, "is_adult", lambda: False)():
            return "growth"
        return "learn"

    def _yield_phase(self) -> dict:
        """Human-facing growth / Yield state for the shared Learn badge."""
        yld = self._recipe_yield()
        verdict = str(yld.get("verdict", "IDLE"))
        accent = self._yield_verdict_color(verdict)
        armed = int(yld.get("armed", 0))
        pct = float(yld.get("pct", 0.0))
        ok = int(yld.get("ok", 0))
        fail = int(yld.get("fail", 0))
        open_n = int(yld.get("open", 0))
        if armed == 0:
            pre = getattr(self.fly, "recipe_pre", None)
            try:
                n_pre = 0 if pre is None else int(len(pre))
            except TypeError:
                n_pre = 0
            if n_pre == 0:
                badge = "GROWTH · no recipe"
            else:
                badge = "GROWTH · Start to arm"
            phase = "IDLE"
            ready = False
        elif verdict == "RUNNING":
            phase = "RUNNING"
            badge = f"GROWING · {pct:.0f}% · still {open_n:,}"
            ready = False
        elif verdict == "CONNECTED":
            phase = "CONNECTED"
            badge = f"CONNECTED · {pct:.0f}% · Adult?"
            ready = True
        elif verdict == "PARTIAL":
            phase = "PARTIAL"
            badge = f"PARTIAL · {pct:.0f}% · Adult?"
            ready = open_n == 0
        else:
            phase = "FAILED"
            badge = f"FAILED · {pct:.0f}%"
            ready = False
        # Weakest Out classes (armed only), for competence strip.
        weak: list[dict] = []
        for row in yld.get("by_class", []) or []:
            if int(row.get("out_armed", 0)) <= 0:
                continue
            weak.append(row)
        weak.sort(key=lambda r: float(r.get("out_pct", 0.0)))
        return {
            "kind": "growth",
            "phase": phase,
            "accent": accent,
            "badge": badge,
            "yld": yld,
            "pct": pct,
            "ok": ok,
            "fail": fail,
            "open": open_n,
            "armed": armed,
            "ready": ready,
            "weak": weak[:4],
            "verdict": verdict,
        }

    def _yield_detail_blocks(self) -> list[tuple]:
        """Growth story inside the Learn badge panel."""
        st = self._yield_phase()
        yld = st["yld"]
        rows: list[tuple] = [
            ("head", str(st["badge"])),
            ("sec", "Progress"),
            (
                "kv",
                "Yield",
                f"{float(st['pct']):.0f}%  ·  {int(st['ok']):,}/{int(st['armed']):,}",
            ),
            ("kv", "open / fail", f"{int(st['open']):,}  /  {int(st['fail']):,}"),
            ("kv", "verdict", str(st["verdict"])),
        ]
        rows.append(("sec", "Weak Out"))
        weak = st.get("weak") or []
        if not weak:
            rows.append(("hint", "No armed tips yet — Start / Fuse"))
        else:
            for row in weak:
                name = CLASS_LABELS.get(
                    row["name"], str(row["name"]).replace("_", " ").title()
                )
                rows.append(
                    (
                        "kv",
                        name,
                        f"Out {float(row['out_pct']):.0f}%  "
                        f"{int(row['out_ok']):,}/{int(row['out_armed']):,}"
                        + (
                            f"  · still {int(row['out_open']):,}"
                            if int(row.get("out_open", 0))
                            else ""
                        ),
                    )
                )
        rows.append(("sec", "Ready"))
        if st["ready"]:
            rows.append(("kv", "Ready", "growth settled · enter Adult"))
        elif st["phase"] == "RUNNING":
            rows.append(("kv", "Ready", "still growing — learn blocked"))
        elif st["phase"] == "IDLE":
            rows.append(("kv", "Ready", "arm tips first"))
        else:
            rows.append(("kv", "Ready", "weak yield — grow or abandon"))
        rows.append(("hint", "Learn / Freeze unlock after Adult"))

        mark = "▾" if getattr(self, "_learn_rule_open", False) else "▸"
        rows.append(("rule", f"By class {mark}"))
        if getattr(self, "_learn_rule_open", False):
            rows.append(
                ("hint", "Out% = axons from class · In% = aimed at class")
            )
            for row in yld.get("by_class", []) or []:
                if int(row.get("out_armed", 0)) == 0 and int(row.get("in_armed", 0)) == 0:
                    continue
                name = CLASS_LABELS.get(
                    row["name"], str(row["name"]).replace("_", " ").title()
                )
                out_s = (
                    f"Out {float(row['out_pct']):.0f}%"
                    if int(row.get("out_armed", 0))
                    else "Out —"
                )
                in_s = (
                    f"In {float(row['in_pct']):.0f}%"
                    if int(row.get("in_armed", 0))
                    else "In —"
                )
                rows.append(("kv", name, f"{out_s}  ·  {in_s}"))
        return rows

    def _port_nn_active(self) -> bool:
        """True when Sensor-port NN owns (or is training) the effector."""
        if getattr(self, "sensor_port_nn", None) is None:
            return False
        return bool(getattr(self, "sensor_port_on", False)) or bool(
            getattr(self, "sensor_port_train", False)
        )

    def _learn_phase(self) -> dict:
        """Human-facing learn state for badge + detail panel."""
        # AI Brain is a separate learner — never show CNS "Learn off" while it runs.
        port = getattr(self, "sensor_port_nn", None)
        if port is not None and self._port_nn_active():
            try:
                badge, accent = port.format_corner_badge(self)
            except Exception:
                badge, accent = "AI Brain", (200, 180, 120)
            p = port.progress(self) if hasattr(port, "progress") else {}
            trk = self._track_snapshot()
            return {
                "kind": "port",
                "phase": str(p.get("mode", "port")).upper(),
                "accent": accent,
                "badge": badge,
                "err_deg": float(p.get("eps_deg", float("nan"))),
                "start_deg": float("nan"),
                "freeze_deg": float("nan"),
                "trend": str(p.get("trend", "·")),
                "rpe": 0.0,
                "dw": float(p.get("loss", 0.0) or 0.0),
                "track": float(trk.get("score", float("nan"))),
                "track_moving": bool(trk.get("moving", False)),
                "track_spin": float(trk.get("spin", 0.0) or 0.0),
                "stim_spin_set": float(getattr(self, "lamp_speed", 0.0) or 0.0),
                "track_window_s": float(trk.get("window_s", 0.0) or 0.0),
                "track_early": float(getattr(self, "_track_early", float("nan"))),
                "track_rms": float(trk.get("rms_deg", float("nan"))),
                "track_lag": float(trk.get("lag_deg", float("nan"))),
                "track_slips": int(trk.get("slips", 0) or 0),
                "track_n": int(trk.get("n", 0) or 0),
                "not_learning": False,
                "jitter": False,
                "sock_on": False,
                "frozen": bool(getattr(port, "frozen", False)),
                "live": True,
                "ready": False,
                "levels": None,
                "report": {},
            }

        tr = getattr(self, "_trainer", None)
        r: dict = {}
        if tr is not None and hasattr(tr, "report"):
            try:
                r = dict(tr.report({}))
            except Exception:
                r = {}
        err_deg = float(r.get("err_deg", float("nan")))
        if not math.isfinite(err_deg):
            err_deg = self._learn_err_deg()
        self._learn_note_err(err_deg)
        rpe = float(r.get("rpe", getattr(tr, "last_rpe", 0.0) if tr else 0.0) or 0.0)
        sock_on = bool(getattr(self, "motor_socket_on", False)) and getattr(
            self, "motor_socket", None
        ) is not None
        frozen = bool(
            sock_on and getattr(self.motor_socket, "frozen", False)
        )
        if sock_on:
            dw = float(getattr(self, "motor_socket_dw_ema", 0.0) or 0.0)
        else:
            dw = float(r.get("dw_ema", getattr(self, "learn_dw_ema", 0.0)) or 0.0)
        live = self._learn_is_live()
        trk = self._track_snapshot()
        track = float(trk["score"])
        track_ok = bool(
            trk["moving"]
            and math.isfinite(track)
            and track >= float(TRACK_READY)
            and int(trk["slips"]) <= int(TRACK_SETTLE_SLIPS)
            and int(trk["n"]) >= 20
        )
        # Progress meters updated in _update_turret_aim_metric (once / frame).
        not_learning = bool(getattr(self, "_learn_not_learning", False))
        trained = self._learn_is_trained(track, track_ok, dw, rpe)
        # SETTLING only when the net is actually trained (not one-shot ε).
        ready = bool(live and not frozen and trained)
        start = float(getattr(self, "_learn_err_start_deg", float("nan")))
        freeze_e = float(getattr(self, "_learn_err_freeze_deg", float("nan")))
        # Trend vs session start (fallback: RPE sign).
        trend = "·"
        if math.isfinite(err_deg) and math.isfinite(start) and abs(err_deg - start) > 0.8:
            trend = "↓" if err_deg < start else "↑"
        elif live and abs(rpe) > 0.002:
            trend = "↓" if rpe > 0 else "↑"

        def _trk_badge() -> str:
            if not math.isfinite(track):
                return "Track —"
            if not trk["moving"]:
                return f"Track {track:.0%}·still"
            return f"Track {track:.0%}"

        jitter_on = self._stim_jitter_active()

        if not self.learn_on:
            phase, accent = "OFF", (120, 120, 140)
            badge = "Learn off"
        elif not self.turret_on:
            phase, accent = "BLOCKED", (150, 150, 160)
            badge = "BLOCKED · need turret"
        elif frozen:
            phase, accent = "FROZEN", (200, 230, 140)
            if math.isfinite(track):
                badge = f"FROZEN · {_trk_badge()}"
            elif math.isfinite(err_deg):
                badge = f"FROZEN · ε {err_deg:.0f}°"
            else:
                badge = "FROZEN · W locked"
        elif live and ready:
            phase, accent = "SETTLING", (180, 230, 160)
            badge = f"SETTLING · {_trk_badge()} · Freeze?"
        elif live and not_learning:
            phase, accent = "STUCK", (230, 160, 90)
            badge = f"⚠ not learning · {_trk_badge()}"
        elif live:
            phase, accent = "TRAINING", (160, 210, 255)
            mode = "socket" if sock_on else "Brain"
            jtag = " ±20%" if jitter_on else ""
            if math.isfinite(track):
                badge = (
                    f"TRAINING · {mode} · {_trk_badge()}{jtag} · ε {err_deg:.0f}° {trend}"
                ).rstrip()
            else:
                badge = f"TRAINING · {mode} · ε {err_deg:.0f}° {trend}".rstrip()
        elif self.learn_on and self.turret_on:
            phase, accent = "ARMED", (140, 170, 210)
            badge = "ARMED · Run"
        else:
            phase, accent = "ARMED", (140, 170, 210)
            badge = "ARMED"

        levels = self._learn_socket_levels()
        return {
            "kind": "learn",
            "phase": phase,
            "accent": accent,
            "badge": badge,
            "err_deg": err_deg,
            "start_deg": start,
            "freeze_deg": freeze_e,
            "trend": trend,
            "rpe": rpe,
            "dw": dw,
            "ready": ready,
            "trained": trained,
            "not_learning": not_learning,
            "jitter": jitter_on,
            "stim_spin": float(self.effective_lamp_speed()),
            "stim_spin_set": float(getattr(self, "lamp_speed", 0.0) or 0.0),
            "track": track,
            "track_rms": float(trk["rms_deg"]),
            "track_lag": float(trk["lag_deg"]),
            "track_slips": int(trk["slips"]),
            "track_n": int(trk["n"]),
            "track_moving": bool(trk["moving"]),
            "track_spin": float(trk["spin"]),
            "track_window_s": float(trk["window_s"]),
            "track_early": float(getattr(self, "_track_early", float("nan"))),
            "live_s": float(getattr(self, "_learn_live_s", 0.0) or 0.0),
            "dw_peak": float(getattr(self, "_learn_dw_peak", 0.0) or 0.0),
            "live": live,
            "frozen": frozen,
            "sock_on": sock_on,
            "levels": levels,
            "report": r,
        }

    def _port_nn_detail_blocks(self, port: Any) -> list[tuple]:
        """Learn-panel detail when Sensor-port NN owns the effector."""
        p = port.progress(self)
        mode = str(p.get("mode", "off"))
        loss = float(p.get("loss", float("nan")))
        start = float(p.get("loss_start", float("nan")))
        best = float(p.get("loss_best", float("nan")))
        track = float(p.get("track", float("nan")))
        eps = float(p.get("eps_deg", float("nan")))
        trend = str(p.get("trend", "·"))
        rows: list[tuple] = [
            ("head", f"AI Brain · {mode.upper()}"),
        ]
        rows.append(("sec", "Progress"))
        rows.append(("kv", "steps", f"{int(p.get('step_n', 0))}"))
        loss_s = f"{loss:.4f}" if math.isfinite(loss) else "—"
        start_s = f"{start:.3f}" if math.isfinite(start) else "—"
        best_s = f"{best:.4f}" if math.isfinite(best) else "—"
        band = str(p.get("loss_band", "—"))
        loss_col = tuple(p.get("loss_rgb", (210, 225, 245)))
        if not (isinstance(loss_col, tuple) and len(loss_col) == 3):
            loss_col = (210, 225, 245)
        rows.append(
            (
                "kv",
                "loss",
                f"{loss_s}  {trend}  {band}  · start {start_s}",
                loss_col,
            )
        )
        rows.append(("kv", "best", best_s, loss_col if math.isfinite(best) else (210, 225, 245)))
        rows.append(("kv", "replay buf", f"{int(p.get('buf', 0))}"))
        rows.append(("sec", "Track"))
        if math.isfinite(track):
            rows.append(("kv", "in cone", f"{track:.0%}  · under Spin"))
        else:
            rows.append(("kv", "in cone", "— · need Run + Spin"))
        eps_s = f"{eps:.1f}°" if math.isfinite(eps) else "—"
        rows.append(("kv", "ε now", eps_s))
        rows.append(
            (
                "kv",
                "cmd",
                f"yaw {float(p.get('cmd_yaw', 0.0)):+.2f}  "
                f"pitch {float(p.get('cmd_pitch', 0.0)):+.2f}",
            )
        )
        rows.append(("sec", "Policy"))
        rows.append(("kv", "lr", f"{float(p.get('lr', 0.0)):.3f}"))
        rows.append(
            (
                "kv",
                "W",
                "locked" if bool(p.get("frozen")) else "train + replay",
            )
        )
        rows.append(("kv", "CNS Brain", "frozen · contact_gain unused"))
        if mode == "train":
            rows.append(
                (
                    "hint",
                    "Train until loss↓ and Track↑ under Spin, then Freeze AI Brain.",
                )
            )
        elif mode == "frozen":
            rows.append(
                (
                    "hint",
                    "Frozen RUN — weights locked; Track should hold without Train.",
                )
            )
        else:
            rows.append(("hint", "Drive on · Train off — inference only."))
        rows.append(
            (
                "warn",
                "AI Brain exclusive — CNS Brain hop/carve not driving turret",
            )
        )
        return rows

    def _learn_mode_label(self) -> tuple[str, tuple[int, int, int]]:
        """Short badge text + accent — Yield while rising, plasticity when adult."""
        if self._constructor_panel_kind() == "growth":
            st = self._yield_phase()
            return str(st["badge"]), tuple(st["accent"])  # type: ignore[return-value]
        port = getattr(self, "sensor_port_nn", None)
        if port is not None and self._port_nn_active():
            try:
                return port.format_corner_badge(self)
            except Exception:
                return "AI Brain", (200, 180, 120)
        st = self._learn_phase()
        return str(st["badge"]), tuple(st["accent"])  # type: ignore[return-value]

    def _learn_plasticity_detail_blocks(self) -> list[tuple]:
        """Adult plasticity story (Aim / Socket / Freeze / AI Brain)."""
        if self._port_nn_active():
            port = getattr(self, "sensor_port_nn", None)
            if port is not None:
                return self._port_nn_detail_blocks(port)

        st = self._learn_phase()
        err = float(st["err_deg"])
        start = float(st["start_deg"])
        freeze_e = float(st["freeze_deg"])
        trend = str(st["trend"])
        rpe = float(st["rpe"])
        dw = float(st["dw"])
        rows: list[tuple] = [
            ("head", str(st["badge"])),
        ]
        # Tracking competence (primary KPI under Spin)
        rows.append(("sec", "Track"))
        track = float(st.get("track", float("nan")))
        moving = bool(st.get("track_moving", False))
        spin = float(st.get("track_spin", 0.0) or 0.0)
        spin_set = float(st.get("stim_spin_set", spin) or spin)
        win_s = float(st.get("track_window_s", TRACK_WINDOW_MIN_S) or TRACK_WINDOW_MIN_S)
        if bool(st.get("not_learning")):
            rows.append(
                (
                    "warn",
                    "Not learning — Track flat. Check Fuse ocelli→motor, Spin, hops.",
                )
            )
        if math.isfinite(track):
            tag = "under Spin" if moving else "still · Spin↑"
            rows.append(("kv", "in cone", f"{track:.0%}  ·  {tag}"))
        else:
            rows.append(("kv", "in cone", "— · need Run"))
        early = float(st.get("track_early", float("nan")))
        if math.isfinite(early) and math.isfinite(track):
            rows.append(
                ("kv", "early→now", f"{early:.0%} → {track:.0%}")
            )
        rms = float(st.get("track_rms", float("nan")))
        lag = float(st.get("track_lag", float("nan")))
        slips = int(st.get("track_slips", 0) or 0)
        n = int(st.get("track_n", 0) or 0)
        if math.isfinite(rms):
            rows.append(("kv", "RMS ε", f"{rms:.1f}°"))
        if math.isfinite(lag):
            rows.append(("kv", "lag", f"{lag:.1f}°"))
        rows.append(("kv", "slips", f"{slips}  ·  n={n} · {win_s:.1f}s"))
        if bool(st.get("jitter")):
            rows.append(
                (
                    "kv",
                    "Spin live",
                    f"{spin:.2f}  ← set {spin_set:.2f} ±{TRACK_STIM_JITTER:.0%}",
                )
            )
        else:
            rows.append(
                (
                    "kv",
                    "Spin",
                    f"{spin:.2f} rev/s"
                    + ("" if moving else f"  ·  need ≥{TRACK_MIN_SPIN:.2f}"),
                )
            )
        rows.append(
            (
                "hint",
                "CNS: slew from hop motors · Learn = RPE carve on corridor "
                f"· Track≥{TRACK_READY:.0%}",
            )
        )

        # Aim snapshot (instant ε — not the ready gate)
        rows.append(("sec", "Aim"))
        now_s = f"{err:.1f}°" if math.isfinite(err) else "—"
        start_s = f"{start:.1f}°" if math.isfinite(start) else "—"
        aim_line = f"{now_s}  ←  start {start_s}"
        if trend in ("↓", "↑"):
            aim_line += f"  {trend}"
        rows.append(("kv", "ε now", aim_line))
        if math.isfinite(freeze_e):
            rows.append(("kv", "at freeze", f"{freeze_e:.1f}°"))
        cone = float(TURRET_AIM_HALF_DEG)
        if math.isfinite(err):
            in_cone = err <= cone
            rows.append(
                (
                    "kv",
                    "cone",
                    f"{'in' if in_cone else 'out'} · ≤{cone:.0f}°",
                )
            )

        # Drive path: CNS hop vs legacy socket
        if st.get("sock_on"):
            rows.append(("sec", "Socket map"))
            levels = st.get("levels")
            if levels is not None:
                rows.append(
                    (
                        "bars",
                        list(zip(("LA", "LP", "RA", "RP"), levels)),
                        bool(st["frozen"]),
                    )
                )
                rows.append(("kv", "drive", self._learn_decode_socket(list(levels))))
                alpha = float(getattr(self, "_socket_drive_alpha", 0.0) or 0.0)
                if alpha > 1e-3 and not bool(st["frozen"]):
                    rows.append(
                        (
                            "kv",
                            "slew mix",
                            f"{alpha:.0%} teacher · {1.0 - alpha:.0%} W",
                        )
                    )
                rows.append(
                    (
                        "kv",
                        "W",
                        "locked" if st["frozen"] else "train",
                    )
                )
            rows.append(("warn", "Socket ON = bypass CNS (legacy)"))
        else:
            rows.append(("sec", "CNS path"))
            rows.append(("kv", "slew", "hop motor clamps"))
            rule = getattr(self, "rule", None)
            tmix = float(getattr(rule, "teacher_mix", 0.0) or 0.0) if rule else 0.0
            rows.append(("kv", "teacher_mix", f"{tmix:.2f} → clamps only"))
            n_g = int(getattr(self, "learn_n_edges", 0) or 0)
            rows.append(("kv", "carve edges", f"{n_g}"))
            rows.append(
                ("hint", "ocelli seed → hop → motors · RPE ±carve on corridor")
            )

        # Pulse / ready
        rows.append(("sec", "Pulse"))
        rows.append(("kv", "RPE", f"{rpe:+.3f}"))
        rows.append(("kv", "Δw", f"{dw:.3f}"))
        if st["frozen"]:
            rows.append(("kv", "Ready", "frozen — Train again to resume"))
        elif st["ready"]:
            rows.append(("kv", "Ready", "SETTLING · Track ok"))
        elif st.get("not_learning"):
            rows.append(("kv", "Ready", "⚠ not learning"))
        elif st["live"] and not moving:
            rows.append(("kv", "Ready", "raise Spin/swing to score Track"))
        elif st["live"] and math.isfinite(track) and track < float(TRACK_READY):
            rows.append(
                ("kv", "Ready", f"Track {track:.0%} < {TRACK_READY:.0%}")
            )
        elif st["live"]:
            rows.append(("kv", "Ready", "still training"))
        else:
            rows.append(("kv", "Ready", "—"))

        # Frozen meaning
        rows.append(("sec", "Frozen means"))
        if st["frozen"]:
            rows.append(("hint", "W locked · teacher off · hop + sense live"))
        else:
            rows.append(("hint", "not frozen — teacher can still update W"))

        # Advanced knobs
        mark = "▾" if getattr(self, "_learn_rule_open", False) else "▸"
        rows.append(("rule", f"Rule {mark}"))
        if getattr(self, "_learn_rule_open", False):
            rule = getattr(self, "rule", None)
            learn_n = int(getattr(self, "learn_every", 8) or 8)
            hops = int(getattr(self, "loop_hops", 1) or 1)
            r = st.get("report") or {}
            rows.append(("kv", "Cadence", f"hop×{hops} · learn/{learn_n}"))
            rows.append(
                (
                    "kv",
                    "Gains / flash",
                    f"{int(r.get('n_gains', getattr(self, 'learn_n_edges', 0) or 0))} · "
                    f"{int(r.get('plast', getattr(self, '_learn_flash_n', 0) or 0))}Δ",
                )
            )
            if rule is not None:
                rows.append(
                    ("kv", "Teacher mix", f"{float(getattr(rule, 'teacher_mix', 0.0)):.3f}")
                )
                rows.append(("kv", "Hebb", f"{float(getattr(rule, 'hebb', 0.0)):.3f}"))
                rows.append(
                    (
                        "kv",
                        "Path credit",
                        f"{float(getattr(rule, 'path_credit', 0.0)):.3f}",
                    )
                )
                rows.append(
                    ("kv", "Carve lr", f"{float(getattr(rule, 'carve_lr', 0.0)):.4f}")
                )
                rows.append(("kv", "Lr", f"{float(getattr(rule, 'lr', 0.0)):.4f}"))
            nan_g = int(r.get("nan_gains", 0) or 0)
            if nan_g:
                rows.append(("kv", "WARN nan", str(nan_g)))
        return rows

    def _learn_detail_blocks(self) -> list[tuple]:
        """Structured rows for the learn badge — growth Yield or adult plasticity."""
        if self._constructor_panel_kind() == "growth":
            return self._yield_detail_blocks()
        return self._learn_plasticity_detail_blocks()

    def _learn_detail_rows(self) -> list[tuple[str, str]]:
        """Compat flat view (tests / callers). Prefer ``_learn_detail_blocks``."""
        out: list[tuple[str, str]] = []
        for row in self._learn_detail_blocks():
            kind = row[0]
            if kind in ("head", "sec", "hint", "warn", "rule"):
                out.append((str(row[1]), ""))
            elif kind == "kv":
                out.append((str(row[1]), str(row[2])))
            elif kind == "bars":
                bits = [
                    f"{lab}:{float(val):.2f}" for lab, val in row[1]  # type: ignore[misc]
                ]
                tag = "locked" if row[2] else "live"
                out.append((f"sites ({tag})", " ".join(bits)))
        return out

    def _draw_fps(self) -> None:
        """Update FPS EMA (chip is drawn in ``_draw_top_bar``)."""
        raw = float(self.clock.get_fps())
        if raw > 0.5:
            self._fps_smooth = 0.82 * float(self._fps_smooth) + 0.18 * raw
        self._fps_panel_rect = None
        self._fps_corner_h = 0

    def _draw_fps_panel(self) -> None:
        """Frame-time breakdown under the left badge stack."""
        if not self._fps_panel_open:
            self._fps_panel_rect = None
            return
        fps = float(self._fps_smooth)
        frame_ms = 1000.0 / max(fps, 0.1)
        order = [
            ("sim.hop", "Hop / activity"),
            ("sim.learn", "Learn / plasticity"),
            ("sim.sense", "Sense (loop)"),
            ("sim.actuate", "Actuate"),
            ("draw.tissue", "Draw tissue"),
            ("draw.overlays", "Draw overlays"),
            ("draw.panel", "Control panel"),
            ("draw.legend", "Legend"),
            ("draw.flip", "Present / flip"),
            ("sense.ui", "Sensors (UI)"),
            ("events", "Events"),
            ("other", "Other / idle"),
        ]
        ema = dict(self._perf_ema)
        known = sum(float(ema.get(k, 0.0)) for k, _ in order if k != "other")
        ema["other"] = max(0.0, frame_ms - known)

        rows: list[tuple[str, float]] = []
        for key, label in order:
            ms = float(ema.get(key, 0.0))
            if ms < 0.05 and key != "other":
                continue
            rows.append((label, ms))
        rows.sort(key=lambda r: -r[1])
        if not rows:
            rows = [("(no samples yet)", 0.0)]

        cache = "hit" if self._tissue_cache_hit else "miss"
        gl = "on" if getattr(self, "use_gl_tissue", False) else "off"
        sim_hz = float(getattr(self, "adult_sim_hz", 20.0) or 20.0)
        learn_n = int(getattr(self, "learn_every", 8) or 8)
        hops = int(getattr(self, "loop_hops", 1) or 1)
        meta = [
            f"frame {frame_ms:.1f} ms  ·  {fps:.0f} FPS",
            f"tissue cache {cache}  ·  GL {gl}",
            f"sim {sim_hz:.0f} Hz  ·  hop×{hops}  ·  learn/{learn_n}",
        ]

        row_h = self.hint_font.get_height() + 2
        meta_h = len(meta) * row_h + 6
        bar_max_w = 110
        label_w = 118
        ms_w = 52
        content_w = label_w + bar_max_w + ms_w + 24
        title = self.small_font.render("Frame brakes", True, (210, 220, 240))
        content_w = max(content_w, title.get_width() + 16)
        for line in meta:
            content_w = max(content_w, self.hint_font.size(line)[0] + 16)
        panel_h = 10 + title.get_height() + 6 + meta_h + len(rows) * row_h + 10
        panel_w = content_w + 12

        anchor = self._fps_badge_rect
        if anchor is None:
            return
        px = int(anchor.x)
        max_right = (
            self.dock_x - 8 if int(getattr(self, "dock_w", 0) or 0) > 0
            else self.width - 8
        )
        if px + panel_w > max_right:
            px = max(8, max_right - panel_w)
        py = int(self.top_h + 4)
        max_bottom = self.view_y + self.view_h - 8
        if py + panel_h > max_bottom:
            panel_h = max(40, max_bottom - py)

        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (14, 16, 22, 230), surf.get_rect(), border_radius=4)
        pygame.draw.rect(
            surf, (100, 120, 150, 200), surf.get_rect(), width=1, border_radius=4
        )
        y = 8
        surf.blit(title, (8, y))
        y += title.get_height() + 4
        for line in meta:
            surf.blit(self.hint_font.render(line, True, (140, 150, 170)), (8, y))
            y += row_h
        y += 4
        pygame.draw.line(surf, (50, 55, 70), (8, y), (panel_w - 8, y), 1)
        y += 6
        denom = max(frame_ms, 1e-3)
        for label, ms in rows:
            if y + row_h > panel_h - 6:
                break
            frac = float(np.clip(ms / denom, 0.0, 1.0))
            if frac > 0.35:
                col = (220, 120, 90)
            elif frac > 0.18:
                col = (210, 180, 100)
            else:
                col = (100, 170, 140)
            name = self.hint_font.render(label, True, (190, 200, 215))
            surf.blit(name, (8, y))
            bx0 = 8 + label_w
            by0 = y + 3
            pygame.draw.rect(
                surf, (40, 44, 56), (bx0, by0, bar_max_w, row_h - 6), border_radius=2
            )
            bw = max(1, int(bar_max_w * frac)) if ms > 0 else 0
            if bw > 0:
                pygame.draw.rect(
                    surf, col, (bx0, by0, bw, row_h - 6), border_radius=2
                )
            ms_lab = self.hint_font.render(f"{ms:.1f}", True, (200, 210, 225))
            surf.blit(ms_lab, (bx0 + bar_max_w + 6, y))
            y += row_h

        self.screen.blit(surf, (px, py))
        self._fps_panel_rect = pygame.Rect(px, py, panel_w, panel_h)

    def _perf_absorb_sim(self, info: dict | None) -> None:
        """Fold ClosedLoop timing fields into this frame's perf dict."""
        if not info:
            return
        mapping = (
            ("t_sense_ms", "sim.sense"),
            ("t_hop_ms", "sim.hop"),
            ("t_actuate_ms", "sim.actuate"),
            ("t_learn_ms", "sim.learn"),
        )
        for src, dst in mapping:
            if src in info:
                self._perf_ms[dst] = float(info[src])

    def _perf_commit_frame(self) -> None:
        """EMA-smooth raw marks for the FPS panel."""
        for k, v in self._perf_ms.items():
            prev = float(self._perf_ema.get(k, v))
            self._perf_ema[k] = 0.82 * prev + 0.18 * float(v)
        # Keep stale keys from dominating forever when a path is skipped.
        for k in list(self._perf_ema.keys()):
            if k not in self._perf_ms:
                self._perf_ema[k] = float(self._perf_ema[k]) * 0.92

    def _body_orbit_frame(self) -> tuple[np.ndarray, float] | None:
        """World center + radius beyond the farthest soma (tail included)."""
        cloud = self.points_xyz
        if cloud is None or len(cloud) == 0:
            return None
        key = (id(cloud), int(len(cloud)))
        if key == self._orbit_cache_key and self._orbit_cache is not None:
            return self._orbit_cache
        pts = cloud.astype(np.float64)
        center = pts.mean(axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        # Cover outliers (VNC tip / "ass") — not just axis-aligned half-span.
        far = float(np.quantile(dist, 0.995)) if len(dist) > 20 else float(dist.max())
        far = max(far, float(dist.max()) * 0.98, 1e-3)
        out = (center, far * float(COMPASS_RING_FRAC))
        self._orbit_cache_key = key
        self._orbit_cache = out
        return out

    def _begin_frame_caches(self) -> None:
        """Drop per-frame pose memos (anchors/orbit stay until soma cloud changes)."""
        self._frame_ocelli_pose = None
        self._frame_turret_pose = None
        self._frame_eye_pose = {}
        self._frame_jo_pose = {}
        self._frame_pose_stamp = None

    def _overlay_surface(self, view_lim_y: int) -> pygame.Surface:
        """Reusable full-view SRCALPHA buffer (cleared each use)."""
        need = (int(self.view_w), int(view_lim_y))
        buf = self._overlay_buf
        if buf is None or buf.get_size() != need:
            buf = pygame.Surface(need, pygame.SRCALPHA)
            self._overlay_buf = buf
        else:
            buf.fill((0, 0, 0, 0))
        return buf

    def _draw_orbit_compass(
        self, cx: float, cy: float, view_lim_y: int
    ) -> None:
        """Compass / gyro rings around the fly (Matrix world-fixed, outside the body).

        Cyan horizon: N=anterior (−forward), E=right, S=posterior, W=left.
        Amber sagittal: UP=world up, DN=world down.
        Magenta coronal: roll about the body axis.
        """
        assert self._view_surf is not None
        frame = self._body_orbit_frame()
        if frame is None:
            return
        tilt_q = round(float(self._world_swing_angle()), 4)
        cam_key = (
            round(float(self.view_angle_x), 4),
            round(float(self.view_angle_y), 4),
            round(float(self.scale_3d), 3),
            round(float(self.pan_x), 1),
            round(float(self.pan_y), 1),
            int(self.view_w),
            int(view_lim_y),
            id(self.points_xyz),
            int(len(self.points_xyz)),
            tilt_q,
        )
        if (
            cam_key == self._compass_blit_key
            and self._compass_blit_surf is not None
            and self._compass_blit_surf.get_size() == (self.view_w, view_lim_y)
        ):
            self._view_surf.blit(self._compass_blit_surf, (0, 0))
            return
        center, radius = frame
        overlay = pygame.Surface((self.view_w, view_lim_y), pygame.SRCALPHA)

        def _project(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            sx, sy = project_3d_batch(
                xyz,
                self.view_angle_x,
                self.view_angle_y,
                cx,
                cy,
                self.scale_3d,
            )
            ax = float(self.view_angle_x)
            ay = float(self.view_angle_y)
            z_rot = xyz[:, 0] * math.sin(ay) + xyz[:, 2] * math.cos(ay)
            y_rot = xyz[:, 1] * math.cos(ax) - z_rot * math.sin(ax)
            return sx, sy, y_rot

        def _draw_ring(
            u: np.ndarray,
            v: np.ndarray,
            col: tuple[int, int, int, int],
            *,
            width: int = 2,
        ) -> None:
            ang = np.linspace(0.0, math.tau, COMPASS_SEGS, endpoint=False)
            xyz = (
                center[None, :]
                + radius * np.cos(ang)[:, None] * u[None, :]
                + radius * np.sin(ang)[:, None] * v[None, :]
            ).astype(np.float32)
            xyz = np.concatenate([xyz, xyz[:1]], axis=0)
            sx, sy, depth = _project(xyz)
            order = np.argsort(0.5 * (depth[:-1] + depth[1:]))
            for i in order.tolist():
                x0, y0 = int(sx[i]), int(sy[i])
                x1, y1 = int(sx[i + 1]), int(sy[i + 1])
                if (
                    (0 <= x0 < self.view_w and 0 <= y0 < view_lim_y)
                    or (0 <= x1 < self.view_w and 0 <= y1 < view_lim_y)
                ):
                    pygame.draw.line(overlay, col, (x0, y0), (x1, y1), width)

        def _mark_ring(
            u: np.ndarray,
            v: np.ndarray,
            *,
            zero_dir: np.ndarray,
            tick_rgb: tuple[int, int, int],
            cardinals: dict[int, str],
        ) -> None:
            """Degree ticks + labels on one ring. 0° = zero_dir in the u–v plane."""
            u = u / max(float(np.linalg.norm(u)), 1e-8)
            v = v / max(float(np.linalg.norm(v)), 1e-8)
            z = zero_dir.astype(np.float64)
            cu = float(np.dot(z, u))
            cv = float(np.dot(z, v))
            n2 = cu * cu + cv * cv
            if n2 < 1e-12:
                cu, cv = 1.0, 0.0
                n2 = 1.0
            inv = 1.0 / math.sqrt(n2)
            cu, cv = cu * inv, cv * inv
            # 0° and 90° directions in-plane.
            e0 = cu * u + cv * v
            e90 = -cv * u + cu * v

            tick_a: list[np.ndarray] = []
            tick_b: list[np.ndarray] = []
            tick_w: list[int] = []
            labels: list[tuple[np.ndarray, str]] = []
            for deg in range(0, 360, COMPASS_TICK_MINOR):
                th = math.radians(float(deg))
                dvec = math.cos(th) * e0 + math.sin(th) * e90
                outer = center + radius * dvec
                major = deg % COMPASS_TICK_MAJOR == 0
                inner = center + radius * (0.93 if major else 0.965) * dvec
                tick_a.append(inner.astype(np.float32))
                tick_b.append(outer.astype(np.float32))
                tick_w.append(2 if major else 1)
                if major:
                    lab = center + radius * 1.10 * dvec
                    name = cardinals.get(deg)
                    text = f"{name} {deg}°" if name else f"{deg}°"
                    labels.append((lab.astype(np.float32), text))

            if tick_a:
                a = np.stack(tick_a, axis=0)
                b = np.stack(tick_b, axis=0)
                sa, ta, _ = _project(a)
                sb, tb, _ = _project(b)
                tick_col = (*tick_rgb, int(COMPASS_TICK_ALPHA))
                for i in range(len(a)):
                    pygame.draw.line(
                        overlay,
                        tick_col,
                        (int(sa[i]), int(ta[i])),
                        (int(sb[i]), int(tb[i])),
                        tick_w[i],
                    )
            for pos, text in labels:
                sx, sy, _ = _project(pos.reshape(1, 3))
                x, y = int(sx[0]), int(sy[0])
                if not (0 <= x < self.view_w and 0 <= y < view_lim_y):
                    continue
                surf = self.hint_font.render(text, True, tick_rgb)
                surf.set_alpha(int(COMPASS_LABEL_ALPHA))
                sh = self.hint_font.render(text, True, (8, 10, 14))
                sh.set_alpha(int(COMPASS_LABEL_ALPHA * 0.55))
                overlay.blit(
                    sh, (x - surf.get_width() // 2 + 1, y - surf.get_height() // 2 + 1)
                )
                overlay.blit(
                    surf, (x - surf.get_width() // 2, y - surf.get_height() // 2)
                )

        up, right, forward = self._world_frame()

        _draw_ring(right, forward, COMPASS_COL_HZ, width=1)
        _draw_ring(up, forward, COMPASS_COL_MER, width=1)
        _draw_ring(right, up, COMPASS_COL_COR, width=1)

        # Horizon: 0° = anterior (−forward).
        _mark_ring(
            right,
            forward,
            zero_dir=-forward,
            tick_rgb=COMPASS_COL_HZ_RGB,
            cardinals={0: "N", 90: "E", 180: "S", 270: "W"},
        )
        # Sagittal: 0° = Matrix up.
        _mark_ring(
            up,
            forward,
            zero_dir=up,
            tick_rgb=COMPASS_COL_MER_RGB,
            cardinals={0: "UP", 90: "POST", 180: "DN", 270: "ANT"},
        )
        # Coronal: 0° = Matrix up, 90° = right.
        _mark_ring(
            right,
            up,
            zero_dir=up,
            tick_rgb=COMPASS_COL_COR_RGB,
            cardinals={0: "UP", 90: "R", 180: "DN", 270: "L"},
        )

        self._view_surf.blit(overlay, (0, 0))
        self._compass_blit_key = cam_key
        self._compass_blit_surf = overlay

    def _draw_gravity_pendulum(
        self, cx: float, cy: float, view_lim_y: int
    ) -> None:
        """Arrow along Matrix gravity (down) — fly fixed, world swinging."""
        assert self._view_surf is not None
        frame = self._body_orbit_frame()
        if frame is None:
            return
        center, radius = frame
        center = np.asarray(center, dtype=np.float64)
        down = self._world_down()
        length = float(radius) * float(GRAV_ARROW_FRAC)
        tip = center + down * length
        # Shaft samples for depth-sorted segments.
        t = np.linspace(0.08, 1.0, 12, dtype=np.float64)
        shaft = (center[None, :] + t[:, None] * (tip - center)[None, :]).astype(
            np.float32
        )
        # Arrowhead in the sagittal plane (down × right).
        _, right, _ = self._world_frame()
        head_len = 0.14 * length
        head_w = 0.07 * length
        back = tip - down * head_len
        wing_a = back + right * head_w
        wing_b = back - right * head_w
        head = np.stack([tip, wing_a, tip, wing_b], axis=0).astype(np.float32)

        overlay = self._overlay_surface(view_lim_y)
        all_pts = np.concatenate([shaft, head], axis=0)
        sx, sy = project_3d_batch(
            all_pts,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        a = int(GRAV_ARROW_ALPHA)
        col = (*GRAV_ARROW_RGB, a)
        for i in range(len(shaft) - 1):
            x0, y0 = int(sx[i]), int(sy[i])
            x1, y1 = int(sx[i + 1]), int(sy[i + 1])
            if (
                (0 <= x0 < self.view_w and 0 <= y0 < view_lim_y)
                or (0 <= x1 < self.view_w and 0 <= y1 < view_lim_y)
            ):
                pygame.draw.line(overlay, col, (x0, y0), (x1, y1), 3)
        n_s = len(shaft)
        # Head: tip→wing_a, tip→wing_b
        for aa, bb in ((n_s, n_s + 1), (n_s + 2, n_s + 3)):
            x0, y0 = int(sx[aa]), int(sy[aa])
            x1, y1 = int(sx[bb]), int(sy[bb])
            if (
                (0 <= x0 < self.view_w and 0 <= y0 < view_lim_y)
                or (0 <= x1 < self.view_w and 0 <= y1 < view_lim_y)
            ):
                pygame.draw.line(overlay, col, (x0, y0), (x1, y1), 3)
        # Fill head triangle
        tri = [
            (int(sx[n_s]), int(sy[n_s])),
            (int(sx[n_s + 1]), int(sy[n_s + 1])),
            (int(sx[n_s + 3]), int(sy[n_s + 3])),
        ]
        pygame.draw.polygon(
            overlay, (*GRAV_ARROW_RGB, int(GRAV_ARROW_HEAD_ALPHA)), tri
        )
        # Pivot hub
        hx, hy = project_3d_batch(
            center.reshape(1, 3).astype(np.float32),
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        hx_i, hy_i = int(hx[0]), int(hy[0])
        if 0 <= hx_i < self.view_w and 0 <= hy_i < view_lim_y:
            pygame.draw.circle(overlay, (*GRAV_ARROW_RGB, a), (hx_i, hy_i), 4)

        # Deviation badge at the tip (° from anatomical down).
        tip_x, tip_y = int(sx[n_s]), int(sy[n_s])
        deg = math.degrees(float(self._world_swing_angle()))
        badge_txt = f"{deg:+.0f}°"
        lab = self.hint_font.render(badge_txt, True, GRAV_BADGE_FG)
        pad_x, pad_y = 5, 2
        bw = lab.get_width() + pad_x * 2
        bh = lab.get_height() + pad_y * 2
        # Offset past tip along projected shaft direction when possible.
        if len(shaft) >= 2:
            dx = float(sx[n_s - 1] - sx[n_s - 2])
            dy = float(sy[n_s - 1] - sy[n_s - 2])
            dn = math.hypot(dx, dy)
            if dn > 1e-3:
                ux, uy = dx / dn, dy / dn
            else:
                ux, uy = 0.0, 1.0
        else:
            ux, uy = 0.0, 1.0
        bx = int(tip_x + ux * 10 - bw // 2)
        by = int(tip_y + uy * 10 - bh // 2)
        bx = int(np.clip(bx, 2, max(2, self.view_w - bw - 2)))
        by = int(np.clip(by, 2, max(2, view_lim_y - bh - 2)))
        badge = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(badge, GRAV_BADGE_BG, badge.get_rect(), border_radius=3)
        pygame.draw.rect(
            badge,
            (*GRAV_ARROW_RGB, 120),
            badge.get_rect(),
            width=1,
            border_radius=3,
        )
        badge.blit(lab, (pad_x, pad_y))
        overlay.blit(badge, (bx, by))

        # Swing arc ticks when active (amplitude envelope).
        if getattr(self, "world_swing_on", False) and float(self.world_swing_amp) > 1e-4:
            amp = float(self.world_swing_amp)
            arc_r = 0.42 * length
            angs = np.linspace(-amp, amp, 17, dtype=np.float64)
            arc_pts = []
            for ang in angs:
                ct, st = math.cos(ang), math.sin(ang)
                # Rest down = −Y; swing about +X → down(a) = (0, −ct, −st)
                d = np.array([0.0, -ct, -st], dtype=np.float64)
                arc_pts.append(center + d * arc_r)
            arc = np.stack(arc_pts, axis=0).astype(np.float32)
            ax, ay = project_3d_batch(
                arc,
                self.view_angle_x,
                self.view_angle_y,
                cx,
                cy,
                self.scale_3d,
            )
            arc_col = (*GRAV_ARROW_RGB, 70)
            for i in range(len(arc) - 1):
                pygame.draw.line(
                    overlay,
                    arc_col,
                    (int(ax[i]), int(ay[i])),
                    (int(ax[i + 1]), int(ay[i + 1])),
                    1,
                )
        self._view_surf.blit(overlay, (0, 0))

    def _world_gravity(self) -> np.ndarray:
        """Anatomical down for tissue gravity wells (−Y). Funnels stay body-fixed."""
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)

    def _device_funnel_centers(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Raw device anchors: ocelli lenses mean + turret pad centroid."""
        if not (self.ocelli_on and self.turret_on):
            return None
        oc = self._ocelli_rig_pose()
        tu = self._turret_mech_pose()
        if oc is None or tu is None:
            return None
        lenses = np.stack([np.asarray(o["lens"], dtype=np.float64) for o in oc], axis=0)
        c_oc = lenses.mean(axis=0)
        # Seat turret well on the pads (tissue), not a floating pedestal.
        sites = np.asarray(tu["sites"], dtype=np.float64)
        c_tu = sites.mean(axis=0) if len(sites) else np.asarray(tu["mount"], dtype=np.float64)
        return c_oc, c_tu

    def _body_span(self) -> float:
        frame = self._body_orbit_frame()
        if frame is not None:
            return max(float(frame[1]), 1e-3)
        if len(self.fly.soma_xyz) == 0:
            return 1.0
        return max(float(np.ptp(self.fly.soma_xyz, axis=0).max()), 1e-3)

    def _tissue_centroid_near(
        self, anchor: np.ndarray, seed_r: float
    ) -> np.ndarray:
        """Mean of somata near ``anchor`` — seat the well inside real tissue."""
        a = np.asarray(anchor, dtype=np.float64)
        if len(self.fly.soma_xyz) == 0 or seed_r <= 0:
            return a
        xyz = np.asarray(self.fly.soma_xyz, dtype=np.float64)
        d = np.linalg.norm(xyz - a[None, :], axis=1)
        near = d <= float(seed_r)
        n = int(np.count_nonzero(near))
        if n < 12:
            # Fall back to K nearest.
            k = min(int(DEVICE_FUNNEL_LOCAL_K), len(d))
            if k < 8:
                return a
            idx = np.argpartition(d, k - 1)[:k]
            return xyz[idx].mean(axis=0)
        return xyz[near].mean(axis=0)

    def _local_well_radius(self, center: np.ndarray, gap: float) -> float:
        """Seed rim radius from local soma cloud (before grow-toward-partner)."""
        span = self._body_span()
        r_min = float(DEVICE_FUNNEL_R_MIN_FRAC) * span
        r_max = min(
            float(DEVICE_FUNNEL_R_MAX_FRAC) * span,
            float(DEVICE_FUNNEL_R_MAX_GAP) * max(gap, 1e-6),
        )
        r_max = max(r_max, r_min)
        if len(self.fly.soma_xyz) == 0:
            return 0.5 * (r_min + r_max)
        xyz = np.asarray(self.fly.soma_xyz, dtype=np.float64)
        d = np.linalg.norm(xyz - np.asarray(center, dtype=np.float64)[None, :], axis=1)
        k = min(int(DEVICE_FUNNEL_LOCAL_K), len(d))
        if k < 8:
            return 0.5 * (r_min + r_max)
        nearest = np.partition(d, k - 1)[:k]
        r = float(np.quantile(nearest, float(DEVICE_FUNNEL_LOCAL_Q)))
        return float(np.clip(r, r_min, r_max))

    def _well_radius_cap(self, gap: float) -> float:
        span = self._body_span()
        return float(
            max(
                float(DEVICE_FUNNEL_R_MIN_FRAC) * span,
                min(
                    float(DEVICE_FUNNEL_R_MAX_FRAC) * span,
                    float(DEVICE_FUNNEL_R_MAX_GAP) * max(gap, 1e-6),
                ),
            )
        )

    def _neurons_in_ball(self, center: np.ndarray, radius: float) -> set[int]:
        """Soma ids inside a ball (well volume for funnel sets)."""
        if len(self.fly.soma_xyz) == 0 or radius <= 0:
            return set()
        xyz = np.asarray(self.fly.soma_xyz, dtype=np.float64)
        ids = np.asarray(self.fly.soma_ids, dtype=np.int64)
        c = np.asarray(center, dtype=np.float64)
        d = np.linalg.norm(xyz - c[None, :], axis=1)
        return set(int(i) for i in ids[d <= float(radius)].tolist())

    def _neck_class_ids_in(self, ids: set[int]) -> set[int]:
        """Ascending/descending members of ``ids``."""
        if not ids or len(self.points_cls) == 0:
            return set()
        want = {
            i
            for i, n in enumerate(self.class_names)
            if n in FUSE_NECK_CLASSES
        }
        if not want:
            return set()
        lut = self._soma_id_lut()
        out: set[int] = set()
        cls = np.asarray(self.points_cls, dtype=np.int32)
        for nid in ids:
            k = lut.get(int(nid))
            if k is None:
                continue
            if int(cls[k]) in want:
                out.add(int(nid))
        return out

    def _funnel_bridge_path_count(
        self, a_set: set[int], n_set: set[int], b_set: set[int]
    ) -> int:
        """How many ∩ neurons lie on some atlas∪Fuse path sources→sinks."""
        if not n_set:
            return 0
        sources = set(self._ocelli_flat_ids()) & (a_set | n_set)
        sinks = set(self._turret_flat_ids()) & (b_set | n_set)
        if not sources or not sinks:
            sources = set(a_set) - set(n_set) or set(a_set)
            sinks = set(b_set) - set(n_set) or set(b_set)
        if not sources or not sinks:
            return 0
        allow = a_set | b_set | n_set
        # Probe on unclipped hop graph (scope may already clip hop_allow).
        saved = np.asarray(
            getattr(self.fly, "hop_allow_ids", np.zeros((0,), dtype=np.int32)),
            dtype=np.int32,
        ).copy()
        try:
            self.fly.hop_allow_ids = np.zeros((0,), dtype=np.int32)
            self.fly._hop_recipe_key = None
            pre, post = self.fly.activity_graph_edges()
        finally:
            self.fly.hop_allow_ids = saved
            self.fly._hop_recipe_key = None
        if len(pre) == 0:
            return 0
        adj: dict[int, list[int]] = {}
        rev: dict[int, list[int]] = {}
        for p, q in zip(pre.tolist(), post.tolist()):
            pi, qi = int(p), int(q)
            if pi not in allow or qi not in allow:
                continue
            adj.setdefault(pi, []).append(qi)
            rev.setdefault(qi, []).append(pi)

        def _bfs(starts: set[int], graph: dict[int, list[int]]) -> set[int]:
            seen = set(starts)
            stack = list(starts)
            while stack:
                u = stack.pop()
                for v in graph.get(u, ()):
                    if v in seen or v not in allow:
                        continue
                    seen.add(v)
                    stack.append(v)
            return seen

        on_path = n_set & _bfs(sources, adj) & _bfs(sinks, rev)
        return int(len(on_path))


    def _well_grow_params(self) -> WellGrowParams:
        wg = getattr(self, "well_grow", None)
        if wg is not None:
            return wg
        return WellGrowParams.from_funnel_gate(
            getattr(self, "funnel_gate", None) or _DEFAULT_GATE
        )

    def _funnel_sets_for_radii(
        self, c_oc: np.ndarray, c_tu: np.ndarray, r_oc: float, r_tu: float
    ) -> dict[str, set[int]]:
        """Geometric A/B/∩ for wells only — never writes ``_training_scope``."""
        from wells import WellSpec, neurons_in_ball, DEFAULT_COLORS

        soma_xyz = np.asarray(self.fly.soma_xyz, dtype=np.float64)
        soma_ids = np.asarray(self.fly.soma_ids, dtype=np.int64)
        ids_a = neurons_in_ball(soma_xyz, soma_ids, c_oc, r_oc) | set(
            self._ocelli_flat_ids()
        )
        ids_b = neurons_in_ball(soma_xyz, soma_ids, c_tu, r_tu) | set(
            self._turret_flat_ids()
        )
        wf = getattr(self, "_well_field", None)
        if wf is None:
            wf = WellField()
            self._well_field = wf
        wf.specs = [
            WellSpec(
                name="ocelli",
                center=c_oc,
                radius=float(r_oc),
                rgb=DEFAULT_COLORS["ocelli"],
                neuron_ids=ids_a,
            ),
            WellSpec(
                name="turret",
                center=c_tu,
                radius=float(r_tu),
                rgb=DEFAULT_COLORS["turret"],
                neuron_ids=ids_b,
            ),
        ]
        # Probe cache for grow loop / diagnostics — not TrainingScope.
        layers = wf.as_layers()
        self._funnel_probe_layers = layers
        return layers

    def _device_funnel_pair(self) -> list[dict] | None:
        """Seat wells in tissue; grow until WellGrowParams size stop.

        Never assigns ``_training_scope``. Drawn funnel uses same centre/r.
        """
        pair = self._device_funnel_centers()
        if pair is None:
            wf = getattr(self, "_well_field", None)
            if wf is not None:
                wf.clear()
            self._funnel_pair_cache = None
            self._funnel_pair_cache_key = None
            self._funnel_probe_layers = None
            return None
        params = self._well_grow_params()
        # Sync module grow knobs into params (shell constants stay source for UI).
        params = WellGrowParams(
            min_a=int(params.min_a),
            min_b=int(params.min_b),
            min_n=int(params.min_n),
            min_neck_class=int(params.min_neck_class),
            local_k=int(DEVICE_FUNNEL_LOCAL_K),
            local_q=float(DEVICE_FUNNEL_LOCAL_Q),
            r_min_frac=float(DEVICE_FUNNEL_R_MIN_FRAC),
            r_max_frac=float(DEVICE_FUNNEL_R_MAX_FRAC),
            r_max_gap=float(DEVICE_FUNNEL_R_MAX_GAP),
            grow_steps=int(DEVICE_FUNNEL_GROW_STEPS),
            grow_frac=float(DEVICE_FUNNEL_GROW_FRAC),
            rising_bucket=int(params.rising_bucket),
        )
        self.well_grow = params
        wf = getattr(self, "_well_field", None)
        if wf is None:
            wf = WellField()
            self._well_field = wf
        key = WellField.stable_cache_key(self, pair, params)
        if key == wf.cache_key and wf.specs:
            out = [s.as_draw_dict() for s in wf.specs]
            self._funnel_pair_cache_key = key
            self._funnel_pair_cache = out
            self._funnel_probe_layers = wf.as_layers()
            if wf.specs:
                self._funnel_probe_radii = (
                    float(wf.specs[0].radius),
                    float(wf.specs[1].radius),
                )
            return list(out)
        specs = wf.ensure_pair(
            self,
            pair,
            params=params,
            span=float(self._body_span()),
            extra_ids={
                "ocelli": set(self._ocelli_flat_ids()),
                "turret": set(self._turret_flat_ids()),
            },
            neck_class_count=lambda n: len(self._neck_class_ids_in(n)),
        )
        if not specs:
            self._funnel_pair_cache = None
            self._funnel_pair_cache_key = None
            self._funnel_probe_layers = None
            return None
        self._funnel_probe_layers = wf.as_layers()
        self._funnel_probe_radii = (
            float(specs[0].radius),
            float(specs[1].radius),
        )
        out = [s.as_draw_dict() for s in specs]
        self._funnel_pair_cache_key = key
        self._funnel_pair_cache = out
        return list(out)

    def _funnel_cone_frame(
        self, center: np.ndarray, radius: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Rim centre, tip, depth — same geometry as ``_draw_gravity_well_3d``."""
        g = self._world_gravity()
        r = float(radius)
        depth = r * float(DEVICE_FUNNEL_DEPTH)
        tip = np.asarray(center, dtype=np.float64) + g * (0.15 * depth)
        rim_c = np.asarray(center, dtype=np.float64) - g * depth
        return rim_c, tip, depth

    def _neurons_in_funnel_cone(
        self, center: np.ndarray, radius: float
    ) -> set[int]:
        """Soma ids inside the drawn funnel cone (debug / legacy)."""
        if len(self.fly.soma_xyz) == 0 or radius <= 0:
            return set()
        rim_c, tip, _depth = self._funnel_cone_frame(center, radius)
        xyz = np.asarray(self.fly.soma_xyz, dtype=np.float64)
        ids = np.asarray(self.fly.soma_ids, dtype=np.int64)
        axis = tip - rim_c
        h = float(np.linalg.norm(axis))
        if h < 1e-9:
            return set()
        axis_u = axis / h
        rel = xyz - rim_c[None, :]
        t = np.dot(rel, axis_u) / h
        radial = np.linalg.norm(rel - t[:, None] * axis[None, :], axis=1)
        t_c = np.clip(t, 0.0, 1.0)
        rr = float(radius) * (1.0 - 0.82 * np.power(t_c, 0.85))
        rr = np.maximum(rr, 0.02 * float(radius))
        slack = float(DEVICE_FUNNEL_SLACK)
        inside = (t >= -0.02) & (t <= 1.05) & (radial <= rr * slack)
        return set(int(i) for i in ids[inside].tolist())

    def _funnel_layer_sets(self) -> dict[str, set[int]] | None:
        """Well A/B/∩ from WellField (render diagnostics — not graph neck)."""
        specs = self._device_funnel_pair()
        if not specs or len(specs) < 2:
            return None
        wf = getattr(self, "_well_field", None)
        if wf is not None and wf.specs:
            return wf.as_layers()
        probed = getattr(self, "_funnel_probe_layers", None)
        if isinstance(probed, dict) and "inter" in probed:
            return probed
        return None

    def _sync_device_funnel_gravity(self) -> None:
        """Push visible wells into tip attractors (throttled via WellField key)."""
        fly = getattr(self, "fly", None)
        if fly is None:
            return
        show_oc = bool(getattr(self, "show_ocelli_funnel", False))
        show_tu = bool(getattr(self, "show_turret_funnel", False))
        if not (show_oc or show_tu):
            wf = getattr(self, "_well_field", None)
            if wf is not None:
                wf.sync_gravity(fly, visible_names=set(), gravity=0.0)
            else:
                fly.device_funnel_xyz = np.zeros((0, 3), dtype=np.float32)
                fly.device_funnel_r = np.zeros((0,), dtype=np.float32)
                fly.device_funnel_w = 0.0
            return
        # Ensure wells exist (cached).
        self._device_funnel_pair()
        wf = getattr(self, "_well_field", None)
        if wf is None:
            return
        names: set[str] = set()
        if show_oc:
            names.add("ocelli")
        if show_tu:
            names.add("turret")
        wf.sync_gravity(
            fly, visible_names=names, gravity=float(DEVICE_FUNNEL_GRAVITY)
        )

    def _device_funnel_visible_specs(self) -> list[dict]:
        """Well draw specs filtered by per-device Funnel checkboxes."""
        show_oc = bool(getattr(self, "show_ocelli_funnel", False))
        show_tu = bool(getattr(self, "show_turret_funnel", False))
        # Skip WellField.rebuild_pair (≈140k ball queries × grow steps) when
        # nothing will be drawn — first _draw was paying ~300ms for hidden funnels.
        if not (show_oc or show_tu):
            return []
        pair = self._device_funnel_pair()
        if not pair:
            return []
        out: list[dict] = []
        for s in pair:
            name = str(s.get("name", ""))
            if name == "ocelli" and show_oc:
                out.append(s)
            elif name == "turret" and show_tu:
                out.append(s)
        return out

    def _corridor_neuron_ids(self) -> set[int]:
        """Graph path bed ∪ endpoints only — never geometric well ∩."""
        scope = getattr(self, "_training_scope", None)
        if (
            scope is not None
            and getattr(scope, "path_neurons", None)
            and (getattr(self, "scope_on", False))
        ):
            corridor = set(scope.corridor)
            self._corridor_ids = corridor
            return set(corridor)
        if getattr(self, "scope_on", False):
            try:
                scope = self.build_training_scope(with_wells=False)
                corridor = set(scope.corridor)
                self._corridor_ids = corridor
                self._training_scope = scope
                return set(corridor)
            except Exception:
                pass
        # Scope off / refused: empty corridor (do not fall back to wells).
        self._corridor_ids = set()
        return set()

    def _scope_path_edge_set(self) -> set[tuple[int, int]]:
        scope = getattr(self, "_training_scope", None)
        if scope is None:
            return set()
        raw = getattr(scope, "path_edges", None) or ()
        out: set[tuple[int, int]] = set()
        for a, b in raw:
            out.add((int(a), int(b)))
        return out

    def _refresh_corridor_topo(self) -> None:
        """Cache teach-corridor path_edges for standing topology overlay."""
        scope_on = bool(
            getattr(self, "scope_on", False)
        )
        scope = getattr(self, "_training_scope", None)
        if not scope_on or scope is None:
            self._corridor_topo_pairs = []
            self._corridor_topo_ids = set()
            self._corridor_topo_key = None
            return
        edges = getattr(scope, "path_edges", None) or ()
        corridor = set(getattr(scope, "corridor", None) or ())
        if not corridor:
            corridor = set(getattr(scope, "path_neurons", None) or ())
        key = (id(scope), len(edges), len(corridor), int(getattr(scope, "best_hops", -1) or -1))
        if key == getattr(self, "_corridor_topo_key", None) and self._corridor_topo_pairs:
            return
        pairs = [(int(a), int(b)) for a, b in edges]
        # Prefer CNS bridges over pure device-local for overlay budget.
        device = set()
        if hasattr(self, "_turret_flat_ids"):
            try:
                device |= set(int(x) for x in self._turret_flat_ids())
            except Exception:
                pass
        if hasattr(self, "_ocelli_flat_ids"):
            try:
                device |= set(int(x) for x in self._ocelli_flat_ids())
            except Exception:
                pass

        def _prio(ab: tuple[int, int]) -> int:
            a, b = ab
            da, db = a in device, b in device
            if da and db:
                return 0
            if da != db:
                return 3
            return 2

        pairs.sort(key=_prio, reverse=True)
        max_e = 900
        if len(pairs) > max_e:
            pairs = pairs[:max_e]
        ids: set[int] = set()
        for a, b in pairs:
            ids.add(a)
            ids.add(b)
        # Fill soma blooms from corridor mass if edges sparse.
        if len(ids) < 120 and corridor:
            for u in list(corridor)[: max(0, 400 - len(ids))]:
                ids.add(int(u))
        self._corridor_topo_pairs = pairs
        self._corridor_topo_ids = ids
        self._corridor_topo_key = key
        self._corridor_ids = set(corridor) if corridor else ids

    def _draw_corridor_topology(self, cx: float, cy: float, view_lim_y: int) -> None:
        """Standing scope bed (path_edges) — reachability map, not carved Δw.

        Super-class gates: neurite bed → A/D; soma beads → N. Empty = hide.
        """
        if self._view_surf is None:
            return
        scope_on = bool(
            getattr(self, "scope_on", False)
        )
        if not scope_on:
            return
        soma_f = set(self.focus_soma_indices)
        neur_f = set(self.focus_axon_indices) | set(self.focus_dendrite_indices)
        if not soma_f and not neur_f:
            return
        self._refresh_corridor_topo()
        pairs = getattr(self, "_corridor_topo_pairs", None) or []
        engaged = set(getattr(self, "_corridor_topo_ids", set()) or ())
        if not pairs and not engaged:
            return
        ids = self.fly.soma_ids
        xyz = self.fly.soma_xyz
        if len(ids) == 0 or len(xyz) != len(ids):
            return
        device = set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
        polys = self._learn_topo_polylines(list(pairs), engaged, device=device)
        overlay = self._overlay_surface(view_lim_y)
        pulse = float(self._pulse())
        # Quiet map when no carve — pulse only if contact_gain is painting teal.
        has_carve = bool(getattr(self.fly, "contact_gain", None))
        if not has_carve:
            pulse = 0.12
        # Indigo bed was flat blue — now class hue (scope reachability).
        n_drawn = 0
        if neur_f:
            for pts3, owner in polys:
                if len(pts3) < 2:
                    continue
                if not self._nid_in_focus(int(owner), neur_f):
                    continue
                if len(pts3) > 28:
                    step = max(1, (len(pts3) + 27) // 28)
                    pts3 = pts3[::step]
                sx, sy = project_3d_batch(
                    pts3, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
                )
                pts = [
                    (int(x), int(y))
                    for x, y in zip(sx, sy)
                    if np.isfinite(x) and np.isfinite(y)
                ]
                if len(pts) < 2:
                    continue
                if not any(0 <= x < self.view_w and 0 <= y < view_lim_y for x, y in pts):
                    continue
                rgb = self._soma_rgb_for_nid(int(owner))
                pygame.draw.lines(
                    overlay,
                    self._rgba_class(rgb, int(55 + 35 * pulse), lift=0.10),
                    False,
                    pts,
                    max(2, int(3 + pulse)),
                )
                pygame.draw.lines(
                    overlay,
                    self._rgba_class(rgb, int(90 + 50 * pulse), lift=0.40),
                    False,
                    pts,
                    max(1, int(1 + pulse)),
                )
                n_drawn += 1
                if n_drawn >= 260:
                    break
        # Soma beads along corridor (covers atlas edges without grown axons).
        bed = self._soma_indices_for_ids(engaged) if soma_f else []
        if bed:
            if len(bed) > 500:
                step = max(1, (len(bed) + 499) // 500)
                bed = bed[::step][:500]
            e_xyz = xyz[np.asarray(bed, dtype=np.int32)]
            e_ids = [int(ids[i]) for i in bed]
            sx, sy = project_3d_batch(
                e_xyz, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
            )
            for nid, x, y in zip(e_ids, sx.tolist(), sy.tolist()):
                if not self._nid_in_focus(int(nid), soma_f):
                    continue
                xi, yi = int(x), int(y)
                if 0 <= xi < self.view_w and 0 <= yi < view_lim_y:
                    rgb = self._soma_rgb_for_nid(int(nid))
                    pygame.draw.circle(
                        overlay,
                        self._rgba_class(rgb, int(70 + 40 * pulse), lift=0.30),
                        (xi, yi),
                        max(2, int(2 + 1.5 * pulse)),
                    )
                    n_drawn += 1
        if n_drawn <= 0:
            return
        n_cor = len(engaged)
        lab = self.hint_font.render(
            f"scope {len(pairs)}e / {n_cor}n",
            True,
            (150, 170, 230),
        )
        overlay.blit(lab, (10, max(8, view_lim_y - lab.get_height() - 48)))
        self._view_surf.blit(overlay, (0, 0))

    def _update_engagement_from_hop(self, info: dict | None = None) -> None:
        """Every hop beat: light hop ∩ scope. No path_edges flood when Δw empty."""
        scope_on = bool(
            getattr(self, "scope_on", False)
            or getattr(self, "motor_socket_on", False)
            or getattr(self, "learn_on", False)
        )
        if not scope_on:
            return
        if not bool(getattr(self, "_activity_running", False)) and not bool(
            getattr(self, "learn_on", False)
        ):
            # Still keep standing bed; decay wave only.
            amp = float(getattr(self, "_learn_flash_amp", 0.0) or 0.0)
            self._learn_flash_amp = amp * 0.94
            return

        path_edges = self._scope_path_edge_set()
        self._refresh_corridor_topo()
        hop_pre = None if info is None else info.get("flash_pre")
        hop_post = None if info is None else info.get("flash_post")
        device = set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
        corridor = set(getattr(self, "_corridor_topo_ids", set()) or ())
        if not corridor:
            corridor = self._corridor_neuron_ids()

        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def _add(a: int, b: int) -> None:
            key = (int(a), int(b))
            if key in seen:
                return
            seen.add(key)
            pairs.append(key)

        # 1) Hop edges on the corridor path graph first (true engaged topology).
        if (
            hop_pre is not None
            and hop_post is not None
            and len(hop_pre) == len(hop_post)
            and len(hop_pre)
        ):
            ranked: list[tuple[int, int, int]] = []
            for a, b in zip(hop_pre.tolist(), hop_post.tolist()):
                a, b = int(a), int(b)
                on_path = (a, b) in path_edges or (b, a) in path_edges
                in_cor = (a in corridor) or (b in corridor) or not corridor
                if not in_cor and not on_path:
                    continue
                score = 0
                if on_path:
                    score += 6
                if a in corridor and b in corridor:
                    score += 3
                if (a in device) != (b in device):
                    score += 2
                if a not in device and b not in device:
                    score += 2
                ranked.append((score, a, b))
            ranked.sort(key=lambda t: -t[0])
            for score, a, b in ranked[:500]:
                if a == b or (a in device and b in device):
                    _add(a, a)
                    _add(b, b)
                elif self._contact_has_topo(a, b):
                    _add(a, b)
                else:
                    _add(a, a)
                    _add(b, b)

        # 2) Mid-corridor fill only when carve is real — empty contact_gain must
        # not look like a living Δw tract (scope bed is the indigo overlay).
        gains = getattr(self.fly, "contact_gain", {}) or {}
        topo_pairs = getattr(self, "_corridor_topo_pairs", None) or []
        if gains and len(pairs) < 80 and topo_pairs:
            for a, b in topo_pairs[:400]:
                if a in device and b in device:
                    continue
                _add(a, a)
                _add(b, b)
                if len(pairs) >= 220:
                    break

        if not pairs:
            self._learn_flash_amp = float(getattr(self, "_learn_flash_amp", 0.0)) * 0.9
            return

        self._learn_flash_pre = np.asarray([p[0] for p in pairs], dtype=np.int32)
        self._learn_flash_post = np.asarray([p[1] for p in pairs], dtype=np.int32)
        self._learn_flash_n = int(sum(1 for a, b in pairs if a != b))
        self._learn_flash_amp = max(
            float(getattr(self, "_learn_flash_amp", 0.0)),
            0.55 if self._learn_flash_n else 0.40,
        )
        self._learn_hot_ids = {int(a) for a, _ in pairs} | {int(b) for _, b in pairs}

    def _refresh_channel_bed(self) -> None:
        """Pick carved русло edges: |g−1| high and touching the corridor."""
        corridor = self._corridor_neuron_ids()
        gains = getattr(self.fly, "contact_gain", {}) or {}
        scored: list[tuple[float, int, int]] = []
        thr = float((getattr(self, "rule", None) or _DEFAULT_RULE).channel_show_dw)
        for (a, b), w in gains.items():
            wf = float(w)
            if not math.isfinite(wf):
                continue
            score = abs(wf - 1.0)
            if score < thr:
                continue
            a, b = int(a), int(b)
            if a not in corridor and b not in corridor:
                continue
            scored.append((score, a, b))
        scored.sort(reverse=True)
        top = scored[: int((getattr(self, "rule", None) or _DEFAULT_RULE).channel_max_edges)]
        self._channel_pairs = [(a, b) for _, a, b in top]
        ids: set[int] = set()
        for a, b in self._channel_pairs:
            ids.add(a)
            ids.add(b)
        self._channel_ids = ids

    def _draw_channel_bed(self, cx: float, cy: float, view_lim_y: int) -> None:
        """Highlight the carved river bed (not the whole funnel flood).

        Super-class gates: neurite bed → A/D; soma beads → N. Empty = hide.
        """
        if self._view_surf is None:
            return
        pairs = getattr(self, "_channel_pairs", None) or []
        if not pairs:
            return
        soma_f = set(self.focus_soma_indices)
        neur_f = set(self.focus_axon_indices) | set(self.focus_dendrite_indices)
        if not soma_f and not neur_f:
            return
        ids = self.fly.soma_ids
        xyz = self.fly.soma_xyz
        if len(ids) == 0:
            return
        engaged = set(getattr(self, "_channel_ids", set()) or ())
        if not engaged:
            for a, b in pairs:
                engaged.add(int(a))
                engaged.add(int(b))
        device = set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
        polys = self._learn_topo_polylines(pairs, engaged, device=device)
        overlay = self._overlay_surface(view_lim_y)
        pulse = float(self._pulse())
        # Carved river — class hue (was flat amber–teal).
        n_drawn = 0
        if neur_f:
            for pts3, owner in polys:
                if len(pts3) < 2:
                    continue
                if not self._nid_in_focus(int(owner), neur_f):
                    continue
                if len(pts3) > 32:
                    step = max(1, (len(pts3) + 31) // 32)
                    pts3 = pts3[::step]
                sx, sy = project_3d_batch(
                    pts3, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
                )
                pts = [
                    (int(x), int(y))
                    for x, y in zip(sx, sy)
                    if np.isfinite(x) and np.isfinite(y)
                ]
                if len(pts) < 2:
                    continue
                if not any(0 <= x < self.view_w and 0 <= y < view_lim_y for x, y in pts):
                    continue
                rgb = self._soma_rgb_for_nid(int(owner))
                pygame.draw.lines(
                    overlay,
                    self._rgba_class(rgb, int(70 + 40 * pulse), lift=0.05),
                    False,
                    pts,
                    max(4, int(5 + 2 * pulse)),
                )
                pygame.draw.lines(
                    overlay,
                    self._rgba_class(rgb, int(160 + 70 * pulse), lift=0.45),
                    False,
                    pts,
                    max(2, int(2 + pulse)),
                )
                n_drawn += 1
        # Soma beads on bed when topo missing for an edge.
        bed = self._soma_indices_for_ids(engaged) if soma_f else []
        if bed and n_drawn < max(8, len(pairs) // 4):
            e_xyz = xyz[np.asarray(bed, dtype=np.int32)]
            e_ids = [int(ids[i]) for i in bed]
            sx, sy = project_3d_batch(
                e_xyz, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
            )
            for nid, x, y in zip(e_ids, sx.tolist(), sy.tolist()):
                if not self._nid_in_focus(int(nid), soma_f):
                    continue
                xi, yi = int(x), int(y)
                if 0 <= xi < self.view_w and 0 <= yi < view_lim_y:
                    rgb = self._soma_rgb_for_nid(int(nid))
                    pygame.draw.circle(
                        overlay,
                        self._rgba_class(rgb, 120, lift=0.35),
                        (xi, yi),
                        max(2, int(2 + pulse)),
                    )
                    n_drawn += 1
        if n_drawn <= 0:
            return
        # Compact HUD.
        n_cor = len(getattr(self, "_corridor_ids", ()) or ())
        lab = self.hint_font.render(
            f"русло {len(pairs)}e / ∩{n_cor}n",
            True,
            (160, 230, 200),
        )
        overlay.blit(lab, (10, max(8, view_lim_y - lab.get_height() - 28)))
        self._view_surf.blit(overlay, (0, 0))

    def _draw_device_funnels(self, cx: float, cy: float, view_lim_y: int) -> None:
        """Draw checked device gravity funnels (per-panel Funnel toggles)."""
        if self._view_surf is None:
            return
        specs = self._device_funnel_visible_specs()
        if not specs:
            return
        # Cache full funnel layer while camera + device pose rest (orbit still
        # invalidates via angles; learning turret motion rebuilds as needed).
        pose_bits: list = []
        for s in specs:
            c = np.asarray(s["center"], dtype=np.float64).reshape(3)
            pose_bits.append(
                (
                    str(s.get("name", "")),
                    round(float(s["r"]), 3),
                    round(float(c[0]), 2),
                    round(float(c[1]), 2),
                    round(float(c[2]), 2),
                    tuple(int(x) for x in s["rgb"]),
                )
            )
        blit_key = (
            round(float(self.view_angle_x), 4),
            round(float(self.view_angle_y), 4),
            round(float(self.scale_3d), 3),
            round(float(self.pan_x), 1),
            round(float(self.pan_y), 1),
            int(self.view_w),
            int(view_lim_y),
            round(float(getattr(self, "turret_yaw", 0.0)), 3),
            round(float(getattr(self, "turret_pitch", 0.0)), 3),
            int(BASIN_RING_STEPS),
            int(BASIN_RING_SEGS),
            tuple(pose_bits),
        )
        if (
            blit_key == self._device_funnel_blit_key
            and self._device_funnel_blit_surf is not None
            and self._device_funnel_blit_surf.get_size() == (self.view_w, view_lim_y)
        ):
            self._view_surf.blit(self._device_funnel_blit_surf, (0, 0))
            return

        g = self._world_gravity()
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, g))) > 0.85:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u = np.cross(g, ref)
        u /= max(float(np.linalg.norm(u)), 1e-8)
        v = np.cross(g, u)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        ax = float(self.view_angle_x)
        ay = float(self.view_angle_y)
        tips = np.stack([s["center"] for s in specs], axis=0)
        z_rot = tips[:, 0] * math.sin(ay) + tips[:, 2] * math.cos(ay)
        y_rot = tips[:, 1] * math.cos(ax) - z_rot * math.sin(ax)
        order = np.argsort(y_rot)

        saved = self._view_surf
        layer = pygame.Surface((self.view_w, view_lim_y), pygame.SRCALPHA)
        layer.fill((0, 0, 0, 0))
        self._view_surf = layer
        try:
            for i in order.tolist():
                s = specs[i]
                self._draw_gravity_well_3d(
                    s["center"],
                    float(s["r"]),
                    g,
                    u,
                    v,
                    tuple(int(c) for c in s["rgb"]),
                    label=str(s["name"]),
                    focused=True,
                    cx=cx,
                    cy=cy,
                    view_lim_y=view_lim_y,
                    alpha_mul=float(DEVICE_FUNNEL_ALPHA),
                    depth_mul=float(DEVICE_FUNNEL_DEPTH),
                )
        finally:
            self._view_surf = saved
        self._device_funnel_blit_key = blit_key
        self._device_funnel_blit_surf = layer
        saved.blit(layer, (0, 0))

    def _draw_basins_layer(
        self,
        cx: float,
        cy: float,
        view_lim_y: int,
        focus: set[int],
    ) -> None:
        """Cosmology-style gravity wells as world-fixed 3D funnels."""
        if self._view_surf is None:
            return
        self._refresh_basins(focus=focus if focus else None)
        if len(self._basin_cls) == 0:
            return
        foc_key = tuple(sorted(int(i) for i in focus)) if focus else ()
        blit_key = (
            round(float(self.view_angle_x), 4),
            round(float(self.view_angle_y), 4),
            round(float(self.scale_3d), 3),
            round(float(self.pan_x), 1),
            round(float(self.pan_y), 1),
            int(self.view_w),
            int(view_lim_y),
            self._basin_cache_key,
            foc_key,
            round(float(self.basin_form), 3),
            len(self._basin_cls),
            int(BASIN_RING_STEPS),
            int(BASIN_RING_SEGS),
        )
        if (
            blit_key == self._basin_blit_key
            and self._basin_blit_surf is not None
            and self._basin_blit_surf.get_size() == (self.view_w, view_lim_y)
        ):
            self._view_surf.blit(self._basin_blit_surf, (0, 0))
            return

        # Draw into an offscreen layer so we can reuse while the camera rests.
        saved = self._view_surf
        layer = pygame.Surface((self.view_w, view_lim_y), pygame.SRCALPHA)
        layer.fill((0, 0, 0, 0))
        self._view_surf = layer
        try:
            g = self._world_gravity()
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, g))) > 0.85:
                ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = np.cross(g, ref)
            u /= max(float(np.linalg.norm(u)), 1e-8)
            v = np.cross(g, u)
            v /= max(float(np.linalg.norm(v)), 1e-8)

            tips = self._basin_xyz.astype(np.float64)
            ax = float(self.view_angle_x)
            ay = float(self.view_angle_y)
            sin_x, cos_x = math.sin(ax), math.cos(ax)
            sin_y, cos_y = math.sin(ay), math.cos(ay)
            z_rot = tips[:, 0] * sin_y + tips[:, 2] * cos_y
            y_rot = tips[:, 1] * cos_x - z_rot * sin_x
            order = np.argsort(y_rot)  # farther first
            vis_scale = float(self._basin_form_knobs()["vis_scale"])

            for i in order.tolist():
                cidx = int(self._basin_cls[i])
                col = self._class_color(cidx)
                focused = bool(focus) and cidx in focus
                label = self._class_label(cidx)
                if len(label) > 14:
                    label = label[:13] + "…"
                r_world = max(
                    float(BASIN_VIS_MIN_PX) / max(float(self.scale_3d), 1e-6),
                    float(self._basin_r[i]) * vis_scale,
                )
                self._draw_gravity_well_3d(
                    tips[i],
                    r_world,
                    g,
                    u,
                    v,
                    col,
                    label=label,
                    focused=focused,
                    cx=cx,
                    cy=cy,
                    view_lim_y=view_lim_y,
                )
        finally:
            self._view_surf = saved
        saved.blit(layer, (0, 0))
        self._basin_blit_key = blit_key
        self._basin_blit_surf = layer

    def _draw_gravity_well_3d(
        self,
        center: np.ndarray,
        r_world: float,
        g: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        col: tuple[int, int, int],
        *,
        label: str = "",
        focused: bool,
        cx: float,
        cy: float,
        view_lim_y: int,
        alpha_mul: float | None = None,
        depth_mul: float | None = None,
    ) -> None:
        """World-fixed funnel: sparse rings + spokes (no wall-quad mesh).

        Shared by device funnels and class basins — keep cheap so Learn/orbit
        stay interactive when wells are visible.
        """
        assert self._view_surf is not None
        steps = max(2, int(BASIN_RING_STEPS))
        n_spokes = max(3, int(BASIN_SPOKES))
        n_ring = max(8, int(BASIN_RING_SEGS))
        depth = r_world * float(1.35 if depth_mul is None else depth_mul)
        boost = 1.25 if focused else 1.0
        a_mul = float(
            BASIN_ALPHA if alpha_mul is None else alpha_mul
        ) * boost

        tip = center + g * (0.15 * depth)
        rim_c = center - g * depth
        ang = np.linspace(0.0, math.tau, n_ring, endpoint=False)
        cos_a = np.cos(ang)[:, None]
        sin_a = np.sin(ang)[:, None]

        rings_xyz: list[np.ndarray] = []
        for s in range(steps):
            t = s / float(max(1, steps - 1))
            rr = r_world * (1.0 - 0.82 * (t ** 0.85))
            c = rim_c + (tip - rim_c) * (t ** 1.1)
            rings_xyz.append(
                (
                    c[None, :]
                    + rr * cos_a * u[None, :]
                    + rr * sin_a * v[None, :]
                ).astype(np.float32)
            )

        tip_xyz = tip.astype(np.float32)[None, :]
        all_xyz = np.concatenate(rings_xyz + [tip_xyz], axis=0)
        sx, sy = project_3d_batch(
            all_xyz,
            self.view_angle_x,
            self.view_angle_y,
            cx,
            cy,
            self.scale_3d,
        )
        rings_sc: list[list[tuple[int, int]]] = []
        off = 0
        for _ in range(steps):
            xs = sx[off : off + n_ring]
            ys = sy[off : off + n_ring]
            rings_sc.append(
                [
                    (int(round(float(x))), int(round(float(y))))
                    for x, y in zip(xs, ys)
                ]
            )
            off += n_ring
        tip_sx = float(sx[-1])
        tip_sy = float(sy[-1])

        xs_all = [p[0] for ring in rings_sc for p in ring] + [int(tip_sx)]
        ys_all = [p[1] for ring in rings_sc for p in ring] + [int(tip_sy)]
        pad = 6
        legend_w = self.hint_font.size(label)[0] + 18 if label else 0
        left = min(xs_all) - pad
        right = max(xs_all) + pad + legend_w
        top = min(ys_all) - pad
        bot = max(ys_all) + pad + (self.hint_font.get_height() if label else 0)
        if right < 0 or bot < 0 or left >= self.view_w or top >= view_lim_y:
            return
        bw = max(8, right - left)
        bh = max(8, bot - top)
        well = pygame.Surface((bw, bh), pygame.SRCALPHA)

        def _local(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
            return [(x - left, y - top) for x, y in pts]

        # Soft fills on rings only (no wall quads).
        for s in range(steps):
            t = s / float(max(1, steps - 1))
            fill_a = int(np.clip((28 + 55 * t) * a_mul, 14, 100))
            shade = 0.55 + 0.40 * (1.0 - t)
            fc = (
                int(np.clip(col[0] * shade, 0, 255)),
                int(np.clip(col[1] * shade, 0, 255)),
                int(np.clip(col[2] * shade, 0, 255)),
                fill_a,
            )
            poly = _local(rings_sc[s])
            if len(poly) >= 3:
                pygame.draw.polygon(well, fc, poly)

        for s in range(steps):
            t = s / float(max(1, steps - 1))
            line_a = int(np.clip((120 + 70 * t) * a_mul, 55, 170))
            poly = _local(rings_sc[s])
            if len(poly) >= 3:
                pygame.draw.polygon(
                    well,
                    (*col, line_a),
                    poly,
                    width=max(1, 2 if (focused or s == 0) else 1),
                )

        for k in range(n_spokes):
            idx = int(round(k * n_ring / n_spokes)) % n_ring
            pts = _local([rings_sc[s][idx] for s in range(steps)])
            # Close spoke to tip.
            pts.append((int(tip_sx) - left, int(tip_sy) - top))
            if len(pts) >= 2:
                pygame.draw.lines(
                    well,
                    (*col, int(np.clip(110 * a_mul, 45, 150))),
                    False,
                    pts,
                    1,
                )

        rim_poly = _local(rings_sc[0])
        if len(rim_poly) >= 3:
            pygame.draw.polygon(
                well,
                (*col, int(np.clip(210 * a_mul, 90, 220))),
                rim_poly,
                width=max(2, 3 if focused else 2),
            )
        core = max(4, int(r_world * float(self.scale_3d) * 0.08))
        tx, ty = int(tip_sx) - left, int(tip_sy) - top
        pygame.draw.circle(
            well,
            (255, 255, 255, int(np.clip(180 * a_mul, 70, 200))),
            (tx, ty),
            core,
        )
        pygame.draw.circle(
            well,
            (*col, int(np.clip(220 * a_mul, 100, 220))),
            (tx, ty),
            max(2, core - 1),
        )

        if label:
            lx = tx + core + 6
            ly = ty - self.hint_font.get_height() // 2
            sw = 7
            pygame.draw.rect(
                well,
                (*col, int(np.clip(220 * a_mul, 120, 255))),
                pygame.Rect(lx, ly + 2, sw, sw),
            )
            pygame.draw.rect(
                well,
                (230, 230, 240, int(np.clip(200 * a_mul, 100, 255))),
                pygame.Rect(lx, ly + 2, sw, sw),
                1,
            )
            shadow = self.hint_font.render(label, True, (10, 10, 16))
            text = self.hint_font.render(label, True, (235, 235, 245))
            shadow.set_alpha(int(np.clip(160 * a_mul, 80, 200)))
            text.set_alpha(int(np.clip(230 * a_mul, 120, 255)))
            well.blit(shadow, (lx + sw + 4 + 1, ly + 1))
            well.blit(text, (lx + sw + 4, ly))

        dst = pygame.Rect(left, top, bw, bh)
        view = pygame.Rect(0, 0, self.view_w, view_lim_y)
        clip = dst.clip(view)
        if clip.width <= 0 or clip.height <= 0:
            return
        src = pygame.Rect(clip.x - left, clip.y - top, clip.width, clip.height)
        self._view_surf.blit(well, clip.topleft, src)

    def _recipe_soma_row_edges(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Map atlas recipe edges → soma-row indices (cached).

        Returns (pre_row, post_row, weight) or None.
        """
        fly = self.fly
        if len(fly.recipe_pre) == 0 or self.n_shown == 0:
            return None
        ids = fly.soma_ids
        key = (
            id(fly.recipe_pre),
            int(len(fly.recipe_pre)),
            id(ids),
            int(len(ids)),
            int(self.n_shown),
        )
        cached = getattr(self, "_recipe_soma_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        contiguous = (
            len(ids) == self.n_shown
            and int(ids[0]) == 0
            and int(ids[-1]) == len(ids) - 1
        )
        if contiguous:
            ra = np.asarray(fly.recipe_pre, dtype=np.int32)
            rb = np.asarray(fly.recipe_post, dtype=np.int32)
        else:
            lut = {int(i): k for k, i in enumerate(ids.tolist())}
            ra = np.fromiter(
                (lut.get(int(x), -1) for x in fly.recipe_pre.tolist()),
                dtype=np.int32,
                count=len(fly.recipe_pre),
            )
            rb = np.fromiter(
                (lut.get(int(x), -1) for x in fly.recipe_post.tolist()),
                dtype=np.int32,
                count=len(fly.recipe_post),
            )
        w = np.asarray(fly.recipe_weight, dtype=np.float32)
        if len(w) != len(ra):
            w = np.ones(len(ra), dtype=np.float32)
        self._recipe_soma_cache = (key, (ra, rb, w))
        return ra, rb, w

    def _draw_tract_sample(
        self, cx: float, cy: float, view_lim_y: int
    ) -> None:
        """Stratified recipe chords — Super-class T only.

        Quiet: equal budget per class (top weight).
        Wave: when T is on, prefer edges whose pre/post soma is firing.
        Empty T → nothing (empty = hide). Scope corridor must not resurrect
        blueprint chords after Reset→rising; scope bed is ``_draw_corridor_topology``.
        """
        if self._view_surf is None or self.n_shown == 0:
            return
        tract_f = self._tract_focus_set()
        if not tract_f:
            return

        n = self.n_shown
        act = getattr(self, "soma_activity", None)
        wave = bool(
            act is not None
            and len(act) == n
            and float(np.max(act)) > 0.05
            and (
                getattr(self, "_activity_running", False)
                or bool(getattr(self, "learn_on", False))
            )
        )

        mapped = self._recipe_soma_row_edges()
        if mapped is None:
            return
        ra, rb, w = mapped
        valid = (ra >= 0) & (rb >= 0) & (ra < n) & (rb < n)
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return

        # Optional: when scope is on, keep T chords inside the teach corridor.
        if bool(getattr(self, "scope_on", False)):
            cor_ids = self._corridor_neuron_ids()
            if not cor_ids:
                cor_ids = set(getattr(self, "_corridor_topo_ids", ()) or ())
            if cor_ids:
                corridor_rows = set(self._soma_indices_for_ids(cor_ids))
                if corridor_rows:
                    cr = np.fromiter(
                        corridor_rows, dtype=np.int32, count=len(corridor_rows)
                    )
                    in_cor = np.isin(ra[idx], cr) & np.isin(rb[idx], cr)
                    idx = idx[in_cor]
                    if len(idx) == 0:
                        return

        cls = self.points_cls
        pre_cls = cls[ra[idx]]
        post_cls = cls[rb[idx]]
        gf = list(tract_f)
        keep = np.isin(pre_cls, gf) | np.isin(post_cls, gf)
        idx = idx[keep]
        if len(idx) == 0:
            return
        pre_cls = cls[ra[idx]]
        post_cls = cls[rb[idx]]

        bucket_cls = np.where(np.isin(pre_cls, list(tract_f)), pre_cls, post_cls)
        classes = sorted({int(c) for c in tract_f if np.any(bucket_cls == int(c))})
        if not classes:
            return

        budget = int(DRAW_TRACT_SAMPLE_MAX)
        # Wide focus: still draw, but share budget across more classes.
        per = max(1, budget // len(classes))
        act_f = (
            np.asarray(act, dtype=np.float32)
            if wave
            else None
        )

        def _edge_score(edge_i: np.ndarray) -> np.ndarray:
            base = w[edge_i].astype(np.float32)
            if act_f is None:
                return base
            ap = act_f[ra[edge_i]]
            bp = act_f[rb[edge_i]]
            live = np.maximum(ap, bp)
            return base * (0.15 + 0.85 * np.clip(live, 0.0, 1.0))

        picked: list[np.ndarray] = []
        for c in classes:
            sub = np.flatnonzero(bucket_cls == int(c))
            if len(sub) == 0:
                continue
            edge_i = idx[sub]
            if len(edge_i) > per:
                score = _edge_score(edge_i)
                if act_f is not None:
                    # Prefer firing edges; fill remainder with weight.
                    live = np.maximum(act_f[ra[edge_i]], act_f[rb[edge_i]])
                    hot = np.flatnonzero(live > 0.05)
                    cold = np.flatnonzero(live <= 0.05)
                    take: list[np.ndarray] = []
                    if len(hot):
                        hh = edge_i[hot]
                        hs = score[hot]
                        n_hot = min(per, len(hh))
                        if len(hh) > n_hot:
                            part = np.argpartition(-hs, n_hot - 1)[:n_hot]
                            hh = hh[part]
                        take.append(hh)
                    need = per - (len(take[0]) if take else 0)
                    if need > 0 and len(cold):
                        cc = edge_i[cold]
                        cs = score[cold]
                        if len(cc) > need:
                            part = np.argpartition(-cs, need - 1)[:need]
                            cc = cc[part]
                        take.append(cc)
                    edge_i = np.concatenate(take) if take else edge_i[:per]
                else:
                    part = np.argpartition(-score, per - 1)[:per]
                    edge_i = edge_i[part]
            picked.append(edge_i)
        if not picked:
            return
        sel = np.concatenate(picked)
        if len(sel) > budget:
            # Keep the hottest when over budget.
            if act_f is not None:
                live = np.maximum(act_f[ra[sel]], act_f[rb[sel]])
                order = np.argsort(-live)[:budget]
                sel = sel[order]
            else:
                sel = sel[:budget]

        a = self.points_xyz[ra[sel]]
        b = self.points_xyz[rb[sel]]
        edge_cls = cls[ra[sel]]
        rgb = _axon_rgb_for_cls(edge_cls).astype(np.float32)
        if act_f is not None:
            e_act = np.maximum(act_f[ra[sel]], act_f[rb[sel]])
            # Wide dynamic range: near-black quiet ↔ punched lit (class hue kept).
            # gamma > 1 pushes mid activity down so only real fire pops.
            live = np.clip(e_act, 0.0, 1.0) ** 1.45
            factor = 0.08 + 1.25 * live
            rgb = rgb * factor[:, None]
            rgb = np.clip(rgb + live[:, None] * 130.0, 0, 255)
        else:
            rgb = np.clip(rgb + 40.0, 0, 255)
        rgb_u = rgb.astype(np.uint8)

        sa, ta = project_3d_batch(
            a, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
        )
        sb, tb = project_3d_batch(
            b, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
        )
        if act_f is not None:
            e_act = np.maximum(act_f[ra[sel]], act_f[rb[sel]])
        else:
            e_act = None
        for i in range(len(a)):
            x0, y0 = int(sa[i]), int(ta[i])
            x1, y1 = int(sb[i]), int(tb[i])
            if not (
                (0 <= x0 < self.view_w and 0 <= y0 < view_lim_y)
                or (0 <= x1 < self.view_w and 0 <= y1 < view_lim_y)
            ):
                continue
            c = rgb_u[i]
            if e_act is not None:
                a_i = float(e_act[i])
                if a_i < 0.03:
                    # Faintest map — 1px, heavily dimmed rgb already.
                    width = 1
                elif a_i < 0.15:
                    width = 1
                elif a_i < 0.35:
                    width = 2
                elif a_i < 0.60:
                    width = 3
                else:
                    width = 4
            else:
                width = 2  # T is on (empty T returns earlier)
            pygame.draw.line(
                self._view_surf,
                (int(c[0]), int(c[1]), int(c[2])),
                (x0, y0),
                (x1, y1),
                width,
            )

    def _draw_recipe_graph_overlay(
        self,
        cx: float,
        cy: float,
        view_lim_y: int,
        graph_f: set[int],
        *,
        max_edges: int = 2500,
    ) -> None:
        """Sample atlas recipe edges as soma→soma chords (connection graph)."""
        mapped = self._recipe_soma_row_edges()
        if mapped is None:
            return
        ra, rb, w = mapped
        n = self.n_shown
        valid = (ra >= 0) & (rb >= 0) & (ra < n) & (rb < n)
        idx = np.flatnonzero(valid)
        if graph_f and len(idx):
            gf = list(graph_f)
            keep = np.isin(self.points_cls[ra[idx]], gf) | np.isin(
                self.points_cls[rb[idx]], gf
            )
            idx = idx[keep]
        if len(idx) == 0:
            return
        if len(idx) > max_edges:
            part = np.argpartition(-w[idx], max_edges - 1)[:max_edges]
            idx = idx[part]
        a = self.points_xyz[ra[idx]]
        b = self.points_xyz[rb[idx]]
        sa, ta = project_3d_batch(
            a, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
        )
        sb, tb = project_3d_batch(
            b, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
        )
        col = (55, 70, 90)
        for i in range(len(a)):
            x0, y0 = int(sa[i]), int(ta[i])
            x1, y1 = int(sb[i]), int(tb[i])
            if (
                (0 <= x0 < self.view_w and 0 <= y0 < view_lim_y)
                or (0 <= x1 < self.view_w and 0 <= y1 < view_lim_y)
            ):
                pygame.draw.line(self._view_surf, col, (x0, y0), (x1, y1), 1)

    def _draw_growth_overlay(
        self,
        cx: float,
        cy: float,
        view_lim_y: int,
        *,
        axon_f: set[int],
        dend_f: set[int],
    ) -> None:
        """Draw axon trails, synapse marks, dendrite stubs (bundles form via peer pull)."""
        if not axon_f and not dend_f:
            return
        # Any A/D focus uses the same readable stroke weight (no half-class demotion).
        axon_on = bool(axon_f)
        dend_on = bool(dend_f)
        # Never pass empty/None as "all classes" — empty means hide.
        cls_filter: set[int] | None = set(axon_f) if axon_f else None
        ray_max = DRAW_FOCUS_RAY_MAX if axon_on else GROW_RAY_DRAW_MAX
        tip_max = DRAW_FOCUS_RAY_MAX if axon_on else GROW_TIP_DRAW_MAX
        poly_pts = DRAW_FOCUS_POLY_POINTS if axon_on else None
        seg_max = DRAW_SEG_FOCUS_MAX if axon_on else GROW_SEG_DRAW_MAX
        line_w = FOCUS_LINE_WIDTH if axon_on else 1

        def _cap_by_class(
            a: np.ndarray,
            b: np.ndarray,
            cols: np.ndarray,
            cls: np.ndarray,
            seg_cap: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Subsample segments with equal share per class (multi-focus fair)."""
            if len(a) <= seg_cap:
                return a, b, cols
            classes = np.unique(cls)
            per = max(1, seg_cap // max(1, len(classes)))
            keep: list[np.ndarray] = []
            for c in classes.tolist():
                m = np.flatnonzero(cls == int(c))
                if len(m) > per:
                    step = max(1, (len(m) + per - 1) // per)
                    m = m[::step][:per]
                keep.append(m)
            idx = np.concatenate(keep) if keep else np.zeros((0,), dtype=np.int32)
            return a[idx], b[idx], cols[idx]

        def _draw_segments(
            a: np.ndarray,
            b: np.ndarray,
            cols: np.ndarray,
            *,
            width: int = 1,
            seg_cap: int = DRAW_SEG_MAX,
            cls: np.ndarray | None = None,
        ) -> None:
            if len(a) == 0:
                return
            if len(a) > seg_cap:
                if cls is not None and len(cls) == len(a):
                    a, b, cols = _cap_by_class(a, b, cols, cls, seg_cap)
                else:
                    step = max(1, (len(a) + seg_cap - 1) // seg_cap)
                    a = a[::step][:seg_cap]
                    b = b[::step][:seg_cap]
                    cols = cols[::step][:seg_cap]
            sa, ta = project_3d_batch(
                a, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
            )
            sb, tb = project_3d_batch(
                b, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
            )
            for i in range(len(a)):
                x0, y0 = int(sa[i]), int(ta[i])
                x1, y1 = int(sb[i]), int(tb[i])
                if (
                    (0 <= x0 < self.view_w and 0 <= y0 < view_lim_y)
                    or (0 <= x1 < self.view_w and 0 <= y1 < view_lim_y)
                ):
                    c = cols[i]
                    pygame.draw.line(
                        self._view_surf,
                        (int(c[0]), int(c[1]), int(c[2])),
                        (x0, y0),
                        (x1, y1),
                        width,
                    )

        # Skip expensive overlay fetch when A/D both off (caller should gate too).
        if not (
            (self.show_layer_axons and axon_f)
            or (self.show_layer_dendrites and dend_f)
        ):
            return
        (
            _tip_xyz,
            _tip_code,
            ray_a,
            ray_b,
            ray_cls,
            dend_a,
            dend_b,
            dend_cls,
        ) = self.fly.growth_overlay_arrays(
            tip_max=tip_max,
            ray_max=ray_max,
            cls_filter=cls_filter,
            poly_points=poly_pts,
            seg_max=seg_max,
        )

        if self.show_layer_axons and axon_f and len(ray_a):
            # cls_filter already applied upstream when axon focus is set.
            ra, rb, rc = ray_a, ray_b, ray_cls
            _draw_segments(
                ra,
                rb,
                _axon_rgb_for_cls(rc),
                width=line_w,
                seg_cap=DRAW_SEG_FOCUS_MAX if axon_on else DRAW_SEG_MAX,
                cls=rc,
            )

        # Cervical connective: keep neck-crossing trails visible (Yield≠neck fill).
        if self.show_layer_axons and axon_f and hasattr(self.fly, "neck_trail_segments"):
            na, nb, nc = self.fly.neck_trail_segments(
                max_segs=1600 if axon_on else 900,
                cls_filter=cls_filter,
            )
            if len(na):
                _draw_segments(
                    na,
                    nb,
                    _axon_rgb_for_cls(nc),
                    width=max(2, line_w + 1),
                    seg_cap=len(na),
                    cls=nc,
                )

        if self.show_layer_axons and axon_f and len(self.fly.growth_tip):
            tip = self.fly.growth_tip
            code = self.fly.growth_status.astype(np.int32)
            cls = self.fly.growth_cls
            mask = np.isin(cls, list(axon_f)) & (code != 4)
            idx = np.flatnonzero(mask)
            # Equal tip budget per enabled class.
            if axon_f and len(idx):
                parts_t: list[np.ndarray] = []
                classes = sorted(axon_f)
                per_t = max(1, tip_max // max(1, len(classes)))
                for c in classes:
                    sub = idx[cls[idx] == int(c)]
                    if len(sub) > per_t:
                        step = max(1, (len(sub) + per_t - 1) // per_t)
                        sub = sub[::step][:per_t]
                    if len(sub):
                        parts_t.append(sub)
                idx = (
                    np.concatenate(parts_t)
                    if parts_t
                    else np.zeros((0,), dtype=np.int32)
                )
            elif len(idx) > tip_max:
                step = max(1, len(idx) // tip_max)
                idx = idx[::step][:tip_max]
            if len(idx):
                tip_xyz = tip[idx]
                tip_code = code[idx]
                tip_cls = cls[idx]
                base = _axon_rgb_for_cls(tip_cls).astype(np.int16)
                tip_rgb = base.copy()
                g = tip_code == 1
                s = tip_code == 2
                c = tip_code == 3
                if np.any(g):
                    tip_rgb[g] = np.minimum(255, tip_rgb[g] + 18)
                if np.any(s):
                    tip_rgb[s] = np.minimum(255, tip_rgb[s] + 28)
                if np.any(c):
                    # Synapse mark — slight lift, still below soma brightness.
                    tip_rgb[c] = np.minimum(255, tip_rgb[c] + 35)
                tip_rgb = tip_rgb.astype(np.uint8)
                sx, sy = project_3d_batch(
                    tip_xyz,
                    self.view_angle_x,
                    self.view_angle_y,
                    cx,
                    cy,
                    self.scale_3d,
                )
                xi = np.rint(sx).astype(np.int32)
                yi = np.rint(sy).astype(np.int32)
                m = (xi >= 0) & (xi < self.view_w) & (yi >= 0) & (yi < view_lim_y)
                if np.any(m):
                    px = pygame.surfarray.pixels3d(self._view_surf)
                    for mask_m, stamp in (
                        (m & (tip_code == 3), ((0, 0), (1, 0))),
                        (
                            m & ((tip_code == 1) | (tip_code == 2)),
                            ((0, 0),),
                        ),
                    ):
                        if not np.any(mask_m):
                            continue
                        for dx, dy in stamp:
                            xx = xi[mask_m] + dx
                            yy = yi[mask_m] + dy
                            ok = (
                                (xx >= 0)
                                & (xx < self.view_w)
                                & (yy >= 0)
                                & (yy < view_lim_y)
                            )
                            px[xx[ok], yy[ok]] = np.maximum(
                                px[xx[ok], yy[ok]], tip_rgb[mask_m][ok]
                            )
                    del px

        if not self.show_layer_dendrites or not dend_f or len(dend_a) == 0:
            return
        m = np.isin(dend_cls, list(dend_f))
        if not np.any(m):
            return
        _draw_segments(
            dend_a[m],
            dend_b[m],
            _dendrite_rgb_for_cls(dend_cls[m]),
            width=FOCUS_LINE_WIDTH if dend_on else 1,
            seg_cap=DRAW_SEG_FOCUS_MAX if dend_on else DRAW_SEG_MAX,
            cls=dend_cls[m],
        )
        # Synapse contact also stamped in dendrite color (meets axon tip).
        syn = dend_a[m]
        syn_rgb = _dendrite_rgb_for_cls(dend_cls[m])
        sx, sy = project_3d_batch(
            syn, self.view_angle_x, self.view_angle_y, cx, cy, self.scale_3d
        )
        xi = np.rint(sx).astype(np.int32)
        yi = np.rint(sy).astype(np.int32)
        vis = (xi >= 0) & (xi < self.view_w) & (yi >= 0) & (yi < view_lim_y)
        if np.any(vis):
            px = pygame.surfarray.pixels3d(self._view_surf)
            for dx, dy in ((0, 0),):
                xx = xi[vis] + dx
                yy = yi[vis] + dy
                ok = (
                    (xx >= 0)
                    & (xx < self.view_w)
                    & (yy >= 0)
                    & (yy < view_lim_y)
                )
                px[xx[ok], yy[ok]] = np.maximum(
                    px[xx[ok], yy[ok]], syn_rgb[vis][ok]
                )
            del px

    def _class_label(self, idx: int) -> str:
        name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        return CLASS_LABELS.get(name, name.replace("_", " ").title())

    def _class_name(self, idx: int) -> str:
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"

    def _class_help(self, idx: int) -> str:
        return CLASS_HELP.get(
            self._class_name(idx),
            "Группа суперкласса MaleCNS (сомы, не аксоны).",
        )

    def _wrap_text(self, text: str, font: pygame.font.Font, max_w: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
        cur = words[0]
        for w in words[1:]:
            trial = f"{cur} {w}"
            if font.size(trial)[0] <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def _ensure_class_centroids(self) -> np.ndarray:
        """Mean soma xyz per class (proc mass fallback for sparse classes)."""
        n = len(self.class_names)
        key = (id(self.points_xyz), int(self.n_shown), int(self.n_proc_shown), n)
        if (
            self._class_cent_xyz is not None
            and self._class_cent_key == key
            and len(self._class_cent_xyz) == n
        ):
            return self._class_cent_xyz
        cents = np.zeros((n, 3), dtype=np.float64)
        have = np.zeros(n, dtype=bool)
        if self.n_shown > 0 and len(self.points_cls) == self.n_shown:
            for i in range(n):
                m = self.points_cls == i
                if np.any(m):
                    cents[i] = self.points_xyz[m].mean(axis=0)
                    have[i] = True
        if self.n_proc_shown > 0 and len(self.proc_cls) == self.n_proc_shown:
            for i in range(n):
                if have[i]:
                    continue
                m = self.proc_cls == i
                if np.any(m):
                    cents[i] = self.proc_xyz[m].mean(axis=0)
                    have[i] = True
        self._class_cent_xyz = cents
        self._class_cent_key = key
        return cents

    def _view_project_center(self) -> tuple[float, float]:
        return (
            self.view_w * 0.5 + float(self.pan_x),
            (self.view_h - NAV_PANEL_H) * 0.5 + float(self.pan_y),
        )

    def _hover_tip_content(
        self, tip: tuple[str, int | str]
    ) -> tuple[str, str, tuple[int, int, int]] | None:
        kind, key = tip
        if kind == "class":
            idx = int(key)
            if idx < 0 or idx >= len(self.class_names):
                return None
            return (
                self._class_label(idx),
                self._class_help(idx),
                self._class_color(idx),
            )
        if kind == "device":
            name = str(key)
            title = {"lamp": "Lamp", "ocelli": "Ocelli", "turret": "Turret"}.get(
                name, name
            )
            body = DEVICE_HELP.get(name, "")
            accent = {
                "lamp": (240, 210, 120),
                "ocelli": (160, 210, 255),
                "turret": (180, 220, 160),
            }.get(name, (180, 180, 200))
            if not body:
                return None
            return title, body, accent
        return None

    def _update_hover_tip(self, pos: tuple[int, int]) -> None:
        """Pick class / device under cursor → floating tip in the 3D view."""
        self._hover_tip = None
        self._hover_tip_pos = pos
        if self.exit_confirm_modal.is_visible():
            return
        if self._schematic_active():
            return
        # Legend class row / mini chip under cursor.
        hit = self._hit_legend(pos)
        if hit is not None:
            kind, idx = hit
            if kind in ("soma", "neuron", "group", "class") or kind in (
                "graph",
                "axon",
                "dendrite",
                "tract",
                "basin",
            ):
                if 0 <= idx < len(self.class_names):
                    self._hover_tip = ("class", idx)
                    return
            return  # other legend controls — no tip

        if not self._in_view(pos):
            return
        if self.legend_panel_rect is not None and self.legend_panel_rect.collidepoint(
            pos
        ):
            return
        for prect in getattr(self, "_side_panel_rects", {}).values():
            if prect is not None and prect.collidepoint(pos):
                return
        if self._hit_nav_button(pos):
            return

        mx, my = float(pos[0]), float(pos[1])
        cx_v, cy_v = self._view_project_center()
        # Screen coords are view-local in project_3d_batch; convert to absolute.
        # project returns coords relative to view surface origin via center_x/y
        # used when drawing into _view_surf; on screen we add view_x/view_y.
        best: tuple[float, tuple[str, int | str]] | None = None
        hit_r2 = 42.0 * 42.0

        def _consider(sx: float, sy: float, tip: tuple[str, int | str]) -> None:
            nonlocal best
            dx = (self.view_x + sx) - mx
            dy = (self.view_y + sy) - my
            d2 = dx * dx + dy * dy
            if d2 > hit_r2:
                return
            if best is None or d2 < best[0]:
                best = (d2, tip)

        # Devices first (few anchors).
        if self.lamp_on and getattr(self, "_lamp_xyz", None) is not None:
            xyz = np.asarray(self._lamp_xyz, dtype=np.float64).reshape(1, 3)
            sx, sy = project_3d_batch(
                xyz,
                self.view_angle_x,
                self.view_angle_y,
                cx_v,
                cy_v,
                self.scale_3d,
            )
            _consider(float(sx[0]), float(sy[0]), ("device", "lamp"))
        if self.ocelli_on:
            pose = self._ocelli_rig_pose()
            if pose is not None:
                lenses = np.stack(
                    [np.asarray(o["lens"], dtype=np.float64) for o in pose], axis=0
                )
                sx, sy = project_3d_batch(
                    lenses,
                    self.view_angle_x,
                    self.view_angle_y,
                    cx_v,
                    cy_v,
                    self.scale_3d,
                )
                for i in range(len(sx)):
                    _consider(float(sx[i]), float(sy[i]), ("device", "ocelli"))
        if self.turret_on:
            tu = self._turret_mech_pose()
            if tu is not None:
                sites = np.asarray(tu.get("sites", []), dtype=np.float64)
                if len(sites):
                    sx, sy = project_3d_batch(
                        sites,
                        self.view_angle_x,
                        self.view_angle_y,
                        cx_v,
                        cy_v,
                        self.scale_3d,
                    )
                    for i in range(len(sx)):
                        _consider(float(sx[i]), float(sy[i]), ("device", "turret"))

        # Class mass centroids.
        cents = self._ensure_class_centroids()
        if len(cents):
            sx, sy = project_3d_batch(
                cents.astype(np.float64),
                self.view_angle_x,
                self.view_angle_y,
                cx_v,
                cy_v,
                self.scale_3d,
            )
            for i in range(len(cents)):
                if i < len(self.class_counts) and int(self.class_counts[i]) <= 0:
                    # still allow if we have a centroid from proc mass
                    if float(np.linalg.norm(cents[i])) < 1e-9:
                        continue
                _consider(float(sx[i]), float(sy[i]), ("class", i))

        if best is not None:
            self._hover_tip = best[1]

    def _draw_hover_tip(self) -> None:
        tip = getattr(self, "_hover_tip", None)
        if tip is None:
            return
        content = self._hover_tip_content(tip)
        if content is None:
            return
        title, body, col = content
        max_w = 280
        pad_in = 8
        bar_w = 3
        text_w = max_w - pad_in * 2 - bar_w - 4
        title_font = self.small_font
        title_lines = self._wrap_text(title, title_font, text_w)
        body_lines = self._wrap_text(body, self.hint_font, text_w)
        line_h = self.hint_font.get_height() + 2
        block_h = (
            pad_in
            + len(title_lines) * (title_font.get_height() + 2)
            + 2
            + len(body_lines) * line_h
            + pad_in
        )
        mx, my = self._hover_tip_pos
        # Prefer below-right of cursor; flip to stay inside the 3D view.
        x = mx + 14
        y = my + 16
        left = self.view_x + 8
        right = self.view_x + self.view_w - 8
        top = self.view_y + 8
        bottom = self.view_y + self.view_h - NAV_PANEL_H - 8
        if self.legend_panel_rect is not None and not self.legend_mini:
            right = min(right, self.legend_panel_rect.left - 6)
        if x + max_w > right:
            x = mx - max_w - 14
        if x < left:
            x = left
        if y + block_h > bottom:
            y = my - block_h - 12
        if y < top:
            y = top
        tip_bg = pygame.Surface((max_w, block_h), pygame.SRCALPHA)
        tip_bg.fill((22, 24, 34, 210))
        self.screen.blit(tip_bg, (x, y))
        pygame.draw.rect(self.screen, col, (x, y, bar_w, block_h))
        pygame.draw.rect(
            self.screen,
            (col[0] // 2, col[1] // 2, col[2] // 2),
            (x, y, max_w, block_h),
            1,
        )
        ty = y + pad_in
        accent = (
            min(255, col[0] + 50),
            min(255, col[1] + 50),
            min(255, col[2] + 50),
        )
        for line in title_lines:
            surf = title_font.render(line, True, accent)
            self.screen.blit(surf, (x + bar_w + pad_in, ty))
            ty += title_font.get_height() + 2
        ty += 2
        for line in body_lines:
            surf = self.hint_font.render(line, True, (200, 200, 220))
            self.screen.blit(surf, (x + bar_w + pad_in, ty))
            ty += line_h

    def _draw_checkbox(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        size: int,
        checked: bool,
        accent: tuple[int, int, int],
        *,
        partial: bool = False,
    ) -> pygame.Rect:
        """Checkbox; ``partial`` = indeterminate (some but not all)."""
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surface, (45, 45, 55), rect)
        border = (
            accent
            if partial
            else ((200, 200, 230) if checked else (100, 100, 120))
        )
        pygame.draw.rect(surface, border, rect, 1)
        if partial:
            bar = pygame.Rect(x + 2, y + size // 2 - 1, size - 4, 3)
            pygame.draw.rect(surface, accent, bar)
        elif checked:
            pygame.draw.lines(
                surface,
                accent,
                False,
                [
                    (x + 2, y + size // 2),
                    (x + size // 2 - 1, y + size - 3),
                    (x + size - 2, y + 3),
                ],
                2,
            )
        return rect


    def _draw_inset_panel(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        fill: tuple[int, int, int, int] = (28, 30, 42, 170),
        border: tuple[int, int, int] = (70, 74, 95),
        accent: tuple[int, int, int] | None = None,
    ) -> None:
        """Soft card behind a legend section (matches Help tip chrome)."""
        if h <= 1 or w <= 1:
            return
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(fill)
        self.screen.blit(surf, (x, y))
        if accent is not None:
            pygame.draw.rect(self.screen, accent, (x, y, 3, h))
        pygame.draw.rect(self.screen, border, (x, y, w, h), 1)

    def _legend_state_key(self) -> tuple:
        """UI state for legend panel reuse. Live I/aim live on 3D plates, not here."""
        yld = self._recipe_yield()
        return (
            bool(self.legend_mini),
            tuple(
                (p, bool(self.side_panel_mini.get(p, False)), int(self.side_panel_scroll.get(p, 0)))
                for p in SIDE_PANELS
            ),
            bool(self.draw_thin_open),
            str(self.draw_thin_mode),
            tuple(
                sorted(
                    (k, round(float(v), 3))
                    for k, v in (self.draw_density or {}).items()
                )
            ),
            bool(self.show_layer_neurons),
            bool(self.show_layer_graph),
            bool(self.show_layer_axons),
            bool(self.show_layer_dendrites),
            bool(self.show_layer_basins),
            round(float(self.basin_form), 2),
            round(float(getattr(self, "view_detail", 1.0)), 2),
            bool(self.light_enabled),
            bool(self.lamp_on),
            round(float(self.sky_fill), 2),
            round(float(self.lamp_speed), 2),
            bool(self.ground_on),
            round(float(self.ground_relief), 2),
            bool(getattr(self, "world_swing_on", False)),
            round(float(getattr(self, "world_swing_amp", 0.0)), 3),
            round(float(getattr(self, "world_swing_hz", 0.0)), 3),
            bool(self.ocelli_on),
            bool(self.turret_on),
            bool(self.learn_on),
            bool(getattr(self, "show_ocelli_funnel", False)),
            bool(getattr(self, "show_turret_funnel", False)),
            bool(getattr(self, "scope_on", False)),
            bool(getattr(self, "motor_socket_on", False)),
            bool(
                getattr(getattr(self, "motor_socket", None), "frozen", False)
            ),
            # AI Brain controls (must invalidate legend cache on toggle).
            bool(getattr(self, "sensor_port_on", False)),
            bool(getattr(self, "sensor_port_train", False)),
            bool(
                getattr(getattr(self, "sensor_port_nn", None), "frozen", False)
            ),
            (
                int(getattr(getattr(self, "sensor_port_nn", None), "step_n", 0) or 0)
                // 30
                if bool(getattr(self, "sensor_port_on", False))
                else 0
            ),
            (
                int(
                    round(
                        float(
                            getattr(
                                getattr(self, "sensor_port_nn", None),
                                "loss_ema",
                                0.0,
                            )
                            or 0.0
                        )
                        * 50.0
                    )
                )
                if bool(getattr(self, "sensor_port_on", False))
                else 0
            ),
            bool(getattr(self, "eye_left_on", False)),
            bool(getattr(self, "eye_right_on", False)),
            bool(getattr(self, "jo_left_on", False)),
            bool(getattr(self, "jo_right_on", False)),
            bool(self.fly.is_adult()) if getattr(self, "fly", None) else False,
            str(self._constructor_panel_kind()),
            round(float(yld.get("pct", 0.0)), 1),
            str(yld.get("verdict", "")),
            int(yld.get("open", 0) or 0),
            int(len(getattr(getattr(self, "_training_scope", None), "path_neurons", ()) or ())),
            int(len(getattr(getattr(self, "_well_field", None), "specs", ()) or ())),
            int(round(float(getattr(self, "learn_dw_ema", 0.0)) * 40.0)),
            int(getattr(self, "learn_n_edges", 0) or 0),
            int(getattr(self, "_learn_flash_n", 0) or 0),
            bool(
                self.learn_on
                and self.turret_on
                and self.sim_running
                and getattr(self, "_activity_running", False)
            ),
            frozenset(self.focus_soma_indices),
            frozenset(self.focus_graph_indices),
            frozenset(self.focus_axon_indices),
            frozenset(self.focus_dendrite_indices),
            frozenset(self.focus_tract_indices),
            frozenset(self.focus_basin_indices),
            int(self.n_shown),
            int(self.n_proc_shown),
            int(self.fly.growth_trail_point_count()),
            int(self.view_w),
            int(self.view_h),
            int(self.view_x),
            "legend_v9_tract_col",
        )

    def _draw_legend(self) -> None:
        """Right overlay over 3D — mini strip or full legend (cached when stable)."""
        key = self._legend_state_key()
        if (
            key == self._legend_blit_key
            and self._legend_blit_surf is not None
        ):
            self.screen.blit(self._legend_blit_surf, self._legend_blit_pos)
            self._legend_hits = list(self._legend_hits_cached)
            self.legend_panel_rect = self._legend_panel_rect_cached
            self.legend_toggle_rect = self._legend_toggle_rect_cached
            self._side_panel_rects = dict(
                getattr(self, "_side_panel_rects_cached", {}) or {}
            )
            self._side_panel_toggle_rects = dict(
                getattr(self, "_side_panel_toggle_rects_cached", {}) or {}
            )
            self.legend_help_rect = None
            return
        real = self.screen
        cap = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        cap.fill((0, 0, 0, 0))
        self.screen = cap
        try:
            self._draw_legend_into_screen()
        finally:
            self.screen = real
        r = self.legend_panel_rect
        if r is not None:
            clip = r.clip(cap.get_rect())
            if clip.width > 0 and clip.height > 0:
                self._legend_blit_surf = cap.subsurface(clip).copy()
                self._legend_blit_pos = (clip.x, clip.y)
                self._legend_blit_key = key
                self._legend_hits_cached = list(self._legend_hits)
                self._legend_panel_rect_cached = self.legend_panel_rect
                self._legend_toggle_rect_cached = self.legend_toggle_rect
                self._side_panel_rects_cached = dict(self._side_panel_rects)
                self._side_panel_toggle_rects_cached = dict(
                    self._side_panel_toggle_rects
                )
                self._legend_help_rect_cached = None
        real.blit(cap, (0, 0))

    def _side_panel_rows(
        self, panel_id: str, total: int
    ) -> tuple[list[tuple], list[str | None]]:
        """Build rows + card tags for one independent right-side panel."""
        rows: list[tuple] = []
        block_of: list[str | None] = []

        def _add(row: tuple, block: str | None) -> None:
            rows.append(row)
            block_of.append(block)

        def _add_gap() -> None:
            _add(("gap",), None)

        if panel_id == "filters":
            # Order by significance: Super-class first, then density, basins, meters.
            order = sorted(
                range(len(self.class_names)),
                key=lambda i: int(self.class_counts[i]) if i < len(self.class_counts) else 0,
                reverse=True,
            )
            _add(("panel_head", "Super-class", (140, 170, 200)), "classes")
            _add(
                (
                    "hint",
                    "N·G·A·D·T·B on = show · T = recipe tracts · empty = hide",
                    (120, 120, 140),
                ),
                "classes",
            )
            _add(("quick",), "classes")
            for i in order:
                cnt = int(self.class_counts[i]) if i < len(self.class_counts) else 0
                if cnt <= 0:
                    continue
                pct = 100.0 * cnt / total
                _add(("class", i, self._class_label(i), cnt, pct), "classes")
                if self.draw_thin_mode == "manual":
                    _add(("thin_slider", i), "classes")
            active_cls = self._active_class_indices()
            n_soma_f = len(self.focus_soma_indices & active_cls)
            n_graph_f = len(self.focus_graph_indices & active_cls)
            n_axon_f = len(self.focus_axon_indices & active_cls)
            n_dend_f = len(self.focus_dendrite_indices & active_cls)
            n_tract_f = len(self.focus_tract_indices & active_cls)
            n_basin_f = len(self.focus_basin_indices & active_cls)
            if n_soma_f or n_graph_f or n_axon_f or n_dend_f or n_tract_f or n_basin_f:
                _add(
                    (
                        "hint",
                        f"Focus N{n_soma_f} G{n_graph_f} A{n_axon_f} D{n_dend_f} "
                        f"T{n_tract_f} B{n_basin_f}",
                        (200, 200, 120),
                    ),
                    "classes",
                )
            else:
                _add(
                    (
                        "hint",
                        "Quick: N/G/A/D/T/B · empty = hide CNS · G=atlas T=chords",
                        (120, 120, 140),
                    ),
                    "classes",
                )
            _add_gap()

            _add(("panel_head", "Thinning", (140, 170, 200)), "thinning")
            _add(("thin_bar",), "thinning")
            if self.draw_thin_open:
                for mi, mode in enumerate(("auto", "manual", "off")):
                    _add(("thin_opt", mi, mode), "thinning")
            _add_gap()

            _add(("panel_head", "Basins", (140, 170, 200)), "basins")
            _add(("basin_form",), "basins")
            _add(
                ("hint", "Split  few large funnels  ← →  many wells", (110, 115, 130)),
                "basins",
            )
            _add_gap()

            _add(("panel_head", "Cloud", (140, 170, 200)), "cloud")
            soma_drawn = self._soma_drawn_count()
            _add(
                (
                    "text",
                    f"Somata  {soma_drawn:,} / {self.n_source_points:,} drawn",
                    (160, 160, 180),
                ),
                "cloud",
            )
            _add(
                (
                    "text",
                    f"Atlas ghost {self.n_proc_shown:,} pts",
                    (160, 160, 180) if self.n_proc_shown else (120, 80, 80),
                ),
                "cloud",
            )
            grown_n = self.fly.growth_trail_point_count()
            _add(
                (
                    "text",
                    f"Grown trails {grown_n:,} pts",
                    (160, 170, 190) if grown_n else (120, 120, 140),
                ),
                "cloud",
            )
            _add_gap()

            yld = self._recipe_yield()
            _add(("panel_head", "Yield", (140, 170, 200)), "yield")
            head, head_col = self._yield_headline(yld)
            _add(("text", head, head_col), "yield")
            _add(
                (
                    "hint",
                    "Detail → Learn badge (top-left) · Growth then plasticity",
                    (110, 115, 130),
                ),
                "yield",
            )

        elif panel_id == "env":
            _add(("panel_head", "Environment", (140, 170, 200)), "lights")
            _add(("light_lamp",), "lights")
            _add(("light_sky",), "lights")
            _add(("light_speed",), "lights")
            _add(("ground_on",), "lights")
            _add(("ground_relief",), "lights")
            _add(("world_swing",), "lights")
            _add(("world_swing_amp",), "lights")
            _add(("world_swing_hz",), "lights")
            _add(
                (
                    "hint",
                    "Lamp · ground · Matrix world swing (fly fixed)",
                    (110, 115, 130),
                ),
                "lights",
            )

        elif panel_id == "devices":
            _add(("panel_head", "Training scope", (160, 190, 150)), "blueprint")
            for brow in blueprint_panel_rows():
                _add(("prow", brow, None), "blueprint")
            _add_gap()
            devices = list(
                getattr(self, "device_list", None)
                or getattr(getattr(self, "blueprint", None), "device_list", None)
                or []
            )
            for d in devices:
                kind = getattr(d, "kind", getattr(d, "name", "device"))
                _add(("panel_head", panel_title(d), panel_color(d)), kind)
                for row in d.panel_rows():
                    _add(("prow", row, d), kind)
                _add_gap()
            if not devices:
                _add(("panel_head", "Devices", (140, 170, 200)), "devices")
                _add(
                    ("hint", "No devices in Blueprint.device_list", (110, 115, 130)),
                    "devices",
                )

        elif panel_id == "learn":
            if self._constructor_panel_kind() == "growth":
                st = self._yield_phase()
                _add(("panel_head", "Growth → Learn", (180, 200, 160)), "learn")
                _add(("text", str(st["badge"]), tuple(st["accent"])), "learn")
                _add(
                    (
                        "hint",
                        "Rising: badge shows Yield · Learn/Freeze after Adult",
                        (110, 115, 130),
                    ),
                    "learn",
                )
                if st.get("ready"):
                    _add(
                        (
                            "hint",
                            "Ready · enter Adult to unlock plasticity",
                            (160, 200, 150),
                        ),
                        "learn",
                    )
            else:
                _add(("panel_head", "Learn", (160, 200, 170)), "learn")
                for lrow in learn_panel_rows():
                    _add(("prow", lrow, None), "learn")
                _add(
                    (
                        "hint",
                        "Badge = CNS Track · hop carve (socket = legacy bypass)",
                        (110, 115, 130),
                    ),
                    "learn",
                )

        else:  # view
            kn = self._view_detail_knobs()
            _add(("panel_head", "View", (160, 180, 210)), "view")
            _add(("view_detail",), "view")
            _add(("hint", str(kn["label"]), (160, 175, 195)), "view")
            _add(
                (
                    "hint",
                    "Detail ↓ = fewer Hop/Activity ticks + lighter overlays",
                    (110, 115, 130),
                ),
                "view",
            )

        return rows, block_of

    def _paint_legend_visible_rows(
        self,
        *,
        rows: list[tuple],
        vis0: int,
        vis1: int,
        corner_x: int,
        corner_y: int,
        content_x: int,
        panel_width: int,
        line_h: int,
        check_sz: int,
        pulse: float,
        prow_ctx: PanelDrawCtx,
    ) -> None:
        """Paint one visible window of legend rows (shared by side panels)."""
        swatch = 10
        mode_labels = {
            "auto": "Auto",
            "manual": "Manual",
            "off": "Off",
        }
        y_off = 0
        for item in rows[vis0:vis1]:
            kind = item[0]
            if kind == "gap":
                pass
            elif kind == "title":
                _, text_s, color = item
                surf = self.font.render(text_s, True, color)
                self.screen.blit(
                    surf,
                    (
                        corner_x,
                        corner_y + y_off + (line_h - surf.get_height()) // 2,
                    ),
                )
            elif kind == "tabs":
                cur = str(getattr(self, "legend_tab", "atlas"))
                x0 = content_x
                gap_t = 4
                for ti, tname in enumerate(LEGEND_TABS):
                    lab = LEGEND_TAB_LABELS[tname]
                    on = cur == tname
                    ts = self.hint_font.render(
                        lab, True, (230, 235, 250) if on else (130, 140, 160)
                    )
                    tw = ts.get_width() + 12
                    th = line_h - 4
                    bx = x0
                    by = corner_y + y_off + 2
                    fill = (50, 70, 100) if on else (32, 36, 48)
                    edge = (160, 200, 240) if on else (70, 80, 100)
                    pygame.draw.rect(
                        self.screen, fill, (bx, by, tw, th), border_radius=3
                    )
                    pygame.draw.rect(
                        self.screen, edge, (bx, by, tw, th), width=1, border_radius=3
                    )
                    self.screen.blit(
                        ts, (bx + 6, by + (th - ts.get_height()) // 2)
                    )
                    self._legend_hits.append(
                        (pygame.Rect(bx, by, tw, th).inflate(2, 2), "legend_tab", ti)
                    )
                    x0 += tw + gap_t
            elif kind == "panel_head":
                _, text_s, color = item
                surf = self.small_font.render(text_s, True, color)
                self.screen.blit(
                    surf,
                    (
                        content_x + 4,
                        corner_y + y_off + (line_h - surf.get_height()) // 2,
                    ),
                )
            elif kind == "hint":
                # Static tuple hint ("hint", text, color) — same renderer as Hint row.
                _, text_s, color = item
                prow_ctx.y_off = corner_y + y_off
                render_hint(prow_ctx, Hint(str(text_s), tuple(color)))
            elif kind == "prow":
                _, row, device = item
                prow_ctx.y_off = corner_y + y_off
                render_panel_row(prow_ctx, row, device)
            elif kind == "thin_bar":
                cur = mode_labels.get(self.draw_thin_mode, "Auto")
                arrow = "^" if self.draw_thin_open else "v"
                btn_txt = f"{cur} {arrow}"
                btn_s = self.small_font.render(btn_txt, True, (220, 220, 235))
                bw = max(78, btn_s.get_width() + 16)
                bx = corner_x + panel_width - bw - 18
                by = corner_y + y_off + 2
                bh = line_h - 4
                pygame.draw.rect(self.screen, (40, 44, 58), (bx, by, bw, bh))
                pygame.draw.rect(self.screen, (110, 120, 150), (bx, by, bw, bh), 1)
                self.screen.blit(
                    btn_s,
                    (
                        bx + (bw - btn_s.get_width()) // 2,
                        by + (bh - btn_s.get_height()) // 2,
                    ),
                )
                lab = self.small_font.render("Mode", True, (150, 155, 170))
                self.screen.blit(
                    lab,
                    (
                        content_x + 4,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                self._legend_hits.append(
                    (pygame.Rect(bx, by, bw, bh).inflate(4, 4), "thin_toggle", 0)
                )
            elif kind == "thin_opt":
                _, mi, mode = item
                on = self.draw_thin_mode == mode
                hints = {
                    "auto": "keep small groups",
                    "manual": "per-group density",
                    "off": "global stride",
                }
                mark = "*" if on else "o"
                col = (200, 220, 255) if on else (140, 145, 160)
                txt = f"  {mark} {mode_labels[mode]} - {hints[mode]}"
                surf = self.small_font.render(txt, True, col)
                self.screen.blit(surf, (content_x, corner_y + y_off + 2))
                hit = pygame.Rect(
                    content_x - 2, corner_y + y_off, panel_width - 16, line_h
                )
                self._legend_hits.append((hit, "thin_mode", mi))
            elif kind == "thin_slider":
                _, idx = item
                name = self.class_names[idx] if idx < len(self.class_names) else ""
                frac = float(self.draw_density.get(name, 1.0))
                active = self.draw_thin_mode == "manual"
                track_x = content_x + 12
                track_w = panel_width - 86
                track_h = 7
                track_y = corner_y + y_off + (line_h - track_h) // 2
                track = pygame.Rect(track_x, track_y, track_w, track_h)
                self._density_slider_rects[idx] = track
                bg_col = (50, 52, 68) if active else (36, 38, 48)
                fill_col = self._class_color(idx) if active else (70, 72, 88)
                pygame.draw.rect(self.screen, bg_col, track)
                fill_w = max(0, int(track_w * np.clip(frac, 0.0, 1.0)))
                if fill_w > 0:
                    pygame.draw.rect(
                        self.screen,
                        fill_col,
                        (track_x, track_y, fill_w, track_h),
                    )
                pygame.draw.rect(
                    self.screen,
                    (100, 105, 130) if active else (60, 62, 75),
                    track,
                    1,
                )
                pct_s = self.hint_font.render(
                    f"{int(round(frac * 100))}%",
                    True,
                    (200, 205, 220) if active else (90, 92, 105),
                )
                self.screen.blit(
                    pct_s,
                    (
                        track_x + track_w + 6,
                        corner_y + y_off + (line_h - pct_s.get_height()) // 2,
                    ),
                )
                if active:
                    self._legend_hits.append((track.inflate(6, 10), "thin_slider", idx))
            elif kind == "basin_form":
                frac = float(np.clip(self.basin_form, 0.0, 1.0))
                knobs = self._basin_form_knobs()
                lab = self.small_font.render("Split", True, (200, 190, 160))
                self.screen.blit(
                    lab,
                    (
                        content_x + 4,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                track_x = content_x + 52
                track_w = panel_width - 130
                track_h = 8
                track_y = corner_y + y_off + (line_h - track_h) // 2
                track = pygame.Rect(track_x, track_y, track_w, track_h)
                self._basin_form_rect = track
                pygame.draw.rect(self.screen, (50, 48, 40), track)
                fill_w = max(0, int(track_w * frac))
                if fill_w > 0:
                    pygame.draw.rect(
                        self.screen,
                        (200, 170, 110),
                        (track_x, track_y, fill_w, track_h),
                    )
                pygame.draw.rect(self.screen, (140, 120, 80), track, 1)
                pct_s = self.hint_font.render(
                    f"{knobs['max_modes']}",
                    True,
                    (220, 200, 150),
                )
                self.screen.blit(
                    pct_s,
                    (
                        track_x + track_w + 6,
                        corner_y + y_off + (line_h - pct_s.get_height()) // 2,
                    ),
                )
                self._legend_hits.append((track.inflate(6, 10), "basin_form", 0))
            elif kind == "light_lamp":
                on = bool(self.lamp_on)
                cb_y = corner_y + y_off + (line_h - check_sz) // 2
                cb = self._draw_checkbox(
                    self.screen, content_x, cb_y, check_sz, on, (220, 190, 120)
                )
                lab = self.small_font.render(
                    "Point lamp", True, (220, 210, 180) if on else (130, 130, 145)
                )
                self.screen.blit(
                    lab,
                    (
                        content_x + check_sz + 8,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                hit = pygame.Rect(
                    content_x - 2, corner_y + y_off, panel_width - 16, line_h
                )
                self._legend_hits.append((hit, "light_lamp", 0))
                self._legend_hits.append((cb.inflate(4, 4), "light_lamp", 0))
            elif kind in ("light_sky", "light_speed"):
                if kind == "light_sky":
                    label, key, val = "Sky", "sky", float(self.sky_fill)
                    fill_col = (100, 140, 190)
                else:
                    label, key, val = "Spin", "speed", float(
                        np.clip(self.lamp_speed / float(LAMP_SPEED_MAX), 0.0, 1.0)
                    )
                    fill_col = (220, 180, 100)
                lab = self.small_font.render(label, True, (180, 185, 200))
                self.screen.blit(
                    lab,
                    (
                        content_x + 4,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                track_x = content_x + 48
                track_w = panel_width - 120
                track_h = 8
                track_y = corner_y + y_off + (line_h - track_h) // 2
                track = pygame.Rect(track_x, track_y, track_w, track_h)
                self._light_slider_rects[key] = track
                pygame.draw.rect(self.screen, (40, 42, 55), track)
                fw = max(0, int(track_w * np.clip(val, 0.0, 1.0)))
                if fw > 0:
                    pygame.draw.rect(
                        self.screen, fill_col, (track_x, track_y, fw, track_h)
                    )
                pygame.draw.rect(self.screen, (100, 105, 125), track, 1)
                if kind == "light_sky":
                    disp = f"{int(round(self.sky_fill * 100))}%"
                else:
                    disp = f"{self.lamp_speed:.2f}"
                pct_s = self.hint_font.render(disp, True, (200, 205, 220))
                self.screen.blit(
                    pct_s,
                    (
                        track_x + track_w + 6,
                        corner_y + y_off + (line_h - pct_s.get_height()) // 2,
                    ),
                )
                self._legend_hits.append(
                    (track.inflate(6, 10), kind, 0)
                )
            elif kind == "view_detail":
                frac = float(np.clip(getattr(self, "view_detail", 1.0), 0.0, 1.0))
                lab = self.small_font.render("Detail", True, (180, 195, 220))
                self.screen.blit(
                    lab,
                    (
                        content_x + 4,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                track_x = content_x + 56
                track_w = panel_width - 130
                track_h = 8
                track_y = corner_y + y_off + (line_h - track_h) // 2
                track = pygame.Rect(track_x, track_y, track_w, track_h)
                self._view_detail_rect = track
                pygame.draw.rect(self.screen, (40, 45, 58), track)
                fw = max(0, int(track_w * frac))
                if fw > 0:
                    pygame.draw.rect(
                        self.screen,
                        (140, 180, 220),
                        (track_x, track_y, fw, track_h),
                    )
                pygame.draw.rect(self.screen, (100, 120, 150), track, 1)
                pct_s = self.hint_font.render(
                    f"{int(round(frac * 100))}%",
                    True,
                    (200, 210, 230),
                )
                self.screen.blit(
                    pct_s,
                    (
                        track_x + track_w + 6,
                        corner_y + y_off + (line_h - pct_s.get_height()) // 2,
                    ),
                )
                self._legend_hits.append((track.inflate(6, 10), "view_detail", 0))
            elif kind == "ground_on":
                on = bool(self.ground_on)
                cb_y = corner_y + y_off + (line_h - check_sz) // 2
                cb = self._draw_checkbox(
                    self.screen, content_x, cb_y, check_sz, on, (120, 150, 90)
                )
                lab = self.small_font.render(
                    "Ground", True, (190, 200, 170) if on else (130, 130, 145)
                )
                self.screen.blit(
                    lab,
                    (
                        content_x + check_sz + 8,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                hit = pygame.Rect(
                    content_x - 2, corner_y + y_off, panel_width - 16, line_h
                )
                self._legend_hits.append((hit, "ground_on", 0))
                self._legend_hits.append((cb.inflate(4, 4), "ground_on", 0))
            elif kind == "ground_relief":
                frac = float(np.clip(self.ground_relief, 0.0, 1.0))
                lab = self.small_font.render("Relief", True, (180, 185, 200))
                self.screen.blit(
                    lab,
                    (
                        content_x + 4,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                track_x = content_x + 52
                track_w = panel_width - 120
                track_h = 8
                track_y = corner_y + y_off + (line_h - track_h) // 2
                track = pygame.Rect(track_x, track_y, track_w, track_h)
                self._ground_relief_rect = track
                pygame.draw.rect(self.screen, (40, 42, 55), track)
                fw = max(0, int(track_w * frac))
                if fw > 0:
                    pygame.draw.rect(
                        self.screen, (120, 150, 90), (track_x, track_y, fw, track_h)
                    )
                pygame.draw.rect(self.screen, (100, 105, 125), track, 1)
                pct_s = self.hint_font.render(
                    f"{int(round(frac * 100))}%", True, (200, 205, 220)
                )
                self.screen.blit(
                    pct_s,
                    (
                        track_x + track_w + 6,
                        corner_y + y_off + (line_h - pct_s.get_height()) // 2,
                    ),
                )
                self._legend_hits.append((track.inflate(6, 10), "ground_relief", 0))
            elif kind == "world_swing":
                on = bool(self.world_swing_on)
                cb_y = corner_y + y_off + (line_h - check_sz) // 2
                cb = self._draw_checkbox(
                    self.screen, content_x, cb_y, check_sz, on, GRAV_ARROW_RGB
                )
                lab = self.small_font.render(
                    "World swing",
                    True,
                    (220, 200, 160) if on else (130, 130, 145),
                )
                self.screen.blit(
                    lab,
                    (
                        content_x + check_sz + 8,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                hit = pygame.Rect(
                    content_x - 2, corner_y + y_off, panel_width - 16, line_h
                )
                self._legend_hits.append((hit, "world_swing", 0))
                self._legend_hits.append((cb.inflate(4, 4), "world_swing", 0))
            elif kind in ("world_swing_amp", "world_swing_hz"):
                if kind == "world_swing_amp":
                    label, key = "Amp", "swing_amp"
                    max_v = float(WORLD_SWING_AMP_MAX)
                    cur = float(self.world_swing_amp)
                    disp = f"{math.degrees(cur):.0f}°"
                    fill_col = (200, 160, 90)
                else:
                    label, key = "Rate", "swing_hz"
                    max_v = float(WORLD_SWING_HZ_MAX)
                    cur = float(self.world_swing_hz)
                    disp = f"{cur:.2f}"
                    fill_col = (180, 150, 100)
                val = float(np.clip(cur / max(max_v, 1e-9), 0.0, 1.0))
                lab = self.small_font.render(label, True, (180, 185, 200))
                self.screen.blit(
                    lab,
                    (
                        content_x + 4,
                        corner_y + y_off + (line_h - lab.get_height()) // 2,
                    ),
                )
                track_x = content_x + 48
                track_w = panel_width - 120
                track_h = 8
                track_y = corner_y + y_off + (line_h - track_h) // 2
                track = pygame.Rect(track_x, track_y, track_w, track_h)
                self._light_slider_rects[key] = track
                pygame.draw.rect(self.screen, (40, 42, 55), track)
                fw = max(0, int(track_w * val))
                if fw > 0:
                    pygame.draw.rect(
                        self.screen, fill_col, (track_x, track_y, fw, track_h)
                    )
                pygame.draw.rect(self.screen, (100, 105, 125), track, 1)
                pct_s = self.hint_font.render(disp, True, (200, 205, 220))
                self.screen.blit(
                    pct_s,
                    (
                        track_x + track_w + 6,
                        corner_y + y_off + (line_h - pct_s.get_height()) // 2,
                    ),
                )
                self._legend_hits.append((track.inflate(6, 10), kind, 0))
            elif kind == "quick":
                gap_cb = 3
                layers_q = (
                    ("soma_col", self.focus_soma_indices, "N", (200, 210, 230)),
                    ("graph_col", self.focus_graph_indices, "G", (120, 150, 190)),
                    ("axon_col", self.focus_axon_indices, "A", (140, 200, 220)),
                    ("dendrite_col", self.focus_dendrite_indices, "D", (190, 150, 210)),
                    ("tract_col", self.focus_tract_indices, "T", (220, 160, 100)),
                    ("basin_col", self.focus_basin_indices, "B", (200, 170, 110)),
                )
                active = self._active_class_indices()
                x_cb = content_x
                cb_y = corner_y + y_off + (line_h - check_sz) // 2
                for kind_l, focus_set, letter, accent in layers_q:
                    inter = focus_set & active
                    all_on = bool(active) and inter == active
                    partial = bool(inter) and not all_on
                    cb = self._draw_checkbox(
                        self.screen,
                        x_cb,
                        cb_y,
                        check_sz,
                        all_on,
                        accent,
                        partial=partial,
                    )
                    self._legend_hits.append((cb.inflate(4, 4), kind_l, 0))
                    x_cb += check_sz + gap_cb
                hint = self.hint_font.render(
                    "column = show/hide · name fills N+A+D", True, (110, 110, 130)
                )
                self.screen.blit(
                    hint,
                    (x_cb + 4, corner_y + y_off + (line_h - hint.get_height()) // 2),
                )
            elif kind == "class":
                _, idx, label, cnt, pct = item
                col = self._class_color(idx)
                row_rect = pygame.Rect(
                    content_x - 2, corner_y + y_off, panel_width - 16, line_h
                )
                any_on = (
                    idx in self.focus_soma_indices
                    or idx in self.focus_graph_indices
                    or idx in self.focus_axon_indices
                    or idx in self.focus_dendrite_indices
                    or idx in self.focus_tract_indices
                    or idx in self.focus_basin_indices
                )
                if any_on:
                    # Static tint — no pulse (avoids Super-class row blink).
                    glow = 32
                    hi = pygame.Surface((row_rect.w, row_rect.h), pygame.SRCALPHA)
                    hi.fill((col[0], col[1], col[2], glow))
                    self.screen.blit(hi, row_rect.topleft)

                cb_y = corner_y + y_off + (line_h - check_sz) // 2
                gap_cb = 3
                layers = (
                    ("soma", self.focus_soma_indices, (200, 210, 230)),
                    ("graph", self.focus_graph_indices, (120, 150, 190)),
                    ("axon", self.focus_axon_indices, (140, 200, 220)),
                    ("dendrite", self.focus_dendrite_indices, (190, 150, 210)),
                    ("tract", self.focus_tract_indices, (220, 160, 100)),
                    ("basin", self.focus_basin_indices, (200, 170, 110)),
                )
                x_cb = content_x
                for kind_l, focus_set, accent in layers:
                    on = idx in focus_set
                    cb = self._draw_checkbox(
                        self.screen, x_cb, cb_y, check_sz, on, accent if on else col
                    )
                    self._legend_hits.append((cb.inflate(4, 4), kind_l, idx))
                    x_cb += check_sz + gap_cb
                sw_x = x_cb + 2
                sw_rect = pygame.Rect(
                    sw_x, corner_y + y_off + (line_h - swatch) // 2, swatch, swatch
                )
                pygame.draw.rect(self.screen, col, sw_rect)
                pygame.draw.rect(
                    self.screen,
                    (160, 160, 180) if any_on else (80, 80, 100),
                    sw_rect,
                    1,
                )
                name_col = (240, 240, 250) if any_on else (190, 190, 200)
                name_s = self.small_font.render(label, True, name_col)
                name_pos = (sw_x + swatch + 6, corner_y + y_off + 2)
                self.screen.blit(name_s, name_pos)
                num_s = self.small_font.render(f"{cnt:,}", True, (170, 200, 230))
                num_x = corner_x + panel_width - num_s.get_width() - 18
                self.screen.blit(num_s, (num_x, corner_y + y_off + 2))
                group_hit = pygame.Rect(
                    sw_x,
                    corner_y + y_off,
                    max(40, num_x - sw_x - 4),
                    line_h,
                )
                self._legend_hits.append((group_hit, "group", idx))
            else:
                # text / hint
                _, text_s, color = item
                surf = self.small_font.render(text_s, True, color)
                self.screen.blit(surf, (content_x + 4, corner_y + y_off + 2))
            y_off += line_h



    def _draw_legend_into_screen(self) -> None:
        """Right dock column (push layout). Collapsed = top-bar Panels only."""
        total = int(self.class_counts.sum()) or 1
        pad = 5
        btn_w, btn_h = 24, 22
        top_limit = self.view_y + 8
        bottom_limit = self.view_y + self.view_h - 8
        self._legend_hits = []
        self.legend_help_rect = None
        self._side_panel_rects = {}
        self._side_panel_toggle_rects = {}
        # Keep top-bar Panels as the master toggle when collapsed.
        panels_chrome = getattr(self, "_chrome_rects", {}).get("Panels")
        self.legend_toggle_rect = panels_chrome
        pulse = self._pulse()

        if not getattr(self, "side_panel_mini", None):
            self.side_panel_mini = {p: False for p in SIDE_PANELS}
        if not getattr(self, "side_panel_scroll", None):
            self.side_panel_scroll = {p: 0 for p in SIDE_PANELS}
        for p in SIDE_PANELS:
            self.side_panel_mini.setdefault(p, False)
            self.side_panel_scroll.setdefault(p, 0)

        # --- Collapsed: no overlay on projection (toggle lives in top bar) ---
        if self.legend_mini:
            self.legend_panel_rect = None
            return

        # --- Expanded dock: fills reserved dock_w column ---
        line_h = 22
        panel_width = max(160, int(self.dock_w) - 2 * pad - 4)
        check_sz = 12
        gap = 2
        strip_h = 24
        dock_head_h = 30
        avail = max(120, bottom_limit - top_limit)
        corner_x = int(self.dock_x) + pad + 2

        built: dict[str, tuple[list[tuple], list[str | None]]] = {
            pid: self._side_panel_rows(pid, total) for pid in SIDE_PANELS
        }
        desired: dict[str, int] = {}
        for pid in SIDE_PANELS:
            if self.side_panel_mini.get(pid, False):
                desired[pid] = strip_h
            else:
                rows, _ = built[pid]
                content_h = max(line_h, len(rows) * line_h + 8)
                desired[pid] = strip_h + min(content_h, 360)
        inner_gaps = gap * (len(SIDE_PANELS) - 1)
        body_des = sum(desired.values()) + inner_gaps
        body_avail = max(80, avail - dock_head_h - pad)
        if body_des > body_avail:
            expandable = [p for p in SIDE_PANELS if not self.side_panel_mini.get(p, False)]
            over = body_des - body_avail
            while over > 0 and expandable:
                tallest = max(expandable, key=lambda p: desired[p])
                room = desired[tallest] - (strip_h + 48)
                if room <= 0:
                    expandable.remove(tallest)
                    continue
                cut = min(room, over)
                desired[tallest] -= cut
                over -= cut

        dock_h = max(avail, dock_head_h + sum(desired.values()) + inner_gaps + pad)
        dock_h = min(dock_h, avail)
        dock_rect = pygame.Rect(
            int(self.dock_x),
            int(self.top_h),
            int(self.dock_w),
            int(self.view_h),
        )
        # Column already filled in _draw; light inner card.
        inner = pygame.Rect(
            corner_x - pad, top_limit - pad, panel_width + pad * 2, dock_h + pad
        )
        pygame.draw.rect(self.screen, (14, 16, 24), inner)
        pygame.draw.rect(self.screen, (70, 85, 110), inner, 1)

        # Dock header
        head_y = top_limit
        title = self.small_font.render("Panels", True, (190, 205, 225))
        self.screen.blit(
            title, (corner_x + 6, head_y + (dock_head_h - title.get_height()) // 2)
        )
        n_open = sum(1 for p in SIDE_PANELS if not self.side_panel_mini.get(p, False))
        meta = self.hint_font.render(
            f"{n_open}/{len(SIDE_PANELS)} open", True, (120, 130, 150)
        )
        self.screen.blit(
            meta,
            (
                corner_x + 6 + title.get_width() + 8,
                head_y + (dock_head_h - meta.get_height()) // 2,
            ),
        )
        btn_x = corner_x - pad + panel_width - btn_w
        btn_y = head_y + (dock_head_h - btn_h) // 2
        pygame.draw.rect(self.screen, (70, 75, 95), (btn_x, btn_y, btn_w, btn_h))
        pygame.draw.rect(self.screen, (140, 150, 170), (btn_x, btn_y, btn_w, btn_h), 1)
        bcx, bcy = btn_x + btn_w // 2, btn_y + btn_h // 2
        # Chevron right = hide dock
        pygame.draw.polygon(
            self.screen,
            (210, 215, 230),
            [(bcx - 5, bcy - 6), (bcx - 5, bcy + 6), (bcx + 5, bcy)],
        )
        self.legend_toggle_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        # Whole header also toggles hide
        head_hit = pygame.Rect(
            corner_x - pad, top_limit - pad, panel_width - btn_w - 4, dock_head_h + pad
        )
        self._legend_hits.append((self.legend_toggle_rect.inflate(2, 2), "side_dock_toggle", 0))
        # Don't use head_hit for dock toggle — title area is reserved; only chevron hides.
        # (Accordion section toggles handle their own strips.)

        self._density_slider_rects = {}
        self._basin_form_rect = None
        self._light_slider_rects = {}
        self._ground_relief_rect = None
        self._prow_targets = []
        content_x = corner_x + 6
        prow_ctx = PanelDrawCtx(
            screen=self.screen,
            host=self,
            content_x=content_x,
            y_off=0,
            line_h=line_h,
            panel_width=panel_width,
            check_sz=check_sz,
            small_font=self.small_font,
            hint_font=self.hint_font,
            draw_checkbox=self._draw_checkbox,
            hits=self._legend_hits,
            prow_targets=self._prow_targets,
        )

        card_accent = {
            "thinning": (110, 150, 190),
            "layers": (130, 160, 200),
            "basins": (200, 170, 110),
            "classes": (150, 140, 180),
            "cloud": (120, 150, 170),
            "yield": (160, 150, 120),
            "lights": (180, 170, 120),
            "blueprint": (140, 180, 140),
            "learn": (150, 190, 160),
            "view": (150, 170, 200),
            "devices": (140, 170, 160),
        }

        cy = top_limit + dock_head_h
        # Thin rule under header
        pygame.draw.line(
            self.screen,
            (50, 55, 70),
            (corner_x, cy - 1),
            (corner_x + panel_width - 8, cy - 1),
            1,
        )

        for pid in SIDE_PANELS:
            rows, block_of = built[pid]
            panel_h = int(desired[pid])
            mini = bool(self.side_panel_mini.get(pid, False))
            # Section strip (no separate floating card — lives inside dock)
            sec_bg = (22, 24, 34) if not mini else (18, 20, 28)
            pygame.draw.rect(
                self.screen,
                sec_bg,
                (corner_x - 2, cy, panel_width - 6, panel_h),
            )
            prect = pygame.Rect(corner_x - 2, cy, panel_width - 6, panel_h)
            self._side_panel_rects[pid] = prect

            title_s = SIDE_PANEL_LABELS.get(pid, pid)
            tcol = (180, 200, 220) if not mini else (130, 140, 160)
            tsurf = self.small_font.render(title_s, True, tcol)
            self.screen.blit(
                tsurf, (corner_x + 6, cy + (strip_h - tsurf.get_height()) // 2)
            )
            # Section collapse chevron
            sbtn_x = corner_x - 2 + panel_width - 6 - btn_w - 2
            sbtn_y = cy + 1
            pygame.draw.rect(self.screen, (55, 58, 72), (sbtn_x, sbtn_y, btn_w, btn_h))
            pygame.draw.rect(self.screen, (110, 115, 135), (sbtn_x, sbtn_y, btn_w, btn_h), 1)
            scx, scy = sbtn_x + btn_w // 2, sbtn_y + btn_h // 2
            if mini:
                pygame.draw.polygon(
                    self.screen,
                    (200, 200, 220),
                    [(scx - 4, scy - 5), (scx - 4, scy + 5), (scx + 5, scy)],
                )
            else:
                pygame.draw.polygon(
                    self.screen,
                    (200, 200, 220),
                    [(scx - 5, scy + 3), (scx + 5, scy + 3), (scx, scy - 5)],
                )
            trect = pygame.Rect(sbtn_x, sbtn_y, btn_w, btn_h)
            self._side_panel_toggle_rects[pid] = trect
            self._legend_hits.append(
                (
                    pygame.Rect(corner_x - 2, cy, panel_width - btn_w - 14, strip_h),
                    "side_panel_toggle",
                    SIDE_PANELS.index(pid),
                )
            )
            self._legend_hits.append(
                (trect.inflate(2, 2), "side_panel_toggle", SIDE_PANELS.index(pid))
            )

            if not mini:
                list_top = cy + strip_h
                list_h = max(line_h, panel_h - strip_h)
                visible = max(1, (list_h - 4) // line_h)
                scroll_max = max(0, len(rows) - visible)
                scroll = min(int(self.side_panel_scroll.get(pid, 0)), scroll_max)
                self.side_panel_scroll[pid] = scroll
                if pid == "filters":
                    self.legend_scroll = scroll

                block_ranges: list[tuple[str, int, int]] = []
                i = 0
                while i < len(block_of):
                    b = block_of[i]
                    if b is None:
                        i += 1
                        continue
                    j = i + 1
                    while j < len(block_of) and block_of[j] == b:
                        j += 1
                    block_ranges.append((b, i, j))
                    i = j
                vis0 = scroll
                vis1 = vis0 + visible
                for bname, a, b in block_ranges:
                    if b <= vis0 or a >= vis1:
                        continue
                    top_i = max(a, vis0)
                    bot_i = min(b, vis1)
                    y0 = list_top + (top_i - vis0) * line_h
                    y1 = list_top + (bot_i - vis0) * line_h
                    self._draw_inset_panel(
                        corner_x,
                        y0 + 1,
                        panel_width - 14,
                        max(2, y1 - y0 - 2),
                        accent=card_accent.get(bname),
                    )

                clip = pygame.Rect(corner_x - 2, list_top - 1, panel_width, list_h + 2)
                prev = self.screen.get_clip()
                self.screen.set_clip(clip.clip(dock_rect))
                self._paint_legend_visible_rows(
                    rows=rows,
                    vis0=vis0,
                    vis1=vis1,
                    corner_x=corner_x,
                    corner_y=list_top,
                    content_x=content_x,
                    panel_width=panel_width,
                    line_h=line_h,
                    check_sz=check_sz,
                    pulse=pulse,
                    prow_ctx=prow_ctx,
                )
                self.screen.set_clip(prev)

                if scroll_max > 0:
                    sb_w = 4
                    sb_x = corner_x - 2 + panel_width - 12
                    sb_y = list_top
                    sb_h = list_h
                    pygame.draw.rect(self.screen, (40, 42, 55), (sb_x, sb_y, sb_w, sb_h))
                    thumb_h = max(10, int(sb_h * visible / max(1, len(rows))))
                    thumb_y = sb_y + int((sb_h - thumb_h) * scroll / scroll_max)
                    pygame.draw.rect(
                        self.screen, (110, 115, 140), (sb_x, thumb_y, sb_w, thumb_h)
                    )

            cy += panel_h + gap

        self.legend_panel_rect = dock_rect
        self._side_panel_rects["dock"] = dock_rect

    def _handle_control_panel_click(self, pos: tuple[int, int]) -> bool:
        px = int(getattr(self, "panel_x", 0))
        if pos[0] < px or pos[0] >= px + self.panel_w:
            return False
        if pos[1] < int(getattr(self, "top_h", 0)) or pos[1] >= int(
            getattr(self, "top_h", 0)
        ) + int(getattr(self, "view_h", self.height)):
            return False
        if self.panel_collapsed:
            return True  # consume clicks on strip; expand via toggle button
        for rect, action in self._panel_hits:
            if not rect.collidepoint(pos):
                continue
            if action == "save_fly":
                self._save_fly()
            elif action == "save_fly_as":
                self._save_fly_as_dialog()
            elif action == "load_fly":
                self._load_fly_dialog()
            elif action == "import_malecns":
                self._import_malecns()
            elif action == "sim_start":
                self._on_sim_start()
            elif action == "sim_pause":
                # Same path as Space when running.
                if self.sim_running or self._fuse_running:
                    self._toggle_sim_run_pause()
            elif action == "sim_step":
                self._on_sim_step()
            elif action == "sim_adult":
                self._enter_adult()
            elif action == "sim_fuse":
                self._on_fuse_start()
            elif action == "sim_reset":
                self.sim_running = False
                self._activity_running = False
                self._fuse_running = False
                self.fly.sim_time = 0
                self.fly.clear_growth_tips()
                self.fly.enter_rising()
                self.fly.clear_activity()
                self._sync_activity_view()
                self.grown_xyz = self.fly.neurite_xyz
                self.grown_cls = self.fly.neurite_cls
                self.grown_rgb = _axon_rgb_for_cls(self.grown_cls)
                # Drop cached tissue / carve overlays so Rising cannot resurrect
                # a prior blueprint blit independent of Super-class.
                self._tissue_blit_key = None
                self._tissue_blit_surf = None
                self._gl_geom_key = None
                self._channel_pairs = []
                self._channel_ids = set()
                self._corridor_topo_pairs = []
                self._corridor_topo_ids = set()
                self._corridor_topo_key = None
                self._set_io_status(
                    f"Reset → rising · genome {int(self.fly.genome_quota.sum()):,} tips · "
                    f"atlas {len(self.fly.recipe_pre):,} edges"
                )
            elif action == "sock_freeze_toggle":
                sock = getattr(self, "motor_socket", None)
                frozen = bool(sock is not None and getattr(sock, "frozen", False))
                handle_action_click(
                    self,
                    Action(
                        "socket_unfreeze" if frozen else "socket_freeze",
                        "",
                    ),
                )
                self._save_ui_state()
            elif action == "sock_reset":
                handle_action_click(self, Action("socket_reset", ""))
                self._save_ui_state()
            else:
                return False
            return True
        return True  # consume click in panel even on empty space

    def _stimulus_target(self) -> tuple[int, str]:
        """Class to light on Run: legend focus, else optic, else first class."""
        if self.focus_soma_indices:
            idx = min(self.focus_soma_indices)
            if 0 <= idx < len(self.class_names):
                return idx, self.class_names[idx]
        if (
            self.focus_hint_class is not None
            and 0 <= self.focus_hint_class < len(self.class_names)
        ):
            return self.focus_hint_class, self.class_names[self.focus_hint_class]
        if "optic" in self.class_names:
            idx = int(self.class_names.index("optic"))
            return idx, "optic"
        if self.class_names:
            return 0, self.class_names[0]
        return 0, ""

    def _tick_adult_closed_loop(self, dt: float) -> dict:
        """One adult frame via ClosedLoop (sense → hop → actuate → learn)."""
        loop = getattr(self, "_closed_loop", None)
        if loop is None:
            from aim_task import AimTask
            from loop import ClosedLoop
            from trainer import Trainer

            task = self._ensure_aim_task()
            tr = getattr(self, "_trainer", None) or Trainer(self, task=task)
            self._trainer = tr
            loop = ClosedLoop(self, task=task, trainer=tr)
            self._closed_loop = loop
        return loop.step(float(dt))

    def _log_learn_console(self, info: dict) -> None:
        """Periodic Learn / Port-NN telemetry to the terminal."""
        port = getattr(self, "sensor_port_nn", None)
        port_on = port is not None and bool(getattr(self, "sensor_port_on", False))
        if port_on:
            try:
                print(port.format_console_line(self), flush=True)
            except Exception:
                print("[ai-brain] (telemetry error)", flush=True)
            return
        if getattr(self, "motor_socket_on", False) and getattr(
            self, "motor_socket", None
        ) is not None:
            sock = self.motor_socket
            aim = float(self.turret_aim_err)
            aim_s = (
                f"{aim * 180.0 / math.pi:.1f}°"
                if math.isfinite(aim)
                else "nan"
            )
            g = [round(float(x), 2) for x in sock.gains]
            levels = [self._turret_site_motor_level(si) for si in range(4)]
            raw = (
                [self._turret_site_motor_level_raw(si) for si in range(4)]
                if hasattr(self, "_turret_site_motor_level_raw")
                else levels
            )
            left = 0.5 * (levels[0] + levels[1])
            right = 0.5 * (levels[2] + levels[3])
            print(
                f"[socket] ε={aim_s}  yaw={left - right:+.3f}  "
                f"{'FROZEN' if sock.frozen else 'train'}  "
                f"lvl={[round(x, 2) for x in levels]}  "
                f"ΔW={float(getattr(self, 'motor_socket_dw_ema', 0.0)):.4f}  "
                f"hop={int(info.get('hops', getattr(self, 'loop_hops', 1) or 1))}",
                flush=True,
            )
            return
        tr = getattr(self, "_trainer", None)
        if tr is not None:
            print(tr.format_report_line(info), flush=True)
            return
        aim = float(self.turret_aim_err)
        if math.isfinite(aim):
            aim_s = f"{aim * 180.0 / math.pi:.1f}°"
        else:
            aim_s = "nan"
        dw = float(self.learn_dw_ema)
        dw_s = f"{dw:.4f}" if math.isfinite(dw) else "nan/inf"
        device = set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
        gains = getattr(self.fly, "contact_gain", {}) or {}
        n_cns = n_br = n_dev = 0
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
            a_d, b_d = int(a) in device, int(b) in device
            if a_d and b_d:
                n_dev += 1
            elif a_d != b_d:
                n_br += 1
            else:
                n_cns += 1
        mean_abs = abs_sum / abs_n if abs_n else 0.0
        warn = ""
        if nan_g or not math.isfinite(dw) or not math.isfinite(aim):
            warn = "  WARN=unstable"
        print(
            f"[learn] t={self.fly.sim_time}  ε={aim_s}  Δw={dw_s}  "
            f"gains={len(gains)} (cns={n_cns} br={n_br} dev={n_dev}"
            + (f" nan={nan_g}" if nan_g else "")
            + f")  |g-1|={mean_abs:.4f}  "
            f"plast={int(self._learn_flash_n)}  "
            f"hop={int(info.get('edges', 0))}  "
            f"lit={int(info.get('lit', 0))}  "
            f"touch={int(info.get('touched', 0))}{warn}",
            flush=True,
        )

    def _merge_learn_chain_flash(
        self,
        *,
        hop_pre: np.ndarray | None,
        hop_post: np.ndarray | None,
    ) -> None:
        """Select engaged neurons/contacts that have real neurite topography.

        Atlas hop edges without a grown axon become soma blooms only — no
        invented soma→soma chords.
        """
        motors = set(self._turret_motor_neuron_targets().keys())
        device = set(self._turret_flat_ids()) | set(self._ocelli_flat_ids())
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def _add(a: int, b: int) -> None:
            key = (int(a), int(b))
            if key in seen:
                return
            seen.add(key)
            pairs.append(key)

        def _add_edge_or_soma(a: int, b: int) -> None:
            a, b = int(a), int(b)
            if a == b:
                _add(a, a)
                return
            # Pack-local wires: soma bloom only (normal axon layer draws the mesh).
            if a in device and b in device:
                _add(a, a)
                _add(b, b)
                return
            if self._contact_has_topo(a, b):
                _add(a, b)
            else:
                # Honest: light the cells, not a fake cable.
                _add(a, a)
                _add(b, b)

        # 1) Plastic edges from the last plasticity step (topo only as cables).
        p_pre = getattr(self, "_learn_flash_pre", None)
        p_post = getattr(self, "_learn_flash_post", None)
        # Snapshot plastic pairs before we overwrite flash arrays.
        plastic_pairs: list[tuple[int, int]] = []
        if p_pre is not None and p_post is not None and len(p_pre) == len(p_post):
            for a, b in zip(p_pre.tolist(), p_post.tolist()):
                plastic_pairs.append((int(a), int(b)))
        # Clear working set — rebuild from topo-aware sources.
        pairs.clear()
        seen.clear()
        for a, b in plastic_pairs:
            _add_edge_or_soma(a, b)

        # Path_edges are the scope map (indigo bed), not plasticity flash.
        # Only real hop / plastic / grown contacts light cyan.
        path_edges = self._scope_path_edge_set()

        # 2) Grown Fuse contacts that are hop carriers / co-active (always have topo).
        g_pre, g_post = self.fly.grown_contact_edges()
        grown_set = set(zip(g_pre.tolist(), g_post.tolist())) if len(g_pre) else set()

        if (
            hop_pre is not None
            and hop_post is not None
            and len(hop_pre) == len(hop_post)
            and len(hop_pre)
        ):
            ranked: list[tuple[int, int, int]] = []
            for a, b in zip(hop_pre.tolist(), hop_post.tolist()):
                a, b = int(a), int(b)
                score = 0
                if (a, b) in path_edges or (b, a) in path_edges:
                    score += 6
                if (a, b) in grown_set:
                    score += 5
                if b in motors or a in motors:
                    score += 3
                if (a in device) != (b in device):
                    score += 2
                if a not in device and b not in device:
                    score += 2
                ranked.append((score, a, b))
            ranked.sort(key=lambda t: -t[0])
            for score, a, b in ranked[:400]:
                if score <= 0 and len(pairs) > 60:
                    break
                _add_edge_or_soma(a, b)

        # 3) Living wave on grown contacts (real cables) + hop carriers as somata.
        act = getattr(self, "soma_activity", None)
        ids = self.fly.soma_ids
        # Plastic + hop already give a rich flash — skip heavy CNS scans.
        rich_flash = len(pairs) >= 100
        if (
            not rich_flash
            and act is not None
            and len(ids) == len(act)
            and len(act) > 0
        ):
            contiguous = (
                hasattr(self.fly, "_activity_ids_contiguous")
                and self.fly._activity_ids_contiguous()
            )
            thr_pre, thr_post = 0.12, 0.08
            if contiguous:
                # id == index: no 139k lut.
                if len(g_pre):
                    gpa = g_pre.astype(np.int64, copy=False)
                    gpb = g_post.astype(np.int64, copy=False)
                    n_max = len(act)
                    ok = (
                        (gpa >= 0)
                        & (gpb >= 0)
                        & (gpa < n_max)
                        & (gpb < n_max)
                    )
                    gpa, gpb = gpa[ok], gpb[ok]
                    pa = act[gpa]
                    pb = act[gpb]
                    live = (pa >= thr_pre) & (pb >= thr_post)
                    for a, b in zip(gpa[live].tolist(), gpb[live].tolist()):
                        a, b = int(a), int(b)
                        if a in device and b in device:
                            continue
                        _add(a, b)

                    # Reverse BFS from motors through co-active grown tissue.
                    active = act > 0.08
                    rev: dict[int, list[int]] = {}
                    both = active[gpa] & active[gpb]
                    for a, b in zip(gpa[both].tolist(), gpb[both].tolist()):
                        rev.setdefault(int(b), []).append(int(a))
                    frontier = [m for m in motors if 0 <= int(m) < n_max]
                    visited = set(frontier)
                    for _depth in range(5):
                        nxt: list[int] = []
                        for node in frontier:
                            for src in rev.get(int(node), []):
                                if src in visited:
                                    continue
                                visited.add(src)
                                _add(src, node)
                                nxt.append(src)
                        frontier = nxt
                        if not frontier:
                            break

                for mid in motors:
                    mid = int(mid)
                    if 0 <= mid < len(act) and float(act[mid]) > 0.05:
                        _add(mid, mid)

                # Lit CNS somata — blooms only.
                hot_idx = np.flatnonzero(act >= 0.18)
                if len(hot_idx):
                    # Drop device terminals.
                    if device:
                        keep = np.fromiter(
                            (int(i) not in device for i in hot_idx.tolist()),
                            dtype=bool,
                            count=len(hot_idx),
                        )
                        hot_idx = hot_idx[keep]
                    if len(hot_idx) > 420:
                        order = np.argsort(-act[hot_idx])[:420]
                        hot_idx = hot_idx[order]
                    for nid in hot_idx.tolist():
                        _add(int(nid), int(nid))
            else:
                lut = {int(i): k for k, i in enumerate(ids.tolist())}
                for a, b in zip(g_pre.tolist(), g_post.tolist()):
                    a, b = int(a), int(b)
                    if a in device and b in device:
                        continue
                    ia, ib = lut.get(a), lut.get(b)
                    if ia is None or ib is None:
                        continue
                    pa, pb = float(act[ia]), float(act[ib])
                    if pa < thr_pre or pb < thr_post:
                        continue
                    _add(a, b)

                rev = {}
                active = {
                    int(i) for i, k in lut.items() if float(act[k]) > 0.08
                }
                for a, b in zip(g_pre.tolist(), g_post.tolist()):
                    a, b = int(a), int(b)
                    if a in active and b in active:
                        rev.setdefault(b, []).append(a)
                frontier = [m for m in motors if m in lut]
                visited = set(frontier)
                for _depth in range(5):
                    nxt = []
                    for node in frontier:
                        for src in rev.get(int(node), []):
                            if src in visited:
                                continue
                            visited.add(src)
                            _add(src, node)
                            nxt.append(src)
                    frontier = nxt
                    if not frontier:
                        break

                for mid in motors:
                    ia = lut.get(int(mid))
                    if ia is not None and float(act[ia]) > 0.05:
                        _add(mid, mid)

                cns_hot: list[tuple[float, int]] = []
                for nid, k in lut.items():
                    if int(nid) in device:
                        continue
                    a = float(act[k])
                    if a >= 0.18:
                        cns_hot.append((a, int(nid)))
                cns_hot.sort(key=lambda t: -t[0])
                for _, nid in cns_hot[: min(420, len(cns_hot))]:
                    _add(nid, nid)

        if not pairs:
            self._learn_flash_amp *= 0.88
            return

        def _prio(ab: tuple[int, int]) -> int:
            a, b = ab
            if a == b:
                return 0
            if (a, b) in grown_set:
                return 4
            if (a in device) != (b in device):
                return 3
            if a not in device and b not in device:
                return 2
            return 1

        pairs.sort(key=_prio, reverse=True)
        self._learn_flash_pre = np.asarray([p[0] for p in pairs], dtype=np.int32)
        self._learn_flash_post = np.asarray([p[1] for p in pairs], dtype=np.int32)
        self._learn_flash_n = int(sum(1 for a, b in pairs if a != b))
        self._learn_flash_amp = max(float(self._learn_flash_amp), 0.85)
        nodes = {int(a) for a, _ in pairs} | {int(b) for _, b in pairs}
        self._learn_hot_ids = nodes

    def _toggle_sim_run_pause(self) -> None:
        """Space / Run↔Pause: start if idle, pause if running (incl. Fuse)."""
        if self.sim_running or self._fuse_running:
            self.sim_running = False
            self._activity_running = False
            if self._fuse_running:
                self._fuse_running = False
                self.fly.enter_adult(abandon_open=False)
                self._set_io_status("Fuse paused · adult")
            else:
                self._set_io_status(f"Paused · {self.fly.phase}")
            return
        self._on_sim_start()

    def _on_sim_start(self) -> None:
        """Rising → arm growth; adult → closed-loop Run (devices + hop)."""
        if self.fly.is_adult() or self._settle_if_idle():
            self._start_stimulus_run()
            return
        n = self._arm_growth_channel()
        if n > 0:
            self.fly.enter_rising()
            self._activity_running = False
            self.sim_running = True
            q = (
                int(self.fly.genome_quota.sum())
                if self.fly.genome_quota.size
                else n
            )
            self._set_io_status(
                f"Rising · genome armed {n:,} tips (quota {q:,})"
            )
        else:
            self.sim_running = False
            self._activity_running = False

    def _on_sim_step(self) -> None:
        self.sim_running = False
        self._activity_running = False
        if self.fly.is_adult() or self._settle_if_idle():
            self.fly.sim_tick()
            info = self._tick_adult_closed_loop(1.0 / float(FPS))
            if info["edges"] == 0:
                self._set_io_status(
                    f"Step · t={self.fly.sim_time} · no contacts · "
                    f"max {info['max']:.2f}"
                )
            else:
                self._set_io_status(
                    f"Step · t={self.fly.sim_time} · lit {info['lit']:,} · "
                    f"max {info['max']:.2f}"
                )
            return
        if self._arm_growth_channel() > 0:
            self.fly.enter_rising()
            self.fly.sim_tick()
            self._settle_if_idle()

    def _start_stimulus_run(self) -> None:
        """Adult Run: device clamps (or class wave), then closed loop each frame."""
        if not self.fly.neurons:
            self._set_io_status("No fly loaded")
            return
        _idx, name = self._stimulus_target()
        devices_on = bool(
            self.ocelli_on
            or self.turret_on
            or getattr(self, "eye_left_on", False)
            or getattr(self, "eye_right_on", False)
            or getattr(self, "jo_left_on", False)
            or getattr(self, "jo_right_on", False)
            or any(d.enabled for d in compound_eyes.iter_eye_devices(self))
            or any(d.enabled for d in johnston_organ.iter_jo_devices(self))
        )
        if not devices_on and not name:
            self._set_io_status("No class to stimulate")
            return
        # Adult hop needs the atlas recipe (CNS tracts through the neck).
        self._ensure_atlas_recipe()
        self.fly.clear_activity()
        if devices_on:
            # Seed only instrument clamps — no whole-class flood through CNS.
            n = 0
            if self.ocelli_on:
                n += int(self._seed_ocelli_activity())
            n += int(self._seed_eye_activity())
            n += int(self._seed_jo_activity())
            if self.turret_on:
                n += int(self._apply_turret_feedback() or 0)
            name = "devices"
        else:
            n = self.fly.stimulate_set(name, 1.0)
        # Learning flashes follow Super-class (empty N/A = no CNS glow).
        if self.learn_on and self.turret_on:
            self.learn_dw_ema = 0.0
            self._learn_flash_amp = 0.0
            self._learn_flash_n = 0
            self._learn_status_i = 0
            self._learn_prev_err = float("nan")
            # Sync motor socket from rule — keep learned W / frozen across Run.
            # Wipe only via Learn → Reset socket W (or Train again unfreezes).
            rule = getattr(self, "rule", None) or _DEFAULT_RULE
            self.motor_socket_on = bool(getattr(rule, "motor_socket_on", False))
            if getattr(self, "motor_socket", None) is None:
                from motor_socket import MotorSocket

                self.motor_socket = MotorSocket(
                    lr=float(getattr(rule, "motor_socket_lr", 0.45) or 0.45)
                )
            else:
                self.motor_socket.lr = float(
                    getattr(rule, "motor_socket_lr", self.motor_socket.lr) or 0.35
                )
            self.motor_socket_dw_ema = 0.0
            sock_frozen = bool(getattr(self.motor_socket, "frozen", False))
            if not sock_frozen:
                self._learn_err_start_deg = float("nan")
                self._learn_err_freeze_deg = float("nan")
                self._track_reset()
                self._learn_progress_reset()
            if self.motor_socket_on and not sock_frozen:
                # Socket train mode: wipe graph Δw so meters/overlays match port-W.
                self.fly.contact_gain = {}
                if hasattr(self, "_refresh_channel_bed"):
                    self._refresh_channel_bed()
                self._learn_flash_amp = 0.0
                self._learn_flash_n = 0
                self._learn_flash_pre = np.zeros(0, dtype=np.int32)
                self._learn_flash_post = np.zeros(0, dtype=np.int32)
                self.learn_n_edges = 0
                self.learn_dw_ema = 0.0
            # CNS mode: keep contact_gain; Trainer will carve sensor→motor.
            tag = (
                "FROZEN W kept"
                if sock_frozen
                else ("socket TRAIN" if self.motor_socket_on else "CNS carve")
            )
            print(
                f"[socket] {tag} · lr={self.motor_socket.lr:.2f} "
                f"lateral={self.motor_socket.lateral_mix:.2f} "
                f"hop_clip={bool(self.hop_clip)}",
                flush=True,
            )
            self._save_ui_state()
            # Drop non-finite gains from prior runaway sessions.
            gains = getattr(self.fly, "contact_gain", {}) or {}
            bad = [k for k, w in gains.items() if not math.isfinite(float(w))]
            for k in bad:
                gains.pop(k, None)
            reach = self._probe_ocelli_motor_reach(hops=10)
            print(
                f"[learn] REACH ocelli→motors hit={reach['hit']}/{reach['motors']} "
                f"best={reach['best']:.3f} ok={reach['ok']} "
                f"(teacher={rule.teacher_mix} hebb={rule.hebb} "
                f"socket={self.motor_socket_on})",
                flush=True,
            )
            if not reach["ok"]:
                self._set_io_status(
                    f"Learn blocked · path weak "
                    f"({reach['hit']}/{reach['motors']} motors) · Fuse CNS→motor",
                    redraw=True,
                )
                print(
                    "[learn] WARN path not reachable — Fuse with CNS→motor efferent, "
                    "then Run again",
                    flush=True,
                )
            else:
                # Learning needs a real sample (corridor). Build/apply scope so
                # carve + path_credit are not fighting an empty mask / off-path erase.
                cor_n = 0
                try:
                    if not (
                        getattr(self, "scope_on", False)
                        and getattr(self, "_training_scope", None) is not None
                        and getattr(self._training_scope, "ok", False)
                    ):
                        ok_scope = bool(self.apply_training_scope())
                    else:
                        ok_scope = True
                    scope = getattr(self, "_training_scope", None)
                    cor_n = (
                        len(getattr(scope, "corridor", ()) or ())
                        if scope is not None
                        else 0
                    )
                    print(
                        f"[learn] SCOPE {'ok' if ok_scope else 'fail'} "
                        f"corridor={cor_n:,} hop_clip={bool(self.hop_clip)}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[learn] SCOPE failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
            # Re-seed after probe cleared activity.
            self.fly.clear_activity()
            n = 0
            if self.ocelli_on:
                n += int(self._seed_ocelli_activity())
            n += int(self._seed_eye_activity())
            n += int(self._seed_jo_activity())
            if self.turret_on:
                n += int(self._apply_turret_feedback() or 0)
        # First beat so partners light before the closed loop.
        self._activity_running = True
        self.sim_running = True
        info = self.fly.tick_activity(hop_gain=0.9, decay=0.97)
        self._sync_activity_view()
        self._tick_turret(1.0 / float(FPS))
        if self.turret_on:
            self._apply_turret_feedback()
        # Teacher mix is 0 by default — left as optional hook.
        rule = getattr(self, "rule", None) or _DEFAULT_RULE
        if self.learn_on and self.turret_on and float(rule.teacher_mix) > 0.01:
            self._apply_aim_teacher_mix()
            self._sync_activity_view()
        n_edges = int(info["edges"])
        n_grown = len(self.fly.grown_contact_edges()[0])
        devices = []
        if self.ocelli_on:
            devices.append("ocelli")
        if self.turret_on:
            devices.append("turret")
        loop = " · closed loop" if devices else " · wave"
        learn = " · learn" if (self.learn_on and self.turret_on) else ""
        dev = f" · {'+'.join(devices)}" if devices else ""
        hop = f" · hop {n_edges:,} (grown {n_grown:,})"
        if n_edges == 0:
            self._set_io_status(
                f"Run · {name} ({n:,}) · no hop graph — local fade{dev}{loop}{learn}"
            )
        else:
            self._set_io_status(
                f"Run · {name} ({n:,}){hop}{dev}{loop}{learn}"
            )
        if self.learn_on and self.turret_on:
            print(
                f"[learn] RUN start · hop={n_edges:,} grown={n_grown:,} "
                f"seed={n:,} · teacher={rule.teacher_mix} "
                f"lr={rule.lr} hebb={rule.hebb} decay={rule.w_decay}",
                flush=True,
            )

    def _enter_adult(self) -> None:
        """Freeze growth and switch to use-phase (even if Yield < 100%)."""
        self.sim_running = False
        self._activity_running = False
        self._fuse_running = False
        ns = self.fly.neurite_stats()
        open_n = int(ns.get("active", 0))
        self.fly.enter_adult(abandon_open=True)
        self._ensure_atlas_recipe()
        yld = self._recipe_yield()
        head, _ = self._yield_headline(yld)
        n_edges = len(self.fly.grown_contact_edges()[0])
        n_hop = len(self.fly.activity_graph_edges()[0])
        self._set_io_status(
            f"Adult · Fuse {n_edges:,} · hop {n_hop:,} · {head}"
            + (f" · closed {open_n:,} open tips" if open_n else "")
            + " · Run = closed loop",
            redraw=True,
        )

    def _prepare_native_dialog(self) -> None:
        """Let the OS file dialog receive mouse/keyboard (pygame often grabs them)."""
        try:
            pygame.event.set_grab(False)
        except pygame.error:
            pass
        pygame.mouse.set_visible(True)
        pygame.event.clear()
        pygame.display.flip()

    def _resume_after_dialog(self) -> None:
        """Unstick SDL after a native dialog closes."""
        try:
            pygame.event.set_grab(False)
        except pygame.error:
            pass
        pygame.mouse.set_visible(True)
        w, h = self.screen.get_size()
        self._reopen_display(w, h)
        pygame.event.clear()
        for _ in range(10):
            pygame.event.pump()
        pygame.display.flip()

    def _run_file_dialog_cmd(self, cmd: list[str]) -> str | None:
        """Run zenity/kdialog while pumping pygame so the UI does not deadlock."""
        import os
        import subprocess
        import time

        env = os.environ.copy()
        # Prefer classic GTK chooser; portal + pygame often hangs after OK.
        env.setdefault("GTK_USE_PORTAL", "0")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except OSError:
            return None

        while proc.poll() is None:
            pygame.event.pump()
            time.sleep(0.03)

        try:
            stdout, _stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return None
        if proc.returncode != 0:
            return None
        chosen = (stdout or "").strip()
        return chosen or None

    def _set_io_status(self, text: str, *, redraw: bool = False) -> None:
        self._fly_io_status = text
        if redraw:
            self._draw()

    def _pick_fly_path(self, *, save: bool) -> Path | None:
        """Native file dialog (zenity/kdialog), tkinter only as last resort."""
        import shutil

        FLIES_DIR.mkdir(parents=True, exist_ok=True)
        initial = str(FLIES_DIR.resolve())
        cur = self._current_fly_path() if save else None
        save_seed = str(cur) if cur is not None else str(FLIES_DIR / "untitled.fly")
        save_title = "Save As" if save else "Load fly"
        self._prepare_native_dialog()

        if shutil.which("zenity"):
            filename = save_seed if save else f"{initial}/"
            cmd = [
                "zenity",
                "--file-selection",
                f"--title={save_title}",
                f"--filename={filename}",
            ]
            if save:
                cmd.append("--save")
            chosen = self._run_file_dialog_cmd(cmd)
            self._resume_after_dialog()
            if chosen:
                path = Path(chosen)
                if save and path.suffix.lower() != ".fly":
                    path = path.with_suffix(".fly")
                return path
            return None

        if shutil.which("kdialog"):
            if save:
                cmd = [
                    "kdialog",
                    "--getsavefilename",
                    save_seed,
                    "*.fly|Fly model\n*|All files",
                ]
            else:
                cmd = [
                    "kdialog",
                    "--getopenfilename",
                    initial,
                    "*.fly|Fly model\n*|All files",
                ]
            chosen = self._run_file_dialog_cmd(cmd)
            self._resume_after_dialog()
            if chosen:
                path = Path(chosen)
                if save and path.suffix.lower() != ".fly":
                    path = path.with_suffix(".fly")
                return path
            return None

        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            self._resume_after_dialog()
            fallback = FLIES_DIR / "current.fly"
            return fallback if save or fallback.is_file() else None

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        kwargs = {
            "defaultextension": ".fly",
            "filetypes": [("Fly model", "*.fly"), ("All files", "*.*")],
            "initialdir": str(Path(save_seed).parent) if save else initial,
        }
        if save:
            kwargs["initialfile"] = Path(save_seed).name
            path = filedialog.asksaveasfilename(title="Save As", **kwargs)
        else:
            path = filedialog.askopenfilename(title="Load fly", **kwargs)
        root.destroy()
        self._resume_after_dialog()
        return Path(path) if path else None

    def _current_fly_path(self) -> Path | None:
        """Path of the last loaded/saved .fly, or None (MaleCNS / unknown)."""
        last = getattr(self, "_last_model", None)
        if not isinstance(last, str) or not last.strip() or last.strip() == "malecns":
            return None
        path = Path(last.strip())
        if path.suffix.lower() != ".fly":
            return None
        return path

    def _write_fly_to(self, path: Path) -> None:
        """Serialize current fly + shell devices to ``path`` and remember it."""
        if path.suffix.lower() != ".fly":
            path = path.with_suffix(".fly")
        sock_meta = self._motor_socket_snapshot()
        extra = {"shell_devices": self._device_snapshot()}
        if sock_meta is not None:
            extra["motor_socket"] = sock_meta
        save_fly(self.fly, path, extra_meta=extra)
        self._save_ui_state()
        self._remember_model(str(path.resolve()))
        oc = (
            sum(1 for c in self._ocelli_connected_mask() if c)
            if self.ocelli_on
            else 0
        )
        tu = (
            sum(1 for c in self._turret_connected_mask() if c)
            if self.turret_on
            else 0
        )
        fuse = ""
        if self.ocelli_on or self.turret_on:
            fuse = f" · fused {oc}/3+{tu}/{int(TURRET_SITES)}"
        sock_note = ""
        if sock_meta is not None:
            sock_note = (
                " · socket FROZEN" if sock_meta.get("frozen") else " · socket W"
            )
        self._set_io_status(
            f"Saved {path.name}{fuse}{sock_note}",
            redraw=True,
        )

    def _save_fly(self) -> None:
        """Overwrite current .fly; if none, open Save As."""
        if not self.fly.neurons:
            self._set_io_status("Nothing to save — import or load first", redraw=True)
            return
        path = self._current_fly_path()
        if path is None:
            self._save_fly_as_dialog()
            return
        self._set_io_status(f"Saving {path.name}…", redraw=True)
        try:
            self._write_fly_to(path)
        except OSError as exc:
            self._set_io_status(f"Save failed: {exc}", redraw=True)

    def _save_fly_as_dialog(self) -> None:
        """Always pick a destination path (Save As)."""
        if not self.fly.neurons:
            self._set_io_status("Nothing to save — import or load first", redraw=True)
            return
        path = self._pick_fly_path(save=True)
        if path is None:
            self._set_io_status("Save As cancelled", redraw=True)
            return
        self._set_io_status(f"Saving {path.name}…", redraw=True)
        try:
            self._write_fly_to(path)
        except OSError as exc:
            self._set_io_status(f"Save As failed: {exc}", redraw=True)

    def _load_fly_dialog(self) -> None:
        path = self._pick_fly_path(save=False)
        if path is None:
            self._set_io_status("Load cancelled", redraw=True)
            return
        self._set_io_status(f"Loading {path.name}…", redraw=True)
        try:
            loaded = load_fly(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._set_io_status(f"Load failed: {exc}", redraw=True)
            return
        self._adopt_fly(loaded, f"Loaded {path.name}", model_path=path)
        self._draw()

    def _draw_control_panel(self) -> None:
        """Left column — fly transport + layers + file, or collapsed strip."""
        self._panel_hits = []
        self.panel_toggle_rect = None
        btn_w, btn_h = 22, 22
        ox = int(getattr(self, "panel_x", 0))
        oy = int(getattr(self, "top_h", 0))

        if self.panel_collapsed:
            btn_x = ox + (self.panel_w - btn_w) // 2
            btn_y = oy + 12
            pygame.draw.rect(self.screen, (80, 80, 100), (btn_x, btn_y, btn_w, btn_h))
            pygame.draw.rect(self.screen, (140, 140, 160), (btn_x, btn_y, btn_w, btn_h), 1)
            cx, cy = btn_x + btn_w // 2, btn_y + btn_h // 2
            pygame.draw.polygon(
                self.screen,
                (200, 200, 220),
                [(cx - 4, cy - 6), (cx - 4, cy + 6), (cx + 5, cy)],
            )
            self.panel_toggle_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
            return

        margin = 12
        pad = 8
        line_h = 24
        action_h = 36
        gap = 8
        panel_width = self.panel_w - 2 * margin
        inner_w = panel_width - 2 * pad
        corner_x = ox + margin
        corner_y = oy + margin
        content_x = corner_x + pad

        dial_h = 90
        body_h = (
            line_h  # fly time header
            + dial_h
            + line_h  # state
            + 4 * (action_h + gap)  # Start/Pause · Step/Adult · Fuse/Reset · Sock
            + line_h  # file
            + 3 * (action_h + gap)
            + (line_h if self._fly_io_status else 0)
        )

        panel_h = line_h + body_h + 2 * pad + 8
        panel_h = min(panel_h, max(200, self.view_h - 2 * margin))
        bg = pygame.Surface((panel_width, panel_h))
        bg.set_alpha(200)
        bg.fill((20, 20, 30))
        self.screen.blit(bg, (corner_x, corner_y))

        x = content_x
        y = corner_y + pad
        right = corner_x + panel_width - pad

        def row_text(text: str, color: tuple[int, int, int], yy: int) -> int:
            t = self.small_font.render(text, True, color)
            self.screen.blit(t, (x, yy + (line_h - t.get_height()) // 2))
            return yy + line_h

        def full_btn(
            label: str,
            yy: int,
            action: str,
            *,
            fill: tuple[int, int, int],
            border: tuple[int, int, int],
            text_color: tuple[int, int, int],
        ) -> int:
            rect = pygame.Rect(x, yy, inner_w, action_h)
            pygame.draw.rect(self.screen, fill, rect)
            pygame.draw.rect(self.screen, border, rect, 1)
            t = self.small_font.render(label, True, text_color)
            self.screen.blit(
                t,
                (
                    rect.x + (rect.w - t.get_width()) // 2,
                    rect.y + (rect.h - t.get_height()) // 2,
                ),
            )
            self._panel_hits.append((rect, action))
            return yy + action_h + gap

        title = self.small_font.render("FlyWire", True, (180, 200, 220))
        self.screen.blit(title, (x, y + (line_h - title.get_height()) // 2))
        toggle_x = right - btn_w
        toggle_y = y + (line_h - btn_h) // 2
        pygame.draw.rect(self.screen, (80, 80, 100), (toggle_x, toggle_y, btn_w, btn_h))
        pygame.draw.rect(self.screen, (140, 140, 160), (toggle_x, toggle_y, btn_w, btn_h), 1)
        tcx, tcy = toggle_x + btn_w // 2, toggle_y + btn_h // 2
        pygame.draw.polygon(
            self.screen,
            (200, 200, 220),
            [(tcx + 4, tcy - 6), (tcx + 4, tcy + 6), (tcx - 5, tcy)],
        )
        self.panel_toggle_rect = pygame.Rect(toggle_x, toggle_y, btn_w, btn_h)
        y += line_h

        half_w = (inner_w - gap) // 2

        def half_btn(
            label: str,
            bx: int,
            by: int,
            action: str,
            *,
            fill: tuple[int, int, int],
            border: tuple[int, int, int],
            text_color: tuple[int, int, int],
            enabled: bool = True,
        ) -> None:
            rect = pygame.Rect(bx, by, half_w, action_h)
            if not enabled:
                fill, border, text_color = (
                    (35, 35, 42),
                    (70, 70, 80),
                    (100, 100, 110),
                )
            pygame.draw.rect(self.screen, fill, rect)
            pygame.draw.rect(self.screen, border, rect, 1)
            t = self.small_font.render(label, True, text_color)
            self.screen.blit(
                t,
                (
                    rect.x + (rect.w - t.get_width()) // 2,
                    rect.y + (rect.h - t.get_height()) // 2,
                ),
            )
            if enabled:
                self._panel_hits.append((rect, action))

        y = row_text("── Fly time ──", (140, 140, 160), y)
        dial_size = min(72, inner_w - 24)
        y = self._draw_fly_clock(x, y, inner_w, dial_size)
        if getattr(self, "_fuse_running", False):
            phase_col = (230, 170, 120)
            phase_txt = "Phase  FUSE · device tips only"
        elif self.fly.is_adult():
            if self.sim_running and self._activity_running:
                phase_col = (210, 180, 255)
                if self.ocelli_on or self.turret_on:
                    if self.learn_on and self.turret_on:
                        phase_txt = "Phase  ADULT · learn+closed"
                    else:
                        phase_txt = "Phase  ADULT · closed loop"
                else:
                    phase_txt = "Phase  ADULT · wave"
            else:
                phase_col = (180, 200, 255)
                phase_txt = "Phase  ADULT · Run = closed loop"
        else:
            phase_col = (160, 210, 170) if self.sim_running else (180, 160, 120)
            phase_txt = (
                "Phase  RISING · run" if self.sim_running else "Phase  RISING · paused"
            )
        y = row_text(phase_txt, phase_col, y)

        adult = self.fly.is_adult()
        half_btn(
            "Run" if adult and not self._fuse_running else "Start",
            x,
            y,
            "sim_start",
            fill=(55, 45, 70) if adult and not self._fuse_running else (36, 58, 48),
            border=(180, 140, 220) if adult and not self._fuse_running else (100, 180, 130),
            text_color=(230, 200, 255) if adult and not self._fuse_running else (180, 230, 190),
            enabled=not self.sim_running
            and (self.fly.is_rising() or adult)
            and not self._fuse_running,
        )
        half_btn(
            "Pause",
            x + half_w + gap,
            y,
            "sim_pause",
            fill=(55, 45, 40),
            border=(200, 140, 100),
            text_color=(240, 200, 170),
            enabled=self.sim_running,
        )
        y += action_h + gap
        half_btn(
            "Step +1s",
            x,
            y,
            "sim_step",
            fill=(40, 50, 65),
            border=(120, 160, 200),
            text_color=(180, 210, 240),
        )
        half_btn(
            "→ Adult",
            x + half_w + gap,
            y,
            "sim_adult",
            fill=(40, 48, 70),
            border=(140, 160, 220),
            text_color=(190, 205, 255),
            enabled=self.fly.is_rising() or self._fuse_running,
        )
        y += action_h + gap
        need_fuse = self._devices_need_fuse()
        half_btn(
            "Fuse",
            x,
            y,
            "sim_fuse",
            fill=(70, 42, 36) if need_fuse or self._fuse_running else (35, 35, 42),
            border=(230, 140, 110) if need_fuse or self._fuse_running else (70, 70, 80),
            text_color=(255, 200, 170) if need_fuse or self._fuse_running else (100, 100, 110),
            enabled=(need_fuse or self._fuse_running)
            and not self._activity_running
            and bool(self.fly.neurons),
        )
        half_btn(
            "Reset → rising",
            x + half_w + gap,
            y,
            "sim_reset",
            fill=(45, 45, 58),
            border=(130, 130, 155),
            text_color=(200, 200, 220),
        )
        y += action_h + gap
        sock_on = bool(getattr(self, "motor_socket_on", False))
        sock = getattr(self, "motor_socket", None)
        frozen = bool(sock is not None and getattr(sock, "frozen", False))
        half_btn(
            "Sock freeze" if not frozen else "Sock train",
            x,
            y,
            "sock_freeze_toggle",
            fill=(45, 60, 48) if sock_on else (35, 35, 42),
            border=(160, 220, 150) if sock_on else (70, 70, 80),
            text_color=(200, 240, 190) if sock_on else (100, 100, 110),
            enabled=bool(self.fly.is_adult()),
        )
        half_btn(
            "Sock reset",
            x + half_w + gap,
            y,
            "sock_reset",
            fill=(55, 42, 40) if sock_on else (35, 35, 42),
            border=(220, 160, 130) if sock_on else (70, 70, 80),
            text_color=(240, 200, 170) if sock_on else (100, 100, 110),
            enabled=bool(self.fly.is_adult()),
        )
        y += action_h + gap

        y = row_text("── File ──", (140, 140, 160), y)
        y = full_btn(
            "Import MaleCNS",
            y,
            "import_malecns",
            fill=(50, 48, 40),
            border=(200, 170, 110),
            text_color=(230, 210, 160),
        )
        half_btn(
            "Save",
            x,
            y,
            "save_fly",
            fill=(40, 50, 65),
            border=(120, 160, 200),
            text_color=(180, 210, 240),
        )
        half_btn(
            "Save As",
            x + half_w + gap,
            y,
            "save_fly_as",
            fill=(40, 50, 65),
            border=(120, 160, 200),
            text_color=(180, 210, 240),
        )
        y += action_h + gap
        y = full_btn(
            "Load .fly",
            y,
            "load_fly",
            fill=(40, 50, 65),
            border=(120, 160, 200),
            text_color=(180, 210, 240),
        )
        if self._fly_io_status:
            y = row_text(self._fly_io_status, (160, 170, 140), y)

    def _draw_fly_clock(
        self, x: int, y: int, width: int, size: int
    ) -> int:
        """Analog dial for fly lifetime. 1 revolution = FLY_SECONDS_PER_REVOLUTION."""
        cx = x + width // 2
        cy = y + size // 2
        r = size // 2 - 4
        pygame.draw.circle(self.screen, (32, 34, 44), (cx, cy), r)
        pygame.draw.circle(self.screen, (90, 100, 120), (cx, cy), r, 2)
        # hour marks (12 ticks)
        for i in range(12):
            ang = -math.pi / 2 + i * (math.tau / 12)
            c, s = math.cos(ang), math.sin(ang)
            inner = r - (10 if i % 3 == 0 else 6)
            pygame.draw.line(
                self.screen,
                (150, 160, 180) if i % 3 == 0 else (80, 85, 100),
                (cx + int(inner * c), cy + int(inner * s)),
                (cx + int((r - 2) * c), cy + int((r - 2) * s)),
                2 if i % 3 == 0 else 1,
            )
        t = int(self.fly.sim_time)
        # Hand: 0 at 12 o'clock, clockwise, one turn per 60 fly-seconds
        frac = (t % FLY_SECONDS_PER_REVOLUTION) / float(FLY_SECONDS_PER_REVOLUTION)
        ang = -math.pi / 2 + frac * math.tau
        hand_len = r - 14
        hx = cx + int(hand_len * math.cos(ang))
        hy = cy + int(hand_len * math.sin(ang))
        pygame.draw.line(self.screen, (220, 180, 100), (cx, cy), (hx, hy), 3)
        pygame.draw.circle(self.screen, (230, 200, 140), (cx, cy), 4)
        label = self.small_font.render(f"{t:,} fly-s", True, (200, 210, 230))
        ly = y + size + 2
        self.screen.blit(label, (cx - label.get_width() // 2, ly))
        sub = self.hint_font.render(
            f"1 tick = 1 s · dial {FLY_SECONDS_PER_REVOLUTION}s/turn",
            True,
            (120, 130, 150),
        )
        self.screen.blit(sub, (cx - sub.get_width() // 2, ly + label.get_height() + 2))
        return ly + label.get_height() + sub.get_height() + 10

    def _draw(self) -> None:
        import time as _time

        t0 = _time.perf_counter()
        self.screen.fill(BG_VIEW)
        # Chrome strips (outside projection).
        self._draw_fps()  # EMA before top-bar chip text
        self._draw_top_bar()
        self._draw_toolbar()
        # Blueprint column
        pygame.draw.rect(
            self.screen,
            BG_PANEL,
            (self.panel_x, self.top_h, self.panel_w, self.view_h),
        )
        pygame.draw.line(
            self.screen,
            EDGE,
            (self.panel_x + self.panel_w - 1, self.top_h),
            (self.panel_x + self.panel_w - 1, self.top_h + self.view_h),
            1,
        )
        # Dock column fill when pushed open
        if int(getattr(self, "dock_w", 0) or 0) > 0:
            pygame.draw.rect(
                self.screen,
                CHROME_BG,
                (self.dock_x, self.top_h, self.dock_w, self.view_h),
            )
            pygame.draw.line(
                self.screen,
                CHROME_EDGE,
                (self.dock_x, self.top_h),
                (self.dock_x, self.top_h + self.view_h),
                1,
            )
        self._draw_control_panel()
        self._perf_ms["draw.panel"] = (_time.perf_counter() - t0) * 1000.0
        self._draw_particles()
        assert self._view_surf is not None
        self.screen.blit(self._view_surf, (self.view_x, self.view_y))
        t1 = _time.perf_counter()
        self._draw_legend()
        self._perf_ms["draw.legend"] = (_time.perf_counter() - t1) * 1000.0
        self._draw_nav()
        self._draw_hover_tip()
        self._draw_learn_badge()
        if self._fps_panel_open:
            self._draw_fps_panel()
        if self.exit_confirm_modal.is_visible():
            self.exit_confirm_modal.render(
                self.screen, self.width, self.height, self.font, self.small_font
            )
        t2 = _time.perf_counter()
        pygame.display.flip()
        self._perf_ms["draw.flip"] = (_time.perf_counter() - t2) * 1000.0

    def run(self) -> None:
        try:
            import time as _time

            _log_every = 30  # console progress while sim_running
            dt = 1.0 / float(FPS)
            self._last_frame_wall = _time.perf_counter()
            while self.running:
                self._perf_ms = {}
                # Events before tick so Pause/Start respond without waiting a draw.
                t0 = _time.perf_counter()
                self._handle_events()
                self._perf_ms["events"] = (_time.perf_counter() - t0) * 1000.0
                now = _time.perf_counter()
                wall_dt = min(0.1, max(1e-4, now - float(self._last_frame_wall)))
                self._last_frame_wall = now
                self._frame_dt = float(wall_dt)
                self._begin_frame_caches()
                t0 = _time.perf_counter()
                self._tick_stim_jitter(dt)
                self._tick_lamp(dt)
                self._tick_world_swing(dt)
                # Sensors once per frame (draw used to recompute poses every call).
                if self.ocelli_on:
                    self._update_ocelli_sensors()
                self._update_jo_sensors()
                if self.turret_on:
                    self._update_turret_sensors()
                    self._update_turret_aim_metric()
                if getattr(self, "show_ocelli_funnel", False) or getattr(
                    self, "show_turret_funnel", False
                ):
                    self._sync_device_funnel_gravity()
                self._perf_ms["sense.ui"] = (_time.perf_counter() - t0) * 1000.0
                if self.sim_running:
                    self.fly.sim_tick()
                    if self.fly.is_adult() and self._activity_running:
                        # Decouple: closed-loop at View Detail × adult_sim_hz.
                        sim_hz = float(self._view_detail_knobs()["sim_hz"])
                        sim_hz = float(np.clip(sim_hz, 5.0, float(FPS)))
                        step = 1.0 / sim_hz
                        self._sim_accum = float(getattr(self, "_sim_accum", 0.0)) + wall_dt
                        # At most one closed-loop tick per display frame so draw
                        # still gets a turn when tissue/funnels are heavy.
                        if self._sim_accum >= step:
                            info = self._tick_adult_closed_loop(step)
                            self._perf_absorb_sim(info)
                            self._sim_accum -= step
                            if self._sim_accum > 2.0 * step:
                                self._sim_accum = step
                            if (
                                not (self.ocelli_on or self.turret_on)
                                and info["max"] < 0.04
                            ):
                                self.sim_running = False
                                self._activity_running = False
                                self._set_io_status(
                                    f"Wave faded · t={self.fly.sim_time}"
                                )
                    elif not self.fly.is_adult():
                        if not self._activity_running and (
                            self.ocelli_on or self.turret_on
                        ):
                            # Early device drive once tips connect (rising).
                            seeded = False
                            if self.ocelli_on and self._seed_ocelli_activity():
                                seeded = True
                            if self._seed_eye_activity():
                                seeded = True
                            if self._seed_jo_activity():
                                seeded = True
                            if self.turret_on and self._seed_turret_activity():
                                seeded = True
                            if seeded:
                                self.fly.tick_activity(hop_gain=0.8, decay=0.92)
                                self._sync_activity_view()
                        self.grown_xyz = self.fly.neurite_xyz
                        self.grown_cls = self.fly.neurite_cls
                        self.grown_rgb = _axon_rgb_for_cls(self.grown_cls)
                        ns = self.fly.neurite_stats()
                        if (
                            self.fly.sim_time > 0
                            and self.fly.sim_time % _log_every == 0
                        ):
                            yld = self._recipe_yield()
                            print(
                                f"[sim] t={self.fly.sim_time}  "
                                f"conn={ns['status'].get('connected', 0)}  "
                                f"act={ns['active']}  "
                                f"fail={ns['status'].get('failed', 0)}  "
                                f"yield={yld.get('pct', 0):.1f}% {yld.get('verdict', '')}  "
                                f"trails={self.fly.growth_trail_point_count()}",
                                flush=True,
                            )
                        if self._settle_if_idle():
                            n_edges = len(self.fly.grown_contact_edges()[0])
                            print(
                                f"[sim] ADULT — contacts={n_edges}",
                                flush=True,
                            )
                        # Turret pose held outside Run (no idle spin).
                    # else: adult idle / Fuse — hold turret; motion only in closed loop.
                # else: sim paused — hold turret.
                self._draw()
                self._perf_commit_frame()
                self.clock.tick(FPS)
        except KeyboardInterrupt:
            self.running = False
        finally:
            self._save_ui_state()
            pygame.quit()


def main() -> None:
    try:
        GenomeShell().run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)

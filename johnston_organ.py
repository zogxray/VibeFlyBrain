"""Johnston's organ sockets: paired antennae → Matrix tilt → sensory Fuse.

Virtual sensor + clamp pack per side. No vibration channel (tilt only).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from device import POLICY_JO, JohnstonOrganDevice

JO_GROUP_N = 4
JO_CLUSTER_FRAC = 0.012
JO_INSET_FRAC = 0.04
JO_FUSE_SEEK = 1200.0
# Degrees that map to tilt = ±1 (match WORLD_SWING_AMP_MAX ≈ 40°).
JO_TILT_REF_DEG = 40.0
JO_TILT_REF_RAD = math.radians(JO_TILT_REF_DEG)
# Antenna geometry vs body orbit r.
JO_ROOT_ANT = 0.16
JO_ROOT_UP = 0.14
JO_ROOT_LAT = 0.11
JO_SHAFT_ANT = 0.14
JO_SHAFT_UP = 0.05
JO_SHAFT_LAT = 0.05
JO_PEDICEL_T = 0.32  # JO seat along root→tip


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def iter_jo_devices(host: Any) -> list[JohnstonOrganDevice]:
    out: list[JohnstonOrganDevice] = []
    for d in getattr(host, "iter_devices", lambda: [])():
        if isinstance(d, JohnstonOrganDevice):
            out.append(d)
    if out:
        return out
    for d in getattr(getattr(host, "blueprint", None), "device_list", []) or []:
        if isinstance(d, JohnstonOrganDevice):
            out.append(d)
    return out


def jo_rig_pose(host: Any, side: str) -> dict | None:
    """Antenna root/tip + JO seat for one side."""
    cache = getattr(host, "_frame_jo_pose", None)
    if not isinstance(cache, dict):
        cache = {}
        host._frame_jo_pose = cache
    side_l = "right" if str(side).lower().startswith("r") else "left"
    if side_l in cache and cache[side_l] is not None:
        return cache[side_l]
    frame = host._body_orbit_frame() if hasattr(host, "_body_orbit_frame") else None
    if frame is None:
        return None
    c = np.asarray(frame[0], dtype=np.float64)
    r = float(frame[1])
    ant = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    lat = float(JO_ROOT_LAT) * r * (1.0 if side_l == "right" else -1.0)
    root = (
        c
        + ant * (float(JO_ROOT_ANT) * r)
        + up * (float(JO_ROOT_UP) * r)
        + right * lat
    )
    tip = (
        root
        + ant * (float(JO_SHAFT_ANT) * r)
        + up * (float(JO_SHAFT_UP) * r)
        + right * (float(JO_SHAFT_LAT) * r * (1.0 if side_l == "right" else -1.0))
    )
    t = float(JO_PEDICEL_T)
    seat = root * (1.0 - t) + tip * t
    axis = _norm(tip - root)
    col = (220, 110, 110) if side_l == "left" else (200, 100, 100)
    pose = {
        "side": side_l,
        "root": root.astype(np.float64),
        "tip": tip.astype(np.float64),
        "seat": seat.astype(np.float64),
        "axis": axis.astype(np.float64),
        "color": col,
        "radius": float(r),
        "center": c.copy(),
    }
    cache[side_l] = pose
    return pose


def update_jo_sensors(host: Any) -> None:
    """World swing angle → jo_left_tilt / jo_right_tilt ∈ [−1, 1]."""
    if hasattr(host, "_world_swing_angle"):
        ang = float(host._world_swing_angle())
    else:
        ang = 0.0
    tilt = float(np.clip(ang / max(float(JO_TILT_REF_RAD), 1e-9), -1.0, 1.0))
    for side, attr in (("left", "jo_left_tilt"), ("right", "jo_right_tilt")):
        setattr(host, attr, tilt)
        for d in iter_jo_devices(host):
            if d.side == side:
                d.tilt = tilt


def _place_pack(
    host: Any,
    pose: dict,
    groups: list[list[int]],
    terminals: list[int],
    *,
    move_only: bool,
) -> np.ndarray:
    """Pack somata around JO seat; clamp inset toward body center."""
    seat = np.asarray(pose["seat"], dtype=np.float64)
    axis = np.asarray(pose["axis"], dtype=np.float64)
    r = float(pose["radius"])
    c = np.asarray(pose["center"], dtype=np.float64)
    cl_r = float(JO_CLUSTER_FRAC) * r
    inset = float(JO_INSET_FRAC) * r
    to_c = c - seat
    tn = float(np.linalg.norm(to_c))
    origin = seat + (to_c / tn) * inset if tn > 1e-8 else seat
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, axis))) > 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = _norm(np.cross(axis, ref))
    v = _norm(np.cross(axis, u))
    n = int(JO_GROUP_N)
    rng = np.random.default_rng(17 + (1 if pose["side"] == "right" else 0))
    ang = rng.uniform(0.0, math.tau, size=n)
    rad = cl_r * rng.uniform(0.25, 1.0, size=n)
    depth = cl_r * rng.uniform(0.1, 0.45, size=n)
    pos = (
        origin[None, :]
        + rad[:, None] * np.cos(ang)[:, None] * u[None, :]
        + rad[:, None] * np.sin(ang)[:, None] * v[None, :]
        - depth[:, None] * axis[None, :]
    ).astype(np.float32)
    if move_only and groups and groups[0]:
        for j, nid in enumerate(groups[0]):
            neu = host.fly.neurons.get(int(nid))
            if neu is not None and j < len(pos):
                neu.soma = pos[j]
        if terminals:
            tneu = host.fly.neurons.get(int(terminals[0]))
            if tneu is not None:
                tneu.soma = origin.astype(np.float32)
    return pos


def ensure_jo_neurons(host: Any, device: JohnstonOrganDevice) -> bool:
    """Create/update JO pack + clamp for one antenna."""
    if not device.enabled or not host.fly.neurons or not host.class_names:
        return False
    cache = getattr(host, "_frame_jo_pose", None)
    if isinstance(cache, dict):
        cache.pop(device.side, None)
    pose = jo_rig_pose(host, device.side)
    if pose is None:
        return False
    try:
        sens_cls = host.class_names.index("sensory")
    except ValueError:
        sens_cls = 0
    need_pack = int(JO_GROUP_N)
    need_term = 1
    flat = device.flat_ids()
    alive_pack = sum(1 for g in device.groups for i in g if i in host.fly.neurons)
    alive_term = sum(1 for i in device.terminals if i in host.fly.neurons)
    if (
        alive_pack != need_pack
        or alive_term != need_term
        or len(device.groups) != 1
    ):
        for i in flat:
            host.fly.neurons.pop(i, None)
        for s in host.fly.sets.values():
            s.members.difference_update(flat)
        device.groups = [[]]
        device.terminals = []
        pos = _place_pack(
            host, pose, device.groups, device.terminals, move_only=False
        )
        gids: list[int] = []
        for j in range(len(pos)):
            n = host.fly.add_neuron(pos[j], sens_cls, set_name="sensory")
            gids.append(int(n.id))
        device.groups = [gids]
        seat = np.asarray(pose["seat"], dtype=np.float64)
        c = np.asarray(pose["center"], dtype=np.float64)
        r = float(pose["radius"])
        to_c = c - seat
        tn = float(np.linalg.norm(to_c))
        inset = float(JO_INSET_FRAC) * r
        clamp = (seat + (to_c / tn) * inset if tn > 1e-8 else seat).astype(np.float32)
        t = host.fly.add_neuron(clamp, sens_cls, set_name="sensory")
        device.terminals = [int(t.id)]
        _wire_jo_device(host, device)
        host.fly.rebuild_caches()
        if hasattr(host, "_sync_from_fly"):
            host._sync_from_fly()
        try:
            if "sensory" in host.class_names:
                host.focus_soma_indices.add(host.class_names.index("sensory"))
            if hasattr(host, "_sync_layers_from_focus"):
                host._sync_layers_from_focus()
        except Exception:
            pass
        setattr(host, f"{device.name}_on", True)
        return True
    _place_pack(host, pose, device.groups, device.terminals, move_only=True)
    _wire_jo_device(host, device)
    host.fly.rebuild_caches()
    host._basin_cache_key = None
    setattr(host, f"{device.name}_on", True)
    return True


def _wire_jo_device(host: Any, device: JohnstonOrganDevice) -> None:
    flat = set(device.flat_ids())
    if len(flat) < 2:
        return
    if hasattr(host, "_strip_ocelli_device_wires"):
        host._strip_ocelli_device_wires(flat)
    gids = [int(x) for x in (device.groups[0] if device.groups else [])]
    if hasattr(host, "_device_wire_mesh"):
        host._device_wire_mesh(gids)
    if device.terminals and hasattr(host, "_device_wire_star"):
        host._device_wire_star(int(device.terminals[0]), gids)


def jo_connected_mask(host: Any, jo_name: str) -> list[bool]:
    for d in iter_jo_devices(host):
        if d.name != jo_name:
            continue
        if not d.terminals:
            return [False]
        gids = list(d.groups[0]) if d.groups else []
        gids.append(int(d.terminals[0]))
        if hasattr(host, "_owners_fused_to_cns"):
            return [bool(host._owners_fused_to_cns(gids))]
        return [False]
    return [False]


def arm_jo_tips(
    host: Any, device: JohnstonOrganDevice, *, missing_only: bool = False
) -> int:
    """Append Fuse growth tips from JO clamp → sensory/ascending."""
    if not device.enabled:
        return 0
    if getattr(host.fly, "is_adult", lambda: False)() and not missing_only:
        return 0
    if not ensure_jo_neurons(host, device):
        return 0
    pose = jo_rig_pose(host, device.side)
    if pose is None or not device.terminals:
        return 0
    flat = np.asarray(device.flat_ids(), dtype=np.int32)
    if (
        not missing_only
        and len(host.fly.growth_owner)
        and len(flat)
        and np.any(np.isin(host.fly.growth_owner, flat))
    ):
        return 0
    ban = set(host.all_device_neuron_ids()) if hasattr(host, "all_device_neuron_ids") else set()
    used: set[int] = set()
    conn = jo_connected_mask(host, device.name) if missing_only else [False]
    tid = int(device.terminals[0])
    if missing_only and conn and conn[0]:
        return 0
    if missing_only and hasattr(host, "_device_owner_has_open_tip"):
        if host._device_owner_has_open_tip(tid):
            return 0
    probe = np.asarray(pose["seat"], dtype=np.float64)
    tips_per = int(POLICY_JO.tips_per_site)
    gids = host._pick_cns_near(
        probe,
        tips_per,
        prefer=POLICY_JO.prefer,
        fallback=POLICY_JO.fallback,
        ban=ban,
        used=used,
    )
    if not gids:
        return 0
    owners = [tid] * len(gids)
    return int(
        host.fly.append_growth_tips(
            np.asarray(owners, dtype=np.int32),
            np.asarray(gids, dtype=np.int32),
            seek_limit=float(JO_FUSE_SEEK),
        )
    )


def fuse_jo_device(host: Any, device: JohnstonOrganDevice) -> int:
    """Instant connect JO clamp → nearest sensory/ascending posts."""
    if not device.enabled:
        return 0
    if not ensure_jo_neurons(host, device):
        return 0
    pose = jo_rig_pose(host, device.side)
    if pose is None or not device.terminals:
        return 0
    ban = set(host.all_device_neuron_ids()) if hasattr(host, "all_device_neuron_ids") else set()
    used: set[int] = set()
    n = 0
    tips_per = int(POLICY_JO.tips_per_site)
    tid = int(device.terminals[0])
    probe = np.asarray(pose["seat"], dtype=np.float64)
    goals = host._pick_cns_near(
        probe,
        tips_per,
        prefer=POLICY_JO.prefer,
        fallback=POLICY_JO.fallback,
        ban=ban,
        used=used,
    )
    for g in goals:
        if host.fly.connect_pair(tid, int(g)):
            n += 1
    return n


def seed_jo_activity(host: Any) -> int:
    update_jo_sensors(host)
    n = 0
    for d in iter_jo_devices(host):
        if not d.enabled:
            continue
        levels = d.sense(host)
        if levels and hasattr(host.fly, "stimulate_neurons"):
            n += int(host.fly.stimulate_neurons(levels))
    return n


def draw_johnston_organ(
    host: Any, device: JohnstonOrganDevice, cx: float, cy: float, view_lim_y: int
) -> None:
    """Antenna shaft + JO seat + fuse plate + tilt readout."""
    import pygame

    from genome_shell import project_3d_batch

    if not device.enabled or host._view_surf is None:
        return
    pose = jo_rig_pose(host, device.side)
    if pose is None:
        return
    update_jo_sensors(host)
    tilt = float(getattr(host, f"{device.name}_tilt", device.tilt))
    conn = jo_connected_mask(host, device.name)
    ok = bool(conn[0]) if conn else False
    root = np.asarray(pose["root"], dtype=np.float64)
    tip = np.asarray(pose["tip"], dtype=np.float64)
    seat = np.asarray(pose["seat"], dtype=np.float64)
    col = pose["color"]
    bri = 0.40 + 0.60 * abs(tilt)
    rgb = tuple(int(np.clip(c * bri, 0, 255)) for c in col)

    # Fuse plate around seat + pack.
    pts3: list[np.ndarray] = [seat, root, tip]
    for gids in device.groups:
        for nid in gids:
            neu = host.fly.neurons.get(int(nid))
            if neu is not None:
                pts3.append(np.asarray(neu.soma, dtype=np.float64))
    if device.terminals:
        tneu = host.fly.neurons.get(int(device.terminals[0]))
        if tneu is not None:
            pts3.append(np.asarray(tneu.soma, dtype=np.float64))
    cloud = np.stack(pts3, axis=0).astype(np.float32)
    sx, sy = project_3d_batch(
        cloud, host.view_angle_x, host.view_angle_y, cx, cy, host.scale_3d
    )
    finite = np.isfinite(sx) & np.isfinite(sy)
    if not np.any(finite):
        return
    xs = sx[finite]
    ys = sy[finite]
    pad = 10
    left = int(np.clip(xs.min() - pad, 0, host.view_w - 1))
    right = int(np.clip(xs.max() + pad, 0, host.view_w - 1))
    top = int(np.clip(ys.min() - pad, 0, view_lim_y - 1))
    bot = int(np.clip(ys.max() + pad, 0, view_lim_y - 1))
    bw = max(1, right - left)
    bh = max(1, bot - top)
    overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
    box_c = (*col, 55)
    pygame.draw.rect(overlay, box_c, overlay.get_rect(), width=1, border_radius=3)
    if ok:
        st_txt, st_rgb, st_bg = "fused", (120, 220, 150), (16, 36, 24, 200)
    else:
        st_txt, st_rgb, st_bg = "not fused", (230, 140, 140), (40, 18, 18, 200)
    st_lab = host.hint_font.render(st_txt, True, st_rgb)
    pad_x, pad_y, dot_gap = 6, 3, 12
    pw = st_lab.get_width() + pad_x * 2 + dot_gap
    ph = st_lab.get_height() + pad_y * 2
    px = int(np.clip(4, 2, max(2, bw - pw - 2)))
    py = int(np.clip(4, 2, max(2, bh - ph - 2)))
    plate = pygame.Surface((pw, ph), pygame.SRCALPHA)
    pygame.draw.rect(plate, st_bg, plate.get_rect(), border_radius=3)
    pygame.draw.rect(plate, (*st_rgb, 200), plate.get_rect(), width=1, border_radius=3)
    pygame.draw.circle(plate, (*st_rgb, 230), (pad_x + 3, ph // 2), 3)
    plate.blit(st_lab, (pad_x + dot_gap, pad_y))
    overlay.blit(plate, (px, py))
    host._view_surf.blit(overlay, (left, top))

    title = host.hint_font.render(device.panel_heading.lower(), True, col)
    title.set_alpha(160)
    host._view_surf.blit(
        title, (left + 4, max(2, top - title.get_height() - 1))
    )

    # Shaft + JO blob.
    shaft = np.stack([root, tip], axis=0).astype(np.float32)
    ax, ay = project_3d_batch(
        shaft, host.view_angle_x, host.view_angle_y, cx, cy, host.scale_3d
    )
    if np.isfinite(ax).all() and np.isfinite(ay).all():
        pygame.draw.line(
            host._view_surf,
            rgb,
            (int(ax[0]), int(ay[0])),
            (int(ax[1]), int(ay[1])),
            2 if ok else 1,
        )
    seat_p = seat.reshape(1, 3).astype(np.float32)
    sx0, sy0 = project_3d_batch(
        seat_p, host.view_angle_x, host.view_angle_y, cx, cy, host.scale_3d
    )
    if np.isfinite(sx0[0]) and np.isfinite(sy0[0]):
        lx, ly = int(sx0[0]), int(sy0[0])
        core = max(3, int(4 + 5 * abs(tilt)))
        pygame.draw.circle(host._view_surf, rgb, (lx, ly), core)
        deg = tilt * float(JO_TILT_REF_DEG)
        txt = f"{deg:+.0f}°" + ("*" if ok else "")
        lab = host.hint_font.render(txt, True, (255, 180, 180))
        lab.set_alpha(200)
        host._view_surf.blit(lab, (lx + core + 3, ly - lab.get_height() // 2))

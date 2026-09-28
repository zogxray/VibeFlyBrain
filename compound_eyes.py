"""Compound-eye sockets: Left/Right × (4 azimuth corners + Up + Down).

Runtime sense only (brightness cones). Fuse prefers optic / visual_projection.
Teacher / lamp XYZ must stay out of the motor closed loop.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from device import (
    EYE_N_SITES,
    EYE_SITE_LABELS,
    POLICY_EYE,
    CompoundEyeDevice,
)

EYE_GROUP_N = 4
EYE_CLUSTER_FRAC = 0.010
EYE_INSET_FRAC = 0.035
EYE_FACING_BIAS = 0.12
EYE_FACING_POW = 1.05
EYE_FUSE_SEEK = 1200.0
# Hemisphere shell around optic lobe somata (contour for future surface).
EYE_SHELL_PCT = 88.0  # radius = this percentile of optic cloud distances
EYE_SHELL_PAD = 1.10  # slight inflate outside tissue
EYE_SHELL_MIN_FRAC = 0.08  # floor vs body orbit r if optic cloud thin
EYE_CONTOUR_THETAS = (0.40, 0.75, 1.10)  # rad from outward pole
EYE_CONTOUR_SEGS = 28
# Corner azimuths in the eye's outward horizontal plane (full circle).
EYE_CORNER_AZIMUTH = (0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi)
EYE_CORNER_THETA = 0.85  # rad from outward pole for NE/SE/SW/NW band
EYE_POLE_THETA = 0.55  # U/D closer to outward pole, tilted dorsal/ventral


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _facing(axis: np.ndarray, direction: np.ndarray) -> float:
    c = float(np.dot(axis, direction))
    bias = float(EYE_FACING_BIAS)
    raw = max(0.0, (c + bias) / (1.0 + bias))
    return float(raw ** float(EYE_FACING_POW))


def iter_eye_devices(host: Any) -> list[CompoundEyeDevice]:
    out: list[CompoundEyeDevice] = []
    for d in getattr(host, "iter_devices", lambda: [])():
        if isinstance(d, CompoundEyeDevice):
            out.append(d)
    if out:
        return out
    # Fallback: blueprint list
    for d in getattr(getattr(host, "blueprint", None), "device_list", []) or []:
        if isinstance(d, CompoundEyeDevice):
            out.append(d)
    return out


def _optic_lobe_shell(host: Any, side: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray] | None:
    """Center, radius, and local basis (outward, fwd, dorsal) for one optic lobe."""
    fly = getattr(host, "fly", None)
    if fly is None or not getattr(fly, "neurons", None):
        return None
    names = list(getattr(fly, "class_names", None) or getattr(host, "class_names", []) or [])
    xyz = getattr(fly, "soma_xyz", None)
    cls = getattr(fly, "soma_cls", None)
    if xyz is None or cls is None or len(xyz) == 0 or len(xyz) != len(cls):
        return None
    want: list[int] = []
    for name in ("optic", "visual_projection"):
        if name in names:
            want.append(int(names.index(name)))
    if not want:
        return None
    mask = np.isin(cls.astype(np.int32), np.asarray(want, dtype=np.int32))
    cloud = np.asarray(xyz[mask], dtype=np.float64)
    if len(cloud) < 30:
        return None
    # Body mid-X: left = −X side of optic mass.
    mid_x = float(np.median(xyz[:, 0]))
    side_l = "right" if str(side).lower().startswith("r") else "left"
    if side_l == "left":
        cloud = cloud[cloud[:, 0] < mid_x]
    else:
        cloud = cloud[cloud[:, 0] >= mid_x]
    if len(cloud) < 20:
        return None
    center = cloud.mean(axis=0)
    dist = np.linalg.norm(cloud - center[None, :], axis=1)
    radius = float(np.percentile(dist, float(EYE_SHELL_PCT))) * float(EYE_SHELL_PAD)
    frame = host._body_orbit_frame() if hasattr(host, "_body_orbit_frame") else None
    body_r = float(frame[1]) if frame is not None else 1.0
    radius = max(radius, float(EYE_SHELL_MIN_FRAC) * body_r)
    body_c = np.asarray(frame[0], dtype=np.float64) if frame is not None else cloud.mean(axis=0)
    ant = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    outward = _norm(center - body_c)
    # Prefer lateral if outward is degenerate.
    if float(np.linalg.norm(outward)) < 0.2:
        outward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if side_l == "left":
            outward = -outward
    fwd = _norm(ant - outward * float(np.dot(ant, outward)))
    dorsal = _norm(np.cross(outward, fwd))
    if float(np.dot(dorsal, up)) < 0:
        dorsal = -dorsal
    fwd = _norm(np.cross(dorsal, outward))
    return (
        center.astype(np.float64),
        float(radius),
        outward.astype(np.float64),
        fwd.astype(np.float64),
        dorsal.astype(np.float64),
    )


def _hemisphere_dir(
    outward: np.ndarray,
    fwd: np.ndarray,
    dorsal: np.ndarray,
    *,
    theta: float,
    phi: float,
) -> np.ndarray:
    """Unit direction on outward hemisphere: theta from outward pole, phi from fwd."""
    return _norm(
        math.cos(theta) * outward
        + math.sin(theta) * (math.cos(phi) * fwd + math.sin(phi) * dorsal)
    )


def eye_rig_pose(host: Any, side: str) -> list[dict] | None:
    """Six look sites on the optic-lobe hemisphere: corners + U/D."""
    cache = getattr(host, "_frame_eye_pose", None)
    if not isinstance(cache, dict):
        cache = {}
        host._frame_eye_pose = cache
    side_l = "right" if str(side).lower().startswith("r") else "left"
    if side_l in cache and cache[side_l] is not None:
        return cache[side_l]
    shell = _optic_lobe_shell(host, side_l)
    if shell is None:
        return None
    center, radius, outward, fwd, dorsal = shell
    col = (90, 200, 210) if side_l == "left" else (80, 180, 200)
    sites: list[dict] = []
    for i, lab in enumerate(EYE_SITE_LABELS):
        if lab == "U":
            axis = _hemisphere_dir(
                outward, fwd, dorsal, theta=float(EYE_POLE_THETA), phi=0.5 * math.pi
            )
        elif lab == "D":
            axis = _hemisphere_dir(
                outward, fwd, dorsal, theta=float(EYE_POLE_THETA), phi=-0.5 * math.pi
            )
        else:
            phi = float(EYE_CORNER_AZIMUTH[i])
            axis = _hemisphere_dir(
                outward, fwd, dorsal, theta=float(EYE_CORNER_THETA), phi=phi
            )
        lens = center + radius * axis
        sites.append(
            {
                "name": lab,
                "lens": lens.astype(np.float64),
                "axis": axis.astype(np.float64),
                "color": col,
                "side": side_l,
                "center": center.copy(),
                "radius": float(radius),
                "outward": outward.copy(),
                "fwd": fwd.copy(),
                "dorsal": dorsal.copy(),
            }
        )
    cache[side_l] = sites
    return sites


def eye_hemisphere_contour(host: Any, side: str) -> list[np.ndarray]:
    """Closed polylines on the outward hemisphere (rings + meridians)."""
    pose = eye_rig_pose(host, side)
    if not pose:
        return []
    center = np.asarray(pose[0]["center"], dtype=np.float64)
    radius = float(pose[0]["radius"])
    outward = np.asarray(pose[0]["outward"], dtype=np.float64)
    fwd = np.asarray(pose[0]["fwd"], dtype=np.float64)
    dorsal = np.asarray(pose[0]["dorsal"], dtype=np.float64)
    segs = int(EYE_CONTOUR_SEGS)
    polylines: list[np.ndarray] = []
    for theta in EYE_CONTOUR_THETAS:
        ring = np.zeros((segs + 1, 3), dtype=np.float32)
        for k in range(segs + 1):
            phi = (math.tau * k) / segs
            ring[k] = center + radius * _hemisphere_dir(
                outward, fwd, dorsal, theta=float(theta), phi=phi
            )
        polylines.append(ring)
    n_mer = max(4, segs // 2)
    for phi in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        merid = np.zeros((n_mer + 1, 3), dtype=np.float32)
        for k in range(n_mer + 1):
            theta = (0.5 * math.pi * k) / n_mer
            merid[k] = center + radius * _hemisphere_dir(
                outward, fwd, dorsal, theta=float(theta), phi=phi
            )
        polylines.append(merid)
    return polylines


def update_eye_sensors(host: Any) -> None:
    """Sky + point lamp → eye_left_I / eye_right_I (6 channels each)."""
    sky_a = (
        float(np.clip(getattr(host, "sky_fill", 0.28), 0.0, 1.0))
        if getattr(host, "light_enabled", True)
        else 0.15
    )
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if hasattr(host, "_update_lamp_xyz"):
        host._update_lamp_xyz()
    lamp = np.asarray(getattr(host, "_lamp_xyz", np.zeros(3)), dtype=np.float64)
    inten = (
        float(np.clip(getattr(host, "lamp_intensity", 1.0), 0.0, 2.0))
        if getattr(host, "lamp_on", True)
        else 0.0
    )
    frame = host._body_orbit_frame() if hasattr(host, "_body_orbit_frame") else None
    body_r = float(frame[1]) if frame is not None else 1.0
    light_on = bool(getattr(host, "light_enabled", True))

    for side, attr in (("left", "eye_left_I"), ("right", "eye_right_I")):
        pose = eye_rig_pose(host, side)
        if pose is None:
            setattr(host, attr, np.zeros(EYE_N_SITES, dtype=np.float64))
            continue
        vals: list[float] = []
        for oc in pose:
            axis = oc["axis"]
            lens = oc["lens"]
            sky = sky_a * _facing(axis, up)
            to_l = lamp - lens
            dist = float(np.linalg.norm(to_l))
            if dist < 1e-6 or inten <= 0.0 or not light_on:
                lamp_t = 0.0
            else:
                L = to_l / dist
                facing = _facing(axis, L)
                atten = (body_r * body_r) / (dist * dist + (0.35 * body_r) ** 2)
                lamp_t = inten * facing * atten
            # Eyes: less sky wash than ocelli — lamp bearing dominates.
            vals.append(
                float(np.clip(0.08 * sky_a + 0.25 * sky + 1.0 * lamp_t, 0.0, 1.0))
            )
        arr = np.asarray(vals, dtype=np.float64)
        setattr(host, attr, arr)
        # Mirror onto typed device if present.
        for d in iter_eye_devices(host):
            if d.side == side:
                d.intensity = arr.copy()


def eye_light_vector(host: Any) -> tuple[np.ndarray, float] | None:
    """Population vector over both eyes → unit direction to the lamp.

    Each of the 12 look sites answers roughly ``I ≈ base + k·max(0, axis·L)``.
    Summing ``(I_i − mean I)·axis_i`` is the classic labelled-line readout: a
    fixed anatomical map (site → preferred direction), not a lamp oracle.
    Measured on the frozen plant it calls the lag side right ~93% of the time
    (~89% inside 30°), which is what the motor credit needs.

    Returns (unit vector in body coords, contrast confidence 0..1).
    """
    axes: list[np.ndarray] = []
    vals: list[float] = []
    for side, attr in (("left", "eye_left_I"), ("right", "eye_right_I")):
        if not bool(getattr(host, f"eye_{side}_on", False)):
            continue
        pose = eye_rig_pose(host, side)
        if pose is None:
            continue
        I = np.asarray(getattr(host, attr, np.zeros(EYE_N_SITES)), dtype=np.float64)
        if len(I) < len(pose):
            I = np.pad(I, (0, len(pose) - len(I)))
        for i, oc in enumerate(pose):
            axes.append(np.asarray(oc["axis"], dtype=np.float64))
            vals.append(float(I[i]))
    if len(axes) < 4:
        return None
    A = np.stack(axes, axis=0)
    v = np.asarray(vals, dtype=np.float64)
    span = float(np.max(v) - np.min(v))
    if span < 1e-4 or not math.isfinite(span):
        return None
    c = v - float(np.mean(v))
    vec = (c[:, None] * A).sum(axis=0)
    n = float(np.linalg.norm(vec))
    if n < 1e-12 or not math.isfinite(n):
        return None
    conf = float(np.clip(span / 0.18, 0.0, 1.0))
    return (vec / n).astype(np.float64), conf


def eye_lamp_azimuth(host: Any) -> tuple[float, float] | None:
    """Eye-derived lamp azimuth (rad) in the turret's yaw convention + confidence."""
    out = eye_light_vector(host)
    if out is None:
        return None
    vec, conf = out
    return float(math.atan2(float(vec[0]), float(vec[2]))), float(conf)


def eye_probe_light_angle(host: Any, device: CompoundEyeDevice) -> float | None:
    """Lamp-orbit azimuth that puts the lamp in front of this eye.

    The eye hemisphere faces ``outward``; the lamp rides a fixed-elevation orbit
    parametrised by ``lamp_angle`` about world-up. Projecting ``outward`` onto
    the orbit plane gives the azimuth where this eye actually sees the lamp.
    Used by the reach probe so a wave is measured on a lit eye, not a shadowed
    one — the left eye is dark at the right eye's azimuth and vice versa.
    """
    pose = eye_rig_pose(host, device.side)
    if not pose:
        return None
    outward = np.asarray(pose[0]["outward"], dtype=np.float64)
    frame = host._world_frame() if hasattr(host, "_world_frame") else None
    if frame is None:
        return None
    _up, right, forward = frame
    x = float(np.dot(outward, np.asarray(right, dtype=np.float64)))
    y = float(np.dot(outward, np.asarray(forward, dtype=np.float64)))
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return None
    return float(math.atan2(y, x))


def _place_clusters(
    host: Any, pose: list[dict], groups: list[list[int]], terminals: list[int], *, move_only: bool
) -> list[np.ndarray]:
    frame = host._body_orbit_frame()
    if frame is None:
        c = np.zeros(3, dtype=np.float64)
        r = 1.0
    else:
        c = np.asarray(frame[0], dtype=np.float64)
        r = float(frame[1])
    cl_r = float(EYE_CLUSTER_FRAC) * r
    inset = float(EYE_INSET_FRAC) * r
    all_pos: list[np.ndarray] = []
    for gi, oc in enumerate(pose):
        lens = np.asarray(oc["lens"], dtype=np.float64)
        axis = oc["axis"]
        to_c = c - lens
        tn = float(np.linalg.norm(to_c))
        origin = lens + (to_c / tn) * inset if tn > 1e-8 else lens
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, axis))) > 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = np.cross(axis, ref)
        u /= max(float(np.linalg.norm(u)), 1e-8)
        v = np.cross(axis, u)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        rng = np.random.default_rng(2200 + gi + (0 if oc.get("side") == "left" else 17))
        n = int(EYE_GROUP_N)
        pos = np.zeros((n, 3), dtype=np.float64)
        ang = np.linspace(0.0, math.tau, n, endpoint=False) + float(rng.uniform(0, 0.2))
        rad = cl_r * np.sqrt(rng.uniform(0.2, 1.0, size=n))
        depth = cl_r * rng.uniform(0.1, 0.5, size=n)
        pos[:] = (
            origin[None, :]
            + rad[:, None] * np.cos(ang)[:, None] * u[None, :]
            + rad[:, None] * np.sin(ang)[:, None] * v[None, :]
            - depth[:, None] * axis[None, :]
        )
        all_pos.append(pos.astype(np.float32))
        if move_only and gi < len(groups):
            for j, nid in enumerate(groups[gi]):
                neu = host.fly.neurons.get(int(nid))
                if neu is not None and j < len(pos):
                    neu.soma = pos[j]
        if move_only and gi < len(terminals):
            tneu = host.fly.neurons.get(int(terminals[gi]))
            if tneu is not None:
                tneu.soma = origin.astype(np.float32)
    return all_pos


def ensure_eye_neurons(host: Any, device: CompoundEyeDevice) -> bool:
    """Create/update photoreceptor packs + clamps for one eye."""
    if not device.enabled or not host.fly.neurons or not host.class_names:
        return False
    # Bust pose cache so shell tracks optic cloud if fly moved.
    cache = getattr(host, "_frame_eye_pose", None)
    if isinstance(cache, dict):
        cache.pop(device.side, None)
    pose = eye_rig_pose(host, device.side)
    if pose is None:
        return False
    frame = host._body_orbit_frame()
    if frame is None:
        return False
    c = np.asarray(frame[0], dtype=np.float64)
    r = float(frame[1])
    try:
        sens_cls = host.class_names.index("sensory")
    except ValueError:
        sens_cls = 0
    need_pack = EYE_N_SITES * int(EYE_GROUP_N)
    need_term = EYE_N_SITES
    flat = device.flat_ids()
    alive_pack = sum(1 for g in device.groups for i in g if i in host.fly.neurons)
    alive_term = sum(1 for i in device.terminals if i in host.fly.neurons)
    if (
        alive_pack != need_pack
        or alive_term != need_term
        or len(device.groups) != EYE_N_SITES
    ):
        for i in flat:
            host.fly.neurons.pop(i, None)
        for s in host.fly.sets.values():
            s.members.difference_update(flat)
        device.groups = [[] for _ in range(EYE_N_SITES)]
        device.terminals = []
        positions = _place_clusters(host, pose, device.groups, device.terminals, move_only=False)
        for gi, pos in enumerate(positions):
            gids: list[int] = []
            for j in range(len(pos)):
                n = host.fly.add_neuron(pos[j], sens_cls, set_name="sensory")
                gids.append(int(n.id))
            device.groups[gi] = gids
            lens = np.asarray(pose[gi]["lens"], dtype=np.float64)
            to_c = c - lens
            tn = float(np.linalg.norm(to_c))
            inset = float(EYE_INSET_FRAC) * r
            clamp = (lens + (to_c / tn) * inset if tn > 1e-8 else lens).astype(np.float32)
            t = host.fly.add_neuron(clamp, sens_cls, set_name="sensory")
            device.terminals.append(int(t.id))
        _wire_eye_device(host, device)
        host.fly.rebuild_caches()
        if hasattr(host, "_sync_from_fly"):
            host._sync_from_fly()
        # Show eye packs in Super-class N only — do NOT touch focus_basin
        # (that flips show_layer_basins / Filter B funnels on).
        try:
            if "sensory" in host.class_names:
                host.focus_soma_indices.add(host.class_names.index("sensory"))
            if hasattr(host, "_sync_layers_from_focus"):
                host._sync_layers_from_focus()
        except Exception:
            pass
        setattr(host, f"{device.name}_on", True)
        return True
    _place_clusters(host, pose, device.groups, device.terminals, move_only=True)
    _wire_eye_device(host, device)
    host.fly.rebuild_caches()
    host._basin_cache_key = None
    setattr(host, f"{device.name}_on", True)
    return True


def _wire_eye_device(host: Any, device: CompoundEyeDevice) -> None:
    flat = set(device.flat_ids())
    if len(flat) < 2:
        return
    if hasattr(host, "_strip_ocelli_device_wires"):
        host._strip_ocelli_device_wires(flat)
    terms = [int(t) for t in device.terminals]
    for gi, gids in enumerate(device.groups):
        if hasattr(host, "_device_wire_mesh"):
            host._device_wire_mesh([int(x) for x in gids])
        if gi < len(terms) and hasattr(host, "_device_wire_star"):
            host._device_wire_star(terms[gi], [int(x) for x in gids])
    for i in range(len(terms)):
        host.fly.connect_pair(terms[i], terms[(i + 1) % len(terms)])
        host.fly.connect_pair(terms[(i + 1) % len(terms)], terms[i])


def eye_connected_mask(host: Any, eye_name: str) -> list[bool]:
    from device import EYE_N_SITES

    for d in iter_eye_devices(host):
        if d.name != eye_name:
            continue
        if not d.terminals:
            return [False] * EYE_N_SITES
        out: list[bool] = []
        for i, tid in enumerate(d.terminals):
            gids = list(d.groups[i]) if i < len(d.groups) else []
            gids.append(int(tid))
            if hasattr(host, "_owners_fused_to_cns"):
                out.append(bool(host._owners_fused_to_cns(gids)))
            else:
                out.append(False)
        while len(out) < EYE_N_SITES:
            out.append(False)
        return out[:EYE_N_SITES]
    return [False] * EYE_N_SITES


def arm_eye_tips(host: Any, device: CompoundEyeDevice, *, missing_only: bool = False) -> int:
    """Append Fuse growth tips from eye clamps → optic/VPN."""
    if not device.enabled:
        return 0
    if getattr(host.fly, "is_adult", lambda: False)() and not missing_only:
        return 0
    if not ensure_eye_neurons(host, device):
        return 0
    pose = eye_rig_pose(host, device.side)
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
    conn = eye_connected_mask(host, device.name) if missing_only else [False] * EYE_N_SITES
    owners: list[int] = []
    goals: list[int] = []
    for gi, tid in enumerate(device.terminals):
        if missing_only and gi < len(conn) and conn[gi]:
            continue
        if missing_only and hasattr(host, "_device_owner_has_open_tip"):
            if host._device_owner_has_open_tip(int(tid)):
                continue
        probe = np.asarray(pose[gi]["lens"], dtype=np.float64)
        gids = eye_site_goals(host, device, gi, probe, ban=ban, used=used)
        for g in gids:
            owners.append(int(tid))
            goals.append(int(g))
    if not owners:
        return 0
    return int(
        host.fly.append_growth_tips(
            np.asarray(owners, dtype=np.int32),
            np.asarray(goals, dtype=np.int32),
            seek_limit=float(EYE_FUSE_SEEK),
        )
    )


def eye_site_goals(
    host: Any,
    device: CompoundEyeDevice,
    site: int,
    probe: np.ndarray,
    *,
    ban: set[int],
    used: set[int],
) -> list[int]:
    """CNS posts for one look site: stereo plan first, nearest optic/VPN else."""
    plan = host.stereo_plan() if hasattr(host, "stereo_plan") else None
    if plan is not None:
        goals = plan.goals_for(device.name, site)
        if goals:
            used.update(int(g) for g in goals)
            return [int(g) for g in goals]
    return host._pick_cns_near(
        probe,
        int(POLICY_EYE.tips_per_site),
        prefer=POLICY_EYE.prefer,
        fallback=POLICY_EYE.fallback,
        ban=ban,
        used=used,
    )


def fuse_eye_device(host: Any, device: CompoundEyeDevice) -> int:
    """Instant connect clamps → the optic/VPN posts this eye's channel uses."""
    if not device.enabled:
        return 0
    if not ensure_eye_neurons(host, device):
        return 0
    pose = eye_rig_pose(host, device.side)
    if pose is None:
        return 0
    ban = set(host.all_device_neuron_ids()) if hasattr(host, "all_device_neuron_ids") else set()
    used: set[int] = set()
    n = 0
    for gi, tid in enumerate(device.terminals):
        probe = np.asarray(pose[gi]["lens"], dtype=np.float64)
        goals = eye_site_goals(host, device, gi, probe, ban=ban, used=used)
        for g in goals:
            if host.fly.connect_pair(int(tid), int(g)):
                n += 1
    return n


def seed_eye_activity(host: Any) -> int:
    update_eye_sensors(host)
    n = 0
    for d in iter_eye_devices(host):
        if not d.enabled:
            continue
        levels = d.sense(host)
        if levels and hasattr(host.fly, "stimulate_neurons"):
            n += int(host.fly.stimulate_neurons(levels))
    return n


def _draw_eye_group_box(
    host: Any,
    device: CompoundEyeDevice,
    pose: list[dict],
    connected: list[bool],
    cx: float,
    cy: float,
    view_lim_y: int,
) -> None:
    """Screen box + fused/fusing/not-fused plate (same language as ocelli/turret)."""
    import pygame

    from genome_shell import project_3d_batch

    assert host._view_surf is not None
    pts3: list[np.ndarray] = []
    for oc in pose:
        pts3.append(np.asarray(oc["lens"], dtype=np.float64))
    for gids in device.groups:
        for nid in gids:
            neu = host.fly.neurons.get(int(nid))
            if neu is not None:
                pts3.append(np.asarray(neu.soma, dtype=np.float64))
    if len(pts3) < 2:
        return
    arr = np.stack(pts3, axis=0).astype(np.float32)
    sx, sy = project_3d_batch(
        arr, host.view_angle_x, host.view_angle_y, cx, cy, host.scale_3d
    )
    xs = [float(x) for x in sx if np.isfinite(x)]
    ys = [float(y) for y in sy if np.isfinite(y)]
    if len(xs) < 2:
        return
    pad = 14
    left = max(0, int(min(xs) - pad))
    top = max(0, int(min(ys) - pad))
    right = min(int(host.view_w) - 1, int(max(xs) + pad))
    bot = min(view_lim_y - 1, int(max(ys) + pad))
    if right - left < 8 or bot - top < 8:
        return
    bw, bh = right - left, bot - top
    n_ok = sum(1 for c in connected if c)
    n_tot = max(1, len(connected))
    if n_ok == n_tot:
        fill, edge, tick_c, title_c = (
            (36, 70, 72, 28),
            (110, 190, 190, 55),
            (150, 220, 210, 90),
            (160, 210, 210),
        )
    elif n_ok > 0:
        fill, edge, tick_c, title_c = (
            (70, 60, 30, 32),
            (200, 180, 90, 70),
            (220, 200, 120, 95),
            (210, 200, 140),
        )
    else:
        fill, edge, tick_c, title_c = (
            (70, 28, 28, 36),
            (210, 90, 90, 90),
            (230, 120, 120, 110),
            (220, 150, 150),
        )
    overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, bw, bh)
    pygame.draw.rect(overlay, fill, rect, border_radius=8)
    pygame.draw.rect(overlay, edge, rect, width=1, border_radius=8)
    tick = 10
    for x0, y0, dx, dy in (
        (0, 0, 1, 1),
        (bw - 1, 0, -1, 1),
        (0, bh - 1, 1, -1),
        (bw - 1, bh - 1, -1, -1),
    ):
        pygame.draw.line(overlay, tick_c, (x0, y0), (x0 + dx * tick, y0), 1)
        pygame.draw.line(overlay, tick_c, (x0, y0), (x0, y0 + dy * tick), 1)
    title = host.hint_font.render(device.panel_heading.lower(), True, title_c)
    title.set_alpha(160)
    host._view_surf.blit(
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
    st_lab = host.hint_font.render(st_txt, True, st_rgb)
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
    host._view_surf.blit(overlay, (left, top))


def draw_compound_eye(
    host: Any, device: CompoundEyeDevice, cx: float, cy: float, view_lim_y: int
) -> None:
    """Hemisphere contour + look ticks + fuse plate + per-site sensor plates."""
    import pygame

    from genome_shell import project_3d_batch

    if not device.enabled or host._view_surf is None:
        return
    pose = eye_rig_pose(host, device.side)
    if pose is None:
        return
    update_eye_sensors(host)
    I = np.asarray(getattr(host, f"{device.name}_I", device.intensity), dtype=np.float64)
    frame = host._body_orbit_frame()
    tip_len = 0.022 * float(frame[1]) if frame is not None else 0.05
    conn = eye_connected_mask(host, device.name)
    _draw_eye_group_box(host, device, pose, conn, cx, cy, view_lim_y)
    overlay = (
        host._overlay_surface(view_lim_y)
        if hasattr(host, "_overlay_surface")
        else None
    )
    surf = overlay if overlay is not None else host._view_surf
    base_col = pose[0]["color"]
    rim_rgb = tuple(int(np.clip(c * 0.55, 0, 255)) for c in base_col)
    for poly in eye_hemisphere_contour(host, device.side):
        sx, sy = project_3d_batch(
            poly, host.view_angle_x, host.view_angle_y, cx, cy, host.scale_3d
        )
        pts2 = [
            (int(sx[i]), int(sy[i]))
            for i in range(len(sx))
            if np.isfinite(sx[i]) and np.isfinite(sy[i])
        ]
        if len(pts2) >= 2:
            pygame.draw.aalines(surf, rim_rgb, False, pts2)
    for i, oc in enumerate(pose):
        val = float(I[i]) if i < len(I) else 0.0
        col = oc["color"]
        bri = 0.35 + 0.65 * val
        rgb = tuple(int(np.clip(c * bri, 0, 255)) for c in col)
        lens = oc["lens"]
        axis = oc["axis"]
        tip = lens + axis * (tip_len * 2.4)
        pts = np.stack([lens, tip], axis=0).astype(np.float32)
        sx, sy = project_3d_batch(
            pts, host.view_angle_x, host.view_angle_y, cx, cy, host.scale_3d
        )
        if not (np.isfinite(sx).all() and np.isfinite(sy).all()):
            continue
        ok = bool(conn[i]) if i < len(conn) else False
        lx, ly = int(sx[0]), int(sy[0])
        pygame.draw.line(
            surf,
            rgb,
            (lx, ly),
            (int(sx[1]), int(sy[1])),
            2 if ok else 1,
        )
        core = max(2, int(3 + 4 * val))
        pygame.draw.circle(surf, (*rgb, 220) if overlay is not None else rgb, (lx, ly), core)
        tag = EYE_SITE_LABELS[i] + ("*" if ok else "")
        txt = f"{tag} {val:.2f}"
        lab = host.hint_font.render(txt, True, (200, 240, 235))
        pad_x, pad_y = 5, 2
        bw = lab.get_width() + pad_x * 2
        bh = lab.get_height() + pad_y * 2
        bx = lx + core + 4
        by = ly - bh // 2
        bx = int(np.clip(bx, 2, max(2, int(host.view_w) - bw - 2)))
        by = int(np.clip(by, 2, max(2, view_lim_y - bh - 2)))
        plate = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(
            plate, (16, 28, 30, 200), plate.get_rect(), border_radius=3
        )
        pygame.draw.rect(
            plate, (*rgb, 210), plate.get_rect(), width=1, border_radius=3
        )
        bar_w = max(1, int((bw - 4) * np.clip(val, 0.0, 1.0)))
        pygame.draw.rect(plate, (*rgb, 160), pygame.Rect(2, bh - 3, bar_w, 2))
        plate.blit(lab, (pad_x, pad_y))
        surf.blit(plate, (bx, by))
    if overlay is not None:
        host._view_surf.blit(overlay, (0, 0))

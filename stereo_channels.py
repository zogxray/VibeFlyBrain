"""Two eye→wing channels: pick Fuse partners by reach, not only by distance.

Fuse used to hand every clamp its geometrically nearest legal partner. For a
compound eye that lands on optic-lobe cells that go nowhere, so the only route
left to the wing pads was one long shared corridor: both eyes reached all four
pads at the same hop depth and ``yaw_cmd`` came out ~0 — a common-mode
projection no credit rule can turn into a steer signal.

Here the same two pick sites are chosen by *where they lead*: an eye enters
through optic/VPN somata that reach its own pad pair in fewer hops than the
other pair, and a pad listens to descending/motor somata that its own eye
reaches first. Candidates stay geometrically local (nearest-N to the lens / to
the pad) and pads stay side-matched, so no edge skips tissue — only the choice
among legal partners changes.

Which eye owns which pad comes from the turret's own convention rather than
from device names: ``_turret_motor_commands`` makes pads 0,1 add +yaw and
``_turret_rig_pose`` aims +yaw toward +X, so pads 0,1 belong to the eye facing
+X. Pads sit opposite the eye that drives them — the channel is crossed, as
wing steering is in the fly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from device import POLICY_EYE

STEREO_ROUNDS = 2  # alternations of (eye entry ↔ pad partner)
STEREO_EYE_CAND = 300  # nearest optic/VPN somata considered per look site
STEREO_PAD_CAND = 300  # nearest descending/motor somata considered per pad
STEREO_PAD_KEEP = 260  # kept after the same-side gate
STEREO_BIAS_W = 1.0  # weight on (own-side hops − other-side hops)
STEREO_DEPTH_W = 0.35  # also keep the absolute path short
STEREO_MAX_ADV = 4  # cap the bias term: a private partner is good, not infinite
STEREO_UNREACHED = 99  # hop stand-in for "no path"
STEREO_HOP_LIMIT = 32

EYE_CLASSES = ("optic", "visual_projection")
PAD_CLASSES = ("descending", "motor")


@dataclass
class StereoPlan:
    """Fuse partners for one CNS: per eye look site, per wing pad."""

    eye_goals: dict[str, list[list[int]]] = field(default_factory=dict)
    pad_sources: list[list[int]] = field(default_factory=list)
    pad_eye: list[str] = field(default_factory=list)
    readout: str = ""

    def goals_for(self, device_name: str, site: int) -> list[int] | None:
        rows = self.eye_goals.get(str(device_name))
        if not rows or not (0 <= int(site) < len(rows)):
            return None
        return list(rows[int(site)]) or None

    def sources_for(self, site: int) -> list[int] | None:
        if not (0 <= int(site) < len(self.pad_sources)):
            return None
        return list(self.pad_sources[int(site)]) or None


def _adjacency(fly: Any) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Forward / reverse CNS tract graph (hop recipe, before any scope clip)."""
    pre, post = fly._ensure_hop_recipe()
    fwd: dict[int, list[int]] = {}
    rev: dict[int, list[int]] = {}
    for a, b in zip(pre.tolist(), post.tolist()):
        fwd.setdefault(int(a), []).append(int(b))
        rev.setdefault(int(b), []).append(int(a))
    return fwd, rev


def _hops(adj: dict[int, list[int]], sources: list[int]) -> dict[int, int]:
    dist = {int(s): 0 for s in sources}
    q = deque(dist.keys())
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= STEREO_HOP_LIMIT:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = d + 1
                q.append(v)
    return dist


def _class_pool(host: Any, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Atlas somata of the given classes (device clamps/packs excluded)."""
    fly = host.fly
    class_names = list(getattr(fly, "class_names", None) or [])
    want = [class_names.index(n) for n in names if n in class_names]
    ids = np.asarray(fly.soma_ids, dtype=np.int64)
    cls = np.asarray(fly.soma_cls, dtype=np.int32)
    xyz = np.asarray(fly.soma_xyz, dtype=np.float64)
    if not want or len(ids) == 0 or len(ids) != len(cls):
        return np.zeros((0,), np.int64), np.zeros((0, 3), np.float64)
    m = np.isin(cls, np.asarray(want, dtype=np.int32))
    dev = host.all_device_neuron_ids() if hasattr(host, "all_device_neuron_ids") else set()
    if dev:
        m &= ~np.isin(ids, np.fromiter(dev, dtype=np.int64, count=len(dev)))
    return ids[m], xyz[m]


def _score(own: dict[int, int], other: dict[int, int], nid: int) -> float:
    """Lower = better partner: own side close, other side far.

    A cell the other eye cannot reach at all is the cleanest channel, but the
    bonus is capped — otherwise the pick runs off to cells nothing reaches,
    which reads as a perfect split and carries no drive.
    """
    a = own.get(int(nid), STEREO_UNREACHED)
    if a >= STEREO_UNREACHED:
        return float("inf")
    b = other.get(int(nid), STEREO_UNREACHED)
    adv = max(float(a - b), -float(STEREO_MAX_ADV))
    return STEREO_BIAS_W * adv + STEREO_DEPTH_W * float(a)


def _pick(
    cands: list[int],
    own: dict[int, int],
    other: dict[int, int],
    *,
    n: int,
    used: set[int],
) -> list[int]:
    scored = [(_score(own, other, g), k, int(g)) for k, g in enumerate(cands)]
    scored.sort()
    out: list[int] = []
    for s, _k, g in scored:
        if s == float("inf"):
            break
        if g in used or g in out:
            continue
        out.append(g)
        if len(out) >= n:
            break
    if not out:  # nothing reachable — keep the nearest legal partner
        out = [int(g) for g in cands[:n]]
    used.update(out)
    return out


def _pad_eye_owner(host: Any, sites: np.ndarray, eyes: list[Any]) -> list[str]:
    """Pad index → name of the eye whose bearing that pad must null.

    Pads 0,1 push ``yaw_cmd`` positive (``_turret_motor_commands``) and +yaw
    aims the barrel toward +X (``_turret_rig_pose``), so they answer to the eye
    that looks toward +X. Nothing reads the device name.
    """
    from compound_eyes import eye_rig_pose

    facing: dict[str, float] = {}
    for d in eyes:
        pose = eye_rig_pose(host, d.side)
        if not pose:
            continue
        facing[str(d.name)] = float(np.asarray(pose[0]["outward"], dtype=np.float64)[0])
    if len(facing) < 2:
        return []
    plus_x = max(facing, key=lambda k: facing[k])
    minus_x = min(facing, key=lambda k: facing[k])
    if plus_x == minus_x:
        return []
    return [plus_x if si in (0, 1) else minus_x for si in range(len(sites))]


def plan_stereo_channels(host: Any, *, rounds: int | None = None) -> StereoPlan | None:
    """Choose eye entry points and wing-pad partners so the two sides separate."""
    from compound_eyes import eye_rig_pose, iter_eye_devices
    from genome_shell import TURRET_TIPS_PER_SITE

    rounds = STEREO_ROUNDS if rounds is None else int(rounds)
    fly = getattr(host, "fly", None)
    if fly is None or not getattr(fly, "neurons", None):
        return None
    pose = host._turret_rig_pose() if hasattr(host, "_turret_rig_pose") else None
    if pose is None:
        return None
    eyes = [d for d in iter_eye_devices(host) if getattr(d, "enabled", True)]
    if len(eyes) < 2:
        return None
    sites = np.asarray(pose["sites"], dtype=np.float64)
    pad_eye = _pad_eye_owner(host, sites, eyes)
    if not pad_eye:
        return None

    pad_ids, pad_xyz = _class_pool(host, PAD_CLASSES)
    eye_ids, eye_xyz = _class_pool(host, EYE_CLASSES)
    if len(pad_ids) == 0 or len(eye_ids) == 0:
        return None

    mid_x = float(np.median(np.asarray(fly.soma_xyz, dtype=np.float64)[:, 0]))
    pad_cands: list[list[int]] = []
    for s in sites:
        d = np.linalg.norm(pad_xyz - s[None, :], axis=1)
        near = np.argsort(d)[: int(STEREO_PAD_CAND)]
        same = (pad_xyz[near, 0] < mid_x) == (float(s[0]) < mid_x)
        keep = near[same][: int(STEREO_PAD_KEEP)]
        if len(keep) == 0:
            keep = near[: int(STEREO_PAD_KEEP)]
        pad_cands.append([int(pad_ids[j]) for j in keep])

    eye_cands: dict[str, list[list[int]]] = {}
    for d in eyes:
        rig = eye_rig_pose(host, d.side)
        if not rig:
            return None
        rows: list[list[int]] = []
        for oc in rig:
            lens = np.asarray(oc["lens"], dtype=np.float64)
            dist = np.linalg.norm(eye_xyz - lens[None, :], axis=1)
            near = np.argsort(dist)[: int(STEREO_EYE_CAND)]
            rows.append([int(eye_ids[j]) for j in near])
        eye_cands[str(d.name)] = rows

    fwd, rev = _adjacency(fly)
    eye_names = sorted(eye_cands.keys())
    if len(eye_names) != 2:
        return None
    other_eye = {eye_names[0]: eye_names[1], eye_names[1]: eye_names[0]}
    per_eye = int(POLICY_EYE.tips_per_site)
    per_pad = int(TURRET_TIPS_PER_SITE)

    # Seed with today's behaviour: the nearest legal partner per pad.
    pad_sources = [c[: int(per_pad)] for c in pad_cands]
    eye_goals: dict[str, list[list[int]]] = {
        name: [row[: int(per_eye)] for row in rows] for name, rows in eye_cands.items()
    }

    for _ in range(max(1, int(rounds))):
        by_eye: dict[str, list[int]] = {n: [] for n in eye_names}
        for si, gs in enumerate(pad_sources):
            by_eye[pad_eye[si]].extend(gs)
        if not all(by_eye[n] for n in eye_names):
            return None
        # Upstream: hops from a soma down to each eye's own pad pair.
        reach = {n: _hops(rev, by_eye[n]) for n in eye_names}
        for name in eye_names:
            own, other = reach[name], reach[other_eye[name]]
            used: set[int] = set()
            eye_goals[name] = [
                _pick(row, own, other, n=per_eye, used=used)
                for row in eye_cands[name]
            ]
        # Downstream: hops from each eye's entry points out to the candidates.
        seen = {n: _hops(fwd, [g for row in eye_goals[n] for g in row]) for n in eye_names}
        used_pads: set[int] = set()
        for si, cands in enumerate(pad_cands):
            own_name = pad_eye[si]
            pad_sources[si] = _pick(
                cands,
                seen[own_name],
                seen[other_eye[own_name]],
                n=per_pad,
                used=used_pads,
            )

    seen = {n: _hops(fwd, [g for row in eye_goals[n] for g in row]) for n in eye_names}
    parts: list[str] = []
    for si, gs in enumerate(pad_sources):
        own_name = pad_eye[si]
        d_own = min((seen[own_name].get(g, STEREO_UNREACHED) for g in gs), default=-1)
        d_oth = min(
            (seen[other_eye[own_name]].get(g, STEREO_UNREACHED) for g in gs), default=-1
        )
        parts.append(f"pad{si}←{own_name.replace('eye_', '')} {d_own}/{d_oth}")
    return StereoPlan(
        eye_goals=eye_goals,
        pad_sources=pad_sources,
        pad_eye=pad_eye,
        readout="stereo " + " · ".join(parts) + " (own/other hops)",
    )

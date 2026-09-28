"""Training scope: graph paths sensor→motor (not geometric funnel ∩).

Funnel wells stay in DeviceRegion for render/growth only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeviceRegion:
    """One well / ball around a device — render & tip gravity only."""

    name: str
    neuron_ids: set[int] = field(default_factory=set)
    center: np.ndarray | None = None
    radius: float = 0.0

    def __post_init__(self) -> None:
        if self.center is not None:
            self.center = np.asarray(self.center, dtype=np.float64)


def _bfs(
    graph: dict[int, list[int]],
    starts: set[int],
    *,
    max_hops: int | None = None,
) -> set[int]:
    return set(_bfs_dist(graph, starts, max_hops=max_hops).keys())


def _bfs_dist(
    graph: dict[int, list[int]],
    starts: set[int],
    *,
    max_hops: int | None = None,
) -> dict[int, int]:
    """BFS distances from starts (optional hop cap)."""
    dist = {int(s): 0 for s in starts}
    frontier = list(dist.keys())
    i = 0
    while i < len(frontier):
        u = frontier[i]
        i += 1
        if max_hops is not None and dist[u] >= int(max_hops):
            continue
        for v in graph.get(u, ()):
            v = int(v)
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            frontier.append(v)
    return dist


def build_adj(
    pre: np.ndarray, post: np.ndarray
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    adj: dict[int, list[int]] = {}
    rev: dict[int, list[int]] = {}
    for a, b in zip(pre.tolist(), post.tolist()):
        ai, bi = int(a), int(b)
        adj.setdefault(ai, []).append(bi)
        rev.setdefault(bi, []).append(ai)
    return adj, rev


@dataclass
class ScopeMetrics:
    """UI / log meters for a live training scope."""

    n_sensors: int = 0
    n_motors: int = 0
    motors_hit: int = 0
    path_neurons: int = 0
    path_edges: int = 0
    dw_edges: int = 0  # contact_gain edges touching corridor
    best_hops: int | None = None
    hops_floor: int | None = None
    group_best_hops: dict[str, int] = field(default_factory=dict)
    ok: bool = False
    detail: str = ""

    def readout(self) -> str:
        hops = "—" if self.hops_floor is None else str(self.hops_floor)
        return (
            f"paths {self.motors_hit}/{self.n_motors} · "
            f"n={self.path_neurons:,} · e={self.path_edges:,} · "
            f"Δw {self.dw_edges} · hops≥{hops}"
        )

    def groups_readout(self) -> str:
        if not self.group_best_hops:
            return "—"
        return " ".join(
            f"{name}:{hops}" for name, hops in sorted(self.group_best_hops.items())
        )


@dataclass
class TrainingScope:
    """Learn/hop domain = neurons on near-shortest sensor→motor paths.

    The bed is built **per sensor group** (one group per device: ocelli, eyeL,
    eyeR, joL, joR, turret proprio) and unioned. A single multi-source BFS
    measures every node from its *nearest* sensor, so a short proprio→motor
    geodesic clips the long optic→DN corridor out of the scope entirely.

    Per group g: ∪_motors { u | dist(g→u) + dist(u→m) ≤ dist(g→m) + margin }.
    ``corridor`` = path_neurons ∪ endpoints (clamps).
    ``regions`` — optional geometric wells (funnel draw only).
    """

    path_neurons: set[int] = field(default_factory=set)
    endpoint_ids: set[int] = field(default_factory=set)
    sensor_ids: set[int] = field(default_factory=set)
    motor_ids: set[int] = field(default_factory=set)
    motors_reached: set[int] = field(default_factory=set)
    path_edges: set[tuple[int, int]] = field(default_factory=set)
    best_hops: int | None = None
    # Per sensor group: shortest group→motor geodesic / motors reached.
    group_best_hops: dict[str, int] = field(default_factory=dict)
    group_motors_hit: dict[str, int] = field(default_factory=dict)
    path_margin: int = 2
    regions: list[DeviceRegion] = field(default_factory=list)  # funnel only
    max_hops: int = 16

    @property
    def corridor(self) -> set[int]:
        return set(self.path_neurons) | set(self.endpoint_ids)

    @property
    def hops_floor(self) -> int | None:
        """Beats needed for the *slowest* sensor group's wave to land.

        ``best_hops`` is the nearest sensor (usually turret proprio); hopping
        that few beats never lets the eye wave out of the optic lobe.
        """
        if self.group_best_hops:
            return int(max(self.group_best_hops.values()))
        return self.best_hops

    @property
    def ok(self) -> bool:
        return bool(self.motors_reached) and bool(self.path_neurons)

    def metrics(self, *, dw_edges: int = 0) -> ScopeMetrics:
        return ScopeMetrics(
            n_sensors=len(self.sensor_ids),
            n_motors=len(self.motor_ids),
            motors_hit=len(self.motors_reached),
            path_neurons=len(self.path_neurons),
            path_edges=len(self.path_edges),
            dw_edges=int(dw_edges),
            best_hops=self.best_hops,
            hops_floor=self.hops_floor,
            group_best_hops=dict(self.group_best_hops),
            ok=self.ok,
            detail="",
        )

    def readout(self, *, dw_edges: int = 0) -> str:
        m = self.metrics(dw_edges=dw_edges)
        hops = "—" if m.hops_floor is None else str(m.hops_floor)
        near = "—" if m.best_hops is None else str(m.best_hops)
        return (
            f"paths {m.motors_hit}/{m.n_motors} · "
            f"n={m.path_neurons:,} · e={m.path_edges:,} · "
            f"Δw {m.dw_edges} · hops≥{hops} (near {near}) · +{int(self.path_margin)}"
        )

    def region_map(self) -> dict[str, set[int]]:
        return {r.name: set(r.neuron_ids) for r in self.regions}

    @classmethod
    def from_regions(
        cls,
        regions: list[DeviceRegion],
        *,
        endpoint_ids: set[int] | None = None,
    ) -> TrainingScope:
        """Attach wells for draw only — path_neurons stay empty (not a train scope)."""
        return cls(
            regions=list(regions),
            endpoint_ids=set(endpoint_ids or ()),
            path_neurons=set(),
        )

    @classmethod
    def from_graph(
        cls,
        pre: np.ndarray,
        post: np.ndarray,
        *,
        sensors: set[int],
        motors: set[int],
        max_hops: int = 16,
        path_margin: int = 2,
        regions: list[DeviceRegion] | None = None,
        sensor_groups: dict[str, set[int]] | None = None,
    ) -> TrainingScope:
        """Union of per-sensor-group near-shortest beds (geodesic + margin).

        ``sensor_groups`` — one entry per device (eye_left, jo_right, …). Each
        group gets its own BFS and its own geodesic limit, so a long optic path
        is not clipped by a short proprio one. Without groups this degrades to
        the old single multi-source BFS.
        """
        sensors = {int(s) for s in sensors}
        motors = {int(m) for m in motors}
        margin = max(0, int(path_margin))
        groups: dict[str, set[int]] = {}
        for name, ids in (sensor_groups or {}).items():
            gid = {int(x) for x in ids}
            if gid:
                groups[str(name)] = gid
                sensors |= gid
        if not groups and sensors:
            groups = {"sensors": set(sensors)}
        if not sensors or not motors or len(pre) == 0:
            return cls(
                sensor_ids=sensors,
                motor_ids=motors,
                endpoint_ids=sensors | motors,
                regions=list(regions or ()),
                max_hops=int(max_hops),
                path_margin=margin,
            )
        adj, rev = build_adj(pre, post)
        path: set[int] = set()
        motors_hit: set[int] = set()
        best: int | None = None
        group_best: dict[str, int] = {}
        group_hit: dict[str, int] = {}
        # One reverse ball per (motor, limit) — groups often share limits.
        rev_balls: dict[tuple[int, int], dict[int, int]] = {}
        for gname, gids in groups.items():
            dist_s = _bfs_dist(adj, gids, max_hops=max_hops)
            g_best: int | None = None
            g_hit = 0
            for m in motors:
                if m not in dist_s:
                    continue
                d_m = int(dist_s[m])
                motors_hit.add(m)
                g_hit += 1
                if g_best is None or d_m < g_best:
                    g_best = d_m
                # Nodes on paths group→m of length ≤ shortest + margin.
                lim = d_m + margin
                dist_from_m = rev_balls.get((m, lim))
                if dist_from_m is None:
                    dist_from_m = _bfs_dist(rev, {m}, max_hops=lim)
                    rev_balls[(m, lim)] = dist_from_m
                for u, d_um in dist_from_m.items():
                    d_su = dist_s.get(u)
                    if d_su is None:
                        continue
                    if int(d_su) + int(d_um) <= lim:
                        path.add(int(u))
            group_hit[gname] = g_hit
            if g_best is not None:
                group_best[gname] = int(g_best)
                if best is None or g_best < best:
                    best = int(g_best)
        path_edges: set[tuple[int, int]] = set()
        if path:
            for a, b in zip(pre.tolist(), post.tolist()):
                ai, bi = int(a), int(b)
                if ai in path and bi in path:
                    path_edges.add((ai, bi))
        return cls(
            path_neurons=path,
            endpoint_ids=sensors | motors,
            sensor_ids=sensors,
            motor_ids=motors,
            motors_reached=motors_hit,
            path_edges=path_edges,
            best_hops=best,
            group_best_hops=group_best,
            group_motors_hit=group_hit,
            path_margin=margin,
            regions=list(regions or ()),
            max_hops=int(max_hops),
        )

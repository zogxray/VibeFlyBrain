"""Geometric device wells — render / tip gravity only. Not TrainingScope.

Training corridor lives in ``scope.TrainingScope`` (graph paths).
Well balls are independent: grow thresholds ≠ path-gate / FunnelGate.learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np

from scope import DeviceRegion


@dataclass
class WellGrowParams:
    """When to stop growing wells toward each other (geometry only)."""

    min_a: int = 30
    min_b: int = 30
    min_n: int = 20  # |A∩B| soft stop
    min_neck_class: int = 0  # 0 = ignore AN/DN count for grow
    local_k: int = 140
    local_q: float = 0.55
    r_min_frac: float = 0.04
    r_max_frac: float = 0.42
    r_max_gap: float = 0.72
    grow_steps: int = 28
    grow_frac: float = 0.035
    # Cache: bucket soma count while rising so wells aren't rebuilt every tip.
    rising_bucket: int = 500

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> WellGrowParams:
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: raw[k] for k in known if k in raw})

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_funnel_gate(cls, gate: Any | None) -> WellGrowParams:
        """Map legacy FunnelGate size fields; ignore min_paths (training-only)."""
        if gate is None:
            return cls()
        return cls(
            min_a=int(getattr(gate, "min_a", 30)),
            min_b=int(getattr(gate, "min_b", 30)),
            min_n=int(getattr(gate, "min_n", 20)),
            min_neck_class=int(getattr(gate, "min_neck_class", 0)),
        )


@dataclass
class WellSpec:
    name: str
    center: np.ndarray
    radius: float
    rgb: tuple[int, int, int] = (140, 170, 200)
    neuron_ids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.radius = float(self.radius)
        self.neuron_ids = {int(x) for x in self.neuron_ids}

    def as_draw_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "center": self.center,
            "r": float(self.radius),
            "rgb": self.rgb,
        }

    def to_region(self) -> DeviceRegion:
        return DeviceRegion(
            name=self.name,
            neuron_ids=set(self.neuron_ids),
            center=self.center.copy(),
            radius=float(self.radius),
        )


DEFAULT_COLORS: dict[str, tuple[int, int, int]] = {
    "ocelli": (120, 190, 230),
    "turret": (230, 160, 110),
}


def neurons_in_ball(
    soma_xyz: np.ndarray,
    soma_ids: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> set[int]:
    if len(soma_xyz) == 0 or radius <= 0:
        return set()
    xyz = np.asarray(soma_xyz, dtype=np.float64)
    ids = np.asarray(soma_ids, dtype=np.int64)
    c = np.asarray(center, dtype=np.float64)
    d = np.linalg.norm(xyz - c[None, :], axis=1)
    return set(int(i) for i in ids[d <= float(radius)].tolist())


def tissue_centroid_near(
    soma_xyz: np.ndarray,
    anchor: np.ndarray,
    seed_r: float,
    *,
    local_k: int = 140,
) -> np.ndarray:
    a = np.asarray(anchor, dtype=np.float64)
    if len(soma_xyz) == 0 or seed_r <= 0:
        return a
    xyz = np.asarray(soma_xyz, dtype=np.float64)
    d = np.linalg.norm(xyz - a[None, :], axis=1)
    near = d <= float(seed_r)
    n = int(np.count_nonzero(near))
    if n < 12:
        k = min(int(local_k), len(d))
        if k < 8:
            return a
        idx = np.argpartition(d, k - 1)[:k]
        return xyz[idx].mean(axis=0)
    return xyz[near].mean(axis=0)


def local_well_radius(
    soma_xyz: np.ndarray,
    center: np.ndarray,
    gap: float,
    span: float,
    params: WellGrowParams,
) -> float:
    r_min = float(params.r_min_frac) * span
    r_max = min(
        float(params.r_max_frac) * span,
        float(params.r_max_gap) * max(gap, 1e-6),
    )
    r_max = max(r_max, r_min)
    if len(soma_xyz) == 0:
        return 0.5 * (r_min + r_max)
    xyz = np.asarray(soma_xyz, dtype=np.float64)
    d = np.linalg.norm(xyz - np.asarray(center, dtype=np.float64)[None, :], axis=1)
    k = min(int(params.local_k), len(d))
    if k < 8:
        return 0.5 * (r_min + r_max)
    nearest = np.partition(d, k - 1)[:k]
    r = float(np.quantile(nearest, float(params.local_q)))
    return float(np.clip(r, r_min, r_max))


def well_radius_cap(gap: float, span: float, params: WellGrowParams) -> float:
    return float(
        max(
            float(params.r_min_frac) * span,
            min(
                float(params.r_max_frac) * span,
                float(params.r_max_gap) * max(gap, 1e-6),
            ),
        )
    )


@dataclass
class WellField:
    """Cached pair (or list) of device wells. Never writes TrainingScope."""

    specs: list[WellSpec] = field(default_factory=list)
    cache_key: tuple | None = None
    gravity_key: tuple | None = None  # last push to fly.device_funnel_*

    def clear(self) -> None:
        self.specs = []
        self.cache_key = None
        self.gravity_key = None

    def region_map(self) -> dict[str, set[int]]:
        return {s.name: set(s.neuron_ids) for s in self.specs}

    def regions(self) -> list[DeviceRegion]:
        return [s.to_region() for s in self.specs]

    def as_layers(self) -> dict[str, set[int]]:
        """A/B/∩ for grow diagnostics (not training neck)."""
        by = self.region_map()
        a = by.get("ocelli", set())
        b = by.get("turret", set())
        if not a and not b and len(self.specs) >= 2:
            a = set(self.specs[0].neuron_ids)
            b = set(self.specs[1].neuron_ids)
        inter = a & b if a and b else set()
        return {
            "ocelli": a,
            "turret": b,
            "inter": inter,
            "union": a | b,
            **{f"region:{k}": set(v) for k, v in by.items()},
        }

    def size_ok(
        self,
        params: WellGrowParams,
        *,
        neck_class_count: Callable[[set[int]], int] | None = None,
    ) -> bool:
        layers = self.as_layers()
        if len(layers["ocelli"]) < int(params.min_a):
            return False
        if len(layers["turret"]) < int(params.min_b):
            return False
        inter = layers["inter"]
        if len(inter) < int(params.min_n):
            return False
        if int(params.min_neck_class) > 0 and neck_class_count is not None:
            if int(neck_class_count(inter)) < int(params.min_neck_class):
                return False
        return True

    def draw_specs(self, *, names: set[str] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in self.specs:
            if names is not None and s.name not in names:
                continue
            out.append(s.as_draw_dict())
        return out

    def metrics_dict(self) -> dict[str, Any]:
        layers = self.as_layers()
        return {
            "A": len(layers["ocelli"]),
            "B": len(layers["turret"]),
            "N": len(layers["inter"]),
            "wells": len(self.specs),
            "names": [s.name for s in self.specs],
        }

    @staticmethod
    def stable_cache_key(
        host: Any,
        centers: tuple[np.ndarray, np.ndarray],
        params: WellGrowParams,
    ) -> tuple:
        """Avoid ``len(soma_xyz)`` every frame — bucket while rising, ignore adult."""
        fly = host.fly
        a_oc, a_tu = centers
        adult = bool(fly.is_adult()) if hasattr(fly, "is_adult") else False
        xyz = getattr(fly, "soma_xyz", None)
        n = int(len(xyz)) if xyz is not None else 0
        if adult:
            n_stamp = -1
        else:
            bucket = max(1, int(params.rising_bucket))
            n_stamp = n // bucket
        return (
            id(fly),
            n_stamp,
            round(float(a_oc[0]), 3),
            round(float(a_oc[1]), 3),
            round(float(a_oc[2]), 3),
            round(float(a_tu[0]), 3),
            round(float(a_tu[1]), 3),
            round(float(a_tu[2]), 3),
            int(params.min_a),
            int(params.min_b),
            int(params.min_n),
        )

    def rebuild_pair(
        self,
        host: Any,
        c_oc_anchor: np.ndarray,
        c_tu_anchor: np.ndarray,
        *,
        params: WellGrowParams,
        span: float,
        extra_ids: dict[str, set[int]] | None = None,
        neck_class_count: Callable[[set[int]], int] | None = None,
    ) -> list[WellSpec]:
        """Seat + grow two wells. Does not touch TrainingScope."""
        soma_xyz = np.asarray(host.fly.soma_xyz, dtype=np.float64)
        soma_ids = np.asarray(host.fly.soma_ids, dtype=np.int64)
        extra = extra_ids or {}
        gap0 = float(np.linalg.norm(c_tu_anchor - c_oc_anchor))
        if gap0 < 1e-6:
            self.specs = []
            return []
        seed = max(0.14 * gap0, 0.09 * span)
        c_oc = tissue_centroid_near(
            soma_xyz, c_oc_anchor, seed, local_k=params.local_k
        )
        c_tu = tissue_centroid_near(
            soma_xyz, c_tu_anchor, seed, local_k=params.local_k
        )
        gap = float(np.linalg.norm(c_tu - c_oc))
        if gap < 1e-6:
            gap = gap0
        r_oc = local_well_radius(soma_xyz, c_oc, gap, span, params)
        r_tu = local_well_radius(soma_xyz, c_tu, gap, span, params)
        r_cap = well_radius_cap(gap, span, params)
        step = float(params.grow_frac) * gap

        def _fill(ro: float, rt: float) -> list[WellSpec]:
            ids_a = neurons_in_ball(soma_xyz, soma_ids, c_oc, ro) | set(
                extra.get("ocelli", ())
            )
            ids_b = neurons_in_ball(soma_xyz, soma_ids, c_tu, rt) | set(
                extra.get("turret", ())
            )
            return [
                WellSpec(
                    name="ocelli",
                    center=c_oc,
                    radius=float(ro),
                    rgb=DEFAULT_COLORS["ocelli"],
                    neuron_ids=ids_a,
                ),
                WellSpec(
                    name="turret",
                    center=c_tu,
                    radius=float(rt),
                    rgb=DEFAULT_COLORS["turret"],
                    neuron_ids=ids_b,
                ),
            ]

        specs = _fill(r_oc, r_tu)
        self.specs = specs
        for _ in range(int(params.grow_steps)):
            if self.size_ok(params, neck_class_count=neck_class_count):
                break
            if r_oc >= r_cap - 1e-9 and r_tu >= r_cap - 1e-9:
                break
            r_oc = float(min(r_oc + step, r_cap))
            r_tu = float(min(r_tu + step, r_cap))
            specs = _fill(r_oc, r_tu)
            self.specs = specs
        return list(self.specs)

    def ensure_pair(
        self,
        host: Any,
        centers: tuple[np.ndarray, np.ndarray] | None,
        *,
        params: WellGrowParams,
        span: float,
        extra_ids: dict[str, set[int]] | None = None,
        neck_class_count: Callable[[set[int]], int] | None = None,
    ) -> list[WellSpec]:
        if centers is None:
            self.clear()
            return []
        key = self.stable_cache_key(host, centers, params)
        if key == self.cache_key and self.specs:
            return list(self.specs)
        self.rebuild_pair(
            host,
            centers[0],
            centers[1],
            params=params,
            span=span,
            extra_ids=extra_ids,
            neck_class_count=neck_class_count,
        )
        self.cache_key = key
        return list(self.specs)

    def sync_gravity(
        self,
        fly: Any,
        *,
        visible_names: set[str],
        gravity: float,
    ) -> bool:
        """Push wells into fly tip attractors. Returns True if updated."""
        vis = [s for s in self.specs if s.name in visible_names]
        gkey = (
            tuple(sorted(visible_names)),
            tuple(
                (s.name, round(float(s.radius), 4), tuple(np.round(s.center, 4)))
                for s in vis
            ),
            round(float(gravity), 4),
        )
        if gkey == self.gravity_key:
            return False
        self.gravity_key = gkey
        if not vis:
            fly.device_funnel_xyz = np.zeros((0, 3), dtype=np.float32)
            fly.device_funnel_r = np.zeros((0,), dtype=np.float32)
            fly.device_funnel_w = 0.0
            return True
        fly.device_funnel_xyz = np.stack([s.center for s in vis], axis=0).astype(
            np.float32
        )
        fly.device_funnel_r = np.asarray([s.radius for s in vis], dtype=np.float32)
        fly.device_funnel_w = (
            float(gravity) if len(vis) >= 2 else 0.55 * float(gravity)
        )
        return True

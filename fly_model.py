"""CNS fly model: neurons (soma + axons/dendrites) organized in sets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

NeuriteKind = Literal["axon", "dendrite"]
NeuriteStatus = Literal["idle", "growing", "seeking", "connected", "failed"]
# Life-cycle: rising = developmental growth; adult = use assembled wiring.
FlyPhase = Literal["rising", "adult"]

# Default seek budget in fly-seconds (1 sim tick = 1 fly-second).
DEFAULT_SEEK_LIMIT = 300.0
FLY_SECONDS_PER_REVOLUTION = 60  # dial: one full turn = 60 fly-seconds

# Recipe-guided axon growth: tips realize atlas edges (pre→post), not random hops.
GROW_STEP_FRAC = 0.0012  # tip step per tick as fraction of soma span
GROW_CONNECT_FRAC = 0.0012  # connect radius as fraction of soma span
GROW_SEEK_LIMIT = 600.0  # fly-seconds to reach assigned partner
GROW_MAX_RECIPE_EDGES = 20000  # tips armed per Start (from genome_quota)
# Adult hop: atlas recipe ∩ tract classes ∪ top weights (∪ grown Fuse bridges).
HOP_RECIPE_MAX = 120_000
HOP_TRACT_CLASSES = (
    "ascending",
    "descending",
    "sensory",
    "motor",
    "central",
    "optic",
    "visual_projection",
)
GROW_TRAIL_MAX = 64  # polyline samples per tip (soma → … → tip)
GROW_TIP_DRAW_MAX = 1800  # subsample tips for overlay if larger
GROW_RAY_DRAW_MAX = 700  # axon polylines drawn per frame (full CNS)
GROW_POLY_DRAW_POINTS = 8  # max samples kept per axon trail when drawing
GROW_SEG_DRAW_MAX = 3200  # hard cap on axon segments per frame
GROW_DEND_STUB_FRAC = 0.03  # dendrite stub length as fraction of soma span
GROW_DEND_STUB_MAX = 0.9  # hard cap on dendrite stub length
GROW_DEND_POLY_N = 5  # samples on materialized dendrite (contact → cable)
# Chemotaxis: group + cell fields instruct direction. Tissue is substrate only.
GROW_TISSUE_K = 6  # nearest process samples (substrate probe)
GROW_TISSUE_TREE_MAX = 80000  # KDTree subsample of process cloud
GROW_TISSUE_MODES = 2  # density modes of process cloud (e.g. brain / cord)
GROW_SUBSTRATE_FRAC = 0.018  # farther than this × span from tissue → air correction
GROW_SUBSTRATE_W = 0.55  # max weight of substrate pull when off the cloud
GROW_SUBSTRATE_NECK_W = 0.40  # soft lock in neck slab (hard lock trapped AN/DN)
GROW_GROUP_CELL_BLEND = 0.18  # residual cell pull while still in foreign mode
GROW_NECK_CAPTURE_FRAC = 0.09  # cross-mode tips aim at neck until within this × span
GROW_NECK_DOMINATE = 0.92  # weight toward neck waypoint before crossing
# Contact domain: attractive basin around post process neighborhood.
GROW_CONTACT_K = 8  # process neighborhood size around post soma
GROW_CONTACT_MIN_FRAC = 0.008  # prefer sites at least this × span from post soma
GROW_BASIN_FRAC = 0.028  # Connected = tip inside this × span of domain center
GROW_BASIN_SPREAD = 1.15  # basin_r at least spread of neighborhood samples
# Final approach: last tips near basin were orbiting (fasciculation) → stuck ~99%.
# Latch must cover the late stall band (observed d/br ≈ 3.5–8 for AN/DN).
GROW_FINAL_APPROACH_MUL = 8.0  # d ≤ mul·basin → pure chemo, no peer pull
GROW_FINAL_CONNECT_MUL = 1.30  # soft connect radius in final approach
GROW_FINAL_SNAP_MUL = 3.0  # SEEKING + within mul·basin → snap connect
GROW_FINAL_STEP_MUL = 2.8  # stride boost while homing into the basin
GROW_CROSS_SEEK_EXTRA = 2.0  # seek budget multiplier for brain↔VNC trips
GROW_CROSS_STEP_MUL = 1.85  # longer stride while navigating the neck
# Legacy names kept for call-site compat (no longer blend attractors).
GROW_TISSUE_BLEND = 0.0
GROW_TISSUE_NEAR_GOAL = 0.04
GROW_RIDGE_BLEND = 0.0
GROW_CONNECT_DOMAIN_FRAC = GROW_BASIN_FRAC
GROW_CONNECT_FRAC = 0.0012
# Mutual tip attraction: nearby axons of the same cohort coalesce into cables.
BUNDLE_RADIUS_FRAC = 0.12  # neighbor search radius × soma span
BUNDLE_K = 16  # max peers considered
BUNDLE_BLEND = 0.78  # weight of peer pull vs chemotaxis (ramps with neighbors)
BUNDLE_MIN_PEERS = 1  # start lateral bind with a single nearby peer
BUNDLE_COHORT_MIN = 4  # min tips to form a (pre_cls, post_cls[, side]) cohort
# Pioneer → fasciculation: one navigator per cohort; followers lock to its trail.
# Far from goal: chemotaxis dominates (was full 0.82 trail → AN/DN timed out mid-neck).
PIONEER_TRAIL_BLEND = 0.82  # max follower trail weight near the goal basin
PIONEER_TRAIL_FAR = 0.18  # trail weight floor while still far (long-haul path finding)
PIONEER_TRAIL_SAMPLES = 12  # subsample pioneer poly for nearest-point search
PIONEER_AHEAD_FRAC = 0.20  # prefer a trail sample ahead of nearest (along poly)
PIONEER_PEER_BLEND = 0.22  # residual follower↔follower tip bind (of leftover)

STATUS_IDLE = 0
STATUS_GROWING = 1
STATUS_SEEKING = 2
STATUS_CONNECTED = 3
STATUS_FAILED = 4

_STATUS_TO_CODE: dict[NeuriteStatus, int] = {
    "idle": STATUS_IDLE,
    "growing": STATUS_GROWING,
    "seeking": STATUS_SEEKING,
    "connected": STATUS_CONNECTED,
    "failed": STATUS_FAILED,
}
_CODE_TO_STATUS: dict[int, NeuriteStatus] = {
    v: k for k, v in _STATUS_TO_CODE.items()
}


@dataclass
class Neurite:
    id: int
    kind: NeuriteKind
    points: np.ndarray  # float32 [M, 3]
    status: NeuriteStatus = "idle"
    partner_id: int | None = None  # partner neuron id when connected
    seek_t: float = 0.0  # elapsed time while growing/seeking
    seek_limit: float = DEFAULT_SEEK_LIMIT  # fail if seek_t exceeds this


@dataclass
class Neuron:
    id: int
    soma: np.ndarray  # float32 [3]
    cls: int
    activity: float = 0.0
    axons: list[Neurite] = field(default_factory=list)
    dendrites: list[Neurite] = field(default_factory=list)

    def iter_neurites(self):
        yield from self.axons
        yield from self.dendrites


@dataclass
class NeuronSet:
    name: str
    members: set[int] = field(default_factory=set)


@dataclass
class Fly:
    """In-memory CNS: neurons with neurites, grouped into named sets."""

    neurons: dict[int, Neuron] = field(default_factory=dict)
    sets: dict[str, NeuronSet] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    _next_neurite_id: int = 0
    sim_time: int = 0  # fly-seconds; +1 per full simulation tick
    phase: FlyPhase = "rising"  # rising = grow wiring; adult = run on contacts
    # Active wiring: which atlas sets sprout / receive (names).
    growth_sources: list[str] = field(default_factory=list)
    growth_targets: list[str] = field(default_factory=list)
    # Atlas body ids aligned with soma_ids / neuron indices after load.
    soma_body_id: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int64)
    )
    # Full atlas edge recipe (neuron indices into this Fly).
    recipe_pre: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    recipe_post: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    recipe_weight: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    recipe_armed: int = 0  # edges selected for the current growth run
    # Plastic gain on realized contacts (pre, post) → multiplier; missing ⇒ 1.0.
    contact_gain: dict = field(default_factory=dict)
    # Cached hop-recipe subset (neuron ids); invalidated when recipe changes.
    _hop_recipe_pre: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32), repr=False
    )
    _hop_recipe_post: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32), repr=False
    )
    _hop_recipe_key: object = field(default=None, repr=False)
    # Optional shell extras from .fly meta (device id maps, etc.).
    shell_meta: dict = field(default_factory=dict)
    # Device attractor funnels (shell): centers + radii pull tips in the corridor.
    device_funnel_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    device_funnel_r: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    device_funnel_w: float = 0.0
    # Adult hop lock: only recipe edges with both ends in this id set (empty = all).
    hop_allow_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    # Editable genome: tips quota per (source_class × target_class).
    # Atlas edges only instantiate candidates; this matrix is what Start arms.
    genome_quota: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int32)
    )
    # Per-cell atlas edge indices (weight-sorted) — candidates, not the recipe.
    _genome_edge_bins: dict = field(default_factory=dict, repr=False)
    # Target-atlas process cloud (render/recipe only — not live neurites).
    atlas_process_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    atlas_process_cls: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    # Vectorized growth field (tips) — not per-neuron Python objects.
    growth_owner: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    growth_soma: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    growth_tip: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    growth_status: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.uint8)
    )
    growth_partner: np.ndarray = field(
        default_factory=lambda: np.full((0,), -1, dtype=np.int32)
    )
    # Fixed partner chosen at spawn (not nearest-at-each-tick).
    growth_goal: np.ndarray = field(
        default_factory=lambda: np.full((0,), -1, dtype=np.int32)
    )
    growth_goal_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    # Attractive basin radius around growth_goal_xyz (contact domain).
    growth_basin_r: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    growth_seek_t: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    growth_seek_limit: np.ndarray = field(
        default_factory=lambda: np.full((0,), GROW_SEEK_LIMIT, dtype=np.float32)
    )
    growth_cls: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    # Polyline samples along each tip's path (tissue-guided, not a chord).
    growth_poly: np.ndarray = field(
        default_factory=lambda: np.zeros((0, GROW_TRAIL_MAX, 3), dtype=np.float32)
    )
    growth_poly_n: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    last_growth_step: dict[str, int] = field(
        default_factory=lambda: {"moved": 0, "connected": 0}
    )
    _growth_tree: object | None = field(default=None, repr=False)
    _process_tree: object | None = field(default=None, repr=False)
    _process_tree_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32), repr=False
    )
    # Per-class process KDTrees for contact-domain goals (set cloud).
    _class_process_trees: dict = field(default_factory=dict, repr=False)
    _class_process_xyz: dict = field(default_factory=dict, repr=False)
    # Density modes of the process cloud (inter-cluster ridge routing).
    _tissue_mode_centers: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32), repr=False
    )
    _tissue_mode_labels: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32), repr=False
    )
    # Tissue constriction between modes (cervical connective / "neck").
    _neck_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32), repr=False
    )
    _growth_materialized: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=bool), repr=False
    )
    _growth_tree_tgt_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32), repr=False
    )
    _growth_tree_tgt_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32), repr=False
    )
    # Cohort ids for mutual tip bundling (same pre/post class, optional L/R).
    growth_bundle_id: np.ndarray = field(
        default_factory=lambda: np.full((0,), -1, dtype=np.int32)
    )
    # Per tip: global tip index of that cohort's pioneer (-1 = unassigned).
    growth_pioneer_idx: np.ndarray = field(
        default_factory=lambda: np.full((0,), -1, dtype=np.int32)
    )

    # Render caches (rebuilt from structure)
    soma_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    soma_cls: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    soma_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    soma_activity: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    neurite_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    neurite_cls: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    # Frame-stable hop graph cache (invalidated when recipe/allow/grown change).
    _activity_graph_key: tuple | None = field(default=None, repr=False)
    _activity_graph_cached: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, repr=False
    )
    _elig_rev_key: tuple | None = field(default=None, repr=False)
    _elig_rev_adj: dict[int, list[int]] | None = field(default=None, repr=False)
    # CSR-like hop edges: parallel pre/post/weight (invalidated with graph/gains).
    _hop_edge_key: tuple | None = field(default=None, repr=False)
    _hop_pre_i: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32), repr=False
    )
    _hop_post_i: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32), repr=False
    )
    _hop_w: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32), repr=False
    )
    _contact_gain_version: int = field(default=0, repr=False)

    def alloc_neurite_id(self) -> int:
        nid = self._next_neurite_id
        self._next_neurite_id = nid + 1
        return nid

    def iter_neurites(self):
        for neuron in self.neurons.values():
            yield from neuron.iter_neurites()

    def neurite_status_counts(self) -> dict[NeuriteStatus, int]:
        return self.neurite_stats()["status"]

    def connection_count(self) -> int:
        return int(self.neurite_stats()["status"]["connected"])

    def neurite_stats(self) -> dict:
        """Summary for UI — growth field + any structural neurites."""
        status: dict[NeuriteStatus, int] = {
            "idle": 0,
            "growing": 0,
            "seeking": 0,
            "connected": 0,
            "failed": 0,
        }
        n_ax = 0
        n_den = 0
        if len(self.growth_owner):
            n_ax = int(len(self.growth_owner))
            for neuron in self.neurons.values():
                n_den += len(neuron.dendrites)
        else:
            for neuron in self.neurons.values():
                n_ax += len(neuron.axons)
                n_den += len(neuron.dendrites)
                for neurite in neuron.iter_neurites():
                    status[neurite.status] = status.get(neurite.status, 0) + 1
        if len(self.growth_status):
            for code, name in (
                (STATUS_GROWING, "growing"),
                (STATUS_SEEKING, "seeking"),
                (STATUS_CONNECTED, "connected"),
                (STATUS_FAILED, "failed"),
                (STATUS_IDLE, "idle"),
            ):
                status[name] += int(np.sum(self.growth_status == code))
        elif not self.neurons:
            pass
        # When growth field empty, structural statuses already counted above.
        active_mask = (self.growth_status == STATUS_GROWING) | (
            self.growth_status == STATUS_SEEKING
        )
        active = int(np.sum(active_mask)) if len(self.growth_status) else 0
        timer_min: float | None = None
        timer_max: float | None = None
        if active:
            rem = np.maximum(
                0.0,
                self.growth_seek_limit[active_mask] - self.growth_seek_t[active_mask],
            )
            timer_min = float(rem.min())
            timer_max = float(rem.max())
        return {
            "status": status,
            "axons": n_ax,
            "dendrites": n_den,
            "active": active,
            "timer_min": timer_min,
            "timer_max": timer_max,
            "last_moved": int(self.last_growth_step.get("moved", 0)),
            "last_connected": int(self.last_growth_step.get("connected", 0)),
            "recipe_total": int(len(self.recipe_pre)),
            "recipe_armed": int(self.recipe_armed),
            "realized": int(status["connected"]),
        }

    def recipe_yield(self) -> dict:
        """Recipe Yield: fraction of armed tips that reached their partner.

        Honest denominator is the armed tip set (Start attempt), not full atlas.
        Per-class Out = tips owned by that class; In = tips aimed at that class.
        """
        empty_cls: list[dict] = []
        n = int(len(self.growth_status))
        if n == 0:
            return {
                "armed": 0,
                "ok": 0,
                "fail": 0,
                "open": 0,
                "pct": 0.0,
                "verdict": "IDLE",
                "by_class": empty_cls,
            }

        status = self.growth_status
        ok_m = status == STATUS_CONNECTED
        fail_m = status == STATUS_FAILED
        open_m = (status == STATUS_GROWING) | (status == STATUS_SEEKING)
        n_ok = int(np.sum(ok_m))
        n_fail = int(np.sum(fail_m))
        n_open = int(np.sum(open_m))

        if n_open > 0:
            pct = 100.0 * n_ok / n
            verdict = "RUNNING"
        else:
            decided = n_ok + n_fail
            pct = (100.0 * n_ok / decided) if decided else 0.0
            if decided == 0:
                verdict = "IDLE"
            elif pct >= 70.0:
                verdict = "CONNECTED"
            elif pct >= 30.0:
                verdict = "PARTIAL"
            else:
                verdict = "FAILED"

        owner_cls = self.growth_cls.astype(np.int32, copy=False)
        goal_ids = self.growth_goal
        goal_cls = np.full(n, -1, dtype=np.int32)
        ids = self.soma_ids
        contiguous = (
            len(ids) == len(self.soma_cls)
            and len(ids) > 0
            and int(ids[0]) == 0
            and int(ids[-1]) == len(ids) - 1
        )
        if contiguous:
            valid = (goal_ids >= 0) & (goal_ids < len(self.soma_cls))
            goal_cls[valid] = self.soma_cls[goal_ids[valid]]
        else:
            lut = {
                int(i): int(c)
                for i, c in zip(ids.tolist(), self.soma_cls.tolist())
            }
            for i, gid in enumerate(goal_ids.tolist()):
                goal_cls[i] = lut.get(int(gid), -1)

        n_classes = max(
            int(len(self.class_names)),
            int(owner_cls.max()) + 1 if len(owner_cls) else 0,
            int(goal_cls.max()) + 1 if np.any(goal_cls >= 0) else 0,
        )
        by_class: list[dict] = []
        for c in range(n_classes):
            out_m = owner_cls == c
            in_m = goal_cls == c
            out_armed = int(np.sum(out_m))
            in_armed = int(np.sum(in_m))
            if out_armed == 0 and in_armed == 0:
                continue
            out_ok = int(np.sum(out_m & ok_m))
            out_fail = int(np.sum(out_m & fail_m))
            out_open = int(np.sum(out_m & open_m))
            in_ok = int(np.sum(in_m & ok_m))
            in_fail = int(np.sum(in_m & fail_m))
            in_open = int(np.sum(in_m & open_m))
            if out_open > 0:
                out_pct = 100.0 * out_ok / out_armed if out_armed else 0.0
            else:
                out_dec = out_ok + out_fail
                out_pct = 100.0 * out_ok / out_dec if out_dec else 0.0
            if in_open > 0:
                in_pct = 100.0 * in_ok / in_armed if in_armed else 0.0
            else:
                in_dec = in_ok + in_fail
                in_pct = 100.0 * in_ok / in_dec if in_dec else 0.0
            name = (
                self.class_names[c]
                if c < len(self.class_names)
                else f"class_{c}"
            )
            by_class.append(
                {
                    "idx": c,
                    "name": name,
                    "out_armed": out_armed,
                    "out_ok": out_ok,
                    "out_fail": out_fail,
                    "out_open": out_open,
                    "out_pct": out_pct,
                    "in_armed": in_armed,
                    "in_ok": in_ok,
                    "in_fail": in_fail,
                    "in_open": in_open,
                    "in_pct": in_pct,
                }
            )
        by_class.sort(key=lambda r: r["out_armed"], reverse=True)
        return {
            "armed": n,
            "ok": n_ok,
            "fail": n_fail,
            "open": n_open,
            "pct": pct,
            "verdict": verdict,
            "by_class": by_class,
        }

    def tick_neurites(self, dt: float) -> int:
        """Advance growth/seek timers in fly-seconds. Timed-out tips → failed."""
        if dt <= 0 or len(self.growth_status) == 0:
            return 0
        active = (self.growth_status == STATUS_GROWING) | (
            self.growth_status == STATUS_SEEKING
        )
        if not np.any(active):
            return 0
        self.growth_seek_t[active] = self.growth_seek_t[active] + float(dt)
        timed = active & (self.growth_seek_t >= np.maximum(1e-6, self.growth_seek_limit))
        n_fail = int(np.sum(timed))
        if n_fail:
            self.growth_status[timed] = STATUS_FAILED
            self.growth_partner[timed] = -1
        return n_fail

    def _soma_span(self) -> float:
        if len(self.soma_xyz) < 2:
            return 1.0
        span = float(np.ptp(self.soma_xyz, axis=0).max())
        return max(span, 1e-3)

    def _ensure_process_tree(self) -> bool:
        """KDTree over atlas process cloud (tissue prior for tip guidance)."""
        cloud = self.atlas_process_xyz
        if cloud is None or len(cloud) < 8:
            self._process_tree = None
            self._process_tree_xyz = np.zeros((0, 3), dtype=np.float32)
            return False
        pts = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
        if (
            self._process_tree is not None
            and len(self._process_tree_xyz) > 0
            and len(self._process_tree_xyz) <= len(pts)
            and np.allclose(self._process_tree_xyz[0], pts[0], atol=1e-5)
        ):
            return True
        if len(pts) > GROW_TISSUE_TREE_MAX:
            step = max(1, len(pts) // GROW_TISSUE_TREE_MAX)
            pts = pts[::step][:GROW_TISSUE_TREE_MAX]
        from scipy.spatial import cKDTree

        self._process_tree_xyz = pts
        self._process_tree = cKDTree(pts)
        # Cloud changed → drop class trees / modes so they rebuild.
        self._class_process_trees = {}
        self._class_process_xyz = {}
        self._tissue_mode_centers = np.zeros((0, 3), dtype=np.float32)
        self._tissue_mode_labels = np.zeros((0,), dtype=np.int32)
        self._neck_xyz = np.zeros((0, 3), dtype=np.float32)
        return True

    @staticmethod
    def _kmeans_modes(pts: np.ndarray, k: int = 2, n_iter: int = 18) -> tuple[np.ndarray, np.ndarray]:
        """Simple k-means on process samples. Returns (labels, centers)."""
        n = len(pts)
        if n == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float32)
        k = max(1, min(int(k), n))
        mean = pts.mean(axis=0)
        x = pts - mean
        # Init along principal axis extremes.
        try:
            _, _, vh = np.linalg.svd(x, full_matrices=False)
            axis = vh[0]
        except np.linalg.LinAlgError:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        proj = x @ axis
        qs = np.linspace(0.15, 0.85, k)
        centers = np.stack(
            [pts[int(np.argmin(np.abs(proj - np.quantile(proj, q))))] for q in qs],
            axis=0,
        ).astype(np.float64)
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(n_iter):
            d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1).astype(np.int32)
            for c in range(k):
                m = labels == c
                if np.any(m):
                    centers[c] = pts[m].mean(axis=0)
                else:
                    centers[c] = pts[int(np.random.randint(0, n))]
        return labels, centers.astype(np.float32)

    def _ensure_tissue_modes(self) -> bool:
        """Cluster process cloud into density modes for inter-cluster ridges."""
        if len(self._tissue_mode_centers) >= 2:
            return True
        if not self._ensure_process_tree():
            self._tissue_mode_centers = np.zeros((0, 3), dtype=np.float32)
            self._tissue_mode_labels = np.zeros((0,), dtype=np.int32)
            return False
        pts = self._process_tree_xyz
        # Subsample for clustering speed.
        if len(pts) > 25000:
            step = max(1, len(pts) // 25000)
            sample = pts[::step]
        else:
            sample = pts
        labels_s, centers = self._kmeans_modes(sample, k=int(GROW_TISSUE_MODES))
        if len(centers) < 2:
            return False
        # Label full tree points by nearest center.
        d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1).astype(np.int32)
        self._tissue_mode_centers = centers
        self._tissue_mode_labels = labels
        self._neck_xyz = np.zeros((0, 3), dtype=np.float32)
        _ = labels_s
        return True

    def _ensure_neck_waypoint(self) -> bool:
        """Tissue centroid of the constriction between density modes (the neck)."""
        if len(self._neck_xyz) == 1:
            return True
        if not self._ensure_tissue_modes() or not self._ensure_process_tree():
            self._neck_xyz = np.zeros((0, 3), dtype=np.float32)
            return False
        centers = self._tissue_mode_centers
        if len(centers) < 2:
            return False
        pts = self._process_tree_xyz.astype(np.float64)
        c0 = centers[0].astype(np.float64)
        c1 = centers[1].astype(np.float64)
        axis = c1 - c0
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-6:
            return False
        u = axis / axis_len
        proj = (pts - c0) @ u
        # Slab around the mid-ridge between modes — cervical connective.
        mid = 0.5 * axis_len
        half = max(0.04 * self._soma_span(), 0.08 * axis_len)
        band = np.abs(proj - mid) <= half
        if int(band.sum()) < 12:
            # Widen once if the constriction is extremely thin.
            band = np.abs(proj - mid) <= 2.0 * half
        if int(band.sum()) < 8:
            neck = (0.5 * (c0 + c1)).astype(np.float32)
        else:
            neck = pts[band].mean(axis=0).astype(np.float32)
        self._neck_xyz = neck.reshape(1, 3)
        return True

    def _point_modes(self, xyz: np.ndarray) -> np.ndarray:
        """Assign each point to nearest tissue density mode (−1 if unknown)."""
        n = len(xyz)
        out = np.full(n, -1, dtype=np.int32)
        if n == 0 or not self._ensure_tissue_modes():
            return out
        centers = self._tissue_mode_centers
        d2 = ((xyz[:, None, :].astype(np.float64) - centers[None, :, :]) ** 2).sum(
            axis=2
        )
        return np.argmin(d2, axis=1).astype(np.int32)

    def _ridge_targets(
        self,
        tip_xyz: np.ndarray,
        tip_mode: np.ndarray,
        goal_mode: np.ndarray,
        goal_xyz: np.ndarray,
    ) -> np.ndarray:
        """Steering points for tips: mode centroid when crossing, else contact goal.

        Inter-mode path follows the ridge between density centers (not a free-space
        chord and not local kNN that pulls back into the source blob).
        """
        out = goal_xyz.astype(np.float32, copy=True)
        if len(tip_xyz) == 0 or not self._ensure_tissue_modes():
            return out
        centers = self._tissue_mode_centers
        cross = (tip_mode >= 0) & (goal_mode >= 0) & (tip_mode != goal_mode)
        if not np.any(cross):
            return out
        # Waypoint = goal-mode centroid, until tip enters that mode.
        gc = centers[goal_mode[cross]]
        # Mix centroid with final goal so path ends at the contact domain.
        # Weight toward centroid while still far from goal mode.
        tip_c = tip_xyz[cross]
        d_cent = np.linalg.norm(gc - tip_c, axis=1, keepdims=True)
        d_goal = np.linalg.norm(goal_xyz[cross] - tip_c, axis=1, keepdims=True)
        # Prefer centroid while it is a meaningful fraction of remaining trip.
        w = np.clip(d_cent / np.maximum(d_goal, 1e-6), 0.35, 0.9)
        out[cross] = (w * gc + (1.0 - w) * goal_xyz[cross]).astype(np.float32)
        return out

    def _ensure_class_process_trees(self) -> bool:
        """Per-class KDTrees over atlas process cloud (contact domains)."""
        cloud = self.atlas_process_xyz
        cls = self.atlas_process_cls
        if cloud is None or len(cloud) < 8 or len(cls) != len(cloud):
            self._class_process_trees = {}
            self._class_process_xyz = {}
            return False
        if self._class_process_trees and self._class_process_xyz:
            return True
        from scipy.spatial import cKDTree

        trees: dict = {}
        xyzs: dict = {}
        pts = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
        carr = np.asarray(cls, dtype=np.int32).reshape(-1)
        for c in np.unique(carr).tolist():
            mask = carr == int(c)
            sub = pts[mask]
            if len(sub) < 4:
                continue
            if len(sub) > GROW_TISSUE_TREE_MAX:
                step = max(1, len(sub) // GROW_TISSUE_TREE_MAX)
                sub = sub[::step][:GROW_TISSUE_TREE_MAX]
            trees[int(c)] = cKDTree(sub)
            xyzs[int(c)] = sub
        self._class_process_trees = trees
        self._class_process_xyz = xyzs
        return bool(trees)

    def _contact_domains(
        self, post_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Contact basins: center + radius in post process neighborhood.

        Center = mean of nearby process samples (neuropil domain), not the soma.
        Connected when tip enters the basin (radius), not a single sample point.
        """
        n = int(len(post_ids))
        centers = np.zeros((n, 3), dtype=np.float32)
        radii = np.zeros((n,), dtype=np.float32)
        if n == 0:
            return centers, radii
        soma = np.stack(
            [self.neurons[int(i)].soma for i in post_ids.tolist()], axis=0
        ).astype(np.float32)
        centers[:] = soma
        span = self._soma_span()
        base_r = float(GROW_BASIN_FRAC) * span
        radii[:] = base_r
        post_cls = self._neuron_cls_array(post_ids.astype(np.int32, copy=False))
        min_d = float(GROW_CONTACT_MIN_FRAC) * span
        k_want = int(GROW_CONTACT_K)
        used = np.zeros(n, dtype=bool)

        def _fill_from_tree(sel: np.ndarray, tree, pts: np.ndarray) -> None:
            if len(sel) == 0 or tree is None or pts is None or len(pts) == 0:
                return
            k = min(k_want, len(pts))
            d, ix = tree.query(soma[sel], k=k)
            if k == 1:
                d = np.asarray(d, dtype=np.float64)[:, None]
                ix = np.asarray(ix, dtype=np.int64)[:, None]
            else:
                d = np.asarray(d, dtype=np.float64)
                ix = np.asarray(ix, dtype=np.int64)
            for row, si in enumerate(sel.tolist()):
                far = np.flatnonzero(d[row] >= min_d)
                pick = far if len(far) else np.arange(k)
                # Stable subset if many tips share a post.
                if len(pick) > 3:
                    off = int(si) % len(pick)
                    pick = np.roll(pick, -off)[: max(3, len(pick) // 2)]
                samples = pts[ix[row, pick]]
                c = samples.mean(axis=0)
                centers[si] = c.astype(np.float32)
                spread = float(np.median(np.linalg.norm(samples - c, axis=1)))
                radii[si] = max(base_r, float(GROW_BASIN_SPREAD) * spread)
                used[si] = True

        if self._ensure_class_process_trees():
            for c in np.unique(post_cls[post_cls >= 0]).tolist():
                sel = np.flatnonzero(post_cls == int(c))
                _fill_from_tree(
                    sel,
                    self._class_process_trees.get(int(c)),
                    self._class_process_xyz.get(int(c)),
                )
        miss = np.flatnonzero(~used)
        if len(miss) and self._ensure_process_tree():
            _fill_from_tree(miss, self._process_tree, self._process_tree_xyz)
        return centers, radii.astype(np.float32)

    def _contact_goal_xyz(self, post_ids: np.ndarray) -> np.ndarray:
        """Backward-compatible: basin centers only."""
        centers, _r = self._contact_domains(post_ids)
        return centers

    def _substrate_correction(
        self, tip_xyz: np.ndarray, instruct: np.ndarray
    ) -> np.ndarray:
        """Keep tips on the tissue manifold without competing with chemotaxis.

        If far from process cloud (in air), blend a pull toward nearest tissue.
        On-cloud: return instructive direction unchanged. In the neck slab the
        lock is stronger so cross-brain↔VNC axons do not chord through the void.
        """
        n = len(tip_xyz)
        if n == 0:
            return instruct
        out = instruct.astype(np.float32, copy=True)
        if not self._ensure_process_tree():
            return out
        span = self._soma_span()
        sub_r = float(GROW_SUBSTRATE_FRAC) * span
        tree = self._process_tree
        pts = self._process_tree_xyz
        d, ix = tree.query(tip_xyz, k=1)
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        ix = np.asarray(ix, dtype=np.int64).reshape(-1)
        off = d > sub_r
        if not np.any(off):
            return out
        toward = pts[ix[off]] - tip_xyz[off]
        tlen = np.linalg.norm(toward, axis=1, keepdims=True)
        toward = toward / np.maximum(tlen, 1e-6)
        w_max = float(GROW_SUBSTRATE_W)
        # Stronger tissue lock near the cervical connective.
        if self._ensure_neck_waypoint():
            neck = self._neck_xyz[0].astype(np.float64)
            near_neck = (
                np.linalg.norm(tip_xyz[off].astype(np.float64) - neck, axis=1)
                < float(GROW_NECK_CAPTURE_FRAC) * 2.5 * span
            )
            w_cap = np.full(int(np.sum(off)), w_max, dtype=np.float64)
            w_cap[near_neck] = float(GROW_SUBSTRATE_NECK_W)
        else:
            w_cap = w_max
        w = np.clip((d[off] - sub_r) / max(sub_r, 1e-6), 0.0, 1.0) * w_cap
        mixed = instruct[off] * (1.0 - w)[:, None] + toward * w[:, None]
        mlen = np.linalg.norm(mixed, axis=1, keepdims=True)
        out[off] = (mixed / np.maximum(mlen, 1e-6)).astype(np.float32)
        return out

    def _chemotaxis_dir(
        self,
        tip_xyz: np.ndarray,
        tip_mode: np.ndarray,
        goal_mode: np.ndarray,
        goal_xyz: np.ndarray,
    ) -> np.ndarray:
        """Unit instructive direction: neck waypoint when crossing, else cell.

        Cross-mode (brain↔VNC) tips are steered through the tissue neck first so
        Yield does not quietly chord across the empty cervical gap.
        """
        n = len(tip_xyz)
        cell_delta = goal_xyz - tip_xyz
        cell_dist = np.linalg.norm(cell_delta, axis=1, keepdims=True)
        cell_dir = cell_delta / np.maximum(cell_dist, 1e-6)
        out = cell_dir.astype(np.float32)
        if n == 0 or not self._ensure_tissue_modes():
            return out
        centers = self._tissue_mode_centers
        cross = (tip_mode >= 0) & (goal_mode >= 0) & (tip_mode != goal_mode)
        if not np.any(cross):
            return out
        span = self._soma_span()
        gc = centers[goal_mode[cross]]
        tip_c = tip_xyz[cross].astype(np.float64)
        goal_c = goal_xyz[cross].astype(np.float64)
        gc64 = gc.astype(np.float64)

        # Default: group (goal-mode center) + residual cell.
        group_delta = gc64 - tip_c
        group_dist = np.linalg.norm(group_delta, axis=1, keepdims=True)
        group_dir = group_delta / np.maximum(group_dist, 1e-6)
        w_cell = float(GROW_GROUP_CELL_BLEND)
        near = (group_dist[:, 0] < 0.12 * span).astype(np.float64)
        w = np.clip(w_cell + (1.0 - w_cell) * near, 0.0, 1.0)
        mixed = group_dir * (1.0 - w)[:, None] + cell_dir[cross] * w[:, None]

        # Route via cervical connective until the tip has crossed the neck.
        if self._ensure_neck_waypoint():
            neck = self._neck_xyz[0].astype(np.float64)
            to_neck = neck - tip_c
            d_neck = np.linalg.norm(to_neck, axis=1)
            neck_dir = to_neck / np.maximum(d_neck[:, None], 1e-6)
            # Goal-mode side of the neck plane = already through the connective.
            crossed = ((tip_c - neck) * (gc64 - neck)).sum(axis=1) > 0.0
            capture = float(GROW_NECK_CAPTURE_FRAC) * span
            need_neck = (~crossed) & (d_neck > 0.35 * capture)
            if np.any(need_neck):
                # Dominate toward neck; keep a little cell aim so paths fan less.
                w_n = float(GROW_NECK_DOMINATE)
                mixed[need_neck] = (
                    w_n * neck_dir[need_neck]
                    + (1.0 - w_n) * cell_dir[cross][need_neck]
                )
            approaching = (~crossed) & (d_neck <= 0.35 * capture) & (
                d_neck > 1e-6
            )
            if np.any(approaching):
                mixed[approaching] = (
                    0.55 * neck_dir[approaching]
                    + 0.25 * group_dir[approaching]
                    + 0.20 * cell_dir[cross][approaching]
                )

        mlen = np.linalg.norm(mixed, axis=1, keepdims=True)
        out = out.copy()
        out[cross] = (mixed / np.maximum(mlen, 1e-6)).astype(np.float32)
        return out

    def _device_funnel_dir(
        self, tip_xyz: np.ndarray, cell_dir: np.ndarray
    ) -> np.ndarray:
        """Blend tip direction toward device funnel corridor (shell attractors).

        Inside either well, pull toward the midpoint between funnels so growth
        prefers the overlapping neck volume (river between two poles).
        """
        out = cell_dir.astype(np.float32, copy=True)
        w = float(getattr(self, "device_funnel_w", 0.0) or 0.0)
        centers = getattr(self, "device_funnel_xyz", None)
        radii = getattr(self, "device_funnel_r", None)
        if w < 1e-6 or centers is None or radii is None:
            return out
        if len(centers) < 2 or len(radii) < 2 or len(tip_xyz) == 0:
            return out
        c = np.asarray(centers, dtype=np.float64)
        r = np.asarray(radii, dtype=np.float64)
        tip = tip_xyz.astype(np.float64)
        mid = 0.5 * (c[0] + c[1])
        # Inside either funnel (slightly expanded) → corridor pull.
        d0 = np.linalg.norm(tip - c[0][None, :], axis=1)
        d1 = np.linalg.norm(tip - c[1][None, :], axis=1)
        inside = (d0 <= 1.15 * float(r[0])) | (d1 <= 1.15 * float(r[1]))
        if not np.any(inside):
            return out
        pull = mid[None, :] - tip
        plen = np.linalg.norm(pull, axis=1, keepdims=True)
        pdir = pull / np.maximum(plen, 1e-6)
        blend = out.astype(np.float64)
        blend[inside] = (1.0 - w) * blend[inside] + w * pdir[inside]
        nrm = np.linalg.norm(blend, axis=1, keepdims=True)
        out = (blend / np.maximum(nrm, 1e-6)).astype(np.float32)
        return out

    def _clear_bundle_cohorts(self) -> None:
        self.growth_bundle_id = np.full((0,), -1, dtype=np.int32)
        self.growth_pioneer_idx = np.full((0,), -1, dtype=np.int32)

    def _assign_bundle_cohorts(self) -> int:
        """Tag tips into cohorts and pick one pioneer navigator per cohort.

        Pioneer = shortest soma→goal chord in the cohort (clearest path finder).
        Followers fasciculate along the pioneer's trail during growth.
        """
        n = len(self.growth_owner)
        self.growth_bundle_id = np.full(n, -1, dtype=np.int32)
        self.growth_pioneer_idx = np.full(n, -1, dtype=np.int32)
        if n < BUNDLE_COHORT_MIN:
            return 0
        goal_cls = self._neuron_cls_array(self.growth_goal.astype(np.int32, copy=False))
        pre_cls = self.growth_cls.astype(np.int32, copy=False)
        pairs: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            pairs.setdefault((int(pre_cls[i]), int(goal_cls[i])), []).append(i)
        assign = np.full(n, -1, dtype=np.int32)
        pioneer = np.full(n, -1, dtype=np.int32)
        next_id = 0

        def _emit(idxs: list[int]) -> None:
            nonlocal next_id
            if len(idxs) < BUNDLE_COHORT_MIN:
                return
            ii = np.asarray(idxs, dtype=np.int32)
            assign[ii] = next_id
            # Shortest soma→goal = pioneer navigator for this cable.
            d = np.linalg.norm(
                self.growth_goal_xyz[ii] - self.growth_soma[ii], axis=1
            )
            piv = int(ii[int(np.argmin(d))])
            pioneer[ii] = piv
            next_id += 1

        for _key, idxs in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
            ii = np.asarray(idxs, dtype=np.int32)
            mid_x = 0.5 * (
                self.growth_soma[ii, 0] + self.growth_goal_xyz[ii, 0]
            )
            left = ii[mid_x < 0.0].tolist()
            right = ii[mid_x >= 0.0].tolist()
            if len(left) >= BUNDLE_COHORT_MIN and len(right) >= BUNDLE_COHORT_MIN:
                _emit(left)
                _emit(right)
            else:
                _emit(idxs)
        self.growth_bundle_id = assign
        self.growth_pioneer_idx = pioneer
        return int(next_id)

    def _pioneer_trail_target(
        self, tip: np.ndarray, piv: int, goal: np.ndarray
    ) -> np.ndarray | None:
        """Nearest-ahead sample on pioneer polyline (ECM-like cable path)."""
        if piv < 0 or piv >= len(self.growth_poly_n):
            return None
        k = int(self.growth_poly_n[piv])
        if k < 2:
            return None
        trail = self.growth_poly[piv, :k].astype(np.float64)
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        # Subsample for speed on long trails.
        n_keep = min(int(PIONEER_TRAIL_SAMPLES), k)
        if n_keep < k:
            idx = np.linspace(0, k - 1, n_keep).astype(np.int32)
            samp = trail[idx]
            samp_i = idx
        else:
            samp = trail
            samp_i = np.arange(k, dtype=np.int32)
        d = np.linalg.norm(samp - tip[None, :], axis=1)
        nearest = int(np.argmin(d))
        # Prefer a point slightly ahead along the poly toward the goal.
        ahead = max(1, int(round(float(PIONEER_AHEAD_FRAC) * (len(samp) - 1))))
        j = min(nearest + ahead, len(samp) - 1)
        # If ahead is not more goal-progressed, keep nearest.
        gprog = float(np.dot(samp[j] - tip, goal - tip))
        if gprog < 0.0 and nearest < len(samp) - 1:
            j = nearest
        return samp[j].astype(np.float64)

    def _peer_bundle_dir(
        self,
        tip_xyz: np.ndarray,
        tip_idx: np.ndarray,
        goal_xyz: np.ndarray,
        cell_dir: np.ndarray,
    ) -> np.ndarray:
        """Pioneer chemotaxis + follower fasciculation along pioneer trail.

        Pioneer tips ignore peers (pure path finding). Followers lock onto the
        pioneer polyline — the cable channel — with a light residual tip bind.
        Connected / failed pioneers still guide via their frozen trail.
        """
        out = cell_dir.astype(np.float32, copy=True)
        n = len(tip_xyz)
        if n == 0:
            return out
        if len(self.growth_bundle_id) != len(self.growth_owner):
            return out
        if len(getattr(self, "growth_pioneer_idx", ())) != len(self.growth_owner):
            return out
        cohorts = self.growth_bundle_id[tip_idx]
        pioneers = self.growth_pioneer_idx[tip_idx]
        span = self._soma_span()
        radius = float(BUNDLE_RADIUS_FRAC) * span
        from scipy.spatial import cKDTree

        tree = cKDTree(tip_xyz.astype(np.float64))
        k = min(int(BUNDLE_K) + 1, max(n, 1))
        dists, nbrs = tree.query(tip_xyz, k=k, distance_upper_bound=radius)
        if k == 1:
            dists = np.asarray(dists, dtype=np.float64)[:, None]
            nbrs = np.asarray(nbrs, dtype=np.int64)[:, None]
        else:
            dists = np.asarray(dists, dtype=np.float64)
            nbrs = np.asarray(nbrs, dtype=np.int64)

        for i in range(n):
            cid = int(cohorts[i])
            if cid < 0:
                continue
            gi = int(tip_idx[i])
            piv = int(pioneers[i])
            if piv < 0:
                piv = int(self.growth_pioneer_idx[gi]) if gi < len(self.growth_pioneer_idx) else -1
            # Pioneer: chemotaxis only (path finder).
            if piv == gi:
                continue
            cell = cell_dir[i].astype(np.float64)
            c_n = float(np.linalg.norm(cell))
            if c_n > 1e-9:
                cell = cell / c_n
            else:
                cell = np.zeros(3, dtype=np.float64)

            trail_dir = None
            target = self._pioneer_trail_target(tip_xyz[i], piv, goal_xyz[i])
            if target is not None:
                pull = target - tip_xyz[i].astype(np.float64)
                # Lateral component vs chemo — hug the cable, don't fight the goal.
                if c_n > 1e-9:
                    pull = pull - cell * float(np.dot(pull, cell))
                p_n = float(np.linalg.norm(pull))
                if p_n > 1e-9:
                    trail_dir = pull / p_n

            # Light residual bind to nearby follower tips (same cohort).
            peer_dir = None
            peer_pos = []
            peer_w = []
            for j in range(k):
                nj = int(nbrs[i, j])
                if nj < 0 or nj >= n or nj == i:
                    continue
                d = float(dists[i, j])
                if not np.isfinite(d) or d > radius or d < 1e-8:
                    continue
                if int(cohorts[nj]) != cid:
                    continue
                # Skip binding hard to the pioneer tip itself (trail handles that).
                if int(tip_idx[nj]) == piv:
                    continue
                ahead = float(
                    np.dot(
                        tip_xyz[nj] - tip_xyz[i],
                        goal_xyz[i] - tip_xyz[i],
                    )
                )
                w = (1.0 / (d + 1e-4)) * (1.0 + 0.35 * max(0.0, np.sign(ahead)))
                peer_pos.append(tip_xyz[nj])
                peer_w.append(w)
            if len(peer_pos) >= BUNDLE_MIN_PEERS:
                pos = np.stack(peer_pos, axis=0)
                ww = np.asarray(peer_w, dtype=np.float64)
                ww = ww / ww.sum()
                target_p = (pos * ww[:, None]).sum(axis=0)
                pull_p = target_p - tip_xyz[i]
                if c_n > 1e-9:
                    pull_p = pull_p - cell * float(np.dot(pull_p, cell))
                p_n = float(np.linalg.norm(pull_p))
                if p_n > 1e-9:
                    peer_dir = pull_p / p_n

            if trail_dir is None and peer_dir is None:
                continue
            d_goal = float(np.linalg.norm(goal_xyz[i] - tip_xyz[i]))
            # Near the contact basin: drop fasciculation — lateral trail pull
            # orbits followers just outside the connect radius (~99% hang).
            br_i = (
                float(self.growth_basin_r[gi])
                if gi < len(self.growth_basin_r)
                else float(GROW_BASIN_FRAC) * span
            )
            if d_goal <= float(GROW_FINAL_APPROACH_MUL) * max(br_i, 1e-6):
                continue
            near = float(np.clip(1.0 - d_goal / max(0.10 * span, 1e-6), 0.0, 1.0))
            fade = 1.0 - 0.65 * near
            # Long-haul: keep chemotaxis dominant until close to the basin.
            # Full trail blend at distance was parking AN/DN mid-neck (Yield ~33/42%).
            approach = float(
                np.clip(1.0 - d_goal / max(0.35 * span, 1e-6), 0.0, 1.0)
            )
            trail_scale = float(PIONEER_TRAIL_FAR) + (
                float(PIONEER_TRAIL_BLEND) - float(PIONEER_TRAIL_FAR)
            ) * approach
            if trail_dir is not None:
                w_trail = trail_scale * fade
                fasc = trail_dir
                if peer_dir is not None:
                    w_peer = float(PIONEER_PEER_BLEND) * fade * approach
                    fasc = (1.0 - w_peer) * trail_dir + w_peer * peer_dir
                    fn = float(np.linalg.norm(fasc))
                    if fn > 1e-9:
                        fasc = fasc / fn
                mixed = w_trail * fasc + (1.0 - w_trail) * cell
            else:
                dens = min(1.0, len(peer_pos) / float(BUNDLE_K))
                w_peer = (
                    float(BUNDLE_BLEND)
                    * (0.50 + 0.50 * dens)
                    * fade
                    * (0.25 + 0.75 * approach)
                )
                mixed = w_peer * peer_dir + (1.0 - w_peer) * cell
            m_n = float(np.linalg.norm(mixed))
            if m_n > 1e-9:
                out[i] = (mixed / m_n).astype(np.float32)
        return out

    def _tissue_pull(self, tip_xyz: np.ndarray, tip_idx: np.ndarray | None = None) -> np.ndarray:
        """Unit vectors toward local process density (0 if no atlas cloud).

        Caches per-tip pull for a few ticks — KDTree on ~20k tips/frame is costly.
        """
        n = len(tip_xyz)
        out = np.zeros((n, 3), dtype=np.float32)
        if n == 0 or not self._ensure_process_tree():
            return out
        # Reuse cache when querying the same active set within the refresh window.
        refresh = 3
        cache = getattr(self, "_tissue_pull_cache", None)
        cache_tick = getattr(self, "_tissue_pull_cache_tick", -999)
        if (
            tip_idx is not None
            and cache is not None
            and len(cache) == len(self.growth_owner)
            and self.sim_time - cache_tick < refresh
        ):
            return cache[tip_idx].astype(np.float32, copy=False)

        tree = self._process_tree
        pts = self._process_tree_xyz
        k = min(GROW_TISSUE_K, len(pts))
        _d, ix = tree.query(tip_xyz, k=k)
        if k == 1:
            ix = ix[:, None]
            _d = _d[:, None]
        nbr = pts[ix]
        delta = nbr - tip_xyz[:, None, :]
        dist = np.linalg.norm(delta, axis=2)
        w = 1.0 / np.maximum(dist, 1e-4)
        w = np.where(dist < 1e-5, 0.0, w)
        wsum = w.sum(axis=1, keepdims=True)
        none = wsum[:, 0] < 1e-8
        w = w / np.maximum(wsum, 1e-8)
        pull = (delta * w[:, :, None]).sum(axis=1)
        plen = np.linalg.norm(pull, axis=1, keepdims=True)
        ok = (~none) & (plen[:, 0] > 1e-8)
        out[ok] = (pull[ok] / plen[ok]).astype(np.float32)

        if tip_idx is not None and len(self.growth_owner):
            if cache is None or len(cache) != len(self.growth_owner):
                cache = np.zeros((len(self.growth_owner), 3), dtype=np.float32)
            cache[tip_idx] = out
            self._tissue_pull_cache = cache
            self._tissue_pull_cache_tick = int(self.sim_time)
        return out

    def _set_member_ids(self, name: str) -> list[int]:
        s = self.sets.get(name)
        if s is None:
            return []
        return [nid for nid in sorted(s.members) if nid in self.neurons]

    def _member_id_array(self, names: list[str]) -> np.ndarray:
        ids: list[int] = []
        seen: set[int] = set()
        for name in names:
            for nid in self._set_member_ids(name):
                if nid not in seen:
                    seen.add(nid)
                    ids.append(nid)
        return np.asarray(ids, dtype=np.int32)

    def _target_xyz_for_ids(self, tgt_ids: np.ndarray) -> np.ndarray:
        if len(tgt_ids) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(
            [self.neurons[int(i)].soma for i in tgt_ids.tolist()], axis=0
        ).astype(np.float32)

    @staticmethod
    def _map_bodies_to_index(
        bodies: np.ndarray, body_arr: np.ndarray
    ) -> np.ndarray:
        """Map body ids → indices into body_arr; missing → -1."""
        if len(bodies) == 0 or len(body_arr) == 0:
            return np.full(len(bodies), -1, dtype=np.int32)
        sorter = np.argsort(body_arr)
        sorted_b = body_arr[sorter]
        pos = np.searchsorted(sorted_b, bodies)
        pos = np.clip(pos, 0, len(sorted_b) - 1)
        ok = sorted_b[pos] == bodies
        out = np.full(len(bodies), -1, dtype=np.int32)
        out[ok] = sorter[pos[ok]].astype(np.int32)
        return out

    def set_recipe_from_body_edges(
        self,
        pre_body: np.ndarray,
        post_body: np.ndarray,
        weight: np.ndarray,
    ) -> int:
        """Install atlas edges (body ids) as neuron-index recipe."""
        if len(self.soma_body_id) == 0:
            # Align body ids with current neuron indices if missing.
            self.soma_body_id = self.soma_ids.astype(np.int64)
        pre_i = self._map_bodies_to_index(
            np.asarray(pre_body, dtype=np.int64), self.soma_body_id
        )
        post_i = self._map_bodies_to_index(
            np.asarray(post_body, dtype=np.int64), self.soma_body_id
        )
        w = np.asarray(weight, dtype=np.float32)
        ok = (pre_i >= 0) & (post_i >= 0) & (pre_i != post_i)
        self.recipe_pre = pre_i[ok].astype(np.int32)
        self.recipe_post = post_i[ok].astype(np.int32)
        self.recipe_weight = w[ok]
        self.recipe_armed = 0
        self._hop_recipe_key = None  # invalidate adult hop cache
        self.rebuild_genome_from_atlas()
        return int(len(self.recipe_pre))

    def _neuron_cls_array(self, neuron_ids: np.ndarray) -> np.ndarray:
        """Map neuron indices → class ids (−1 if unknown)."""
        n_soma = len(self.soma_cls)
        if n_soma == 0 or len(neuron_ids) == 0:
            return np.full(len(neuron_ids), -1, dtype=np.int32)
        ids = self.soma_ids
        contiguous = (
            len(ids) == n_soma
            and int(ids[0]) == 0
            and int(ids[-1]) == n_soma - 1
        )
        if contiguous:
            valid = (neuron_ids >= 0) & (neuron_ids < n_soma)
            return np.where(
                valid,
                self.soma_cls[np.clip(neuron_ids, 0, n_soma - 1)],
                -1,
            ).astype(np.int32)
        lut = {int(i): int(c) for i, c in zip(ids.tolist(), self.soma_cls.tolist())}
        return np.array(
            [lut.get(int(p), -1) for p in neuron_ids.tolist()], dtype=np.int32
        )

    def rebuild_genome_from_atlas(
        self, *, total_tips: int = GROW_MAX_RECIPE_EDGES
    ) -> np.ndarray:
        """Fill genome_quota from atlas edge mass (set→set), then editable.

        Score ∝ edge count per (pre_class, post_class); allocate `total_tips`
        by largest-remainder. Zero cells stay zero (no invented pathways).
        """
        n = len(self.class_names)
        if n == 0 or len(self.recipe_pre) == 0:
            self.genome_quota = np.zeros((n, n), dtype=np.int32)
            self._genome_edge_bins = {}
            return self.genome_quota
        pre_c = self._neuron_cls_array(self.recipe_pre)
        post_c = self._neuron_cls_array(self.recipe_post)
        ok = (pre_c >= 0) & (post_c >= 0)
        pre_c, post_c = pre_c[ok], post_c[ok]
        edge_i = np.flatnonzero(ok)
        counts = np.zeros((n, n), dtype=np.float64)
        # Bin edge indices per cell (for arming).
        bins: dict[tuple[int, int], list[int]] = {}
        w = self.recipe_weight[ok]
        for e, a, b in zip(edge_i.tolist(), pre_c.tolist(), post_c.tolist()):
            counts[a, b] += 1.0
            bins.setdefault((a, b), []).append(int(e))
        # Sort each bin by descending weight once.
        self._genome_edge_bins = {}
        for key, idxs in bins.items():
            arr = np.asarray(idxs, dtype=np.int32)
            if len(arr) > 1:
                order = np.argsort(-self.recipe_weight[arr])
                arr = arr[order]
            self._genome_edge_bins[key] = arr

        score = np.sqrt(counts)
        total = float(score.sum())
        quota = np.zeros((n, n), dtype=np.int32)
        if total <= 0 or total_tips <= 0:
            self.genome_quota = quota
            return quota
        raw = score / total * float(total_tips)
        base = np.floor(raw).astype(np.int32)
        # Don't allocate more than available edges in a cell.
        for i in range(n):
            for j in range(n):
                avail = int(counts[i, j])
                if avail <= 0:
                    base[i, j] = 0
                else:
                    base[i, j] = min(int(base[i, j]), avail)
        used = int(base.sum())
        remain = int(total_tips) - used
        if remain > 0:
            frac = raw - np.floor(raw)
            # Prefer cells that still have unused atlas edges.
            order = np.argsort(-frac, axis=None)
            for flat in order.tolist():
                if remain <= 0:
                    break
                i, j = divmod(int(flat), n)
                avail = int(counts[i, j]) - int(base[i, j])
                if avail <= 0:
                    continue
                base[i, j] += 1
                remain -= 1
        self.genome_quota = base
        return self.genome_quota

    def genome_rules(self) -> list[tuple[int, int, int]]:
        """Non-zero genome cells as (src_cls, tgt_cls, tips), heaviest first."""
        q = self.genome_quota
        if q.size == 0:
            return []
        rules = [
            (i, j, int(q[i, j]))
            for i in range(q.shape[0])
            for j in range(q.shape[1])
            if q[i, j] > 0
        ]
        rules.sort(key=lambda t: t[2], reverse=True)
        return rules

    def adjust_genome_rule(self, src: int, tgt: int, delta: int) -> int:
        """Change one cell's tip quota (clamped to available atlas edges)."""
        n = len(self.class_names)
        if not (0 <= src < n and 0 <= tgt < n):
            return 0
        if self.genome_quota.shape != (n, n):
            self.rebuild_genome_from_atlas()
        avail = len(getattr(self, "_genome_edge_bins", {}).get((src, tgt), ()))
        if avail <= 0 and delta > 0:
            # Allow inventing a quota only if atlas has that pathway.
            return int(self.genome_quota[src, tgt])
        new = int(self.genome_quota[src, tgt]) + int(delta)
        new = max(0, min(new, avail if avail > 0 else new))
        self.genome_quota[src, tgt] = new
        return new

    def scale_genome(self, factor: float) -> int:
        """Scale all quotas; clamp to atlas edge availability. Returns total tips."""
        n = len(self.class_names)
        if self.genome_quota.shape != (n, n):
            self.rebuild_genome_from_atlas()
        bins = getattr(self, "_genome_edge_bins", {})
        q = self.genome_quota.astype(np.float64) * float(factor)
        out = np.zeros_like(self.genome_quota)
        for i in range(n):
            for j in range(n):
                avail = len(bins.get((i, j), ()))
                if avail <= 0:
                    out[i, j] = 0
                else:
                    out[i, j] = int(max(0, min(avail, round(q[i, j]))))
        self.genome_quota = out.astype(np.int32)
        return int(self.genome_quota.sum())

    def select_recipe_edges(
        self,
        sources: list[str],
        targets: list[str],
        *,
        max_edges: int = GROW_MAX_RECIPE_EDGES,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arm tips from genome_quota; atlas edges only fill each cell.

        `sources`/`targets` mask which genome rows/cols are active.
        Falls back to rebuild-from-atlas if the matrix is empty.
        """
        empty = (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
        if len(self.recipe_pre) == 0:
            return empty
        n = len(self.class_names)
        if n == 0:
            return empty
        if self.genome_quota.shape != (n, n) or int(self.genome_quota.sum()) == 0:
            self.rebuild_genome_from_atlas(total_tips=max_edges)
        elif not self._genome_edge_bins:
            # Bins missing but quota edited — rebuild candidates without wiping rules.
            saved = np.asarray(self.genome_quota, dtype=np.int32).copy()
            self.rebuild_genome_from_atlas(total_tips=max(int(saved.sum()), max_edges))
            for i in range(n):
                for j in range(n):
                    avail = len(self._genome_edge_bins.get((i, j), ()))
                    self.genome_quota[i, j] = (
                        min(int(saved[i, j]), avail) if avail else 0
                    )

        name_to_i = {name: i for i, name in enumerate(self.class_names)}
        src_i = {name_to_i[s] for s in sources if s in name_to_i}
        tgt_i = {name_to_i[t] for t in targets if t in name_to_i}
        if not src_i or not tgt_i:
            return empty

        parts: list[np.ndarray] = []
        total = 0
        for i in sorted(src_i):
            for j in sorted(tgt_i):
                want = int(self.genome_quota[i, j])
                if want <= 0:
                    continue
                bin_idx = self._genome_edge_bins.get((i, j))
                if bin_idx is None or len(bin_idx) == 0:
                    continue
                take = min(want, len(bin_idx), max(0, max_edges - total))
                if take <= 0:
                    continue
                parts.append(bin_idx[:take])
                total += take
                if total >= max_edges:
                    break
            if total >= max_edges:
                break
        if not parts:
            return empty
        sel = np.concatenate(parts)
        return (
            self.recipe_pre[sel].astype(np.int32),
            self.recipe_post[sel].astype(np.int32),
            self.recipe_weight[sel].astype(np.float32),
        )

    @staticmethod
    def _stratified_indices(
        indices: np.ndarray, cls: np.ndarray, max_n: int
    ) -> np.ndarray:
        """Subsample indices with roughly equal share per class."""
        if len(indices) <= max_n:
            return indices
        classes = np.unique(cls[indices])
        if len(classes) == 0:
            return indices[:max_n]
        per = max(1, max_n // int(len(classes)))
        parts: list[np.ndarray] = []
        budget = max_n
        for c in classes:
            if budget <= 0:
                break
            sub = indices[cls[indices] == c]
            take = min(per, budget, len(sub))
            if len(sub) > take:
                step = max(1, len(sub) // take)
                sub = sub[::step][:take]
            parts.append(sub)
            budget -= len(sub)
        if not parts:
            return indices[:max_n]
        sel = np.concatenate(parts)
        if len(sel) > max_n:
            sel = sel[:max_n]
        return sel

    def _seek_limits_for_trips(
        self,
        owner_xyz: np.ndarray,
        goal_xyz: np.ndarray,
        *,
        step_frac: float = GROW_STEP_FRAC,
        base: float = GROW_SEEK_LIMIT,
    ) -> np.ndarray:
        span = self._soma_span()
        step = max(1e-6, float(step_frac) * span)
        dist = np.linalg.norm(goal_xyz - owner_xyz, axis=1)
        need = dist / step * 1.25 + 30.0
        return np.maximum(float(base), need).astype(np.float32)

    def _tip_spawn_offset(
        self, soma_xyz: np.ndarray, goal_xyz: np.ndarray
    ) -> np.ndarray:
        """Start tip a short step toward the goal so it never sits on the soma."""
        delta = goal_xyz - soma_xyz
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        step = 0.01 * self._soma_span()
        safe = np.maximum(dist, 1e-6)
        offset = delta / safe * np.minimum(step, 0.12 * safe)
        none = dist[:, 0] < 1e-8
        if np.any(none):
            offset = offset.copy()
            offset[none] = np.array([step, 0.0, 0.0], dtype=np.float32)
        return offset.astype(np.float32)

    def _install_growth_tips(
        self,
        owners: np.ndarray,
        goals: np.ndarray,
        *,
        seek_limit: float = GROW_SEEK_LIMIT,
    ) -> int:
        """Replace growth field with tips for given pre→post neuron pairs."""
        n = int(len(owners))
        self.recipe_armed = n
        self._ensure_process_tree()
        if n == 0:
            self.growth_owner = np.zeros((0,), dtype=np.int32)
            self.growth_soma = np.zeros((0, 3), dtype=np.float32)
            self.growth_tip = np.zeros((0, 3), dtype=np.float32)
            self.growth_status = np.zeros((0,), dtype=np.uint8)
            self.growth_partner = np.full((0,), -1, dtype=np.int32)
            self.growth_goal = np.full((0,), -1, dtype=np.int32)
            self.growth_goal_xyz = np.zeros((0, 3), dtype=np.float32)
            self.growth_basin_r = np.zeros((0,), dtype=np.float32)
            self.growth_seek_t = np.zeros((0,), dtype=np.float32)
            self.growth_seek_limit = np.full((0,), GROW_SEEK_LIMIT, dtype=np.float32)
            self.growth_cls = np.zeros((0,), dtype=np.int32)
            self.growth_poly = np.zeros((0, GROW_TRAIL_MAX, 3), dtype=np.float32)
            self.growth_poly_n = np.zeros((0,), dtype=np.int32)
            self._growth_materialized = np.zeros((0,), dtype=bool)
            self.last_growth_step = {"moved": 0, "connected": 0}
            self._clear_bundle_cohorts()
            return 0
        soma = np.stack(
            [self.neurons[int(i)].soma for i in owners.tolist()], axis=0
        ).astype(np.float32)
        # Goal = contact-domain basin near post (process neighborhood).
        gxyz, basin_r = self._contact_domains(goals)
        tip = soma + self._tip_spawn_offset(soma, gxyz)
        # Seek from path length: soma → neck → goal-mode → basin (cross-mode).
        span = self._soma_span()
        step = max(1e-6, float(GROW_STEP_FRAC) * span)
        path = np.linalg.norm(gxyz - soma, axis=1)
        if self._ensure_tissue_modes():
            centers = self._tissue_mode_centers
            sm = self._point_modes(soma)
            gm = self._point_modes(gxyz)
            cross = (sm >= 0) & (gm >= 0) & (sm != gm)
            if np.any(cross):
                gc = centers[gm[cross]]
                path = path.copy()
                if self._ensure_neck_waypoint():
                    neck = self._neck_xyz[0]
                    path[cross] = (
                        np.linalg.norm(neck - soma[cross], axis=1)
                        + np.linalg.norm(gc - neck, axis=1)
                        + np.linalg.norm(gxyz[cross] - gc, axis=1)
                    )
                else:
                    path[cross] = np.linalg.norm(
                        gc - soma[cross], axis=1
                    ) + np.linalg.norm(gxyz[cross] - gc, axis=1)
                path[cross] *= float(GROW_CROSS_SEEK_EXTRA)
        need = path / step * 1.45 + 40.0
        seek = np.maximum(float(seek_limit), need).astype(np.float32)

        self.growth_owner = owners.astype(np.int32, copy=False)
        self.growth_soma = soma
        self.growth_tip = tip
        self.growth_status = np.full(n, STATUS_GROWING, dtype=np.uint8)
        self.growth_partner = np.full(n, -1, dtype=np.int32)
        self.growth_goal = goals.astype(np.int32, copy=False)
        self.growth_goal_xyz = gxyz
        self.growth_basin_r = basin_r.astype(np.float32, copy=False)
        self.growth_seek_t = np.zeros(n, dtype=np.float32)
        self.growth_seek_limit = seek
        self.growth_cls = np.asarray(
            [self.neurons[int(i)].cls for i in owners.tolist()], dtype=np.int32
        )
        self.growth_poly = np.zeros((n, GROW_TRAIL_MAX, 3), dtype=np.float32)
        self.growth_poly[:, 0, :] = soma
        self.growth_poly[:, 1, :] = tip
        self.growth_poly_n = np.full(n, 2, dtype=np.int32)
        self._growth_materialized = np.zeros(n, dtype=bool)
        self.last_growth_step = {"moved": 0, "connected": 0}
        self._assign_bundle_cohorts()
        return n

    def append_growth_tips(
        self,
        owners: np.ndarray,
        goals: np.ndarray,
        *,
        seek_limit: float = GROW_SEEK_LIMIT,
    ) -> int:
        """Append tips for pre→post pairs without wiping existing growth field.

        Used by peripheral devices (ocelli): same contact/basin rules as atlas tips.
        Replaces any prior tips owned by the same ``owners``.
        """
        if self.phase == "adult":
            return 0
        owners = np.asarray(owners, dtype=np.int32).reshape(-1)
        goals = np.asarray(goals, dtype=np.int32).reshape(-1)
        if len(owners) == 0 or len(owners) != len(goals):
            return 0
        # Drop previous tips from these owners.
        if len(self.growth_owner):
            keep = ~np.isin(self.growth_owner, owners)
            if not np.all(keep):
                self.growth_owner = self.growth_owner[keep]
                self.growth_soma = self.growth_soma[keep]
                self.growth_tip = self.growth_tip[keep]
                self.growth_status = self.growth_status[keep]
                self.growth_partner = self.growth_partner[keep]
                self.growth_goal = self.growth_goal[keep]
                self.growth_goal_xyz = self.growth_goal_xyz[keep]
                self.growth_basin_r = self.growth_basin_r[keep]
                self.growth_seek_t = self.growth_seek_t[keep]
                self.growth_seek_limit = self.growth_seek_limit[keep]
                self.growth_cls = self.growth_cls[keep]
                self.growth_poly = self.growth_poly[keep]
                self.growth_poly_n = self.growth_poly_n[keep]
                if len(self._growth_materialized) == len(keep):
                    self._growth_materialized = self._growth_materialized[keep]
                else:
                    self._growth_materialized = np.zeros(
                        len(self.growth_owner), dtype=bool
                    )
                if len(getattr(self, "growth_bundle_id", ())) == len(keep):
                    self.growth_bundle_id = self.growth_bundle_id[keep]
                # Pioneer tip indices are absolute — rebuild after any drop.
                self._assign_bundle_cohorts()
        if len(self.growth_owner) == 0:
            return self._install_growth_tips(owners, goals, seek_limit=seek_limit)

        # Build a temporary batch, then concatenate onto the live field.
        saved = {
            "owner": self.growth_owner,
            "soma": self.growth_soma,
            "tip": self.growth_tip,
            "status": self.growth_status,
            "partner": self.growth_partner,
            "goal": self.growth_goal,
            "goal_xyz": self.growth_goal_xyz,
            "basin_r": self.growth_basin_r,
            "seek_t": self.growth_seek_t,
            "seek_limit": self.growth_seek_limit,
            "cls": self.growth_cls,
            "poly": self.growth_poly,
            "poly_n": self.growth_poly_n,
            "mat": self._growth_materialized,
            "armed": self.recipe_armed,
            "sources": list(self.growth_sources),
            "targets": list(self.growth_targets),
        }
        n_new = self._install_growth_tips(owners, goals, seek_limit=seek_limit)
        if n_new <= 0:
            # restore
            self.growth_owner = saved["owner"]
            self.growth_soma = saved["soma"]
            self.growth_tip = saved["tip"]
            self.growth_status = saved["status"]
            self.growth_partner = saved["partner"]
            self.growth_goal = saved["goal"]
            self.growth_goal_xyz = saved["goal_xyz"]
            self.growth_basin_r = saved["basin_r"]
            self.growth_seek_t = saved["seek_t"]
            self.growth_seek_limit = saved["seek_limit"]
            self.growth_cls = saved["cls"]
            self.growth_poly = saved["poly"]
            self.growth_poly_n = saved["poly_n"]
            self._growth_materialized = saved["mat"]
            self.recipe_armed = saved["armed"]
            self.growth_sources = saved["sources"]
            self.growth_targets = saved["targets"]
            return 0
        self.growth_owner = np.concatenate([saved["owner"], self.growth_owner])
        self.growth_soma = np.concatenate([saved["soma"], self.growth_soma], axis=0)
        self.growth_tip = np.concatenate([saved["tip"], self.growth_tip], axis=0)
        self.growth_status = np.concatenate([saved["status"], self.growth_status])
        self.growth_partner = np.concatenate([saved["partner"], self.growth_partner])
        self.growth_goal = np.concatenate([saved["goal"], self.growth_goal])
        self.growth_goal_xyz = np.concatenate(
            [saved["goal_xyz"], self.growth_goal_xyz], axis=0
        )
        self.growth_basin_r = np.concatenate([saved["basin_r"], self.growth_basin_r])
        self.growth_seek_t = np.concatenate([saved["seek_t"], self.growth_seek_t])
        self.growth_seek_limit = np.concatenate(
            [saved["seek_limit"], self.growth_seek_limit]
        )
        self.growth_cls = np.concatenate([saved["cls"], self.growth_cls])
        # Polylines may differ in trail length dim — both use GROW_TRAIL_MAX.
        self.growth_poly = np.concatenate([saved["poly"], self.growth_poly], axis=0)
        self.growth_poly_n = np.concatenate([saved["poly_n"], self.growth_poly_n])
        self._growth_materialized = np.concatenate(
            [saved["mat"], self._growth_materialized]
        )
        self.recipe_armed = int(saved["armed"]) + int(n_new)
        self.growth_sources = saved["sources"]
        self.growth_targets = saved["targets"]
        self._assign_bundle_cohorts()
        return int(n_new)

    def stimulate_neurons(
        self, levels: dict[int, float]
    ) -> int:
        """Set activity on specific neuron ids (max with current). Returns count.

        Updates ``soma_activity`` in place and writebacks only touched neurons
        (no full 140k rebuild).
        """
        if not levels or not self.neurons:
            return 0
        if len(self.soma_ids) == 0 or len(self.soma_activity) != len(self.soma_ids):
            self.rebuild_activity_cache()
        ids = self.soma_ids
        act = self.soma_activity
        if len(ids) == 0 or len(act) != len(ids):
            # Fallback: object path + rebuild.
            n = 0
            for nid, val in levels.items():
                neuron = self.neurons.get(int(nid))
                if neuron is None:
                    continue
                v = float(np.clip(val, 0.0, 1.0))
                neuron.activity = max(float(neuron.activity), v)
                n += 1
            if n:
                self.rebuild_activity_cache()
            return n
        if not act.flags.writeable:
            act = np.array(act, dtype=np.float32, copy=True)
            self.soma_activity = act
        contiguous = self._activity_ids_contiguous()
        lut = None if contiguous else {int(i): k for k, i in enumerate(ids.tolist())}
        n = 0
        for nid, val in levels.items():
            nid_i = int(nid)
            if contiguous:
                if nid_i < 0 or nid_i >= len(act) or nid_i not in self.neurons:
                    continue
                ia = nid_i
            else:
                ia = lut.get(nid_i)  # type: ignore[union-attr]
                if ia is None:
                    continue
            v = float(np.clip(val, 0.0, 1.0))
            nv = max(float(act[ia]), v)
            act[ia] = nv
            self.neurons[nid_i].activity = nv
            n += 1
        return n

    def ensure_growth_tips(
        self,
        sources: list[str],
        targets: list[str],
        *,
        seek_limit: float = GROW_SEEK_LIMIT,
        max_edges: int = GROW_MAX_RECIPE_EDGES,
    ) -> int:
        """Spawn tips for atlas recipe edges in sources → targets."""
        if self.phase == "adult":
            return int(len(self.growth_owner))
        req_src = [s for s in sources if s in self.sets]
        req_tgt = [t for t in targets if t in self.sets]
        self._growth_tree = None
        if len(self.recipe_pre) == 0:
            self.recipe_armed = 0
            return 0

        # Already armed: keep mask; only revive failed tips (no silent reselect).
        if len(self.growth_owner):
            if sorted(req_src) != sorted(self.growth_sources) or sorted(req_tgt) != sorted(
                self.growth_targets
            ):
                return int(len(self.growth_owner))
            failed = self.growth_status == STATUS_FAILED
            if np.any(failed):
                f_idx = np.flatnonzero(failed)
                gxyz = self.growth_goal_xyz[f_idx]
                self.growth_status[f_idx] = STATUS_GROWING
                self.growth_partner[f_idx] = -1
                self.growth_seek_t[f_idx] = 0.0
                self.growth_seek_limit[f_idx] = self._seek_limits_for_trips(
                    self.growth_soma[f_idx], gxyz, base=float(seek_limit)
                )
                self.growth_tip[f_idx] = self.growth_soma[f_idx] + self._tip_spawn_offset(
                    self.growth_soma[f_idx], gxyz
                )
                self.growth_poly[f_idx, 0, :] = self.growth_soma[f_idx]
                self.growth_poly[f_idx, 1, :] = self.growth_tip[f_idx]
                self.growth_poly_n[f_idx] = 2
            return int(len(self.growth_owner))

        self.growth_sources = req_src
        self.growth_targets = req_tgt
        pre, post, _w = self.select_recipe_edges(
            self.growth_sources, self.growth_targets, max_edges=max_edges
        )
        return self._install_growth_tips(pre, post, seek_limit=seek_limit)

    def restart_growth_tips(self, *, seek_limit: float = GROW_SEEK_LIMIT) -> int:
        """Reset armed recipe tips to growing (same atlas partners)."""
        if self.phase == "adult":
            return 0
        if len(self.growth_owner) == 0:
            if self.growth_sources and self.growth_targets:
                return self.ensure_growth_tips(
                    self.growth_sources, self.growth_targets, seek_limit=seek_limit
                )
            return 0
        return self._install_growth_tips(
            self.growth_owner, self.growth_goal, seek_limit=seek_limit
        )

    def _poly_append(self, tip_idx: np.ndarray, xyz: np.ndarray) -> None:
        """Append positions to per-tip polylines; keep soma+tip when compressing."""
        if len(tip_idx) == 0:
            return
        ns = self.growth_poly_n[tip_idx]
        not_full = ns < GROW_TRAIL_MAX
        if np.any(not_full):
            ii = tip_idx[not_full]
            kk = ns[not_full]
            self.growth_poly[ii, kk, :] = xyz[not_full]
            self.growth_poly_n[ii] = kk + 1
        full = ~not_full
        if np.any(full):
            ii = tip_idx[full]
            # Preserve endpoints: soma at [0], decimate middle, new tip at end.
            old = self.growth_poly[ii]
            keep = GROW_TRAIL_MAX - 1
            # Take soma + evenly spaced interior from old[1:].
            # Result length keep, then overwrite last with new xyz.
            n_old = GROW_TRAIL_MAX
            # indices into old for slots 0..keep-2 (last slot = new tip)
            slots = keep - 1  # number of retained old samples including soma
            idx_f = np.linspace(0, n_old - 1, slots).astype(np.int32)
            idx_f[0] = 0
            compressed = old[:, idx_f, :]
            self.growth_poly[ii, :slots, :] = compressed
            self.growth_poly[ii, slots, :] = xyz[full]
            self.growth_poly_n[ii] = GROW_TRAIL_MAX


    def step_growth(
        self,
        *,
        step_frac: float = GROW_STEP_FRAC,
        connect_frac: float = GROW_CONNECT_FRAC,
        tissue_blend: float = GROW_TISSUE_BLEND,
    ) -> dict[str, int]:
        """Tip motion via chemotaxis; tissue is substrate, not a rival attractor.

        U ≈ U_group (far / cross-mode) + U_cell (domain basin). Connected = tip
        entered the contact basin, not the post soma. `connect_frac` /
        `tissue_blend` kept for call-site compat.
        """
        out = {"moved": 0, "connected": 0}
        _ = (connect_frac, tissue_blend)
        # Device Fuse arms tips without atlas sources/targets — still advance
        # whatever is already in the growth field.
        if len(self.growth_owner) == 0:
            self.last_growth_step = out
            return out
        if len(self.growth_goal) != len(self.growth_owner):
            self.last_growth_step = out
            return out
        if len(self.growth_poly_n) != len(self.growth_owner):
            n = len(self.growth_owner)
            self.growth_poly = np.zeros((n, GROW_TRAIL_MAX, 3), dtype=np.float32)
            self.growth_poly[:, 0, :] = self.growth_soma
            self.growth_poly[:, 1, :] = self.growth_tip
            self.growth_poly_n = np.full(n, 2, dtype=np.int32)
        if len(self._growth_materialized) != len(self.growth_owner):
            self._growth_materialized = np.zeros(len(self.growth_owner), dtype=bool)
        if len(self.growth_basin_r) != len(self.growth_owner):
            span = self._soma_span()
            self.growth_basin_r = np.full(
                len(self.growth_owner), float(GROW_BASIN_FRAC) * span, dtype=np.float32
            )

        active = (self.growth_status == STATUS_GROWING) | (
            self.growth_status == STATUS_SEEKING
        )
        if not np.any(active):
            self.last_growth_step = out
            return out
        idx = np.flatnonzero(active)
        tip_xyz = self.growth_tip[idx]
        goals = self.growth_goal_xyz[idx]
        partners = self.growth_goal[idx]
        basin_r = self.growth_basin_r[idx]
        dist = np.linalg.norm(goals - tip_xyz, axis=1)

        span = self._soma_span()
        step = float(step_frac) * span

        # Soft connect near basin; snap SEEKING tips that reached the domain mouth.
        soft_r = basin_r * float(GROW_FINAL_CONNECT_MUL)
        was_seeking = self.growth_status[idx] == STATUS_SEEKING
        snap_r = basin_r * float(GROW_FINAL_SNAP_MUL)
        connected = (dist <= soft_r) | (was_seeking & (dist <= snap_r))
        moving = (~connected) & (dist > 1e-12)
        if np.any(connected):
            c_idx = idx[connected]
            self.growth_tip[c_idx] = goals[connected]
            self.growth_status[c_idx] = STATUS_CONNECTED
            self.growth_partner[c_idx] = partners[connected]
            self._poly_append(c_idx, goals[connected])
            out["connected"] = int(np.sum(connected))
            self._materialize_connections(c_idx)
        if np.any(moving):
            m_idx = idx[moving]
            tip_m = tip_xyz[moving]
            goal_m = goals[moving]
            d = dist[moving]
            br = basin_r[moving]
            final = d <= float(GROW_FINAL_APPROACH_MUL) * br
            tip_mode = self._point_modes(tip_m)
            goal_mode = self._point_modes(goal_m)
            instruct = self._chemotaxis_dir(tip_m, tip_mode, goal_mode, goal_m)
            # Final approach: home straight into the contact domain (no peer orbit).
            if np.any(final):
                cell = goal_m[final] - tip_m[final]
                clen = np.linalg.norm(cell, axis=1, keepdims=True)
                instruct = instruct.copy()
                instruct[final] = (cell / np.maximum(clen, 1e-6)).astype(np.float32)
            if np.any(~final):
                far_idx = np.flatnonzero(~final)
                instruct_far = self._peer_bundle_dir(
                    tip_m[far_idx], m_idx[far_idx], goal_m[far_idx], instruct[far_idx]
                )
                instruct_far = self._device_funnel_dir(tip_m[far_idx], instruct_far)
                instruct = instruct.copy()
                instruct[far_idx] = instruct_far
            direction = self._substrate_correction(tip_m, instruct)
            # Don't let substrate yank tips off the final home-in.
            if np.any(final):
                direction = direction.copy()
                direction[final] = instruct[final]
            cross = (tip_mode >= 0) & (goal_mode >= 0) & (tip_mode != goal_mode)
            step_i = np.full(len(m_idx), step, dtype=np.float32)
            step_i[cross] = step * float(GROW_CROSS_STEP_MUL)
            step_i[final] = np.maximum(
                step_i[final], step * float(GROW_FINAL_STEP_MUL)
            )
            # Far: leave a small standoff; final: always close the remaining gap.
            advance = np.minimum(step_i, np.maximum(d - 0.25 * br, 0.0))
            advance = np.where(final, np.minimum(step_i, d), advance)[:, None]
            finish = d <= np.maximum(2.5 * step_i, 1.5 * br)
            advance = np.where(finish[:, None], np.minimum(step_i, d)[:, None], advance)
            new_tip = tip_m + direction * advance
            self.growth_tip[m_idx] = new_tip
            self._poly_append(m_idx, new_tip)
            self.growth_status[m_idx] = np.where(
                finish | final, STATUS_SEEKING, STATUS_GROWING
            ).astype(np.uint8)
            out["moved"] = int(np.sum(moving))
        self.last_growth_step = dict(out)
        return out

    def _make_dendrite_points(
        self,
        partner_soma: np.ndarray,
        contact: np.ndarray,
        axon_pts: np.ndarray,
        span: float,
    ) -> np.ndarray:
        """Multi-point dendrite stub: synapse → back along axon cable toward post.

        Hugs the arriving axon trail so post arbors meet in the same dense cable
        mouth; slight bow toward the post soma keeps ownership readable.
        """
        stub_len = min(GROW_DEND_STUB_FRAC * span, GROW_DEND_STUB_MAX)
        contact = np.asarray(contact, dtype=np.float64).reshape(3)
        soma = np.asarray(partner_soma, dtype=np.float64).reshape(3)
        ax = np.asarray(axon_pts, dtype=np.float64)
        n = max(3, int(GROW_DEND_POLY_N))
        to_soma = soma - contact
        ts = float(np.linalg.norm(to_soma))
        to_soma_u = to_soma / ts if ts > 1e-8 else np.zeros(3, dtype=np.float64)

        # Walk back along the axon poly from the tip (contact) for stub_len.
        path: list[np.ndarray] = [contact.copy()]
        budget = float(stub_len)
        if len(ax) >= 2:
            # Ensure tip ≈ contact for distance walk.
            tip = ax[-1]
            if float(np.linalg.norm(tip - contact)) > 1e-6:
                ax = np.concatenate([ax, contact.reshape(1, 3)], axis=0)
            for i in range(len(ax) - 1, 0, -1):
                a = ax[i]
                b = ax[i - 1]
                seg = b - a
                seg_len = float(np.linalg.norm(seg))
                if seg_len < 1e-9:
                    continue
                if seg_len <= budget:
                    path.append(b.copy())
                    budget -= seg_len
                    if budget <= 1e-9:
                        break
                else:
                    path.append(a + seg * (budget / seg_len))
                    budget = 0.0
                    break
        if len(path) < 2:
            # Fallback: short chord toward post soma.
            reach = min(stub_len, 0.35 * max(ts, stub_len))
            if ts < 1e-8:
                toward = np.array([reach, 0.0, 0.0], dtype=np.float64)
            else:
                toward = to_soma_u * reach
            path = [contact.copy(), contact + toward]

        raw = np.stack(path, axis=0)
        # Resample to fixed N; bow mid samples toward post soma.
        if len(raw) == 2:
            t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
            pts = raw[0][None, :] * (1.0 - t) + raw[1][None, :] * t
        else:
            chord = np.cumsum(
                np.concatenate(
                    [
                        [0.0],
                        np.linalg.norm(np.diff(raw, axis=0), axis=1),
                    ]
                )
            )
            total = float(chord[-1])
            if total < 1e-9:
                pts = np.repeat(contact.reshape(1, 3), n, axis=0)
            else:
                samples = np.linspace(0.0, total, n, dtype=np.float64)
                pts = np.column_stack(
                    [
                        np.interp(samples, chord, raw[:, d])
                        for d in range(3)
                    ]
                )
        reach = float(np.linalg.norm(pts[-1] - pts[0]))
        if reach > 1e-9 and ts > 1e-8:
            for i in range(1, n - 1):
                t = i / (n - 1)
                bow = math.sin(math.pi * t) * 0.12 * reach
                pts[i] = pts[i] + to_soma_u * bow
        pts[0] = contact
        return pts.astype(np.float32)

    def _materialize_connections(self, tip_idx: np.ndarray) -> None:
        """Commit connected tips as axon + dendrite neurites (many per neuron)."""
        if len(tip_idx) == 0:
            return
        if len(self._growth_materialized) != len(self.growth_owner):
            self._growth_materialized = np.zeros(len(self.growth_owner), dtype=bool)
        span = self._soma_span()
        any_new = False
        for i in tip_idx.tolist():
            i = int(i)
            if self._growth_materialized[i]:
                continue
            owner_id = int(self.growth_owner[i])
            partner_id = int(self.growth_goal[i])
            owner = self.neurons.get(owner_id)
            partner = self.neurons.get(partner_id)
            if owner is None or partner is None:
                continue
            k = int(self.growth_poly_n[i])
            if k < 2:
                pts = np.stack(
                    [self.growth_soma[i], self.growth_tip[i]], axis=0
                ).astype(np.float32)
            else:
                pts = self.growth_poly[i, :k].astype(np.float32, copy=True)
            ax_id = self._next_neurite_id
            self._next_neurite_id += 1
            owner.axons.append(
                Neurite(
                    id=ax_id,
                    kind="axon",
                    points=pts,
                    status="connected",
                    partner_id=partner_id,
                    seek_t=float(self.growth_seek_t[i]),
                    seek_limit=float(self.growth_seek_limit[i]),
                )
            )
            contact = pts[-1]
            dend_pts = self._make_dendrite_points(
                partner.soma, contact, pts, span
            )
            den_id = self._next_neurite_id
            self._next_neurite_id += 1
            partner.dendrites.append(
                Neurite(
                    id=den_id,
                    kind="dendrite",
                    points=dend_pts,
                    status="connected",
                    partner_id=owner_id,
                )
            )
            self._growth_materialized[i] = True
            any_new = True
        if any_new:
            self.rebuild_caches()

    def growth_trail_point_count(self) -> int:
        """Samples along live growth polylines (or materialized neurites if idle)."""
        if len(self.growth_poly_n):
            return int(self.growth_poly_n.sum())
        return int(len(self.neurite_xyz))

    def growth_tip_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tip positions / status codes / owner cls for overlay render."""
        if len(self.growth_owner) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        return (
            self.growth_tip.astype(np.float32, copy=False),
            self.growth_status.astype(np.int32, copy=False),
            self.growth_cls.astype(np.int32, copy=False),
        )

    def neck_trail_segments(
        self,
        *,
        max_segs: int = 1200,
        radius_frac: float = 0.12,
        cls_filter: set[int] | frozenset[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Polyline segments that pass near the cervical connective.

        Yield can be 100% while the neck looks empty if overlay sampling
        thins away the few trails that actually cross — keep those explicit.
        """
        empty = (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
        if len(self.growth_owner) == 0 or len(self.growth_poly_n) == 0:
            return empty
        if not self._ensure_neck_waypoint():
            return empty
        neck = self._neck_xyz[0].astype(np.float64)
        rad = float(radius_frac) * self._soma_span()
        parts_a: list[np.ndarray] = []
        parts_b: list[np.ndarray] = []
        parts_c: list[np.ndarray] = []
        n = len(self.growth_owner)
        order = np.arange(n)
        if cls_filter:
            order = order[np.isin(self.growth_cls, list(cls_filter))]
        # Prefer longer trails (more likely cross-mode completed paths).
        if len(order):
            order = order[np.argsort(-self.growth_poly_n[order])]
        budget = int(max_segs)
        for i in order.tolist():
            if budget <= 0:
                break
            k = int(self.growth_poly_n[i])
            if k < 2:
                continue
            poly = self.growth_poly[i, :k].astype(np.float64)
            d = np.linalg.norm(poly - neck, axis=1)
            near = np.flatnonzero(d <= rad)
            if len(near) == 0:
                continue
            # Emit consecutive segment pairs that touch the neck slab.
            cls_i = int(self.growth_cls[i])
            for j in near.tolist():
                if budget <= 0:
                    break
                if j <= 0:
                    a, b = poly[0], poly[min(1, k - 1)]
                elif j >= k - 1:
                    a, b = poly[k - 2], poly[k - 1]
                else:
                    a, b = poly[j], poly[j + 1]
                parts_a.append(a.astype(np.float32).reshape(1, 3))
                parts_b.append(b.astype(np.float32).reshape(1, 3))
                parts_c.append(np.array([cls_i], dtype=np.int32))
                budget -= 1
        if not parts_a:
            return empty
        return (
            np.concatenate(parts_a, axis=0),
            np.concatenate(parts_b, axis=0),
            np.concatenate(parts_c, axis=0),
        )

    def growth_overlay_arrays(
        self,
        *,
        tip_max: int = GROW_TIP_DRAW_MAX,
        ray_max: int = GROW_RAY_DRAW_MAX,
        cls_filter: set[int] | frozenset[int] | None = None,
        poly_points: int | None = None,
        seg_max: int | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Tips + tissue polylines (path samples) + dendrite stubs at synapses.

        Axon: consecutive segments along growth_poly (not a free-space chord).
        Stratified by class so a majority class does not hide others.
        When cls_filter is set, budget is spent only on those classes (focus).
        """
        empty = (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
        if len(self.growth_owner) == 0:
            return empty
        tip = self.growth_tip.astype(np.float32, copy=False)
        code = self.growth_status.astype(np.int32, copy=False)
        cls = self.growth_cls.astype(np.int32, copy=False)
        soma = self.growth_soma.astype(np.float32, copy=False)
        active = (code == STATUS_GROWING) | (code == STATUS_SEEKING)
        connected = code == STATUS_CONNECTED
        axon_draw = active | connected
        if cls_filter:
            in_f = np.isin(cls, list(cls_filter))
            active = active & in_f
            connected = connected & in_f
            axon_draw = axon_draw & in_f
        n = len(code)
        poly_keep = int(GROW_POLY_DRAW_POINTS if poly_points is None else poly_points)
        seg_budget_cap = int(GROW_SEG_DRAW_MAX if seg_max is None else seg_max)
        tip_pool = np.flatnonzero(in_f) if cls_filter else np.arange(n)
        if len(tip_pool) <= tip_max:
            tip_out, code_out = tip[tip_pool], code[tip_pool]
        else:
            pool_act = np.flatnonzero(active)
            act_idx = self._stratified_indices(pool_act, cls, tip_max)
            remain = tip_max - len(act_idx)
            if remain > 0:
                if cls_filter:
                    idle_pool = tip_pool[~np.isin(tip_pool, act_idx)]
                    # idle among filtered, not already taken
                    idle_mask = np.ones(n, dtype=bool)
                    idle_mask[act_idx] = False
                    idle_pool = np.flatnonzero(idle_mask & in_f & ~active)
                else:
                    idle_pool = np.flatnonzero(~active)
                idle_idx = self._stratified_indices(idle_pool, cls, remain)
                sel = (
                    np.concatenate([act_idx, idle_idx])
                    if len(idle_idx)
                    else act_idx
                )
            else:
                sel = act_idx
            tip_out = tip[sel]
            code_out = code[sel]

        ax_idx = np.flatnonzero(axon_draw)
        if len(ax_idx) == 0:
            rays_a = np.zeros((0, 3), dtype=np.float32)
            rays_b = np.zeros((0, 3), dtype=np.float32)
            ray_cls = np.zeros((0,), dtype=np.int32)
        elif len(self.growth_poly_n) != n:
            ax_idx = self._stratified_indices(ax_idx, cls, ray_max)
            rays_a = soma[ax_idx]
            rays_b = tip[ax_idx]
            ray_cls = cls[ax_idx]
        else:
            ax_idx = self._stratified_indices(ax_idx, cls, ray_max)
            # Equal segment budget per class so multi-focus never starves one type.
            classes_present = np.unique(cls[ax_idx])
            n_c = max(1, int(len(classes_present)))
            per_cls_seg = max(1, seg_budget_cap // n_c)
            parts_a: list[np.ndarray] = []
            parts_b: list[np.ndarray] = []
            parts_c: list[np.ndarray] = []
            for c in classes_present.tolist():
                sub = ax_idx[cls[ax_idx] == int(c)]
                seg_budget = per_cls_seg
                for i in sub.tolist():
                    if seg_budget <= 0:
                        break
                    k = int(self.growth_poly_n[i])
                    if k < 2:
                        parts_a.append(soma[i : i + 1])
                        parts_b.append(tip[i : i + 1])
                        parts_c.append(cls[i : i + 1])
                        seg_budget -= 1
                        continue
                    poly = self.growth_poly[i, :k]
                    keep = min(poly_keep, k)
                    if keep < k:
                        sel = np.linspace(0, k - 1, keep).astype(np.int32)
                        sel[0] = 0
                        sel[-1] = k - 1
                        poly = poly[sel]
                        k = keep
                    nseg = k - 1
                    if nseg > seg_budget:
                        poly = poly[: seg_budget + 1]
                        nseg = int(len(poly) - 1)
                    parts_a.append(poly[:-1])
                    parts_b.append(poly[1:])
                    parts_c.append(np.full(nseg, int(cls[i]), dtype=np.int32))
                    seg_budget -= nseg
            if parts_a:
                rays_a = np.concatenate(parts_a, axis=0).astype(np.float32)
                rays_b = np.concatenate(parts_b, axis=0).astype(np.float32)
                ray_cls = np.concatenate(parts_c, axis=0).astype(np.int32)
            else:
                rays_a = np.zeros((0, 3), dtype=np.float32)
                rays_b = np.zeros((0, 3), dtype=np.float32)
                ray_cls = np.zeros((0,), dtype=np.int32)

        # Dendrite stubs: synapse → short step back toward pre (same chord).
        conn_idx = np.flatnonzero(connected)
        if len(conn_idx) == 0:
            dend_a = np.zeros((0, 3), dtype=np.float32)
            dend_b = np.zeros((0, 3), dtype=np.float32)
            dend_cls = np.zeros((0,), dtype=np.int32)
        else:
            conn_idx = self._stratified_indices(conn_idx, cls, ray_max)
            syn = tip[conn_idx]
            pre = soma[conn_idx]
            delta = pre - syn
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            stub = min(GROW_DEND_STUB_FRAC * self._soma_span(), GROW_DEND_STUB_MAX)
            safe = np.maximum(dist, 1e-6)
            back = delta / safe * np.minimum(stub, 0.35 * dist)
            dend_a = syn.astype(np.float32, copy=False)
            dend_b = (syn + back).astype(np.float32)
            # Color by post (target) class when resolvable.
            goals = self.growth_goal[conn_idx]
            dend_cls = np.full(len(conn_idx), -1, dtype=np.int32)
            ids = self.soma_ids
            contiguous = (
                len(ids) == len(self.soma_cls)
                and len(ids) > 0
                and int(ids[0]) == 0
                and int(ids[-1]) == len(ids) - 1
            )
            if contiguous:
                valid = (goals >= 0) & (goals < len(self.soma_cls))
                dend_cls[valid] = self.soma_cls[goals[valid]]
            else:
                lut = {
                    int(i): int(c)
                    for i, c in zip(ids.tolist(), self.soma_cls.tolist())
                }
                for i, gid in enumerate(goals.tolist()):
                    dend_cls[i] = lut.get(int(gid), -1)
            # Fallback to owner class if goal class unknown.
            miss = dend_cls < 0
            if np.any(miss):
                dend_cls[miss] = cls[conn_idx[miss]]

        return tip_out, code_out, rays_a, rays_b, ray_cls, dend_a, dend_b, dend_cls

    def clear_growth_tips(self) -> int:
        """Clear vectorized growth field and neuron axons/dendrites; atlas stays.

        Does not change phase — caller must `enter_rising()` or `enter_adult()`.
        """
        removed = int(len(self.growth_owner))
        for neuron in self.neurons.values():
            removed += len(neuron.axons) + len(neuron.dendrites)
            neuron.axons = []
            neuron.dendrites = []
        self.growth_sources = []
        self.growth_targets = []
        self.growth_owner = np.zeros((0,), dtype=np.int32)
        self.growth_soma = np.zeros((0, 3), dtype=np.float32)
        self.growth_tip = np.zeros((0, 3), dtype=np.float32)
        self.growth_status = np.zeros((0,), dtype=np.uint8)
        self.growth_partner = np.full((0,), -1, dtype=np.int32)
        self.growth_goal = np.full((0,), -1, dtype=np.int32)
        self.growth_goal_xyz = np.zeros((0, 3), dtype=np.float32)
        self.growth_basin_r = np.zeros((0,), dtype=np.float32)
        self.growth_seek_t = np.zeros((0,), dtype=np.float32)
        self.growth_seek_limit = np.full((0,), GROW_SEEK_LIMIT, dtype=np.float32)
        self.growth_cls = np.zeros((0,), dtype=np.int32)
        self.growth_poly = np.zeros((0, GROW_TRAIL_MAX, 3), dtype=np.float32)
        self.growth_poly_n = np.zeros((0,), dtype=np.int32)
        self._growth_materialized = np.zeros((0,), dtype=bool)
        self.last_growth_step = {"moved": 0, "connected": 0}
        self.recipe_armed = 0
        self._growth_tree = None
        self._process_tree = None
        self._process_tree_xyz = np.zeros((0, 3), dtype=np.float32)
        self._class_process_trees = {}
        self._class_process_xyz = {}
        self._tissue_mode_centers = np.zeros((0, 3), dtype=np.float32)
        self._tissue_mode_labels = np.zeros((0,), dtype=np.int32)
        self._neck_xyz = np.zeros((0, 3), dtype=np.float32)
        self._tissue_pull_cache = None
        self._tissue_pull_cache_tick = -999
        self._growth_tree_tgt_ids = np.zeros((0,), dtype=np.int32)
        self._growth_tree_tgt_xyz = np.zeros((0, 3), dtype=np.float32)
        self._clear_bundle_cohorts()
        self.rebuild_caches()
        return removed

    def enter_rising(self) -> FlyPhase:
        """Developmental phase: tips may grow toward contact domains."""
        self.phase = "rising"
        return self.phase

    def enter_adult(self, *, abandon_open: bool = True) -> FlyPhase:
        """Use phase: freeze growth; keep connected contacts for signals.

        May be called before 100% Yield — open tips stop moving.
        By default open tips are marked failed so Yield matches the freeze.
        """
        if abandon_open and len(self.growth_status):
            open_m = (self.growth_status == STATUS_GROWING) | (
                self.growth_status == STATUS_SEEKING
            )
            if np.any(open_m):
                self.growth_status[open_m] = STATUS_FAILED
                if len(self.growth_partner) == len(self.growth_status):
                    self.growth_partner[open_m] = -1
        self.phase = "adult"
        return self.phase

    def is_adult(self) -> bool:
        return self.phase == "adult"

    def is_rising(self) -> bool:
        return self.phase == "rising"

    def sim_tick(self) -> int:
        """One full simulation step. Returns fly-time after the tick."""
        self.sim_time += 1
        if self.phase == "rising":
            self.step_growth()
            self.tick_neurites(1.0)
        return self.sim_time

    def class_counts(self) -> np.ndarray:
        n = len(self.class_names)
        return np.bincount(self.soma_cls, minlength=n).astype(np.int32)

    def class_basin_fields(
        self,
        *,
        top_n: int = 8,
        min_count: int = 400,
        class_filter: set[int] | None = None,
        max_modes: int = 4,
        min_frac: float = 0.06,
        min_frac_afferent: float | None = None,
        sep_factor: float = 1.20,
        sep_factor_afferent: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Spatial basins = dense tissue agglomerations (attractor wells).

        One class can yield several basins (e.g. left/right optic lobes).
        Prefers soma clouds when a class has a real soma mass; otherwise uses
        the dense core of the process cloud (not the full axon span — that
        falsely inflates projection classes like sensory).

        ``max_modes`` / ``min_frac`` / ``sep_factor`` control how split the
        wells are (UI basin-form slider). Afferent overrides default to the
        ``*_afferent`` kwargs when provided.

        Returns cls_idx [K], centers [K,3], radii [K], n_pts [K] (dense-core size).
        """
        empty = (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
        n_cls = len(self.class_names)
        if n_cls == 0:
            return empty

        has_proc = len(self.atlas_process_xyz) and len(self.atlas_process_cls) == len(
            self.atlas_process_xyz
        )
        has_soma = len(self.soma_xyz) and len(self.soma_cls) == len(self.soma_xyz)
        if not has_proc and not has_soma:
            return empty

        soma_counts = (
            np.bincount(self.soma_cls, minlength=n_cls).astype(np.int32)
            if has_soma
            else np.zeros(n_cls, dtype=np.int32)
        )
        proc_counts = (
            np.bincount(self.atlas_process_cls, minlength=n_cls).astype(np.int32)
            if has_proc
            else np.zeros(n_cls, dtype=np.int32)
        )
        # Rank default list by soma mass — neuropil first. Also admit
        # process-heavy afferents whose somata sit outside the CNS atlas
        # (sensory: head hairs / eye / dorsal entry sites).
        thr_soma = max(80, int(min_count) // 5)
        thr_proc = max(400, int(min_count))
        if class_filter is not None:
            cand = [int(c) for c in sorted(class_filter) if 0 <= int(c) < n_cls]
        else:
            order = np.argsort(-soma_counts)
            cand = [
                int(c)
                for c in order.tolist()
                if int(soma_counts[int(c)]) >= thr_soma
            ][: max(0, int(top_n))]
            seen = set(cand)
            # Peripheral-soma classes: large process mass, few atlas somata.
            proc_order = np.argsort(-proc_counts)
            for c in proc_order.tolist():
                c = int(c)
                if c in seen:
                    continue
                if int(soma_counts[c]) < thr_soma and int(proc_counts[c]) >= thr_proc:
                    cand.append(c)
                    seen.add(c)
            if not cand:
                # Tiny atlases: fall back to any class with enough somata or processes.
                order_p = np.argsort(-(soma_counts * 10 + proc_counts))
                cand = [
                    int(c)
                    for c in order_p.tolist()
                    if int(soma_counts[int(c)]) + int(proc_counts[int(c)]) >= 30
                ][: max(0, int(top_n))]
        if not cand:
            return empty

        span = max(float(self._soma_span()), 1e-3)
        floor_r = float(GROW_BASIN_FRAC) * span

        cls_out: list[int] = []
        ctr_out: list[np.ndarray] = []
        rad_out: list[float] = []
        n_out: list[int] = []
        for c in cand:
            n_soma_c = int(soma_counts[int(c)]) if has_soma else 0
            n_proc_c = int(proc_counts[int(c)]) if has_proc else 0
            # Soma cloud when it is a real CNS agglomeration, or whenever this
            # class is focused (live / device somata use the same well rules).
            focused = class_filter is not None and int(c) in class_filter
            use_soma = has_soma and n_soma_c >= 3 and (
                focused
                or (n_soma_c >= 80 and n_soma_c >= max(1, n_proc_c // 30))
                or n_proc_c < 3
            )
            clouds: list[tuple[np.ndarray, bool]] = []
            if use_soma:
                clouds.append((self.soma_xyz[self.soma_cls == int(c)], False))
            elif has_proc and n_proc_c >= 3:
                clouds.append(
                    (self.atlas_process_xyz[self.atlas_process_cls == int(c)], True)
                )
            elif has_soma and n_soma_c >= 3:
                clouds.append((self.soma_xyz[self.soma_cls == int(c)], False))
            else:
                continue
            modes_cap = max(1, int(max_modes))
            for pts, afferent_cloud in clouds:
                n_pts = len(pts)
                if n_pts < 3:
                    continue
                if n_pts > 12000:
                    step = max(1, n_pts // 8000)
                    pts = pts[::step]
                afferent = bool(afferent_cloud) and n_proc_c >= thr_proc
                mf = float(
                    min_frac_afferent
                    if afferent and min_frac_afferent is not None
                    else min_frac
                )
                sf = float(
                    sep_factor_afferent
                    if afferent and sep_factor_afferent is not None
                    else sep_factor
                )
                for center, radius, n_mode in self._agglomeration_basins(
                    pts,
                    floor_r=floor_r,
                    max_modes=modes_cap,
                    min_frac=mf,
                    sep_factor=sf,
                ):
                    cls_out.append(int(c))
                    ctr_out.append(center)
                    rad_out.append(float(radius))
                    n_out.append(int(n_mode))

        if not cls_out:
            return empty
        order = np.argsort(-np.asarray(n_out, dtype=np.int32))
        return (
            np.asarray(cls_out, dtype=np.int32)[order],
            np.stack(ctr_out, axis=0).astype(np.float32)[order],
            np.asarray(rad_out, dtype=np.float32)[order],
            np.asarray(n_out, dtype=np.int32)[order],
        )

    @staticmethod
    def _dense_core_points(
        pts: np.ndarray, *, keep_frac: float = 0.35, knn: int = 8
    ) -> np.ndarray:
        """Keep the locally densest fraction — drop long axon tails."""
        from scipy.spatial import cKDTree

        n = len(pts)
        if n < 12:
            return pts
        k = min(int(knn) + 1, n)
        try:
            d, _ = cKDTree(pts).query(pts, k=k)
        except Exception:
            return pts
        if k == 1:
            knn_mean = np.asarray(d, dtype=np.float64).reshape(-1)
        else:
            knn_mean = np.asarray(d, dtype=np.float64)[:, 1:].mean(axis=1)
        dens = 1.0 / (knn_mean + 1e-9)
        thr = float(np.quantile(dens, max(0.0, 1.0 - float(keep_frac))))
        core = pts[dens >= thr]
        return core if len(core) >= 8 else pts

    def _agglomeration_basins(
        self,
        pts: np.ndarray,
        *,
        floor_r: float,
        max_modes: int = 4,
        min_frac: float = 0.10,
        sep_factor: float = 1.55,
    ) -> list[tuple[np.ndarray, float, int]]:
        """Split a class cloud into well-separated dense modes (e.g. two eyes).

        ``min_frac`` is soft: small secondary peaks must still qualify —
        absolute floor is modest so satellite modes are not dropped.
        """
        from scipy.spatial import cKDTree

        n = len(pts)
        if n < 3:
            return []

        def _mode_stats(sub: np.ndarray) -> tuple[np.ndarray, float, int] | None:
            if len(sub) < 3:
                return None
            # Dense core only — drop long projection tails.
            core_pts = self._dense_core_points(sub)
            center = core_pts.mean(axis=0).astype(np.float32)
            # Packing scale (kNN), not convex hull of the set.
            k = min(9, len(core_pts))
            try:
                d_knn, _ = cKDTree(core_pts).query(core_pts, k=k)
                if k == 1:
                    med_knn = float(np.median(d_knn))
                else:
                    med_knn = float(np.median(d_knn[:, 1:]))
            except Exception:
                med_knn = float(np.median(np.linalg.norm(core_pts - center, axis=1)))
            dist = np.linalg.norm(core_pts.astype(np.float64) - center, axis=1)
            half_extent = float(np.percentile(dist, 50))
            # Well = local tissue pack; cap by half-extent so diffuse clouds stay small.
            pack_r = float(GROW_BASIN_SPREAD) * med_knn * 10.0
            radius = max(float(floor_r), min(half_extent, pack_r))
            if half_extent > 0:
                radius = max(float(floor_r), min(radius, half_extent))
            return center, float(radius), int(len(core_pts))

        def _try_k(k: int) -> list[tuple[np.ndarray, float, int]] | None:
            k = max(1, min(int(k), n))
            if k == 1:
                one = _mode_stats(pts)
                return [one] if one is not None else None
            labels, centers = self._kmeans_modes(pts, k=k)
            # Soft floor: fraction of the cloud, not a hard 25 (blocks small
            # dense modes such as local sensory clusters).
            min_n = max(3, min(int(min_frac * n), n // (2 * k)))
            modes: list[tuple[np.ndarray, float, int]] = []
            for ci in range(k):
                sub = pts[labels == ci]
                if len(sub) < min_n:
                    return None
                st = _mode_stats(sub)
                if st is None:
                    return None
                modes.append(st)
            for i in range(len(modes)):
                for j in range(i + 1, len(modes)):
                    d = float(np.linalg.norm(modes[i][0] - modes[j][0]))
                    need = float(sep_factor) * 0.5 * (modes[i][1] + modes[j][1])
                    if d < need:
                        return None
            _ = centers
            return modes

        def _density_valley_cuts(
            vals: np.ndarray, *, max_bands: int
        ) -> list[float] | None:
            """Cut elongated clouds at 1D density valleys between modes."""
            from scipy.ndimage import gaussian_filter1d

            vals = np.asarray(vals, dtype=np.float64)
            if len(vals) < 40:
                return None
            n_bins = max(24, min(48, len(vals) // 8))
            hist, edges = np.histogram(vals, bins=n_bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            smooth = gaussian_filter1d(hist.astype(np.float64), sigma=1.0)
            floor = max(5.0, 0.10 * float(smooth.max()))
            peaks: list[int] = []
            for i in range(1, len(smooth) - 1):
                if (
                    smooth[i] >= smooth[i - 1]
                    and smooth[i] >= smooth[i + 1]
                    and smooth[i] >= floor
                ):
                    peaks.append(i)
            if len(peaks) < 2:
                return None
            # Merge nearly-adjacent peaks (weak canyon).
            bin_w = float(edges[1] - edges[0])
            merged: list[int] = []
            for i in peaks:
                if not merged:
                    merged.append(i)
                    continue
                prev = merged[-1]
                if float(centers[i] - centers[prev]) < 1.5 * bin_w:
                    if smooth[i] > smooth[prev]:
                        merged[-1] = i
                else:
                    merged.append(i)
            peaks = merged
            if len(peaks) > max_bands:
                keep = sorted(peaks, key=lambda i: smooth[i], reverse=True)[
                    : max_bands
                ]
                peaks = sorted(keep)
            if len(peaks) < 2:
                return None
            cuts: list[float] = []
            for a, b in zip(peaks[:-1], peaks[1:]):
                seg = range(int(a), int(b) + 1)
                vi = min(seg, key=lambda i: smooth[i])
                cuts.append(float(centers[vi]))
            return cuts

        def _banded_bilateral(
            axis: int, n_bands: int
        ) -> list[tuple[np.ndarray, float, int]] | None:
            """Cut along the long axis (density valleys), then L/R split per band."""
            vals = pts[:, int(axis)]
            n_bands = max(2, min(int(n_bands), 8))
            cuts = _density_valley_cuts(vals, max_bands=n_bands)
            if cuts:
                edges = np.unique(
                    np.asarray(
                        [float(vals.min()) - 1e-6, *cuts, float(vals.max()) + 1e-6],
                        dtype=np.float64,
                    )
                )
            else:
                edges = np.unique(
                    np.asarray(
                        np.quantile(vals, np.linspace(0.0, 1.0, n_bands + 1)),
                        dtype=np.float64,
                    )
                )
            if len(edges) < 3:
                return None
            out: list[tuple[np.ndarray, float, int]] = []
            lat = 0 if int(axis) != 0 else 1  # bilateral ≈ x for Z-long body
            for b in range(len(edges) - 1):
                lo, hi = float(edges[b]), float(edges[b + 1])
                if b == len(edges) - 2:
                    mask = (vals >= lo) & (vals <= hi)
                else:
                    mask = (vals >= lo) & (vals < hi)
                sub = pts[mask]
                if len(sub) < 20:
                    continue
                # Force L/R by lateral sign when the band is bilateral.
                if (
                    len(sub) >= 40
                    and float(np.ptp(sub[:, lat])) > 1.2
                    and int((sub[:, lat] < 0).sum()) >= 12
                    and int((sub[:, lat] >= 0).sum()) >= 12
                ):
                    pieces = (sub[sub[:, lat] < 0], sub[sub[:, lat] >= 0])
                    for piece in pieces:
                        if len(piece) < 12:
                            continue
                        st = _mode_stats(piece)
                        if st is not None:
                            out.append(st)
                    continue
                st = _mode_stats(sub)
                if st is not None:
                    out.append(st)
            return out if len(out) >= 2 else None

        chosen: list[tuple[np.ndarray, float, int]] | None = None
        # Try all k; keep the richest split that still validates.
        for k in range(1, max(1, int(max_modes)) + 1):
            modes = _try_k(k)
            if modes is not None:
                chosen = modes

        # Elongated clouds: density-valley bands × bilateral split (geometry only).
        extent = np.ptp(pts, axis=0)
        long_ax = int(np.argmax(extent))
        if (
            n >= 200
            and float(extent[long_ax]) > 2.2 * float(np.median(extent))
            and int(max_modes) >= 4
        ):
            banded = _banded_bilateral(long_ax, min(5, int(max_modes)))
            if banded is not None and (
                chosen is None or len(banded) >= len(chosen)
            ):
                chosen = banded
        return chosen or []

    def neurite_counts(self) -> np.ndarray:
        n = len(self.class_names)
        if len(self.neurite_cls) == 0:
            return np.zeros(n, dtype=np.int32)
        return np.bincount(self.neurite_cls, minlength=n).astype(np.int32)

    def rebuild_caches(self) -> None:
        if not self.neurons:
            self.soma_xyz = np.zeros((0, 3), dtype=np.float32)
            self.soma_cls = np.zeros((0,), dtype=np.int32)
            self.soma_ids = np.zeros((0,), dtype=np.int32)
            self.soma_activity = np.zeros((0,), dtype=np.float32)
            self.neurite_xyz = np.zeros((0, 3), dtype=np.float32)
            self.neurite_cls = np.zeros((0,), dtype=np.int32)
            return

        ids = sorted(self.neurons.keys())
        self.soma_ids = np.asarray(ids, dtype=np.int32)
        self.soma_xyz = np.stack(
            [self.neurons[i].soma for i in ids], axis=0
        ).astype(np.float32)
        self.soma_cls = np.asarray(
            [self.neurons[i].cls for i in ids], dtype=np.int32
        )
        self.soma_activity = np.asarray(
            [self.neurons[i].activity for i in ids], dtype=np.float32
        )

        chunks_xyz: list[np.ndarray] = []
        chunks_cls: list[np.ndarray] = []
        for i in ids:
            neuron = self.neurons[i]
            for neurite in neuron.axons:
                if len(neurite.points) == 0:
                    continue
                chunks_xyz.append(neurite.points.astype(np.float32, copy=False))
                chunks_cls.append(
                    np.full(len(neurite.points), neuron.cls, dtype=np.int32)
                )
            for neurite in neuron.dendrites:
                if len(neurite.points) == 0:
                    continue
                chunks_xyz.append(neurite.points.astype(np.float32, copy=False))
                chunks_cls.append(
                    np.full(len(neurite.points), neuron.cls, dtype=np.int32)
                )
        if chunks_xyz:
            self.neurite_xyz = np.concatenate(chunks_xyz, axis=0)
            self.neurite_cls = np.concatenate(chunks_cls, axis=0)
        else:
            self.neurite_xyz = np.zeros((0, 3), dtype=np.float32)
            self.neurite_cls = np.zeros((0,), dtype=np.int32)

    def rebuild_activity_cache(self) -> None:
        if not self.neurons or len(self.soma_ids) == 0:
            self.soma_activity = np.zeros((0,), dtype=np.float32)
            return
        self.soma_activity = np.asarray(
            [self.neurons[int(i)].activity for i in self.soma_ids],
            dtype=np.float32,
        )

    def _activity_ids_contiguous(self) -> bool:
        ids = self.soma_ids
        return bool(
            len(ids) > 0
            and int(ids[0]) == 0
            and int(ids[-1]) == len(ids) - 1
            and len(ids) == len(self.neurons)
        )

    def _writeback_activity_indices(self, indices: np.ndarray) -> None:
        """Copy ``soma_activity[indices]`` into ``Neuron.activity`` (sparse)."""
        if indices is None or len(indices) == 0:
            return
        act = self.soma_activity
        ids = self.soma_ids
        neurons = self.neurons
        if self._activity_ids_contiguous():
            for i in indices.tolist():
                i = int(i)
                neurons[i].activity = float(act[i])
        else:
            for i in indices.tolist():
                i = int(i)
                neurons[int(ids[i])].activity = float(act[i])

    def stimulate_set(self, name: str, value: float = 1.0) -> int:
        """Set activity on all members of a named set. Returns how many neurons."""
        s = self.sets.get(name)
        if s is None:
            return 0
        n = 0
        v = float(np.clip(value, 0.0, 1.0))
        for nid in s.members:
            neuron = self.neurons.get(nid)
            if neuron is None:
                continue
            neuron.activity = v
            n += 1
        self.rebuild_activity_cache()
        return n

    def decay_activity(self, factor: float = 0.9) -> float:
        """Multiply soma activity by factor (vectorized). Returns max remaining.

        Only previously-active slots are touched; sparse writeback to Neuron.
        """
        f = float(np.clip(factor, 0.0, 1.0))
        if not self.neurons:
            self.soma_activity = np.zeros((0,), dtype=np.float32)
            return 0.0
        if len(self.soma_ids) == 0 or len(self.soma_activity) != len(self.soma_ids):
            self.rebuild_activity_cache()
        act = self.soma_activity
        if len(act) == 0:
            return 0.0
        if not act.flags.writeable:
            act = np.array(act, dtype=np.float32, copy=True)
            self.soma_activity = act
        alive = act > 0.0
        if not np.any(alive):
            return 0.0
        idx = np.flatnonzero(alive)
        scaled = act[idx] * f
        scaled = np.where(scaled < 1e-4, 0.0, scaled).astype(np.float32, copy=False)
        act[idx] = scaled
        self._writeback_activity_indices(idx)
        return float(np.max(scaled)) if len(scaled) else 0.0

    def tick_activity(
        self, *, hop_gain: float = 0.85, decay: float = 0.90
    ) -> dict:
        """One network beat: hop along contacts, then fade.

        Returns {edges, lit, touched, max}.
        """
        hop = self.propagate_activity(gain=hop_gain)
        mx = self.decay_activity(decay)
        return {
            "edges": int(hop["edges"]),
            "lit": int(hop["lit"]),
            "touched": int(hop["touched"]),
            "max": float(mx),
            "flash_pre": hop.get("flash_pre"),
            "flash_post": hop.get("flash_post"),
        }

    def contact_edge_gain(self, pre_id: int, post_id: int) -> float:
        """Synaptic multiplier for a grown contact (default 1.0)."""
        g = self.contact_gain.get((int(pre_id), int(post_id)))
        return 1.0 if g is None else float(g)

    def set_contact_gain(
        self,
        pre_id: int,
        post_id: int,
        gain: float,
        *,
        lo: float = 0.05,
        hi: float = 2.5,
    ) -> float:
        """Clamp and store plastic gain; returns stored value."""
        v = float(np.clip(gain, lo, hi))
        key = (int(pre_id), int(post_id))
        if abs(v - 1.0) < 1e-4:
            if key in self.contact_gain:
                self.contact_gain.pop(key, None)
                self._contact_gain_version = int(self._contact_gain_version) + 1
                self._hop_edge_key = None
            return 1.0
        prev = self.contact_gain.get(key)
        self.contact_gain[key] = v
        if prev is None or abs(float(prev) - v) > 1e-8:
            self._contact_gain_version = int(self._contact_gain_version) + 1
            self._hop_edge_key = None
        return v

    def adjust_contact_gain(
        self,
        pre_id: int,
        post_id: int,
        delta: float,
        *,
        lo: float = 0.05,
        hi: float = 2.5,
    ) -> float:
        """Add delta to contact gain (creates override from 1.0 if needed)."""
        cur = self.contact_edge_gain(pre_id, post_id)
        return self.set_contact_gain(pre_id, post_id, cur + float(delta), lo=lo, hi=hi)

    def grown_contact_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """pre→post neuron ids for realized (connected) contacts.

        Union of growth-field CONNECTED tips and materialized axon partners
        (deduped) so Load/adult and live growth share one graph for Run.
        """
        empty = (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
        pairs: set[tuple[int, int]] = set()
        if len(self.growth_owner) and len(self.growth_status) == len(self.growth_owner):
            m = self.growth_status == STATUS_CONNECTED
            if np.any(m):
                for o, g in zip(
                    self.growth_owner[m].tolist(),
                    self.growth_goal[m].tolist(),
                ):
                    pairs.add((int(o), int(g)))
        for neuron in self.neurons.values():
            for ax in neuron.axons:
                if ax.partner_id is None or ax.status != "connected":
                    continue
                if ax.partner_id not in self.neurons:
                    continue
                pairs.add((int(neuron.id), int(ax.partner_id)))
        if not pairs:
            return empty
        pre = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
        post = np.fromiter((p[1] for p in pairs), dtype=np.int32, count=len(pairs))
        return pre, post

    def _recipe_as_neuron_ids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map recipe indices → neuron ids (contiguous atlas: id == index)."""
        empty = (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
        if len(self.recipe_pre) == 0:
            return empty
        ids = self.soma_ids
        if len(ids) == 0:
            self.rebuild_caches()
            ids = self.soma_ids
        if len(ids) == 0:
            return empty
        pre = np.asarray(self.recipe_pre, dtype=np.int32)
        post = np.asarray(self.recipe_post, dtype=np.int32)
        w = np.asarray(self.recipe_weight, dtype=np.float32)
        contiguous = (
            int(ids[0]) == 0
            and int(ids[-1]) == len(ids) - 1
            and len(ids) == len(self.neurons)
        )
        n_atlas = len(self.soma_cls) if len(self.soma_cls) else len(ids)
        # Device neurons sit past the atlas; recipe only indexes atlas somas.
        if contiguous:
            ok = (
                (pre >= 0)
                & (post >= 0)
                & (pre < n_atlas)
                & (post < n_atlas)
                & (pre < len(ids))
                & (post < len(ids))
            )
            return pre[ok], post[ok], w[ok]
        # Sparse: recipe stores indices into soma_* arrays.
        n = len(ids)
        ok = (pre >= 0) & (post >= 0) & (pre < n) & (post < n)
        return ids[pre[ok]].astype(np.int32), ids[post[ok]].astype(np.int32), w[ok]

    def _ensure_hop_recipe(self, *, max_edges: int = HOP_RECIPE_MAX) -> tuple[np.ndarray, np.ndarray]:
        """Cached atlas subset for adult hop: tract classes ∪ top weights."""
        empty = (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
        key = (int(len(self.recipe_pre)), int(max_edges), tuple(self.class_names))
        if (
            self._hop_recipe_key == key
            and len(self._hop_recipe_pre) == len(self._hop_recipe_post)
        ):
            return self._hop_recipe_pre, self._hop_recipe_post
        pre, post, w = self._recipe_as_neuron_ids()
        if len(pre) == 0:
            self._hop_recipe_pre, self._hop_recipe_post = empty
            self._hop_recipe_key = key
            return empty
        max_edges = max(1000, int(max_edges))
        if len(pre) <= max_edges:
            self._hop_recipe_pre = pre.astype(np.int32, copy=False)
            self._hop_recipe_post = post.astype(np.int32, copy=False)
            self._hop_recipe_key = key
            return self._hop_recipe_pre, self._hop_recipe_post

        name_to_i = {n: i for i, n in enumerate(self.class_names)}
        tract = {
            name_to_i[n]
            for n in HOP_TRACT_CLASSES
            if n in name_to_i
        }
        # Prefer ascending/descending endpoints; also sensory→* and *→motor.
        prefer = set()
        for n in ("ascending", "descending", "sensory", "motor", "central"):
            if n in name_to_i:
                prefer.add(name_to_i[n])
        pre_c = self._neuron_cls_array(pre)
        post_c = self._neuron_cls_array(post)
        useful = np.zeros(len(pre), dtype=bool)
        if prefer:
            useful |= np.isin(pre_c, list(prefer)) | np.isin(post_c, list(prefer))
        if tract:
            useful |= np.isin(pre_c, list(tract)) & np.isin(post_c, list(tract))
        # Always keep strongest synapses to fill the budget.
        order_w = np.argsort(-w)
        sel = np.zeros(len(pre), dtype=bool)
        u_idx = np.flatnonzero(useful)
        if len(u_idx) > max_edges:
            # Among useful, keep top weight.
            u_order = u_idx[np.argsort(-w[u_idx])]
            sel[u_order[:max_edges]] = True
        else:
            sel[u_idx] = True
            remain = max_edges - int(sel.sum())
            if remain > 0:
                for j in order_w.tolist():
                    if sel[j]:
                        continue
                    sel[j] = True
                    remain -= 1
                    if remain <= 0:
                        break
        self._hop_recipe_pre = pre[sel].astype(np.int32, copy=False)
        self._hop_recipe_post = post[sel].astype(np.int32, copy=False)
        self._hop_recipe_key = key
        return self._hop_recipe_pre, self._hop_recipe_post

    def _activity_graph_cache_key(self) -> tuple:
        allow = getattr(self, "hop_allow_ids", None)
        n_conn = 0
        if len(self.growth_status):
            n_conn = int(np.sum(self.growth_status == STATUS_CONNECTED))
        return (
            int(len(self.recipe_pre)),
            int(getattr(self, "recipe_armed", 0) or 0),
            int(len(self.growth_owner)),
            n_conn,
            int(len(allow)) if allow is not None else -1,
            int(id(allow)) if allow is not None else 0,
        )

    def invalidate_activity_graph_cache(self) -> None:
        self._activity_graph_key = None
        self._activity_graph_cached = None
        self._elig_rev_key = None
        self._elig_rev_adj = None
        self._hop_edge_key = None

    def _ensure_hop_edge_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached (pre_i, post_i, weight) for scatter-max hop."""
        pre, post = self.activity_graph_edges()
        key = (self._activity_graph_key, int(self._contact_gain_version), len(pre))
        if (
            self._hop_edge_key == key
            and len(self._hop_pre_i) == len(pre)
            and len(self._hop_w) == len(pre)
        ):
            return self._hop_pre_i, self._hop_post_i, self._hop_w
        n = len(pre)
        if n == 0:
            empty_i = np.zeros((0,), dtype=np.int32)
            empty_w = np.zeros((0,), dtype=np.float32)
            self._hop_pre_i = empty_i
            self._hop_post_i = empty_i
            self._hop_w = empty_w
            self._hop_edge_key = key
            return empty_i, empty_i, empty_w
        pre_i = np.asarray(pre, dtype=np.int32)
        post_i = np.asarray(post, dtype=np.int32)
        w = np.ones(n, dtype=np.float32)
        gains = self.contact_gain
        if gains:
            if len(gains) < 4000:
                for (a, b), gw in gains.items():
                    gwf = float(gw)
                    if not np.isfinite(gwf):
                        continue
                    hit = (pre_i == int(a)) & (post_i == int(b))
                    if np.any(hit):
                        w[hit] = np.float32(gwf)
            else:
                keys = (pre_i.astype(np.int64) << 32) | (
                    post_i.astype(np.int64) & np.int64(0xFFFFFFFF)
                )
                order = np.argsort(keys)
                keys_s = keys[order]
                for (a, b), gw in gains.items():
                    gwf = float(gw)
                    if not np.isfinite(gwf):
                        continue
                    k = (np.int64(int(a)) << 32) | (
                        np.int64(int(b)) & np.int64(0xFFFFFFFF)
                    )
                    j = int(np.searchsorted(keys_s, k))
                    if j < len(keys_s) and keys_s[j] == k:
                        w[order[j]] = np.float32(gwf)
        self._hop_pre_i = pre_i
        self._hop_post_i = post_i
        self._hop_w = w
        self._hop_edge_key = key
        return pre_i, post_i, w

    def activity_graph_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Adult hop graph: Fuse-grown contacts ∪ capped atlas recipe.

        Grown bridges attach devices; recipe carries CNS tracts (AN/DN neck).
        If ``hop_allow_ids`` is set, atlas recipe is clipped to that neuron set
        (funnel model); grown Fuse edges always stay.
        """
        key = self._activity_graph_cache_key()
        cached = self._activity_graph_cached
        if cached is not None and self._activity_graph_key == key:
            return cached
        g_pre, g_post = self.grown_contact_edges()
        r_pre, r_post = self._ensure_hop_recipe()
        allow = getattr(self, "hop_allow_ids", None)
        if (
            allow is not None
            and len(allow) > 0
            and len(r_pre) > 0
        ):
            allow_a = np.asarray(allow, dtype=np.int32)
            keep = np.isin(r_pre, allow_a) & np.isin(r_post, allow_a)
            r_pre = r_pre[keep]
            r_post = r_post[keep]
        if len(r_pre) == 0:
            out = (g_pre, g_post)
        elif len(g_pre) == 0:
            out = (r_pre, r_post)
        else:
            out = (
                np.concatenate([g_pre, r_pre]).astype(np.int32, copy=False),
                np.concatenate([g_post, r_post]).astype(np.int32, copy=False),
            )
        self._activity_graph_key = key
        self._activity_graph_cached = out
        # Rev-adj / hop weights depend on the same edges.
        self._elig_rev_key = None
        self._elig_rev_adj = None
        self._hop_edge_key = None
        return out

    def reverse_activity_adj(
        self, allow: set[int] | None = None
    ) -> dict[int, list[int]]:
        """Cached post→[pre] adjacency for eligibility BFS."""
        pre, post = self.activity_graph_edges()
        allow_sig = (
            (len(allow), id(allow)) if allow is not None else (0, 0)
        )
        key = (self._activity_graph_key, allow_sig)
        if self._elig_rev_adj is not None and self._elig_rev_key == key:
            return self._elig_rev_adj
        rev: dict[int, list[int]] = {}
        if allow is None:
            for a, b in zip(pre.tolist(), post.tolist()):
                rev.setdefault(int(b), []).append(int(a))
        else:
            for a, b in zip(pre.tolist(), post.tolist()):
                a, b = int(a), int(b)
                if a not in allow or b not in allow:
                    continue
                rev.setdefault(b, []).append(a)
        self._elig_rev_key = key
        self._elig_rev_adj = rev
        return rev

    def propagate_activity(self, *, gain: float = 0.9) -> dict:
        """One hop of activity along the adult activity graph (pre → post).

        Graph = grown Fuse contacts ∪ atlas hop-recipe. post := max(post,
        gain * contact_gain * pre). Stimulated sources stay lit.
        Returns {edges, lit, touched, flash_pre, flash_post} — hot edge neuron ids.
        """
        empty_f = np.zeros((0,), dtype=np.int32)
        out = {
            "edges": 0,
            "lit": 0,
            "touched": 0,
            "flash_pre": empty_f,
            "flash_post": empty_f,
        }
        if not self.neurons:
            return out
        if len(self.soma_ids) == 0:
            self.rebuild_caches()
        ids = self.soma_ids
        if len(ids) == 0:
            return out
        contiguous = self._activity_ids_contiguous()
        if contiguous:
            pre_i, post_i, edge_w = self._ensure_hop_edge_arrays()
            out["edges"] = int(len(pre_i))
            if len(pre_i) == 0:
                return out
            act = self.soma_activity.astype(np.float32, copy=True)
            if len(act) != len(ids):
                self.rebuild_activity_cache()
                act = self.soma_activity.astype(np.float32, copy=True)
            n_max = len(act)
            # Active-frontier: only edges whose pre is lit.
            alive = act > 1e-4
            if not np.any(alive):
                return out
            valid = (
                (pre_i >= 0)
                & (post_i >= 0)
                & (pre_i < n_max)
                & (post_i < n_max)
                & alive[pre_i]
            )
            if not np.any(valid):
                return out
            pre_i = pre_i[valid]
            post_i = post_i[valid]
            src = act[pre_i] * float(gain) * edge_w[valid]
            hot = src > 1e-4
            if not np.any(hot):
                return out
            pre_i, post_i, src = pre_i[hot], post_i[hot], src[hot]
            incoming = np.zeros_like(act)
            np.maximum.at(incoming, post_i, src)
            before = act.copy()
            act = np.maximum(act, incoming)
            newly = (before <= 1e-4) & (act > 1e-4)
            out["lit"] = int(np.sum(newly))
            out["touched"] = int(np.sum(incoming > 1e-4))
            order = np.argsort(-src)
            keep = order[: min(600, len(order))]
            out["flash_pre"] = pre_i[keep].astype(np.int32, copy=False)
            out["flash_post"] = post_i[keep].astype(np.int32, copy=False)
            self.soma_activity = act
            changed = np.flatnonzero(act > before)
            self._writeback_activity_indices(changed)
            return out

        # Sparse id map fallback.
        pre, post = self.activity_graph_edges()
        out["edges"] = int(len(pre))
        if len(pre) == 0:
            return out
        gains = self.contact_gain
        lut = {int(i): k for k, i in enumerate(ids.tolist())}
        act = self.soma_activity.astype(np.float32, copy=True)
        if len(act) != len(ids):
            self.rebuild_activity_cache()
            act = self.soma_activity.astype(np.float32, copy=True)
        incoming = np.zeros_like(act)
        g = float(gain)
        hot_a: list[int] = []
        hot_b: list[int] = []
        hot_s: list[float] = []
        for a, b in zip(pre.tolist(), post.tolist()):
            ia, ib = lut.get(int(a)), lut.get(int(b))
            if ia is None or ib is None:
                continue
            if act[ia] <= 1e-4:
                continue
            src = act[ia] * g
            if gains:
                gw = gains.get((int(a), int(b)))
                if gw is not None:
                    gwf = float(gw)
                    if np.isfinite(gwf):
                        src *= gwf
            if src <= 1e-4:
                continue
            if src > incoming[ib]:
                incoming[ib] = src
            hot_a.append(int(a))
            hot_b.append(int(b))
            hot_s.append(float(src))
        before = act.copy()
        act = np.maximum(act, incoming)
        newly = (before <= 1e-4) & (act > 1e-4)
        out["lit"] = int(np.sum(newly))
        out["touched"] = int(np.sum(incoming > 1e-4))
        if hot_a:
            order = np.argsort(-np.asarray(hot_s, dtype=np.float32))
            keep = order[: min(600, len(order))]
            out["flash_pre"] = np.asarray([hot_a[i] for i in keep], dtype=np.int32)
            out["flash_post"] = np.asarray([hot_b[i] for i in keep], dtype=np.int32)
        self.soma_activity = act
        changed = np.flatnonzero(act > before)
        self._writeback_activity_indices(changed)
        return out

    def clear_activity(self) -> None:
        for neuron in self.neurons.values():
            neuron.activity = 0.0
        self.rebuild_activity_cache()

    def add_neuron(
        self,
        soma: np.ndarray,
        cls: int,
        *,
        neuron_id: int | None = None,
        set_name: str | None = None,
    ) -> Neuron:
        if neuron_id is None:
            neuron_id = (max(self.neurons.keys()) + 1) if self.neurons else 0
        neuron = Neuron(
            id=int(neuron_id),
            soma=np.asarray(soma, dtype=np.float32).reshape(3),
            cls=int(cls),
        )
        self.neurons[neuron.id] = neuron
        if set_name is not None:
            self.sets.setdefault(set_name, NeuronSet(name=set_name)).members.add(
                neuron.id
            )
        elif 0 <= cls < len(self.class_names):
            name = self.class_names[cls]
            self.sets.setdefault(name, NeuronSet(name=name)).members.add(neuron.id)
        return neuron

    def connect_pair(
        self,
        pre_id: int,
        post_id: int,
        *,
        bend: np.ndarray | None = None,
    ) -> bool:
        """Structural axon pre→post + dendrite stub on post (device / local wire)."""
        pre = self.neurons.get(int(pre_id))
        post = self.neurons.get(int(post_id))
        if pre is None or post is None or int(pre_id) == int(post_id):
            return False
        a = np.asarray(pre.soma, dtype=np.float32).reshape(3)
        b = np.asarray(post.soma, dtype=np.float32).reshape(3)
        mid = 0.5 * (a + b)
        if bend is not None:
            mid = mid + np.asarray(bend, dtype=np.float32).reshape(3)
        else:
            # Slight off-axis bend so the cable reads in 3D.
            d = b - a
            dn = float(np.linalg.norm(d))
            if dn > 1e-8:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                if abs(float(np.dot(d / dn, ref))) > 0.9:
                    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                side = np.cross(d / dn, ref)
                sn = float(np.linalg.norm(side))
                if sn > 1e-8:
                    mid = mid + side / sn * (0.12 * dn)
        ax_pts = np.stack([a, mid.astype(np.float32), b], axis=0).astype(np.float32)
        ax_id = self.alloc_neurite_id()
        pre.axons.append(
            Neurite(
                id=ax_id,
                kind="axon",
                points=ax_pts,
                status="connected",
                partner_id=int(post_id),
            )
        )
        span = self._soma_span()
        den_pts = self._make_dendrite_points(post.soma, b, ax_pts, span)
        den_id = self.alloc_neurite_id()
        post.dendrites.append(
            Neurite(
                id=den_id,
                kind="dendrite",
                points=den_pts,
                status="connected",
                partner_id=int(pre_id),
            )
        )
        return True

    def remove_neurons(self, ids: set[int]) -> None:
        for nid in ids:
            self.neurons.pop(nid, None)
        for s in self.sets.values():
            s.members.difference_update(ids)

    def copy_baseline(self) -> "Fly":
        """Deep-ish copy for Reset (somata + atlas recipe + genome quotas)."""
        out = Fly(class_names=list(self.class_names))
        out._next_neurite_id = self._next_neurite_id
        out.sim_time = 0
        out.phase = "rising"
        out.growth_sources = []
        out.growth_targets = []
        out.atlas_process_xyz = np.asarray(self.atlas_process_xyz, dtype=np.float32).copy()
        out.atlas_process_cls = np.asarray(self.atlas_process_cls, dtype=np.int32).copy()
        out.soma_body_id = np.asarray(self.soma_body_id, dtype=np.int64).copy()
        out.recipe_pre = np.asarray(self.recipe_pre, dtype=np.int32).copy()
        out.recipe_post = np.asarray(self.recipe_post, dtype=np.int32).copy()
        out.recipe_weight = np.asarray(self.recipe_weight, dtype=np.float32).copy()
        out.recipe_armed = 0
        out.contact_gain = {}
        out.genome_quota = np.asarray(self.genome_quota, dtype=np.int32).copy()
        out._genome_edge_bins = {
            k: np.asarray(v, dtype=np.int32).copy()
            for k, v in getattr(self, "_genome_edge_bins", {}).items()
        }
        for nid, neuron in self.neurons.items():
            # Baseline: somata + sets only — no developmental neurites.
            clone = Neuron(
                id=nid,
                soma=neuron.soma.copy(),
                cls=neuron.cls,
                activity=0.0,
                axons=[],
                dendrites=[],
            )
            out.neurons[nid] = clone
        for name, s in self.sets.items():
            out.sets[name] = NeuronSet(name=name, members=set(s.members))
        out.rebuild_caches()
        return out


def load_fly_from_malecns(
    somas_path: Path,
    processes_path: Path | None = None,
    edges_path: Path | None = None,
) -> Fly:
    """Build a developing Fly from MaleCNS as a *target atlas / set recipe*.

    Somata + class sets define who exists and where. Process cloud is an atlas
    ghost for display. Edge recipe (pre→post) drives growth of live neurites.
    """
    data = np.load(somas_path, allow_pickle=True)
    xyz = data["xyz"].astype(np.float32)
    cls = data["cls"].astype(np.int32)
    class_names = [str(x) for x in data["class_names"].tolist()]
    if "body_id" in data.files:
        body = data["body_id"].astype(np.int64)
    else:
        body = np.arange(len(xyz), dtype=np.int64)

    fly = Fly(class_names=class_names)
    for i in range(len(xyz)):
        c = int(cls[i])
        name = class_names[c] if 0 <= c < len(class_names) else f"class_{c}"
        fly.add_neuron(xyz[i], c, neuron_id=i, set_name=name)

    # body[i] matches neuron_id == i at construction time.
    fly.soma_body_id = body
    if processes_path is not None and processes_path.is_file():
        pdata = np.load(processes_path, allow_pickle=True)
        fly.atlas_process_xyz = pdata["xyz"].astype(np.float32)
        fly.atlas_process_cls = pdata["cls"].astype(np.int32)

    if edges_path is None:
        edges_path = Path(somas_path).resolve().parent / "malecns_edges.npz"
    if edges_path is not None and Path(edges_path).is_file():
        ed = np.load(edges_path, allow_pickle=True)
        fly.set_recipe_from_body_edges(ed["pre"], ed["post"], ed["weight"])

    fly.rebuild_caches()
    fly.soma_body_id = np.asarray(
        [body[int(i)] for i in fly.soma_ids.tolist()], dtype=np.int64
    )
    return fly


# --- .fly binary format -------------------------------------------------------
# magic "FLY1" | version u16 | meta_json_len u32 | meta_json utf-8
# n_neurons u32 | ids[i32] | cls[i32] | soma[f32 Nx3]
# [v2+] activity[f32 N]
# n_neurites u32 | neurite_id[i32] | neuron_id[i32] | kind[u8] | n_points[u32]
# points[f32 total_pts x 3]
# [v3+] status[u8] | partner[i32] | seek_t[f32] | seek_limit[f32]  (per neurite)
# n_sets u32 | for each: name_len u16 | name utf-8 | n_members u32 | members[i32]
# [v4+] n_atlas u32 | atlas_xyz[f32 Nx3] | atlas_cls[i32 N]

FLY_MAGIC = b"FLY1"
FLY_VERSION = 4
FLY_VERSION_MIN = 1
KIND_AXON = 0
KIND_DENDRITE = 1


def save_fly(fly: Fly, path: Path, *, extra_meta: dict | None = None) -> None:
    """Write Fly model to a .fly file.

    ``extra_meta`` is merged into the JSON header (e.g. shell device id maps).
    """
    import json
    import struct

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ids = sorted(fly.neurons.keys())
    n = len(ids)
    id_arr = np.asarray(ids, dtype=np.int32)
    cls_arr = np.asarray([fly.neurons[i].cls for i in ids], dtype=np.int32)
    soma_arr = (
        np.stack([fly.neurons[i].soma for i in ids], axis=0).astype(np.float32)
        if n
        else np.zeros((0, 3), dtype=np.float32)
    )
    activity_arr = (
        np.asarray([fly.neurons[i].activity for i in ids], dtype=np.float32)
        if n
        else np.zeros((0,), dtype=np.float32)
    )

    neurite_ids: list[int] = []
    neurite_owners: list[int] = []
    neurite_kinds: list[int] = []
    neurite_npts: list[int] = []
    neurite_status: list[int] = []
    neurite_partner: list[int] = []
    neurite_seek_t: list[float] = []
    neurite_seek_limit: list[float] = []
    point_chunks: list[np.ndarray] = []
    for nid in ids:
        neuron = fly.neurons[nid]
        for neurite in neuron.iter_neurites():
            pts = np.asarray(neurite.points, dtype=np.float32).reshape(-1, 3)
            neurite_ids.append(neurite.id)
            neurite_owners.append(nid)
            neurite_kinds.append(
                KIND_AXON if neurite.kind == "axon" else KIND_DENDRITE
            )
            neurite_npts.append(len(pts))
            neurite_status.append(_STATUS_TO_CODE.get(neurite.status, STATUS_IDLE))
            neurite_partner.append(
                -1 if neurite.partner_id is None else int(neurite.partner_id)
            )
            neurite_seek_t.append(float(neurite.seek_t))
            neurite_seek_limit.append(float(neurite.seek_limit))
            if len(pts):
                point_chunks.append(pts)

    m = len(neurite_ids)
    points = (
        np.concatenate(point_chunks, axis=0).astype(np.float32)
        if point_chunks
        else np.zeros((0, 3), dtype=np.float32)
    )

    meta = {
        "format": "flywire-fly",
        "version": FLY_VERSION,
        "class_names": list(fly.class_names),
        "next_neurite_id": int(fly._next_neurite_id),
        "n_neurons": n,
        "n_neurites": m,
        "sim_time": int(fly.sim_time),
        "phase": str(fly.phase),
        "growth_sources": list(fly.growth_sources),
        "growth_targets": list(fly.growth_targets),
        "contact_gain": [
            [int(a), int(b), float(w)]
            for (a, b), w in sorted(getattr(fly, "contact_gain", {}).items())
        ],
    }
    if extra_meta:
        # Shell / device registries — must survive Load without recreating packs.
        for k, v in extra_meta.items():
            if k not in meta:
                meta[k] = v
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )

    with path.open("wb") as f:
        f.write(FLY_MAGIC)
        f.write(struct.pack("<H", FLY_VERSION))
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack("<I", n))
        if n:
            id_arr.tofile(f)
            cls_arr.tofile(f)
            soma_arr.tofile(f)
            activity_arr.tofile(f)
        f.write(struct.pack("<I", m))
        if m:
            np.asarray(neurite_ids, dtype=np.int32).tofile(f)
            np.asarray(neurite_owners, dtype=np.int32).tofile(f)
            np.asarray(neurite_kinds, dtype=np.uint8).tofile(f)
            np.asarray(neurite_npts, dtype=np.uint32).tofile(f)
            points.tofile(f)
            np.asarray(neurite_status, dtype=np.uint8).tofile(f)
            np.asarray(neurite_partner, dtype=np.int32).tofile(f)
            np.asarray(neurite_seek_t, dtype=np.float32).tofile(f)
            np.asarray(neurite_seek_limit, dtype=np.float32).tofile(f)
        set_items = sorted(fly.sets.items(), key=lambda kv: kv[0])
        f.write(struct.pack("<I", len(set_items)))
        for name, s in set_items:
            nb = name.encode("utf-8")
            members = np.asarray(sorted(s.members), dtype=np.int32)
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(members)))
            if len(members):
                members.tofile(f)
        atlas_xyz = np.asarray(fly.atlas_process_xyz, dtype=np.float32).reshape(-1, 3)
        atlas_cls = np.asarray(fly.atlas_process_cls, dtype=np.int32).reshape(-1)
        n_atlas = int(len(atlas_xyz))
        f.write(struct.pack("<I", n_atlas))
        if n_atlas:
            atlas_xyz.tofile(f)
            atlas_cls[:n_atlas].tofile(f)


def load_fly(path: Path) -> Fly:
    """Read Fly model from a .fly file."""
    import json
    import struct

    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != FLY_MAGIC:
            raise ValueError(f"Not a .fly file (bad magic): {path}")
        (version,) = struct.unpack("<H", f.read(2))
        if version < FLY_VERSION_MIN or version > FLY_VERSION:
            raise ValueError(
                f"Unsupported .fly version {version} "
                f"(supported {FLY_VERSION_MIN}..{FLY_VERSION})"
            )
        (meta_len,) = struct.unpack("<I", f.read(4))
        meta = json.loads(f.read(meta_len).decode("utf-8"))
        class_names = [str(x) for x in meta.get("class_names", [])]
        fly = Fly(class_names=class_names)
        # Preserve shell extras (device id maps + learned socket) for GenomeShell.
        shell_keys = ("shell_devices", "devices", "shell", "motor_socket")
        fly.shell_meta = {
            k: meta[k] for k in shell_keys if k in meta
        }
        fly._next_neurite_id = int(meta.get("next_neurite_id", 0))
        fly.sim_time = int(meta.get("sim_time", 0))
        phase = str(meta.get("phase", "")).strip().lower()
        if phase in ("rising", "adult"):
            fly.phase = phase  # type: ignore[assignment]
        gs = meta.get("growth_sources", meta.get("growth_source"))
        gt = meta.get("growth_targets", meta.get("growth_target"))
        if isinstance(gs, str) and gs:
            fly.growth_sources = [gs]
        elif isinstance(gs, list):
            fly.growth_sources = [str(x) for x in gs]
        if isinstance(gt, str) and gt:
            fly.growth_targets = [gt]
        elif isinstance(gt, list):
            fly.growth_targets = [str(x) for x in gt]
        cg = meta.get("contact_gain")
        if isinstance(cg, list):
            gains: dict[tuple[int, int], float] = {}
            for row in cg:
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    continue
                try:
                    a, b, w = int(row[0]), int(row[1]), float(row[2])
                except (TypeError, ValueError):
                    continue
                if a == b:
                    continue
                gains[(a, b)] = float(np.clip(w, 0.05, 2.5))
            fly.contact_gain = gains

        (n,) = struct.unpack("<I", f.read(4))
        if n:
            id_arr = np.fromfile(f, dtype=np.int32, count=n)
            cls_arr = np.fromfile(f, dtype=np.int32, count=n)
            soma_arr = np.fromfile(f, dtype=np.float32, count=n * 3).reshape(n, 3)
            if version >= 2:
                activity_arr = np.fromfile(f, dtype=np.float32, count=n)
            else:
                activity_arr = np.zeros(n, dtype=np.float32)
            for i in range(n):
                nid = int(id_arr[i])
                fly.neurons[nid] = Neuron(
                    id=nid,
                    soma=soma_arr[i].copy(),
                    cls=int(cls_arr[i]),
                    activity=float(activity_arr[i]),
                )

        (m,) = struct.unpack("<I", f.read(4))
        if m:
            neurite_ids = np.fromfile(f, dtype=np.int32, count=m)
            owners = np.fromfile(f, dtype=np.int32, count=m)
            kinds = np.fromfile(f, dtype=np.uint8, count=m)
            npts = np.fromfile(f, dtype=np.uint32, count=m)
            total_pts = int(npts.sum())
            points = (
                np.fromfile(f, dtype=np.float32, count=total_pts * 3).reshape(
                    total_pts, 3
                )
                if total_pts
                else np.zeros((0, 3), dtype=np.float32)
            )
            if version >= 3:
                status_arr = np.fromfile(f, dtype=np.uint8, count=m)
                partner_arr = np.fromfile(f, dtype=np.int32, count=m)
                seek_t_arr = np.fromfile(f, dtype=np.float32, count=m)
                seek_limit_arr = np.fromfile(f, dtype=np.float32, count=m)
            else:
                status_arr = np.zeros(m, dtype=np.uint8)
                partner_arr = np.full(m, -1, dtype=np.int32)
                seek_t_arr = np.zeros(m, dtype=np.float32)
                seek_limit_arr = np.full(m, DEFAULT_SEEK_LIMIT, dtype=np.float32)
            cursor = 0
            for i in range(m):
                np_ = int(npts[i])
                pts = points[cursor : cursor + np_].copy()
                cursor += np_
                kind: NeuriteKind = (
                    "axon" if int(kinds[i]) == KIND_AXON else "dendrite"
                )
                partner_raw = int(partner_arr[i])
                neurite = Neurite(
                    id=int(neurite_ids[i]),
                    kind=kind,
                    points=pts,
                    status=_CODE_TO_STATUS.get(int(status_arr[i]), "idle"),
                    partner_id=None if partner_raw < 0 else partner_raw,
                    seek_t=float(seek_t_arr[i]),
                    seek_limit=float(seek_limit_arr[i]) or DEFAULT_SEEK_LIMIT,
                )
                owner = int(owners[i])
                neuron = fly.neurons.get(owner)
                if neuron is None:
                    continue
                if kind == "axon":
                    neuron.axons.append(neurite)
                else:
                    neuron.dendrites.append(neurite)
            if m and len(neurite_ids):
                fly._next_neurite_id = max(
                    fly._next_neurite_id, int(neurite_ids.max()) + 1
                )

        (n_sets,) = struct.unpack("<I", f.read(4))
        for _ in range(n_sets):
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode("utf-8")
            (n_mem,) = struct.unpack("<I", f.read(4))
            members = (
                set(int(x) for x in np.fromfile(f, dtype=np.int32, count=n_mem))
                if n_mem
                else set()
            )
            fly.sets[name] = NeuronSet(name=name, members=members)

        if version >= 4:
            (n_atlas,) = struct.unpack("<I", f.read(4))
            if n_atlas:
                fly.atlas_process_xyz = np.fromfile(
                    f, dtype=np.float32, count=n_atlas * 3
                ).reshape(n_atlas, 3)
                fly.atlas_process_cls = np.fromfile(
                    f, dtype=np.int32, count=n_atlas
                )
        else:
            # Legacy: idle process-as-axon clouds → atlas ghost, strip from neurons.
            chunks: list[np.ndarray] = []
            cls_chunks: list[np.ndarray] = []
            for neuron in fly.neurons.values():
                keep: list[Neurite] = []
                for axon in neuron.axons:
                    if axon.status == "idle" and len(axon.points):
                        chunks.append(np.asarray(axon.points, dtype=np.float32))
                        cls_chunks.append(
                            np.full(len(axon.points), neuron.cls, dtype=np.int32)
                        )
                    else:
                        keep.append(axon)
                neuron.axons = keep
            if chunks:
                fly.atlas_process_xyz = np.concatenate(chunks, axis=0)
                fly.atlas_process_cls = np.concatenate(cls_chunks, axis=0)

    fly.rebuild_caches()
    # Legacy files without phase: wired contacts → adult (use), else rising.
    if str(meta.get("phase", "")).strip().lower() not in ("rising", "adult"):
        n_conn = len(fly.grown_contact_edges()[0])
        fly.phase = "adult" if n_conn > 0 else "rising"
    return fly

"""Headless CNS plant: Fly + devices + Blueprint rule.

GenomeShell subclasses this and *is* the public API (methods live on the shell
class). ``CNSRuntime.open`` returns a GenomeShell instance — never setattr's
shell methods onto this base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from blueprint import Blueprint, DeviceBundle, FunnelGate, PlasticityRule
from device import (
    OcelliDevice,
    TurretDevice,
    all_device_ids,
    devices_from_shell,
)
from fly_model import Fly, load_fly
from scope import DeviceRegion, TrainingScope
from wells import WellField, WellGrowParams


class CNSRuntime:
    """CNS plant base. Prefer ``GenomeShell.open`` / ``CNSRuntime.open`` (→ shell)."""

    def __init__(
        self,
        fly: Fly | None = None,
        blueprint: Blueprint | None = None,
    ) -> None:
        bp = blueprint or Blueprint.default()
        self.blueprint = bp
        self.rule: PlasticityRule = bp.plasticity
        self.funnel_gate: FunnelGate = bp.funnel_gate
        self._init_runtime_state(fly if fly is not None else Fly(), bp)

    def _init_runtime_state(self, fly: Fly, bp: Blueprint) -> None:
        """Shared field init for headless open() and GenomeShell.__init__."""
        self.fly = fly
        self.blueprint = bp
        self.rule = bp.plasticity
        self.funnel_gate = bp.funnel_gate
        self._training_scope: TrainingScope = TrainingScope()
        self._well_field: WellField = WellField()
        self.well_grow: WellGrowParams = WellGrowParams.from_funnel_gate(
            bp.funnel_gate
        )

        # Typed devices (source of truth); flat shell fields synced below.
        if not bp.device_list:
            bp.device_list = devices_from_shell(
                bp.devices.to_dict(),
                ocelli_on=bp.ocelli_on,
                turret_on=bp.turret_on,
            )
        self.device_list = list(bp.device_list)
        self._apply_yaw_offset_to_turret(bp.yaw_offset)
        self._sync_devices_to_shell_fields()

        self.ocelli_on = bool(bp.ocelli_on)
        self.turret_on = bool(bp.turret_on)
        self.learn_on = bool(bp.learn_on)
        self.scope_on = bool(bp.scope_on)
        self.hop_clip = bool(bp.hop_clip)

        self.ocelli_I = np.zeros(3, dtype=np.float64)
        self.turret_S = np.zeros(2, dtype=np.float64)
        self.turret_M = np.zeros(2, dtype=np.float64)
        self.turret_aim_err = float("nan")
        self._yaw_lag = float("nan")

        self.show_ocelli_funnel = True
        self.show_turret_funnel = True
        self.show_turret_aim = True
        self._scope_mask: np.ndarray | None = None
        self._funnel_pair_cache = None
        self._funnel_pair_cache_key = None
        self._funnel_probe_layers = None
        self._funnel_probe_radii = None

        self._learn_prev_err = float("nan")
        self.learn_dw_ema = 0.0
        self.learn_n_edges = 0
        self._learn_status_i = 0
        self._learn_flash_n = 0
        self._learn_flash_amp = 0.0
        self._learn_flash_pre = np.zeros(0, dtype=np.int32)
        self._learn_flash_post = np.zeros(0, dtype=np.int32)

        from motor_socket import MotorSocket

        self.motor_socket_on = bool(getattr(self.rule, "motor_socket_on", False))
        self.motor_socket = MotorSocket(
            lr=float(getattr(self.rule, "motor_socket_lr", 0.45) or 0.45)
        )
        self.motor_socket_dw_ema = 0.0

        from sensor_port_nn import SensorPortNN

        self.sensor_port_nn = SensorPortNN(lr=0.02)
        self.sensor_port_on = False
        self.sensor_port_train = False

        # Closed-loop + Trainer (sense→hop→actuate / RPE plasticity).
        from aim_task import AimTask
        from loop import ClosedLoop
        from trainer import Trainer

        self._aim_task = AimTask()
        self._trainer = Trainer(self, task=self._aim_task)
        self._closed_loop = ClosedLoop(
            self, task=self._aim_task, trainer=self._trainer
        )
        # Headless defaults: full hops, learn every frame.
        self.loop_hops = int(self.rule.hops)
        self.learn_every = 1
        self._devices_dirty = True

        self._stereo_plan = None
        self._channel_pairs: list[tuple[int, int]] = []
        self._channel_ids: set[int] = set()
        self._corridor_cache_key = None
        self._corridor_ids: set[int] = set()

        self.lamp_on = True
        self.light_enabled = True
        self.lamp_speed = float(bp.lamp_speed)
        self.lamp_angle = float(bp.lamp_angle)
        self.lamp_intensity = 1.0
        self.sky_fill = 0.28
        self._lamp_xyz = np.zeros(3, dtype=np.float64)

        self.sim_running = True
        self._activity_running = True
        self._fuse_running = False
        self._fly_io_status = ""
        self._last_model = Path(bp.fly_path) if bp.fly_path else None

        # Draw-facing caches (harmless headless; shell overwrites on sync).
        self.class_names = list(getattr(fly, "class_names", ()) or ())
        self.points_xyz = getattr(fly, "soma_xyz", np.zeros((0, 3), np.float32))
        self.points_cls = getattr(fly, "soma_cls", np.zeros((0,), np.int32))
        self.soma_activity = getattr(fly, "soma_activity", np.zeros((0,), np.float32))
        self.n_shown = len(self.points_xyz)
        self.grown_xyz = getattr(fly, "neurite_xyz", np.zeros((0, 3), np.float32))
        self.grown_cls = getattr(fly, "neurite_cls", np.zeros((0,), np.int32))
        self.proc_xyz = getattr(fly, "atlas_process_xyz", np.zeros((0, 3), np.float32))
        self.proc_cls = getattr(fly, "atlas_process_cls", np.zeros((0,), np.int32))
        self._coord_span = (
            float(np.ptp(self.points_xyz, axis=0).max()) if len(self.points_xyz) else 1.0
        )
        self.draw_density = {n: 0.45 for n in self.class_names}
        self.draw_thin_mode = "auto"
        self.focus_soma_indices: set[int] = set()
        self.focus_graph_indices: set[int] = set()
        self.focus_axon_indices: set[int] = set()
        self.focus_dendrite_indices: set[int] = set()
        self.focus_tract_indices: set[int] = set()
        self.focus_basin_indices: set[int] = set()
        self.focus_proc_indices: set[int] = set()
        self.show_layer_neurons = True
        self.show_layer_graph = False
        self.show_layer_axons = True
        self.show_layer_dendrites = True
        self.show_layer_basins = False
        self._basin_cache_key = None
        self._legend_blit_key = None
        self.view_angle_x = 0.0
        self.view_angle_y = 0.0
        self.scale_3d = 20.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.ground_on = True
        self.ground_relief = 0.55
        self._frame_turret_pose = None
        self._frame_ocelli_pose = None
        self._orbit_cache = None
        self._orbit_cache_key = None
        self._wing_anchor_key = None
        self._wing_anchor_cache = None

    def _apply_yaw_offset_to_turret(self, yaw_offset: float) -> None:
        for d in self.blueprint.device_list:
            if isinstance(d, TurretDevice):
                d.yaw = float(d.yaw) + float(yaw_offset)

    def _sync_devices_to_shell_fields(self) -> None:
        """Mirror device_list into legacy flat attrs used by GenomeShell methods."""
        oc = self.blueprint.get_ocelli()
        tu = self.blueprint.get_turret()
        if oc is None:
            oc = OcelliDevice(enabled=self.blueprint.ocelli_on)
        if tu is None:
            tu = TurretDevice(enabled=self.blueprint.turret_on)
        self.ocelli_groups = [list(g) for g in oc.groups] or [[], [], []]
        self.ocelli_terminals = list(oc.terminals)
        self.ocelli_ids = list(oc.flat_ids())
        self.ocelli_on = bool(oc.enabled)
        self.turret_motor_groups = [list(g) for g in tu.motor_groups] or [
            [],
            [],
            [],
            [],
        ]
        self.turret_sensory_groups = [list(g) for g in tu.sensory_groups] or [
            [],
            [],
            [],
            [],
        ]
        self.turret_motor_terminals = list(tu.motor_terminals)
        self.turret_sensory_terminals = list(tu.sensory_terminals)
        self.turret_bus_terminal = tu.bus_terminal
        self.turret_yaw = float(tu.yaw)
        self.turret_pitch = float(tu.pitch)
        self.turret_on = bool(tu.enabled)
        self.device_list = list(self.blueprint.device_list)
        self.blueprint.sync_bundle_from_list()

    def iter_devices(self):
        for d in self.blueprint.device_list:
            yield d

    def all_device_neuron_ids(self) -> set[int]:
        return all_device_ids(list(self.iter_devices()))

    # --- UI hooks (no-ops headless; shell overrides) ---------------------------

    def _set_io_status(self, *args: Any, **kwargs: Any) -> None:
        if args:
            self._fly_io_status = str(args[0])

    def _log_learn_console(self, info: dict) -> None:
        """Shell prints telemetry; headless learn_console has its own logger."""
        del info

    # --- construction ---------------------------------------------------------

    def _boot_headless_extras(self) -> None:
        """Shell overrides: stub display attrs. Plant base needs nothing."""

    @classmethod
    def open(
        cls,
        fly_path: str | Path,
        *,
        blueprint: Blueprint | None = None,
        shell_devices: dict[str, Any] | None = None,
        yaw_offset: float | None = None,
        lamp_speed: float | None = None,
        lamp_angle: float | None = None,
        scope_on: bool | None = None,
        keep_gains: bool = False,
    ) -> CNSRuntime:
        """Load .fly + Blueprint. Shell is the API — never setattr onto base.

        ``CNSRuntime.open`` redirects to ``GenomeShell.open`` so plant methods
        resolve via normal inheritance (no ``bind_api_from_shell`` monkey-patch).
        """
        if cls is CNSRuntime:
            from genome_shell import GenomeShell

            return GenomeShell.open(
                fly_path,
                blueprint=blueprint,
                shell_devices=shell_devices,
                yaw_offset=yaw_offset,
                lamp_speed=lamp_speed,
                lamp_angle=lamp_angle,
                scope_on=scope_on,
                keep_gains=keep_gains,
            )

        path = Path(fly_path)
        fly = load_fly(path)
        meta = getattr(fly, "shell_meta", {}) or {}
        dev = shell_devices or meta.get("shell_devices") or {}
        if blueprint is None:
            bp = Blueprint.from_fly_meta(
                path,
                shell_devices=dev,
                yaw_offset=float(yaw_offset or 0.0),
                lamp_speed=float(lamp_speed if lamp_speed is not None else 0.0),
                lamp_angle=float(lamp_angle if lamp_angle is not None else 0.8),
                scope_on=True if scope_on is None else bool(scope_on),
            )
        else:
            bp = blueprint
            if yaw_offset is not None:
                bp.yaw_offset = float(yaw_offset)
            if lamp_speed is not None:
                bp.lamp_speed = float(lamp_speed)
            if lamp_angle is not None:
                bp.lamp_angle = float(lamp_angle)
            if scope_on is not None:
                bp.scope_on = bool(scope_on)
            if not bp.devices.has_ocelli_turret() and dev:
                bp.devices = DeviceBundle.from_dict(dev)
                bp.device_list = bp.devices.to_device_list(
                    ocelli_on=bp.ocelli_on, turret_on=bp.turret_on
                )
            bp.fly_path = str(path)

        # Real API class (GenomeShell), no pygame window, no setattr bridge.
        rt = object.__new__(cls)
        rt._boot_headless_extras()
        CNSRuntime._init_runtime_state(rt, fly, bp)
        rt._wire_turret_device()
        if not keep_gains:
            rt.fly.contact_gain = {}
        return rt

    def instant_fuse(self) -> int:
        """Fuse every enabled device via its channels / attach-policy."""
        self._sync_devices_to_shell_fields()
        self._clear_device_cns_fuse()
        self._ensure_atlas_recipe()  # the plan reads the CNS tract graph
        self.refresh_stereo_plan()
        n = 0
        for d in self.iter_devices():
            if not getattr(d, "enabled", True):
                continue
            n += int(self._fuse_one_device(d))
        self.fly.enter_adult(abandon_open=True)
        self.fly.rebuild_caches()
        self._ensure_atlas_recipe()
        self._devices_dirty = True
        return n

    def _fuse_one_device(self, d: Any) -> int:
        """Connect one device using its channel attach policies."""
        kind = getattr(d, "kind", "")
        n = 0
        if kind == "ocelli":
            goals = self._ocelli_pick_targets()
            terms = list(getattr(d, "terminals", None) or self.ocelli_terminals)
            # Policy lives on channel — pick_targets still uses FUSE aliases
            # that match POLICY_OCELLI; wiring direction = to_cns.
            if goals and len(terms) >= len(goals):
                for gi, tid in enumerate(terms[: len(goals)]):
                    for g in goals[gi]:
                        if self.fly.connect_pair(int(tid), int(g)):
                            n += 1
            return n
        if kind == "turret":
            pairs = self._turret_pick_targets()
            m_terms = list(
                getattr(d, "motor_terminals", None) or self.turret_motor_terminals
            )
            s_terms = list(
                getattr(d, "sensory_terminals", None) or self.turret_sensory_terminals
            )
            if pairs and len(m_terms) >= len(pairs) and len(s_terms) >= len(pairs):
                for si, (src_m, goal_s) in enumerate(pairs):
                    for g in src_m:
                        if self.fly.connect_pair(int(g), int(m_terms[si])):
                            n += 1
                    for g in goal_s:
                        if self.fly.connect_pair(int(s_terms[si]), int(g)):
                            n += 1
            return n
        if kind == "eye":
            from compound_eyes import fuse_eye_device

            return int(fuse_eye_device(self, d))
        if kind == "jo":
            from johnston_organ import fuse_jo_device

            return int(fuse_jo_device(self, d))
        # Future devices: iterate channels() + AttachPolicy.
        return 0

    def refresh_stereo_plan(self) -> Any:
        """Re-pick the eye→wing partners so the two sides do not share a path."""
        from stereo_channels import plan_stereo_channels

        plan = None
        if bool(getattr(self.blueprint, "stereo_channels", True)):
            try:
                plan = plan_stereo_channels(self)
            except Exception as exc:  # anatomy too thin → fall back to nearest
                print(f"[stereo] plan failed: {exc}", flush=True)
                plan = None
        self._stereo_plan = plan
        if plan is not None:
            print(f"[stereo] {plan.readout}", flush=True)
        return plan

    def stereo_plan(self) -> Any:
        return getattr(self, "_stereo_plan", None)


    def sensor_terminal_ids(self) -> set[int]:
        """All sensory clamps across enabled devices."""
        ids: set[int] = set()
        for d in self.iter_devices():
            if not getattr(d, "enabled", True):
                continue
            for ch in d.channels():
                if ch.kind == "sensor":
                    ids.update(ch.terminal_ids())
        if not ids:
            ids.update(int(x) for x in (getattr(self, "ocelli_terminals", []) or []))
            ids.update(
                int(x) for x in (getattr(self, "turret_sensory_terminals", []) or [])
            )
        return ids

    def sensor_terminal_groups(self) -> dict[str, set[int]]:
        """Sensory clamps split per device — one BFS source set per modality.

        The scope corridor is the union of per-group beds; merging eye, JO,
        ocelli and turret proprio into one source set would measure every node
        from the nearest modality and clip the long optic→DN path.
        """
        out: dict[str, set[int]] = {}
        for d in self.iter_devices():
            if not getattr(d, "enabled", True):
                continue
            ids: set[int] = set()
            for ch in d.channels():
                if ch.kind == "sensor":
                    ids.update(int(x) for x in ch.terminal_ids())
            if not ids:
                continue
            name = str(getattr(d, "name", None) or getattr(d, "kind", "device"))
            out.setdefault(name, set()).update(ids)
        if not out:
            oc = {int(x) for x in (getattr(self, "ocelli_terminals", []) or [])}
            tu = {
                int(x) for x in (getattr(self, "turret_sensory_terminals", []) or [])
            }
            if oc:
                out["ocelli"] = oc
            if tu:
                out["turret"] = tu
        return out

    def motor_terminal_ids(self) -> set[int]:
        ids: set[int] = set()
        for d in self.iter_devices():
            if not getattr(d, "enabled", True):
                continue
            for ch in d.channels():
                if ch.kind == "motor":
                    ids.update(ch.terminal_ids())
        if not ids:
            ids.update(
                int(x) for x in (getattr(self, "turret_motor_terminals", []) or [])
            )
        try:
            targets = self._turret_motor_neuron_targets()
            if targets:
                ids.update(int(k) for k in targets.keys())
        except Exception:
            pass
        return ids

    def build_well_regions(self) -> list[DeviceRegion]:
        """Geometric wells — render/growth only. Lazy; does not write TrainingScope."""
        field = self.ensure_well_field()
        return field.regions()

    def ensure_well_field(self) -> WellField:
        """Build/cached WellField for draw & tip gravity (not training corridor)."""
        wf = getattr(self, "_well_field", None)
        if wf is None:
            wf = WellField()
            self._well_field = wf
        # Prefer shell geometry helpers when present (GenomeShell).
        pair_fn = getattr(self, "_device_funnel_pair", None)
        if callable(pair_fn):
            # Shell path populates WellField via _device_funnel_pair.
            pair_fn()
            return self._well_field
        return wf

    def build_training_scope(
        self, *, max_hops: int | None = None, with_wells: bool = False
    ) -> TrainingScope:
        """Graph scope: near-shortest sensor→motor paths (+ blueprint margin)."""
        self._ensure_atlas_recipe()
        pre, post = self.fly.activity_graph_edges()
        regions: list[DeviceRegion] | None = None
        if with_wells:
            try:
                regions = self.build_well_regions()
            except Exception:
                regions = None
        margin = int(getattr(self.blueprint, "scope_path_margin", 2) or 0)
        if max_hops is None:
            max_hops = int(getattr(self.blueprint, "scope_max_hops", 24) or 24)
        scope = TrainingScope.from_graph(
            pre,
            post,
            sensors=self.sensor_terminal_ids(),
            motors=self.motor_terminal_ids(),
            max_hops=int(max_hops),
            path_margin=margin,
            regions=regions,
            sensor_groups=self.sensor_terminal_groups(),
        )
        self._training_scope = scope
        return scope

    def apply_training_scope(self) -> bool:
        """Activate graph TrainingScope; optionally clip hop_allow."""
        scope = self.build_training_scope(with_wells=False)
        met = scope.metrics()
        if not scope.ok:
            self.scope_on = False
            self.blueprint.scope_on = False
            self._scope_mask = None
            self.fly.hop_allow_ids = np.zeros((0,), dtype=np.int32)
            self.fly._hop_recipe_key = None
            self._set_io_status(f"Scope refused · {met.readout()}")
            print(f"[scope] REFUSED {met.readout()}", flush=True)
            return False
        self.scope_on = True
        self.blueprint.scope_on = True
        ids = np.asarray(self.fly.soma_ids, dtype=np.int64)
        if self.hop_clip and scope.corridor:
            allow = np.fromiter(
                scope.corridor, dtype=np.int64, count=len(scope.corridor)
            )
            self.fly.hop_allow_ids = allow.astype(np.int32, copy=False)
            self._scope_mask = np.isin(ids, allow)
        else:
            self.fly.hop_allow_ids = np.zeros((0,), dtype=np.int32)
            cor = scope.corridor
            self._scope_mask = (
                np.isin(
                    ids,
                    np.fromiter(cor, dtype=np.int64, count=len(cor)),
                )
                if cor
                else None
            )
        self.fly._hop_recipe_key = None
        self._corridor_ids = set(scope.corridor)
        self._corridor_cache_key = None
        self._refresh_channel_bed()
        print(
            f"[scope] ON {scope.readout()} hop_clip={bool(self.hop_clip)}",
            flush=True,
        )
        self._set_io_status(f"Scope · {scope.readout()}")
        return True

    def clear_training_scope(self) -> None:
        self.scope_on = False
        self.blueprint.scope_on = False
        self._scope_mask = None
        self._training_scope = TrainingScope()
        self.fly.hop_allow_ids = np.zeros((0,), dtype=np.int32)
        self.fly._hop_recipe_key = None
        self._corridor_ids = set()
        self._set_io_status("Scope off · full hop graph")
        print("[scope] OFF", flush=True)

    def sync_training_scope(self) -> bool:
        """Apply or clear TrainingScope from ``blueprint.scope_on``."""
        if not self.blueprint.scope_on:
            self.clear_training_scope()
            return False
        return bool(self.apply_training_scope())

    def _apply_probe_light(self, device: Any) -> float | None:
        """Move the lamp onto the azimuth where ``device`` faces it.

        Returns the previous ``lamp_angle`` when the lamp moved, else ``None``.
        Devices that do not track a point light (JO, turret proprio) have no
        ``probe_light_angle`` and keep the world lighting as-is.
        """
        fn = getattr(device, "probe_light_angle", None)
        if not callable(fn):
            return None
        az = fn(self)
        if az is None:
            return None
        prev = float(getattr(self, "lamp_angle", 0.0))
        self.lamp_angle = float(az)
        self._update_lamp_xyz()
        return prev

    def probe_device_motor_reach(
        self,
        device: Any,
        *,
        hops: int = 24,
        hop_gain: float = 0.9,
        decay: float = 0.97,
        threshold: float = 0.02,
        reference_light: bool = True,
    ) -> dict[str, Any]:
        """Frozen-weight wave: one device's own sense() → turret motor clamps.

        Reports the beat each motor first lit, so a FAIL says how many hops the
        path would need instead of hiding behind a shortcut to the motors.

        With ``reference_light`` the lamp is first swung onto its own orbit
        azimuth in front of the probed device. Reach is a property of the fused
        corridor, and a compound eye on the far side of the body only sees sky
        wash (~0.06 vs ~0.9 lit) — probing every device at one static lamp pose
        reads that shadow as a dead path. ``drive`` / ``drive_ref`` keep both
        numbers visible so an eye that is genuinely dark right now still says so.
        """
        name = str(getattr(device, "name", None) or getattr(device, "kind", "device"))
        motors = sorted(int(m) for m in self.motor_terminal_ids())
        out: dict[str, Any] = {
            "device": name,
            "seed": 0,
            "motors": len(motors),
            "hit": 0,
            "best": 0.0,
            "mean_hit": 0.0,
            "first_hop": None,
            "all_hop": None,
            "hops": int(hops),
            "drive": 0.0,
            "drive_ref": 0.0,
            "lit_az": None,
            "ok": False,
        }
        if not motors:
            return out
        saved_gains = dict(getattr(self.fly, "contact_gain", {}) or {})
        saved_enabled = bool(getattr(device, "enabled", True))
        saved_az: float | None = None
        self.fly.contact_gain = {}
        self.fly.clear_activity()
        self._update_lamp_xyz()
        try:
            device.enabled = True
            world_levels = device.sense(self) or {}
            out["drive"] = float(max(world_levels.values())) if world_levels else 0.0
            if reference_light:
                saved_az = self._apply_probe_light(device)
                if saved_az is not None:
                    out["lit_az"] = float(self.lamp_angle)
            levels = device.sense(self) or {}
            out["drive_ref"] = float(max(levels.values())) if levels else 0.0
            out["seed"] = int(self.fly.stimulate_neurons(levels)) if levels else 0
            if not levels:
                return out
            peak = {m: 0.0 for m in motors}
            lit_at: dict[int, int] = {}
            for beat in range(1, max(1, int(hops)) + 1):
                self.fly.tick_activity(hop_gain=float(hop_gain), decay=float(decay))
                self.fly.rebuild_activity_cache()
                for m in motors:
                    neu = self.fly.neurons.get(m)
                    if neu is None:
                        continue
                    a = float(neu.activity)
                    if a > peak[m]:
                        peak[m] = a
                    if a > float(threshold) and m not in lit_at:
                        lit_at[m] = beat
        finally:
            self.fly.contact_gain = saved_gains
            device.enabled = saved_enabled
            if saved_az is not None:
                self.lamp_angle = float(saved_az)
                self._update_lamp_xyz()
                device.sense(self)  # re-read under world light: panel bars must not keep probe values
        hit = {m: a for m, a in peak.items() if a > float(threshold)}
        out.update(
            hit=len(hit),
            best=float(max(peak.values()) if peak else 0.0),
            mean_hit=float(sum(hit.values()) / len(hit)) if hit else 0.0,
            first_hop=(int(min(lit_at.values())) if lit_at else None),
            all_hop=(int(max(lit_at.values())) if len(lit_at) == len(motors) else None),
            ok=len(hit) >= max(1, len(motors) // 4),
        )
        return out

    def probe_sensor_reach_by_device(
        self, *, hops: int = 24, threshold: float = 0.02, reference_light: bool = True
    ) -> list[dict[str, Any]]:
        """Per-device reach — eye→motor is reported apart from ocelli→motor."""
        rows: list[dict[str, Any]] = []
        for d in self.iter_devices():
            if not any(ch.kind == "sensor" for ch in d.channels()):
                continue
            row = self.probe_device_motor_reach(
                d,
                hops=hops,
                threshold=threshold,
                reference_light=reference_light,
            )
            row["enabled"] = bool(getattr(d, "enabled", True))
            rows.append(row)
        return rows

    def count_dw_corridor_edges(self) -> int:
        corridor = self._corridor_neuron_ids()
        gains = getattr(self.fly, "contact_gain", {}) or {}
        return sum(1 for (a, b) in gains if int(a) in corridor or int(b) in corridor)

    def prepare_activity(self) -> None:
        self.fly.clear_activity()
        loop = getattr(self, "_closed_loop", None)
        if loop is not None:
            loop.sense_all()
        else:
            self._seed_ocelli_activity()
            self._apply_turret_feedback()
        self._update_lamp_xyz()
        self._update_turret_aim_metric()

    def tick(self, dt: float) -> dict:
        loop = getattr(self, "_closed_loop", None)
        if loop is not None:
            return loop.step(float(dt))
        return self._tick_adult_closed_loop(float(dt))

    def trainer(self):
        """Public Trainer for learn_console / probes."""
        tr = getattr(self, "_trainer", None)
        if tr is None:
            from aim_task import AimTask
            from trainer import Trainer

            task = getattr(self, "_aim_task", None) or AimTask()
            self._aim_task = task
            tr = Trainer(self, task=task)
            self._trainer = tr
        return tr

    def device_snapshot(self) -> dict[str, Any]:
        self._sync_shell_fields_to_devices()
        return self._device_snapshot()

    def _sync_shell_fields_to_devices(self) -> None:
        oc = self.blueprint.get_ocelli()
        tu = self.blueprint.get_turret()
        if oc is not None:
            oc.groups = [list(g) for g in self.ocelli_groups]
            oc.terminals = list(self.ocelli_terminals)
            oc.enabled = bool(self.ocelli_on)
        if tu is not None:
            tu.motor_groups = [list(g) for g in self.turret_motor_groups]
            tu.sensory_groups = [list(g) for g in self.turret_sensory_groups]
            tu.motor_terminals = list(self.turret_motor_terminals)
            tu.sensory_terminals = list(self.turret_sensory_terminals)
            tu.bus_terminal = self.turret_bus_terminal
            tu.yaw = float(self.turret_yaw)
            tu.pitch = float(self.turret_pitch)
            tu.enabled = bool(self.turret_on)
        self.blueprint.sync_bundle_from_list()

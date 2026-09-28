"""CNS Brain schematic — iterative block diagram of hop / carve / motor path.

Honest: MaleCNS graph + contact_gain plasticity, not an MLP. When AI Brain
owns the effector this view reports bypass (CNS Brain unused for slew).
Click a block for detail; stage chips walk the pipeline left→right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_TITLE = (200, 220, 235)
_BODY = (150, 165, 185)
_MUTED = (138, 150, 168)
_EDGE = (60, 95, 120)
_EDGE_HOT = (120, 190, 210)
_PANEL_BG = (16, 22, 30, 235)
_PANEL_EDGE = (100, 160, 190)
_BG_GRID = (20, 26, 32)

_STAGES = (
    ("sense", "1 Sense"),
    ("scope", "2 Scope"),
    ("hop", "3 Hop"),
    ("plastic", "4 Carve"),
    ("motor", "5 Motor"),
    ("plant", "6 Plant"),
)


@dataclass(frozen=True)
class BrainBlock:
    id: str
    title: str
    stage: str
    summary: tuple[str, ...]
    detail: tuple[str, ...]
    accent: tuple[int, int, int]
    live: bool = False
    bars: tuple[float, ...] = ()


@dataclass
class BrainSchematic:
    blocks: list[BrainBlock] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    mode: str = "off"
    stage: str = "sense"
    subtitle: str = ""

    @classmethod
    def from_host(cls, host: Any, *, stage: str | None = None) -> BrainSchematic:
        port_on = bool(getattr(host, "sensor_port_on", False))
        learn_on = bool(getattr(host, "learn_on", False))
        sock_on = bool(getattr(host, "motor_socket_on", False))
        sock = getattr(host, "motor_socket", None)
        sock_frozen = bool(sock is not None and getattr(sock, "frozen", False))
        turret_on = bool(getattr(host, "turret_on", False))
        scope_on = bool(getattr(host, "scope_on", False))
        hop_clip = bool(getattr(host, "hop_clip", False))

        if port_on:
            mode = "bypass"
        elif sock_on and sock_frozen:
            mode = "socket-frozen"
        elif sock_on:
            mode = "socket"
        elif learn_on and turret_on:
            mode = "train"
        elif turret_on:
            mode = "run"
        else:
            mode = "off"

        # Sensors
        I = np.asarray(getattr(host, "ocelli_I", np.zeros(3)), dtype=float).ravel()
        if len(I) < 3:
            I = np.pad(I, (0, 3 - len(I)))
        el = np.asarray(getattr(host, "eye_left_I", np.zeros(6)), dtype=float).ravel()[:6]
        er = np.asarray(getattr(host, "eye_right_I", np.zeros(6)), dtype=float).ravel()[:6]
        if len(el) < 6:
            el = np.pad(el, (0, 6 - len(el)))
        if len(er) < 6:
            er = np.pad(er, (0, 6 - len(er)))

        # Scope / corridor
        scope = getattr(host, "_training_scope", None)
        path_n = int(len(getattr(scope, "path_neurons", ()) or ())) if scope else 0
        scope_ok = bool(scope is not None and getattr(scope, "ok", False))
        hop_allow = getattr(getattr(host, "fly", None), "hop_allow_ids", None)
        allow_n = int(len(hop_allow)) if hop_allow is not None else 0

        # Hop / activity
        fly = getattr(host, "fly", None)
        act = getattr(host, "soma_activity", None)
        if act is None and fly is not None:
            act = getattr(fly, "soma_activity", None)
        act_arr = np.asarray(act, dtype=float).ravel() if act is not None else np.zeros(0)
        lit = int(np.count_nonzero(act_arr > 0.05)) if act_arr.size else 0
        amax = float(np.max(act_arr)) if act_arr.size else 0.0
        hops_ui = int(getattr(host, "loop_hops", 1) or 1)
        eng = float(getattr(host, "_hop_engagement", getattr(host, "hop_engagement", 0.0)) or 0.0)

        # Plasticity / contact_gain
        gains = getattr(fly, "contact_gain", None) or {} if fly is not None else {}
        n_gains = int(len(gains))
        dw = float(getattr(host, "learn_dw_ema", 0.0) or 0.0)
        rpe = 0.0
        tr = getattr(host, "_trainer", None)
        if tr is not None:
            rpe = float(getattr(tr, "last_rpe", 0.0) or 0.0)
            if hasattr(tr, "report"):
                try:
                    rep = tr.report({})
                    rpe = float(rep.get("rpe", rpe) or rpe)
                    dw = float(rep.get("dw_ema", dw) or dw)
                except Exception:
                    pass
        dw_edges = 0
        if hasattr(host, "count_dw_corridor_edges"):
            try:
                dw_edges = int(host.count_dw_corridor_edges())
            except Exception:
                dw_edges = 0

        # Motor decode
        levels = [0.0, 0.0, 0.0, 0.0]
        if hasattr(host, "_turret_site_motor_level"):
            try:
                levels = [float(host._turret_site_motor_level(i)) for i in range(4)]
            except Exception:
                levels = [0.0, 0.0, 0.0, 0.0]
        yaw_cmd = pitch_cmd = 0.0
        if hasattr(host, "_turret_motor_commands") and not port_on:
            try:
                yaw_cmd, pitch_cmd = host._turret_motor_commands()
            except Exception:
                pass
        elif port_on:
            port = getattr(host, "sensor_port_nn", None)
            if port is not None:
                cmd = np.asarray(getattr(port, "last_cmd", [0, 0]), dtype=float).ravel()
                if len(cmd) >= 2:
                    yaw_cmd, pitch_cmd = float(cmd[0]), float(cmd[1])

        err = float(getattr(host, "turret_aim_err", float("nan")))
        eps = float(math.degrees(err)) if math.isfinite(err) else float("nan")
        track = float(getattr(host, "_track_score", float("nan")))
        yaw = float(getattr(host, "turret_yaw", 0.0) or 0.0)
        pitch = float(getattr(host, "turret_pitch", 0.0) or 0.0)

        rule = getattr(host, "rule", None)
        teacher_mix = float(getattr(rule, "teacher_mix", 0.0) or 0.0) if rule else 0.0
        carve_lr = float(getattr(rule, "carve_lr", 0.0) or 0.0) if rule else 0.0
        elig = int(getattr(rule, "elig_depth", 0) or 0) if rule else 0

        tissue_live = (not port_on) and turret_on
        learn_live = tissue_live and learn_on and not sock_on

        blocks = [
            BrainBlock(
                id="world",
                title="World / lamp",
                stage="sense",
                summary=(
                    f"spin {float(getattr(host, 'lamp_speed', 0.0) or 0.0):.2f}",
                    f"lamp {'on' if getattr(host, 'lamp_on', False) else 'off'}",
                ),
                detail=(
                    f"lamp_angle = {float(getattr(host, 'lamp_angle', 0.0) or 0.0):.3f}",
                    "Light → ocelli / eyes → CNS Brain inject (not Port MLP).",
                ),
                accent=(200, 180, 110),
                live=bool(getattr(host, "lamp_on", False)),
            ),
            BrainBlock(
                id="ocelli",
                title="Ocelli · 3",
                stage="sense",
                summary=(
                    f"max {float(np.max(I)):.2f}",
                    f"on={'yes' if getattr(host, 'ocelli_on', False) else 'no'}",
                ),
                detail=tuple(f"I[{i}] = {float(I[i]):.3f}" for i in range(3))
                + ("Fuse → optic / visual_projection seeds.",),
                accent=(120, 210, 160),
                live=bool(getattr(host, "ocelli_on", False)) and float(np.max(I)) > 0.05,
                bars=tuple(float(np.clip(v, 0.0, 1.0)) for v in I),
            ),
            BrainBlock(
                id="eyes",
                title="Compound eyes",
                stage="sense",
                summary=(
                    f"L {float(np.max(el)):.2f}",
                    f"R {float(np.max(er)):.2f}",
                ),
                detail=(
                    f"eye_L max = {float(np.max(el)):.3f}",
                    f"eye_R max = {float(np.max(er)):.3f}",
                    "Optional afferents into atlas; Port can read same sensors.",
                ),
                accent=(100, 180, 190),
                live=(
                    bool(getattr(host, "eye_left_on", False) or getattr(host, "eye_right_on", False))
                    and max(float(np.max(el)), float(np.max(er))) > 0.05
                ),
                bars=tuple(
                    float(np.clip(v, 0.0, 1.0))
                    for v in list(el[::2]) + list(er[::2])
                ),
            ),
            BrainBlock(
                id="scope",
                title="Training scope",
                stage="scope",
                summary=(
                    f"{'ON' if scope_on else 'off'} · path {path_n}",
                    f"clip={'yes' if hop_clip else 'no'} · allow {allow_n}",
                ),
                detail=(
                    f"scope_on = {scope_on}",
                    f"hop_clip = {hop_clip}",
                    f"path_neurons = {path_n}",
                    f"hop_allow_ids = {allow_n}",
                    f"scope.ok = {scope_ok}",
                    "Corridor = sensor→motor paths · optional hop lock.",
                ),
                accent=(160, 210, 140) if scope_on else (110, 130, 120),
                live=scope_on and path_n > 0,
            ),
            BrainBlock(
                id="hop",
                title="Hop activity",
                stage="hop",
                summary=(
                    f"hops/frame {hops_ui}",
                    f"lit {lit} · max {amax:.2f}",
                ),
                detail=(
                    f"loop_hops = {hops_ui}",
                    f"lit somata (>0.05) = {lit}",
                    f"activity max = {amax:.4f}",
                    f"engagement = {eng:.3f}",
                    "tick_activity on atlas edges — CNS Brain graph, not MLP.",
                    "Port ON → hop quenched (bypass).",
                ),
                accent=(140, 190, 220) if tissue_live else (90, 100, 110),
                live=tissue_live and lit > 0,
                bars=tuple(
                    float(np.clip(v, 0.0, 1.0))
                    for v in (act_arr[:: max(1, act_arr.size // 12)][:12] if act_arr.size else [])
                ),
            ),
            BrainBlock(
                id="plastic",
                title="Carve · contact_gain",
                stage="plastic",
                summary=(
                    f"Δw {dw:.3f} · RPE {rpe:+.3f}",
                    f"gains {n_gains} · edges {dw_edges}",
                ),
                detail=(
                    f"learn_on = {learn_on}",
                    f"dw_ema = {dw:.4f}",
                    f"RPE = {rpe:+.4f}",
                    f"contact_gain entries = {n_gains}",
                    f"corridor Δw edges = {dw_edges}",
                    f"carve_lr = {carve_lr:.4f} · elig_depth = {elig}",
                    f"teacher_mix = {teacher_mix:.3f} (clamps only)",
                    "Plasticity on CNS Brain edges — not Port W.",
                    "Port ON / socket → carve idle.",
                ),
                accent=(160, 210, 255) if learn_live else (100, 110, 130),
                live=learn_live and (abs(dw) > 1e-4 or abs(rpe) > 1e-4),
            ),
            BrainBlock(
                id="motor",
                title="Motor decode",
                stage="motor",
                summary=(
                    f"yaw {yaw_cmd:+.2f}",
                    f"pitch {pitch_cmd:+.2f}",
                ),
                detail=(
                    f"sites Llo/Lhi/Rlo/Rhi = "
                    f"{levels[0]:.2f}/{levels[1]:.2f}/{levels[2]:.2f}/{levels[3]:.2f}",
                    f"yaw_cmd = {yaw_cmd:+.4f}",
                    f"pitch_cmd = {pitch_cmd:+.4f}",
                    "Hop motor packs → L−R yaw, lo−hi pitch.",
                    (
                        "BYPASS: AI Brain owns slew — CNS Brain decode unused."
                        if port_on
                        else (
                            "Socket W readout (legacy bypass)."
                            if sock_on
                            else "CNS hop decode drives turret."
                        )
                    ),
                ),
                accent=(220, 160, 120) if tissue_live and not port_on else (100, 100, 110),
                live=tissue_live and not port_on and (abs(yaw_cmd) + abs(pitch_cmd)) > 0.04,
                bars=tuple(float(np.clip(v, 0.0, 1.0)) for v in levels),
            ),
            BrainBlock(
                id="plant",
                title="Turret plant",
                stage="plant",
                summary=(
                    f"ε {eps:.0f}°" if math.isfinite(eps) else "ε —",
                    f"Track {track:.0%}" if math.isfinite(track) else "Track —",
                ),
                detail=(
                    f"yaw = {yaw:.3f} · pitch = {pitch:.3f}",
                    f"aim ε = {eps:.2f}°" if math.isfinite(eps) else "aim ε = —",
                    f"Track = {track:.1%}" if math.isfinite(track) else "Track = —",
                    f"turret_on = {turret_on}",
                    "Closed-loop plant under Spin — same meters as AI Brain plant.",
                ),
                accent=(150, 200, 140),
                live=turret_on and math.isfinite(track) and track > 0.2,
            ),
        ]

        if sock_on:
            blocks.append(
                BrainBlock(
                    id="socket",
                    title="Motor socket W",
                    stage="plastic",
                    summary=(
                        "FROZEN" if sock_frozen else "train",
                        f"Δw {float(getattr(host, 'motor_socket_dw_ema', 0.0) or 0.0):.3f}",
                    ),
                    detail=(
                        "Legacy port map on site levels — bypasses hop carve.",
                        f"frozen = {sock_frozen}",
                        "Prefer CNS carve or AI Brain for new work.",
                    ),
                    accent=(180, 220, 150),
                    live=True,
                )
            )

        edges = [
            ("world", "ocelli", "light"),
            ("world", "eyes", "light"),
            ("ocelli", "scope", "afferent"),
            ("eyes", "scope", "afferent"),
            ("scope", "hop", "corridor"),
            ("hop", "plastic", "elig"),
            ("plastic", "hop", "Δg"),
            ("hop", "motor", "packs"),
            ("motor", "plant", "cmd"),
        ]
        if sock_on:
            edges.append(("hop", "socket", "sites"))
            edges.append(("socket", "motor", "W"))

        st = stage if stage in {s for s, _ in _STAGES} else "sense"
        if mode == "bypass":
            sub = (
                "CNS Brain · BYPASS · AI Brain owns drive · hop/carve unused for slew"
            )
        else:
            sub = (
                f"CNS Brain · {mode} · hop + contact_gain · "
                "click block · stage chips iterate"
            )
        return cls(blocks=blocks, edges=edges, mode=mode, stage=st, subtitle=sub)


def layout_brain(schematic: BrainSchematic, area: Any) -> dict[str, Any]:
    """Place blocks in stage columns + detail strip."""
    import pygame

    pad = 14
    top = area.y + 36
    bottom = area.bottom - 52
    left = area.x + pad
    right = area.right - pad
    stage_ids = [s for s, _ in _STAGES]
    cols = len(stage_ids)
    cw = max(96, (right - left - (cols - 1) * 10) // cols)
    gap = 10
    rects: dict[str, Any] = {}
    by_stage: dict[str, list[BrainBlock]] = {s: [] for s in stage_ids}
    for b in schematic.blocks:
        by_stage.setdefault(b.stage, []).append(b)

    for ci, sid in enumerate(stage_ids):
        xs = left + ci * (cw + gap)
        items = by_stage.get(sid, [])
        if not items:
            continue
        bh = max(54, min(88, (bottom - top - (len(items) - 1) * 8) // max(1, len(items))))
        for bi, b in enumerate(items):
            ys = top + bi * (bh + 8)
            rects[b.id] = pygame.Rect(xs, ys, cw, bh)

    chip_y = area.bottom - 40
    chip_w = max(70, (right - left - (cols - 1) * 6) // cols)
    chips = []
    for ci, (sid, label) in enumerate(_STAGES):
        chips.append(
            (
                sid,
                label,
                pygame.Rect(left + ci * (chip_w + 6), chip_y, chip_w, 26),
            )
        )
    rects["__chips__"] = chips
    rects["__area__"] = area
    return rects


def draw_brain(
    surf: Any,
    schematic: BrainSchematic,
    rects: dict[str, Any],
    *,
    fonts: dict[str, Any],
    selected: str | None = None,
    stage: str | None = None,
) -> list[tuple[Any, str]]:
    """Draw CNS Brain block diagram. Returns local (rect, id) hits."""
    import pygame

    hits: list[tuple[Any, str]] = []
    area = rects.get("__area__")
    if area is not None:
        for gx in range(area.x, area.right, 28):
            pygame.draw.line(surf, _BG_GRID, (gx, area.y), (gx, area.bottom), 1)
        for gy in range(area.y, area.bottom, 28):
            pygame.draw.line(surf, _BG_GRID, (area.x, gy), (area.right, gy), 1)

    title_f = fonts.get("title") or fonts.get("small")
    body_f = fonts.get("hint") or fonts.get("small")
    small_f = fonts.get("small") or body_f

    cur_stage = stage or schematic.stage
    mode_col = (
        (200, 160, 100)
        if schematic.mode == "bypass"
        else ((160, 210, 255) if schematic.mode == "train" else _TITLE)
    )
    header = title_f.render(
        f"CNS Brain · {schematic.mode.upper()} · hop / carve schema",
        True,
        mode_col,
    )
    surf.blit(header, (area.x + 12 if area else 12, (area.y + 8) if area else 8))
    sub = body_f.render(schematic.subtitle, True, _MUTED)
    surf.blit(sub, (area.x + 12 if area else 12, (area.y + 24) if area else 24))

    id_to_block = {b.id: b for b in schematic.blocks}
    for src, dst, lab in schematic.edges:
        ra = rects.get(src)
        rb = rects.get(dst)
        if ra is None or rb is None:
            continue
        a = (ra.centerx, ra.centery)
        b = (rb.centerx, rb.centery)
        hot = False
        if cur_stage:
            sa = id_to_block.get(src)
            sb = id_to_block.get(dst)
            if sa and sb and (sa.stage == cur_stage or sb.stage == cur_stage):
                hot = True
        if selected in (src, dst):
            hot = True
        col = _EDGE_HOT if hot else _EDGE
        pygame.draw.line(surf, col, a, b, 2 if hot else 1)
        if lab:
            mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
            t = small_f.render(lab, True, col)
            surf.blit(t, (mx - t.get_width() // 2, my - t.get_height() - 2))

    for b in schematic.blocks:
        r = rects.get(b.id)
        if r is None:
            continue
        dim = cur_stage not in ("", None) and b.stage != cur_stage and b.id != selected
        fill = tuple(max(0, c - (18 if dim else 0)) for c in (24, 30, 38))
        accent = b.accent if not dim else tuple(max(0, c - 30) for c in b.accent)
        if b.id == selected:
            fill = (28, 36, 44)
        pygame.draw.rect(surf, fill, r, border_radius=4)
        border_w = 2 if (b.id == selected or b.live) else 1
        pygame.draw.rect(surf, accent, r, border_w, border_radius=4)
        if b.live:
            pygame.draw.circle(surf, accent, (r.right - 8, r.y + 8), 3)

        title = small_f.render(b.title, True, accent)
        surf.blit(title, (r.x + 6, r.y + 4))
        yy = r.y + 4 + title.get_height() + 2
        for line in b.summary[:2]:
            lab = body_f.render(str(line), True, _BODY if not dim else _MUTED)
            surf.blit(lab, (r.x + 6, yy))
            yy += lab.get_height() + 1
        if b.bars and yy + 8 < r.bottom - 4:
            bw = r.w - 12
            n = len(b.bars)
            sw = max(2, (bw - (n - 1)) // max(1, n))
            for i, v in enumerate(b.bars):
                h = max(1, int(10 * float(np.clip(v, 0, 1))))
                bx = r.x + 6 + i * (sw + 1)
                by = r.bottom - 4 - h
                pygame.draw.rect(surf, accent, (bx, by, sw, h))
        hits.append((r, b.id))

    for sid, label, cr in rects.get("__chips__", []):
        on = sid == cur_stage
        pygame.draw.rect(
            surf,
            (35, 50, 60) if on else (28, 32, 40),
            cr,
            border_radius=3,
        )
        pygame.draw.rect(
            surf,
            (120, 190, 210) if on else (70, 85, 100),
            cr,
            1,
            border_radius=3,
        )
        t = small_f.render(label, True, (200, 230, 240) if on else _MUTED)
        surf.blit(
            t,
            (cr.x + (cr.w - t.get_width()) // 2, cr.y + (cr.h - t.get_height()) // 2),
        )
        hits.append((cr, f"stage:{sid}"))

    if selected and selected in id_to_block and area is not None:
        blk = id_to_block[selected]
        pw = min(340, max(220, area.w // 3))
        ph = min(area.h - 80, 28 + 16 * (2 + len(blk.detail)))
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(_PANEL_BG)
        pygame.draw.rect(panel, _PANEL_EDGE, panel.get_rect(), 1)
        yy = 8
        ht = title_f.render(blk.title, True, blk.accent)
        panel.blit(ht, (10, yy))
        yy += ht.get_height() + 6
        for line in blk.detail:
            lt = body_f.render(str(line)[:54], True, _BODY)
            panel.blit(lt, (10, yy))
            yy += lt.get_height() + 2
            if yy > ph - 12:
                break
        px = area.right - pw - 10
        py = area.y + 40
        surf.blit(panel, (px, py))
        hits.append((pygame.Rect(px, py, pw, ph), selected))

    return hits


def hit_brain(
    pos: tuple[int, int],
    hits: list[tuple[Any, str]],
    *,
    origin: tuple[int, int] = (0, 0),
) -> str | None:
    ox, oy = origin
    x, y = pos[0] - ox, pos[1] - oy
    for rect, bid in reversed(hits):
        if rect.collidepoint(x, y):
            return bid
    return None


def next_stage(cur: str) -> str:
    ids = [s for s, _ in _STAGES]
    if cur not in ids:
        return ids[0]
    return ids[(ids.index(cur) + 1) % len(ids)]

"""AI Brain schematic — iterative block diagram of the sensor→MLP→turret path.

Honest: this is the *separate* AI Brain MLP (CNS Brain unused), not CNS hop.
Click a block for detail; stage chips walk the pipeline left→right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_TITLE = (210, 220, 240)
_BODY = (155, 165, 185)
_MUTED = (140, 150, 170)
_EDGE = (70, 90, 120)
_EDGE_HOT = (180, 160, 100)
_PANEL_BG = (18, 22, 30, 235)
_PANEL_EDGE = (160, 140, 90)
_BG_GRID = (22, 24, 30)

_STAGES = (
    ("sense", "1 Sense"),
    ("feat", "2 Features"),
    ("hid", "3 Hidden"),
    ("out", "4 Cmd"),
    ("plant", "5 Plant"),
    ("train", "Train"),
)


@dataclass(frozen=True)
class AiBrainBlock:
    id: str
    title: str
    stage: str  # sense|feat|hid|out|plant|train
    summary: tuple[str, ...]
    detail: tuple[str, ...]
    accent: tuple[int, int, int]
    live: bool = False
    bars: tuple[float, ...] = ()  # optional 0..1 meters


@dataclass
class AiBrainSchematic:
    blocks: list[AiBrainBlock] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # src,dst,label
    mode: str = "off"
    stage: str = "sense"
    subtitle: str = ""

    @classmethod
    def from_host(cls, host: Any, *, stage: str | None = None) -> AiBrainSchematic:
        from sensor_port_nn import (
            N_HIDDEN,
            N_IN,
            N_OUT,
            features,
            loss_band,
            loss_rgb,
            training_targets,
        )
        port = getattr(host, "sensor_port_nn", None)
        on = bool(getattr(host, "sensor_port_on", False))
        train = bool(getattr(host, "sensor_port_train", False))
        frozen = bool(getattr(port, "frozen", False)) if port is not None else False
        if frozen:
            mode = "frozen"
        elif train and on:
            mode = "train"
        elif on:
            mode = "run"
        else:
            mode = "off"

        # Refresh live forward for diagram meters.
        x = h = y = None
        if port is not None:
            try:
                y, h, x = port.forward(features(host))
                port.last_cmd = y.copy()
            except Exception:
                x = getattr(port, "last_x", None)
                h = getattr(port, "last_h", None)
                y = getattr(port, "last_cmd", None)

        el = np.asarray(getattr(host, "eye_left_I", np.zeros(6)), dtype=float).ravel()[:6]
        er = np.asarray(getattr(host, "eye_right_I", np.zeros(6)), dtype=float).ravel()[:6]
        if len(el) < 6:
            el = np.pad(el, (0, 6 - len(el)))
        if len(er) < 6:
            er = np.pad(er, (0, 6 - len(er)))
        jl = float(getattr(host, "jo_left_tilt", 0.0) or 0.0)
        jr = float(getattr(host, "jo_right_tilt", 0.0) or 0.0)
        S = np.asarray(getattr(host, "turret_S", np.zeros(2)), dtype=float).ravel()
        if len(S) < 2:
            S = np.pad(S, (0, 2 - len(S)))
        yaw = float(getattr(host, "turret_yaw", 0.0) or 0.0)
        pitch = float(getattr(host, "turret_pitch", 0.0) or 0.0)
        err = float(getattr(host, "turret_aim_err", float("nan")))
        eps = float(math.degrees(err)) if math.isfinite(err) else float("nan")
        track = float(getattr(host, "_track_score", float("nan")))

        eye_conf = 0.0
        ehat = 0.0
        try:
            from compound_eyes import eye_lamp_azimuth, eye_light_vector

            az = eye_lamp_azimuth(host)
            if az is not None:
                ehat = float((az[0] - yaw + math.pi) % math.tau - math.pi)
                eye_conf = float(az[1])
            vec = eye_light_vector(host)
            if vec is not None:
                eye_conf = max(eye_conf, float(vec[1]))
        except Exception:
            pass

        tgt = None
        try:
            tgt = training_targets(host)
        except Exception:
            tgt = None

        loss = float(getattr(port, "loss_ema", float("nan"))) if port else float("nan")
        loss_accent = loss_rgb(loss) if math.isfinite(loss) else (220, 150, 100)
        band = loss_band(loss) if math.isfinite(loss) else "—"
        step_n = int(getattr(port, "step_n", 0) or 0) if port else 0
        buf_n = len(getattr(port, "_buf_x", []) or []) if port else 0
        lr = float(getattr(port, "lr", 0.0) or 0.0) if port else 0.0

        cmd = np.asarray(y if y is not None else getattr(port, "last_cmd", [0, 0]), dtype=float).ravel()
        if len(cmd) < 2:
            cmd = np.zeros(2)

        hid = np.asarray(h if h is not None else getattr(port, "last_h", np.zeros(N_HIDDEN)), dtype=float).ravel()
        if len(hid) < N_HIDDEN:
            hid = np.pad(hid, (0, max(0, N_HIDDEN - len(hid))))[:N_HIDDEN]
        feat = np.asarray(x if x is not None else getattr(port, "last_x", np.zeros(N_IN)), dtype=float).ravel()
        if len(feat) < N_IN:
            feat = np.pad(feat, (0, max(0, N_IN - len(feat))))[:N_IN]

        # Hidden strip: downsample to 16 bars for UI.
        hid_bars = tuple(float(np.clip(v, 0.0, 1.0)) for v in hid[:: max(1, N_HIDDEN // 16)][:16])
        feat_bars = tuple(float(np.clip(abs(v), 0.0, 1.5) / 1.5) for v in feat[: min(12, len(feat))])

        blocks = [
            AiBrainBlock(
                id="world",
                title="World / lamp",
                stage="sense",
                summary=(
                    f"spin {float(getattr(host, 'lamp_speed', 0.0) or 0.0):.2f}",
                    f"lamp {'on' if getattr(host, 'lamp_on', False) else 'off'}",
                ),
                detail=(
                    f"lamp_angle = {float(getattr(host, 'lamp_angle', 0.0) or 0.0):.3f}",
                    f"sky_fill = {float(getattr(host, 'sky_fill', 0.0) or 0.0):.3f}",
                    "Photocurrents into compound eyes (not CNS hop).",
                ),
                accent=(200, 180, 110),
                live=bool(getattr(host, "lamp_on", False)),
            ),
            AiBrainBlock(
                id="eye_l",
                title="Eye L · 6 sites",
                stage="sense",
                summary=(
                    f"max {float(np.max(el)):.2f}",
                    f"on={'yes' if getattr(host, 'eye_left_on', False) else 'no'}",
                ),
                detail=tuple(f"I[{i}] = {float(el[i]):.3f}" for i in range(6)),
                accent=(100, 190, 170),
                live=bool(getattr(host, "eye_left_on", False)) and float(np.max(el)) > 0.05,
                bars=tuple(float(np.clip(v, 0.0, 1.0)) for v in el),
            ),
            AiBrainBlock(
                id="eye_r",
                title="Eye R · 6 sites",
                stage="sense",
                summary=(
                    f"max {float(np.max(er)):.2f}",
                    f"on={'yes' if getattr(host, 'eye_right_on', False) else 'no'}",
                ),
                detail=tuple(f"I[{i}] = {float(er[i]):.3f}" for i in range(6)),
                accent=(90, 175, 190),
                live=bool(getattr(host, "eye_right_on", False)) and float(np.max(er)) > 0.05,
                bars=tuple(float(np.clip(v, 0.0, 1.0)) for v in er),
            ),
            AiBrainBlock(
                id="jo",
                title="JO L/R",
                stage="sense",
                summary=(f"L {jl:+.2f}", f"R {jr:+.2f}"),
                detail=(
                    f"jo_left_tilt = {jl:+.4f}",
                    f"jo_right_tilt = {jr:+.4f}",
                    f"enabled L/R = {bool(getattr(host, 'jo_left_on', False))}/"
                    f"{bool(getattr(host, 'jo_right_on', False))}",
                ),
                accent=(140, 160, 200),
                live=abs(jl) + abs(jr) > 1e-3,
            ),
            AiBrainBlock(
                id="proprio",
                title="Proprio",
                stage="sense",
                summary=(
                    f"S=[{float(S[0]):.2f},{float(S[1]):.2f}]",
                    f"pitch {pitch:+.2f}",
                ),
                detail=(
                    f"turret_yaw = {yaw:.4f} rad",
                    f"turret_pitch = {pitch:.4f}",
                    f"turret_S = [{float(S[0]):.3f}, {float(S[1]):.3f}]",
                    f"eye_az err ≈ {math.degrees(ehat):+.1f}°  conf={eye_conf:.2f}",
                ),
                accent=(160, 150, 180),
                live=True,
                bars=(float(S[0]), float(S[1]), float(np.clip((pitch + 0.35) / 1.2, 0, 1))),
            ),
            AiBrainBlock(
                id="feat",
                title=f"Features · {N_IN}",
                stage="feat",
                summary=(
                    f"‖x‖₁={float(np.sum(np.abs(feat))):.2f}",
                    f"conf {eye_conf:.2f}",
                ),
                detail=(
                    "Layout: eyeL(6)+eyeR(6)+JO(2)+S(2)+sin/cos yaw+pitch",
                    "+ ehat_yaw + ehat_pitch + eye_conf",
                    f"N_IN = {N_IN}",
                    "No hop / no contact_gain in this vector.",
                ),
                accent=(180, 170, 120),
                live=on,
                bars=feat_bars,
            ),
            AiBrainBlock(
                id="hid",
                title=f"Hidden ReLU · {N_HIDDEN}",
                stage="hid",
                summary=(
                    f"active {int(np.count_nonzero(hid > 1e-4))}/{N_HIDDEN}",
                    f"mean {float(np.mean(hid)):.3f}",
                ),
                detail=(
                    "h = ReLU(W1·x + b1)",
                    f"W1 shape = ({N_HIDDEN}, {N_IN})",
                    f"‖W1‖₂ ≈ {float(np.linalg.norm(port.W1)):.2f}" if port is not None else "W1 —",
                    "CNS Brain gains never touched.",
                ),
                accent=(210, 160, 100),
                live=on and float(np.max(hid)) > 1e-4,
                bars=hid_bars,
            ),
            AiBrainBlock(
                id="out",
                title=f"Out tanh · {N_OUT}",
                stage="out",
                summary=(
                    f"yaw {float(cmd[0]):+.2f}",
                    f"pitch {float(cmd[1]):+.2f}",
                ),
                detail=(
                    "y = tanh(W2·h + b2)",
                    f"W2 shape = ({N_OUT}, {N_HIDDEN})",
                    f"cmd yaw/pitch = ({float(cmd[0]):+.4f}, {float(cmd[1]):+.4f})",
                    "Pre-update cmd drives slew (matches Freeze RUN).",
                ),
                accent=(220, 180, 90),
                live=on and (abs(float(cmd[0])) + abs(float(cmd[1])) > 0.02),
                bars=(
                    float(np.clip(0.5 + 0.5 * float(cmd[0]), 0, 1)),
                    float(np.clip(0.5 + 0.5 * float(cmd[1]), 0, 1)),
                ),
            ),
            AiBrainBlock(
                id="plant",
                title="Turret plant",
                stage="plant",
                summary=(
                    f"ε {eps:.1f}°" if math.isfinite(eps) else "ε —",
                    f"Track {track:.0%}" if math.isfinite(track) else "Track —",
                ),
                detail=(
                    f"aim ε = {eps:.2f}°" if math.isfinite(eps) else "aim ε = —",
                    f"Track = {track:.1%}" if math.isfinite(track) else "Track = —",
                    f"pose yaw={yaw:.3f} pitch={pitch:.3f}",
                    "Slew from AI Brain cmds only (CNS hop quiet).",
                    f"contact_gain n = {len(getattr(getattr(host, 'fly', None), 'contact_gain', {}) or {})}",
                ),
                accent=(150, 200, 140),
                live=on,
            ),
            AiBrainBlock(
                id="expert",
                title="Expert labels",
                stage="train",
                summary=(
                    (
                        f"t=({float(tgt[0][0]):+.2f},{float(tgt[0][1]):+.2f})"
                        if tgt is not None
                        else "t = —"
                    ),
                    f"mode {mode}",
                ),
                detail=(
                    "Imitation: desired_aim − pose → scaled cmds",
                    (
                        f"target = ({float(tgt[0][0]):+.4f}, {float(tgt[0][1]):+.4f})  "
                        f"conf={float(tgt[1]):.2f}"
                        if tgt is not None
                        else "no desired_aim"
                    ),
                    "Privileged teacher for labels only — features stay sensory.",
                ),
                accent=(200, 140, 160),
                live=train and not frozen and tgt is not None,
            ),
            AiBrainBlock(
                id="replay",
                title="Replay + SGD",
                stage="train",
                summary=(
                    f"n={step_n}",
                    (
                        f"buf={buf_n}  L={loss:.3f} {band}"
                        if math.isfinite(loss)
                        else f"buf={buf_n}"
                    ),
                ),
                detail=(
                    f"steps = {step_n}",
                    (
                        f"loss_ema = {loss:.4f}  [{band}]"
                        if math.isfinite(loss)
                        else "loss_ema = —"
                    ),
                    f"loss_best = {float(getattr(port, 'loss_best', float('nan'))):.4f}"
                    if port is not None
                    else "loss_best = —",
                    f"replay_k = {int(getattr(port, 'replay_k', 0) or 0)}  "
                    f"phantom_k = {int(getattr(port, 'phantom_k', 0) or 0)}",
                    f"lr = {lr:.4f}  momentum = {float(getattr(port, 'momentum', 0.0) or 0.0):.2f}"
                    if port is not None
                    else "lr = —",
                    "Frozen → no SGD; drive uses locked W.",
                ),
                accent=loss_accent,
                live=train and not frozen and step_n > 0,
            ),
        ]

        edges = [
            ("world", "eye_l", "light"),
            ("world", "eye_r", "light"),
            ("eye_l", "feat", ""),
            ("eye_r", "feat", ""),
            ("jo", "feat", ""),
            ("proprio", "feat", ""),
            ("feat", "hid", "x"),
            ("hid", "out", "h"),
            ("out", "plant", "cmd"),
            ("expert", "replay", "t"),
            ("feat", "replay", "x"),
            ("replay", "hid", "ΔW"),
            ("replay", "out", "ΔW"),
        ]

        st = stage if stage in {s for s, _ in _STAGES} else "sense"
        sub = (
            f"AI Brain · {mode} · CNS Brain unused · click block · stage chips iterate pipeline"
        )
        return cls(blocks=blocks, edges=edges, mode=mode, stage=st, subtitle=sub)


def layout_ai_brain(schematic: AiBrainSchematic, area: Any) -> dict[str, Any]:
    """Place blocks in stage columns + detail strip."""
    import pygame

    pad = 14
    top = area.y + 36
    bottom = area.bottom - 52
    left = area.x + pad
    right = area.right - pad
    stage_ids = [s for s, _ in _STAGES]
    # Collapse unused train column visually but keep layout slots.
    cols = len(stage_ids)
    cw = max(96, (right - left - (cols - 1) * 10) // cols)
    gap = 10
    rects: dict[str, Any] = {}
    by_stage: dict[str, list[AiBrainBlock]] = {s: [] for s in stage_ids}
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

    # Stage chip row
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


def draw_ai_brain(
    surf: Any,
    schematic: AiBrainSchematic,
    rects: dict[str, Any],
    *,
    fonts: dict[str, Any],
    selected: str | None = None,
    stage: str | None = None,
) -> list[tuple[Any, str]]:
    """Draw AI Brain block diagram. Returns local (rect, id) hits."""
    import pygame

    hits: list[tuple[Any, str]] = []
    area = rects.get("__area__")
    if area is not None:
        # Subtle grid
        for gx in range(area.x, area.right, 28):
            pygame.draw.line(surf, _BG_GRID, (gx, area.y), (gx, area.bottom), 1)
        for gy in range(area.y, area.bottom, 28):
            pygame.draw.line(surf, _BG_GRID, (area.x, gy), (area.right, gy), 1)

    title_f = fonts.get("title") or fonts.get("small")
    body_f = fonts.get("hint") or fonts.get("small")
    small_f = fonts.get("small") or body_f

    cur_stage = stage or schematic.stage
    header = title_f.render(
        f"AI Brain · {schematic.mode.upper()} · iterative block schema",
        True,
        _TITLE,
    )
    surf.blit(header, (area.x + 12 if area else 12, (area.y + 8) if area else 8))
    sub = body_f.render(schematic.subtitle, True, _MUTED)
    surf.blit(sub, (area.x + 12 if area else 12, (area.y + 24) if area else 24))

    # Edges under blocks
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
        fill = tuple(max(0, c - (18 if dim else 0)) for c in (28, 30, 38))
        accent = b.accent if not dim else tuple(max(0, c - 30) for c in b.accent)
        if b.id == selected:
            fill = (36, 34, 28)
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
            sw = max(2, (bw - (n - 1)) // n)
            for i, v in enumerate(b.bars):
                h = max(1, int(10 * float(np.clip(v, 0, 1))))
                bx = r.x + 6 + i * (sw + 1)
                by = r.bottom - 4 - h
                pygame.draw.rect(surf, accent, (bx, by, sw, h))
        hits.append((r, b.id))

    # Stage chips
    for sid, label, cr in rects.get("__chips__", []):
        on = sid == cur_stage
        pygame.draw.rect(
            surf,
            (50, 45, 35) if on else (30, 32, 40),
            cr,
            border_radius=3,
        )
        pygame.draw.rect(
            surf,
            (200, 180, 110) if on else (80, 85, 100),
            cr,
            1,
            border_radius=3,
        )
        t = small_f.render(label, True, (220, 210, 180) if on else _MUTED)
        surf.blit(
            t,
            (cr.x + (cr.w - t.get_width()) // 2, cr.y + (cr.h - t.get_height()) // 2),
        )
        hits.append((cr, f"stage:{sid}"))

    # Detail panel when selected
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
            lt = body_f.render(str(line)[:52], True, _BODY)
            panel.blit(lt, (10, yy))
            yy += lt.get_height() + 2
            if yy > ph - 12:
                break
        px = area.right - pw - 10
        py = area.y + 40
        surf.blit(panel, (px, py))
        hits.append((pygame.Rect(px, py, pw, ph), selected))

    return hits


def hit_ai_brain(
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

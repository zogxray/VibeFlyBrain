"""CNS Growing policy schematic — how tips / synapses are grown.

Honest genome constructor view: atlas recipe → genome quota → arm tips →
wells/basins → tip step policies (chemotaxis, pioneer, bundle) → Yield.
Not an MLP and not adult hop/carve (see CNS Brain / AI Brain).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_TITLE = (220, 210, 170)
_BODY = (165, 160, 145)
_MUTED = (145, 140, 125)
_EDGE = (90, 85, 70)
_EDGE_HOT = (210, 180, 110)
_PANEL_BG = (24, 22, 16, 235)
_PANEL_EDGE = (180, 160, 100)
_BG_GRID = (28, 26, 20)

_STAGES = (
    ("recipe", "1 Recipe"),
    ("genome", "2 Genome"),
    ("arm", "3 Arm"),
    ("guide", "4 Guide"),
    ("tip", "5 Tip"),
    ("yield", "6 Yield"),
)


@dataclass(frozen=True)
class GrowBlock:
    id: str
    title: str
    stage: str
    summary: tuple[str, ...]
    detail: tuple[str, ...]
    accent: tuple[int, int, int]
    live: bool = False
    bars: tuple[float, ...] = ()


@dataclass
class GrowSchematic:
    blocks: list[GrowBlock] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    mode: str = "idle"
    stage: str = "recipe"
    subtitle: str = ""

    @classmethod
    def from_host(cls, host: Any, *, stage: str | None = None) -> GrowSchematic:
        fly = getattr(host, "fly", None)
        phase = str(getattr(fly, "phase", "rising") or "rising") if fly else "—"

        n_recipe = 0
        if fly is not None:
            try:
                pre = getattr(fly, "recipe_pre", None)
                n_recipe = 0 if pre is None else int(len(pre))
            except Exception:
                n_recipe = 0
        n_soma = 0
        n_cls = 0
        if fly is not None:
            try:
                n_soma = int(len(getattr(fly, "soma_ids", []) or []))
            except Exception:
                try:
                    n_soma = int(np.asarray(getattr(fly, "soma_ids", [])).shape[0])
                except Exception:
                    n_soma = 0
            try:
                n_cls = int(len(list(getattr(fly, "class_names", []) or [])))
            except Exception:
                n_cls = 0

        # Genome quota
        quota = getattr(fly, "genome_quota", None) if fly is not None else None
        q_sum = 0
        q_cells = 0
        top_pairs: list[str] = []
        if quota is not None:
            try:
                q = np.asarray(quota, dtype=np.int32)
                if q.size:
                    q_sum = int(np.sum(q))
                    q_cells = int(np.count_nonzero(q > 0))
                    names = list(getattr(fly, "class_names", []) or [])
                    flat = np.argwhere(q > 0)
                    if len(flat):
                        vals = q[flat[:, 0], flat[:, 1]]
                        order = np.argsort(-vals)[:4]
                        for k in order:
                            i, j = int(flat[k, 0]), int(flat[k, 1])
                            si = names[i] if i < len(names) else str(i)
                            sj = names[j] if j < len(names) else str(j)
                            top_pairs.append(f"{si}→{sj} ×{int(q[i, j])}")
            except Exception:
                pass

        # Tips / status
        st = getattr(fly, "growth_status", None) if fly is not None else None
        n_tips = 0
        n_ok = n_fail = n_open = n_idle = 0
        if st is not None:
            try:
                arr = np.asarray(st)
                n_tips = int(arr.shape[0])
            except Exception:
                arr = None
                n_tips = 0
            if n_tips and arr is not None:
                from fly_model import (
                    STATUS_CONNECTED,
                    STATUS_FAILED,
                    STATUS_GROWING,
                    STATUS_IDLE,
                    STATUS_SEEKING,
                )

                n_ok = int(np.sum(arr == STATUS_CONNECTED))
                n_fail = int(np.sum(arr == STATUS_FAILED))
                n_open = int(
                    np.sum((arr == STATUS_GROWING) | (arr == STATUS_SEEKING))
                )
                n_idle = int(np.sum(arr == STATUS_IDLE))

        srcs = list(getattr(fly, "growth_sources", []) or []) if fly else []
        tgts = list(getattr(fly, "growth_targets", []) or []) if fly else []

        # Yield
        yld: dict = {}
        if hasattr(host, "_recipe_yield"):
            try:
                yld = dict(host._recipe_yield())
            except Exception:
                yld = {}
        elif fly is not None and hasattr(fly, "recipe_yield"):
            try:
                yld = dict(fly.recipe_yield())
            except Exception:
                yld = {}
        verdict = str(yld.get("verdict", "IDLE"))
        pct = float(yld.get("pct", 0.0) or 0.0)
        armed = int(yld.get("armed", n_tips) or 0)

        # Wells / guide
        wells = getattr(host, "well_field", None) or getattr(host, "_well_field", None)
        n_wells = 0
        well_names: list[str] = []
        if wells is not None:
            specs = getattr(wells, "specs", None) or getattr(wells, "wells", None) or []
            try:
                n_wells = len(specs)
                for s in list(specs)[:4]:
                    well_names.append(str(getattr(s, "name", s)))
            except Exception:
                n_wells = 0
        wg = getattr(host, "well_grow", None)
        if wg is None and hasattr(host, "_well_grow_params"):
            try:
                wg = host._well_grow_params()
            except Exception:
                wg = None

        # Tip policy constants (honest readout of live Fly policy knobs)
        try:
            from fly_model import (
                BUNDLE_BLEND,
                PIONEER_TRAIL_BLEND,
                PIONEER_TRAIL_FAR,
                GROW_SEEK_LIMIT,
            )
        except Exception:
            BUNDLE_BLEND = PIONEER_TRAIL_BLEND = PIONEER_TRAIL_FAR = 0.0
            GROW_SEEK_LIMIT = 0.0

        fuse_on = bool(getattr(host, "_fuse_running", False))
        rising = phase == "rising"
        if phase == "adult":
            mode = "adult"
        elif n_open > 0 or fuse_on:
            mode = "growing"
        elif armed > 0 and n_open == 0:
            mode = str(verdict).lower() if verdict else "settled"
        elif q_sum > 0 or n_recipe > 0:
            mode = "armed-ready" if armed else "idle"
        else:
            mode = "idle"

        grow_live = rising and (n_open > 0 or fuse_on)

        # Top class out bars from yield
        class_bars: list[float] = []
        for row in (yld.get("by_class") or [])[:8]:
            class_bars.append(float(np.clip(float(row.get("out_pct", 0.0)) / 100.0, 0, 1)))

        blocks = [
            GrowBlock(
                id="atlas",
                title="Atlas recipe",
                stage="recipe",
                summary=(
                    f"edges {n_recipe:,}",
                    f"soma {n_soma:,} · cls {n_cls}",
                ),
                detail=(
                    "MaleCNS connectome edges = one successful blueprint.",
                    f"recipe_pre/post edges = {n_recipe:,}",
                    f"somata = {n_soma:,} · classes = {n_cls}",
                    "Import / Fuse loads atlas — not learned weights.",
                    "Constructor goal: compress this into editable genome.",
                ),
                accent=(180, 200, 140),
                live=n_recipe > 0,
            ),
            GrowBlock(
                id="sets",
                title="Set → set",
                stage="recipe",
                summary=(
                    f"src {len(srcs)}",
                    f"tgt {len(tgts)}",
                ),
                detail=(
                    f"growth_sources = {', '.join(srcs[:6]) or '—'}",
                    f"growth_targets = {', '.join(tgts[:6]) or '—'}",
                    "Tips only between armed atlas set pairs.",
                    "Device Fuse adds ocelli/eye/JO/turret owners separately.",
                ),
                accent=(160, 190, 150),
                live=bool(srcs or tgts),
            ),
            GrowBlock(
                id="quota",
                title="Genome quota",
                stage="genome",
                summary=(
                    f"tips {q_sum:,}",
                    f"cells {q_cells}",
                ),
                detail=(
                    "genome_quota[i,j] = how many tips class i → class j.",
                    f"Σ quota = {q_sum:,} · nonzero cells = {q_cells}",
                    *(top_pairs or ("(empty — Start fills from atlas mass)",)),
                    "Editable recipe knob — mutate → different brain.",
                ),
                accent=(210, 180, 100),
                live=q_sum > 0,
            ),
            GrowBlock(
                id="arm",
                title="Arm tips",
                stage="arm",
                summary=(
                    f"armed {armed:,}",
                    f"phase {phase}",
                ),
                detail=(
                    "Start / Rising: ensure_growth_tips from genome_quota.",
                    "Fuse: device↔CNS tips only (no atlas re-grow).",
                    f"live tips = {n_tips:,} · idle {n_idle}",
                    f"fuse_running = {fuse_on}",
                    "Adult phase refuses re-arm (Reset → rising).",
                ),
                accent=(200, 160, 120) if rising else (110, 110, 100),
                live=armed > 0 and rising,
            ),
            GrowBlock(
                id="wells",
                title="Wells / basins",
                stage="guide",
                summary=(
                    f"wells {n_wells}",
                    (
                        f"∩≥{int(getattr(wg, 'min_n', 0))} · steps {int(getattr(wg, 'grow_steps', 0))}"
                        if wg is not None
                        else "params —"
                    ),
                ),
                detail=(
                    "Geometric device wells — tip gravity, not TrainingScope.",
                    f"wells = {', '.join(well_names) or '—'}",
                    (
                        f"WellGrowParams min_a/b/n = "
                        f"{int(getattr(wg,'min_a',0))}/"
                        f"{int(getattr(wg,'min_b',0))}/"
                        f"{int(getattr(wg,'min_n',0))}"
                        if wg is not None
                        else "WellGrowParams —"
                    ),
                    "growth_basin_r = contact domain around goal soma.",
                    "Independent of hop corridor (CNS Brain Scope).",
                ),
                accent=(140, 190, 210),
                live=n_wells > 0,
            ),
            GrowBlock(
                id="chemo",
                title="Chemotaxis",
                stage="guide",
                summary=(
                    "goal pull",
                    f"seek≤{float(GROW_SEEK_LIMIT):.0f}s",
                ),
                detail=(
                    "Tip steps toward growth_goal_xyz (+ tissue near goal).",
                    f"seek_limit base ≈ {float(GROW_SEEK_LIMIT):.1f} fly-s",
                    "Timeout → STATUS_FAILED (honest miss, not silent).",
                    "Well gravity blends when tip is inside device wells.",
                ),
                accent=(120, 200, 180),
                live=grow_live,
            ),
            GrowBlock(
                id="pioneer",
                title="Pioneer trail",
                stage="tip",
                summary=(
                    f"trail {float(PIONEER_TRAIL_BLEND):.2f}",
                    f"far {float(PIONEER_TRAIL_FAR):.2f}",
                ),
                detail=(
                    "One pioneer navigator per (pre_cls, post_cls[, side]) cohort.",
                    "Followers fasciculate on pioneer polyline.",
                    f"PIONEER_TRAIL_BLEND = {float(PIONEER_TRAIL_BLEND):.2f} near goal",
                    f"PIONEER_TRAIL_FAR = {float(PIONEER_TRAIL_FAR):.2f} long-haul",
                    "Far: chemotaxis dominates so AN/DN don't stall mid-neck.",
                ),
                accent=(220, 170, 120),
                live=grow_live and n_open > 0,
            ),
            GrowBlock(
                id="bundle",
                title="Bundle peers",
                stage="tip",
                summary=(
                    f"blend {float(BUNDLE_BLEND):.2f}",
                    f"open {n_open:,}",
                ),
                detail=(
                    "Same-cohort tip↔tip attraction → cables / fascicles.",
                    f"BUNDLE_BLEND = {float(BUNDLE_BLEND):.2f}",
                    "Residual peer bind after pioneer trail weight.",
                    f"growing+seeking now = {n_open:,}",
                ),
                accent=(200, 140, 160),
                live=grow_live and n_open > 1,
            ),
            GrowBlock(
                id="connect",
                title="Connect / fail",
                stage="tip",
                summary=(
                    f"ok {n_ok:,}",
                    f"fail {n_fail:,}",
                ),
                detail=(
                    "STATUS_CONNECTED when tip enters partner basin.",
                    "STATUS_FAILED on seek timeout — counted in Yield.",
                    f"connected = {n_ok:,} · failed = {n_fail:,} · open = {n_open:,}",
                    "Connected trails materialize as grown axons/dendrites.",
                ),
                accent=(150, 210, 140) if n_ok else (180, 120, 110),
                live=n_ok > 0 or n_fail > 0,
                bars=(
                    float(n_ok / max(1, n_tips)),
                    float(n_open / max(1, n_tips)),
                    float(n_fail / max(1, n_tips)),
                ),
            ),
            GrowBlock(
                id="yield",
                title="Recipe Yield",
                stage="yield",
                summary=(
                    f"{verdict} · {pct:.0f}%",
                    f"{int(yld.get('ok', n_ok)):,}/{armed:,}",
                ),
                detail=(
                    "Yield = connected / armed tips (Start attempt).",
                    f"verdict = {verdict} · pct = {pct:.1f}%",
                    f"open = {int(yld.get('open', n_open)):,} · "
                    f"fail = {int(yld.get('fail', n_fail)):,}",
                    "CONNECTED/PARTIAL → Adult unlocks Learn.",
                    "Per-class Out/In strips = competence of the recipe.",
                ),
                accent=(200, 210, 140) if verdict in ("CONNECTED", "PARTIAL") else (180, 150, 100),
                live=armed > 0,
                bars=tuple(class_bars) if class_bars else (),
            ),
            GrowBlock(
                id="adult",
                title="Adult handoff",
                stage="yield",
                summary=(
                    f"now {phase}",
                    "Learn after settle",
                ),
                detail=(
                    "Rising grows wiring; Adult runs CNS Brain / AI Brain.",
                    "contact_gain carve (CNS Brain) starts after Adult.",
                    "AI Brain MLP never grows tips — separate learner.",
                    "Success test: mutate genome quota → different still-working net.",
                ),
                accent=(160, 200, 220) if phase == "adult" else (120, 120, 110),
                live=phase == "adult",
            ),
        ]

        edges = [
            ("atlas", "sets", "sets"),
            ("sets", "quota", "mass"),
            ("quota", "arm", "Σ tips"),
            ("arm", "wells", "owners"),
            ("wells", "chemo", "gravity"),
            ("chemo", "pioneer", "step"),
            ("pioneer", "bundle", "cohort"),
            ("bundle", "connect", "contact"),
            ("chemo", "connect", "basin"),
            ("connect", "yield", "ok/fail"),
            ("yield", "adult", "settle"),
        ]

        stg = stage if stage in {s for s, _ in _STAGES} else "recipe"
        sub = (
            f"CNS Growing · {mode} · genome tips · policies: chemo / pioneer / bundle · "
            "click block · stage chips iterate"
        )
        return cls(blocks=blocks, edges=edges, mode=mode, stage=stg, subtitle=sub)


def layout_grow(schematic: GrowSchematic, area: Any) -> dict[str, Any]:
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
    by_stage: dict[str, list[GrowBlock]] = {s: [] for s in stage_ids}
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


def draw_grow(
    surf: Any,
    schematic: GrowSchematic,
    rects: dict[str, Any],
    *,
    fonts: dict[str, Any],
    selected: str | None = None,
    stage: str | None = None,
) -> list[tuple[Any, str]]:
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
        (180, 230, 160)
        if schematic.mode in ("connected", "adult")
        else (
            (220, 190, 110)
            if schematic.mode == "growing"
            else ((200, 140, 120) if schematic.mode == "failed" else _TITLE)
        )
    )
    header = title_f.render(
        f"CNS Growing · {schematic.mode.upper()} · tip / synapse policy",
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
        fill = tuple(max(0, c - (18 if dim else 0)) for c in (32, 28, 22))
        accent = b.accent if not dim else tuple(max(0, c - 30) for c in b.accent)
        if b.id == selected:
            fill = (40, 34, 24)
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
            (48, 42, 30) if on else (30, 28, 24),
            cr,
            border_radius=3,
        )
        pygame.draw.rect(
            surf,
            (210, 180, 110) if on else (90, 85, 75),
            cr,
            1,
            border_radius=3,
        )
        t = small_f.render(label, True, (230, 220, 190) if on else _MUTED)
        surf.blit(
            t,
            (cr.x + (cr.w - t.get_width()) // 2, cr.y + (cr.h - t.get_height()) // 2),
        )
        hits.append((cr, f"stage:{sid}"))

    if selected and selected in id_to_block and area is not None:
        blk = id_to_block[selected]
        pw = min(360, max(220, area.w // 3))
        ph = min(area.h - 80, 28 + 16 * (2 + len(blk.detail)))
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(_PANEL_BG)
        pygame.draw.rect(panel, _PANEL_EDGE, panel.get_rect(), 1)
        yy = 8
        ht = title_f.render(blk.title, True, blk.accent)
        panel.blit(ht, (10, yy))
        yy += ht.get_height() + 6
        for line in blk.detail:
            lt = body_f.render(str(line)[:56], True, _BODY)
            panel.blit(lt, (10, yy))
            yy += lt.get_height() + 2
            if yy > ph - 12:
                break
        px = area.right - pw - 10
        py = area.y + 40
        surf.blit(panel, (px, py))
        hits.append((pygame.Rect(px, py, pw, ph), selected))

    return hits


def hit_grow(
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

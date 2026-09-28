"""Research map — project-state flowchart (not a tissue meter).

Honest constructor log: atlas → grow → failed CNS-train branch → pivot to
AI Brain reference → plan to graft a working net into CNS → memory /
periphery → working organism. Branching language: done / dead / pivot / now / next.

Not a paper twin diagram; terminal node = still-working nervous system you can
mutate, with periphery — the constructor success test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BlockStatus = Literal["done", "dead", "active", "partial", "planned", "goal"]
EdgeKind = Literal["forward", "pivot", "dead"]

_TITLE = (210, 215, 230)
_BODY = (160, 168, 185)
_MUTED = (140, 148, 165)
_EDGE = (70, 80, 100)
_EDGE_HOT = (200, 180, 120)
_EDGE_DEAD = (140, 70, 70)
_EDGE_PIVOT = (200, 150, 90)
_PANEL_BG = (16, 18, 26, 235)
_PANEL_EDGE = (140, 150, 180)
_BG_GRID = (24, 26, 34)

# Lane chips — filter/highlight branches of the research story.
_LANES = (
    ("past", "1 Past"),
    ("branch", "2 Branch"),
    ("now", "3 Now"),
    ("next", "4 Next"),
    ("goal", "5 Goal"),
)

# Fixed flowchart positions (nx, ny) in 0..1 content box — enables branching.
_LAYOUT: dict[str, tuple[float, float]] = {
    "atlas": (0.06, 0.38),
    "grow": (0.20, 0.38),
    "cns_train": (0.34, 0.72),  # dead branch — lower lane
    "ai_ref": (0.36, 0.38),
    "graft": (0.50, 0.38),
    "drive": (0.62, 0.38),
    "memory": (0.74, 0.38),
    "peri": (0.86, 0.22),
    "organism": (0.86, 0.55),
}


@dataclass(frozen=True)
class ResearchBlock:
    id: str
    title: str
    lane: str  # past|branch|now|next|goal
    status: BlockStatus
    summary: tuple[str, ...]
    detail: tuple[str, ...]
    accent: tuple[int, int, int]
    live: bool = False


@dataclass(frozen=True)
class ResearchEdge:
    src: str
    dst: str
    label: str
    kind: EdgeKind = "forward"


@dataclass
class ResearchSchematic:
    blocks: list[ResearchBlock] = field(default_factory=list)
    edges: list[ResearchEdge] = field(default_factory=list)
    mode: str = "map"
    stage: str = "now"  # lane chip
    subtitle: str = ""

    @classmethod
    def from_host(cls, host: Any, *, stage: str | None = None) -> ResearchSchematic:
        fly = getattr(host, "fly", None)
        phase = str(getattr(fly, "phase", "") or "") if fly else ""
        n_soma = 0
        if fly is not None:
            try:
                import numpy as np

                n_soma = int(np.asarray(getattr(fly, "soma_ids", [])).shape[0])
            except Exception:
                try:
                    n_soma = int(len(getattr(fly, "soma_ids", []) or []))
                except Exception:
                    n_soma = 0

        n_tips = 0
        if fly is not None:
            st = getattr(fly, "growth_status", None)
            if st is not None:
                try:
                    import numpy as np

                    n_tips = int(np.asarray(st).shape[0])
                except Exception:
                    n_tips = 0

        port = getattr(host, "sensor_port_nn", None)
        port_on = False
        try:
            port_on = bool(host._port_nn_active())  # type: ignore[attr-defined]
        except Exception:
            port_on = bool(getattr(host, "sensor_port_nn_on", False))
        port_steps = int(getattr(port, "steps", 0) or 0) if port is not None else 0

        ocelli = bool(getattr(host, "ocelli_on", False))
        turret = bool(getattr(host, "turret_on", False))
        peri_n = int(ocelli) + int(turret)

        atlas_done = n_soma > 0
        grow_done = phase in ("adult",) or n_tips > 0 or atlas_done
        ai_active = port is not None and (port_on or port_steps > 0)
        ai_present = port is not None
        peri_partial = peri_n > 0

        blocks = [
            ResearchBlock(
                id="atlas",
                title="FlyWire atlas",
                lane="past",
                status="done" if atlas_done else "planned",
                summary=(
                    f"{n_soma:,} somata" if atlas_done else "not loaded",
                    "connectome blueprint",
                ),
                detail=(
                    "Started here: MaleCNS / FlyWire as one successful wiring instance.",
                    "Raw material for recipes — not the product.",
                    f"somata = {n_soma:,}",
                    "Track: genome compression from this blueprint.",
                ),
                accent=(120, 180, 220) if atlas_done else (90, 100, 120),
                live=atlas_done,
            ),
            ResearchBlock(
                id="grow",
                title="Grow synapses",
                lane="past",
                status="done" if grow_done else "planned",
                summary=(
                    f"phase {phase or '—'}",
                    f"tips {n_tips:,}" if n_tips else "Rising tips",
                ),
                detail=(
                    "Grew tips / synapses on the atlas recipe (CNS Growing).",
                    "Genome quota → arm → chemo / pioneer / bundle → Yield.",
                    f"phase = {phase or '—'} · tips = {n_tips:,}",
                    "Still the right substrate; training path branched next.",
                ),
                accent=(200, 180, 110) if grow_done else (90, 100, 120),
                live=phase == "rising" and n_tips > 0,
            ),
            ResearchBlock(
                id="cns_train",
                title="CNS Learn attempt",
                lane="branch",
                status="dead",
                summary=(
                    "FAILED",
                    "hop / carve · Track flat",
                ),
                detail=(
                    "Tried to train plasticity on CNS edges (contact_gain / hop).",
                    "Did not lock Track under Spin — wrong direction for now.",
                    "Kept as a dead branch so we remember the wrong turn.",
                    "Pivot ↓ to a frozen-topology reference learner (AI Brain).",
                    "Forbidden trap avoided later: not Shiu whole-brain LIF redo.",
                ),
                accent=(200, 90, 90),
                live=False,
            ),
            ResearchBlock(
                id="ai_ref",
                title="AI Brain reference",
                lane="now",
                status="active" if ai_active else ("partial" if ai_present else "planned"),
                summary=(
                    "MLP · eyes+JO→turret" if ai_present else "not armed",
                    f"steps {port_steps}" if port_steps else "imitation + replay",
                ),
                detail=(
                    "Reference net: learn stimulation / commands, freeze CNS topology.",
                    "Exclusive drive while on — CNS hop not driving turret.",
                    "Purpose: prove Track competence, then graft lessons into CNS.",
                    f"on = {port_on} · steps = {port_steps}",
                    "Track (2): stimulation policy — not end-to-end 50M synapses.",
                ),
                accent=(220, 170, 100) if ai_active else (180, 150, 90),
                live=ai_active,
            ),
            ResearchBlock(
                id="graft",
                title="Graft into CNS",
                lane="next",
                status="planned",
                summary=("NEXT", "grow net inside tissue"),
                detail=(
                    "Plan: grow / seat a working policy inside the fly CNS,",
                    "not only beside it as Port MLP.",
                    "Socket / module I/O as bridge; genome rules long-term.",
                    "Success: mutate recipe → different still-working net.",
                ),
                accent=(140, 160, 190),
                live=False,
            ),
            ResearchBlock(
                id="drive",
                title="Make it work",
                lane="next",
                status="planned",
                summary=("NEXT", "closed loop in tissue"),
                detail=(
                    "Drive the grafted net to hold Track / competence under Spin.",
                    "Honest meters: hop / activation / carve — no lying overlays.",
                    "If it fails, branch again — map keeps the dead end.",
                ),
                accent=(140, 160, 190),
                live=False,
            ),
            ResearchBlock(
                id="memory",
                title="Add memory",
                lane="next",
                status="planned",
                summary=("NEXT", "state over time"),
                detail=(
                    "Add memory mechanisms the organism needs (not MB paper clone).",
                    "Must show on meters honestly when live.",
                ),
                accent=(140, 160, 190),
                live=False,
            ),
            ResearchBlock(
                id="peri",
                title="Periphery",
                lane="next" if not peri_partial else "now",
                status="partial" if peri_partial else "planned",
                summary=(
                    f"ocelli {'on' if ocelli else 'off'} · turret {'on' if turret else 'off'}",
                    "sensors / effectors",
                ),
                detail=(
                    "Full periphery the working organism needs.",
                    "Already probing: ocelli, compound eyes, JO, turret.",
                    "Expand until sockets cover the behaviors we care about.",
                ),
                accent=(120, 190, 160) if peri_partial else (140, 160, 190),
                live=peri_partial,
            ),
            ResearchBlock(
                id="organism",
                title="Working organism",
                lane="goal",
                status="goal",
                summary=("GOAL", "constructor · not twin"),
                detail=(
                    "Destination: full-function virtual organism you can reassemble.",
                    "Success test: from one connectome → editable recipe;",
                    "change it → different still-working nervous system.",
                    "Not 'digital twin of the fly' as the product story.",
                    "Visuals stay honest to what the CNS is doing.",
                ),
                accent=(180, 200, 255),
                live=False,
            ),
        ]

        edges = [
            ResearchEdge("atlas", "grow", "grow", "forward"),
            ResearchEdge("grow", "cns_train", "train CNS", "dead"),
            ResearchEdge("grow", "ai_ref", "pivot", "pivot"),
            ResearchEdge("cns_train", "ai_ref", "retreat", "pivot"),
            ResearchEdge("ai_ref", "graft", "graft", "forward"),
            ResearchEdge("graft", "drive", "close loop", "forward"),
            ResearchEdge("drive", "memory", "state", "forward"),
            ResearchEdge("drive", "peri", "I/O", "forward"),
            ResearchEdge("memory", "organism", "compose", "forward"),
            ResearchEdge("peri", "organism", "embody", "forward"),
        ]

        stg = stage if stage in {s for s, _ in _LANES} else "now"
        mode = "active" if ai_active else ("past" if atlas_done else "empty")
        sub = (
            "Research map · branch = wrong turns kept · pivot = retreat + new path · "
            "click block · lane chips filter"
        )
        return cls(blocks=blocks, edges=edges, mode=mode, stage=stg, subtitle=sub)


def layout_research(schematic: ResearchSchematic, area: Any) -> dict[str, Any]:
    import pygame

    pad = 14
    top = area.y + 40
    bottom = area.bottom - 52
    left = area.x + pad
    right = area.right - pad
    bw = max(108, min(148, (right - left) // 8))
    bh = 64
    rects: dict[str, Any] = {}
    for b in schematic.blocks:
        nx, ny = _LAYOUT.get(b.id, (0.5, 0.5))
        cx = left + nx * (right - left)
        cy = top + ny * (bottom - top)
        rects[b.id] = pygame.Rect(int(cx - bw // 2), int(cy - bh // 2), bw, bh)

    cols = len(_LANES)
    chip_y = area.bottom - 40
    chip_w = max(70, (right - left - (cols - 1) * 6) // cols)
    chips = []
    for ci, (sid, label) in enumerate(_LANES):
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


def _status_fill(status: BlockStatus) -> tuple[int, int, int]:
    if status == "dead":
        return (36, 20, 22)
    if status == "active":
        return (36, 32, 24)
    if status == "goal":
        return (24, 28, 40)
    if status == "partial":
        return (28, 34, 32)
    if status == "done":
        return (24, 30, 34)
    return (26, 28, 34)


def draw_research(
    surf: Any,
    schematic: ResearchSchematic,
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

    cur_lane = stage or schematic.stage
    header = title_f.render(
        f"Research map · {schematic.mode.upper()} · project flowchart",
        True,
        _TITLE,
    )
    surf.blit(header, (area.x + 12 if area else 12, (area.y + 8) if area else 8))
    sub = body_f.render(schematic.subtitle, True, _MUTED)
    surf.blit(sub, (area.x + 12 if area else 12, (area.y + 24) if area else 24))

    id_to_block = {b.id: b for b in schematic.blocks}

    def _lane_hot(bid: str) -> bool:
        blk = id_to_block.get(bid)
        if blk is None:
            return False
        if cur_lane in ("", None):
            return True
        return blk.lane == cur_lane or bid == selected

    # Edges under blocks
    for e in schematic.edges:
        ra = rects.get(e.src)
        rb = rects.get(e.dst)
        if ra is None or rb is None:
            continue
        a = (ra.centerx, ra.centery)
        bpt = (rb.centerx, rb.centery)
        hot = _lane_hot(e.src) or _lane_hot(e.dst) or selected in (e.src, e.dst)
        if e.kind == "dead":
            col = _EDGE_DEAD if hot else tuple(max(0, c - 40) for c in _EDGE_DEAD)
            width = 2 if hot else 1
        elif e.kind == "pivot":
            col = _EDGE_PIVOT if hot else tuple(max(0, c - 35) for c in _EDGE_PIVOT)
            width = 2 if hot else 1
        else:
            col = _EDGE_HOT if hot else _EDGE
            width = 2 if hot else 1
        # Elbow for dead branch (grow → cns_train) so it reads as a fork.
        if e.kind == "dead" and e.src == "grow" and e.dst == "cns_train":
            mid = (a[0], bpt[1])
            pygame.draw.lines(surf, col, False, [a, mid, bpt], width)
        elif e.kind == "pivot" and e.src == "cns_train":
            mid = (bpt[0], a[1])
            pygame.draw.lines(surf, col, False, [a, mid, bpt], width)
        else:
            pygame.draw.line(surf, col, a, bpt, width)
        if e.label:
            mx, my = (a[0] + bpt[0]) // 2, (a[1] + bpt[1]) // 2
            if e.kind == "dead":
                my = (a[1] + bpt[1]) // 2
            t = small_f.render(e.label, True, col)
            surf.blit(t, (mx - t.get_width() // 2, my - t.get_height() - 2))

    for b in schematic.blocks:
        r = rects.get(b.id)
        if r is None:
            continue
        dim = cur_lane not in ("", None) and b.lane != cur_lane and b.id != selected
        fill = _status_fill(b.status)
        if dim:
            fill = tuple(max(0, c - 12) for c in fill)
        accent = b.accent if not dim else tuple(max(0, c - 30) for c in b.accent)
        if b.id == selected:
            fill = tuple(min(255, c + 10) for c in fill)
        pygame.draw.rect(surf, fill, r, border_radius=4)
        border_w = 2 if (b.id == selected or b.live or b.status in ("dead", "goal", "active")) else 1
        # Dead branch: dashed feel via double thin border + X in corner
        pygame.draw.rect(surf, accent, r, border_w, border_radius=4)
        if b.status == "dead":
            pygame.draw.line(
                surf,
                accent,
                (r.right - 14, r.y + 6),
                (r.right - 6, r.y + 14),
                2,
            )
            pygame.draw.line(
                surf,
                accent,
                (r.right - 6, r.y + 6),
                (r.right - 14, r.y + 14),
                2,
            )
        elif b.live:
            pygame.draw.circle(surf, accent, (r.right - 8, r.y + 8), 3)

        title = small_f.render(b.title, True, accent)
        surf.blit(title, (r.x + 6, r.y + 4))
        yy = r.y + 4 + title.get_height() + 2
        for line in b.summary[:2]:
            lab = body_f.render(str(line), True, _BODY if not dim else _MUTED)
            surf.blit(lab, (r.x + 6, yy))
            yy += lab.get_height() + 1
        # Status tag
        tag = b.status.upper()
        tag_c = accent if not dim else _MUTED
        ts = body_f.render(tag, True, tag_c)
        if yy + ts.get_height() < r.bottom - 2:
            surf.blit(ts, (r.x + 6, r.bottom - ts.get_height() - 4))
        hits.append((r, b.id))

    for sid, label, cr in rects.get("__chips__", []):
        on = sid == cur_lane
        pygame.draw.rect(
            surf,
            (40, 44, 58) if on else (28, 30, 38),
            cr,
            border_radius=3,
        )
        pygame.draw.rect(
            surf,
            (180, 190, 210) if on else (80, 85, 100),
            cr,
            1,
            border_radius=3,
        )
        t = small_f.render(label, True, (220, 225, 235) if on else _MUTED)
        surf.blit(
            t,
            (cr.x + (cr.w - t.get_width()) // 2, cr.y + (cr.h - t.get_height()) // 2),
        )
        hits.append((cr, f"stage:{sid}"))

    if selected and selected in id_to_block and area is not None:
        blk = id_to_block[selected]
        pw = min(380, max(240, area.w // 3))
        ph = min(area.h - 80, 28 + 16 * (2 + len(blk.detail)))
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(_PANEL_BG)
        pygame.draw.rect(panel, _PANEL_EDGE, panel.get_rect(), 1)
        yy = 8
        ht = title_f.render(blk.title, True, blk.accent)
        panel.blit(ht, (10, yy))
        yy += ht.get_height() + 4
        st = body_f.render(
            f"{blk.status} · lane {blk.lane}",
            True,
            _MUTED,
        )
        panel.blit(st, (10, yy))
        yy += st.get_height() + 6
        for line in blk.detail:
            lt = body_f.render(str(line)[:58], True, _BODY)
            panel.blit(lt, (10, yy))
            yy += lt.get_height() + 2
            if yy > ph - 12:
                break
        px = area.right - pw - 10
        py = area.y + 40
        surf.blit(panel, (px, py))
        hits.append((pygame.Rect(px, py, pw, ph), selected))

    return hits


def hit_research(
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
    ids = [s for s, _ in _LANES]
    if cur not in ids:
        return ids[0]
    return ids[(ids.index(cur) + 1) % len(ids)]

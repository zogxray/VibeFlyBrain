#!/usr/bin/env python3
"""Render short README screenshots (and optional GIF) from GenomeShell.

Honest captures of the live UI — not mockups.

  python scripts/render_readme_shots.py
  python scripts/render_readme_shots.py --gif

Writes to docs/screenshots/ (tracked; see .gitignore allowlist under docs/).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
W, H = 1280, 720


def _boot_shell():
    # Prefer a predictable window before pygame init inside GenomeShell.
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,40")
    sys.path.insert(0, str(ROOT))
    import genome_shell as gs

    gs.WINDOW_W, gs.WINDOW_H = W, H
    # Force non-maximized boot size via display info stub.
    _real_info = gs.pygame.display.Info

    class _Info:
        current_w = W
        current_h = H

    gs.pygame.display.Info = lambda: _Info()  # type: ignore[assignment]
    shell = gs.GenomeShell()
    gs.pygame.display.Info = _real_info
    shell._maximized = False
    shell._reopen_display(W, H)
    shell.legend_mini = True
    if int(getattr(shell, "dock_w", 0) or 0) != 0:
        shell._layout(W, H)
    return gs, shell


def _pump(shell, n: int = 3) -> None:
    import pygame

    for _ in range(n):
        pygame.event.pump()
        shell._draw()
        shell.clock.tick(30)


def _save(shell, name: str) -> Path:
    import pygame

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    pygame.image.save(shell.screen, str(path))
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    return path


def _shot_sequence(gs, shell) -> list[Path]:
    paths: list[Path] = []

    # 1) 3D / tissue (or empty) — clear schematics
    shell.show_port_schema = False
    shell.show_brain_schema = False
    shell.show_grow_schema = False
    shell.show_research_schema = False
    shell._fit_scale()
    _pump(shell, 4)
    paths.append(_save(shell, "01-shell.png"))

    # 2) Research map
    shell.set_view_research_schema(True)
    _pump(shell, 3)
    paths.append(_save(shell, "02-map.png"))

    # 3) AI Brain schema
    shell.set_view_research_schema(False)
    shell.set_view_port_schema(True)
    _pump(shell, 3)
    paths.append(_save(shell, "03-ai-brain.png"))

    # 4) CNS Growing
    shell.set_view_port_schema(False)
    shell.set_view_grow_schema(True)
    _pump(shell, 3)
    paths.append(_save(shell, "04-grow.png"))

    return paths


def _write_gif(paths: list[Path], dest: Path, *, duration_ms: int = 1100) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        print(f"skip gif (Pillow needed): {exc}")
        return
    frames = []
    for p in paths:
        im = Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128)
        frames.append(im)
    if not frames:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dest,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gif", action="store_true", help="Also write docs/screenshots/tour.gif")
    args = ap.parse_args()

    gs, shell = _boot_shell()
    try:
        paths = _shot_sequence(gs, shell)
        if args.gif:
            _write_gif(paths, OUT / "tour.gif")
    finally:
        try:
            shell._save_ui_state()
        except Exception:
            pass
        gs.pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Third device in _REGISTRY gets panel UI without genome_shell edits.

Run:  python scripts/test_panel_third_device.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from device import StubBeaconDevice, TurretDevice, OcelliDevice, _REGISTRY  # noqa: E402
from device_render import (  # noqa: E402
    Bars,
    Hint,
    Meter,
    Toggle,
    handle_row_click,
    iter_panel_rows,
    panel_color,
    panel_title,
)


def main() -> int:
    assert "beacon" in _REGISTRY, "beacon missing from _REGISTRY"
    assert _REGISTRY["beacon"] is StubBeaconDevice

    devices = [
        OcelliDevice(enabled=True),
        TurretDevice(enabled=True),
        StubBeaconDevice(enabled=True, level=0.7),
    ]
    # Titles/colors without genome_shell kind switches.
    titles = [panel_title(d) for d in devices]
    colors = [panel_color(d) for d in devices]
    assert titles == ["Ocelli", "Turret", "Beacon"], titles
    assert colors[2] == (180, 140, 220), colors[2]

    pairs = iter_panel_rows(devices)
    kinds = {type(r).__name__ for _, r in pairs}
    assert Toggle.__name__ in kinds and Bars.__name__ in kinds and Hint.__name__ in kinds
    assert any(isinstance(r, Meter) for _, r in pairs)  # turret aim

    beacon_rows = [r for d, r in pairs if d.kind == "beacon"]
    assert any(isinstance(r, Toggle) and r.key == "enabled" for r in beacon_rows)
    assert any(isinstance(r, Bars) for r in beacon_rows)
    assert any(isinstance(r, Hint) for r in beacon_rows)

    # Click enable without knowing kind == "beacon" in shell.
    class Host:
        ocelli_on = True
        turret_on = True

        def _save_ui_state(self):
            pass

    host = Host()
    beacon = devices[2]
    assert beacon.enabled
    tog = next(r for r in beacon_rows if isinstance(r, Toggle) and r.key == "enabled")
    assert handle_row_click(host, tog, beacon)
    assert beacon.enabled is False

    levels = beacon.bar_levels(host)
    assert levels[0][0] == "A" and levels[0][1] == 0.7

    print(
        f"[panel] ok devices={len(devices)} rows={len(pairs)} "
        f"titles={titles} beacon_off={not beacon.enabled}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

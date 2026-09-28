#!/usr/bin/env bash
# Build MaleCNS soma + process clouds for genome_shell.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

echo "==> MaleCNS clouds (meta + SWC sample → malecns_*.npz)"
python3 "$ROOT/scripts/build_malecns_clouds.py"

echo "==> Done"
ls -lh "$ROOT/data/processed"/malecns_*.npz 2>/dev/null || true

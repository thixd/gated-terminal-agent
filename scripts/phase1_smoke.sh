#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate

echo "[check] python"
python --version

echo "[check] docker"
docker --version
docker info --format '{{.ServerVersion}}'

echo "[check] baseline agent"
python baseline_agent.py --state '{"cwd":"/tmp","stdout":"","stderr":""}'

echo "Phase 1 smoke checks passed."

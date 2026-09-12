#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker compose --env-file Configs/.env -f Configs/docker-compose.yml down
systemctl --user stop hyperdrivewave-llama.service 2>/dev/null || true
systemctl --user stop hyperdrivewave-resource-coordinator.service 2>/dev/null || true

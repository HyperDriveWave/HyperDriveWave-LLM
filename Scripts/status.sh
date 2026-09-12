#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker compose --env-file Configs/.env -f Configs/docker-compose.yml ps
systemctl --user status hyperdrivewave-llama.service --no-pager 2>/dev/null || true
systemctl --user status hyperdrivewave-resource-coordinator.service --no-pager 2>/dev/null || true

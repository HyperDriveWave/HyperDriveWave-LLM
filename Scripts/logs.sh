#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ $# -eq 0 ]; then
  docker compose --env-file Configs/.env -f Configs/docker-compose.yml logs -f
else
  docker compose --env-file Configs/.env -f Configs/docker-compose.yml logs -f "$@"
fi

#!/usr/bin/env bash
set -euo pipefail

test $# -eq 1 || { echo "usage: Scripts/restore.sh <backup.tgz>"; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tar -xzf "$1" -C "$ROOT"
echo "restored from $1"


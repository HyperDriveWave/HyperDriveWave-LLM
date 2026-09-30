#!/usr/bin/env bash
# 统一的 docker compose 入口 —— 手敲命令也走这里，免得漏文件。
#
# 为什么需要它
# ────────────
# 2026-10-01 我自己踩了一次：换 qa-api 镜像时手敲
#
#     docker compose --env-file Configs/.env -f Configs/docker-compose.yml \
#       --profile base ... up -d --force-recreate hdw-qa-api
#
# **漏了 `-f Configs/docker-compose.rebuild-gpu.yml`**。因为 qa-api 声明了
# `depends_on: hdw-rag`，compose 把 hdw-rag 也一起按**没有 GPU 的基础配置**
# 重建了。后果：
#
#   · 容器状态一片正常，`docker compose ps` 全绿
#   · `rag` 的 `/health` 也返回 200
#   · 但一检索就是 503 —— 没有 GPU 时模型加载失败，索引打开也出问题
#
# 也就是说**配置悄悄退化成另一个部署**，而所有"看状态"的手段都看不出来。
# `start.sh` 会自动带齐文件，但手敲的命令不会。与其要求每个人每次都记得，
# 不如把手敲的那条路也收进这里：
#
#     bash Scripts/compose.sh ps
#     bash Scripts/compose.sh up -d hdw-rag
#     bash Scripts/compose.sh logs -f hdw-qa-api
#     bash Scripts/compose.sh down
#
# 其余参数原样透传给 docker compose。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$ROOT/Configs/docker-compose.yml"
ENV_FILE="$ROOT/Configs/.env"

[ -f "$ENV_FILE" ] || { echo "缺少 $ENV_FILE" >&2; exit 1; }

cd "$ROOT"
set -a
. "$ENV_FILE"
set +a

FILES=(-f "$COMPOSE_FILE")

# 与 start.sh 用**同一个判据**：HDW_RAG_DEVICE=cuda 就意味着这份部署叠了
# GPU 覆盖文件。两处判据必须一致，否则 start.sh 起的和手敲起的会是两套配置。
if [ "${HDW_RAG_DEVICE:-cpu}" = "cuda" ]; then
  FILES+=(-f "$ROOT/Configs/docker-compose.rebuild-gpu.yml")
fi

PROFILE_ARGS=()
read -r -a _profiles <<< "${HDW_COMPOSE_PROFILES:-base knowledge web}"
for p in "${_profiles[@]}"; do
  [ -n "$p" ] && PROFILE_ARGS+=(--profile "$p")
done

if [ "${HDW_COMPOSE_QUIET:-0}" != "1" ]; then
  echo "[compose] 文件：${FILES[*]}" >&2
  echo "[compose] profile：${_profiles[*]}" >&2
fi

exec docker compose --env-file "$ENV_FILE" "${FILES[@]}" "${PROFILE_ARGS[@]}" "$@"

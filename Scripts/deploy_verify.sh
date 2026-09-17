#!/usr/bin/env bash
# HyperDriveWave 部署验收 / 冒烟测试。
#
# 用法：
#   bash Scripts/deploy_verify.sh            # 常规验收
#   bash Scripts/deploy_verify.sh --quick    # 跳过端到端问答（只查链路，快）
#   bash Scripts/deploy_verify.sh --deep     # 额外做重启演练（会短暂中断服务）
#
# 为什么不用 healthcheck.sh：
#   healthcheck.sh 是**日常巡检**，它查的 ${HDW_LLM_PORT:-8000} 是旧 FreeToken 引擎的端口，
#   不是真实 llama 的 1919，所以一直误报 llm unavailable。
# 而这个脚本要回答的是「这套东西真的能用吗」，因此**不信任 /health**：
#   · RAG 的 /health 返回的是写死的静态字典（只查路径存在性，不加载模型）
#   · qa-api 的 /health 里 "status":"ok" 也是硬编码字面量
#   · CUDA kernel 不匹配时，llama 的 /health 照样通过
#   这些都必须靠**真的调用一次**才能覆盖。

set -uo pipefail   # 故意不开 -e：验收要把所有项跑完再汇总，不能第一项失败就退出

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/detect.sh
. "$SCRIPT_DIR/lib/detect.sh"
# shellcheck source=lib/models.sh
. "$SCRIPT_DIR/lib/models.sh"

ENV_FILE="$HDW_ROOT/Configs/.env"
COMPOSE_FILE="$HDW_ROOT/Configs/docker-compose.yml"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

DEEP=0
QUICK=0
LOGIN_CODE="${HDW_VERIFY_LOGIN_CODE:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --deep)  DEEP=1; shift ;;
    --quick) QUICK=1; shift ;;
    --login-code) LOGIN_CODE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

PASS=0; FAIL=0; WARN=0
declare -a FAILED_ITEMS=()

pass() { PASS=$((PASS + 1)); printf '  %s✓%s %s\n' "$_C_GRN" "$_C_RESET" "$*"; }
fail() { FAIL=$((FAIL + 1)); FAILED_ITEMS+=("$*"); printf '  %s✗%s %s\n' "$_C_RED" "$_C_RESET" "$*"; }
vwarn() { WARN=$((WARN + 1)); printf '  %s!%s %s\n' "$_C_YEL" "$_C_RESET" "$*"; }
section() { printf '\n%s【%s】%s\n' "$_C_BLU" "$*" "$_C_RESET"; }

LLAMA_PORT="${HDW_LLAMA_PORT:-1919}"
LLAMA_URL="http://127.0.0.1:$LLAMA_PORT"
RAG_PORT="${HDW_RAG_PORT:-8001}"
RAG_URL="http://127.0.0.1:$RAG_PORT"
QA_PORT="${HDW_QA_API_PORT:-8080}"
QA_URL="http://127.0.0.1:$QA_PORT"
API_PORT="${HDW_API_PORT:-8095}"
API_URL="http://127.0.0.1:$API_PORT"
MINERU_PORT="${HDW_MINERU_PORT:-8002}"
MINERU_URL="http://127.0.0.1:$MINERU_PORT"
# Neo4j HTTP。原来在检查里写死 7475，改了端口就会出现
# 「只有 Neo4j 这一项红、其余全绿」，被当成"Neo4j 坏了"往错方向排查。
NEO4J_HTTP_PORT="${HDW_NEO4J_HTTP_PORT:-7475}"

json_get() { python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
for k in sys.argv[1].split('.'):
    if isinstance(d, dict): d=d.get(k)
    elif isinstance(d, list):
        try: d=d[int(k)]
        except Exception: sys.exit(1)
    else: sys.exit(1)
print(d if d is not None else '')
" "$1" 2>/dev/null; }

# ═══ 0. 等待服务就绪 ══════════════════════════════════════════
# 部署刚结束时容器可能还在启动。不等待的话，每个新建的容器都会报一次假失败
# （踩过：deploy.sh 跑完立刻验收，RAG 报了"不通"，十几秒后自己就好了）。
section "等待服务就绪"

WAIT_TIMEOUT="${HDW_VERIFY_WAIT:-180}"
# 先算好 webui 的地址再建数组：变量引用写在数组字面量里会拿到当时还没赋的值
BIND_WAIT="${HDW_WEBUI_BIND:-127.0.0.1}"
[ "$BIND_WAIT" = "0.0.0.0" ] && BIND_WAIT="127.0.0.1"
WEBUI_PORT_WAIT="${HDW_WEBUI_PORT:-3000}"

declare -A _wait_urls=(
  ["llama"]="$LLAMA_URL/health"
  ["rag"]="$RAG_URL/health"
  ["qa-api"]="$QA_URL/health"
  ["hdw-api"]="$API_URL/health"
  ["mineru"]="$MINERU_URL/health"
  ["webui"]="http://$BIND_WAIT:$WEBUI_PORT_WAIT/"
)

declare -A _wait_ok=()
_deadline=$((SECONDS + WAIT_TIMEOUT))
while [ "$SECONDS" -lt "$_deadline" ]; do
  _all=1
  for name in "${!_wait_urls[@]}"; do
    [ -n "${_wait_ok[$name]:-}" ] && continue
    if http_ok "${_wait_urls[$name]}" 4; then
      _wait_ok[$name]=1
    else
      _all=0
    fi
  done
  [ "$_all" = "1" ] && break
  sleep 2
done

for name in "${!_wait_urls[@]}"; do
  if [ -n "${_wait_ok[$name]:-}" ]; then
    pass "$name 已就绪"
  else
    # 这里只报警告，具体失败原因留给后面各节按服务细查
    vwarn "$name 在 ${WAIT_TIMEOUT}s 内未就绪（${_wait_urls[$name]}）"
  fi
done

# ═══ 1. 容器状态 ══════════════════════════════════════════════
section "容器"

CONTAINER_JSON="$(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --profile base --profile knowledge --profile web ps --format json 2>/dev/null)"
if [ -z "$CONTAINER_JSON" ]; then
  fail "读不到容器状态（docker compose ps 无输出）"
else
  BAD_CONTAINERS="$(printf '%s\n' "$CONTAINER_JSON" | python3 -c "
import json,sys
bad=[]
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    try: c=json.loads(line)
    except Exception: continue
    name=c.get('Service') or c.get('Name','?')
    state=(c.get('State') or '').lower()
    health=(c.get('Health') or '').lower()
    # Restarting/Exited 是最典型的'看着起来了实际在崩'
    if state not in ('running',): bad.append(f'{name}: state={state}')
    elif health and health not in ('healthy',''): bad.append(f'{name}: health={health}')
print('\n'.join(bad))
" 2>/dev/null)"
  COUNT="$(printf '%s\n' "$CONTAINER_JSON" | grep -c . || echo 0)"
  if [ -z "$BAD_CONTAINERS" ]; then
    pass "$COUNT 个容器全部 running"
  else
    while IFS= read -r line; do [ -n "$line" ] && fail "容器异常 — $line"; done <<< "$BAD_CONTAINERS"
  fi
fi

# ═══ 2. 基础设施 ══════════════════════════════════════════════
section "基础设施"

if docker exec hyperdrivewave-hdw-postgres-1 pg_isready -U "${POSTGRES_USER:-hdw}" >/dev/null 2>&1; then
  pass "Postgres 就绪"
else
  fail "Postgres 不可用"
fi

if [ "$(docker exec hyperdrivewave-hdw-redis-1 redis-cli -a "${REDIS_PASSWORD:-}" ping 2>/dev/null | tr -d '\r')" = "PONG" ]; then
  pass "Redis 就绪"
else
  fail "Redis 不可用"
fi

NEO_OK=0
# NEO4J_AUTH 用的是 neo4j 自己的 `用户/密码` 约定，而 curl -u 要 `用户:密码`。
# 直接把 `用户/密码` 传给 -u 会让 curl 当成"只有用户名"，转而交互式索要密码。
NEO_AUTH_RAW="${NEO4J_AUTH:-neo4j/change_me}"
NEO_CRED="${NEO_AUTH_RAW%%/*}:${NEO_AUTH_RAW#*/}"
for i in 1 2 3 4 5; do
  # 先落变量再 grep（不用管道）：pipefail 下 `curl | grep -q` 会因 SIGPIPE 误判
  NEO_RESP="$(curl -fsS --max-time 5 -u "$NEO_CRED" \
      -H 'Content-Type: application/json' \
      -d '{"statements":[{"statement":"RETURN 1"}]}' \
      "http://127.0.0.1:${NEO4J_HTTP_PORT}/db/neo4j/tx/commit" 2>/dev/null)"
  if grep -q '"errors":\[\]' <<< "$NEO_RESP"; then
    NEO_OK=1; break
  fi
  sleep 2
done
[ "$NEO_OK" = "1" ] && pass "Neo4j 就绪" || fail "Neo4j 不可用（凭据取自 NEO4J_AUTH）"

# ═══ 3. llama.cpp（重点：不能只看 /health）════════════════════
section "llama.cpp 本地推理"

# 无 GPU 的部署会显式停用本地推理（deploy.sh 写 HDW_SKIP_LOCAL_LLM=true）。
# 这时候报一堆"llama 不通"是误导——它是**按配置就该没在跑**。
# 改为检查"在线 API 是否配好"，那才是这类部署真正该验的东西。
if [ "${HDW_SKIP_LOCAL_LLM:-false}" = "true" ]; then
  info "本地推理已停用（HDW_SKIP_LOCAL_LLM=true），跳过 llama 检查"
  if [ -n "${HDW_ONLINE_LLM_API_KEY:-}" ]; then
    pass "在线 API 已配置：${HDW_ONLINE_LLM_BASE_URL:-<未设>} / ${HDW_ONLINE_LLM_MODEL:-<未设>}"
    _llama_skipped=1
  else
    fail "停用了本地推理，但 HDW_ONLINE_LLM_API_KEY 没配 —— 提问会返回 503"
    _llama_skipped=1
  fi
fi

if [ "${_llama_skipped:-0}" != "1" ]; then
if ! http_ok "$LLAMA_URL/health" 10; then
  fail "llama /health 不通（$LLAMA_URL）"
else
  pass "llama /health 通"

  # 实际加载的是哪个模型
  PROPS="$(curl -fsS --max-time 10 "$LLAMA_URL/props" 2>/dev/null)"
  PROPS_MODEL="$(printf '%s' "$PROPS" | json_get model_path)"
  if [ -n "$PROPS_MODEL" ]; then
    pass "已加载模型：$(basename "$PROPS_MODEL")"
  else
    vwarn "读不到 /props 的 model_path"
  fi

  # ── 真正跑一次生成 ──
  # 这是唯一能证明 CUDA kernel 匹配的检查：kernel 不匹配时 /health 会通过，
  # 但第一次 kernel launch 就崩（no kernel image is available）。
  # 冷加载 12G GGUF 可能要 30-90s，所以超时给足。
  info "  跑一次真实生成（首次可能要等模型加载）…"
  # max_tokens 不能太小：Qwen3.8 是思考模型，思考内容也占 completion_tokens。
  # 给 16 的话会被 reasoning_content 吃光，content 为空——看着像失败其实成功了。
  GEN="$(curl -fsS --max-time 300 -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"只回答两个字：正常"}],"max_tokens":96,"temperature":0,"stream":false}' \
    "$LLAMA_URL/v1/chat/completions" 2>/dev/null)"
  # 判定"生成了"的依据是**产出了 token**，而不是 content 非空：
  # 思考模型可能只填 reasoning_content。这才是 kernel 能否运行的证据。
  GEN_TEXT="$(printf '%s' "$GEN" | json_get choices.0.message.content)"
  GEN_REASON="$(printf '%s' "$GEN" | json_get choices.0.message.reasoning_content)"
  GEN_TOKENS="$(printf '%s' "$GEN" | json_get usage.completion_tokens)"
  if [ "${GEN_TOKENS:-0}" -gt 0 ] 2>/dev/null; then
    pass "生成正常（产出 $GEN_TOKENS 个 token）$(printf '%s' "${GEN_TEXT:-$GEN_REASON}" | tr -d '\n' | head -c 50)"
    TPS="$(printf '%s' "$GEN" | json_get timings.predicted_per_second)"
    if [ -n "$TPS" ]; then
      # 走 GPU 时 27B IQ3_S 大约 20-40 tok/s；个位数基本可以断定掉到 CPU 了
      if awk -v t="$TPS" 'BEGIN { exit !(t > 20) }'; then
        pass "解码速度 ${TPS} tok/s（约 ${TPS%.*}，走 GPU）"
      elif awk -v t="$TPS" 'BEGIN { exit !(t >= 5) }'; then
        vwarn "解码速度仅 ${TPS} tok/s，疑似未走 GPU（检查 --gpu-layers 与显存）"
      else
        fail "解码速度 ${TPS} tok/s，几乎确定跑在 CPU 上"
      fi
    fi
  else
    fail "生成失败 —— 一个 token 都没产出。这是 kernel 不匹配/模型加载失败的典型表现"
  fi

  # 日志窗口按**服务实际启动时间**算，不能写死"最近 N 分钟"：
  # 服务可能是几小时前起的且一直没重启，固定窗口会扫不到启动日志，
  # 导致 MTP 之类的检查全部误报为"没启用"。
  LLAMA_SINCE="$(systemctl --user show hyperdrivewave-llama.service -p ActiveEnterTimestamp --value 2>/dev/null)"
  [ -n "$LLAMA_SINCE" ] || LLAMA_SINCE="-30min"

  # ── 日志扫描：kernel 不匹配最直接的证据 ──
  LOGBAD="$(journalctl --user -u hyperdrivewave-llama.service --since "$LLAMA_SINCE" --no-pager 2>/dev/null \
    | grep -iE "no kernel image is available|CUDA error|failed to load model|error loading model|out of memory" | tail -3)"
  if [ -n "$LOGBAD" ]; then
    fail "llama 日志有错误："
    printf '%s\n' "$LOGBAD" | sed 's/^/      /'
  else
    pass "llama 日志无 kernel/加载错误"
  fi

  # ── 崩溃重启循环 ──
  NR="$(systemctl --user show hyperdrivewave-llama.service -p NRestarts --value 2>/dev/null)"
  if [ -n "$NR" ] && [ "$NR" != "0" ]; then
    vwarn "llama 服务重启过 $NR 次（若非本次操作为 0，说明存在崩溃重启）"
  else
    pass "llama 服务无重启记录"
  fi

  # ── MTP 是否真的启用 ──
  MTP_JOURNAL="$(journalctl --user -u hyperdrivewave-llama.service --since "$LLAMA_SINCE" --no-pager 2>/dev/null)"
  if grep -q "MTP: enabled" <<< "$MTP_JOURNAL"; then
    pass "MTP 已启用（--spec-type draft-mtp + MTP draft context）"
  elif grep -q '"mtp_enabled": *true' "$HDW_ROOT/HDW_Runtime/model-config/config.json" 2>/dev/null; then
    vwarn "配置要求启用 MTP 但日志里没有 — 后端可能是旧版 llama.cpp（不支持 --spec-type），
     或模型 GGUF 不含 nextn 张量（需要 *-mtp.gguf 那个变体）"
  else
    pass "MTP 未启用（配置里也没要求）"
  fi

  # ── --timeout 不能是 0 ──
  # llama.cpp 把它透传给 cpp-httplib 的 set_read_timeout()，而 httplib 把
  # (0,0) 解释成 poll(...,0)——「立即返回」，不是「永不超时」。读大 body 时
  # 只要 socket 缓冲区恰好空一次就判短读，返回空 body 的 400，表现为问答
  # 偶发 `503: LLM generation failed: LLM HTTP 400: `。**间歇、无日志**，
  # 排查时极容易走成「网络/GPU 问题」。这里在验收阶段直接把它挡下来。
  # 扫**全部**匹配进程而不是取第一个：pgrep -f 可能同时命中父 shell 和真正
  # 的服务进程，只看第一条会读到空值或别人的参数。任一带 0 就算失败。
  _to="" ; _zero=0
  for _pid in $(pgrep -f 'llama-server' 2>/dev/null); do
    # 读 /proc 的 cmdline 最可靠：ps 会按终端宽度截断参数
    _v="$(tr '\0' '\n' < "/proc/$_pid/cmdline" 2>/dev/null \
          | awk '/^--timeout$/{getline; print; exit}')"
    if [ -n "$_v" ]; then
      _to="$_v"
      [ "$_v" = "0" ] && _zero=1
    fi
  done
  if [ "$_zero" = "1" ]; then
    fail "llama-server 带着 --timeout 0 在跑 —— 会导致问答偶发 503（空 body 400）。
     改 Configs/.env 的 HDW_LLAMA_TIMEOUT=3600 后重启：systemctl --user restart hyperdrivewave-llama.service"
  elif [ -n "$_to" ]; then
    pass "llama --timeout $_to（非 0，正常）"
  else
    vwarn "读不到 llama-server 的 --timeout（进程在跑的话应能读到）"
  fi
fi
fi   # ← 结束「本地推理是否停用」的分支

# ═══ 4. 本机 RAG ══════════════════════════════════════════════
section "本机 RAG 服务"

RAG_HEALTH="$(curl -fsS --max-time 10 "$RAG_URL/health" 2>/dev/null)"
if [ -z "$RAG_HEALTH" ]; then
  fail "RAG /health 不通（$RAG_URL）"
else
  # 注意：这个 /health 的 status 是写死的 "ok"，只能看它的路径存在性字段
  for f in bge_m3_path_exists reranker_path_exists; do
    v="$(printf '%s' "$RAG_HEALTH" | json_get "$f")"
    [ "$v" = "True" ] && pass "RAG $f" || fail "RAG $f=$v（模型路径不对）"
  done
  z="$(printf '%s' "$RAG_HEALTH" | json_get zvec_index_exists)"
  [ "$z" = "True" ] && pass "RAG zvec 索引存在" || vwarn "RAG zvec 索引不存在（没灌过知识库？）"

  # ── 真正调一次 /embed：/health 通过不代表模型能加载（懒加载）──
  info "  调用 /embed（首次要冷加载 bge-m3，可能 10-30s）…"
  EMB="$(curl -fsS --max-time 300 -H 'Content-Type: application/json' \
    -d '{"texts":["部署自检"]}' "$RAG_URL/embed" 2>/dev/null)"
  DIM="$(printf '%s' "$EMB" | json_get dim)"
  VECLEN="$(printf '%s' "$EMB" | python3 -c "
import json,sys
try: d=json.load(sys.stdin); print(len(d.get('vectors',[[]])[0]))
except Exception: print(0)
" 2>/dev/null)"
  if [ "${VECLEN:-0}" -gt 0 ] && [ "${DIM:-0}" -gt 0 ]; then
    pass "嵌入正常：dim=$DIM 向量长度=$VECLEN"
    [ "$VECLEN" = "$DIM" ] || fail "向量维度与声明不符（$VECLEN ≠ $DIM）"
  else
    fail "嵌入失败 —— bge-m3 加载不起来（localhost 下模型文件在不在？）"
  fi

  # ── /rerank ──
  # documents 必须是对象数组 {id, text, metadata}，传字符串数组会被 422 拒掉
  RR="$(curl -fsS --max-time 120 -H 'Content-Type: application/json' \
    -d '{"query":"凝汽器水位","documents":[{"id":"a","text":"凝汽器液位高报警"},{"id":"b","text":"汽轮机轴封系统作用"}],"top_k":2}' \
    "$RAG_URL/rerank" 2>/dev/null)"
  RRLEN="$(printf '%s' "$RR" | python3 -c "
import json,sys
try: print(len(json.load(sys.stdin).get('results',[])))
except Exception: print(0)
" 2>/dev/null)"
  [ "${RRLEN:-0}" -gt 0 ] && pass "重排正常（返回 $RRLEN 条）" || fail "重排失败 —— bge-reranker 加载不起来"
fi

# ═══ 5. 远端 RAG（配了才查）═══════════════════════════════════
section "远端 RAG 节点"

REMOTE_URLS="${HDW_RAG_REMOTE_URLS:-}"
if [ -z "$REMOTE_URLS" ]; then
  pass "未配置远端 RAG（只用本机），符合预期"
else
  IFS=',' read -r -a _urls <<< "$REMOTE_URLS"
  for u in "${_urls[@]}"; do
    u="${u%/}"; [ -n "$u" ] || continue
    if ! http_ok "$u/health" 6; then
      # 远端不可用是致命配置问题的信号：QA 每个请求都会先吃一次连接超时
      fail "远端 RAG $u 不通（HDW_RAG_REMOTE_URLS 里配了但连不上，问答会每次超时）"
      continue
    fi
    RV="$(curl -fsS --max-time 300 -H 'Content-Type: application/json' \
      -d '{"texts":["远端自检"]}' "$u/embed" 2>/dev/null | json_get dim)"
    if [ -n "$RV" ] && [ "$RV" -gt 0 ] 2>/dev/null; then
      pass "远端 RAG $u 可用（dim=$RV）"
    else
      fail "远端 RAG $u /health 通但 /embed 失败（可能是 CUDA 不可用或模型缺失）"
    fi
  done
fi

# ═══ 6. MinerU ════════════════════════════════════════════════
section "MinerU 文档解析"

MINERU_HEALTH="$(curl -fsS --max-time 10 "$MINERU_URL/health" 2>/dev/null)"
if [ -z "$MINERU_HEALTH" ]; then
  vwarn "MinerU 不可用（不灌新文档的话不影响问答）"
else
  pass "MinerU /health：$(printf '%s' "$MINERU_HEALTH" | json_get status)"
  # 模型是在**镜像构建期**下到容器内的，不在宿主挂载里。
  # 漏了这一步会在真正解析文档时才发现，所以这里直接查。
  if docker exec hyperdrivewave-hdw-mineru-1 test -d /root/.cache/modelscope 2>/dev/null; then
    pass "MinerU 镜像内已含模型（/root/.cache/modelscope）"
  else
    fail "MinerU 镜像内没有模型 —— 镜像构建时的 mineru-models-download 没成功，解析会失败"
  fi
fi

# ═══ 7. WebUI ═════════════════════════════════════════════════
section "WebUI"

BIND="${HDW_WEBUI_BIND:-127.0.0.1}"
[ "$BIND" = "0.0.0.0" ] && BIND="127.0.0.1"
WEBUI_PORT="${HDW_WEBUI_PORT:-3000}"
WEB_BODY="$(curl -fsS --max-time 10 "http://$BIND:$WEBUI_PORT/" 2>/dev/null || true)"
if [ -n "$WEB_BODY" ]; then
  # 只看 200 不够：nginx 配错指到别的目录也会 200。
  # 用 here-string 而不是 `printf | grep -q`：grep -q 一匹配就退出，
  # 上游 printf 收到 SIGPIPE 退出码 141，而脚本开了 pipefail，
  # 整条管道会被判失败——大页面（200KB+）上必然踩到，小字符串上则不会。
  if grep -qiE "<!doctype html|<html" <<< "$WEB_BODY" &&
     grep -qiE "<title>|hyperdrivewave" <<< "$WEB_BODY"; then
    pass "WebUI 可访问：http://$BIND:$WEBUI_PORT"
  else
    vwarn "WebUI 返回内容不像预期页面（可能是 nginx 指错了目录）"
  fi
else
  fail "WebUI 不可访问：http://$BIND:$WEBUI_PORT"
fi

TLS_PORT="${HDW_WEBUI_TLS_PORT:-8443}"
if [ -f "$HDW_ROOT/HDW_Frontend/FRP/certs/server.crt" ]; then
  curl -fsSk --max-time 10 "https://$BIND:$TLS_PORT/" -o /dev/null 2>/dev/null \
    && pass "WebUI TLS 可访问：https://$BIND:$TLS_PORT" \
    || fail "WebUI TLS 不可访问（端口 $TLS_PORT）"
else
  vwarn "找不到 TLS 证书，跳过 HTTPS 检查"
fi

# ═══ 8. QA API ════════════════════════════════════════════════
section "QA API"

QA_HEALTH="$(curl -fsS --max-time 20 "$QA_URL/health" 2>/dev/null)"
if [ -z "$QA_HEALTH" ]; then
  fail "QA API /health 不通（$QA_URL）"
else
  # 顶层 status 是硬编码的 "ok"，必须看三个子对象
  for sub in rag graph llm; do
    sv="$(printf '%s' "$QA_HEALTH" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin).get(sys.argv[1]) or {}
    print(d.get('status') or d.get('ok') or json.dumps(d, ensure_ascii=False)[:80])
except Exception as e: print('读取失败')
" "$sub" 2>/dev/null)"
    case "$sv" in
      ok|OK|true|True) pass "QA API 子项 $sub 正常" ;;
      *) vwarn "QA API 子项 $sub：$sv" ;;
    esac
  done
fi

# ═══ 8.5 对外问答 API（给别的项目调用的那个）══════════════════
section "对外问答 API"

API_HEALTH="$(curl -fsS --max-time 20 "$API_URL/health" 2>/dev/null)"
if [ -z "$API_HEALTH" ]; then
  fail "对外问答 API /health 不通（$API_URL）"
else
  KEY_OK="$(printf '%s' "$API_HEALTH" | json_get key_configured)"
  case "$KEY_OK" in
    True|true) pass "对外问答 API 正常，已配置调用方密钥" ;;
    *) vwarn "对外问答 API 起来了但没配 HDW_API_KEY —— 接口会拒绝所有调用" ;;
  esac
  # 端口通只说明进程活着，不说明门是关着的。这条才验证鉴权真的生效：
  # 无密钥调用必须 401，返回 200 就意味着这个端口对局域网敞开了。
  _api_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    -X POST "$API_URL/api/v1/ask" -H 'Content-Type: application/json' \
    -d '{"question":"ping"}' 2>/dev/null)"
  case "$_api_code" in
    401) pass "无密钥调用被拒（401）" ;;
    000) vwarn "无密钥调用无响应" ;;
    *)   fail "无密钥调用返回 $_api_code（应为 401）—— 接口没有正确实施鉴权" ;;
  esac
fi

# 服务间入口：错误密钥必须被拒。两侧 HDW_API_INTERNAL_KEY 不一致时，
# hdw-api 只会给调用方一个 502，看不出是配置问题，所以在这里先挑出来。
# 用错误密钥发**合法请求体**——FastAPI 先校验 body 再进函数，
# body 不合法的话拿到的是 422，证明不了鉴权这一层。
_api_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
  -X POST "$QA_URL/internal/qa/query" -H 'Content-Type: application/json' \
  -H 'X-HDW-Internal-Key: deliberately-wrong' \
  -d '{"question":"ping"}' 2>/dev/null)"
case "$_api_code" in
  401) pass "服务间入口拒绝错误密钥（401）" ;;
  503) vwarn "服务间入口未启用（HDW_API_INTERNAL_KEY 未配置）" ;;
  000) vwarn "服务间入口无响应" ;;
  *)   fail "服务间入口对错误密钥返回 $_api_code（应为 401）" ;;
esac

# ═══ 9. 模型配置一致性 ════════════════════════════════════════
section "模型配置一致性"

CONFIG_JSON="$HDW_ROOT/HDW_Runtime/model-config/config.json"
CFG_MODEL="$(python3 -c "
import json,sys
try: print(json.load(open(sys.argv[1]))['local']['model'])
except Exception: print('')
" "$CONFIG_JSON" 2>/dev/null)"
ENV_MODEL="$(env_get "$ENV_FILE" HDW_LOCAL_LLM_MODEL || true)"

if [ -z "$CFG_MODEL" ]; then
  fail "读不到 config.json 的 local.model（$CONFIG_JSON）"
else
  # config.json 是真正的事实源（llama/start.sh 与 resource_coordinator 都读它），
  # .env 的 HDW_LOCAL_LLM_MODEL 只影响 qa-api 显示。两者分叉会让
  # 「界面显示的模型」和「实际跑的模型」不一致。
  if [ -n "$PROPS_MODEL" ] && [ "$(basename "$PROPS_MODEL")" = "$CFG_MODEL" ]; then
    pass "实际加载的模型与 config.json 一致：$CFG_MODEL"
  elif [ -n "$PROPS_MODEL" ]; then
    fail "实际加载 $(basename "$PROPS_MODEL")，但 config.json 写的是 $CFG_MODEL"
  fi
  if [ -n "$ENV_MODEL" ] && [ "$ENV_MODEL" != "$CFG_MODEL" ]; then
    vwarn ".env HDW_LOCAL_LLM_MODEL=$ENV_MODEL 与 config.json=$CFG_MODEL 不一致（只影响界面显示）"
  fi
fi

# ═══ 10. 模型权重完整性 ══════════════════════════════════════
section "模型权重完整性"

while IFS=$'\t' read -r kind repo file rel want; do
  [ -n "${kind:-}" ] || continue
  st="$(model_status "$kind" "$repo" "$file" "$rel" "$want")"
  case "$st" in
    OK)       pass "$rel" ;;
    MISSING)  fail "$rel 缺失（跑 bash Scripts/fetch_models.sh 补）" ;;
    BAD_SIZE) fail "$rel 大小不符（下载可能被中断，重跑 fetch_models.sh 会续传）" ;;
  esac
done < <(model_manifest)

# ═══ 11. Linger ═══════════════════════════════════════════════
section "开机自启"

if linger_enabled; then
  pass "linger 已开启，重启后用户级服务会自动恢复"
else
  # 这是「部署完看着好、重启就没了」的元凶，现有脚本零覆盖
  vwarn "linger 未开启：重启机器后 llama/协调器不会自启，要先登录一次。
     修复：sudo loginctl enable-linger $(whoami)"
fi

# ═══ 12. 端到端问答 ═══════════════════════════════════════════
section "端到端问答"

if [ "$QUICK" = "1" ]; then
  vwarn "已跳过（--quick）"
else
  info "  提问中（这一条会同时用到 LLM + RAG + 重排 + 图谱 + MCP，可能要 10-60s）…"
  # /qa/query 和 /v1/chat/completions 都要求登录会话，没有免鉴权的端到端入口。
  # 所以这里用 auth.csv 里的账号自检——**只取 code 用于换会话，绝不打印它**。
  SESSION=""
  WHO=""
  if [ -z "$LOGIN_CODE" ] && [ -f "$HDW_ROOT/HDW_Security/auth/auth.csv" ]; then
    LOGIN_CODE="$(python3 - "$HDW_ROOT/HDW_Security/auth/auth.csv" <<'PY' 2>/dev/null
import csv, sys
with open(sys.argv[1], encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        code = (row.get("code") or "").strip()
        if code:
            print(code); break
PY
)"
  fi

  if [ -n "$LOGIN_CODE" ]; then
    LOGIN_RESP="$(curl -fsS --max-time 20 -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"code": sys.argv[1]}))' "$LOGIN_CODE")" \
      "$QA_URL/auth/login" 2>/dev/null)"
    SESSION="$(printf '%s' "$LOGIN_RESP" | json_get session)"
    WHO="$(printf '%s' "$LOGIN_RESP" | python3 -c "
import json,sys
try:
    u=json.load(sys.stdin).get('user') or {}
    print(u.get('name') or u.get('role') or '')
except Exception: print('')
" 2>/dev/null)"
    [ -n "$SESSION" ] && dim "  已用本地账号登录自检${WHO:+（$WHO）}，登录码未记录"
  fi

  if [ -z "$SESSION" ]; then
    vwarn "拿不到登录会话，跳过端到端问答。
     要跑这一项：bash Scripts/deploy_verify.sh --login-code <auth.csv 里的 code>"
  else
    # 不要传 conversation_id：服务端会校验它存在，随便编一个会 404
    # "conversation not found"。省略则自动新建。
    E2E="$(curl -fsS --max-time 300 -H 'Content-Type: application/json' \
      -H "Authorization: Bearer ${HDW_INTERNAL_API_KEY:-local-dev-key}" \
      -H "X-HDW-Session: $SESSION" \
      -d '{"question":"轴封系统的作用是什么？","inference_mode":"offline"}' \
      "$QA_URL/qa/query" 2>/dev/null)"
    ANS="$(printf '%s' "$E2E" | json_get answer)"
    if [ -n "$ANS" ]; then
      pass "问答返回 $(printf '%s' "$ANS" | wc -c) 字节：$(printf '%s' "$ANS" | tr -d '\n' | head -c 70)…"
      read -r EVID RAG_BACKEND RAG_URL_USED < <(printf '%s' "$E2E" | python3 -c "
import json,sys
try:
    r = json.load(sys.stdin).get('rag') or {}
    print(r.get('deduplicated_count') or 0, r.get('backend') or '-', r.get('url') or '-')
except Exception:
    print('0 - -')
" 2>/dev/null)
      if [ "${EVID:-0}" -gt 0 ] 2>/dev/null; then
        # backend=remote 说明请求真的走了远端节点，这是远端 RAG 配置生效的直接证据
        pass "检索命中 $EVID 条证据（后端 $RAG_BACKEND${RAG_URL_USED:+ @ $RAG_URL_USED}）"
      else
        vwarn "检索 0 条证据 —— 知识库可能是空的（bash Scripts/ingest_knowledge.sh）"
      fi
    else
      fail "问答没有返回答案 —— 这一条同时覆盖 LLM+RAG+重排+图谱+MCP，值得排查"
    fi
  fi
fi

# ═══ 13. 重启演练（--deep）════════════════════════════════════
if [ "$DEEP" = "1" ]; then
  section "重启演练"
  info "  停掉用户服务与容器（不动数据卷），再用 start.sh 拉起…"
  systemctl --user stop hyperdrivewave-llama.service 2>/dev/null || true
  systemctl --user stop hyperdrivewave-resource-coordinator.service 2>/dev/null || true
  ( cd "$HDW_ROOT" && docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" \
      --profile base --profile knowledge --profile web down ) >/dev/null 2>&1 || true

  if bash "$SCRIPT_DIR/start.sh" >/tmp/hdw-deploy-restart.log 2>&1; then
    pass "重启后 start.sh 成功"
    sleep 5
    http_ok "$LLAMA_URL/health" 30 && pass "重启后 llama 恢复" || fail "重启后 llama 未恢复"
    http_ok "$QA_URL/health" 30 && pass "重启后 QA API 恢复" || fail "重启后 QA API 未恢复"
  else
    fail "重启演练失败，详见 /tmp/hdw-deploy-restart.log"
  fi
fi

# ═══ 汇总 ═════════════════════════════════════════════════════
log ""
log "════════════════ 验收结果 ════════════════"
printf '  通过 %s%d%s   警告 %s%d%s   失败 %s%d%s\n' \
  "$_C_GRN" "$PASS" "$_C_RESET" "$_C_YEL" "$WARN" "$_C_RESET" "$_C_RED" "$FAIL" "$_C_RESET"

if [ "$FAIL" -gt 0 ]; then
  log ""
  log "未通过的项："
  for item in "${FAILED_ITEMS[@]}"; do printf '  · %s\n' "$item"; done
  log ""
  exit 1
fi

log ""
ok "验收通过"
exit 0

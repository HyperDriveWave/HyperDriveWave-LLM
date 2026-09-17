#!/usr/bin/env bash
# 端口的**唯一事实源**。被 source，不单独执行。
#
# 一张表 + 几个函数。表里写清每个端口叫什么、默认多少、绑在哪种网卡、
# 能不能改。改端口时靠 `port_apply` 落盘，它会顺带处理连带依赖
# （见 `port_apply_couplings`）——不靠人记。

[ -n "${HDW_PORTS_SH_LOADED:-}" ] && return 0
HDW_PORTS_SH_LOADED=1

# ── 表 ────────────────────────────────────────────────────────
# 格式：键<TAB>环境变量<TAB>默认值<TAB>说明<TAB>绑定范围<TAB>可否更改
#
# 绑定范围：
#   any      = 绑 0.0.0.0，局域网可达（只有 QA API 是这样）
#   lan      = 绑 HDW_WEBUI_BIND 指定的网卡地址
#   loopback = 只绑 127.0.0.1，仅宿主机可访问
#   host     = 不是容器端口，是宿主进程自己监听的
#
# 可否更改：yes / no。标 no 的必须给出理由，不能只是"懒得做"。
ports_manifest() {
  cat <<'EOF'
webui	HDW_WEBUI_PORT	3000	WebUI HTTP	lan	yes
webui_tls	HDW_WEBUI_TLS_PORT	8443	WebUI HTTPS	lan	yes
qa	HDW_QA_API_PORT	8080	QA API	any	yes
llama	HDW_LLAMA_PORT	1919	宿主 llama.cpp	host	yes
llm	HDW_LLM_PORT	8000	旧 FreeToken 引擎	loopback	yes
rag	HDW_RAG_PORT	8001	本机 RAG	loopback	yes
mineru	HDW_MINERU_PORT	8002	文档解析	loopback	yes
neo4j_http	HDW_NEO4J_HTTP_PORT	7475	Neo4j HTTP	loopback	yes
neo4j_bolt	HDW_NEO4J_BOLT_PORT	7688	Neo4j Bolt	loopback	yes
postgres	HDW_POSTGRES_PORT	5432	PostgreSQL	loopback	yes
redis	HDW_REDIS_PORT	6379	Redis	loopback	yes
ingest	HDW_INGEST_PORT	8090	入库 API	loopback	yes
hdw_api	HDW_API_PORT	8095	对外问答 API	any	yes
mcp	HDW_MCP_PORT	8766	MCP 服务	loopback	no
EOF
}

# 为什么 mcp 不能改：
# 容器内监听端口由 Dockerfile 的 CMD ["...", "--port", "8766"] 决定，
# 而命令行参数**压过** HDW_MCP_PORT 环境变量（server.py:459-467）。
# 只改宿主映射而容器内仍是 8766 的话，nginx 的 /api/mcp/ 转发和 qa-api 调用
# 全部会断。要真支持得同时改 Dockerfile + nginx.conf + compose 三处，
# 收益不值这个复杂度——8766 本身也不常冲突。
port_change_note() {
  case "$1" in
    mcp) echo "容器内端口由 Dockerfile 的 --port 8766 固定，改成宿主别的端口会让 nginx 与 qa-api 断链" ;;
    *)   echo "" ;;
  esac
}

# ── 读 ────────────────────────────────────────────────────────

# 从 .env 读，没有则用表里的默认值。
port_get() {
  local key="$1" env_file="${HDW_ENV_FILE:-${HDW_ROOT:-}/Configs/.env}"
  local var def
  read -r _ var def _ _ _ < <(ports_manifest | awk -F'\t' -v k="$key" '$1==k')
  [ -n "$var" ] || return 1
  local v=""
  [ -f "$env_file" ] && v="$(sed -n "s/^${var}=//p" "$env_file" 2>/dev/null | tail -1)"
  echo "${v:-$def}"
}

# 按环境变量名反查键（`--port` 既接受 webui 也接受 HDW_WEBUI_PORT）
port_key_of() {
  local token="$1"
  local k
  k="$(ports_manifest | awk -F'\t' -v t="$token" '$1==t{print $1; exit}')"
  [ -n "$k" ] && { echo "$k"; return 0; }
  k="$(ports_manifest | awk -F'\t' -v t="$token" '$2==t{print $1; exit}')"
  [ -n "$k" ] && { echo "$k"; return 0; }
  return 1
}

port_var() { ports_manifest | awk -F'\t' -v k="$1" '$1==k{print $2}'; }
port_default() { ports_manifest | awk -F'\t' -v k="$1" '$1==k{print $3}'; }
port_desc() { ports_manifest | awk -F'\t' -v k="$1" '$1==k{print $4}'; }
port_scope() { ports_manifest | awk -F'\t' -v k="$1" '$1==k{print $5}'; }
port_changeable() { ports_manifest | awk -F'\t' -v k="$1" '$1==k{print $6}'; }

# 当前值是否偏离默认（用来提示"你改过哪些"）
port_is_default() {
  [ "$(port_get "$1")" = "$(port_default "$1")" ]
}

# ── 连带依赖 ──────────────────────────────────────────────────
# 只有两处需要联动。其余端口改了不影响任何别的地方：
# 容器之间的通信全部走**容器内端口**（hdw-rag:8001、hdw-qa-api:8080 等），
# 与宿主映射无关——这是排查确认过的。
port_apply_couplings() {
  local key="$1" val="$2"
  local env_file="${HDW_ENV_FILE:-${HDW_ROOT:-}/Configs/.env}"
  local runtime="${HDW_ROOT:-}/HDW_Runtime"

  case "$key" in
    llama)
      # ① .env 里两处 base_url 都得跟着走。
      #    不跟的话：start.sh 的"要不要拉起 llama"判断会失效（它是拿
      #    base_url 里的端口和实际端口比对的），llama 直接不启动。
      local old_url new_url
      old_url="$(sed -n 's/^HDW_LOCAL_LLM_BASE_URL=//p' "$env_file" 2>/dev/null | tail -1)"
      new_url="$(printf '%s' "${old_url:-http://host.docker.internal:1919/v1}" \
                 | sed -E "s#(host\.docker\.internal:)[0-9]+#\1${val}#")"
      env_set "$env_file" HDW_LOCAL_LLM_BASE_URL "$new_url"
      env_set "$env_file" HDW_LLM_BASE_URL "$new_url"
      dim "    → HDW_LOCAL_LLM_BASE_URL / HDW_LLM_BASE_URL 已改为 $new_url"

      # ② config.json 的 local.base_url —— **这处不改等于端口没改**。
      #    qa-api 读的是 config.json，.env 只是 fallback
      #    （app/main.py 的 _read_model_config / _effective_profile）。
      #    漏了它的表现极具迷惑性：/health 全绿、模型管理页正常，
      #    一提问就 connection refused。
      local cfg="$runtime/model-config/config.json"
      if [ -f "$cfg" ] && have_cmd python3; then
        python3 - "$cfg" "$new_url" <<'PY' && dim "    → config.json 的 local.base_url 已同步"
import json, sys, pathlib
path, url = sys.argv[1], sys.argv[2]
p = pathlib.Path(path)
cfg = json.loads(p.read_text(encoding="utf-8"))
if isinstance(cfg.get("local"), dict):
    cfg["local"]["base_url"] = url
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    tmp.replace(p)
PY
      fi
      ;;

    webui|webui_tls)
      # FRP 转发配置里的 localPort/localIP 是手写的，不由环境变量生成。
      # 不同步的话：隧道照常"建立成功"，但转发到旧端口——
      # 只有 frpc 日志里有 connection refused，控制中心显示一切正常。
      sync_frpc_config
      ;;

    "") : ;;
  esac
}

# 把 WebUI 的绑定地址与端口同步进 frpc 配置。
sync_frpc_config() {
  local toml="${HDW_ROOT:-}/HDW_Frontend/FRP/conf/frpc_hdw_public.toml"
  [ -f "$toml" ] || return 0
  local env_file="${HDW_ENV_FILE:-${HDW_ROOT:-}/Configs/.env}"
  local bind ip
  bind="$(sed -n 's/^HDW_WEBUI_BIND=//p' "$env_file" 2>/dev/null | tail -1)"
  [ -n "$bind" ] || return 0
  # 0.0.0.0 不能直接写进 localIP（frpc 连不上通配地址），退回环回
  case "$bind" in 0.0.0.0|"") ip="127.0.0.1" ;; *) ip="$bind" ;; esac

  local http_port tls_port changed=0
  http_port="$(port_get webui)"
  tls_port="$(port_get webui_tls)"

  python3 - "$toml" "$ip" "$http_port" "$tls_port" <<'PY' && changed=1
import re, sys, pathlib
path, ip, http_port, tls_port = sys.argv[1:5]
p = pathlib.Path(path)
src = p.read_text(encoding="utf-8")
orig = src

# proxy 块按出现顺序对应 [http, https]。逐个替换 localIP / localPort。
ports = [http_port, tls_port]
idx = [0]
def sub_port(m):
    i = idx[0]; idx[0] += 1
    return f"{m.group(1)}{ports[i] if i < len(ports) else m.group(2)}"
src = re.sub(r"(localPort\s*=\s*)(\d+)", sub_port, src)
src = re.sub(r"(localIP\s*=\s*)\"[^\"]*\"", lambda m: f'{m.group(1)}"{ip}"', src)

if src != orig:
    p.write_text(src, encoding="utf-8")
    sys.exit(0)
sys.exit(1)
PY
  if [ "$changed" = "1" ]; then
    dim "    → FRP 配置已同步：localIP=$ip，localPort=$http_port / $tls_port"
    warn "    FRP 配置改了，需要重启才生效：systemctl --user restart hyperdrivewave-frpc.service"
  fi
}

# ── 写 ────────────────────────────────────────────────────────

port_apply() {
  local key="$1" val="$2"
  local env_file="${HDW_ENV_FILE:-${HDW_ROOT:-}/Configs/.env}"
  local var
  var="$(port_var "$key")"
  [ -n "$var" ] || { warn "未知端口键：$key"; return 1; }
  case "$val" in
    ''|*[!0-9]*) warn "$key 的端口值不是数字：$val"; return 1 ;;
  esac
  [ "$val" -ge 1 ] && [ "$val" -le 65535 ] || { warn "$key 的端口超出范围：$val"; return 1; }

  local cur; cur="$(port_get "$key")"
  [ "$cur" = "$val" ] && return 0
  env_set "$env_file" "$var" "$val"
  info "  $key：$cur → $val（$var）"
  port_apply_couplings "$key" "$val"
}

# 检查所有**本项目会用的**端口有没有被外部占用。
# 输出 "键 端口 说明" 每行一条，空表示无冲突。
ports_conflicts() {
  local key var def desc scope changeable p
  while IFS=$'\t' read -r key var def desc scope changeable; do
    [ -n "$key" ] || continue
    [ "$changeable" = "yes" ] || continue          # mcp 改不了，不参与
    p="$(port_get "$key")"
    if port_busy "$p" && ! port_held_by_hdw "$p"; then
      printf '%s\t%s\t%s\n' "$key" "$p" "$desc"
    fi
  done < <(ports_manifest)
}

# 端口表的人读形式
ports_table() {
  local key var def desc scope changeable p mark
  while IFS=$'\t' read -r key var def desc scope changeable; do
    [ -n "$key" ] || continue
    p="$(port_get "$key")"
    case "$scope" in
      any)      scope="所有网卡" ;;
      lan)      scope="局域网" ;;
      loopback) scope="仅本机" ;;
      host)     scope="仅本机（宿主进程）" ;;
    esac
    mark=""
    [ "$changeable" = "no" ] && mark="  [不可改]"
    [ "$p" != "$def" ] && mark="$mark  [已改，默认 $def]"
    printf '  %-14s %-6s %-22s%s\n' "$desc" "$p" "$scope" "$mark"
  done < <(ports_manifest)
}

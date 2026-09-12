#!/usr/bin/env bash
# 从 Configs/.env 生成 Configs/.env.example。
#
# 为什么需要它：`.env.example` 是新机器唯一的起点（deploy.sh 靠它生成 .env），
# 而手工维护必然会过时——实际发生过：模板比真实配置**少 21 个键**，
# 包括 HDW_RAG_ADMIN_TOKEN、HDW_AUTH_SECRET、HDW_WEBUI_BIND 这些必需的，
# 新机器从这份模板起步会缺一堆配置，且症状分散很难定位。
#
# 用法：
#   bash Scripts/gen_env_example.sh            # 生成并显示会清空哪些值
#   bash Scripts/gen_env_example.sh --check    # 只比对差异，不写文件
#
# 规则：
#   · 保留所有键、注释、顺序
#   · 凭据类键的值换成占位符
#   · deploy.sh 注释掉的绝对路径恢复成注释形式（模板里不该有绝对路径）

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

. "$SCRIPT_DIR/lib/common.sh"

SRC="$HDW_ROOT/Configs/.env"
DST="$HDW_ROOT/Configs/.env.example"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -f "$SRC" ] || die "找不到 $SRC"

if [ "$CHECK_ONLY" = "1" ]; then
  python3 - "$SRC" "$DST" <<'PY'
import pathlib, sys
def keys(p):
    out = []
    for line in pathlib.Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            out.append(s.split("=", 1)[0].strip())
    return out
src, dst = keys(sys.argv[1]), keys(sys.argv[2])
miss = [k for k in src if k not in dst]
print(f"  .env 有 {len(src)} 个键，模板有 {len(dst)} 个")
if miss:
    print(f"  ❌ 模板缺少 {len(miss)} 个（新机器会缺这些配置）：")
    for k in miss: print(f"     {k}")
    sys.exit(1)
print("  ✅ 模板覆盖了 .env 的全部键")
PY
  exit $?
fi

# 清空凭据：只报键名，不回显值
BLANKED="$(python3 - "$SRC" "$DST" <<'PY'
import pathlib, re, sys

# 键名匹配。**必须带词边界**（`_` 或结尾），不能用裸子串：
# 踩过的坑——`KEY|TOKEN|SECRET` 这样的子串匹配会把非凭据也清掉：
#   KEYCLOAK_URL / KEYCLOAK_REALM / KEYCLOAK_CLIENT_ID   （含 "KEY"）
#   HDW_LLM_MAX_TOKENS / HDW_CONTEXT_WINDOW_TOKENS        （含 "TOKEN"）
# 这些都是配置值不是凭据，被清成占位符会让模板直接不可用。
# 加上边界后只命中 _KEY / _TOKEN / _SECRET 这类真正的凭据键。
KEY_PAT = re.compile(r"_(KEY|TOKEN|SECRET|PASSWORD|PASSWD|USERNAME)$", re.I)
# 明确不是凭据的例外
EXEMPT = {"HDW_ENABLE_AUTH", "HDW_INTERNAL_API_KEY"}

def placeholder(key: str) -> str:
    base = key.replace("HDW_", "").lower()
    if key == "HDW_INTERNAL_API_KEY":
        return "local-dev-key"          # 本地开发用的固定值，不是真凭据
    if "USERNAME" in key.upper():
        return "your-username"
    if "PASSWORD" in key.upper() or "PASSWD" in key.upper():
        return "change_me"
    if "TOKEN" in key.upper():
        return "your-token"
    return "your-key"

out, blanked = [], []
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        out.append(line); continue
    k, v = s.split("=", 1)
    k, v = k.strip(), v.strip()
    if k in EXEMPT:
        out.append(line); continue
    if KEY_PAT.search(k) and v and "change_me" not in v and not v.startswith("your"):
        out.append(f"{k}={placeholder(k)}")
        blanked.append(k)
    else:
        out.append(line)

header = """# HyperDriveWave 配置模板 —— 由 Scripts/gen_env_example.sh 从 Configs/.env 生成。
#
# 新机器：cp Configs/.env.example Configs/.env 后按实际环境修改。
# 或直接跑 bash Scripts/deploy.sh，它会自动生成并引导你填关键项。
#
# **不要手工编辑这个文件**——改了会在下次生成时被覆盖。
# 要加新配置项，先加到 Configs/.env，再跑一次生成脚本。
"""
pathlib.Path(sys.argv[2] + ".new").write_text(header + "\n" + "\n".join(out) + "\n", encoding="utf-8")
print("\n".join(blanked))
PY
)"

# 生成失败就别继续——不然会打印"已生成"，而文件根本没写出来
[ -f "$DST.new" ] || die "生成失败（见上面的 Python 报错），$DST 未被改动"

echo "$BLANKED" | while read -r k; do
  [ -n "$k" ] && dim "    已清空：$k"
done
echo "$BLANKED" | grep -c . | xargs echo "  共清空"

mv "$DST.new" "$DST"
ok "已生成 $DST"
dim "  校验一下没有残留真实凭据：bash Scripts/gen_env_example.sh --check"

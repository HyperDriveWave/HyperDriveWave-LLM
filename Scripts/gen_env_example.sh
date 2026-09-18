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
# 这些键在模板里**故意**只以注释形态存在（原因见下面 REMOTE_ONLY_KEYS 处）。
# 不排除的话 --check 会永久报红——红久了的检查等于没有检查。
REMOTE_ONLY = {"HDW_REMOTE_RAG_ROOT"}
miss = [k for k in src if k not in dst and k not in REMOTE_ONLY]
by_design = [k for k in src if k not in dst and k in REMOTE_ONLY]
print(f"  .env 有 {len(src)} 个键，模板有 {len(dst)} 个")
if by_design:
    print(f"  · 其中 {len(by_design)} 个按设计只以注释形态存在：{', '.join(by_design)}")
if miss:
    print(f"  ❌ 模板缺少 {len(miss)} 个（新机器会缺这些配置）：")
    for k in miss: print(f"     {k}")
    sys.exit(1)
print("  ✅ 模板覆盖了 .env 的全部键")
PY
  exit $?
fi

# 清空凭据：只报键名，不回显值
BLANKED="$(python3 - "$SRC" "$DST" "$HDW_ROOT" <<'PY'
import pathlib, re, sys

# 键名匹配。**必须带词边界**（`_` 或结尾），不能用裸子串：
# 踩过的坑——`KEY|TOKEN|SECRET` 这样的子串匹配会把非凭据也清掉：
#   KEYCLOAK_URL / KEYCLOAK_REALM / KEYCLOAK_CLIENT_ID   （含 "KEY"）
#   HDW_LLM_MAX_TOKENS / HDW_CONTEXT_WINDOW_TOKENS        （含 "TOKEN"）
# 这些都是配置值不是凭据，被清成占位符会让模板直接不可用。
# 加上边界后只命中 _KEY / _TOKEN / _SECRET 这类真正的凭据键。
KEY_PAT = re.compile(r"_(KEY|KEYS|TOKEN|SECRET|PASSWORD|PASSWD|USERNAME)$", re.I)
# 明确不是凭据的例外
EXEMPT = {"HDW_ENABLE_AUTH", "HDW_INTERNAL_API_KEY"}

# 指向**别的机器**的路径键。这类键和下面 unset_abs_path_key 处理的那批不同：
# 那批有自定位的相对默认值，注释掉等于「用默认值」；这批没有默认值，
# 注释掉等于「这台机器上必须自己填」。所以在模板里只能以注释形态出现。
REMOTE_ONLY_KEYS = {"HDW_REMOTE_RAG_ROOT"}

# 指向**私有网络或企业内网**的键。这类值不能进模板：
# 公开仓库里一个可访问的企业登录页等于把攻击面直接指出来，内网 IP 则暴露拓扑。
# 和路径那批不同，这些键**必须继续生效**（新机器要填自己的值），所以换成占位符，
# 不能像 REMOTE_ONLY_KEYS 那样整行注释掉。
# 2026-09-16：原先这里只脱敏绝对路径，企业 SIS 门户地址被原样提交了很久。
PRIVATE_HOST_KEYS = {
    "HDW_SIS_BASE_URL": "https://<企业SIS门户>",
    "HDW_SIS_LOGIN_URL": "https://<企业SIS门户>/login.html",
    "HDW_RAG_REMOTE_URLS": "http://<远端RAG主机IP>:8001,http://<远端RAG主机IP>:8003",
    "HDW_REMOTE_RAG_SSH_TARGET": "<远端用户名>@<远端RAG主机IP>",
    # 占位符里**不要再写字面的私有地址**——_scrub 会把它再收一道，
    # 变成「<…，如 <私有IP>>」这种嵌套占位符。
    "HDW_WEBUI_BIND": "<本机内网网卡地址>",
    "HDW_FRP_PUBLIC_HOST": "<中转机公网地址>",
}

# 兜底：注释里出现的私有地址也一并收敛。键值那一层由 PRIVATE_HOST_KEYS 处理，
# 但注释里常顺手写「实测 <某台内网机>:8003 就是这样」——那同样是泄露。
# （写这条规则时我自己就在注释里把真实内网 IP 写进去了，被本脚本扫出来才发现。）
_PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"
)

def placeholder(key: str) -> str:
    base = key.replace("HDW_", "").lower()
    if key == "HDW_INTERNAL_API_KEY":
        return "local-dev-key"          # 本地开发用的固定值，不是真凭据
    if key == "HDW_API_KEYS":
        # 值里**本身含多把密钥**，占位符要把格式也带上，否则看模板的人不知道
        # 该写成什么样。**别在这一行里写真实标签**——_scrub 收不走它们。
        return "your-label1:your-key,your-label2:your-key"
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
    if k in REMOTE_ONLY_KEYS:
        # 指向**别的机器**的路径。不能原样留在模板里：新机器照抄会拿到一个
        # 本机不存在的路径，而且要等 sync_remote_rag.sh 报错才发现。
        # 也不能换成占位符后保持生效——那会让新机器拿到一个字面量路径当真实值。
        # 唯一正确的形态是注释掉，并说明必须按实际部署填。
        out.append("# [deploy] 已注释以保持可移植：这是**远端机**上的项目根，本机代码无法自定位，")
        out.append("# 必须按实际部署填写。留空时 Scripts/sync_remote_rag.sh 会明确报错而不是猜一个路径。")
        out.append(f"# {k}=/home/<远端用户名>/HyperDriveWave-RAG")
        blanked.append(k)
        continue
    if k in PRIVATE_HOST_KEYS:
        out.append(f"{k}={PRIVATE_HOST_KEYS[k]}")
        blanked.append(k)
        continue
    if KEY_PAT.search(k) and v and "change_me" not in v and not v.startswith("your"):
        out.append(f"{k}={placeholder(k)}")
        blanked.append(k)
    else:
        out.append(line)

# 模板里不该出现任何主机绝对路径——它既暴露部署环境，也会让新机器以为可以照抄。
# 项目根换成 <项目根>（注释里的示例路径也一并换），其余 /home/<某用户> 收敛成 <用户>。
_project_root = sys.argv[3].rstrip("/")


def _scrub(text: str) -> str:
    text = text.replace(_project_root, "<项目根>")
    # 注释里的私有地址同样要收：它们暴露内网拓扑，且没有任何保留价值。
    text = _PRIVATE_IP.sub("<私有IP>", text)
    return re.sub(r"/home/(?!<)[^/\s\"']+", "/home/<用户>", text)


out = [_scrub(line) for line in out]

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

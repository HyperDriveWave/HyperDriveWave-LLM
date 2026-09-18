# HyperDriveWave 对外问答 API

给**其它项目**调用的问答接口：发一个问题，拿回回答。内网直连，不经 FRP。

两个刻意的设计，用之前先知道：

- **不做上下文管理**。每次提问都是独立的，模型只看到当次问题，看不到同一
  `session_id` 里此前的问答。要追问就把前文自己写进 `question` 里。
- **响应只有回答**，不带引文/证据。检索到的片段留在服务端。

留档（问题+回答）会落到 `HDW_Runtime/chatdata/api/<session_id>.json`。

## 快速开始

把 `<本机内网地址>` 换成部署这台机器的内网 IP（端口默认 8095，见 `Configs/.env`
的 `HDW_API_PORT`）。**密钥由服务端运维单独发给你**（存在 `Configs/.env` 的
`HDW_API_KEYS` 里），别写进任何会提交的文件。

```bash
curl -X POST http://<本机内网地址>:8095/api/v1/ask \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <运维给你的密钥>' \
  -d '{"question":"汽轮机超速保护动作值是多少？"}'
```

```json
{
  "session_id": "api-1789614407811-66x8q6",
  "turn": 1,
  "question": "汽轮机超速保护动作值是多少？",
  "answer": "汽轮机超速保护的动作转速为 3240 r/min。……",
  "elapsed_ms": 8421
}
```

Python（`httpx` 或 `requests` 都行）：

```python
import httpx

BASE = "http://<本机内网地址>:8095"
KEY = "<运维给你的密钥>"        # 放环境变量，别写死在代码里

resp = httpx.post(
    f"{BASE}/api/v1/ask",
    headers={"Authorization": f"Bearer {KEY}"},
    json={"question": "引风机的轴承温度报警值是多少？"},
    timeout=660.0,              # 必须 ≥ 服务端 HDW_API_QA_TIMEOUT（默认 600s）
)
resp.raise_for_status()
print(resp.json()["answer"])
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/ask` | 提问，同步返回回答 |
| GET | `/api/v1/sessions?limit=50` | 列出留档会话（按最近更新倒序） |
| GET | `/api/v1/sessions/{session_id}` | 某会话的全部问答 |
| DELETE | `/api/v1/sessions/{session_id}` | 删除某会话留档 |
| GET | `/health` | 探活，**无需鉴权**（不返回任何配置值） |

### POST /api/v1/ask

请求体：

| 字段 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `question` | 是 | — | 1–8000 字 |
| `session_id` | 否 | 自动生成 | 用于留档分组。自带一个能把同一业务的问答归到一起；不传则每次新建，返回体里会给生成的 id |
| `inference_mode` | 否 | `offline` | `offline` 走本机模型；`online` 是否可用由服务端策略决定，未开放时返回 403 |
| `top_k` | 否 | 40 | 检索条数上限，1–40 |

响应字段：`session_id`、`turn`（这是该会话的第几轮）、`question`、`answer`、
`elapsed_ms`。

### 鉴权

两种写法等价，二选一：

```
Authorization: Bearer <你的密钥>
X-HDW-API-Key: <你的密钥>
```

密钥比对用常数时间比较，不匹配一律 `401`（不区分"没带"和"带错了"）。服务端
**一把密钥都没配**时返回 `503` 而不是放行——这是有意的，见下面「安全」。

**每个调用方一把密钥**（`HDW_API_KEYS`，格式 `标签:密钥,标签:密钥`）。
这样吊销其中一个不影响其它——共用一把的话，想停掉 A 就会把 B 也断了。
你的密钥由服务端运维单独发给你；`HDW_API_KEY` 是单密钥形式，
只在服务端没配列表时使用。

### 状态码

| 码 | 含义 | 怎么办 |
| --- | --- | --- |
| 200 | 正常 | — |
| 401 | 密钥不对 | 检查密钥值；注意 `Bearer ` 后面有个空格 |
| 403 | 请求了未开放的在线推理 | 改用 `offline` |
| 404 | `session_id` 不存在（读取/删除时） | — |
| 422 | `question` 为空、`session_id` 非法 | 见下 |
| 502 | 上游问答不可达，或服务间密钥不一致 | 看服务端日志；后者要查 `HDW_API_INTERNAL_KEY` |
| 503 | 服务端没配密钥 | 找运维配 `Configs/.env` |
| 504 | 上游超时 | 调大服务端 `HDW_API_QA_TIMEOUT`，或稍后重试 |

### 超时

一次提问要跑检索 + 重排 + 图谱 + 实时取数 + 生成，**实测单发 5 秒到 2 分钟**，
取决于要不要取实时数据。

这是同步接口——请求会一直挂着直到答案生成完。调用方如果用同步 HTTP 客户端，
注意别设成默认的 30 秒。

**服务端有三层超时，调用方只需要关心最外层**：

```text
调用方客户端超时          ← 你设的，必须 ≥ 600s，建议 660s
  └─ hdw-api  HDW_API_QA_TIMEOUT        600s   ← 超了返回 504
       └─ qa-api  HDW_LLM_TIMEOUT       540s   ← 超了返回 502
```

内层刻意比外层小：生成本身超时的话，让内层先报出「LLM 超时」这个有指向性的
错误，而不是让外层干等到底、只留一个没有原因的 504。

**并发时会更慢**。本机和 WebUI 共享同一个 llama 服务（4 个槽位），两边同时
提问时不会排队等待，但会分摊 GPU 吞吐，各自耗时可能翻几倍。所以单发能
10 秒答完的问题，并发时可能要 30 秒以上。600 秒的上限就是为这种情况留的。

**注意 `HDW_LLM_TIMEOUT` 不是"每次读"的超时，而是整段生成的总时限**。
llama 走非流式（`stream: false`），响应头要等生成结束才发——实测
`time_starttransfer` 与 `time_total` 相等。所以只调大 `HDW_API_QA_TIMEOUT`
而不管 `HDW_LLM_TIMEOUT` 是没用的，生成超时会先在里层抛 `ReadTimeout`。

### session_id 的约束

只接受 `A-Za-z0-9_-`，1–64 字符（因为它会作为文件名）。不满足返回 422。
不合法的 id 被拒在入口，不会碰到磁盘。

## 安全

- 这个端口绑在**所有网卡**上（内网可达）。门是密钥，不是网络位置。
- **两类密钥分工不同，别合并**：
  - `HDW_API_KEYS` —— 调用方密钥，每个调用方一个标签一把，**各持各的**。
    吊销其中一个不影响其它。
  - `HDW_API_INTERNAL_KEY` —— 只有 `hdw-api` 和 `qa-api` 两个容器知道，不外发。
  合并的话，拿到对外密钥的人可以绕过本服务直连 qa-api 的内部入口。
- 密钥列表为空时接口直接拒绝服务（fail-closed），不会静默敞开。
- `/health` 不需要密钥，只返回 `status` / 已配置密钥**数量**这类信息——
  **不返回标签**：那等于告诉任何能访问的人「有哪些项目在调这个接口」。
- 留档文件里记了调用方标签（`caller` 字段），方便翻查是哪个项目问的；
  但**同一 session_id 被两个调用方复用时只记最先创建的那个**。

## 运维

```bash
# 改端口（宿主侧）——容器内固定 8095
bash Scripts/deploy.sh --port hdw_api=18095

# 看日志
docker logs -f hyperdrivewave-hdw-api-1

# 只重建这一个服务
docker compose --env-file Configs/.env -f Configs/docker-compose.yml \
  --profile base --profile web up -d --build hdw-api

# 验收（含鉴权是否真的生效）
bash Scripts/deploy_verify.sh
```

留档文件就是普通 JSON，可以直接看：

```bash
ls HDW_Runtime/chatdata/api/
```

写入是「读-改-写」持锁 + 临时文件原子替换，单进程假设（与 qa-api 的
`_conversation_lock` 同一套前提）。要跑多 worker 得先换成文件锁或数据库。

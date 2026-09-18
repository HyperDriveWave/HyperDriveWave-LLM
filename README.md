# HyperDriveWave

HyperDriveWave 是一个面向工业场景的私有化知识问答系统。它不是只把文档切成片段后交给大模型，而是把工业文档、章节结构、设备实体、故障关系、向量检索、重排序、知识图谱、实时工具和可追溯回答组织成一条可验证链路。

项目根目录：

```text
<项目根>
```

本文档以当前代码和 Compose 配置为准，更新时间：2026-09-18。未来接手本项目的开发者或 AI 应先读本文档，再读 `架构.md`，最后以 `Configs/docker-compose.yml` 和各服务的 Dockerfile 为实际运行依据。

## 1. 设计原则

项目按以下纪律演进：

1. 求证：先查当前代码、接口、配置和容器状态，再修改。
2. 需求：先明确输入、输出和验收方式，再实现。
3. 业务：工业安全边界、启停机条件、报警处置和测点含义不能凭空推断。
4. 复用：优先复用已有 MinerU、FreeToken、MCP SDK、Dify、n8n 和 LangGraph 源码。
5. 测试：每个非平凡改动都保留最小可运行检查。
6. 架构：第三方源码作为依赖或参考，自研胶水层放在 HyperDriveWave 自己的目录中。
7. 坦诚：当前未启用的服务必须标明“预留”或“未接入”，不能把目录存在当成服务已经上线。
8. 迭代：先保持 P0 闭环可用，再增加权限、工具路由、审批和自动化。

## 2. 当前能力总览

当前已形成的主要能力：

| 能力 | 当前实现 | 入口 |
| --- | --- | --- |
| 工业聊天问答 | FastAPI QA API 调用 RAG、Neo4j 和 Qwen；默认返回 JSON，传 `stream: true` 时走 SSE（进度事件 + 心跳 + 最终结果） | WebUI 首页 |
| 图片提问 | 上传图片，自动转写成文字后进入检索；**默认仅管理员可用**，可由管理员开放给普通用户 | WebUI 首页输入框 |
| 视觉来源优先级 | 图片转写有三种来源（在线对话模型 / 本地模型 / MinerU），按 `vision_priority` 顺序尝试 | 模型管理页面 |
| 历史相关性选留 | 每次提问前判断哪些历史轮次与当前问题相关，无关的不进 prompt；实测把 82 轮 / 103k token 的会话压到 1.7k | 自动 |
| 对话内问答导航 | 右侧边缘一列短横线，悬停展开成提问列表，点击跳到那一次问答 | WebUI 右侧 |
| 本地模型加载/卸载 | 管理员可停掉 llama-server 释放约 21.8 GB 显存，再随时加载回来；期间本地问答不可用 | 模型管理页面 |
| Qwen 底座 | `llama.cpp` 常驻加载 `Qwen3.8-27B-GSQ` MTP 模型；FreeToken 仅保留为可选旧方案 | `hyperdrivewave-llama.service` |
| 文档上传 | 局域网 WebUI 上传到 ingest API | 知识库入库页面 |
| MinerU 解析 | PDF、DOCX、PPTX、XLSX 和图片解析为 Markdown/JSON | `hdw-mineru` |
| 顺序解析 | 上层逐文件调用，MinerU API 并发上限为 1 | `MINERU_API_MAX_CONCURRENT_REQUESTS=1` |
| 文档切分 | 按章节、编号标题和文本窗口生成 Chunk | `ingest_documents.py` |
| 向量检索 | BGE-M3 嵌入，Zvec 持久化索引；QA 优先轮询远端双 GPU RAG，失败回本机 CPU RAG | `hdw-rag` 和远端 RAG |
| GPU 知识库维护 | 入库、全量重建、删除重建先卸载本地 llama，临时让本机 RTX 5090 执行 RAG/MinerU，完成后恢复 CPU RAG 和 llama | 资源协调器和知识库入库任务 |
| 查询重排序 | BGE reranker 在查询时重排 | `hdw-rag` |
| 知识图谱 | Neo4j 保存文档、章节、Chunk、设备、参数、故障等关系 | `hdw-neo4j` |
| 图谱增强问答 | 根据召回 Chunk 查找章节、相邻片段和实体关系 | QA API |
| MCP | MCP 服务和工具目录页面，包含 SIS、RTSP 等工具适配 | `hdw-mcp` 和 WebUI MCP 页面 |
| 对话历史 | 每个会话一个 JSON 文件，局域网客户端共享 | `HDW_Runtime/chatdata` |
| 球球交互 | Aora emotion-ball 球球、鼠标跟随、Thinking、主题切换 | WebUI |
| 知识图谱可视化 | SVG 节点、关系、筛选、搜索、拖动和悬停详情 | 知识库图谱页面 |

当前明确的限制：

1. Dify、n8n、LangGraph、Langfuse、Keycloak 和 Open WebUI **当前未被任何代码引用**，
   也没有纳入主 Compose，不能假设它们已经启动。它们的版本记在 `vendor/vendor.lock`
   里但标为「不默认拉取」，要用时 `bash Scripts/fetch_vendors.sh --with dify,n8n`。
   默认部署真正需要的只有 `MinerU`（`knowledge` profile）和 `aora-bot`（`web` profile 的情绪球）。
2. MCP 的**只读**工具已经自动并入 `/qa/query`：SIS 测点现值/历史/趋势、LIEMS 日志由同一次检索规划的模型结论决定取哪些，不需要人工触发。仍未接入的是 RTSP、thermal、lstm、alarm、edge_device、custom_rule。
3. 普通入库和全量重建仍是两种语义。全量重建会重新解析源目录内全部文档，这是为了让源文件、解析结果、Chunk、Neo4j 和 Zvec 一致。
4. 当前没有“取消正在运行的 ingest 任务”接口。停止任务需要先确认任务状态，再停止 ingest API、清理 MinerU 工作进程，并检查数据是否需要恢复。
5. `aora-bot/emotion-ball` 的许可证需要在商业部署前重新核对，不能因为已经能运行就默认可以商业使用。
6. 远端双 GPU RAG 与本机 CPU RAG 的索引必须保持一致；WebUI 入库任务会在本机重建完成并拉起 llama 后自动同步远端两份索引。此外知识库页面有独立的「同步远端」按钮（`POST /sync`），用于**不重建**而只把现有 chunks 重推一遍、刷新文档上的同步状态——`Scripts/sync_remote_rag.sh` 保留作人工恢复工具。

## 3. 项目结构

以下是项目级结构。**带 `[vendor]` 标记的目录不在版本库里**——它们由
`Scripts/fetch_vendors.sh` 按 `vendor/vendor.lock` 里的固定 commit 克隆回来，
项目结构保持不变。同理，模型权重和构建产物也不进库（见 §10.8）：

```text
[vendor]  = 第三方仓库，clone 得到
[fetch]   = 模型权重，Scripts/fetch_models.sh 下载
[build]   = 构建产物，目标机重新编译
```

第三方仓库内部文件不在这里展开：

```text
HyperDriveWave/
├── Configs/
│   ├── .env
│   ├── .env.example
│   ├── docker-compose.yml
│   ├── docker-compose.rebuild-gpu.yml
│   ├── nginx/
│   └── profiles/
├── Scripts/
│   ├── deploy.sh                          # 一键部署主入口
│   ├── deploy_remote_rag.sh               # 远端 RAG 节点单独部署
│   ├── deploy_verify.sh                   # 验收（不信任 /health）
│   ├── fetch_models.sh                    # 从魔搭补齐模型
│   ├── fetch_vendors.sh                   # 按 vendor.lock 克隆第三方源码
│   ├── setup_mirrors.sh                   # 配 Docker/pip/npm/apt 镜像源
│   ├── build_llama.sh                     # 从源码编译 llama.cpp
│   ├── pack_hdw.sh                        # 源机打包
│   ├── lib/                               # 上述脚本共用的库
│   │   ├── common.sh                      #   幂等写文件 / 日志 / 交互
│   │   ├── detect.sh                      #   GPU / 网络 / 端口探测
│   │   ├── ports.sh                       #   端口清单（唯一事实源）
│   │   └── models.sh                      #   模型清单（唯一事实源）
│   ├── init.sh
│   ├── prepare_dirs.sh
│   ├── start.sh                           # 唯一的启动入口
│   ├── stop.sh
│   ├── restart.sh
│   ├── status.sh
│   ├── logs.sh
│   ├── healthcheck.sh                     # 日常巡检（有已知误报，见 §4.2）
│   ├── backup.sh
│   ├── restore.sh
│   ├── ingest_knowledge.sh
│   ├── sync_remote_rag.sh
│   ├── resource_coordinator.py
│   └── hyperdrivewave-resource-coordinator.service
├── vendor/                                # 第三方依赖的「配方」，不含正文
│   ├── vendor.lock                        #   路径 / commit / URL
│   └── overlays/                          #   项目自有的覆盖文件，克隆后拷回原位
├── HDW_Engines/
│   ├── LLM_Models/
│   │   └── Qwen3.8-27B-GSQ/               # 当前在用（含 MTP 层）
│   ├── RAG_Models/
│   │   ├── bge-m3/
│   │   └── bge-reranker-v2-m3/
│   └── MODELS.md
├── HDW_Inference/
│   ├── llama/                    [build] llama.cpp 多架构构建产物
│   ├── FreeToken/                [vendor]
│   └── RAG_Service/
├── HDW_Knowledge/
│   └── MinerU/                   [vendor]
├── HDW_KnowledgeGraph/
│   ├── cypher/
│   ├── entity_aliases.json
│   └── neo4j-data/
├── HDW_DataFoundation/
│   ├── ETL_Pipelines/
│   ├── MCP/
│   │   ├── MCP_Tools/
│   │   ├── python-sdk/
│   │   └── servers/
│   ├── Mapping/
│   └── RelationalDB/
├── HDW_Orchestrator/
│   ├── industrial-qa-api/
│   ├── industrial-ingest-api/
│   ├── langgraph/
│   ├── dify/
│   └── n8n/
├── HDW_API/                               # 对外问答 API（给别的项目调用）
├── HDW_Frontend/
│   ├── industrial-webui/
│   └── open-webui/
├── HDW_Animation/
│   └── aora-bot/
├── HDW_Ops/
│   └── langfuse/
├── HDW_Security/
│   └── keycloak/
├── HDW_Evaluation/
│   └── RAGAS/
├── HDW_VectorDB/
│   └── zvec/
├── HDW_Runtime/
│   ├── chatdata/
│   ├── knowledge_sources/
│   ├── mineru/
│   ├── rag/
│   ├── zvec/
│   ├── neo4j/
│   ├── postgres/
│   ├── redis/
│   ├── ingest/
│   ├── backups/
│   ├── graph_review/
│   ├── dify/
│   ├── n8n/
│   ├── langfuse/
│   ├── keycloak/
│   └── open-webui/
├── icons/
├── README.md
└── 架构.md
```

`.venv/` 是本机开发环境（给 `fetch_models.sh` 用），不是生产服务，也不进版本库。
带 `[vendor]` 标记的目录同理——它们在完整的工作副本里是完整的 git 仓库，
但在版本库里只留 `vendor/vendor.lock` 里那一行「路径 + commit + URL」。

## 4. 文件夹职责

### 4.1 `Configs`

部署入口和运行参数目录。

- `docker-compose.yml`：HyperDriveWave 当前主 Compose。
- `docker-compose.rebuild-gpu.yml`：知识库维护模式的 Compose 覆盖文件，只给 `hdw-rag` 和 `hdw-mineru` 分配本机 RTX 5090。
- `.env.example`：配置模板，包含端口、路径、模型和服务地址样例。
- `.env`：本机实际配置，含密码和局域网绑定信息，不应提交。
- `profiles/`：后续保存 minimal、GPU、生产环境的 profile 参数；当前 GPU 重建使用独立覆盖文件，不改变常态 CPU 后备。
- `nginx/`：未来可把网关、TLS、安全头和上传策略独立出来。

### 4.2 `Scripts`

**部署**（新机器从零到可用）：

- `deploy.sh`：一键全栈部署主入口。探测 GPU 与网络 → 建目录 → 渲染机器相关配置
  → 补第三方依赖与模型 → 建镜像 → 调 `start.sh` → 验收。支持 `--dry-run` 先看会改什么。
- `deploy_remote_rag.sh`：远端 GPU 机只部署 RAG 节点（不需要主站那套）。
- `deploy_verify.sh`：部署验收。**不信任 `/health`**，真跑一次推理和嵌入，
  详见 §10.6。
- `fetch_models.sh`：按清单从魔搭下载模型，字节级校验。
- `fetch_vendors.sh`：按 `vendor/vendor.lock` 克隆第三方源码回原路径。
- `setup_mirrors.sh`：配置 Docker/pip/npm/apt 国内镜像源。
- `build_llama.sh`：从源码编译 llama.cpp（CUDA / Vulkan / CPU 三选一，按显卡自动定）。
  **从 GitHub 克隆后必须跑**，否则没有 llama-server 二进制。
- `pack_hdw.sh`：源机打包（253 G → 25 G），排除用不到的模型与构建产物。
- `lib/`：上述脚本共用的库。`common.sh`（幂等写文件/日志/交互）、
  `detect.sh`（GPU/网络/端口探测）、`ports.sh`（端口清单的**唯一事实源**）、
  `models.sh`（模型清单的**唯一事实源**）。

**运维**：

- `init.sh`：检查 Docker、Compose 和模型目录。
- `prepare_dirs.sh`：创建 `HDW_Runtime` 持久化目录。
- `start.sh`：按 Compose profile 启动服务。**项目唯一的启动入口**，`deploy.sh` 也调它。
- `stop.sh`：停止当前 Compose 项目，不删除数据卷目录。
- `restart.sh`：停止后重新启动全套服务。
- `status.sh`：查看 Compose 容器状态。
- `logs.sh`：查看全部服务或指定服务日志。
- `healthcheck.sh`：日常巡检。**已知缺陷**：查的是 `${HDW_LLM_PORT:-8000}`
  （旧 FreeToken 端口），不是真实 llama 的 `1919`，会误报 llm unavailable。
  要严格验收请用 `deploy_verify.sh`。
- `backup.sh` / `restore.sh`：打包与还原配置、运行数据、关系库初始化和评测数据。
- `ingest_knowledge.sh`：命令行执行解析、切分、图谱导入和 Zvec 重建。
- `sync_remote_rag.sh`：推送 `chunks.jsonl` 到远端节点并触发重建。
  远端地址必须显式配置（不再有写死的默认值）。
- `resource_coordinator.py`：通过 Unix socket 串行协调 llama 与本机 GPU RAG/MinerU 的占用。
- `hyperdrivewave-resource-coordinator.service`：以用户级 systemd 服务常驻资源协调器。

**注意**：`push_github.py` 之类的本机工具不进版本库，见 `.gitignore`。

### 4.3 `HDW_Engines`

模型权重资产层，不放业务代码。

- `LLM_Models/Qwen3.8-27B-GSQ`：当前 `llama.cpp` 常驻加载的本地 Qwen GSQ/MTP 模型。
- `LLM_Models/Qwen3.8-27B-FP8`：FreeToken 可选旧方案的模型目录，当前默认不加载。
- `LLM_Models/Qwen3.8`：原始或备用模型目录，当前不直接加载。
- `RAG_Models/bge-m3`：Embedding 模型，文档 Chunk 和查询问题共用。
- `RAG_Models/bge-reranker-v2-m3`：查询时对候选 Chunk 重排序。
- 模型通过只读 Volume 挂载，镜像不打包大模型。

### 4.4 `HDW_Inference`

推理运行时：

- `llama`：当前 `llama.cpp` 本地推理入口，由 `hyperdrivewave-llama.service` 管理，提供 OpenAI 兼容接口。
- `FreeToken`：保留的旧推理引擎，只有显式启用 `legacy-freetoken` profile 才会启动 `hdw-llm`。
- `RAG_Service`：加载 BGE-M3、BGE reranker 和 Zvec，提供 `/search`、`/health`、`/admin/reindex`；常态 CPU 运行，维护任务临时切换 CUDA。
- `RAG_Service/remote-compose.yml`：远端双 GPU RAG 部署文件；GPU 0 监听 `8001`，GPU 1 监听 `8003`，两个容器使用独立 zvec 副本。

### 4.5 `HDW_Knowledge/MinerU`

文档解析引擎。当前使用项目内的 MinerU 源码和 `docker/hyperdrivewave-pipeline.Dockerfile` 构建 `hdw-mineru`。

解析原则：

1. Markdown 文件直接复制到解析输出。
2. PDF、DOCX、PPTX、XLSX 和图片通过 MinerU `/file_parse` 转换。
3. `parse_documents.py` 对输入文件排序后逐个提交。
4. Compose 设置 `MINERU_API_MAX_CONCURRENT_REQUESTS=1`，MinerU API 不接受多个文档请求并发。
5. 单个文档内部仍可能由 MinerU 使用 OCR 或渲染子进程，这是一个文档内部的计算并行，不代表多个文档同时解析。

### 4.6 `HDW_DataFoundation`

数据接入和工具基础设施：

- `ETL_Pipelines`：解析结果清洗、章节结构识别、Chunk 切分和图谱导入。
- `MCP/MCP_Tools`：HyperDriveWave 自己的 SIS、报警、RTSP、热成像、边缘设备、日志、LSTM 等工具适配。
- `MCP/python-sdk`：MCP Python SDK 上游项目，作为协议和客户端/服务端能力参考。
- `MCP/servers`：MCP servers 上游项目集合，当前不是全部运行服务。
- `Mapping`：SIS 测点映射、设备映射及工具需要读取的本地数据。
- `RelationalDB`：PostgreSQL 初始化 SQL 和关系数据结构。

### 4.7 `HDW_KnowledgeGraph`

图谱规则和图谱导入相关资产：

- `cypher/constraints.cypher`：Neo4j 约束。
- `cypher/examples.cypher`：示例查询。
- `entity_aliases.json`：实体别名归一化。
- `neo4j-data/`：历史或辅助数据目录；主 Compose 实际运行数据在 `HDW_Runtime/neo4j`。

当前图谱节点主要包括：

```text
Document
Section
Chunk
Equipment
Parameter
Fault
Alarm
Action
```

### 4.8 `HDW_Orchestrator`

编排和业务边界：

- `industrial-qa-api`：当前生产问答入口，负责鉴权、RAG 调用、Neo4j 上下文查询、Qwen 调用和对话文件存储。
- `industrial-ingest-api`：上传、文档状态、入库任务、全量重建和删除重建。
- `langgraph`：上游编排框架，未来可把 QA API 的固定流程升级为可观测状态图；当前没有直接把业务逻辑写进上游源码。
- `dify`：低代码应用、Prompt 实验和原型平台目录；当前主 Compose 未启动。
- `n8n`：自动化流程平台目录；当前主 Compose 未启动。

### 4.9 `HDW_Frontend`

- `industrial-webui`：当前用户入口，包含聊天、历史记录、知识库入库、图谱、模型管理、外网访问、MCP 页面、思维深度和主题切换。
- `FRP`：公网隧道资产（配置、二进制、证书、systemd 单元）。详见 `HDW_Frontend/FRP/README_FRP.md`。
- `open-webui`：保留的上游 WebUI，当前不是主入口。

`industrial-webui/nginx.conf` 的代理关系（8080 明文与 8443 TLS 共用同一组 location 规则）：

```text
/api/*            -> hdw-qa-api
/api/knowledge/* -> hdw-ingest
/api/mcp/*       -> hdw-mcp
```

控制中心的页面与权限：

| 菜单项 | 页面 | 权限 |
| --- | --- | --- |
| 知识库入库 | `#knowledgeIngestPage` | 仅管理员可操作 |
| 知识库图谱 | `#knowledgeGraphPage` | 只读 |
| 模型管理 | `#modelManagementPage` | 仅管理员可保存 |
| 外网访问 | `#frpPage` | 仅管理员可切换 |
| 工作流 / MCP | `#toolPlaceholder` / `#mcpPage` | 只读 |

页面可见性与管理员开关由 `data-admin-only` 属性和 `applyUserPermissions()` 统一处理，新增页面照此模式接入。

### 4.10 `HDW_Animation/aora-bot`

球球交互组件，当前 WebUI 以只读方式挂载其 `emotion-ball/js`：

- 首页苏醒、好奇、发呆、睡眠、唤醒。
- 输入框聚焦时等待输入并向下看。
- 鼠标移动时目光跟随。
- Thinking 和回答完成时使用不同表情。
- 对话次数、空闲时间和回答内容会影响表情。

### 4.11 `HDW_Runtime`

运行数据目录，不放进镜像，不应随意删除：

| 目录 | 内容 |
| --- | --- |
| `chatdata` | JSON 对话会话文件 |
| `knowledge_sources` | 用户上传的原始文档 |
| `mineru` | MinerU 上传、解析结果和任务数据 |
| `rag/chunks.jsonl` | 当前 Chunk 索引源文件 |
| `zvec` | Zvec 持久化索引 |
| `neo4j` | Neo4j data/logs |
| `postgres` | PostgreSQL 数据 |
| `redis` | Redis 数据 |
| `ingest` | 入库状态、任务和实体审查文件 |
| `backups` | 备份归档 |
| `graph_review` | 图谱候选实体审查结果 |
| `dify`、`n8n`、`langfuse`、`keycloak`、`open-webui` | 未来服务的持久化目录 |

### 4.12 `HDW_Ops`、`HDW_Security`、`HDW_Evaluation`、`HDW_VectorDB`

- `HDW_Ops/langfuse`：后续记录 LLM trace、Prompt、Token、延迟、检索结果和反馈。
- `HDW_Security/keycloak`：后续 OIDC、SSO、RBAC 和 JWT。
- `HDW_Evaluation/RAGAS`：离线评测、Golden Dataset 和 RAGAS 指标。
- `HDW_VectorDB/zvec`：向量库相关源码或参考资产；运行索引在 `HDW_Runtime/zvec`。

## 5. 当前容器和端口

当前主 Compose 的服务如下：

| 容器 | Profile | 作用 | 默认端口 |
| --- | --- | --- | --- |
| `hdw-postgres` | `base` | 关系数据 | `127.0.0.1:5432` |
| `hdw-redis` | `base` | 缓存和队列基础设施 | `127.0.0.1:6379` |
| `hdw-neo4j` | `base` | 知识图谱 | HTTP `127.0.0.1:7474`，Bolt `127.0.0.1:7687` |
| `hdw-rag` | `base` | BGE、重排、Zvec | `127.0.0.1:8001` |
| `hdw-qa-api` | `base` | 问答和会话 API | `:8080` |
| `hdw-ingest` | `base` | 文档上传和入库任务 | `127.0.0.1:8090` |
| `hdw-api` | `base` | 对外问答 API（给别的项目调用，不做上下文管理） | `:8095` |
| `hdw-mcp` | `base` | MCP 工具服务 | `127.0.0.1:8766` |
| `hyperdrivewave-llama.service` | `systemd` | 当前 Qwen GSQ/MTP 本地推理 | `127.0.0.1:1919` |
| `hdw-llm` | `legacy-freetoken` | 可选 FreeToken 旧推理 | `127.0.0.1:8000` |
| `hdw-mineru` | `knowledge` | MinerU 文档解析 | `127.0.0.1:8002` |
| `hdw-webui` | `web` | Nginx WebUI 和反向代理（含 TLS 终结） | `${HDW_WEBUI_BIND}:3000` 明文、`:8443` TLS |
| `hyperdrivewave-frpc.service` | `systemd` | 可选公网隧道客户端，由控制中心开关 | 无本地监听 |

当前默认是 10 个 Compose 容器，加 1 个由用户级 systemd 管理的 `llama.cpp` 服务，共 11 个运行服务；`hyperdrivewave-frpc.service` 是第 12 个，仅在启用外网访问时运行。FreeToken、Dify、n8n、Langfuse、Keycloak 和 Open WebUI 默认不启动；它们不应被误计为当前在线服务。

当前 WebUI 的局域网地址取决于 `Configs/.env`：

```text
http://<HDW_WEBUI_BIND>:<HDW_WEBUI_PORT>
https://<HDW_WEBUI_BIND>:<HDW_WEBUI_TLS_PORT>   # 自签证书，浏览器会告警
```

本机当前曾使用：

```text
http://<本机局域网IP>:3000
https://<本机局域网IP>:8443
```

不要把这个地址写死到业务代码；网卡地址变化时只修改 `.env`。

## 6. 文档入库链路

### 6.1 WebUI 入库

```text
局域网浏览器
  -> hdw-webui / Nginx
  -> /api/knowledge/upload
  -> hdw-ingest:8090 /upload
  -> HDW_Runtime/knowledge_sources
  -> /api/knowledge/ingest
  -> hdw-ingest 后台任务
  -> pipeline_input
  -> parse_documents.py
  -> MinerU /file_parse，一次一个文件
  -> HDW_Runtime/mineru/parsed
  -> ingest_documents.py
  -> HDW_Runtime/rag/chunks.jsonl
  -> import_graph.py
  -> Neo4j
  -> hdw-rag /admin/reindex
  -> BGE-M3 嵌入
  -> Zvec
```

上传文件先落到临时文件，完成后才替换为正式源文件。待入库文件从 WebUI 移除时，调用的是 staged 删除接口；已经进入任务的文件不能用 staged 删除接口绕过任务一致性。

### 6.2 普通入库

普通入库通过 WebUI 选中文件后提交对应 `document_ids`。任务会经过：

1. MinerU 解析。
2. Markdown 结构化切分。
3. Chunk 稳定化，生成固定 `document_id` 和 `chunk_id`。
4. Neo4j 图谱导入。
5. Zvec 全量索引刷新。
6. 更新 `HDW_Runtime/ingest/state.json`。

执行期间由资源协调器控制本机 5090 的使用，完整阶段为：

```text
大模型卸载
  -> 本机 GPU MinerU 解析、切分、图谱构建、BGE-M3 嵌入与 Zvec 重建
  -> 重建完成
  -> 拉起大模型
  -> 同步远端 RAG
  -> 同步完成
  -> 最终完成
```

常态 `hdw-rag` 保持 CPU，作为远端不可用时的检索后备；GPU 维护模式只在任务期间
重建 `hdw-rag` 和 `hdw-mineru`，不会删除本机 CPU 后备，也不会占用远端 RAG 容器。

当前普通入库仍有一个需要持续修正的边界：如果调用方不传 `document_ids`，API 会使用状态中的文档集合；不能把“上传一个新文件”误解成已经实现完全增量索引。后续应以文档内容哈希为依据做真正增量解析和增量索引。

### 6.3 全量一致性重建

全量重建会读取当前源目录内全部文档，重新执行：

```text
源文档 -> MinerU -> Markdown -> Chunk -> Neo4j replace-all -> BGE/Zvec reindex
```

已入库文档也会再次解析，这是全量重建的定义，不是重复误操作。它用于修复以下不一致：

- 源文档和解析结果不一致。
- Chunk 文件和 Neo4j 不一致。
- Neo4j 和 Zvec 不一致。
- 删除文档后旧关系或旧向量仍残留。

全量重建和删除重建都使用与普通入库相同的 GPU 维护流程。进度页面会依次显示
“大模型卸载”“MinerU 文档解析”“文档切分”“知识图谱构建”“BGE-M3 嵌入与向量索引”
“重建完成”“拉起大模型”“同步远端RAG中”和“最终完成”，并保留当前文档或片段计数。

重建期间不要移动或删除：

```text
HDW_Runtime/knowledge_sources
HDW_Runtime/mineru/parsed
HDW_Runtime/rag/chunks.jsonl
HDW_Runtime/zvec
HDW_Runtime/neo4j
```

不要执行：

```bash
docker compose down -v
```

这会把容器卷一起处理，风险远大于普通 `docker compose down`。

## 7. 在线问答链路

```text
用户问题
  -> industrial-webui
  -> hdw-qa-api /qa/query
  -> hdw-rag /search
  -> BGE-M3 向量召回
  -> Zvec 候选集
  -> bge-reranker-v2-m3 查询时重排序
  -> 返回高质量 Chunk
  -> QA API 根据 chunk_id 查询 Neo4j
  -> 补充文档、章节、前后 Chunk、设备、参数、故障、报警、动作
  -> Qwen3.8-27B-GSQ（MTP 投机解码）
  -> 返回 answer、citations、graph_context、status、llm、context、
     unsupported_numbers、rag
  -> WebUI 展示回答、进度阶段提示、实时测点卡片、趋势图（内联 SVG）、
     可展开依据、以及数字出处缺失时的提示
```

Qwen 是底座模型。RAG、Neo4j、MCP 和球球不是替代模型，而是围绕模型的扩展：

- RAG 提供文档证据。
- reranker 提高候选证据顺序。
- Neo4j 补充知识结构和关系路径。
- MCP 读取受控外部数据或工具结果。
- QA API 负责编排、边界和返回格式。
- WebUI 负责交互和证据展示。

响应的几个字段值得单独说明：

- `citations` 是**真正进了提示词**的那批证据，不是全部召回结果——它和 `unsupported_numbers`
  的核对用的是同一份文本。
- `llm` 里有 `history_select`（历史选留的统计）和 `prompt_tokens_estimate`（实际送出的
  prompt 大小），排查「为什么这次慢」时先看这两个。
- `context` 是压缩/选留的元信息，`unsupported_numbers` 是答案里找不到出处的数字，
  **空列表才是常态**。
- 开 `stream: true` 时这些都在 SSE 的最后一条 `result` 事件里。

**服务间入口 `/internal/qa/query`**（`POST`）是给 `hdw-api` 容器用的，走同一条
管线但**不做上下文管理**：不读也不写任何用户的会话历史，模型只看到当次问题。
它与 `/qa/query` 的关键差别是鉴权——`/qa/query` 要求登录会话，
`/internal/qa/query` 只认 `X-HDW-Internal-Key`（`HDW_API_INTERNAL_KEY`），
且**无条件校验**：`_check_auth` 在 `HDW_ENABLE_AUTH=false` 时是空操作，
而 qa-api 绑在所有网卡上，拿它守这条入口等于敞开，所以这里单独实现
（见 `_require_internal_api_key` 的注释）。密钥没配时返回 503，不放行。

**MCP 只读调用已并入 `/qa/query`**（SIS 测点现值/历史/趋势、LIEMS 日志），
由同一次检索规划的模型结论决定取哪些，不再是"按需人工触发"。仍未接入的是
RTSP、thermal、lstm、alarm、edge_device、custom_rule 这类。后续继续接入时必须先定义：

1. 哪些问题允许读实时测点。
2. 哪些工具只读，哪些工具允许写入。
3. 工具超时、重试和失败文案。
4. SIS 账号、权限和审计记录。
5. 工具结果如何与文档证据分栏展示。

### 7.1 RAG 远端轮询与本机回退

当前问答检索链路如下：

```text
QA API
  ├─ 请求 1 -> <远端RAG主机IP>:8001 -> 远端 GPU 0
  ├─ 请求 2 -> <远端RAG主机IP>:8003 -> 远端 GPU 1
  ├─ 请求 3 -> <远端RAG主机IP>:8001
  └─ ...
       远端当前节点失败 -> 尝试另一个远端节点
       远端全部失败     -> http://hdw-rag:8001（本机 CPU fallback）
```

两个远端容器均使用相同的 `bge-m3`、`bge-reranker-v2-m3`、`chunks.jsonl`
和知识库内容，但各自绑定一张物理 GPU，并各自使用独立的 zvec 索引目录：

```text
远端主机：<远端RAG主机IP>
├── hdw-rag-gpu0 -> GPU 0 -> :8001
└── hdw-rag-gpu1 -> GPU 1 -> :8003
```

主项目的 `hdw-rag` 必须保留。它既是本机入库链路使用的 RAG 服务，也是远端
不可用时的最后检索后备。远端 RAG 不参与本机文档入库，避免在索引尚未同步时
影响入库一致性。

知识库更新后同步远端：

```bash
cd <项目根>
SSHPASS='远端SSH密码' bash Scripts/sync_remote_rag.sh
```

脚本只上传新的 `HDW_Runtime/rag/chunks.jsonl`，并分别请求两个远端节点执行
`/admin/reindex`；模型权重不会重复上传。两个重建任务都成功后才返回成功。

### 7.2 本地推理的故障边界

本地 `llama.cpp` 曾有两个会打断问答的故障点，**已通过升级 llama.cpp 版本解决**。

**① 对话模板解析失败 + 输出污染（已修复）**

旧版 llama.cpp（`src/llama.cpp-8681+dfsg`）会从模型的 Jinja 模板自动生成 PEG 解析器，
把**模型输出**切分成思考/正文/工具调用；模型跑偏时输出不符合语法，上游直接 `throw`，
被 QA API 包装成 `503`，用户侧表现为「提问后返回错误、没有回答」。

同一根因还有一个更隐蔽的表现：**输出污染** —— 模型稳定地在中文里插入英文/阿拉伯文
碎片（典型是「物理」错成 `ysics`）。这些碎片**都是词表里的精确 token**（`ysics`=16735，
`ifs`=21163），所以不是丢字乱码，而是模型**选错了 token**。实测污染率约 20-25%，
且与温度无关（贪心解码逐字节可复现）。

排查中逐项排除的变量（都**不是**原因）：后端（Vulkan/CUDA）、KV 缓存量化（q4_0/f16）、
Flash Attention 开关、采样参数（`temperature` / `presence_penalty`）、量化版本
（IQ3_S 与 Q5_K_M 都复现，且错成同一个 token）。

**升级到 `HDW_Inference/llama/llama.cpp-upstream` 后两者都消失：**

| 配置 | 解码速度 | 输出污染 |
| --- | --- | --- |
| 旧版 + Vulkan | 66.7 tok/s | 2-3/12（~22%）|
| 旧版 + CUDA | 78.6 tok/s | 2-3/12（~22%）|
| 新版 + CUDA | 93.6 tok/s | 0/16 |
| 新版 + CUDA + MTP | 117-124 tok/s | 0/32 |

旧版曾打过自定义补丁（`common/chat.cpp` 等 4 个文件：解析失败降级为纯文本而非抛异常）。
**新版不再需要** —— 上游已改为 `LOG_WRN("unparsed ...")` 记警告不抛异常。补丁源码留在
`src/llama.cpp-8681+dfsg/*.bak_before_parse_fix`，仅作历史记录。

排查命令：

```bash
journalctl --user -u hyperdrivewave-llama.service -n 50 --no-pager | grep -E "unparsed|exception|draft acceptance"
```

**② 前端在失败时删除用户提问**

`industrial-webui` 原先的请求失败分支会把用户刚发出的提问从会话里 `pop()` 掉并
重新存盘，一次瞬时上游抖动就丢掉用户输入。现在改为保留提问和失败气泡，失败信息
作为一条 assistant 消息留痕，用户可直接重问。

### 7.3 本地推理后端与 MTP 投机解码

**后端**由 `Configs/.env` 两行控制，改完 `systemctl --user restart hyperdrivewave-llama.service` 生效：

```bash
HDW_LLAMA_DEVICE=CUDA0
HDW_LLAMA_BINARY=<项目>/HDW_Inference/llama/llama.cpp-upstream/build-cuda/bin/llama-server
```

`start.sh` 会打印实际使用的二进制和设备（`llama.cpp binary: ... (device ...)`），
排查时先看这一行。三个二进制都在，可随时切换：

| 路径 | 版本 | 后端 |
| --- | --- | --- |
| `llama.cpp-upstream/build-cuda/bin/` | 新版（2026-09-08） | CUDA ← **当前使用** |
| `build-cuda/bin/` | 旧版 | CUDA |
| `build/bin/` | 旧版 | Vulkan |

**MTP 投机解码**（Multi-Token Prediction）由控制中心 **模型管理 → 本地推理 → 启用 MTP**
勾选框控制，无需改配置文件。

模型自带 MTP 头（`qwen35.nextn_predict_layers=1`，`blk.64` 的 4 个 `nextn` 张量）。
开启后 llama-server 额外建立 draft 上下文，实测：

```
开启 MTP：解码 117-124 tok/s，接受率 0.45-0.50，mean len 2.34-2.49
关闭 MTP：解码  93.6 tok/s
```

链路：**勾选框 → 模型配置 `local.mtp_enabled` → `PATCH /model-config`（触发 `/switch-llm`
重启）→ `start.sh` 读取该字段 → 加 `--spec-type draft-mtp`**。

两个注意点：

1. **`--spec-type` 只有新版支持。** `start.sh` 会探测二进制的 `--help`，不支持时只打印
   一行提示并跳过 MTP，**不会导致启动失败** —— 所以切回旧版二进制是安全的。
2. **MTP 只影响解码，预填充会略降**（draft 上下文有开销）。解码占问答总耗时的 74-87%，
   所以净收益仍然显著。

启用 MTP 前后的端到端实测（同一组问题）：

| 问题 | 旧版 + Vulkan | 新版 + CUDA + MTP |
| --- | --- | --- |
| 汽轮机轴承温度高怎么处理 | 13.6s | **8.1s** |
| 简要说明 ETS 系统的作用 | 16.0s | **8.0s** |
| 轴封系统与真空系统的关系 | 19.1s | **12.2s** |

### 7.4 检索路由（按需检索）

**问题**：`/qa/query` 之前无条件走 RAG + Neo4j。问「你的上下文长度是多少」这类系统能力问题
时，会检索出 10 条毫不相关的证据（消防参数、循蝶阀文件）挂进回答，既浪费时间也污染界面。

**方案**：检索前先让模型判一次「这个问题要不要查知识库」。判定为不需要时，**完全不碰
RAG 和 Neo4j** —— 下游拿到空 contexts 会自然产出空 `citations`/`graph_context`，
前端的 `addEvidence()` 在两者都空时不渲染依据面板，无需改前端。

链路（**现在的实现**）：

```text
问题 → _plan_retrieval()   ← 一次模型调用，出 5 行：检索 / 测点 / 历史 / 趋势 / 日志
         ├─ 检索=否 → 空 contexts，直接进 LLM
         └─ 检索=是 → _plan_queries()   ← 再让模型把问题拆成 N 路各自独立的检索 query
                       ↓
                     asyncio.gather 并发跑 N 路 RAG，按下标合并去重
                       ↓
                     Neo4j 上下文 → LLM
```

`_plan_is_complete` **只检查结构**（5 行都在吗），不判断内容；失败重试一次
（`_PLANNER_ATTEMPTS=2`），仍然失败才退回保守默认（照常检索）并把 `degraded`
放进返回值——**不静默退化**，因为「本来能答的问题变成答不了」比多查一次糟得多。

多路检索改成并发后，24 题卷子从 38.8s 降到 25.2s。**规划完全由模型完成，没有关键词启发式**——
早期那套「自称词 + 元信息关键词 + 短问题」的规则、以及 MCP 里的 `rag_query_plan` 工具
都已删除。

**拆几条由问题类型决定**，提示词要求先分类再拆：

| 问题类型 | 路数 | 实测 |
| --- | --- | --- |
| 具体事实 / 概念（「是多少」「什么是」） | **1 路** | 汽轮机超速保护动作值：1 路、10 条证据、7.5s |
| 列举（「有哪些」「包括哪些」） | 按子项各一条 | 汽轮机主保护有哪些：10 路、49 条证据、30.9s |
| 一段材料含多个并列问题 | 每个问题一条 | 整份试卷 |

列举型这条是**必须**的，不是锦上添花：单条查询只能覆盖一个子项。
实测问「汽轮机主保护有哪些」时，单路只捞回「主蒸汽温度保护」一类（查询字面撞上
「主…保护」），而「超速保护系统」「EH油压低保护」「差胀保护」这些逐项条目一条都进不了榜——
培训教材里 28 个讲保护的片段，单路只用到 **5 个**，拆子项后覆盖到 **15 个**。
漏掉的子项模型只能靠猜，而猜出来的答案从表面看不出来。

**远端 RAG 不可用时会自动压路数**（`_RAG_FALLBACK_ROUTE_CAP`，默认 2）。
两条路径的单请求延迟差 16 倍，而多路的耗时就是「路数 × 单请求延迟」：

```text
                      单请求      10 路
远端 GPU 节点          0.46s      2.36s      ← 生产路径
本机 CPU RAG           7.55s      75.5s      ← 回退，慢 32 倍
```

所以远端一挂，同一个问题从 2.4 秒变成 75 秒。落到回退路径时把路数压回 2 路，
并在规划结果里写明原因。探测只在路数超过上限时才做。

**注意 RAG 服务自身不能并发**（`_search_zvec` 每次请求拿两次 `_MODEL_LOCK`，
锁覆盖整个 GPU 推理）。实测两个节点**任意并发档位的吞吐都是恒定的**：

| | 吞吐 | 说明 |
| --- | --- | --- |
| 单节点 | 2.15 req/s | 延迟随并发线性增长，加速比 **1.0×** |
| 两节点 | 4.24 req/s | 正好翻倍——**并发能力来自节点数量，不是节点内部** |

所以「多路检索并发」实际是把请求摊到两个节点上，不是单个节点内部并行。
要再提容量只能加节点。

> 下面这张表是**旧版单次判定**（`_needs_retrieval()`，只出 1 个 token）的实测结果，
> 保留作为「该不该路由」这个判断本身的证据；函数本身已被上面的 5 行规划取代。

实测（同一批问题）：

| 问题 | 路由判定 | 证据 | 耗时 |
| --- | --- | --- | --- |
| 你的上下文长度是多少 | no | 0 | 1.4s |
| 你是什么模型 | no | 0 | 1.5s |
| 今天天气怎么样 | no | 0 | 1.4s |
| 汽轮机轴承温度高怎么处理 | yes | 10 | 7.4s |
| 轴封系统与真空系统的关系 | yes | 10 | 11.7s |

路由准确率 8/8，边界情况也对：`你好，请问汽轮机轴承温度高怎么处理` 会正确忽略寒暄去检索，
`我们聊聊别的好吗` 会跳过。

**设计要点**

1. **判断前置到检索之前**，这才省得掉检索开销（事后判断只能解决展示问题）。
2. **路由开销约 185ms**（无证据的小 prompt，只出 1 个 token），相对问答总耗时 5% 以内。
3. **失败一律保守处理**：调用异常、超时、回答无法解析成 yes/no 时，**一律照常检索**。
   漏检一次只是多花几百毫秒，误判成「不需检索」会让本该查知识库的问题答不出来。
4. **不用关键词启发式**。曾试过「自称词 + 元信息关键词 + 短问题」的规则，17 个用例能过，
   但它对换种问法就失效、且规则散落在代码里难维护。模型路由能处理任意说法。
5. **未检索时的提示词不同**：不拼「检索证据：无」这种框架 —— 否则模型会顺着说
   「证据中没有」，而不是直接依据自身设定回答。同时保留防幻觉指令：涉及本系统具体
   配置数值时，不确知就说明无法确认，不要给估计值。

开关：`HDW_RETRIEVAL_ROUTER=false` 可退回「一律检索、不读测点」（无需重建镜像）。

### 7.5 实时测点（MCP sis_point）

**问题**：MCP 有一批工具，但 `/qa/query` 从不调它们。问「现在主汽温度多少」只会拿到
文档里写的定值，不是实测值。

**方案**：和检索路由**合并成同一次规划调用**。那次调用出 **5 行**——检索 / 测点 / 历史 /
趋势 / 日志——历史与趋势按 `关键词|起|止[|间隔秒]`、日志按 `起|止[|重大]` 解析。
测点、历史、趋势、日志并行去查 SIS 与 LIEMS，结果作为**独立区块**进 prompt ——
实时数据与文档证据分栏，不混为一谈。

```text
问题 → _plan_retrieval() 一次出 5 行
        ├─ 测点 → point_query_current_value（并行，有上限）
        ├─ 历史 ┐
        ├─ 趋势 ┤→ point_query_history_series
        ├─ 日志 → log_query_recent | log_query_range | log_query_major_events
        └─ 检索 → _plan_queries → 多路并发 RAG + Neo4j
        ↓
      LLM（实时数据 与 检索证据 分区块）
        ↓
      前端：实时测点卡片 + 趋势图（内联 SVG）；citations 为空也照样显示
```

实时数据抓取与多路检索是**并行**的（`asyncio.create_task` + `gather`），不是串行等。

实测：

| 问题 | 检索 | 测点 | 结果 |
| --- | --- | --- | --- |
| 现在主汽温度是多少 | ✗ | 主蒸汽温度 | 24.265587 ℃ @2026-09-12T03:41:42 |
| 闭式冷却水泵的振动值 | ✗ | 闭式冷却水泵AX向振动 | 0.079346 μm |
| 凝结水和主蒸汽温度分别是多少 | ✗ | 两个 | 46.77 ℃ + 24.27 ℃ |
| 轴封系统的作用是什么 | ✓ | — | 正常走 RAG |
| 你是什么模型 | ✗ | — | 直接回答 |

**SIS 配置**

MCP 的 `runtime.py` **纯读环境变量**（不是配置文件），需要的键在 `Configs/.env`：

```bash
HDW_SIS_BASE_URL / HDW_SIS_LOGIN_URL / HDW_SIS_USERNAME / HDW_SIS_PASSWORD / HDW_SIS_LANGUAGE
```

端点、超时、分片大小都有与 SmartGasTurbine 一致的默认值，不用写。改完要
`--force-recreate hdw-mcp` 才生效（环境变量在容器启动时注入）。

**设计要点**

1. **只接只读工具。** MCP 里有 `alarm_acknowledge`（报警确认）和
   `edge_device_power_action`（设备电源操作）两个**写操作**，工业场景不允许模型自动调。
2. **规划合并成一次调用**，不是两次 —— 两者都在回答「这个问题需要什么」，分开是白花一次往返。
3. **单点失败不拖垮回答**：`_fetch_live_points()` 并行取数，某个测点失败只记进 `live_errors`，
   文档问答照常。缺实时数据不该让整个回答失败。
4. **关键词对不上测点名时的三级兜底**。用户的说法和测点表的命名经常对不上 ——
   表里叫「凝汽器液位」，用户问「凝汽器水位」，整词匹配直接落空。处理链：

   ```text
   ① current_value(query_text=关键词)          快路径，命中就返回
   ② search_points(关键词)                     拿候选
      搜不到 → 逐级去掉尾字再搜（凝汽器水位 → 凝汽器水 → 凝汽器）
   ③ 把候选列表交给模型，由它选最符合意图的 KKS → current_value(kks=选中)
   ```

   ②的「去尾字」只放宽**搜索范围**，选哪个仍由③的模型从真实候选里判断 ——
   所以不是硬编码同义词表（水位→液位 那种），换机组、换命名习惯都不用改。
   即便如此，`一号机凝汽器水位` 这种带前缀的长关键词仍会让测点检索的排序跑偏，
   所以 planner 被要求只产出 2-6 字的短关键词（只描述物理量，不带机组号）。
5. **模型会主动说明边界**：接了实时数据但不检索文档时，模型会说明「无法提供历史趋势、
   报警阈值」，因为那些在文档里 —— 这是期望行为，不是缺陷。
6. **答案模型是最后一道防线**：若选中的测点与问题意图不符（例如问水位却拿到真空度），
   答案模型会主动指出「该测点为 X，不是 Y」。这是最后的安全网，不是可以依赖的常态。
7. **实时数据与文档证据分栏**：live 数据是「此刻的实测值」，文档是「规程里的定值」，
   prompt 里分两个区块，回答里也要说清哪句来自哪个。
8. **采集时间统一按北京时间展示**。SIS 返回的是 UTC ISO-8601（`2026-09-12T04:00:45+00:00`），
   直接给模型和界面看既反直觉，模型还会自己补一句时区换算。换算只在
   [`_fetch_live_points()` 的 `shape()`](HDW_Orchestrator/industrial-qa-api/app/main.py)
   里做一次（`_format_live_time()`，偏移量 `HDW_LIVE_TIME_OFFSET_HOURS`，默认 8），
   输出 `YYYY-MM-DD HH:MM:SS` 无后缀。`_prompt()` 只展示、不再二次换算 ——
   重算会把已经本地化的值再当 UTC 加 8 小时。无时区的输入按 UTC 解释（SIS 侧就是 UTC，
   当成本地时间会少算 8 小时）；解析失败原样返回，不丢采集时间。
**其他工具组的现状**

| 工具组 | 状态 |
| --- | --- |
| liems_log | **已接入**。`/qa/query` 会调 `log_query_recent` / `log_query_range` / `log_query_major_events`，数据在 `HDW_DataFoundation/Mapping/Log_Fetching/data` |
| thermal | 服务 ready，但无匹配测量点 |
| rtsp | 服务 ready，但 `HDW_RTSP_STREAMS_JSON` 为空 → 0 路流 |
| lstm | 服务 ready，但 0 个模型 |
| alarm / edge_device / custom_rule | 缺数据文件，不可用 |

### 7.6 流式返回（SSE）

`POST /qa/query` 默认仍返回一个完整 JSON，老调用方不受影响；传 `stream: true` 时改成
`StreamingResponse`，按事件流推送：

```text
{"stage":"vision"|"retrieve"|"generate","label":"…"}   进度事件（generate 带 evidence 条数）
{"heartbeat":true}                                      心跳，每 HDW_SSE_HEARTBEAT 秒一条
{"result": {…}}                                         最终结果，结构和不流式时完全一样
{"error": {"message":"…"}}                              流内错误
```

为什么要加：一次问答实测要 25–77 秒，中间没有任何反馈时浏览器会以为卡死，
代理也可能提前掐断长连接。

三个实现细节值得记住：

- **前端用 `fetch` + `ReadableStream`，不是 `EventSource`**——后者只支持 GET，
  而 `/qa/query` 必须 POST。
- **校验错误仍走真实 HTTP 状态码**（401/403/422 在开流之前就定了），
  不会变成流里的一条 error 事件。
- nginx 侧要 `X-Accel-Buffering: no`，否则响应会被缓冲到结束才吐出来，流式等于白做。

### 7.7 图片上传与权限

上传图片提问默认**只对管理员开放**，普通用户要由管理员在模型管理页打开
`image_upload_for_users` 才能用。

权限位由 `_permissions_for_user()` 统一给出，前端只负责置灰，**真正的拦截在服务端**
（带图请求会被直接 403）。界面是公开的，只靠前端置灰等于没拦。

这个开关存在 `HDW_Runtime/model-config/config.json` 的 `permissions` 段，**不在 `.env` 里**——
它属于运行时模型配置，不是部署参数。

### 7.8 图片来源优先级

问答链路的图片转写有**三种来源**，由模型配置里的 `vision_priority` 统一驱动
（1–4 项，每项 `{priority, kind, mode, model, enabled}`）：

| kind | 走什么 |
| --- | --- |
| `chat` + `mode=online` | 在线对话模型 |
| `chat` + `mode=offline` | 本地 llama（需要 `mmproj` 投影器） |
| `mineru` | 本地 MinerU `/file_parse` |

两条视图读同一份列表：`_ordered_vision_candidates()` 决定**转写**用哪个来源
（接受 chat 和 mineru），`_vision_candidates()` 决定**答题**用哪个视觉模型
（只接受 chat，选不出就 503）。转写结果会拼进检索问题，原图仍然同时给答题模型，
提示词里明写「图片文字是自动识别，不要当证据引用」。

**缺省 `kind` 当 `chat`**：老配置里没有这个字段，不这样兼容的话升级后所有视觉候选都会被判非法。
`mmproj` 是**启动参数**，改了不重启本地推理不生效。

### 7.9 历史相关性选留

**问题**：历史是一轮一轮线性堆上去的，而解码速度直接由上下文长度决定——短上下文约
150 t/s，5–20k 掉到约 130，5 万以上只剩约 100。对话越长，每个回答越慢。

**方案**：每次提问前多一次小调用，判断哪些历史轮次与当前问题**主题相关**，无关的不进 prompt。
三轮对话「什么是汽轮机 / 电气差动保护是什么 / 汽轮机超速有什么后果」，问第三轮时第二轮不会进 token。

几个刻意的设计：

- **按轮选，不按单条消息选**。一轮 = 一个 user + 其后紧邻的 assistant；
  只选一条会出现「留了答案、丢了问题」这种半截状态。
- **摘要那条 system 消息永远保留**——它是压缩产物，丢了等于丢整段历史。
- **失败方向朝「全留」倒**。选择器超时、报错、输出解析不出来，一律退回全量历史：
  **选错只是慢一点，丢错上下文是答错**。特别地，「模型明确说『无』」（合法结论，返回空）
  和「输出看不懂」（故障，全留）必须分开——混起来就会在解析失败时静默丢掉整段历史。
- **选留放在 `ContextManager.prepare()` 之后**。`context_kept_from` 是存盘的绝对下标，
  在它之前过滤会让存回去的下标指错位置。
- 只把最近 `HDW_HISTORY_SELECT_MAX_TURNS` 轮交给选择器，更早的一律丢弃——这是**有意的
  近因策略**，也让选择器自身的输入有上界。

**和压缩的关系**：`ContextManager` 的 75% 阈值压缩**保留为兜底**——选留之后历史仍然超长
才由它出手。选留是日常手段，每轮都跑。实测最大的一个会话有 82 轮、103,075 token 的历史，
选留后只剩 1,685（省 98%）。

结果记录在响应的 `llm.history_select` 里（选了几轮、丢了哪几轮、花了多久、有没有失败、
前后 token 数），排查时先看这个。`HDW_HISTORY_SELECT=0` 可一键退回全量历史。

## 8. 对话历史文件存储

### 8.1 存储位置

主机目录：

```text
<项目根>/HDW_Runtime/chatdata
```

容器内目录：

```text
/data/chatdata
```

`hdw-qa-api` 通过 Compose 挂载两者，环境变量为：

```text
HDW_CHATDATA_ROOT=/data/chatdata
```

每个会话一个 JSON 文件，**按用户分目录**，例如：

```text
HDW_Runtime/chatdata/<用户code>/<会话id>.json
HDW_Runtime/chatdata/046/local-1788939310439-clb1h6.json
```

### 8.2 文件结构

```json
{
  "id": "local-1788679371-ab12cd",
  "title": "热机运行规程",
  "group": "今天",
  "pinned": false,
  "created_at": "2026-09-06T15:00:00+08:00",
  "updated_at": "2026-09-06T15:02:00+08:00",
  "messages": [
    {
      "role": "user",
      "content": "启动前需要检查什么？",
      "citations": [],
      "graph_context": [],
      "emotion_id": null,
      "created_at": "2026-09-06T15:00:00+08:00"
    },
    {
      "role": "assistant",
      "content": "回答正文",
      "citations": [],
      "graph_context": [],
      "emotion_id": "33",
      "created_at": "2026-09-06T15:02:00+08:00"
    }
  ],
  "context_summary": "",
  "context_kept_from": 0,
  "context_token_estimate": 0,
  "context_compressed_at": null,
  "last_inference_mode": "offline"
}
```

**user 消息也带 `citations` / `graph_context` / `emotion_id` 三个字段**（默认空），
不是只有 assistant 有——所有消息是同一个结构。

后面那五个 `context_*` / `last_inference_mode` 字段是上下文管理的状态：

| 字段 | 含义 |
| --- | --- |
| `context_summary` | 75% 阈值压缩的产物。**实测从未被写过**——压缩线是 0.75 × 262,144 = 196,608 token，而最大的会话也只有 118k |
| `context_kept_from` | 已摘要掉的轮次边界，**存盘的绝对下标** |
| `context_token_estimate` | 上次保存时的历史 token 估算 |
| `context_compressed_at` | 上次压缩时间 |
| `last_inference_mode` | 上次用的是在线还是离线，跨模式时会触发一次压缩 |

历史相关性选留读的就是这个文件——**原文一个字不动**，只是决定哪些轮次进 prompt。

保存采用临时文件写入后 `replace()`，避免浏览器刷新或进程中断时留下半个 JSON。会话 ID 只允许字母、数字、下划线和短横线，防止路径穿越。

### 8.3 会话接口

QA API 提供：

```text
GET    /conversations
GET    /conversations/{id}
PUT    /conversations/{id}
PATCH  /conversations/{id}
DELETE /conversations/{id}
```

对外问答 API（`hdw-api`，给别的项目调用）另有一套，**不做上下文管理**，
留档在 `chatdata/api/`，详见 [HDW_API/README.md](HDW_API/README.md)：

```text
POST   /api/v1/ask                    提问，同步返回回答（不带证据）
GET    /api/v1/sessions?limit=50      列出留档会话
GET    /api/v1/sessions/{id}          某会话的全部问答
DELETE /api/v1/sessions/{id}          删除某会话留档
GET    /health                        探活（无需鉴权）
```

它与 QA API 的关系：`hdw-api` 不实现问答，而是带上服务间密钥调
`qa-api` 的 `POST /internal/qa/query`（见 §7 末尾），把回答落盘后返回。

WebUI 行为：

- 启动时从服务端读取历史摘要。
- 点击左侧历史记录时读取完整 JSON 并恢复消息。
- 新问题先写入用户消息。
- QA 返回后再写入回答、引用和图谱上下文。
- 右键置顶调用 `PATCH`。
- 右键删除调用 `DELETE`，成功后删除当前 JSON。
- 多台局域网电脑访问同一个 WebUI 时共享同一目录。

浏览器 `localStorage` 现在只用于主题模式 `D/N/A`，不再作为对话记录来源。

当前方案适合单机、少量局域网用户。进程内全局锁只解决单个 QA API 进程的并发写入；未来出现多副本、多主机或高并发时，再迁移到 PostgreSQL 或对象存储，不要提前引入复杂存储层。

## 9. 部署前提

建议部署主机具备：

1. Linux x86_64。
2. Docker Engine 和 Docker Compose v2。
3. NVIDIA 驱动和 NVIDIA Container Toolkit，若启用 `gpu` profile。
   **不需要单独安装 CUDA toolkit** —— `HDW_Inference/llama/llama.cpp-upstream/build-cuda-multi/`
   里已随包携带所需的 CUDA runtime 库，`start.sh` 会按自身位置设好 `LD_LIBRARY_PATH`。
4. systemd 用户管理器可用（`systemctl --user`）——llama 与资源协调器都是用户级服务，
   容器或精简系统里通常没有，`deploy.sh` 会前置检查并明确报错。
5. 足够的磁盘空间：Qwen、BGE、MinerU 模型和运行索引都不小。完整部署约需 60G 以上。
6. 能访问本机 Docker 镜像源或已经准备好基础镜像。
7. 局域网网卡有稳定 IP，防火墙允许 WebUI 端口。
8. Qwen、BGE-M3 和 reranker 目录完整——缺失时 `deploy.sh` 会从魔搭自动补齐。

以上条件由 `deploy.sh` 逐项检查，不必手工核对；换机器部署见 §10.4。

不要把以下内容写入公开代码仓库：

- SIS 用户名和密码。
- Neo4j、PostgreSQL、Redis 密码。
- `HDW_INTERNAL_API_KEY`。
- Keycloak 和 Langfuse 密钥。
- 企业内部文档和聊天 JSON。

## 10. 首次部署

### 10.1 准备配置

```bash
cd <项目根>
cp Configs/.env.example Configs/.env
```

修改 `Configs/.env` 中至少这些项目：

```text
HDW_PROJECT_ROOT=<项目根>
HDW_RUNTIME_ROOT=<项目根>/HDW_Runtime
HDW_WEBUI_BIND=0.0.0.0
HDW_WEBUI_PORT=3000
POSTGRES_PASSWORD=<strong-password>
REDIS_PASSWORD=<strong-password>
NEO4J_AUTH=neo4j/<strong-password>
HDW_INTERNAL_API_KEY=<strong-key>
```

`HDW_WEBUI_BIND=0.0.0.0` 代表监听所有网卡。更严格的做法是填**服务器实际对外提供
WebUI 的那张网卡的地址**，避免顺带监听到 RTSP 摄像头所在的网段。
`deploy.sh` 会自动探测默认路由出口地址填入，也可以 `--bind <ip>` 覆盖。

### 10.2 检查

```bash
bash Scripts/init.sh
bash Scripts/prepare_dirs.sh

# 校验 compose 配置。**必须带上 profile**：所有 service 都声明了 profile，
# 不带 profile 时输出的是 `services: {}`，命令返回 0 但什么都没校验。
# 这是个会骗人的假阳性，别照抄网上那种不带 profile 的写法。
docker compose --env-file Configs/.env -f Configs/docker-compose.yml \
  --profile base --profile knowledge --profile web config
```

`init.sh` 会检查实际 Compose 使用的：

```text
HDW_Engines/LLM_Models/Qwen3.8-27B-GSQ/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
HDW_Engines/RAG_Models/bge-m3
HDW_Engines/RAG_Models/bge-reranker-v2-m3
```

模型缺失不用手工拷：`bash Scripts/fetch_models.sh` 会从魔搭按字节数校验下载。
第三方源码同理，见 §10.8。

### 10.3 一键启动

项目启动入口只有 `Scripts/start.sh`。根目录没有 `start.sh`，所以在项目根目录执行 `bash start.sh` 会提示文件不存在。

```bash
cd <项目根>
bash Scripts/start.sh
```

脚本会根据自身位置定位项目根目录，使用相对项目结构读取 `Configs/.env`、`Configs/docker-compose.yml` 和 `Scripts/prepare_dirs.sh`，因此不依赖当前终端所在目录，也不写死项目绝对路径。

默认启动当前 Compose 定义的 9 个容器，并拉起 1 个用户级 systemd 的本地 llama 服务：

```text
PostgreSQL、Redis、Neo4j、RAG、MinerU、Ingest、MCP、QA API、工业 WebUI，以及 llama.cpp 本地 LLM
```

脚本会先单独执行一次 `docker compose build hdw-rag`，再执行
`docker compose up -d --build --remove-orphans`。先建 rag 是因为 `hdw-mineru` 的
Dockerfile 第一行是 `FROM hyperdrivewave-hdw-rag:latest`，而 compose 里 mineru
没声明对 rag 的 `depends_on`，`up --build` 的构建顺序没有保证——全新机器上会随机失败。

重启系统后再次执行同一条命令即可恢复服务；已存在的镜像会复用缓存，只有源码或 Dockerfile 变化时才会重新构建。

如只需要调试 WebUI 和基础问答链路，可以临时覆盖 profile：

```bash
HDW_COMPOSE_PROFILES="base knowledge web" bash Scripts/start.sh
```

等容器启动后检查：

```bash
bash Scripts/status.sh
bash Scripts/healthcheck.sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8001/health
```

预期：

- QA API 返回 `status: ok`。
- llama 服务为 active，`http://127.0.0.1:1919/health` 可访问，并返回当前 GSQ/MTP 模型。
- RAG 状态包含 `bge-m3`、`bge-reranker-v2-m3` 和 Zvec。
- MinerU 返回 `max_concurrent_requests: 1`。
- Neo4j 健康检查通过。

浏览器访问：

```text
http://<服务器172网卡IP>:3000
```

不要使用：

```text
http://192.*:3000
```

除非那确实是服务器提供 WebUI 的网卡；摄像头 RTSP 所在网卡不等于 WebUI 网卡。

### 10.4 部署到另一台服务器（一键）

把项目拷到新机器，跑一条命令即可完成部署，模型缺失时自动从魔搭补齐。

**第一步：在源机打包。** 整个文件夹 253G，其中 239G 是当前用不到的备份模型、
另有 34G 是历史下载残块，直接 `rsync` 整个目录会白搬 270G。

```bash
bash Scripts/pack_hdw.sh --out DIR                 # 带在用的 3 个模型，约 25G
bash Scripts/pack_hdw.sh --app-only --out DIR      # 只带代码，约 7G
bash Scripts/pack_hdw.sh --remote-rag --out DIR    # 只打远端 RAG 节点要的
```

不带 `--out` 时用 `--tar <文件>` 打成单个压缩包。打包前会先算需要多少空间，
不够会直接拒绝——**中途空间不足会产出静默不完整的包，只有到目标机才发现**。

**第二步：目标机部署。**

```bash
cd <目标机上的项目目录>
bash Scripts/deploy.sh
```

脚本按阶段执行：前置检查 → 环境探测 → 建目录 → 渲染配置 → 补模型 →
构建镜像 → 调用 `Scripts/start.sh` → 验收。想先看会改什么而不实际动手：

```bash
bash Scripts/deploy.sh --dry-run
```

**`--dry-run` 一个文件都不写。** 这个退出点必须排在**任何写操作之前**——
早先的版本把它放在"装 systemd 单元 + `daemon-reload`"之后，结果是从一个临时
目录跑 `--dry-run` 时，把真实的 `~/.config/systemd/user/` 下两个单元改写成了
指向那个临时目录。当前进程还在跑旧的，看起来一切正常，但**下次重启就会去跑
一个已经不存在的路径**。现在 `--dry-run` 在阶段 1 结束就退出，连决策快照
`plan.env` 都不写。

常用开关：

| 开关 | 用途 |
| --- | --- |
| `--network online\|mirror\|offline` | 网络模式。不指定则交互式询问；非交互默认 `online` |
| `--proxy <url>` | `mirror` 模式下的 HTTP 代理 |
| `--bind <ip>` | 手动指定 WebUI 绑定地址，默认自动探测默认路由出口 IP |
| `--port <键>=<端口>[,...]` | 直接指定端口，跳过交互（见 §10.11） |
| `--skip-models` | 模型已备好，跳过检查与下载 |
| `--with-frp` | 一并安装 frpc 单元（只安装不启用，启停仍由控制中心控制） |
| `--takeover` | 显式声明接管本机上已有的另一份安装（见下） |

**systemd 单元是机器级的，不是项目级的。** 它们装在
`~/.config/systemd/user/`，路径固定，**不随项目目录走**。所以同一台机器上
从 A 目录跑一次 `deploy.sh`，会把正在运行的 B 目录安装顶掉：单元被改指到 A，
当前进程仍在跑 B（无感），但下次重启就切过去了——如果 A 后来被删掉，
服务就再也起不来。

为此 `deploy.sh` 在装单元前会**对比单元里现有的 `WorkingDirectory=`**：

- 指向的就是本次目录 → 直接继续，不打扰
- 指向别处 → 打印两边路径并要确认；**非交互环境直接报错退出**，
  不会因为 `confirm()` 在非 tty 下默认通过而静默接管。
  确认接管要显式加 `--takeover`。
- 拒绝确认 → 退出，并给出停掉旧安装的命令

在临时目录里试跑部署脚本时，务必用 `--dry-run`，或者把 `XDG_CONFIG_HOME`
指到临时目录（单元会写到那里，碰不到真实安装）：

```bash
XDG_CONFIG_HOME=/tmp/hdw-test bash Scripts/deploy.sh    # 隔离，不碰真实单元
```

`XDG_CONFIG_HOME` 是 `user_unit_dir()` 的取值来源，改了它就等于换了整个
单元目录——**这是在不影响本机运行项目的前提下测试部署脚本的正确做法**。
注意隔离的只是单元文件；`docker compose` 仍会绑同样的宿主端口，
所以真要跑完整流程还是得在另一台机器上进行。

**目标机前置条件**：Docker Engine + Compose v2、systemd 用户管理器可用
（llama 和资源协调器都是用户级服务）、若用 GPU 则需 NVIDIA 驱动 +
nvidia-container-toolkit。**不需要装 CUDA toolkit** ——
`build-cuda-multi/` 里已随包携带所需的 runtime 库。

**关于路径：全部相对，项目可以随便移动。** 代码本身是自定位的
（脚本用 `BASH_SOURCE` 推根目录、compose 用 `../` 相对挂载、`start.sh` 自动选后端
并按自身位置设 `LD_LIBRARY_PATH`），所以部署脚本做的是**删掉 `.env` 里的绝对路径覆盖**，
把控制权还给这些默认值，而不是写一批新的绝对路径进去。
项目在家目录下时 systemd 单元用 `%h/...` 形式，家目录内移动无需重装单元；
换机器或改路径后重跑一次 `deploy.sh` 即可恢复。

**显卡适配**：自动探测 `compute_cap` 并据此选后端，判据是"编的架构能不能在目标卡上跑"
而不是"目录存不存在"。多架构构建 `build-cuda-multi/` 覆盖 sm_80/89/90/120，
4090/A100/H100/5090 都能用 CUDA + MTP；没有匹配的 CUDA 构建时回退 Vulkan
（MTP 失效、吞吐降 2-3 倍），并打印重编命令。无 NVIDIA 卡时仍会拉起 llama，
但走 CPU 并明确标记为慢。

### 10.5 远端 RAG 节点

远端 GPU 机可以只部署 RAG 服务，不需要主站那套 WebUI/数据库/llama。

```bash
cd <远端机上的项目目录>
bash Scripts/deploy_remote_rag.sh               # 自动探测卡数，自动选镜像来源
bash Scripts/deploy_remote_rag.sh --dry-run     # 只生成 compose 不启动
```

服务数量按**实际卡数**决定：0 卡起 1 个 CPU 服务、1 卡起 1 个、2 卡起 2 个。
原 `HDW_Inference/RAG_Service/remote-compose.yml` 写死了两张卡和 IP，
脚本改为按探测结果生成到 `HDW_Runtime/remote-rag/`（不写回仓库，
免得这个文件夹拷到下一台机器时带着上一台的 IP 和卡号）。

镜像三种来源，不指定时自动选：

| 方式 | 说明 |
| --- | --- |
| `--image-tar <文件>` | 用包内的 `docker save` 产物 |
| `--image-from <user@host>` | 从主站直接 `docker save \| ssh \| docker load`（默认优先） |
| `--image-mode build` | 远端本地构建（需要能连公网 PyPI） |

部署完成后脚本会打印主站要改的那一行，**必须照抄**：

```text
HDW_RAG_REMOTE_URLS=http://<远端IP>:8001
```

单卡远端只监听 8001，若主站仍列着 8003，每次问答都会有一半请求打到不存在的端口，
每个都要吃一次 `HDW_RAG_CONNECT_TIMEOUT`（默认 3s）。

然后回主站推数据并重建远端索引：

```bash
SSHPASS='<远端密码>' bash Scripts/sync_remote_rag.sh
```

注意该脚本用的是主站 `.env` 里的 `HDW_REMOTE_RAG_SSH_TARGET` / `HDW_REMOTE_RAG_ROOT`，
换远端时要同步改。

### 10.6 验收

`healthcheck.sh` 是**日常巡检**，`deploy_verify.sh` 是**部署验收**，两者职责不同：

```bash
bash Scripts/healthcheck.sh            # 快，日常看
bash Scripts/deploy_verify.sh          # 全，部署后/交接前
bash Scripts/deploy_verify.sh --quick  # 跳过端到端问答
bash Scripts/deploy_verify.sh --deep   # 额外做重启演练
```

验收脚本刻意**不信任 `/health`**，因为项目里有两处会误导：

- RAG 的 `GET /health` 返回的是写死的静态字典，只查路径存在性、**不加载模型**
  （模型是首次 `/embed` 才懒加载），`status` 恒为 `ok`；
- QA API 的 `/health` 里 `"status":"ok"` 也是硬编码字面量。

所以验收会真调一次 `/embed`（断言维度 1024）、真跑一次生成
（读 `usage.completion_tokens` 而不是 `content` —— Qwen3.8 是思考模型，
token 可能全被 `reasoning_content` 吃掉，只看 `content` 会误判为失败）、
并扫 journal 里的 `no kernel image is available`（CUDA kernel 不匹配时
`/health` 照样通过，只有真推理才暴露）。

端到端问答需要一个登录会话。不指定时脚本会取 `HDW_Security/auth/auth.csv`
里的第一个账号换取会话（**只打印账号名，不打印登录码**），也可用
`--login-code <code>` 显式指定。

### 10.7 镜像源（国内网络建议先做）

```bash
bash Scripts/setup_mirrors.sh --check   # 只报当前状态和会改什么，不动手
bash Scripts/setup_mirrors.sh           # 交互式：显示检测到的默认值，问 Y/n
bash Scripts/setup_mirrors.sh --yes     # 全部采用检测到的值
```

覆盖 **Docker registry mirror / pip / npm / apt** 四项。默认值取自**本机现有配置**
（不是写死的清单）——目的是让新机器配得和现在这台一样。选 `n` 可逐项手工填写。

每一项都是幂等的：内容没变就跳过（**不会白白重启 Docker**）；改前备份到
`<原文件>.hdw-bak-<时间戳>`；apt 改完会跑一次 `apt-get update` 验证，失败自动回滚。

也可以只跑一部分：

```bash
bash Scripts/setup_mirrors.sh --only docker,pip
bash Scripts/setup_mirrors.sh --docker "https://mirror.example.com" --pip "https://mirrors.ustc.edu.cn/pypi/simple/"
```

一键部署时用 `--setup-mirrors` 带上这一步；选 `--network mirror` 但宿主机还没配
registry-mirrors 时会提示你先跑它。

**关键限制：`docker build` 里的 pip 不读宿主机的 `/etc/pip.conf`。**
构建期的源只能通过 build-arg 注入，所以 5 个 Dockerfile 都加了：

```dockerfile
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt
```

`docker-compose.yml` 已经把这 6 个服务的 `build.args` 接上 `.env` 的 `PIP_INDEX_URL`，
所以**切换构建期源只要改 `.env` 一处**：

```bash
# 回官方源
PIP_INDEX_URL=https://pypi.org/simple
```

改完需要 `docker compose build <服务>` 重建（build-arg 变化会失效对应层的缓存）。
MinerU 和 FreeToken 的 Dockerfile 另外支持 `--build-arg APT_MIRROR=<url>` 换 apt 源。

### 10.8 第三方依赖（vendor）

项目用到 12 个第三方仓库，但**它们不进版本库**——仓库里只留一份锁文件，
部署时按需克隆回原路径（项目结构完全不变）。

```bash
bash Scripts/fetch_vendors.sh                 # 只拉默认需要的 4 个（见下）
bash Scripts/fetch_vendors.sh --all           # 全部 12 个（含 8 个预留仓库）
bash Scripts/fetch_vendors.sh --with dify,n8n # 默认的 **加上** 这两个
bash Scripts/fetch_vendors.sh --only MinerU   # 只要这一个
bash Scripts/fetch_vendors.sh --check-only    # 只报状态，不克隆
bash Scripts/fetch_vendors.sh --prefer gitee  # 优先走 gitee（国内网络强烈建议）
```

默认拉的 4 个是：`llama.cpp-upstream`（推理引擎源码，**必须**，见 §10.10）、
`MinerU`（`hdw-mineru` 镜像的构建上下文）、`aora-bot`（WebUI 情绪球）、
`python-sdk`（`hdw-mcp` 镜像构建依赖）。其余 8 个当前零引用，要用时显式 `--with`。

**为什么这么做**：这 12 个目录全部拉齐是 **4.8 GiB，其中 2.7 GiB 是 `.git`**。
把 git 历史搬进项目仓库纯属浪费，而且 12 个仓库里有 8 个现在根本没被引用。
注意大头的分布很悬殊——单独一个 `llama.cpp-upstream` 就占 1.4 GB（源码树本身
就大），其余 11 个合计才 3.5 GiB。默认那 4 个里不含 llama.cpp 时只要 62 MB。

**版本锁在 [`vendor/vendor.lock`](vendor/vendor.lock)**，制表符分隔，每行是
`路径 / commit / 是否默认拉 / 分支 / URL`。用的是 commit SHA 而非分支名——
实测上游推进很快，其中 7 个仓库在克隆后几周内分支头就变了，只写分支名不可复现。

**URL 一栏是「规范上游在前、镜像兜底在后」**，按顺序尝试。这不是摆设：
本机实测 `github.com` 的按 commit 拉取会**无限挂起**（GitHub 对未广告的 SHA
要做一次完整可达性遍历，大仓库上能卡几分钟），靠回退到 `gitee.com` 才装上。

**镜像 URL 可以带自己的 commit，写法是 `URL#commit`**：

```text
https://github.com/ggml-org/llama.cpp,https://gitee.com/mirrors/llama-cpp#b387ddfd84b4...
```

这不是可选的花样，而是**镜像兜底能成立的前提**：镜像是别人的定时同步，
**永远不会有**我们锁定的那个 commit。原先"多个 URL 试同一个 commit"的模型对
镜像不成立，把镜像 URL 直接追加进去只会每一级都失败——看起来配了兜底，实际
一点用没有。

不带 `#` 的 URL 用行首那个锁定 commit（所以老写法完全兼容）。分隔符选 `#`
而不是 `@`，因为 SSH 写法 `git@host:path` 里本来就有 `@`。

走到镜像兜底时脚本会**明确警告**，列出「锁定版本 → 实际拿到的版本」并
要求重跑验收；已有目录落在镜像 commit 上时判为 `就绪·镜像版本`（**不重克隆**，
否则每次跑都会重拉一遍）。`--check-only` 会把每个 URL 各自会检出的 commit
都列出来。

国内网络建议直接加 `--prefer gitee` 把 gitee 提到前面，省掉那次白等：

```text
（不指定）        冷克隆 3 个依赖 54s，其中 45s 花在 github 的超时上
--prefer gitee    同样的 3 个依赖 22s，全部走 gitee 浅取
```

只影响本次运行，不改锁文件——锁文件保持规范上游在前，便于他人复用和溯源。

克隆走三级回退，**层级为主、URL 为辅**（先把最便宜的浅取对所有镜像试一遍，
都不行才升级）：

| 级别 | 做法 | 代价 |
| --- | --- | --- |
| 1 | `git fetch --depth 1 origin <sha>` | 几十 MB，最快 |
| 2 | `git clone --filter=blob:none` + checkout | 中等 |
| 3 | 全量 `git clone` + checkout | 最慢，但一定成功 |

第 1 级依赖服务端支持取任意 SHA（`uploadpack.allowReachableSHA1InWant`）。
**GitHub 和 Gitee 都支持**（Gitee 是 2026-09-12 实测的：对
`gitee.com/mirrors/llama-cpp` 浅取任意 SHA，9 秒成功）——所以镜像兜底通常
就停在最快的那一级，不会退化成全量克隆。
每级都有超时（默认 45s / 150s / 1800s，可用 `HDW_VENDOR_T*_TIMEOUT` 调），
某个 host 三级全挂后**本次运行内不再尝试它**。

**项目自有的 3 个文件**放在 [`vendor/overlays/`](vendor/overlays/)，克隆后自动拷回原位：

```text
HDW_Knowledge/MinerU/docker/hyperdrivewave-pipeline.Dockerfile
HDW_Inference/FreeToken/Dockerfile
HDW_Inference/FreeToken/.dockerignore
```

它们必须在 build context 内部（Docker 要求 Dockerfile 在 context 里，
compose 的 `dockerfile:` 也是相对 context 解析的），所以不能只存一份在外面改指向。

**升级某个依赖**：

```bash
cd <依赖路径> && git fetch && git log --oneline origin/<分支> | head
# 挑好 commit，填进 vendor/vendor.lock 第 2 列
bash Scripts/fetch_vendors.sh --with <名字>
bash Scripts/deploy_verify.sh
```

**注意 `zvec` 不在此列**——它曾经被 vendored 在 `HDW_VectorDB/zvec/`，但**从未被使用**
（RAG 实际用 PyPI 的版本，该目录既没被 COPY 也没被挂载）。现在直接在
`HDW_Inference/RAG_Service/requirements.txt` 里钉 `zvec==0.7.0`。

**许可证**：`vendor/` 下都是别人的代码，各自遵循上游许可证。本项目自有代码
采用 Apache License 2.0，范围与第三方边界见 §18。

### 10.9 推送到版本库前的凭据扫描

**为什么不能只靠正则**：本项目实际发生过一次漏检——第一次扫描报"无密钥"，
但 `README_FRP.md` 里就躺着一个**明文面板账号密码**。它是散文格式
（`` `admin` / `口令` ``），不是 `password=xxx` 这类赋值，正则扫不到。

**可靠的做法是反过来做**：从真实的 `Configs/.env` 里提取凭据**值**，
拿这些值去反查每一个待提交文件。

```bash
cd <项目根>
python3 - <<'PY'
import pathlib, subprocess, re

# 1) 从真实 .env 提取凭据类键的值
env = pathlib.Path("Configs/.env")
creds = {}
for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    k, v = k.strip(), v.strip().strip('"').strip("'")
    if len(v) < 8:                       # 短值会产生大量巧合命中
        continue
    if v.lower() in ("change_me", "local-dev-key") or "change_me" in v or v.startswith("your_"):
        continue                           # 占位符不是凭据
    if not re.search(r"(KEY|TOKEN|SECRET|PASSWORD|AUTH)", k, re.I):
        continue
    creds[k] = v

# 2) 反查每一个**已提交**的文件（用 git show 读，才是真正会上传的内容）
names = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
hits = []
for n in names:
    try:
        text = pathlib.Path(n).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for k, v in creds.items():
        if v in text:
            hits.append((n, k))

print(f"  {len(names)} 个文件，{len(creds)} 个凭据值，命中 {len(hits)}")
for n, k in hits:
    print(f"    ❌ {n}  含 {k}")
# 注意：**脚本本身不要打印凭据值**，连前几位也不要
PY
```

**判定标准**：

- 长值（30+ 字符）一旦命中就是真泄漏——随机巧合不可能让 35 字符的串匹配上。
- 短值（如 3 位数的用户名）在二进制文件里必然出现若干次，是巧合，不是泄漏。
  可以用 `data.count(值)` 看出现次数来区分。

**另外要人工过一遍的**：

- 文档正文里有没有写成散文的凭据（上面那次漏检就是这类）。
- 文件名不典型的凭据文件——`.env.before-*`、`*.env.20260909` 这类备份
  长得不像 `.env`，很容易漏。`.gitignore` 里对每种命名都要有对应规则。
- 配置参考文件里的 token：`frpc_*.toml` / `frps_*.toml` 常常带着真实 token。

### 10.10 从 GitHub 克隆后首次部署

`llama.cpp` 的源码和编译产物**都不进版本库**（源码树 1.4 G、CUDA 产物 974 M），
所以克隆下来的项目里**没有任何 llama-server 二进制**。这是正常的，
`deploy.sh` 会识别出来并在阶段 3.5 现场编译，不会直接报错退出。

完整流程：

```bash
git clone https://github.com/HyperDriveWave/HyperDriveWave-LLM.git
cd HyperDriveWave-LLM

bash Scripts/setup_mirrors.sh --yes   # 国内网络建议先配镜像源
bash Scripts/deploy.sh                # 走完全部流程
```

`deploy.sh` 会依次做完：探测 GPU 与网络 → 端口确认 → 推理方式 →
渲染配置 → **拉第三方源码** → **编译 llama.cpp** → 下模型 → 建镜像 → 启动 → 验收。

**编译这一段要等**：CUDA 单架构约 10 分钟，Vulkan 约 5 分钟，CPU 约 3 分钟。
工具链（cmake / g++ / CUDA toolkit）缺失时会用 apt 自动安装（需要 sudo 密码）。
CUDA toolkit 在 Ubuntu 的 **multiverse** 源里，不需要加 NVIDIA 官方源。

单独重编或用参数控制：

```bash
bash Scripts/build_llama.sh --check      # 只报状态
bash Scripts/build_llama.sh --rebuild    # 强制重编
bash Scripts/build_llama.sh --cpu        # 强制 CPU 版
```

**连不上 github 时走 gitee 镜像**（`vendor.lock` 里已配好兜底）：

```bash
bash Scripts/fetch_vendors.sh --with llama.cpp-upstream --prefer gitee
```

实测 9 秒完成（github 那边会挂起 45 秒才超时）。**但要注意兜底给的是另一份代码**：

镜像是定时同步的，**不会**有我们锁定的那个 commit，所以脚本会用镜像自己的版本
（写在 `vendor.lock` 的 `#<commit>` 里），并明确警告版本不同。这一点已针对
llama.cpp 专门核对过——镜像是 2026-08-28，锁定的是 2026-09-08，落后 11 天，
**MTP 投机解码的代码逐处一致**（`spec_type_draft_mtp` / `opts.download_mtp` /
`--spec-type` 取值集合相同）。但版本差异不会被自动测出来，**务必重跑验收**：

```bash
bash Scripts/deploy_verify.sh
```

要拿到与源机完全一致的版本，用离线包——`pack_hdw.sh` 打出来的包里带着源码和
编译好的产物。另外 `gitee.com/lure_ai/llama.cpp` 虽然存在但停在 2026-04-20
（落后 5 个月），太旧，不要用。

**在这台机器上已经跑着一份 HyperDriveWave 的情况下克隆部署**（比如为了
对比新旧版本），`deploy.sh` 会检测到 systemd 单元正指向另一个目录并要求
确认。想放两份互不干扰地跑，**光改项目目录不够**——单元装在
`~/.config/systemd/user/`，两边会互相顶掉。可行做法是给第二份单独一个
`XDG_CONFIG_HOME`：

```bash
XDG_CONFIG_HOME=$HOME/.config-hdw2 bash Scripts/deploy.sh
```

同时还得把宿主端口错开（见 §10.11），否则两边抢同一组端口。

### 10.11 端口配置

部署时端口可以改，默认值不变。换服务器时 3000/8080/5432 这类端口很容易撞上已有服务。

```bash
bash Scripts/deploy.sh --ports                        # 只看端口表与冲突
bash Scripts/deploy.sh --port qa=18080,webui=13000    # 直接指定
bash Scripts/deploy.sh                                # 交互式：列出现状，问 Y/n，选 n 逐项改
```

冲突会被自动识别并**建议一个空闲端口**（往上找 200 个），确认后自动改。

可改的端口：

| 服务 | 默认 | 绑定 |
| --- | --- | --- |
| WebUI HTTP / HTTPS | 3000 / 8443 | 局域网 |
| QA API | 8080 | **所有网卡** |
| 对外问答 API | 8095 | **所有网卡**（给别的项目调用，靠密钥保护） |
| 宿主 llama.cpp | 1919 | 仅本机 |
| 本机 RAG / MinerU | 8001 / 8002 | 仅本机 |
| Neo4j HTTP / Bolt | 7475 / 7688 | 仅本机 |
| PostgreSQL / Redis | 5432 / 6379 | 仅本机 |
| 入库 API | 8090 | 仅本机 |
| MCP | 8766 | 仅本机（**不可改**，见下） |

**容器之间的通信全部走容器内端口**（`hdw-rag:8001`、`hdw-qa-api:8080` 等），
与宿主映射无关——所以改宿主端口不会影响内部调用，只影响你从外面怎么连。

**两处会连带同步**（脚本自动做，不用手工）：

- 改 **llama 端口** → 同步 `.env` 的 `HDW_LOCAL_LLM_BASE_URL` / `HDW_LLM_BASE_URL`
  和 `HDW_Runtime/model-config/config.json` 的 `local.base_url`。
  最后一处**不同步等于没改**：qa-api 读的是 config.json，`.env` 只是 fallback。
  漏了它的表现极具迷惑性——`/health` 全绿、模型管理页正常，一提问就 connection refused。
- 改 **WebUI 端口或绑定地址** → 同步 `frpc_hdw_public.toml` 的 `localPort`/`localIP`，
  并提示需要重启 frpc。不同步的话隧道照常"建立成功"但转发到旧端口，
  只有 frpc 日志里有 connection refused，控制中心显示一切正常。

**MCP 的 8766 改不了**：容器内监听端口由 `MCP/Dockerfile` 的 `--port 8766` 决定，
命令行参数压过环境变量，只改宿主映射会让 nginx 的 `/api/mcp/` 转发和 qa-api 调用断链。
要真支持得同时改 Dockerfile + nginx.conf + compose 三处，收益不值这个复杂度。

### 10.12 无 GPU 部署（走在线 API）

没有 NVIDIA 显卡时，纯 CPU 跑 27B 模型约 **1 token/s**——问一个问题要等好几分钟，
实际不可用。`deploy.sh` 会检测到并引导切到在线 API。

```text
── 步骤：推理方式（未检测到 GPU）──

  没有可用的 NVIDIA 显卡。这影响两件事：
    · 本地推理：只能走 CPU，27B 模型约 1 token/s，实际不可用
    · 知识入库的 GPU 加速：不可用（入库会慢很多，不影响已有知识库的问答）

  建议改用在线 API：问答立刻可用。代价是每次提问都走外网，
  且提问内容和检索到的文档片段会发给模型提供方。

切换到在线 API？[Y/n]
  base_url [https://api.deepseek.com]:
  model [deepseek-flash]:
  api_key（必填，否则提问会返回 503）:
```

确认后脚本会写 `.env`（`HDW_ONLINE_LLM_*`、`HDW_LLM_MODE=online`、
`HDW_SKIP_LOCAL_LLM=true`），并且**还会改 `HDW_Security/auth/auth.csv`**
里每个用户的 `default_inference_mode`。

最后这一处不改的话前面全白做：每次问答的默认模式取自 auth.csv 里该用户的那一列，
**不是** `.env` 的 `HDW_LLM_MODE`（`main.py:2222 → _user_default_inference_mode`）。
本项目 98 个用户该列都是 `offline`，不翻的话他们提问仍走离线 → 没本地模型 → 直接失败。

**停用本地推理后的边界**：

- 不会启动 llama 服务
- 前端**仍能看到「离线」选项**，但选中会失败——这是刻意的：完全藏掉这个选项
  会让人以为系统坏了
- `deploy_verify.sh` 会跳过 llama 检查，改为验证在线 API 是否配好
  （它不该因为"本地推理按配置就没在跑"而报一堆失败）

**以后加了显卡**：重跑 `bash Scripts/deploy.sh` 即可切回本地推理
（把 `HDW_SKIP_LOCAL_LLM` 设回 `false`，并把 auth.csv 的默认模式改回 `offline` 或按需）。

## 11. 日常运维命令

查看状态：

```bash
bash Scripts/status.sh
```

查看全部日志：

```bash
bash Scripts/logs.sh
```

查看指定服务：

```bash
bash Scripts/logs.sh hdw-qa-api
bash Scripts/logs.sh hdw-ingest
bash Scripts/logs.sh hdw-mineru
bash Scripts/logs.sh hdw-rag
```

停止服务但保留数据：

```bash
bash Scripts/stop.sh
```

修改 QA API 或 WebUI 后只重建相关服务：

```bash
docker compose --env-file Configs/.env \
  -f Configs/docker-compose.yml \
  --profile base --profile web \
  up -d --build hdw-qa-api hdw-webui
```

修改 MinerU Compose 配置后：

```bash
docker compose --env-file Configs/.env \
  -f Configs/docker-compose.yml \
  --profile knowledge \
  up -d --force-recreate hdw-mineru
```

查看 GPU 维护协调器：

```bash
systemctl --user status hyperdrivewave-resource-coordinator.service --no-pager
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock http://localhost/status
```

手动触发维护模式只用于验收或故障排查。它会停止本地 llama，切换本机 RAG/MinerU
到 CUDA；完成后必须恢复：

```bash
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock \
  -X POST http://localhost/prepare
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock \
  -X POST http://localhost/restore
```

资源协调器同时承载外网隧道开关：

```bash
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock http://localhost/frp
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock -X POST http://localhost/frp-enable
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock -X POST http://localhost/frp-disable
```

临时把显存让出来（不想停整个项目时用这个）：

```bash
# 卸载：停掉 llama-server，释放全部显存
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock \
  -X POST http://localhost/unload-llm

# 加载回来
curl --unix-socket HDW_Runtime/maintenance/maintenance.sock \
  -X POST http://localhost/switch-llm
```

日常入口是 WebUI 模型管理页的「卸载本地模型 / 加载本地模型」按钮（管理员可见），
对应 `PATCH /api/local-model`，body `{"loaded": true|false}`——加载复用协调器的
`/switch-llm`，它的语义本来就是「确保配置里的本地模型在跑」，不另造重复端点。

**为什么卸载只能停进程**：llama.cpp 单模型模式**没有任何运行期卸载接口**——
`/models/unload` 只在多模型 router 模式下注册（本项目单模型启动，那条路由根本没挂上）。
而单元是 `Restart=always` + `RestartSec=3`，**kill 掉进程会在 3 秒后自己回来**，
只有 `systemctl --user stop` 这种显式停止才算数。

实测：卸载后显存从 22454 MiB 降到 678 MiB（释放约 21.8 GB），加载回来约 22236 MiB。
卸载期间本地问答不可用，界面会当场提示。

### 11.1 外网访问（公网隧道）

推荐入口：**控制中心 → 外网访问**。管理员可切换，普通用户只能看状态。停用后公网
入口立即失效，局域网不受影响。

命令行等价操作：

```bash
systemctl --user status  hyperdrivewave-frpc.service
systemctl --user enable  --now hyperdrivewave-frpc.service   # 启用
systemctl --user disable --now hyperdrivewave-frpc.service   # 停用
tail -f HDW_Runtime/frp/frpc_hdw_public.log
```

完整的架构图、部署方式、证书更换和回滚步骤见
`HDW_Frontend/FRP/README_FRP.md`。三点必须记住：

1. **单元用 `cp` 安装，不要用 `systemctl --user link`。** `link` 创建的单元，符号链接
   本身就是「启用」机制，`systemctl disable` 会把链接删掉、单元直接消失，导致再也
   启用不了。`hyperdrivewave-llama.service` 和 coordinator 目前仍用 `link`：它们从
   不需要 `disable`，所以没暴露这个问题，但若要给它们做开关必须先改成 `cp`。
2. **公网端口 8080/8443 是共享资源。** 中转 frps 的 `allowPorts` 硬限制为这两个口。
   在 217 上手动重启 SmartGasTurbine 的 FRP 会抢端口、顶掉本项目隧道（frpc 会重试恢复）。
3. **开机自启依赖用户会话。** 本机 `Linger=no`，systemd 用户服务只在用户登录会话存在
   时运行，与 llama/coordinator 行为一致。需要无登录也自启时：
   `sudo loginctl enable-linger xthd`（会影响所有用户服务）。

中转面板（查公网代理是否在线）：`http://<公网中转机IP>:7500`

### 11.2 改动生效方式（易踩的坑）

Compose 里有三类挂载，改完文件后的生效方式不同：

| 挂载方式 | 涉及文件 | 生效方式 |
| --- | --- | --- |
| 单文件 bind mount | `industrial-webui/index.html`、`nginx.conf` | **必须 `--force-recreate`** |
| 目录 bind mount | `HDW_Engines/LLM_API`（qa-api 的 `client.py` 等） | `docker restart` 即可 |
| 烧进镜像 | `HDW_Orchestrator/industrial-qa-api/app/` | `up -d --build` |

单文件 bind mount 锁的是 **inode**。编辑工具通常用 rename 替换文件，产生新 inode，
容器仍指向旧 inode —— 此时 `docker compose up -d` 会判定「无变更」而不重建：

```bash
docker compose --env-file Configs/.env -f Configs/docker-compose.yml \
  --profile base --profile knowledge --profile web \
  up -d --force-recreate --no-deps hdw-webui
```

## 12. 文档入库操作

**这一节的接口全部需要登录会话**，命令行调用必须带 `X-HDW-Session` 头，否则一律 `401 login required`：

```bash
SESSION=<登录会话>     # 浏览器登录后 localStorage 里的 hdw-auth-session 值
ADMIN=<管理员会话>     # 上传、入库、同步、删除、清空都要求 role=admin
```

推荐使用 WebUI：

1. 进入“控制中心”。
2. 打开“知识库入库”。
3. 上传文档或拖拽文档。
4. 等待待入库文件显示上传完成。
5. 需要重新整理全部文档时勾选“全量一致性重建”。
6. 点击“知识库入库”。
7. 等待 MinerU、切分、Neo4j 和 Zvec 全部完成。
8. 远端 RAG 由入库流程自动同步；只有文档上的「远端同步」显示「未配置」而远端其实已有数据时，才需要点「同步远端」单独刷新（见下）。

命令行上传：

```bash
curl -f -X POST http://127.0.0.1:8090/upload \
  -H "X-HDW-Session: $ADMIN" \
  -F "file=@/path/to/document.docx"
```

查看文档：

```bash
curl -fsS http://127.0.0.1:8090/documents -H "X-HDW-Session: $SESSION"
```

查看任务：

```bash
curl -fsS http://127.0.0.1:8090/jobs/<job-id> -H "X-HDW-Session: $SESSION"
```

普通入库请求：

```bash
curl -f -X POST http://127.0.0.1:8090/ingest \
  -H "X-HDW-Session: $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"document_ids":["<document-id>"],"full_rebuild":false}'
```

全量重建请求：

```bash
curl -f -X POST http://127.0.0.1:8090/ingest \
  -H "X-HDW-Session: $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"document_ids":[],"full_rebuild":true}'
```

只同步远端、不重建：

```bash
curl -f -X POST http://127.0.0.1:8090/sync -H "X-HDW-Session: $ADMIN"
```

把**现有的** `chunks.jsonl` 重新推到 `HDW_RAG_REMOTE_URLS` 里的每个节点，并刷新每个文档的 `remote_sync_status`。不解析、不嵌入、不碰 Neo4j、不用大模型、不占显存，所以比入库快得多（实测两个节点各推 10.8 MB、各重建 9021 个片段，整轮约 160 秒）。

另一个任务处于 queued/running 时返回 `409`；没有已入库文档时返回 `400`；一个远端都没配时把状态写成 `not_configured`，不粉饰成「完成」。

**为什么需要这个接口**：`remote_sync_status` 是**入库那一刻写死的快照**，读取时只补缺失字段（`setdefault`）、从不重算。所以远端如果是在某次入库**之后**才配置的，那批文档会永远显示「未配置」——哪怕数据其实早就推过去了。这条路径就是用来不重建而把标签刷成真实值的。

删除已入库文档必须使用文档删除接口，让系统重新构建剩余图谱和向量：

```bash
curl -f -X DELETE \
  http://127.0.0.1:8090/documents/<document-id> \
  -H "X-HDW-Session: $ADMIN"
```

删除“已上传但尚未进入任务”的文件才使用 staged 接口：

```bash
curl -f -X DELETE \
  http://127.0.0.1:8090/documents/<document-id>/staged \
  -H "X-HDW-Session: $ADMIN"
```

## 13. 清空知识库

清空知识库前必须确认没有 running 或 queued 任务：

```bash
python3 - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("HDW_Runtime/ingest/state.json").read_text())
for job in state.get("jobs", {}).values():
    if job.get("status") in {"queued", "running"}:
        print("ACTIVE", job.get("id"), job.get("status"), job.get("stage"))
PY
```

建议流程：

1. 停止 WebUI 发起新入库。
2. 确认所有任务为 `completed` 或 `failed`。
3. 执行 `bash Scripts/backup.sh`。
4. 备份或移走 `HDW_Runtime/knowledge_sources` 内的源文档。
5. 清理 `HDW_Runtime/mineru/parsed`。
6. 清空 `HDW_Runtime/rag/chunks.jsonl`，保留空文件或由流程重建。
7. 通过 `full_rebuild=true` 触发空知识库重建。
8. 检查 Neo4j 节点、Zvec 索引和 WebUI 文档列表。

不要删除整个 `HDW_Runtime/neo4j`、`postgres` 或 `redis`，除非明确要初始化所有基础设施。

## 14. 交给另一个 AI 时的接手流程

把项目交给新的 AI 或开发者时，要求按下面顺序执行：

### 第一步：确认根目录

```bash
cd <项目根>
pwd
find . -maxdepth 2 -type d | sort
```

不要先删除文件，不要先执行 `docker compose down -v`。

### 第二步：阅读入口文档

按顺序阅读：

```text
README.md
架构.md
Configs/docker-compose.yml
Configs/.env.example
Scripts/start.sh
Scripts/stop.sh
HDW_Orchestrator/industrial-qa-api/app/main.py
HDW_Orchestrator/industrial-ingest-api/app.py
HDW_DataFoundation/ETL_Pipelines/parse_documents.py
HDW_DataFoundation/ETL_Pipelines/ingest_documents.py
```

### 第三步：判断什么是真正上线的

检查完整 profile 下的 Compose 服务列表：

```bash
docker compose --env-file Configs/.env \
  -f Configs/docker-compose.yml \
  --profile base --profile gpu --profile knowledge --profile web \
  config --services
```

目录存在只表示源码或预留目录存在；服务是否上线以 Compose、容器状态和 HTTP 健康检查为准。

### 第四步：检查数据和任务

```bash
docker compose --env-file Configs/.env \
  -f Configs/docker-compose.yml ps

curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8090/documents
```

如果有入库任务，先查看：

```bash
cat HDW_Runtime/ingest/state.json
```

正在解析时不要移动源文档、解析结果、Chunk、Zvec 或 Neo4j 数据。

### 第五步：只做最小改动

遵守以下边界：

- 新功能先复用现有容器。
- 不重复引入第二个聊天数据库。
- 不把第三方源码直接改成业务代码。
- 不把模型权重打进镜像。
- 不让 WebUI 直接访问 Neo4j、SIS 或数据库。
- 不绕过 ingest API 直接修改索引。
- 不把账号密码写进 README、前端或提交记录。

### 第六步：完成最小验收

至少执行：

```bash
python3 -m py_compile \
  HDW_Orchestrator/industrial-qa-api/app/config.py \
  HDW_Orchestrator/industrial-qa-api/app/main.py \
  HDW_Orchestrator/industrial-ingest-api/app.py \
  HDW_DataFoundation/ETL_Pipelines/parse_documents.py

# compose 校验要带 profile，否则输出 services: {}，返回 0 但什么都没查
docker compose --env-file Configs/.env -f Configs/docker-compose.yml \
  --profile base --profile knowledge --profile web config

bash Scripts/deploy_verify.sh --quick   # 比 healthcheck.sh 严格得多，见 §10.6
```

**升级到 `deploy_verify.sh` 的原因**：`healthcheck.sh` 是日常巡检，
它查的是 `${HDW_LLM_PORT:-8000}`（旧 FreeToken 引擎的端口），
不是真实 llama 的 `1919`，所以**一直误报 llm unavailable**。
`deploy_verify.sh` 不信任 `/health`——项目里有三处会让它骗人（见 §10.6）。

若修改 WebUI，还要用 Node 做内嵌 JavaScript 语法检查，并手动验证：

1. 新建会话后 `HDW_Runtime/chatdata` 出现 JSON。
2. 刷新页面后历史会话仍可打开。
3. 另一台局域网电脑能看到同一历史记录。
4. 右键置顶会移动到置顶区域。
5. 删除会话后对应 JSON 消失。
6. 回答依据仍能展开。

## 15. 后续接入路线

### P0：当前闭环

```text
Qwen + RAG + reranker + Zvec + Neo4j + MinerU + WebUI + 文件会话
```

### P1：真实增量知识库

1. 使用源文件 SHA-256 判断新增、修改和删除。
2. 只重新解析变化文档。
3. 删除旧文档对应的 Chunk、图谱节点和向量。
4. 让普通入库真正成为增量流程。
5. 给任务增加取消、重试和断点恢复。

### P2：MCP 真实问答链路

1. QA API 增加问题分类。
2. 只读问题允许调用 SIS、报警和 RTSP 工具。
3. 工具结果标注时间、测点、单位、质量码和来源。
4. 工具失败时明确区分“没有数据”和“服务不可用”。
5. 文档证据、图谱证据和实时工具结果分开展示。
6. 所有工具调用写审计日志。

### P3：Dify 和 n8n

- Dify：接入 `hdw-qa-api` 的 OpenAI 兼容接口，用于 Prompt 实验和业务应用原型。
- n8n：接收文档上传事件、定时重建、低置信度告警、审批和通知。
- Dify 和 n8n 不应绕开 QA API 直接操作 Neo4j 或 Zvec。

### P4：生产治理

1. Keycloak 提供 OIDC、SSO 和 RBAC。
2. Langfuse 记录 LLM trace 和用户反馈。
3. PostgreSQL 保存文档元数据、权限、审计和任务索引。
4. RAGAS 和 Golden Dataset 形成版本评测。
5. 单机 JSON 会话迁移到 PostgreSQL 或对象存储。
6. 根据吞吐量再拆分多副本和队列，不提前增加容器。

## 16. 重要配置索引

| 变量 | 作用 |
| --- | --- |
| `HDW_PROJECT_ROOT` | 主机项目根目录 |
| `HDW_RUNTIME_ROOT` | 主机持久化目录 |
| `HDW_COMPOSE_PROFILES` | `Scripts/start.sh` 启动的 Compose profile，默认 `base knowledge web` |
| `HDW_WEBUI_BIND` | WebUI 绑定网卡 |
| `HDW_WEBUI_PORT` | WebUI 明文端口 |
| `HDW_WEBUI_TLS_PORT` | WebUI TLS 端口（自签证书，由 hdw-webui 的 nginx 终结） |
| `HDW_FRP_PUBLIC_HOST` | 公网中转主机地址，仅用于「外网访问」页展示入口地址 |
| `HDW_FRP_SYSTEMD_UNIT` | frpc 的 systemd 用户单元名，默认 `hyperdrivewave-frpc.service` |
| `HDW_LLM_PRESENCE_PENALTY` | 送入上游 LLM 的 `presence_penalty`，默认 `0.3`；抑制模型复读检索原文（复读会撞 `max_tokens` 上限并触发模板解析失败）。调大更不易复读、更易跑题 |
| `HDW_RETRIEVAL_ROUTER` | 检索路由开关，默认 `true`；设为 `false` 退回「一律检索、不读测点」。见 §7.4 |
| `HDW_LIVE_POINTS_MAX` | 单个问题最多并行查几个实时测点，默认 `3` |
| `HDW_LIVE_POINT_CANDIDATES` | 关键词搜不到精确匹配时，交给模型挑选的候选测点数，默认 `5` |
| `HDW_LIVE_TIME_OFFSET_HOURS` | 实时测点采集时间的展示时区偏移，默认 `8`（UTC+8）；只影响展示，输出不带时区后缀。见 §7.5 |
| `HDW_SIS_BASE_URL` / `HDW_SIS_LOGIN_URL` / `HDW_SIS_USERNAME` / `HDW_SIS_PASSWORD` | SIS 实时测点认证，供 MCP 的 `point_query_*` 取实时值。见 §7.5。**改后需 `--force-recreate hdw-mcp`** |
| `HDW_SIS_LANGUAGE` | SIS 接口语言，默认 `zh-Hans` |
| `HDW_LLM_GPU` | FreeToken 旧方案使用的 GPU |
| `HDW_LLM_MEMORY_RATIO` | LLM 显存比例 |
| `HDW_LLM_MODE` | 默认推理模式：`online` 或 `offline` |
| `HDW_LOCAL_LLM_BASE_URL` | 本地 FreeToken OpenAI 兼容地址 |
| `HDW_LOCAL_LLM_MODEL` | 本地模型名称 |
| `HDW_ONLINE_LLM_BASE_URL` | 在线模型供应商地址 |
| `HDW_ONLINE_LLM_MODEL` | 在线模型名称 |
| `HDW_ONLINE_LLM_API_KEY` | 在线模型密钥，只放在被忽略的 `Configs/.env` |
| `HDW_ONLINE_GRAPH_TOP_K` | 在线问答送入 LLM 的图谱上下文上限，默认 40 |
| `HDW_LOCAL_GRAPH_TOP_K` | 离线问答送入 LLM 的图谱上下文上限，默认 10；先由 reranker 排序，再取前 10 条 |
| `HDW_CONTEXT_WINDOW_TOKENS` | 在线上下文估算上限，当前为 **1,000,000**（262,144 是下面那个本地键的值，两者别混） |
| `HDW_LOCAL_CONTEXT_WINDOW_TOKENS` | 离线上下文估算上限 |
| `HDW_CONTEXT_COMPRESSION_THRESHOLD` | 触发上下文压缩的比例，当前为 75%。**这是兜底机制**：选留之后历史仍然超长才由它出手；按 0.75 × 262,144 = 196,608 token 换算，实际从未触发过 |
| `HDW_CONTEXT_COMPRESSION_TARGET` | 压缩后目标比例 |
| `HDW_CONTEXT_COMPRESSION_MAX_TOKENS` | 单次摘要的输出上限，默认 2000 |
| `HDW_HISTORY_SELECT` | 历史相关性选留总开关，默认 1。每次提问前判断哪些历史轮次与当前问题相关，无关的不进 prompt；**置 0 可一键退回「全量历史」** |
| `HDW_HISTORY_SELECT_MIN_TURNS` | 历史少于这么多轮就不做选留（刚开始对话时没意义，白花一次调用），默认 3 |
| `HDW_HISTORY_SELECT_MAX_TURNS` | 最多把最近多少轮交给选择器；更早的一律丢弃（近因策略，也让选择器输入有上界），默认 20 |
| `HDW_SSE_HEARTBEAT` | SSE 心跳间隔秒数，默认 5；防止长连接被代理掐断（仅代码默认值，compose 未透传） |
| `HDW_RAG_DEVICE` | 常态 RAG 使用 CPU 或 GPU；WebUI 维护重建时由覆盖文件临时设为 `cuda` |
| `HDW_LLM_BASE_URL` | QA API 使用的 LLM 地址 |
| `HDW_RAG_BASE_URL` | QA API 使用的 RAG 地址 |
| `HDW_RAG_REMOTE_URLS` | 远端 RAG 地址列表，逗号分隔，QA 按轮询调用，当前为 `<远端RAG主机IP>:8001,8003` |
| `HDW_RAG_CONNECT_TIMEOUT` | RAG 建立连接超时；远端不可达时用于快速进入下一个节点或本机 fallback |
| `HDW_RAG_HEALTH_TIMEOUT` | QA `/health` 检查单个 RAG 节点的超时 |
| `HDW_MCP_TIMEOUT` | MCP 工具的读取超时，默认 60。**必须大于工具内部最慢的那次抓取**——日志工具要等 LIEMS 门户，它自己的 read timeout 就是 30 秒；超时设小了会抢先切断，工具已备好的失败原因一个字都传不回来 |
| `HDW_RAG_ADMIN_TOKEN` | 远端 RAG `/admin/sync` 的鉴权令牌。入库和「同步远端」都靠它，缺了会直接报 `HDW_RAG_ADMIN_TOKEN is not configured` |
| `HDW_REMOTE_RAG_SYNC_TIMEOUT` | 同步远端时**单个 socket 操作**的超时秒数，当前 600（不是总时长，正常上传不受影响）。**置 0 = 不设超时**——远端不回包时作业和文档会永久卡在「同步中」，比原来的「未配置」更糟 |
| `HDW_REMOTE_RAG_SSH_TARGET` | **仅供 `Scripts/sync_remote_rag.sh` 手工运维**用的 SSH 目标，不含密码。应用内的「同步远端」走 HTTP，不需要 SSH |
| `HDW_REMOTE_RAG_ROOT` | 同上，远端 RAG 的独立部署目录 |
| `HDW_MINERU_TIMEOUT` | 问答图片转写走 MinerU 时的超时秒数，默认 180（问答有人在等，和批处理的「不设超时」刻意不同） |
| `HDW_VISION_MINERU_MAX_QUEUE` | MinerU 转写的排队上限，默认 1；队列非空就直接放弃该候选，失败快 |
| `HDW_MINERU_BASE_URL` | ingest 使用的 MinerU 地址 |
| `HDW_MAINTENANCE_SOCKET` | llama/RAG/MinerU 资源协调器 Unix socket |
| `HDW_MAINTENANCE_TIMEOUT` | 维护切换等待上限；`0` 表示不设置总等待上限 |
| `HDW_CHATDATA_ROOT` | QA API 容器内的会话目录 |
| `HDW_ENABLE_AUTH` | 是否启用内部 Bearer 鉴权 |
| `HDW_INTERNAL_API_KEY` | 内部 API key（只被 `_check_auth` 用，且受 `HDW_ENABLE_AUTH` 开关控制） |
| `HDW_API_PORT` | 对外问答 API 的宿主端口（容器内固定 8095） |
| `HDW_API_KEY` | 对外问答 API 的调用方密钥，**发给调用方项目** |
| `HDW_API_INTERNAL_KEY` | `hdw-api` ↔ `qa-api` 的服务间密钥，不外发；两侧必须一致 |
| `HDW_API_QA_TIMEOUT` | `hdw-api` 等上游问答的超时秒数，默认 600 |
| `NEO4J_AUTH` | Neo4j 认证 |
| `HDW_SIS_BASE_URL` | SIS 服务地址 |
| `HDW_SIS_USERNAME` | SIS 用户名，必须放本地环境变量 |
| `HDW_SIS_PASSWORD` | SIS 密码，必须放本地环境变量 |
| `MINERU_API_MAX_CONCURRENT_REQUESTS` | MinerU 请求并发上限，当前固定为 1 |

## 17. 最后检查清单

### 17.1 部署后：先跑自动验收

```bash
bash Scripts/deploy_verify.sh          # 完整，含端到端问答
bash Scripts/deploy_verify.sh --deep   # 额外做重启演练
```

**必须全绿。** 它覆盖了下面这些自动化能测的项，逐条说明它为什么这么测：

- [ ] **推理真的能跑**（不是只看 `/health`）。CUDA kernel 不匹配时 `/health`
      照样通过，第一次 kernel launch 才崩；验收会真发一次生成请求，
      读 `usage.completion_tokens` 而不是 `content`——Qwen3.8 是思考模型，
      token 可能全被 `reasoning_content` 吃掉，只看 `content` 会误判为失败。
- [ ] **journal 里没有** `no kernel image is available` / `CUDA error`
      （kernel 不匹配最直接的证据），且服务 `NRestarts` 为 0。
- [ ] **MTP 确实启用了**：日志里有 `MTP: enabled (--spec-type draft-mtp)`。
      没启用说明后端是旧版 llama.cpp，或 GGUF 不含 nextn 张量。
- [ ] **RAG 真的能嵌入**：调一次 `/embed` 断言维度为 1024。
      RAG 的 `/health` 返回的是**写死的静态字典**，只查路径存在性、
      不加载模型（模型是首次 `/embed` 才懒加载），`status` 恒为 `ok`。
- [ ] **重排能用**：调一次 `/rerank`。注意 `documents` 要传对象数组。
- [ ] **QA API 的三个子项**都正常。它的 `/health` 里 `"status":"ok"`
      也是硬编码字面量，只看顶层等于没看。
- [ ] **模型一致性**：`/props` 的实际加载模型 == `config.json:local.model`
      == `.env:HDW_LOCAL_LLM_MODEL`。三处分叉会导致「界面显示的」和
      「实际跑的」不是同一个模型。
- [ ] **MinerU 镜像里确实烧进了模型**：`docker exec hdw-mineru ls /root/.cache/modelscope`。
      它是在**镜像构建期**下载的，不在宿主挂载里，漏了要到真解析文档时才发现。
- [ ] **远端 RAG 的 URL 条数与实际卡数一致**。单卡远端只监听 8001，
      主站若还列着 8003，每次问答都有一半请求打到不存在的端口、
      每个吃一次 `HDW_RAG_CONNECT_TIMEOUT`（默认 3s）。
- [ ] **模型权重三件套字节数正确**（`fetch_models.sh` 的清单是唯一事实源）。
- [ ] **第三方依赖就位**（`fetch_vendors.sh --check-only`）。缺 MinerU 会导致
      `hdw-mineru` 镜像构建失败，缺 aora-bot 会导致 WebUI 情绪球加载不出来。
- [ ] **WebUI 可访问**，且返回的确实是页面（不是 nginx 指错目录后的 200）。
- [ ] **端口没有冲突**：`bash Scripts/deploy.sh --ports`。改过端口的话，
      确认 `frpc_hdw_public.toml` 的 `localPort` 跟着变了（改了 WebUI 端口却没同步，
      外网访问会静默断掉）。
- [ ] **llama 端口改动后配置自洽**：`HDW_LOCAL_LLM_BASE_URL` 里的端口
      与 `HDW_LLAMA_PORT` 一致，且 `HDW_Runtime/model-config/config.json`
      的 `local.base_url` 也一致。三处分叉的表现是
      `/health` 全绿、模型页正常，一提问就 connection refused。
      `start.sh` 会在检测到不一致时直接报错拦下。
- [ ] **systemd 单元指向的是当前项目目录**（不是上一次部署留下的旧路径）：
      ```bash
      grep -h '^WorkingDirectory=' ~/.config/systemd/user/hyperdrivewave-*.service
      ```
      单元是**机器级**的，不随项目目录走。同一台机器上从另一个目录跑过
      `deploy.sh` 的话，这里会指向那个目录——**服务当前仍在跑旧的（无感），
      重启后才发作**。`deploy.sh` 装单元前会拦下这种情况，但从别的机器
      拷过来的单元不会重新生成，所以值得手工看一眼。

### 17.2 部署后：自动化测不到的，手工确认

- [ ] `Configs/.env` 已按实际环境填写，**没有沿用默认密码**
      （`POSTGRES_PASSWORD` / `REDIS_PASSWORD` / `NEO4J_AUTH` 等还是 `change_me` 的话要改）。
- [ ] **linger 已开启**：`loginctl show-user $USER -p Linger --value` 为 `yes`。
      否则重启机器后 llama 和资源协调器不会自启，要先登录一次——
      这是「部署完看着好、重启就没了」的典型原因。`deploy.sh` 会尝试用 sudo 开。
- [ ] **无 NVIDIA 卡时**接受这些代价：本地推理走 CPU（很慢）、
      知识入库的 GPU 加速路径（`/prepare`）不可用。
- [ ] **非 Blackwell 卡**（cap 不是 12.0）确认后端选择合理：多架构构建
      `build-cuda-multi/` 覆盖 sm_80/89/90/120；不在其中会回退 Vulkan，
      MTP 失效、吞吐降 2-3 倍。
- [ ] **上传一个小文档并完成一次入库**。
- [ ] **问一个能在文档中找到答案的问题**，并展开引用。
- [ ] **问一个需要实时测点的问题**（如「一号机凝汽器水位多少，然后结合知识库回答」），
      确认返回实时值、采集时间显示为北京时间、且注明 KKS。
- [ ] **新建会话后刷新页面**，历史消息可恢复；置顶和删除能在文件目录中体现。
- [ ] **连续问 10 个问题**（含 1 个与知识库无关的），没有 503；
      `journalctl --user -u hyperdrivewave-llama.service | grep -c " 500"` 为 0。
- [ ] **没有执行过 `docker compose down -v`**（`-v` 会删数据卷）。
- [ ] 用外网时：`HDW_ENABLE_AUTH` 已按预期设置；停用外网后公网 `8080`/`8443`
      确实不可达，局域网端口仍正常。

### 17.3 提交到版本库前

- [ ] **没有 gitlink**（嵌套仓库被记成 mode `160000`，clone 下来是空目录、内容全丢，
      而且看起来像正常的子模块引用，极难排查）：
      `git ls-files -s | awk '$1=="160000"'` 应无输出。
- [ ] **没有敏感文件**：
      `git ls-files | grep -E '\.env$|auth\.csv|\.key$|frpc_hdw_public|mapping\.csv|\.env\.before'`
      应无输出。
- [ ] **真实凭据反查**：从 `Configs/.env` 取出凭据**值**，扫每一个待提交文件。
      不要只靠 `password=xxx` 这类正则——实测漏过一个写在散文里的明文面板密码
      （`admin` / `口令` 这种形式），而 35 字符的 API key 一旦命中就是真泄漏，
      不会像短字符串那样产生巧合误报。步骤见 §10.9。
- [ ] **没有超过 50 MB 的文件**（GitHub 警告线，超 100 MB 直接拒收）：
      `git ls-files -z | xargs -0 du -b | awk '$1>50000000'`。
- [ ] **非 ASCII 文件名在远端没被转义破坏**。`git ls-tree` 默认会给中文路径
      加引号并转义成八进制；拿它的输出当路径用会造出名为 `"icons` 的目录。
      用 `git ls-tree -r -z`（NUL 分隔、不转义）。
      **验证时不要拿同一套解析去比两边**——两边都错的话反而显示"一致"。
- [ ] **模型/构建产物/第三方源码没有进库**，见 `.gitignore` 三、六两节。
- [ ] **模拟一次真实克隆**（漏过这类问题，代价很大）：

      ```bash
      rm -rf /tmp/clonetest && mkdir -p /tmp/clonetest
      git archive HEAD | tar -x -C /tmp/clonetest
      ls /tmp/clonetest/Configs/.env.example        # 必须存在
      bash /tmp/clonetest/Scripts/deploy.sh --dry-run
      ```

      **为什么必须做**：`.gitignore` 的排除规则过宽会静默吞掉本该进库的文件。
      实际发生过——`Configs/.env.*` 这条规则把 `.env.example` 一起排除了，
      而它是新机器唯一的配置模板，结果别人克隆后第一步就死：
      「既没有 Configs/.env 也没有 Configs/.env.example」。
      本地一切正常，因为本地**早就有 .env 了**，根本走不到那个分支。

      **教训**：排除规则要写到具体命名形式（`Configs/.env.before*`），
      不要用 `Configs/.env.*` 这种会把模板一起吞掉的通配。
      同理，凭据识别也别用裸子串——`KEY|TOKEN` 会误伤
      `KEYCLOAK_URL` 和 `HDW_LLM_MAX_TOKENS`，要带词边界（`_KEY$`）。

      **`--dry-run` 不能省。** 在 `/tmp` 这种临时目录里跑部署脚本时，
      它会去写**真实的** `~/.config/systemd/user/`——单元目录是机器级的，
      跟项目在哪儿无关。实际踩过：临时目录里跑了一次，把两个在跑的服务的
      单元改写成了指向那个临时目录，当前进程还在跑（毫无异常），
      但只要重启就再也起不来。
      现在 `--dry-run` 在任何写操作之前就退出，所以上面这条命令是安全的；
      **但去掉 `--dry-run` 就不安全了**。要在临时目录里试完整流程，
      至少把单元目录也隔离掉：

      ```bash
      XDG_CONFIG_HOME=/tmp/clonetest/.config bash /tmp/clonetest/Scripts/deploy.sh
      ```

      端口也仍然是真的——临时副本会和本机在跑的项目抢同一组宿主端口，
      所以本机已有一套在跑时，完整流程请换机做。
- [ ] **`Configs/.env.example` 覆盖了 `.env` 的全部键**：
      `bash Scripts/gen_env_example.sh --check`。模板过时会让新机器缺配置，
      且症状分散、很难定位到「模板少了几个键」这个根因。
      `.env` 有改动就重跑 `bash Scripts/gen_env_example.sh` 重新生成。

项目的最短可靠路径是：

```text
先确认任务和数据状态
  -> 读取现有接口
  -> 复用现有容器
  -> 做最小修改
  -> 通过健康检查和端到端测试
  -> 再进入下一阶段
```


## 18. 许可证

本项目**自有的代码**采用 **Apache License 2.0**，全文见仓库根目录的
[LICENSE](LICENSE)。附录里的版权所有者已按 Apache 的要求填好：

```text
Copyright 2026 HyperDriveWave

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

**根目录的 LICENSE 只覆盖本项目自有的代码。** 下面这些不是我们的，各自遵循
上游许可证——它们与本许可证兼容，但署名和分发义务要按各自的要求走：

| 路径 | 是什么 | 许可证 |
| --- | --- | --- |
| `HDW_Frontend/FRP/` | frp 客户端及其二进制 | 自带 `LICENSE`（同为 Apache 2.0） |
| `vendor/overlays/` | 对第三方源码的覆盖文件，衍生于上游 | 随上游（如 MCP python-sdk 为 MIT） |
| `vendor/vendor.lock` | 只是「配方」（路径 / commit / URL） | 不适用——它列出的项目由 `fetch_vendors.sh` 单独克隆，**不在本仓库内** |
| `HDW_Animation/aora-bot/` | WebUI 首页那个情绪球 | **不在本仓库内**；商业部署前需重新核对（见 §2 第 5 条） |

第三方源码、模型权重、构建产物都不进版本库（见 §10.8），所以仓库里属于别人的
内容只有上表前两处。若要对外分发，这两处的上游许可证副本需要一并保留。

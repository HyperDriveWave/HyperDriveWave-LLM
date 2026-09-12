# HyperDriveWave 外网访问（FRP）

公网隧道，把本机 WebUI 暴露到 `<公网中转机IP>`。**默认由控制中心的「外网访问」页开关**，非管理员只能看不能改。

相关文档：项目根目录 `README.md`（§4.9 前端结构、§5 端口、§11.1 运维命令、§16 配置索引）
和 `架构.md`（§4.14.1 隧道架构）。本文件只讲 FRP 自身的部署与运维。

## 架构

```
浏览器 ─┬─ http://<公网中转机IP>:8080 ─┐
        └─ https://<公网中转机IP>:8443 ┤
                                     ↓  frps (公网中转 <公网中转机IP>:7000)
                                     ↓  frpc 隧道
                         hdw-webui nginx (8080 明文 / 8443 TLS)
                                     ├─ /            → 静态页
                                     └─ /api/...     → hdw-qa-api / ingest / mcp
```

控制链路：

```
控制中心界面 → qa-api PATCH /frp → 维护socket → resource_coordinator.py → systemctl --user {start,stop} hyperdrivewave-frpc.service
```

中转服务器 `allowPorts` 硬限制为 **8080 和 8443**，不能用其他端口。

## 目录

| 路径 | 说明 |
|---|---|
| `bin/frpc` `bin/frps` | frp 0.62.1 linux amd64 二进制 |
| `conf/frpc_hdw_public.toml` | **本项目实际使用的隧道配置** |
| `conf/*.toml`、`conf/legacy/` | 从 SmartGasTurbine 复制的全套配置，作参考 |
| `conf/*.reference` | SmartGasTurbine 原公网配置（已停用，留档） |
| `control/` | SmartGasTurbine 的启停脚本，**本项目未使用**（systemd 直接跑 frpc） |
| `systemd/hyperdrivewave-frpc.service` | systemd 用户单元模板 |
| `certs/` | 自签 TLS 证书（10 年） |
| `frp_service.py` | SmartGasTurbine 的 WebUI 集成，**未接线**，留作参考 |
| `dockerfiles/` | frp 镜像构建文件，备用 |

## 日常操作

**推荐：控制中心 → 外网访问**（管理员可切换，带二次确认）

命令行等价操作：

```bash
# 查看状态
systemctl --user status hyperdrivewave-frpc.service

# 开启（等价于界面"启用"）
systemctl --user enable --now hyperdrivewave-frpc.service

# 关闭（等价于界面"停用"）
systemctl --user disable --now hyperdrivewave-frpc.service

# 看隧道日志
tail -f /home/xthd/桌面/HyperDriveWave/HDW_Runtime/frp/frpc_hdw_public.log
```

中转面板（查公网代理是否在线）：`http://<公网中转机IP>:7500` · 账号密码在 frps 自己的配置里，**不要写进任何文档或仓库**

## 单元安装方式（重要）

单元用 **`cp`** 安装，**不是 `systemctl --user link`**：

```bash
cp HDW_Frontend/FRP/systemd/hyperdrivewave-frpc.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hyperdrivewave-frpc.service
```

原因：`link` 创建的单元，其符号链接本身就是"启用"机制，`systemctl disable` 会把链接删掉、单元直接消失，导致再也启用不了。`cp` 安装后 `enable`/`disable` 是标准语义。

> 项目里 `hyperdrivewave-llama.service` 和 `hyperdrivewave-resource-coordinator.service` 仍用 `link`。它们从不需要 `disable`，所以没暴露这个问题。若将来也要给它们做开关，同样要先改成 `cp`。

配置改动后需重新 `cp` + `daemon-reload`。

## 已做的改动

| 文件 | 改动 |
|---|---|
| `HDW_Frontend/FRP/` | 新建（二进制、配置、证书、文档） |
| `HDW_Frontend/industrial-webui/nginx.conf` | server 块加 `listen 8443 ssl` + 证书路径（location 规则与 8080 共用，无重复） |
| `HDW_Frontend/industrial-webui/index.html` | 控制中心加「外网访问」菜单、页面、JS（复用模型管理页样式） |
| `Configs/docker-compose.yml` | `hdw-webui` 加 8443 发布 + 证书挂载；`hdw-qa-api` 加 `HDW_FRP_PUBLIC_HOST` |
| `Configs/.env` | 加 `HDW_WEBUI_TLS_PORT`、`HDW_FRP_PUBLIC_HOST`、`HDW_FRP_SYSTEMD_UNIT` |
| `Scripts/resource_coordinator.py` | 加 `/frp`、`/frp-enable`、`/frp-disable` 路由；`_unit_active()` 改为带 unit 参数 |
| `HDW_Orchestrator/industrial-qa-api/app/main.py` | 加 `GET /frp`、`PATCH /frp`（admin）、`frp_management` 权限位 |
| `HDW_Orchestrator/industrial-qa-api/app/config.py` | 加 `frp_public_host` |

qa-api 的代码是烧进镜像的，改完必须 `docker compose up -d --build hdw-qa-api`。

## 回滚

```bash
# 关闭隧道
systemctl --user disable --now hyperdrivewave-frpc.service

# 卸载单元
rm ~/.config/systemd/user/hyperdrivewave-frpc.service
systemctl --user daemon-reload

# 撤销容器改动
cd /home/xthd/桌面/HyperDriveWave
# 从备份恢复 Configs/、Scripts/、HDW_Orchestrator/、HDW_Frontend/
rsync -a /home/xthd/桌面/backuphdw/<路径>/ <路径>/
docker compose --env-file Configs/.env -f Configs/docker-compose.yml --profile base --profile knowledge --profile web up -d --build
```

完整项目备份在 `/home/xthd/桌面/backuphdw/`，见其 `_BACKUP_MANIFEST.md`。

## 注意事项

1. **开机自启依赖用户会话**：本机 `Linger=no`，systemd 用户服务只在用户登录会话存在时运行。这与 `hyperdrivewave-llama.service`、`hyperdrivewave-resource-coordinator.service` 行为一致。若需无登录也自启：`sudo loginctl enable-linger xthd`（会影响所有用户服务）。

2. **自签证书**：浏览器会提示不安全，属正常。SAN 含 `<公网中转机IP>` 和 `<本机局域网IP>`。换正式证书：替换 `certs/server.crt` 和 `certs/server.key`，然后 `docker restart hyperdrivewave-hdw-webui-1`。

3. **端口是共享资源**：8080/8443 原本由 SmartGasTurbine 占用，2026-09-11 已停用并移交。若有人在 217 上手动重启 SmartGasTurbine 的 FRP，会抢端口、顶掉本项目隧道（frpc 会自动重试恢复）。

4. **当前未开启鉴权**：`Configs/.env` 里 `HDW_ENABLE_AUTH=false`，公网任何人可访问 `/api/qa/query` 并消耗 DeepSeek 额度。隧道关闭时无此风险。开启方式：把 `HDW_ENABLE_AUTH` 改为 `true` 并 `docker compose up -d hdw-qa-api`。

5. **本地目标地址写死在 TOML 里**：`conf/frpc_hdw_public.toml` 的 `localIP`/`localPort` 对应 `Configs/.env` 的 `HDW_WEBUI_BIND`/`HDW_WEBUI_PORT`/`HDW_WEBUI_TLS_PORT`。改端口要同时改两处。

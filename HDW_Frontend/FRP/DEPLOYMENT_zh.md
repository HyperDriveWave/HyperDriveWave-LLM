# frp 部署说明（中文）

本文档用于说明 `SmartGasTurbine/frp` 目录下这套 frp 配置的部署、启动、验证与扩展方式。  
当前文档已经统一为**相对路径启动**，避免因为切换盘符、移动项目目录或混用其他 `frp` 目录导致配置读错。

## 1. 当前部署场景

- 本机客户端：Windows
- 服务端：Linux x86_64
- 公网入口：通过云平台 NAT 转发
- 本机实际业务服务：`SmartGasTurbine/bin/app.py`
- 本机 WebUI 实际监听地址：`http://127.0.0.1:6688/`

注意：

- 这里的本机业务端口是 **6688**
- 不再使用旧文档中出现过的 `8765` 作为本地监听端口
- `8765` 在当前命名中仅表示“对外访问方案的文件命名历史”，不表示本地实际服务端口

## 2. 目录约定

以下命令都以 `SmartGasTurbine/frp` 作为工作目录执行。

目录结构中的关键文件如下：

- `conf/frpc_8765_tcp.toml`：Windows 客户端配置
- `conf/frps_8765.toml`：Linux 服务端最小配置示例
- `bin/frp_0.62.1_windows_amd64/frp_0.62.1_windows_amd64/frpc.exe`：Windows 客户端程序

因此，**推荐启动方式**是：

1. 先切换到 `SmartGasTurbine/frp`
2. 再使用 `./conf/...`、`./bin/...` 这种相对路径执行

不要混用其他目录下的 `frp` 副本，例如：

- `E:\github_project\frp`
- 其他历史测试目录

否则很容易出现：

- 启动的是 A 目录下的 `frpc.exe`
- 读取的是 B 目录下的配置
- 实际排查时又查看了 C 目录下的 `toml`

最终导致“明明配置是 6688，但日志里还是在访问 8765”这类问题。

## 3. 当前端口规划

当前使用的端口如下：

- 公网地址：`<第三方FRP服务器IP>`
- `47000`：公网 NAT 控制端口，转发到 Linux 内部 `7000`
- `7000`：`frps` 服务端控制端口
- `18765`：`frps` 为当前 TCP 代理打开的服务端内部端口
- `48765`：公网访问端口，NAT 转发到 Linux 内部 `18765`
- `6688`：Windows 本机 SmartGasTurbine WebUI 实际监听端口

访问关系如下：

```text
公网浏览器 -> <第三方FRP服务器IP>:48765
           -> NAT 转发到 Linux:18765
           -> frps
           -> frpc
           -> Windows 127.0.0.1:6688
```

## 4. 当前客户端配置

当前项目内使用的客户端配置文件为：

- [conf/frpc_8765_tcp.toml](./conf/frpc_8765_tcp.toml)

当前内容应为：

```toml
serverAddr = "<第三方FRP服务器IP>"
serverPort = 47000

[[proxies]]
name = "local-8765-tcp"
type = "tcp"
localIP = "127.0.0.1"
localPort = 6688
remotePort = 18765
```

重点说明：

- `localPort = 6688` 才是当前正确配置
- 如果这里写成 `8765`，而本机服务实际监听 `6688`，就会出现：

```text
connect to local service [127.0.0.1:8765] error
```

## 5. 当前服务端配置

Linux 服务端最小配置可以是：

```toml
bindPort = 7000
```

对应示例文件：

- [conf/frps_8765.toml](./conf/frps_8765.toml)

## 6. 首次部署步骤

### 6.1 Windows 本机准备 `frpc`

将 Windows 版 frp 解压到当前项目目录下：

```text
./bin/frp_0.62.1_windows_amd64/frp_0.62.1_windows_amd64/
```

其中关键程序为：

```text
./bin/frp_0.62.1_windows_amd64/frp_0.62.1_windows_amd64/frpc.exe
```

### 6.2 Linux 服务端准备 `frps`

Linux 服务器准备对应版本的 `frps`，示例目录如下：

```text
./frp_0.62.1_linux_amd64/
```

然后在 Linux 上执行：

```bash
cd ./frp_0.62.1_linux_amd64
chmod +x ./frps ./frpc
printf 'bindPort = 7000\n' > ./frps.toml
./frps -c ./frps.toml
```

如果服务端已在运行，再次启动会提示端口冲突，例如：

```text
bind: address already in use
```

这时说明旧的 `frps` 已经占用 `7000`，需要先确认旧进程是否仍要继续使用。

### 6.3 配置 NAT 转发

NAT 需要至少配置两条规则。

规则 1：控制通道

- 协议：`TCP`
- 外部端口：`47000`
- 内部端口：`7000`

规则 2：业务访问通道

- 协议：`TCP`
- 外部端口：`48765`
- 内部端口：`18765`

## 7. Windows 本机启动方式

### 7.1 推荐启动命令

先进入 `SmartGasTurbine/frp` 目录，再用相对路径启动：

```powershell
Set-Location E:\my_project\SmartGasTurbine\frp
.\bin\frp_0.62.1_windows_amd64\frp_0.62.1_windows_amd64\frpc.exe -c .\conf\frpc_8765_tcp.toml
```

如果你已经在 `SmartGasTurbine/frp` 目录下，也可以直接执行：

```powershell
.\bin\frp_0.62.1_windows_amd64\frp_0.62.1_windows_amd64\frpc.exe -c .\conf\frpc_8765_tcp.toml
```

### 7.2 不推荐的方式

不推荐这样混用其他目录中的副本：

```powershell
cd E:\github_project\frp
.\bin\frp_0.62.1_windows_amd64\frp_0.62.1_windows_amd64\frpc.exe -c .\conf\frpc_8765_tcp.toml
```

因为这会读取 `E:\github_project\frp\conf\frpc_8765_tcp.toml`，而不是当前项目目录下的配置。

## 8. 验证步骤

建议严格按顺序验证。

### 8.1 先验证本机 WebUI 是否正常监听

在 Windows PowerShell 中执行：

```powershell
curl http://127.0.0.1:6688/
```

或检查监听端口：

```powershell
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -eq 6688 }
```

如果这里不通，`frp` 不可能工作正常。

### 8.2 验证控制端口是否能连通

```powershell
Test-NetConnection <第三方FRP服务器IP> -Port 47000
```

期望结果：

```text
TcpTestSucceeded : True
```

### 8.3 验证 `frpc` 启动日志

正常情况下，客户端启动后应出现类似日志：

```text
login to server success
start proxy success
```

如果出现下面这种错误：

```text
connect to local service [127.0.0.1:8765] error
```

则说明：

- `frpc` 实际读取到的配置里 `localPort` 仍然是 `8765`
- 或你启动的不是当前项目目录下这份配置

### 8.4 验证公网访问

最终公网访问地址为：

```text
http://<第三方FRP服务器IP>:48765/
```

如果该地址返回的就是本机 SmartGasTurbine WebUI 内容，则说明代理链路正常。

## 9. 日常启动与停止

### 9.1 Linux 服务端启动 `frps`

前台启动：

```bash
cd ./frp_0.62.1_linux_amd64
./frps -c ./frps.toml
```

后台启动：

```bash
cd ./frp_0.62.1_linux_amd64
nohup ./frps -c ./frps.toml > ./frps.log 2>&1 &
```

查看进程：

```bash
ps -ef | grep frps
```

停止进程：

```bash
pkill -f "./frps -c ./frps.toml"
```

### 9.2 Windows 本机启动 `frpc`

```powershell
Set-Location E:\my_project\SmartGasTurbine\frp
.\bin\frp_0.62.1_windows_amd64\frp_0.62.1_windows_amd64\frpc.exe -c .\conf\frpc_8765_tcp.toml
```

停止方式：

- 关闭当前 PowerShell 窗口
- 或在当前窗口按 `Ctrl+C`

## 10. 后续新增新的本地服务转发

假设未来你要新增一个本地服务：

- 本机服务：`127.0.0.1:9000`
- 希望公网访问端口：`49000`

### 10.1 规划端口

需要新增两类端口：

1. `frps` 服务端内部代理端口，例如：`19000`
2. NAT 公网映射端口，例如：`49000`

### 10.2 修改客户端配置

在 [conf/frpc_8765_tcp.toml](./conf/frpc_8765_tcp.toml) 中追加：

```toml
[[proxies]]
name = "local-9000-tcp"
type = "tcp"
localIP = "127.0.0.1"
localPort = 9000
remotePort = 19000
```

含义：

- `localPort = 9000`：Windows 本机实际服务端口
- `remotePort = 19000`：Linux 上由 `frps` 打开的服务端内部代理端口

### 10.3 新增 NAT 转发

新增一条 NAT 规则：

- 协议：`TCP`
- 外部端口：`49000`
- 内部端口：`19000`

### 10.4 重新启动 `frpc`

```powershell
Set-Location E:\my_project\SmartGasTurbine\frp
.\bin\frp_0.62.1_windows_amd64\frp_0.62.1_windows_amd64\frpc.exe -c .\conf\frpc_8765_tcp.toml
```

### 10.5 验证新服务

先验证本机：

```powershell
curl http://127.0.0.1:9000/
```

再验证公网：

```text
http://<第三方FRP服务器IP>:49000/
```

## 11. 常见问题

### 11.1 `bind: address already in use`

如果 Linux 上启动 `frps` 报：

```text
bind: address already in use
```

说明 `7000` 已被已有进程占用，通常是旧的 `frps` 已经在运行。

先检查：

```bash
ss -ltnp | grep :7000
ps -ef | grep frps
```

### 11.2 `connect to local service [127.0.0.1:8765] error`

说明客户端正在尝试访问 `8765`，但本机没有服务监听它。

当前项目正确端口是：

```text
127.0.0.1:6688
```

出现这个报错时，优先检查：

1. 当前启动的 `frpc` 是否来自 `SmartGasTurbine/frp`
2. 当前读取的配置文件是否就是 `./conf/frpc_8765_tcp.toml`
3. 配置里的 `localPort` 是否是 `6688`

### 11.3 本机能访问，公网不能访问

常见原因：

- `frpc` 没连上 `frps`
- NAT 外部端口未映射
- NAT 端口映射错了
- `remotePort` 和 NAT 内部端口不一致

### 11.4 如何避免再次混用错误配置

建议固定使用以下启动方式：

```powershell
Set-Location E:\my_project\SmartGasTurbine\frp
.\bin\frp_0.62.1_windows_amd64\frp_0.62.1_windows_amd64\frpc.exe -c .\conf\frpc_8765_tcp.toml
```

这样至少能保证：

- 程序来自当前项目目录
- 配置来自当前项目目录
- 日志排查也只看当前项目目录

## 12. 当前部署结论

当前链路应当是：

```text
公网访问 http://<第三方FRP服务器IP>:48765/
    -> NAT 外部端口 48765
    -> Linux 内部端口 18765
    -> frps
    -> frpc
    -> Windows 127.0.0.1:6688
```

后续新增转发时，保持以下原则即可：

1. 本机真实服务端口写在 `localPort`
2. Linux 服务端内部代理端口写在 `remotePort`
3. NAT 外部端口单独规划
4. 所有启动命令都从 `SmartGasTurbine/frp` 目录下用相对路径执行

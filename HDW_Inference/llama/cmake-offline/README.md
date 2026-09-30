# cmake-offline

`Dockerfile.container` 构建 llama.cpp 时用的**离线 cmake**。

## 为什么需要它

`nvidia/cuda:*-devel-*` 镜像里有 nvcc、gcc、make，**唯独没有 cmake**
（2026-10-01 在 `nvidia/cuda:13.0.2-devel-ubuntu24.04` 上确认）。有外网时在
Dockerfile 里 `apt-get install cmake` 就行；但无外网的机器（麒麟 V10 等）
装不了任何东西，所以得把 cmake 提前带进来。

**不能用宿主现成的 cmake。**宿主那种是按它自己的 glibc 编的，麒麟那类老发行版
根本跑不起来——这正是要走容器方案的原因，绕回去就自相矛盾了。
PyPI 上的 `cmake` wheel 是 manylinux_2_17 的，自包含、不挑 glibc，容器里直接能跑。

## 怎么生成

在**有外网**的机器上：

```bash
bash Scripts/fetch_offline_toolchain.sh
```

脚本会把 wheel 解压到这里，形成 `cmake-offline/cmake/data/bin/cmake`。
`Dockerfile.container` 按这个路径找它。

## 没放 wheel 会怎样

目录里只有本文件时，`COPY cmake-offline /opt/cmake-offline` 仍然成功（拷的是空
目录），构建阶段会退回用镜像自带的 cmake；镜像里没有就会在配置那一步明确报错。
也就是说：**有外网时这个目录可以空着**，只有离线构建才需要真放东西进来。

## 不要提交 wheel 本身

30 MB 的二进制，没必要进仓库。`.gitignore` 里已经把这个目录下的内容排除了，
只留本文件。

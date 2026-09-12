FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 构建期镜像源。**docker build 里不读宿主机的 pip.conf / apt 配置**，
# 唯一能传进去的途径就是 build-arg。基础镜像基于 Ubuntu 24.04。
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu|$APT_MIRROR/ubuntu|g; s|http://security.ubuntu.com/ubuntu|$APT_MIRROR/ubuntu|g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      ninja-build \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN python3 -m venv /opt/venv && \
    pip install -U -i "$PIP_INDEX_URL" pip setuptools wheel && \
    pip install -i "$PIP_INDEX_URL" -e ".[accel]"

EXPOSE 1919

CMD ["ft", "--help"]

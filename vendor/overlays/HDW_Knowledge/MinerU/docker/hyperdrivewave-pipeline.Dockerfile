FROM hyperdrivewave-hdw-rag:latest

WORKDIR /src

# apt 源。构建期同样不读宿主机的 apt 配置，只能靠 build-arg 注入。
# 留空 = 用镜像自带源；需要时传 --build-arg APT_MIRROR=https://mirrors.aliyun.com
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu|$APT_MIRROR/ubuntu|g; s|http://security.ubuntu.com/ubuntu|$APT_MIRROR/ubuntu|g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        fontconfig \
        fonts-noto-cjk \
        fonts-noto-core \
        libgl1 \
        libglib2.0-0 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# 构建期 pip 源。**docker build 里的 pip 不读宿主机的 /etc/pip.conf**，
# 唯一能传进去的途径就是 build-arg，所以这里必须声明 ARG。
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -e '.[pipeline]' && \
    mineru-models-download -s modelscope -m pipeline

ENV MINERU_MODEL_SOURCE=local \
    MINERU_API_OUTPUT_ROOT=/data/api_output

EXPOSE 8002
CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8002"]

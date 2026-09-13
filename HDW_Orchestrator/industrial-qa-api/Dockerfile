FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

# 构建期 pip 源。**docker build 里的 pip 不读宿主机的 /etc/pip.conf**，
# 唯一能传进去的途径就是 build-arg，所以这里必须声明 ARG。
# 换源/回官方源：docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt

# 中文字体：趋势曲线图由后端用 matplotlib 渲染，没有中文字体时
# 图上所有中文都会变成方框。python:3.11-slim 里一个中文字体都没有。
# fonts-noto-cjk 约 60MB；换 apt 源：--build-arg APT_MIRROR=...
ARG APT_MIRROR=""
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]


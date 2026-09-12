FROM python:3.13-slim

WORKDIR /app

COPY python-sdk /opt/mcp-sdk

# 构建期 pip 源。**docker build 里的 pip 不读宿主机的 /etc/pip.conf**，
# 唯一能传进去的途径就是 build-arg，所以这里必须声明 ARG。
# 换源/回官方源：docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -i "$PIP_INDEX_URL" -e /opt/mcp-sdk requests

COPY MCP_Tools /app/MCP_Tools

ENV PYTHONPATH=/app/MCP_Tools
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8766

CMD ["python", "/app/MCP_Tools/server.py", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8766", "--json-response"]

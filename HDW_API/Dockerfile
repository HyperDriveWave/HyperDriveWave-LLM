FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

# 构建期 pip 源。**docker build 里的 pip 不读宿主机的 /etc/pip.conf**，
# 唯一能传进去的途径就是 build-arg，所以这里必须声明 ARG。
# 换源/回官方源：docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt

COPY app.py store.py .

# 容器内监听端口固定 8095。改 HDW_API_PORT 只改宿主映射，不改这里——
# 与 qa-api 同一套约定（它是 "${HDW_QA_API_PORT}:8080"）。本服务不被任何
# 容器内部依赖，所以改宿主端口是安全的。
EXPOSE 8095
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8095"]

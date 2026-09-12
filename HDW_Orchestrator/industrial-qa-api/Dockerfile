FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

# 构建期 pip 源。**docker build 里的 pip 不读宿主机的 /etc/pip.conf**，
# 唯一能传进去的途径就是 build-arg，所以这里必须声明 ARG。
# 换源/回官方源：docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]


FROM python:3.12-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖(这一层不变则命中 docker 缓存,重建秒级)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py backtest.py config.yaml ./
COPY qtrader ./qtrader
COPY tests ./tests

# 状态全部落在 /app/data,compose 里挂 volume 即可实现"容器随便死,状态不丢"
VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)" || exit 1

CMD ["python", "main.py", "serve"]

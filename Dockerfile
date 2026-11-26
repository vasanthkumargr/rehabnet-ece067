# RehabNet server image. Build:  docker build -t rehabnet .
# Run:    docker run -p 8000:8000 -v rehabnet-data:/data -e REHABNET_JWT_SECRET=... -e REHABNET_DATA_KEY=... rehabnet
# Put it behind HTTPS (any cloud load balancer / reverse proxy): browsers only allow the camera on https:// pages.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 REHABNET_DATA_DIR=/data
WORKDIR /srv
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY rehabnet/ rehabnet/
COPY app/__init__.py app/
COPY app/server/ app/server/
COPY app/web/ app/web/
COPY app/data/references/ app/data/references/
RUN useradd -m rehabnet && mkdir -p /data && chown rehabnet /data
USER rehabnet
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "uvicorn", "app.server.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

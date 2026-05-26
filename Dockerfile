FROM python:3.11-slim

# HuggingFace Spaces runs as a non-root user; port 7860 is the default
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps for PyPDF + Redis Stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Redis Stack (for vector search)
# Note: Redis Stack packages may lag on Debian trixie; use bookworm repo for compatibility.
RUN set -eux; \
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb bookworm main" \
      > /etc/apt/sources.list.d/redis.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends redis-stack-server; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FAISS index dir (ephemeral per session — wiped on container restart)
RUN mkdir -p /app/faiss_indexes /app/context

ENV REDIS_URL=redis://localhost:6379
ENV CACHE_ENABLED=true

EXPOSE ${PORT}

CMD ["/bin/sh", "-c", "redis-stack-server --save '' --appendonly no & uvicorn rag_system.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]

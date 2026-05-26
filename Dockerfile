FROM redis/redis-stack-server:latest

# HuggingFace Spaces runs as a non-root user; port 7860 is the default
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

USER root

# System deps for PyPDF + Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  python3-venv \
  build-essential \
  libgomp1 \
  ca-certificates \
  curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# FAISS index dir (ephemeral per session — wiped on container restart)
RUN mkdir -p /app/faiss_indexes /app/context \
  && chmod -R 777 /app/faiss_indexes /app/context

ENV REDIS_URL=redis://localhost:6379
ENV CACHE_ENABLED=true

EXPOSE ${PORT}

CMD ["/bin/sh", "-c", "redis-stack-server --save '' --appendonly no & uvicorn rag_system.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]

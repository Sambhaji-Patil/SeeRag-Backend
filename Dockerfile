FROM python:3.11-slim

# HuggingFace Spaces runs as a non-root user; port 7860 is the default
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps for PyPDF and unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FAISS index dir (ephemeral per session — wiped on container restart)
RUN mkdir -p /app/faiss_indexes /app/context

EXPOSE ${PORT}

CMD uvicorn rag_system.api:app --host 0.0.0.0 --port ${PORT} --workers 1

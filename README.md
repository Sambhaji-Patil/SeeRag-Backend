---
title: RAG System
sdk: docker
app_port: 7860
pinned: false
---

# RAG System

Production-grade Retrieval Augmented Generation backend with FAISS, hybrid retrieval, reranking, and safety guardrails.

## API

- POST /ingest/file
- GET /ingest/jobs/{job_id}/events
- POST /query
- POST /query/pipeline
- GET /collections
- GET /collections/{collection_name}/viz
- POST /collections/{collection_name}/query_similarity
- GET /documents/{collection_name}/raw
- POST /evaluate

## Runtime

This Space uses the Docker runtime and expects a Dockerfile at the repo root.
The FastAPI app listens on port 8000.

## Environment Variables

Required:
- OPENAI_API_KEY

Optional:
- EMBEDDING_DEVICE (default: cuda)
- CACHE_ENABLED (default: false)
- REDIS_URL (default: redis://localhost:6379)
- HF_TOKEN (required only if downloading gated Llama Guard weights)

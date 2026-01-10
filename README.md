# RAG Sentiment Classifier (Ollama)

Minimal, functional prototype for classifying regulatory documents with a local Ollama LLM. This project is structured around clean, testable modules and provides both a REST API and CLI.

## Features
- Local Ollama LLM integration (no external API calls).
- Pydantic models and validated outputs.
- Centralized configuration via environment variables.
- Optional Redis-based LangChain cache.
- FastAPI service and CLI entry point.
- Dockerized workflow with Poetry.

## Quick Start (Local)

1. Install dependencies:

```bash
poetry install
```

2. Start Ollama locally and pull a model:

```bash
ollama serve
ollama pull llama3.1
```

3. Run the API:

```bash
poetry run uvicorn rag_sentiment_classifier.api:app --reload
```

4. Classify a document:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"document_id":"DOC-001","source":"local","content":"SEC Form 10-K annual report..."}'
```

## CLI Usage

```bash
poetry run python -m rag_sentiment_classifier.cli \
  --document-id DOC-001 \
  --source local \
  --content "SEC Form 10-K annual report..."
```

## Docker

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Start services:

```bash
docker compose up --build
```

3. Pull the model inside the Ollama container (one-time):

```bash
docker compose exec ollama ollama pull llama3.1
```

4. Call the API:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"document_id":"DOC-001","source":"docker","content":"AML policy updates..."}'
```

## Environment Variables

See `.env.example` for defaults:
- `OLLAMA_BASE_URL`: Base URL for Ollama (e.g., `http://ollama:11434`).
- `OLLAMA_MODEL`: Local model name (e.g., `llama3.1`).
- `REDIS_HOST` / `REDIS_PORT`: Optional Redis cache.

## Project Layout

```
src/rag_sentiment_classifier/
  api.py
  cli.py
  config/settings.py
  models/document.py
  prompts/classification_prompts.py
  services/classification_service.py
  utils/error_handler.py
```

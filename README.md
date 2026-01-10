# RAG Sentiment Classifier (Ollama)

Minimal, functional prototype for classifying regulatory documents with a local Ollama LLM. This project is structured around clean, testable modules and provides both a REST API and CLI.

## Features
- Local Ollama LLM integration (no external API calls).
- **API key authentication** for secure access control.
- **Rate limiting** to prevent abuse (configurable requests per minute).
- **Comprehensive input validation** and sanitization.
- **Secure Redis connection** with password authentication.
- Pydantic models and validated outputs.
- Centralized configuration via environment variables.
- Optional Redis-based LangChain cache with TTL.
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

> **Note**: API authentication is required if `API_KEY` or `API_KEYS` is configured in your environment.

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
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

4. **Important**: Edit your `.env` file and set secure values:
   - Set `API_KEY` or `API_KEYS` for authentication
   - Set `REDIS_PASSWORD` for Redis security

5. Call the API:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-from-env" \
  -d '{"document_id":"DOC-001","source":"docker","content":"AML policy updates..."}'
```

## Environment Variables

See `.env.example` for all configuration options. Key variables:

### Security (New!)
- `API_KEY`: Primary API key for authentication (leave empty to disable auth in dev)
- `API_KEYS`: Comma-separated list of additional API keys
- `RATE_LIMIT_ENABLED`: Enable/disable rate limiting (default: `true`)
- `RATE_LIMIT_REQUESTS`: Max requests per minute (default: `10`)
- `REDIS_PASSWORD`: Password for Redis authentication
- `REDIS_SSL`: Enable SSL for Redis connection (default: `false`)

### LLM Configuration
- `OLLAMA_BASE_URL`: Base URL for Ollama (e.g., `http://ollama:11434`)
- `OLLAMA_MODEL`: Local model name (e.g., `llama3.1`)
- `OLLAMA_TEMPERATURE`: Temperature for generation (default: `0.2`)
- `OLLAMA_MAX_TOKENS`: Max tokens in response (default: `512`)
- `LLAMA_TIMEOUT`: Request timeout in seconds (default: `60`)

### Cache Configuration
- `REDIS_HOST`: Redis hostname (default: `localhost`)
- `REDIS_PORT`: Redis port (default: `6379`)
- `REDIS_TTL`: Cache TTL in seconds (default: `3600`)

### Application
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `MAX_RETRIES`: Retry attempts on failure (default: `2`)
- `REQUEST_TIMEOUT`: Overall request timeout in seconds (default: `120`)

## Security

### Authentication
The API uses API key authentication via the `X-API-Key` header. Configure valid keys using the `API_KEY` or `API_KEYS` environment variables.

> **Development**: Leave `API_KEY` empty to run without authentication during local development.

> **Production**: Always set strong API keys in production environments!

### Rate Limiting
API endpoints are rate-limited to prevent abuse. Default is 10 requests per minute per IP address. Configure using `RATE_LIMIT_REQUESTS`.

### Input Validation
All inputs are validated and sanitized:
- Document IDs must match pattern: `[A-Za-z0-9-_]{1,100}`
- Content is stripped and sanitized (control characters removed)
- Metadata is size-limited (max 50 entries, values ≤ 1000 chars)

## Project Layout

```
src/rag_sentiment_classifier/
  api.py                          # FastAPI REST API
  cli.py                          # Command-line interface
  config/settings.py              # Configuration management
  models/document.py              # Pydantic data models
  prompts/classification_prompts.py  # LLM prompt templates
  services/classification_service.py # Classification logic
```

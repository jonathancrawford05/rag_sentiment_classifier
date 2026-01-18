# RAG Sentiment Classifier (Ollama)

Production-ready service for classifying regulatory documents with a local Ollama LLM. Built with modern Python architecture patterns including dependency injection, async/await, and comprehensive observability.

## Features

### Core Functionality
- **Local Ollama LLM integration** - No external API calls, complete data privacy
- **Async/await architecture** - Full async stack for optimal performance
- **Concurrent batch processing** - Process multiple documents in parallel with configurable concurrency
- **Dependency injection** - Clean, testable architecture with provider interfaces

### Security & Reliability
- **API key authentication** - Secure access control with multi-key support
- **Rate limiting** - Prevent abuse (configurable requests per minute)
- **Request timeout middleware** - Automatic timeout enforcement with slow request logging
- **Comprehensive input validation** - Sanitized and validated inputs
- **Secure Redis connection** - Password authentication and connection pooling

### Performance & Scalability
- **Redis connection pooling** - Efficient connection reuse with configurable pool size
- **Batch size limits** - Prevent resource exhaustion
- **Configurable concurrency** - Tune concurrent LLM calls to match infrastructure
- **Intelligent caching** - Redis-based LangChain cache with TTL

### Observability & Operations
- **Structured JSON logging** - Production-ready logging for log aggregators
- **Health check endpoints** - Detailed health checks for Kubernetes/Docker
- **FastAPI automatic docs** - Interactive API documentation at /docs
- **Docker health checks** - Container-level health monitoring
- **Resource limits** - Configured CPU and memory limits

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
# Single document classification
curl -X POST http://localhost:8081/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"document_id":"DOC-001","source":"local","content":"SEC Form 10-K annual report..."}'

# Batch classification (async processing)
curl -X POST http://localhost:8081/classify/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '[
    {"document_id":"DOC-001","source":"local","content":"SEC Form 10-K annual report..."},
    {"document_id":"DOC-002","source":"local","content":"GDPR compliance document..."}
  ]'
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
# Single document
curl -X POST http://localhost:8081/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-from-env" \
  -d '{"document_id":"DOC-001","source":"docker","content":"AML policy updates..."}'

# Health check
curl http://localhost:8081/health

# Detailed health check (includes Redis status)
curl http://localhost:8081/health/detailed
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
- `REDIS_MAX_CONNECTIONS`: Connection pool size (default: `50`)
- `REDIS_SOCKET_TIMEOUT`: Socket timeout in seconds (default: `5`)
- `REDIS_SOCKET_CONNECT_TIMEOUT`: Connection timeout in seconds (default: `5`)

### Performance Configuration
- `MAX_CONCURRENT_CLASSIFICATIONS`: Max concurrent LLM calls (default: `5`)
- `MAX_BATCH_SIZE`: Max documents per batch request (default: `50`)
- `ENABLE_REQUEST_TIMEOUT`: Enable request timeout middleware (default: `true`)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: `120`)

### Application
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `JSON_LOGS`: Enable JSON logging for production (default: `false`)
- `MAX_RETRIES`: Retry attempts on failure (default: `2`)
- `RETRY_DELAY`: Initial retry delay in seconds (default: `1.0`)

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

## Architecture

### Provider Pattern
The application uses a provider-based architecture with dependency injection:

- **LLMProvider Protocol** - Interface for LLM implementations
- **CacheProvider Protocol** - Interface for cache implementations
- **OllamaLLMProvider** - Concrete Ollama implementation
- **RedisCacheProvider** - Async Redis cache with connection pooling

### Async/Await Stack
Full async implementation from API endpoints to service methods:
- FastAPI endpoints use `async def`
- Service methods use `asyncio.gather` for concurrent processing
- Semaphore-based concurrency control
- Async Redis client with connection pooling

### Middleware
- **TimeoutMiddleware** - Request timeout enforcement with slow request logging
- Health check endpoints bypass timeout

## Project Layout

```
src/rag_sentiment_classifier/
  api.py                                # FastAPI REST API with lifespan management
  cli.py                                # Command-line interface
  config/
    settings.py                         # Centralized configuration
    logging_config.py                   # JSON logging configuration
  middleware/
    timeout.py                          # Request timeout middleware
  models/
    document.py                         # Pydantic data models
  prompts/
    classification_prompts.py           # LLM prompt templates
  providers/
    llm_provider.py                     # LLM provider protocol
    cache_provider.py                   # Cache provider protocol
    ollama_provider.py                  # Ollama LLM implementation
    redis_provider.py                   # Redis cache implementation
  services/
    classification_service.py           # Classification business logic
```

## API Endpoints

- `POST /classify` - Classify a single document
- `POST /classify/batch` - Classify multiple documents concurrently
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health check with dependency status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src/rag_sentiment_classifier --cov-report=html

# Run specific test categories
poetry run pytest tests/unit/          # Unit tests only
poetry run pytest tests/integration/   # Integration tests only

# View coverage report
open htmlcov/index.html
```

**Test Coverage:**
- 78 passing tests
- Unit tests for all providers, services, and API endpoints
- Integration tests for cache and service interactions
- 80%+ code coverage

## Development

### Code Quality

```bash
# Lint and format code
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/

# Type checking
poetry run mypy src/

# Run all quality checks
poetry run ruff check src/ tests/ && \
poetry run ruff format src/ tests/ && \
poetry run pytest --cov
```

### Local Development Workflow

1. Make code changes
2. Run linter and formatter: `poetry run ruff check . --fix && poetry run ruff format .`
3. Run tests: `poetry run pytest`
4. Test in Docker: `docker compose up --build`
5. Check health: `curl http://localhost:8081/health/detailed`

## Production Deployment

### Environment Configuration

For production, enable these critical settings:

```bash
# Security
API_KEY=your-secure-api-key-here
REDIS_PASSWORD=your-secure-redis-password

# Performance
MAX_CONCURRENT_CLASSIFICATIONS=10
MAX_BATCH_SIZE=100
REDIS_MAX_CONNECTIONS=100

# Logging
JSON_LOGS=true
LOG_LEVEL=INFO

# Timeouts
ENABLE_REQUEST_TIMEOUT=true
REQUEST_TIMEOUT=120
```

### Docker Production Settings

The application includes:
- Multi-stage Docker builds for smaller images
- Health checks for container orchestration
- Resource limits (CPU and memory)
- Non-root user for security
- Automatic Redis connection pooling

### Monitoring

Key metrics to monitor:
- Request duration (TimeoutMiddleware logs slow requests >5s)
- Cache hit rate (check Redis metrics)
- Classification success rate
- Concurrent request count
- Error rates by endpoint

### Scaling

The application supports horizontal scaling:
- Stateless design (cache in Redis)
- Async request handling
- Configurable concurrency limits
- Connection pooling for Redis

## Troubleshooting

### Common Issues

**Redis connection fails:**
- Check `REDIS_HOST` and `REDIS_PORT` settings
- Verify Redis is running: `docker compose ps`
- Check Redis password matches `REDIS_PASSWORD`

**Ollama timeout:**
- Increase `LLAMA_TIMEOUT` setting
- Check Ollama service health: `curl http://localhost:11434/api/tags`
- Ensure model is downloaded: `docker compose exec ollama ollama list`

**Slow classification:**
- Enable caching with Redis
- Increase `MAX_CONCURRENT_CLASSIFICATIONS` for batch processing
- Check `OLLAMA_TEMPERATURE` and `OLLAMA_MAX_TOKENS` settings
- Monitor with `/health/detailed` endpoint

**Request timeout (504):**
- Increase `REQUEST_TIMEOUT` setting
- Check slow request logs for bottlenecks
- Consider reducing batch size with `MAX_BATCH_SIZE`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run quality checks: `poetry run ruff check . && poetry run pytest`
5. Commit with clear messages
6. Push and create a pull request

## License

[Your License Here]

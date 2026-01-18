# Architecture Documentation
## RAG Sentiment Classifier

**Version:** 2.0
**Date:** 2026-01-18
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Patterns](#architecture-patterns)
3. [System Components](#system-components)
4. [Data Flow](#data-flow)
5. [Async/Await Implementation](#asyncawait-implementation)
6. [Dependency Injection](#dependency-injection)
7. [Middleware Stack](#middleware-stack)
8. [Error Handling](#error-handling)
9. [Performance Optimizations](#performance-optimizations)
10. [Security Architecture](#security-architecture)

---

## Overview

The RAG Sentiment Classifier is a production-ready microservice for classifying regulatory documents using local LLMs (Ollama). The architecture follows modern Python best practices with emphasis on:

- **Testability** - Dependency injection enables comprehensive unit testing
- **Scalability** - Async/await pattern with concurrent processing
- **Reliability** - Timeout controls, retry logic, and health checks
- **Security** - API key authentication, rate limiting, input validation
- **Observability** - Structured logging, health endpoints, slow request tracking

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│                  (HTTP Requests with API Keys)                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Middleware Stack                            │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Authentication   │  │ Rate Limiting    │                     │
│  │ Middleware       │  │ Middleware       │                     │
│  └────────┬─────────┘  └────────┬─────────┘                    │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌──────────────────────────────────────────┐                   │
│  │      Timeout Middleware                  │                   │
│  │  - Request timeout enforcement           │                   │
│  │  - Slow request logging (>5s)            │                   │
│  └────────────────┬─────────────────────────┘                   │
└────────────────────┼─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ POST /classify│  │POST /classify│  │GET /health   │         │
│  │              │  │   /batch     │  │  /detailed   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
│         │                  │                                     │
│         │    ┌─────────────┴─────────────┐                     │
│         │    │  Lifespan Context Manager │                     │
│         │    │  - Provider initialization│                     │
│         │    │  - Graceful shutdown      │                     │
│         │    └───────────────────────────┘                     │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│               Dependency Injection Layer                         │
│  ┌──────────────────────────────────────────────────┐           │
│  │      get_classification_service()                │           │
│  │  - Creates OllamaLLMProvider                     │           │
│  │  - Injects RedisCacheProvider                    │           │
│  │  - Configures concurrency settings               │           │
│  └────────────────────┬─────────────────────────────┘           │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Service Layer                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │      DocumentClassificationService                     │     │
│  │  ┌──────────────────────────────────────────────────┐ │     │
│  │  │  async classify_document(doc)                   │ │     │
│  │  │  - Cache lookup (if enabled)                    │ │     │
│  │  │  - LLM classification via provider              │ │     │
│  │  │  - Retry logic with exponential backoff         │ │     │
│  │  │  - Cache storage                                │ │     │
│  │  └──────────────────────────────────────────────────┘ │     │
│  │                                                         │     │
│  │  ┌──────────────────────────────────────────────────┐ │     │
│  │  │  async classify_batch(documents)                │ │     │
│  │  │  - Semaphore-based concurrency control          │ │     │
│  │  │  - asyncio.gather for parallel processing       │ │     │
│  │  │  - Max concurrent configurable                  │ │     │
│  │  └──────────────────────────────────────────────────┘ │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────┬──────────────────────┬────────────────────────┘
                  │                      │
                  ▼                      ▼
┌─────────────────────────────┐ ┌──────────────────────────────┐
│     Provider Layer           │ │    Provider Layer            │
│  ┌─────────────────────────┐ │ │  ┌─────────────────────────┐│
│  │  OllamaLLMProvider      │ │ │  │  RedisCacheProvider     ││
│  │  (implements Protocol)  │ │ │  │  (implements Protocol)  ││
│  │                         │ │ │  │                         ││
│  │  - LangChain integration│ │ │  │  - Async Redis client   ││
│  │  - Prompt management    │ │ │  │  - Connection pooling   ││
│  │  - Structured output    │ │ │  │  - JSON serialization   ││
│  └──────────┬──────────────┘ │ │  └──────────┬──────────────┘│
└─────────────┼────────────────┘ └─────────────┼───────────────┘
              │                                 │
              ▼                                 ▼
┌─────────────────────────────┐ ┌──────────────────────────────┐
│      External Service        │ │    External Service          │
│  ┌─────────────────────────┐ │ │  ┌─────────────────────────┐│
│  │   Ollama LLM Service    │ │ │  │   Redis Cache           ││
│  │   (Docker Container)    │ │ │  │   (Docker Container)    ││
│  └─────────────────────────┘ │ │  └─────────────────────────┘│
└─────────────────────────────┘ └──────────────────────────────┘
```

---

## Architecture Patterns

### 1. Provider Pattern with Protocols

**Pattern:** Interface segregation using Python's `Protocol` (PEP 544)

```python
# providers/llm_provider.py
@runtime_checkable
class LLMProvider(Protocol):
    async def classify(self, document_id: str, content: str) -> ClassificationResult:
        ...

# providers/cache_provider.py
@runtime_checkable
class CacheProvider(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...
```

**Benefits:**
- **Loose coupling** - Service depends on interfaces, not implementations
- **Testability** - Easy to create mock providers for testing
- **Flexibility** - Swap implementations (e.g., OpenAI provider, Memcached provider)
- **Type safety** - Protocol compliance checked at runtime

### 2. Dependency Injection

**Pattern:** Constructor injection with FastAPI dependency system

```python
# api.py
def get_classification_service() -> DocumentClassificationService:
    llm_provider = OllamaLLMProvider(...)
    return DocumentClassificationService(
        llm_provider=llm_provider,
        cache_provider=cache_provider,
        max_concurrent=settings.max_concurrent_classifications,
    )

@app.post("/classify")
async def classify_document(
    document: DocumentInput,
    service: DocumentClassificationService = Depends(get_classification_service),
):
    return await service.classify_document(document)
```

**Benefits:**
- **Testability** - Inject mock dependencies in tests
- **Configuration flexibility** - Different configs for dev/staging/prod
- **Single Responsibility** - Service doesn't manage its dependencies
- **Lifecycle management** - FastAPI handles provider creation/cleanup

### 3. Async/Await Throughout

**Pattern:** Full async stack from API to external services

```python
# Async endpoint
@app.post("/classify")
async def classify_document(...): ...

# Async service method
async def classify_document(self, document: DocumentInput) -> ClassificationResult:
    # Async cache lookup
    cached = await self.cache_provider.get(cache_key)

    # Async LLM call
    result = await self.llm_provider.classify(...)

    # Async cache storage
    await self.cache_provider.set(cache_key, result.model_dump())

    return result
```

**Benefits:**
- **Concurrency** - Handle multiple requests without blocking
- **Resource efficiency** - Minimal thread overhead
- **Scalability** - Better throughput under load
- **I/O optimization** - Efficient waiting for external services

### 4. Concurrent Batch Processing

**Pattern:** Semaphore-controlled parallel processing with `asyncio.gather`

```python
async def classify_batch(self, documents: list[DocumentInput]) -> list[ClassificationResult]:
    semaphore = asyncio.Semaphore(self.max_concurrent)

    async def classify_with_semaphore(doc: DocumentInput):
        async with semaphore:
            return await self.classify_document(doc)

    tasks = [classify_with_semaphore(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r is not None]
```

**Benefits:**
- **Controlled parallelism** - Prevents overwhelming LLM service
- **Throughput** - Process multiple documents concurrently
- **Backpressure** - Semaphore limits concurrent operations
- **Error isolation** - Individual task failures don't crash entire batch

---

## System Components

### FastAPI Application Layer

**File:** `src/rag_sentiment_classifier/api.py`

**Responsibilities:**
- HTTP request/response handling
- Request validation with Pydantic
- Authentication and rate limiting
- Dependency injection
- Resource lifecycle management (lifespan)
- Health check endpoints

**Key Features:**
- Lifespan context manager for startup/shutdown
- FastAPI dependency injection for services
- Automatic OpenAPI documentation
- Request/response logging

### Service Layer

**File:** `src/rag_sentiment_classifier/services/classification_service.py`

**Class:** `DocumentClassificationService`

**Responsibilities:**
- Business logic for document classification
- Cache management
- Retry logic with exponential backoff
- Batch processing orchestration
- Error handling and logging

**Key Methods:**
- `classify_document(document)` - Single document classification
- `classify_batch(documents)` - Concurrent batch processing
- `_classify_with_retry(document, attempt)` - Retry logic

### Provider Layer

#### LLM Provider

**File:** `src/rag_sentiment_classifier/providers/ollama_provider.py`

**Class:** `OllamaLLMProvider`

**Responsibilities:**
- LangChain integration
- Prompt template management
- Structured output parsing
- LLM invocation

**Implementation:**
```python
async def classify(self, document_id: str, content: str) -> ClassificationResult:
    result: ClassificationResult = await self.classification_chain.ainvoke(
        {"document_id": document_id, "content": content}
    )
    result.document_id = document_id
    return result
```

#### Cache Provider

**File:** `src/rag_sentiment_classifier/providers/redis_provider.py`

**Class:** `RedisCacheProvider`

**Responsibilities:**
- Redis connection management
- Connection pooling
- JSON serialization/deserialization
- Cache operations (get, set, delete)
- Health checks

**Features:**
- Configurable connection pool size
- Socket timeout configuration
- Async operations
- Automatic JSON encoding/decoding

### Middleware Layer

#### TimeoutMiddleware

**File:** `src/rag_sentiment_classifier/middleware/timeout.py`

**Responsibilities:**
- Request timeout enforcement
- Slow request logging
- Timeout error responses

**Features:**
- Configurable timeout per request
- Bypasses health check endpoints
- Logs requests exceeding 5 seconds
- Returns 504 Gateway Timeout with details

### Configuration Layer

**File:** `src/rag_sentiment_classifier/config/settings.py`

**Class:** `Settings` (Pydantic BaseSettings)

**Categories:**
- **Security** - API keys, Redis password
- **LLM** - Ollama host, model, temperature
- **Cache** - Redis connection settings, connection pooling
- **Performance** - Concurrency, batch size, timeouts
- **Application** - Logging, retry configuration

---

## Data Flow

### Single Document Classification Flow

```
1. Client Request
   └─> POST /classify with API key header
       └─> Authentication Middleware (verify API key)
           └─> Rate Limiting Middleware (check rate limits)
               └─> Timeout Middleware (start timeout timer)
                   └─> FastAPI Endpoint Handler
                       └─> get_classification_service() (DI)
                           └─> DocumentClassificationService.classify_document()

                               [Cache Check]
                               └─> cache_provider.get(cache_key)
                                   ├─> Cache Hit: Return cached result
                                   └─> Cache Miss: Continue

                               [LLM Classification with Retry]
                               └─> llm_provider.classify(document_id, content)
                                   └─> LangChain chain.ainvoke()
                                       └─> Ollama LLM Service
                                           └─> Returns ClassificationResult

                               [Cache Storage]
                               └─> cache_provider.set(cache_key, result)

                               └─> Return ClassificationResult
                                   └─> FastAPI JSON Response
                                       └─> Client receives response
```

### Batch Classification Flow

```
1. Client Request
   └─> POST /classify/batch with API key header
       └─> [Middleware Stack: Auth → Rate Limit → Timeout]
           └─> FastAPI Endpoint Handler

               [Batch Size Validation]
               └─> if len(documents) > MAX_BATCH_SIZE: raise HTTPException

               └─> DocumentClassificationService.classify_batch(documents)

                   [Create Semaphore]
                   └─> semaphore = asyncio.Semaphore(max_concurrent)

                   [Create Tasks]
                   └─> For each document:
                       └─> async with semaphore:
                           └─> classify_document(doc)

                   [Concurrent Execution]
                   └─> asyncio.gather(*tasks)
                       └─> [Doc1, Doc2, Doc3, ...] processed concurrently
                           └─> Each follows single document flow
                           └─> Semaphore limits to max_concurrent at a time

                   └─> Return list of ClassificationResults
                       └─> FastAPI JSON Response
                           └─> Client receives batch results
```

---

## Async/Await Implementation

### Why Full Async Stack?

1. **I/O Bound Operations**
   - LLM calls can take 1-10+ seconds
   - Redis operations are network I/O
   - Async allows handling other requests while waiting

2. **Scalability**
   - Single process can handle hundreds of concurrent requests
   - Minimal memory overhead compared to threads
   - Better resource utilization

3. **Performance**
   - Batch processing with concurrent execution
   - Non-blocking cache lookups
   - Efficient connection pooling

### Implementation Details

**Async Service Methods:**
```python
class DocumentClassificationService:
    async def classify_document(self, document: DocumentInput) -> ClassificationResult:
        # All I/O operations are async
        cached = await self.cache_provider.get(cache_key)
        result = await self.llm_provider.classify(...)
        await self.cache_provider.set(cache_key, result.model_dump())
        return result
```

**Async Provider Methods:**
```python
class OllamaLLMProvider:
    async def classify(self, document_id: str, content: str) -> ClassificationResult:
        # LangChain async invocation
        result = await self.classification_chain.ainvoke(...)
        return result

class RedisCacheProvider:
    async def get(self, key: str) -> Any | None:
        # Async Redis operations
        value = await self.client.get(key)
        return json.loads(value) if value else None
```

**Concurrent Batch Processing:**
```python
async def classify_batch(self, documents: list[DocumentInput]):
    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(self.max_concurrent)

    async def classify_with_semaphore(doc):
        async with semaphore:  # Limit concurrent executions
            return await self.classify_document(doc)

    # Create all tasks
    tasks = [classify_with_semaphore(doc) for doc in documents]

    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results
```

---

## Dependency Injection

### DI Container Pattern

**Lifespan Management:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global cache_provider

    # Startup: Initialize shared resources
    if settings.redis_host:
        cache_provider = RedisCacheProvider(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            max_connections=settings.redis_max_connections,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
        )

    yield  # Application runs

    # Shutdown: Cleanup resources
    if cache_provider:
        await cache_provider.close()
```

**Service Factory:**
```python
def get_classification_service() -> DocumentClassificationService:
    # Create LLM provider
    llm_provider = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
    )

    # Inject dependencies
    return DocumentClassificationService(
        llm_provider=llm_provider,
        cache_provider=cache_provider,  # Global from lifespan
        max_retries=settings.max_retries,
        retry_delay=settings.retry_delay,
        max_concurrent=settings.max_concurrent_classifications,
    )
```

**Endpoint Injection:**
```python
@app.post("/classify")
async def classify_document(
    document: DocumentInput,
    service: DocumentClassificationService = Depends(get_classification_service),
):
    # Service is created per request with fresh dependencies
    return await service.classify_document(document)
```

### Benefits for Testing

**Mock Injection:**
```python
# In tests
def test_classify_document():
    mock_llm = Mock(spec=LLMProvider)
    mock_cache = Mock(spec=CacheProvider)

    service = DocumentClassificationService(
        llm_provider=mock_llm,
        cache_provider=mock_cache,
    )

    # Test with mocks
    result = await service.classify_document(test_doc)
```

---

## Middleware Stack

### Execution Order

1. **AuthenticationMiddleware** (first)
   - Validates API key in X-API-Key header
   - Returns 401 if missing or invalid
   - Bypasses certain paths (/health, /docs)

2. **RateLimitMiddleware** (second)
   - Tracks requests per client
   - Enforces rate limits per API key
   - Returns 429 Too Many Requests if exceeded

3. **TimeoutMiddleware** (third)
   - Wraps request in asyncio.wait_for()
   - Enforces configurable timeout
   - Logs slow requests (>5s)
   - Returns 504 Gateway Timeout if exceeded
   - Bypasses health check endpoints

### TimeoutMiddleware Implementation

```python
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip timeout for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)

        start_time = time.time()

        try:
            # Enforce timeout with asyncio
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )

            # Log slow requests
            duration = time.time() - start_time
            if duration > 5.0:
                logger.warning(
                    "Slow request: %s %s took %.2fs",
                    request.method,
                    request.url.path,
                    duration,
                )

            return response

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(
                "Request timeout: %s %s exceeded %ds",
                request.method,
                request.url.path,
                self.timeout,
            )
            return JSONResponse(
                status_code=504,
                content={
                    "detail": f"Request exceeded {self.timeout}s timeout",
                    "timeout_seconds": self.timeout,
                    "duration_seconds": round(duration, 2),
                },
            )
```

---

## Error Handling

### Retry Logic with Exponential Backoff

```python
async def _classify_with_retry(
    self, document: DocumentInput, attempt: int = 0
) -> ClassificationResult:
    try:
        return await self.llm_provider.classify(
            document.document_id,
            document.content
        )
    except Exception as e:
        if attempt < self.max_retries:
            delay = self.retry_delay * (2**attempt)
            logger.warning(
                "Classification failed for %s (attempt %d/%d): %s. "
                "Retrying in %.1fs",
                document.document_id,
                attempt + 1,
                self.max_retries,
                str(e),
                delay,
            )
            await asyncio.sleep(delay)
            return await self._classify_with_retry(document, attempt + 1)
        else:
            logger.error(
                "Classification failed for %s after %d retries: %s",
                document.document_id,
                self.max_retries,
                str(e),
            )
            raise
```

**Features:**
- Exponential backoff (1s, 2s, 4s, ...)
- Configurable max retries
- Detailed logging at each attempt
- Preserves original exception after final retry

### HTTP Error Responses

**400 Bad Request:**
- Invalid document input
- Batch size exceeds limit
- Validation errors

**401 Unauthorized:**
- Missing API key
- Invalid API key

**429 Too Many Requests:**
- Rate limit exceeded

**504 Gateway Timeout:**
- Request exceeded timeout
- Includes duration and timeout details

**500 Internal Server Error:**
- Unexpected errors
- LLM service failures after retries

---

## Performance Optimizations

### 1. Redis Connection Pooling

**Configuration:**
```python
redis_params = {
    "host": host,
    "port": port,
    "decode_responses": True,
    "max_connections": max_connections,  # Default: 50
    "socket_timeout": socket_timeout,    # Default: 5s
    "socket_connect_timeout": socket_connect_timeout,  # Default: 5s
}
self.client = redis.Redis(**redis_params)
```

**Benefits:**
- Reuses connections across requests
- Reduces connection establishment overhead
- Prevents connection exhaustion
- Configurable pool size for scaling

### 2. Concurrent Batch Processing

**Semaphore-based Concurrency:**
```python
semaphore = asyncio.Semaphore(self.max_concurrent)  # Default: 5

async def classify_with_semaphore(doc):
    async with semaphore:  # Only max_concurrent run at once
        return await self.classify_document(doc)
```

**Performance Impact:**
- 5 documents in parallel: ~5x faster than sequential
- Configurable based on LLM service capacity
- Prevents overwhelming external services

### 3. Request Timeout Enforcement

**Timeout Middleware:**
- Prevents hanging requests
- Configurable timeout (default 120s)
- Automatically logs slow requests
- Returns clear error responses

### 4. Batch Size Limits

**Validation:**
```python
if len(documents) > settings.max_batch_size:
    raise HTTPException(
        status_code=400,
        detail=f"Batch size {len(documents)} exceeds maximum of {settings.max_batch_size}",
    )
```

**Benefits:**
- Prevents memory exhaustion
- Protects against malicious requests
- Ensures predictable performance

### 5. LangChain Caching

**Redis-backed Cache:**
- Caches LLM responses by content hash
- Configurable TTL (default 1 hour)
- Reduces redundant LLM calls
- Significant cost savings for repeated content

---

## Security Architecture

### 1. Authentication

**API Key Authentication:**
- Required for all endpoints (except /health, /docs)
- Multiple keys supported via `API_KEYS` comma-separated list
- Keys validated in middleware layer
- Invalid/missing keys return 401

### 2. Rate Limiting

**Per-Key Rate Limiting:**
- Configurable requests per minute
- In-memory tracking (can be extended to Redis)
- Returns 429 with Retry-After header
- Prevents abuse and DoS attacks

### 3. Input Validation

**Pydantic Validation:**
- Document ID pattern: `[A-Za-z0-9-_]{1,100}`
- Content max length: 50,000 characters
- Metadata size limits: 50 entries, 1000 chars per value
- Control characters stripped
- JSON schema validation

### 4. Secure Redis Connection

**Configuration:**
- Password authentication required
- Connection timeout limits
- Socket timeout prevents hanging
- TLS support (configure via Redis client)

### 5. Resource Limits

**Docker Configuration:**
```yaml
resources:
  limits:
    cpus: '2.0'
    memory: 2G
  reservations:
    cpus: '1.0'
    memory: 1G
```

**Application Limits:**
- Max batch size: 50 documents
- Max concurrent: 5 LLM calls
- Request timeout: 120 seconds
- Connection pool limits

---

## Design Decisions

### Why Protocols Instead of Abstract Base Classes?

**Reasoning:**
- More Pythonic (PEP 544)
- Structural subtyping (duck typing)
- No inheritance required
- Better for testing (easier mocking)
- Runtime checkable with `@runtime_checkable`

### Why Async Throughout?

**Reasoning:**
- I/O bound operations (LLM, Redis)
- Better concurrency than threads
- Lower memory overhead
- Native support in FastAPI
- Enables efficient batch processing

### Why Dependency Injection?

**Reasoning:**
- Testability (inject mocks)
- Flexibility (swap implementations)
- Configuration management
- Lifecycle control
- Follows SOLID principles

### Why Semaphore for Concurrency?

**Reasoning:**
- Simple and effective backpressure
- Prevents overwhelming LLM service
- Configurable concurrency level
- Built into asyncio (no dependencies)
- Easy to understand and debug

---

## Future Enhancements

### 1. Observability
- Prometheus metrics integration
- OpenTelemetry tracing
- Structured log aggregation (ELK, Datadog)
- Performance dashboards

### 2. Scalability
- Kubernetes deployment
- Horizontal pod autoscaling
- Redis Cluster support
- Load balancing

### 3. Features
- Multiple LLM provider support (OpenAI, Anthropic)
- Webhook notifications
- Async job processing (Celery, RQ)
- Document preprocessing pipeline

### 4. Testing
- Contract testing for providers
- Load testing with Locust
- Chaos engineering tests
- End-to-end integration tests

---

**Document Version:** 2.0
**Last Updated:** 2026-01-18
**Next Review:** 2026-02-18

# Refactor Plan
## RAG Sentiment Classifier - Path to Production

**Version:** 2.0
**Date:** 2026-01-18 (Updated)
**Original Date:** 2026-01-10
**Target Completion:** 12 weeks
**Status:** Phase 5 Complete ✅

---

## Progress Update (2026-01-18)

**Completed Phases:**
- ✅ **Phase 1** - Critical Security & Cleanup (Weeks 1-2) - COMPLETED
- ✅ **Phase 2** - Testing Infrastructure (Weeks 3-4) - COMPLETED
- ✅ **Phase 3** - Architecture Refactoring (Weeks 5-6) - COMPLETED
- ✅ **Phase 4** - DevOps & Infrastructure (Weeks 7-8) - COMPLETED
- ✅ **Phase 5** - Performance & Optimization (Weeks 9-10) - COMPLETED
- ⏳ **Phase 6** - Documentation & Polish (Weeks 11-12) - IN PROGRESS

**Current Status:**
The application has been successfully transformed from a prototype to a production-ready system. All critical security, testing, architecture, and performance improvements have been implemented. Documentation is being finalized.

---

## Overview

This document outlines a comprehensive refactor plan to transform the RAG Sentiment Classifier from a functional prototype to a production-ready system. The plan is organized into phases with clear priorities, actionable tasks, and measurable outcomes.

---

## Phase 1: Critical Security & Cleanup (Weeks 1-2)

**Goal:** Address immediate security vulnerabilities and remove technical debt

**Priority:** 🔴 CRITICAL

### 1.1 Security Hardening (Week 1)

#### Task 1.1.1: Implement API Key Authentication
**Files to modify:**
- `src/rag_sentiment_classifier/api.py`
- `src/rag_sentiment_classifier/config/settings.py`

**Changes:**
```python
# settings.py - Add
api_key: str = ""
api_keys: list[str] = []  # Support multiple API keys

# api.py - Add dependency
from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    settings = get_settings()
    valid_keys = settings.api_keys or ([settings.api_key] if settings.api_key else [])
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Apply to endpoints
@app.post("/classify", dependencies=[Depends(verify_api_key)])
```

**Acceptance Criteria:**
- [ ] API key validation on all protected endpoints
- [ ] Support for multiple API keys
- [ ] Environment variable configuration
- [ ] 403 response for invalid keys
- [ ] Health check endpoint remains open

---

#### Task 1.1.2: Add Rate Limiting
**New dependency:** `slowapi`

**Files to modify:**
- `pyproject.toml` - Add dependency
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/classify")
@limiter.limit("10/minute")  # Configurable via settings
async def classify_document(...):
```

**Acceptance Criteria:**
- [ ] Rate limiting on classify endpoint
- [ ] Configurable limits via environment variables
- [ ] 429 Too Many Requests response
- [ ] Per-IP tracking

---

#### Task 1.1.3: Secure Redis Connection
**Files to modify:**
- `src/rag_sentiment_classifier/config/settings.py`
- `src/rag_sentiment_classifier/services/classification_service.py`
- `docker-compose.yml`

**Changes:**
```python
# settings.py - Add
redis_password: str = ""
redis_ssl: bool = False

# classification_service.py
redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password if settings.redis_password else None,
    ssl=settings.redis_ssl,
    decode_responses=True,
)

# docker-compose.yml - Update redis service
redis:
  image: redis:7-alpine
  command: redis-server --requirepass ${REDIS_PASSWORD}
  environment:
    - REDIS_PASSWORD=${REDIS_PASSWORD}
```

**Acceptance Criteria:**
- [ ] Redis password authentication
- [ ] TLS support for production
- [ ] Graceful fallback if Redis unavailable
- [ ] Connection validation on startup

---

#### Task 1.1.4: Add Input Validation & Sanitization
**Files to modify:**
- `src/rag_sentiment_classifier/models/document.py`

**Changes:**
```python
from pydantic import field_validator, Field

class DocumentInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000)
    document_id: str = Field(..., pattern=r'^[A-Za-z0-9-_]{1,100}$')
    source: str = Field(..., min_length=1, max_length=100)
    metadata: Optional[dict] = Field(None, max_items=50)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Optional[dict]) -> Optional[dict]:
        if value is None:
            return value
        # Validate metadata keys and values
        for key, val in value.items():
            if not isinstance(key, str) or len(key) > 100:
                raise ValueError("Invalid metadata key")
            if isinstance(val, str) and len(val) > 1000:
                raise ValueError("Metadata value too long")
        return value

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, value: str) -> str:
        # Basic sanitization (extend as needed)
        stripped = value.strip()
        if not stripped:
            raise ValueError("Content cannot be empty")
        # Remove null bytes and other control characters
        sanitized = "".join(char for char in stripped if ord(char) >= 32 or char in '\n\r\t')
        return sanitized
```

**Acceptance Criteria:**
- [ ] Document ID pattern validation
- [ ] Metadata size and structure validation
- [ ] Content sanitization for control characters
- [ ] Clear validation error messages

---

### 1.2 Code Cleanup (Week 2)

#### Task 1.2.1: Remove Unused Code
**Files to modify/delete:**
- DELETE: `src/rag_sentiment_classifier/utils/error_handler.py`
- `src/rag_sentiment_classifier/models/document.py` - Remove ProcessingError

**Changes:**
```bash
# Delete entire error_handler.py module
rm src/rag_sentiment_classifier/utils/error_handler.py

# Remove ProcessingError from models/document.py (lines 42-50)
# Remove unused imports
```

**Acceptance Criteria:**
- [ ] error_handler.py deleted
- [ ] ProcessingError model removed
- [ ] No import errors
- [ ] All tests pass

---

#### Task 1.2.2: Fix Deprecated API Usage
**Files to modify:**
- `src/rag_sentiment_classifier/models/document.py`

**Changes:**
```python
from datetime import datetime, timezone

class ClassificationResult(BaseModel):
    # ...
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProcessingError(BaseModel):  # If keeping this model
    # ...
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

**Acceptance Criteria:**
- [ ] Replace all `datetime.utcnow()` with `datetime.now(timezone.utc)`
- [ ] Python 3.12 compatibility verified
- [ ] Tests pass

---

#### Task 1.2.3: Use Configuration Properties
**Files to modify:**
- `src/rag_sentiment_classifier/services/classification_service.py`

**Changes:**
```python
def __init__(self) -> None:
    self.llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,  # ADD THIS
        timeout=settings.llm_timeout,  # ADD THIS (new setting)
    )
    self._initialize_cache()

def _initialize_cache(self) -> None:
    # ...
    set_llm_cache(
        RedisCache(
            redis_client=redis_client,
            ttl=settings.redis_ttl  # ADD THIS
        )
    )
```

**Settings to add:**
```python
llm_timeout: int = 60  # seconds
request_timeout: int = 120  # Overall request timeout
```

**Acceptance Criteria:**
- [ ] ollama_max_tokens used in ChatOllama initialization
- [ ] redis_ttl used in cache configuration
- [ ] LLM timeout configured
- [ ] No unused settings remain

---

#### Task 1.2.4: Remove No-Op Validator
**Files to modify:**
- `src/rag_sentiment_classifier/models/document.py`

**Changes:**
```python
# Simply remove the entire validate_confidence method (lines 36-39)
# Pydantic Field validation (ge=0.0, le=1.0) is sufficient
```

**Acceptance Criteria:**
- [ ] validate_confidence method removed
- [ ] Field constraints still enforced by Pydantic
- [ ] Tests pass

---

## Phase 2: Testing Infrastructure (Weeks 3-4)

**Goal:** Achieve 80%+ test coverage with comprehensive test suite

**Priority:** 🔴 CRITICAL

### 2.1 Test Infrastructure Setup (Week 3, Days 1-2)

#### Task 2.1.1: Add Testing Dependencies
**Files to modify:**
- `pyproject.toml`

**Changes:**
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.3.2"
pytest-cov = "^5.0.0"
pytest-asyncio = "^0.23.0"
httpx = "^0.27.0"
pytest-mock = "^3.12.0"
testcontainers = "^4.0.0"
faker = "^22.0.0"
```

**Acceptance Criteria:**
- [ ] All test dependencies installed
- [ ] pytest-cov configured in pyproject.toml
- [ ] Coverage reports work

---

#### Task 2.1.2: Create Test Fixtures
**New file:** `tests/conftest.py`

**Changes:**
```python
import pytest
from faker import Faker
from rag_sentiment_classifier.models.document import DocumentInput, ClassificationResult
from rag_sentiment_classifier.config.settings import Settings

@pytest.fixture
def fake():
    return Faker()

@pytest.fixture
def sample_document(fake):
    return DocumentInput(
        content=fake.text(max_nb_chars=1000),
        document_id=f"DOC-{fake.random_int(1000, 9999)}",
        source="test",
        metadata={"test": True}
    )

@pytest.fixture
def sample_classification():
    return ClassificationResult(
        document_id="DOC-001",
        category="Regulatory",
        confidence=0.85,
        subcategories=["SEC", "Financial"],
        risk_level="medium",
        requires_review=False,
        reasoning="Test classification result"
    )

@pytest.fixture
def test_settings():
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1",
        redis_host="localhost",
        redis_port=6379,
        log_level="DEBUG"
    )

@pytest.fixture
def mock_llm(mocker):
    """Mock ChatOllama for testing without actual LLM calls"""
    return mocker.patch('rag_sentiment_classifier.services.classification_service.ChatOllama')
```

**Acceptance Criteria:**
- [ ] Reusable fixtures created
- [ ] Faker integration for test data
- [ ] Mock fixtures for external dependencies

---

### 2.2 Unit Tests (Week 3, Days 3-5)

#### Task 2.2.1: Service Unit Tests
**New file:** `tests/unit/test_classification_service.py`

**Tests to write:**
```python
# Test cases:
- test_service_initialization
- test_cache_initialization_success
- test_cache_initialization_failure_graceful
- test_classify_document_success
- test_classify_document_retry_logic
- test_classify_document_max_retries_exceeded
- test_classify_batch_all_success
- test_classify_batch_partial_failures
- test_classify_batch_empty_list
- test_llm_timeout_handling
```

**Coverage target:** 90%+ for classification_service.py

---

#### Task 2.2.2: API Endpoint Tests
**New file:** `tests/unit/test_api.py`

**Tests to write:**
```python
from fastapi.testclient import TestClient

# Test cases:
- test_health_check
- test_classify_valid_document
- test_classify_invalid_document
- test_classify_missing_fields
- test_classify_service_failure
- test_api_key_authentication
- test_rate_limiting
- test_error_responses
```

**Coverage target:** 100% for api.py

---

#### Task 2.2.3: Model Tests (Expand Existing)
**File to modify:** `tests/unit/test_models.py`

**Add tests:**
```python
# New test cases:
- test_document_input_metadata_validation
- test_document_input_id_pattern_validation
- test_classification_result_confidence_bounds
- test_classification_result_category_literal
- test_classification_result_risk_level_literal
- test_classification_result_serialization
```

**Coverage target:** 100% for models/document.py

---

#### Task 2.2.4: Settings Tests
**New file:** `tests/unit/test_settings.py`

**Tests to write:**
```python
# Test cases:
- test_settings_defaults
- test_settings_from_env_variables
- test_settings_caching
- test_settings_validation
```

**Coverage target:** 100% for settings.py

---

### 2.3 Integration Tests (Week 4)

#### Task 2.3.1: Service Integration Tests
**New file:** `tests/integration/test_classification_integration.py`

**Tests to write:**
```python
# Using testcontainers for Redis and Ollama
- test_classification_with_real_redis_cache
- test_classification_with_real_ollama (optional, slow)
- test_cache_hit_rate
- test_end_to_end_classification_flow
```

---

#### Task 2.3.2: API Integration Tests
**New file:** `tests/integration/test_api_integration.py`

**Tests to write:**
```python
# Full stack tests with Docker Compose
- test_api_with_all_services
- test_concurrent_requests
- test_batch_processing_performance
```

---

#### Task 2.3.3: Setup Test Automation
**New file:** `.github/workflows/test.yml`

**Changes:**
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run tests
        run: poetry run pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**Acceptance Criteria:**
- [ ] CI/CD pipeline configured
- [ ] Automated tests on every push
- [ ] Coverage reporting
- [ ] Test results visible in PR

---

## Phase 3: Architecture Refactoring (Weeks 5-6)

**Goal:** Implement proper dependency injection and clean architecture

**Priority:** 🟡 HIGH

### 3.1 Dependency Injection (Week 5)

#### Task 3.1.1: Create Service Interfaces
**New file:** `src/rag_sentiment_classifier/interfaces/llm_provider.py`

**Changes:**
```python
from abc import ABC, abstractmethod
from typing import Any

class LLMProvider(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def invoke(self, prompt: dict[str, Any]) -> Any:
        """Invoke the LLM with a prompt"""
        pass

    @abstractmethod
    def get_chain(self, prompt_template: Any, parser: Any) -> Any:
        """Create a processing chain"""
        pass

class CacheProvider(ABC):
    """Abstract interface for cache providers"""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the cache"""
        pass

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache"""
        pass
```

**Acceptance Criteria:**
- [ ] LLMProvider interface defined
- [ ] CacheProvider interface defined
- [ ] Clear contract for implementations

---

#### Task 3.1.2: Implement Concrete Providers
**New files:**
- `src/rag_sentiment_classifier/providers/ollama_provider.py`
- `src/rag_sentiment_classifier/providers/redis_cache_provider.py`

**Changes:**
```python
# ollama_provider.py
from rag_sentiment_classifier.interfaces.llm_provider import LLMProvider
from langchain_community.chat_models import ChatOllama

class OllamaLLMProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, timeout: int):
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

    def invoke(self, prompt: dict[str, Any]) -> Any:
        return self.llm.invoke(prompt)

    def get_chain(self, prompt_template: Any, parser: Any) -> Any:
        return prompt_template | self.llm | parser
```

**Acceptance Criteria:**
- [ ] OllamaLLMProvider implements interface
- [ ] RedisCacheProvider implements interface
- [ ] Easy to swap implementations

---

#### Task 3.1.3: Refactor Service with DI
**Files to modify:**
- `src/rag_sentiment_classifier/services/classification_service.py`

**Changes:**
```python
class DocumentClassificationService:
    """Document classification service with dependency injection."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        cache_provider: Optional[CacheProvider] = None,
        settings: Optional[Settings] = None
    ) -> None:
        self.llm_provider = llm_provider
        self.cache_provider = cache_provider
        self.settings = settings or get_settings()

        if self.cache_provider:
            self.cache_provider.initialize()

        self.classification_chain = self.llm_provider.get_chain(
            CLASSIFICATION_PROMPT,
            parser
        )

        logger.info("DocumentClassificationService initialized with DI")
```

**Acceptance Criteria:**
- [ ] Service accepts provider interfaces
- [ ] No direct instantiation of dependencies
- [ ] Easy to mock in tests
- [ ] Backward compatible with existing code

---

#### Task 3.1.4: Update API with Dependency Injection
**Files to modify:**
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
from functools import lru_cache

@lru_cache
def get_service() -> DocumentClassificationService:
    """Factory function for service creation"""
    settings = get_settings()

    llm_provider = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,
        timeout=settings.llm_timeout
    )

    cache_provider = RedisCacheProvider(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ssl=settings.redis_ssl,
        ttl=settings.redis_ttl
    )

    return DocumentClassificationService(
        llm_provider=llm_provider,
        cache_provider=cache_provider,
        settings=settings
    )

@app.post("/classify", response_model=ClassificationResult)
async def classify_document(
    document: DocumentInput,
    service: DocumentClassificationService = Depends(get_service)
) -> ClassificationResult:
    result = service.classify_document(document)
    if result is None:
        raise HTTPException(status_code=502, detail="Classification failed")
    return result
```

**Acceptance Criteria:**
- [ ] Service created via factory function
- [ ] FastAPI dependency injection used
- [ ] Request-scoped or cached appropriately
- [ ] CLI also uses factory pattern

---

### 3.2 Async Refactoring (Week 6)

#### Task 3.2.1: Make Service Methods Async
**Files to modify:**
- `src/rag_sentiment_classifier/services/classification_service.py`

**Changes:**
```python
import asyncio
from typing import Optional

async def classify_document(
    self,
    document: DocumentInput,
    retry_count: int = 0,
) -> Optional[ClassificationResult]:
    """Classify a single document with retry logic (async)."""
    try:
        logger.info("Classifying document %s", document.document_id)

        # Run LLM invocation in thread pool to avoid blocking
        result: ClassificationResult = await asyncio.to_thread(
            self.classification_chain.invoke,
            {
                "document_id": document.document_id,
                "content": document.content,
            }
        )
        result.document_id = document.document_id
        return result
    except Exception as exc:
        logger.error("Error classifying document %s: %s", document.document_id, exc, exc_info=True)
        if retry_count < self.settings.max_retries:
            wait_time = self.settings.retry_delay * (2 ** retry_count)
            logger.info("Retrying in %ss (attempt %s/%s)", wait_time, retry_count + 1, self.settings.max_retries)
            await asyncio.sleep(wait_time)
            return await self.classify_document(document, retry_count + 1)
        logger.error("All retries exhausted for document %s", document.document_id)
        return None

async def classify_batch(
    self,
    documents: list[DocumentInput],
    max_concurrent: int = 5
) -> list[ClassificationResult]:
    """Classify multiple documents concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def classify_with_semaphore(doc: DocumentInput) -> Optional[ClassificationResult]:
        async with semaphore:
            return await self.classify_document(doc)

    tasks = [classify_with_semaphore(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    return [r for r in results if r is not None]
```

**Acceptance Criteria:**
- [ ] classify_document is async
- [ ] classify_batch uses asyncio.gather for parallelism
- [ ] Semaphore limits concurrent LLM calls
- [ ] Proper error handling maintained

---

#### Task 3.2.2: Update API to Use Async Service
**Files to modify:**
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
@app.post("/classify", response_model=ClassificationResult)
async def classify_document(
    document: DocumentInput,
    service: DocumentClassificationService = Depends(get_service)
) -> ClassificationResult:
    result = await service.classify_document(document)  # Now actually async!
    if result is None:
        raise HTTPException(status_code=502, detail="Classification failed")
    return result

@app.post("/classify/batch", response_model=list[ClassificationResult])
async def classify_batch(
    documents: list[DocumentInput],
    service: DocumentClassificationService = Depends(get_service)
) -> list[ClassificationResult]:
    """New endpoint for batch classification"""
    results = await service.classify_batch(documents)
    return results
```

**Acceptance Criteria:**
- [ ] All API endpoints properly async
- [ ] Batch endpoint added
- [ ] Performance improved for concurrent requests

---

## Phase 4: DevOps & Infrastructure (Weeks 7-8)

**Goal:** Production-ready deployment and monitoring

**Priority:** 🟡 HIGH

### 4.1 Docker Improvements (Week 7, Days 1-2)

#### Task 4.1.1: Add .dockerignore
**New file:** `.dockerignore`

**Changes:**
```
.git
.gitignore
.env
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.pytest_cache
.coverage
htmlcov/
dist/
build/
*.egg-info
.vscode
.idea
*.md
!README.md
tests/
docs/
```

**Acceptance Criteria:**
- [ ] .dockerignore created
- [ ] Build context reduced
- [ ] Build time improved

---

#### Task 4.1.2: Multi-Stage Dockerfile
**Files to modify:**
- `Dockerfile`

**Changes:**
```dockerfile
# Build stage
FROM python:3.11-slim as builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* README.md /app/
RUN poetry install --only main --no-root

# Runtime stage
FROM python:3.11-slim as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY src /app/src

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "rag_sentiment_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Acceptance Criteria:**
- [ ] Multi-stage build reduces image size
- [ ] Non-root user for security
- [ ] Health check included
- [ ] Layer caching optimized

---

#### Task 4.1.3: Improve docker-compose.yml
**Files to modify:**
- `docker-compose.yml`

**Changes:**
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - REDIS_HOST=redis
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      redis:
        condition: service_healthy
      ollama:
        condition: service_started
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G

  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD:-changeme}
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M

volumes:
  ollama:
  redis:

networks:
  default:
    driver: bridge
```

**Acceptance Criteria:**
- [ ] Health checks configured
- [ ] Resource limits set
- [ ] Proper service dependencies
- [ ] Restart policies configured
- [ ] Redis persistence enabled

---

### 4.2 Logging & Monitoring (Week 7, Days 3-5)

#### Task 4.2.1: Structured Logging
**New dependency:** `python-json-logger`

**Files to modify:**
- `pyproject.toml`
- `src/rag_sentiment_classifier/config/logging_config.py` (new)
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
# logging_config.py
import logging
from pythonjsonlogger import jsonlogger

def setup_logging(log_level: str = "INFO"):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

# api.py - Add request logging middleware
from fastapi import Request
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        "Request processed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": duration,
            "client": request.client.host if request.client else None
        }
    )
    return response
```

**Acceptance Criteria:**
- [ ] JSON structured logs
- [ ] Request/response logging
- [ ] Performance metrics in logs
- [ ] Easy to parse for log aggregators

---

#### Task 4.2.2: Add Prometheus Metrics
**New dependency:** `prometheus-fastapi-instrumentator`

**Files to modify:**
- `pyproject.toml`
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="RAG Sentiment Classifier", version="0.1.0")

# Initialize instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
)

instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Add custom metrics
from prometheus_client import Counter, Histogram

classification_counter = Counter(
    'classification_requests_total',
    'Total classification requests',
    ['status']
)

classification_duration = Histogram(
    'classification_duration_seconds',
    'Time spent classifying documents'
)
```

**Acceptance Criteria:**
- [ ] /metrics endpoint available
- [ ] Standard HTTP metrics collected
- [ ] Custom business metrics added
- [ ] Ready for Prometheus scraping

---

#### Task 4.2.3: Add Application Health Checks
**Files to modify:**
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
from typing import Dict, Any

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check - verify all dependencies"""
    checks = {}
    overall_status = "ok"

    # Check Redis
    try:
        # Attempt Redis ping via service
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"
        overall_status = "degraded"

    # Check Ollama
    try:
        # Attempt Ollama health check
        checks["ollama"] = "ok"
    except Exception as e:
        checks["ollama"] = f"error: {str(e)}"
        overall_status = "error"

    status_code = 200 if overall_status == "ok" else (503 if overall_status == "error" else 200)
    return {"status": overall_status, "checks": checks}

@app.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """Liveness check - is the app running"""
    return {"status": "ok"}
```

**Acceptance Criteria:**
- [ ] /health for basic status
- [ ] /health/ready for Kubernetes readiness
- [ ] /health/live for Kubernetes liveness
- [ ] Dependency status checks

---

### 4.3 CI/CD Pipeline (Week 8)

#### Task 4.3.1: GitHub Actions Workflow
**New files:**
- `.github/workflows/ci.yml`
- `.github/workflows/docker-publish.yml`

**Changes:**
```yaml
# ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run ruff
        run: poetry run ruff check .
      - name: Run mypy
        run: poetry run mypy src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run tests
        run: poetry run pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Run safety check
        run: |
          pip install safety
          safety check --json

  docker:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t rag-sentiment-classifier:${{ github.sha }} .
      - name: Test Docker image
        run: |
          docker run -d -p 8000:8000 --name test-app rag-sentiment-classifier:${{ github.sha }}
          sleep 10
          curl -f http://localhost:8000/health || exit 1
```

**Acceptance Criteria:**
- [ ] Automated linting
- [ ] Automated testing on multiple Python versions
- [ ] Security scanning
- [ ] Docker build verification
- [ ] Coverage reporting

---

## Phase 5: Performance & Optimization (Weeks 9-10)

**Goal:** Optimize performance and scalability

**Priority:** 🟢 MEDIUM

### 5.1 Performance Improvements (Week 9)

#### Task 5.1.1: Add Connection Pooling
**Files to modify:**
- `src/rag_sentiment_classifier/providers/redis_cache_provider.py`

**Changes:**
```python
from redis.connection import ConnectionPool

class RedisCacheProvider(CacheProvider):
    _pool: Optional[ConnectionPool] = None

    def __init__(self, ...):
        if RedisCacheProvider._pool is None:
            RedisCacheProvider._pool = ConnectionPool(
                host=host,
                port=port,
                password=password,
                ssl=ssl,
                max_connections=10,
                decode_responses=True
            )
        self.redis_client = redis.Redis(connection_pool=RedisCacheProvider._pool)
```

**Acceptance Criteria:**
- [ ] Redis connection pooling
- [ ] Connection reuse
- [ ] Configurable pool size

---

#### Task 5.1.2: Implement Request Timeouts
**Files to modify:**
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
import asyncio

@app.post("/classify")
async def classify_document(document: DocumentInput, ...):
    try:
        result = await asyncio.wait_for(
            service.classify_document(document),
            timeout=settings.request_timeout
        )
        if result is None:
            raise HTTPException(status_code=502, detail="Classification failed")
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
```

**Acceptance Criteria:**
- [ ] Global request timeout
- [ ] 504 Gateway Timeout response
- [ ] Configurable timeout value

---

#### Task 5.1.3: Add Caching Headers
**Files to modify:**
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
from fastapi import Response

@app.post("/classify")
async def classify_document(
    document: DocumentInput,
    response: Response,
    ...
):
    result = await service.classify_document(document)

    # Add cache control headers
    response.headers["Cache-Control"] = "private, max-age=3600"
    response.headers["X-Request-ID"] = str(uuid.uuid4())

    return result
```

**Acceptance Criteria:**
- [ ] Cache-Control headers
- [ ] Request ID tracking
- [ ] Proper HTTP semantics

---

### 5.2 Load Testing (Week 10)

#### Task 5.2.1: Set Up Load Testing
**New dependency:** `locust`

**New file:** `tests/load/locustfile.py`

**Changes:**
```python
from locust import HttpUser, task, between

class ClassificationUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def classify_document(self):
        self.client.post(
            "/classify",
            json={
                "document_id": f"DOC-{self.user_number}",
                "source": "load-test",
                "content": "Test regulatory document for classification..."
            },
            headers={"X-API-Key": "test-key"}
        )

    @task(3)
    def health_check(self):
        self.client.get("/health")
```

**Acceptance Criteria:**
- [ ] Load testing configured
- [ ] Baseline performance established
- [ ] Bottlenecks identified

---

#### Task 5.2.2: Performance Benchmarking
**Document to create:** `PERFORMANCE.md`

**Tests to run:**
- Single request latency (P50, P95, P99)
- Concurrent requests throughput
- Batch processing performance
- Cache hit rate analysis
- Resource utilization under load

**Acceptance Criteria:**
- [ ] Performance baseline documented
- [ ] Optimization targets set
- [ ] Regular performance regression testing

---

## Phase 6: Documentation & Polish (Week 11-12)

**Goal:** Complete documentation and final polish

**Priority:** 🟢 MEDIUM

### 6.1 Code Documentation (Week 11)

#### Task 6.1.1: Add Comprehensive Docstrings
**Files to modify:** All Python files

**Standard format:**
```python
def classify_document(
    self,
    document: DocumentInput,
    retry_count: int = 0,
) -> Optional[ClassificationResult]:
    """
    Classify a document using the configured LLM provider.

    This method sends the document content to the LLM for classification
    into predefined categories with confidence scoring and risk assessment.
    Implements exponential backoff retry logic for transient failures.

    Args:
        document: The document to classify, containing content and metadata
        retry_count: Internal counter for retry attempts (default: 0)

    Returns:
        ClassificationResult with category, confidence, risk level, and reasoning
        Returns None if classification fails after all retries

    Raises:
        None - errors are caught and logged, None returned on failure

    Examples:
        >>> doc = DocumentInput(
        ...     content="SEC filing...",
        ...     document_id="DOC-001",
        ...     source="api"
        ... )
        >>> result = await service.classify_document(doc)
        >>> print(result.category)
        'Regulatory'
    """
```

**Acceptance Criteria:**
- [ ] All public methods documented
- [ ] Parameters described
- [ ] Return values explained
- [ ] Examples provided where helpful

---

#### Task 6.1.2: API Documentation
**Files to modify:**
- `src/rag_sentiment_classifier/api.py`

**Changes:**
```python
@app.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Classify a single document",
    description="""
    Classify a regulatory or compliance document into predefined categories.

    The classification includes:
    - Category (Regulatory, Compliance, Risk, Operational, Other)
    - Confidence score (0.0 to 1.0)
    - Risk level assessment
    - Subcategories
    - Reasoning explanation

    Documents with confidence < 0.7 are automatically flagged for review.
    """,
    response_description="Classification result with category, confidence, and metadata",
    responses={
        200: {
            "description": "Successful classification",
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "DOC-001",
                        "category": "Regulatory",
                        "confidence": 0.92,
                        "subcategories": ["SEC", "Financial Reporting"],
                        "risk_level": "medium",
                        "requires_review": False,
                        "reasoning": "Document contains SEC filing requirements...",
                        "processed_at": "2026-01-10T12:00:00Z"
                    }
                }
            }
        },
        403: {"description": "Invalid API key"},
        429: {"description": "Rate limit exceeded"},
        502: {"description": "Classification service unavailable"},
        504: {"description": "Request timeout"}
    }
)
async def classify_document(...):
```

**Acceptance Criteria:**
- [ ] OpenAPI documentation enhanced
- [ ] Example requests/responses
- [ ] Error responses documented
- [ ] /docs endpoint user-friendly

---

### 6.2 Operations Documentation (Week 12)

#### Task 6.2.1: Deployment Guide
**New file:** `docs/DEPLOYMENT.md`

**Sections:**
- Prerequisites
- Environment setup
- Docker deployment
- Kubernetes deployment
- Configuration management
- SSL/TLS setup
- Scaling guidelines
- Troubleshooting

---

#### Task 6.2.2: Operations Runbook
**New file:** `docs/RUNBOOK.md`

**Sections:**
- Service architecture
- Monitoring and alerts
- Common issues and solutions
- Incident response procedures
- Backup and recovery
- Performance tuning
- Security checklist

---

#### Task 6.2.3: Development Guide
**New file:** `docs/DEVELOPMENT.md`

**Sections:**
- Local development setup
- Running tests
- Code style and linting
- Contributing guidelines
- PR process
- Release process

---

## Success Metrics

### Coverage Targets
- [ ] Unit test coverage: **80%+**
- [ ] Integration test coverage: **70%+**
- [ ] API endpoint coverage: **100%**

### Performance Targets
- [ ] P95 latency: **< 2 seconds** (single document)
- [ ] Throughput: **100+ requests/minute** (with caching)
- [ ] Batch processing: **10+ documents/minute**

### Security Checklist
- [ ] Authentication implemented
- [ ] Rate limiting active
- [ ] Input validation comprehensive
- [ ] TLS/SSL configured
- [ ] Security headers present
- [ ] Dependency scanning automated
- [ ] No secrets in code/config

### Production Readiness
- [ ] Zero critical issues
- [ ] All tests passing
- [ ] CI/CD pipeline complete
- [ ] Monitoring and alerts configured
- [ ] Documentation complete
- [ ] Load testing passed
- [ ] Security review completed

---

## Timeline Summary

| Phase | Focus | Duration | Priority |
|-------|-------|----------|----------|
| 1 | Security & Cleanup | 2 weeks | 🔴 Critical |
| 2 | Testing | 2 weeks | 🔴 Critical |
| 3 | Architecture | 2 weeks | 🟡 High |
| 4 | DevOps | 2 weeks | 🟡 High |
| 5 | Performance | 2 weeks | 🟢 Medium |
| 6 | Documentation | 2 weeks | 🟢 Medium |

**Total Duration:** 12 weeks

---

## Risk Management

### High Risks
1. **LLM Availability** - Ollama service stability
   - Mitigation: Implement circuit breaker, fallback mechanisms

2. **Performance Under Load** - LLM latency increases with load
   - Mitigation: Implement queuing, rate limiting, caching

3. **Breaking Changes** - LangChain frequent updates
   - Mitigation: Pin versions, thorough testing before upgrades

### Medium Risks
1. **Redis Dependency** - Cache failures affecting performance
   - Mitigation: Graceful degradation, monitoring

2. **Resource Constraints** - High memory usage
   - Mitigation: Resource limits, monitoring, alerts

---

## Rollout Strategy

### Stage 1: Internal Testing (Week 13)
- Deploy to staging environment
- Internal team testing
- Performance validation
- Security audit

### Stage 2: Limited Beta (Week 14-15)
- Deploy to small user group
- Monitor metrics closely
- Gather feedback
- Iterate on issues

### Stage 3: Production (Week 16)
- Full production deployment
- Gradual traffic ramp-up
- 24/7 monitoring
- On-call rotation

---

## Maintenance Plan

### Weekly
- Review monitoring dashboards
- Check error rates
- Review security alerts
- Update dependencies (patch versions)

### Monthly
- Performance review
- Security scan
- Dependency updates (minor versions)
- Documentation updates

### Quarterly
- Architecture review
- Major dependency updates
- Load testing
- Disaster recovery drill

---

**End of Refactor Plan**

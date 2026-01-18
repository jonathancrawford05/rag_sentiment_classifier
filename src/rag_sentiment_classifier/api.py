"""FastAPI application with dependency injection and async support."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag_sentiment_classifier.config.settings import get_settings
from rag_sentiment_classifier.models.document import (
    ClassificationResult,
    DocumentInput,
)
from rag_sentiment_classifier.providers.ollama_provider import OllamaLLMProvider
from rag_sentiment_classifier.providers.redis_provider import RedisCacheProvider
from rag_sentiment_classifier.services.classification_service import (
    DocumentClassificationService,
)

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Global cache provider for lifecycle management
cache_provider: RedisCacheProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler for startup and shutdown.

    Initializes cache provider on startup and cleans up on shutdown.
    """
    global cache_provider

    # Startup: Initialize cache provider
    try:
        cache_provider = RedisCacheProvider(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            ttl=settings.redis_ttl,
        )
        # Test connection
        if await cache_provider.ping():
            logger.info("Redis cache initialized successfully")
        else:
            logger.warning("Redis not accessible, proceeding without cache")
            cache_provider = None
    except Exception as exc:
        logger.warning("Cache initialization failed: %s. Proceeding without cache.", exc)
        cache_provider = None

    yield

    # Shutdown: Close cache connection
    if cache_provider:
        await cache_provider.close()
        logger.info("Cache connection closed")


# Initialize FastAPI app with lifespan
app = FastAPI(title="RAG Sentiment Classifier", version="0.2.0", lifespan=lifespan)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_valid_api_keys() -> list[str]:
    """Get list of valid API keys from settings."""
    keys = []
    if settings.api_key:
        keys.append(settings.api_key)
    if settings.api_keys:
        # Parse comma-separated list
        keys.extend([k.strip() for k in settings.api_keys.split(",") if k.strip()])
    return keys


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: 403 if API key is invalid or missing
    """
    valid_keys = get_valid_api_keys()

    # If no API keys configured, allow access (for backward compatibility in dev)
    if not valid_keys:
        logger.warning("No API keys configured - running without authentication!")
        return "no-key-configured"

    if not api_key or api_key not in valid_keys:
        logger.warning("Invalid API key attempt from request")
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Provide a valid X-API-Key header.",
        )

    return api_key


def get_classification_service() -> DocumentClassificationService:
    """
    Dependency injection for DocumentClassificationService.

    Creates service with injected LLM and cache providers.

    Returns:
        Configured DocumentClassificationService instance
    """
    # Create LLM provider
    llm_provider = OllamaLLMProvider(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
        max_tokens=settings.ollama_max_tokens,
        timeout=settings.llama_timeout,
    )

    # Create service with providers
    service = DocumentClassificationService(
        llm_provider=llm_provider,
        cache_provider=cache_provider,
        max_retries=settings.max_retries,
        retry_delay=settings.retry_delay,
        max_concurrent=5,  # Can be made configurable
    )

    return service


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint (no authentication required).

    Returns:
        Status dictionary indicating service health
    """
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationResult)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def classify_document(
    request: Request,
    document: DocumentInput,
    api_key: str = Depends(verify_api_key),
    service: DocumentClassificationService = Depends(get_classification_service),
) -> ClassificationResult:
    """
    Classify a single document asynchronously.

    Requires valid API key in X-API-Key header.
    Rate limited to configured requests per minute.

    Args:
        request: FastAPI request (for rate limiting)
        document: Document to classify
        api_key: Validated API key from header
        service: Injected classification service

    Returns:
        Classification result with category, confidence, and risk assessment

    Raises:
        HTTPException: 403 if invalid API key, 429 if rate limit exceeded,
                      502 if classification fails
    """
    logger.info("Classifying document %s", document.document_id)
    result = await service.classify_document(document)
    if result is None:
        logger.error("Classification failed for document %s", document.document_id)
        raise HTTPException(status_code=502, detail="Classification failed")
    return result


@app.post("/classify/batch", response_model=list[ClassificationResult])
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def classify_batch(
    request: Request,
    documents: list[DocumentInput],
    api_key: str = Depends(verify_api_key),
    service: DocumentClassificationService = Depends(get_classification_service),
) -> list[ClassificationResult]:
    """
    Classify multiple documents concurrently.

    Requires valid API key in X-API-Key header.
    Rate limited to configured requests per minute.

    Args:
        request: FastAPI request (for rate limiting)
        documents: List of documents to classify
        api_key: Validated API key from header
        service: Injected classification service

    Returns:
        List of classification results (failed classifications omitted)

    Raises:
        HTTPException: 403 if invalid API key, 429 if rate limit exceeded
    """
    logger.info("Classifying batch of %d documents", len(documents))
    results = await service.classify_batch(documents)
    logger.info(
        "Batch classification complete: %d/%d successful",
        len(results),
        len(documents),
    )
    return results

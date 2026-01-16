import logging

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
from rag_sentiment_classifier.services.classification_service import (
    DocumentClassificationService,
)

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Sentiment Classifier", version="0.1.0")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize service
service = DocumentClassificationService()

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
) -> ClassificationResult:
    """
    Classify a single document.

    Requires valid API key in X-API-Key header.
    Rate limited to configured requests per minute.

    Args:
        document: Document to classify
        api_key: Validated API key from header

    Returns:
        Classification result with category, confidence, and risk assessment

    Raises:
        HTTPException: 403 if invalid API key, 429 if rate limit exceeded,
                      502 if classification fails
    """
    logger.info("Classifying document %s", document.document_id)
    result = service.classify_document(document)
    if result is None:
        logger.error("Classification failed for document %s", document.document_id)
        raise HTTPException(status_code=502, detail="Classification failed")
    return result

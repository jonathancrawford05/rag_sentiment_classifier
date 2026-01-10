import logging

from fastapi import FastAPI, HTTPException

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

app = FastAPI(title="RAG Sentiment Classifier", version="0.1.0")
service = DocumentClassificationService()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationResult)
async def classify_document(document: DocumentInput) -> ClassificationResult:
    result = service.classify_document(document)
    if result is None:
        raise HTTPException(status_code=502, detail="Classification failed")
    return result

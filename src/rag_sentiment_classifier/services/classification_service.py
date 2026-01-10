import logging
import time
from typing import Optional

import redis
from langchain_community.cache import RedisCache
from langchain_community.chat_models import ChatOllama
from langchain_core.globals import set_llm_cache

from rag_sentiment_classifier.config.settings import get_settings
from rag_sentiment_classifier.models.document import (
    ClassificationResult,
    DocumentInput,
)
from rag_sentiment_classifier.prompts.classification_prompts import (
    CLASSIFICATION_PROMPT,
    parser,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentClassificationService:
    """Document classification service using a local Ollama LLM."""

    def __init__(self) -> None:
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            num_predict=settings.ollama_max_tokens,
            timeout=settings.llama_timeout,
        )
        self._initialize_cache()

        self.classification_chain = (
            CLASSIFICATION_PROMPT
            | self.llm
            | parser
        )

        logger.info("DocumentClassificationService initialized")

    def _initialize_cache(self) -> None:
        """Initialize Redis cache with secure connection settings."""
        try:
            # Build Redis connection parameters
            redis_params = {
                "host": settings.redis_host,
                "port": settings.redis_port,
                "decode_responses": True,
            }

            # Add password if configured
            if settings.redis_password:
                redis_params["password"] = settings.redis_password

            # Add SSL if configured
            if settings.redis_ssl:
                redis_params["ssl"] = True
                redis_params["ssl_cert_reqs"] = None  # For self-signed certs in dev

            redis_client = redis.Redis(**redis_params)
            redis_client.ping()
            set_llm_cache(RedisCache(redis_client=redis_client, ttl=settings.redis_ttl))
            logger.info("Redis cache initialized successfully with secure connection")
        except Exception as exc:
            logger.warning(
                "Redis cache initialization failed: %s. Proceeding without cache.",
                exc,
            )

    def classify_document(
        self,
        document: DocumentInput,
        retry_count: int = 0,
    ) -> Optional[ClassificationResult]:
        """Classify a single document with retry logic."""
        try:
            logger.info("Classifying document %s", document.document_id)
            result: ClassificationResult = self.classification_chain.invoke(
                {
                    "document_id": document.document_id,
                    "content": document.content,
                }
            )
            result.document_id = document.document_id
            return result
        except Exception as exc:
            logger.error(
                "Error classifying document %s: %s",
                document.document_id,
                exc,
                exc_info=True,
            )
            if retry_count < settings.max_retries:
                wait_time = settings.retry_delay * (2 ** retry_count)
                logger.info(
                    "Retrying in %ss (attempt %s/%s)",
                    wait_time,
                    retry_count + 1,
                    settings.max_retries,
                )
                time.sleep(wait_time)
                return self.classify_document(document, retry_count + 1)
            logger.error("All retries exhausted for document %s", document.document_id)
            return None

    def classify_batch(self, documents: list[DocumentInput]) -> list[ClassificationResult]:
        """Classify multiple documents sequentially."""
        results: list[ClassificationResult] = []
        for doc in documents:
            result = self.classify_document(doc)
            if result:
                results.append(result)
        return results

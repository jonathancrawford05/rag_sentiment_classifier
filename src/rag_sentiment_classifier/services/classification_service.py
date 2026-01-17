"""Document classification service with dependency injection and async support."""

import asyncio
import hashlib
import logging
from typing import Any

from rag_sentiment_classifier.models.document import (
    ClassificationResult,
    DocumentInput,
)
from rag_sentiment_classifier.providers.cache_provider import CacheProvider
from rag_sentiment_classifier.providers.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class DocumentClassificationService:
    """
    Document classification service using dependency injection.

    This service uses injected LLM and cache providers, making it testable
    and flexible. Supports async operations and concurrent batch processing.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        cache_provider: CacheProvider | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent: int = 5,
    ) -> None:
        """
        Initialize classification service with injected dependencies.

        Args:
            llm_provider: LLM provider for classification
            cache_provider: Optional cache provider for results
            max_retries: Maximum retry attempts for failed classifications
            retry_delay: Initial delay between retries (exponential backoff)
            max_concurrent: Maximum concurrent classifications in batch
        """
        self.llm_provider = llm_provider
        self.cache_provider = cache_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max_concurrent
        logger.info(
            "DocumentClassificationService initialized with max_retries=%d, max_concurrent=%d",
            max_retries,
            max_concurrent,
        )

    def _get_cache_key(self, document_id: str, content: str) -> str:
        """
        Generate a cache key for a document.

        Args:
            document_id: Document identifier
            content: Document content

        Returns:
            Cache key (hash of document_id and content)
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"classify:{document_id}:{content_hash}"

    async def _get_from_cache(
        self, document_id: str, content: str
    ) -> ClassificationResult | None:
        """
        Try to get classification result from cache.

        Args:
            document_id: Document identifier
            content: Document content

        Returns:
            Cached result if found, None otherwise
        """
        if not self.cache_provider:
            return None

        try:
            cache_key = self._get_cache_key(document_id, content)
            cached_data: dict[str, Any] | None = await self.cache_provider.get(cache_key)
            if cached_data:
                logger.info("Cache hit for document %s", document_id)
                return ClassificationResult(**cached_data)
        except Exception as exc:
            logger.warning("Cache retrieval failed for document %s: %s", document_id, exc)

        return None

    async def _save_to_cache(
        self, document_id: str, content: str, result: ClassificationResult
    ) -> None:
        """
        Save classification result to cache.

        Args:
            document_id: Document identifier
            content: Document content
            result: Classification result to cache
        """
        if not self.cache_provider:
            return

        try:
            cache_key = self._get_cache_key(document_id, content)
            # Convert Pydantic model to dict for caching
            await self.cache_provider.set(cache_key, result.model_dump())
            logger.debug("Cached result for document %s", document_id)
        except Exception as exc:
            logger.warning("Cache save failed for document %s: %s", document_id, exc)

    async def classify_document(
        self,
        document: DocumentInput,
        retry_count: int = 0,
    ) -> ClassificationResult | None:
        """
        Classify a single document with retry logic and caching.

        Args:
            document: Document to classify
            retry_count: Current retry attempt (for internal use)

        Returns:
            Classification result or None if all retries exhausted
        """
        logger.info("Classifying document %s", document.document_id)

        # Try cache first
        cached_result = await self._get_from_cache(document.document_id, document.content)
        if cached_result:
            return cached_result

        # Classify with LLM
        try:
            result = await self.llm_provider.classify(
                document.document_id, document.content
            )

            # Save to cache
            await self._save_to_cache(document.document_id, document.content, result)

            logger.info(
                "Document %s classified as %s (confidence: %.2f)",
                document.document_id,
                result.category,
                result.confidence,
            )
            return result

        except Exception as exc:
            logger.error(
                "Error classifying document %s: %s",
                document.document_id,
                exc,
                exc_info=True,
            )

            # Retry logic with exponential backoff
            if retry_count < self.max_retries:
                wait_time = self.retry_delay * (2**retry_count)
                logger.info(
                    "Retrying document %s in %.1fs (attempt %d/%d)",
                    document.document_id,
                    wait_time,
                    retry_count + 1,
                    self.max_retries,
                )
                await asyncio.sleep(wait_time)
                return await self.classify_document(document, retry_count + 1)

            logger.error(
                "All retries exhausted for document %s", document.document_id
            )
            return None

    async def classify_batch(
        self, documents: list[DocumentInput]
    ) -> list[ClassificationResult]:
        """
        Classify multiple documents concurrently with controlled parallelism.

        Uses asyncio.Semaphore to limit concurrent classifications and avoid
        overwhelming the LLM service.

        Args:
            documents: List of documents to classify

        Returns:
            List of successful classification results (failed docs omitted)
        """
        if not documents:
            return []

        logger.info(
            "Classifying batch of %d documents with max_concurrent=%d",
            len(documents),
            self.max_concurrent,
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def classify_with_semaphore(doc: DocumentInput) -> ClassificationResult | None:
            """Classify a document with semaphore-controlled concurrency."""
            async with semaphore:
                return await self.classify_document(doc)

        # Create tasks for all documents
        tasks = [classify_with_semaphore(doc) for doc in documents]

        # Run concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out None values (failed classifications)
        successful_results = [r for r in results if r is not None]

        logger.info(
            "Batch classification complete: %d/%d successful",
            len(successful_results),
            len(documents),
        )

        return successful_results

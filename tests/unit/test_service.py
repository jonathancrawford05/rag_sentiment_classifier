"""Tests for DocumentClassificationService with dependency injection."""

from unittest.mock import AsyncMock

import pytest

from rag_sentiment_classifier.models.document import (
    ClassificationResult,
    DocumentInput,
)
from rag_sentiment_classifier.services.classification_service import (
    DocumentClassificationService,
)


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return DocumentInput(
        document_id="test-123",
        content="This is a regulatory compliance document about banking regulations.",
        source="test-system",
    )


@pytest.fixture
def sample_result():
    """Create a sample classification result."""
    return ClassificationResult(
        document_id="test-123",
        category="Regulatory",
        confidence=0.95,
        subcategories=["Banking", "Compliance"],
        risk_level="high",
        requires_review=True,
        reasoning="Document discusses banking regulations which require compliance review.",
    )


@pytest.fixture
def mock_llm_provider(sample_result):
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.classify = AsyncMock(return_value=sample_result)
    return provider


@pytest.fixture
def mock_cache_provider():
    """Create a mock cache provider."""
    provider = AsyncMock()
    provider.get = AsyncMock(return_value=None)
    provider.set = AsyncMock()
    provider.delete = AsyncMock()
    provider.ping = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def service_with_cache(mock_llm_provider, mock_cache_provider):
    """Create service with mocked providers."""
    return DocumentClassificationService(
        llm_provider=mock_llm_provider,
        cache_provider=mock_cache_provider,
        max_retries=3,
        retry_delay=0.1,  # Faster for tests
        max_concurrent=5,
    )


@pytest.fixture
def service_without_cache(mock_llm_provider):
    """Create service without cache provider."""
    return DocumentClassificationService(
        llm_provider=mock_llm_provider,
        cache_provider=None,
        max_retries=3,
        retry_delay=0.1,
    )


class TestDocumentClassificationService:
    """Tests for DocumentClassificationService."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_llm_provider):
        """Test service initializes correctly."""
        service = DocumentClassificationService(
            llm_provider=mock_llm_provider,
            max_retries=5,
            retry_delay=2.0,
            max_concurrent=10,
        )

        assert service.llm_provider == mock_llm_provider
        assert service.cache_provider is None
        assert service.max_retries == 5
        assert service.retry_delay == 2.0
        assert service.max_concurrent == 10

    @pytest.mark.asyncio
    async def test_classify_document_success(
        self, service_with_cache, sample_document, sample_result
    ):
        """Test successful document classification."""
        result = await service_with_cache.classify_document(sample_document)

        assert result is not None
        assert result.document_id == "test-123"
        assert result.category == "Regulatory"
        assert result.confidence == 0.95

        # Verify LLM was called
        service_with_cache.llm_provider.classify.assert_called_once()

        # Verify result was cached
        service_with_cache.cache_provider.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_document_cache_hit(
        self, service_with_cache, sample_document, sample_result
    ):
        """Test classification with cache hit."""
        # Set up cache to return result
        service_with_cache.cache_provider.get = AsyncMock(return_value=sample_result.model_dump())

        result = await service_with_cache.classify_document(sample_document)

        assert result is not None
        assert result.document_id == "test-123"

        # Verify LLM was NOT called (cache hit)
        service_with_cache.llm_provider.classify.assert_not_called()

        # Verify cache was checked
        service_with_cache.cache_provider.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_document_without_cache(self, service_without_cache, sample_document):
        """Test classification without cache provider."""
        result = await service_without_cache.classify_document(sample_document)

        assert result is not None
        assert result.document_id == "test-123"

        # Verify LLM was called
        service_without_cache.llm_provider.classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_document_llm_failure(self, service_with_cache, sample_document):
        """Test handling of LLM failures."""
        # Make LLM fail
        service_with_cache.llm_provider.classify = AsyncMock(side_effect=Exception("LLM error"))

        result = await service_with_cache.classify_document(sample_document)

        # Should return None after retries exhausted
        assert result is None

        # Verify retries occurred (1 initial + 3 retries = 4 total)
        assert service_with_cache.llm_provider.classify.call_count == 4

    @pytest.mark.asyncio
    async def test_classify_document_retry_success(
        self, service_with_cache, sample_document, sample_result
    ):
        """Test successful retry after initial failure."""
        # Fail twice, then succeed
        service_with_cache.llm_provider.classify = AsyncMock(
            side_effect=[
                Exception("Temporary error"),
                Exception("Another error"),
                sample_result,
            ]
        )

        result = await service_with_cache.classify_document(sample_document)

        assert result is not None
        assert result.document_id == "test-123"

        # Verify 3 attempts were made
        assert service_with_cache.llm_provider.classify.call_count == 3

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, service_with_cache):
        """Test cache key generation is consistent."""
        key1 = service_with_cache._get_cache_key("doc-1", "content")
        key2 = service_with_cache._get_cache_key("doc-1", "content")
        key3 = service_with_cache._get_cache_key("doc-1", "different")

        # Same inputs produce same key
        assert key1 == key2

        # Different content produces different key
        assert key1 != key3

        # Key includes document ID
        assert "doc-1" in key1

    @pytest.mark.asyncio
    async def test_classify_batch_success(self, service_with_cache, sample_result):
        """Test successful batch classification."""
        documents = [
            DocumentInput(
                document_id=f"doc-{i}",
                content=f"Document {i} content",
                source="test",
            )
            for i in range(5)
        ]

        results = await service_with_cache.classify_batch(documents)

        assert len(results) == 5
        # Verify all documents were classified
        assert service_with_cache.llm_provider.classify.call_count == 5

    @pytest.mark.asyncio
    async def test_classify_batch_partial_failure(self, service_with_cache, sample_result):
        """Test batch classification with some failures."""
        documents = [
            DocumentInput(
                document_id=f"doc-{i}",
                content=f"Document {i} content",
                source="test",
            )
            for i in range(5)
        ]

        # Make some classifications fail
        service_with_cache.llm_provider.classify = AsyncMock(
            side_effect=[
                sample_result,  # Success
                Exception("Error"),  # Fail
                sample_result,  # Success
                Exception("Error"),  # Fail
                sample_result,  # Success
            ]
        )

        # Set max_retries to 0 to avoid retry logic in this test
        service_with_cache.max_retries = 0

        results = await service_with_cache.classify_batch(documents)

        # Should return only successful results
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_classify_batch_empty(self, service_with_cache):
        """Test batch classification with empty list."""
        results = await service_with_cache.classify_batch([])

        assert results == []
        service_with_cache.llm_provider.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_batch_concurrency_control(self, service_with_cache, sample_result):
        """Test that batch classification respects concurrency limits."""
        # Create more documents than max_concurrent
        documents = [
            DocumentInput(
                document_id=f"doc-{i}",
                content=f"Document {i} content",
                source="test",
            )
            for i in range(10)
        ]

        service_with_cache.max_concurrent = 3

        results = await service_with_cache.classify_batch(documents)

        # All should succeed
        assert len(results) == 10

        # Verify all were processed
        assert service_with_cache.llm_provider.classify.call_count == 10

    @pytest.mark.asyncio
    async def test_cache_save_failure_handling(self, service_with_cache, sample_document):
        """Test that cache save failures don't break classification."""
        # Make cache save fail
        service_with_cache.cache_provider.set = AsyncMock(side_effect=Exception("Cache error"))

        result = await service_with_cache.classify_document(sample_document)

        # Classification should still succeed
        assert result is not None
        assert result.document_id == "test-123"

    @pytest.mark.asyncio
    async def test_cache_get_failure_handling(self, service_with_cache, sample_document):
        """Test that cache get failures don't break classification."""
        # Make cache get fail
        service_with_cache.cache_provider.get = AsyncMock(side_effect=Exception("Cache error"))

        result = await service_with_cache.classify_document(sample_document)

        # Should fall back to LLM
        assert result is not None
        service_with_cache.llm_provider.classify.assert_called_once()

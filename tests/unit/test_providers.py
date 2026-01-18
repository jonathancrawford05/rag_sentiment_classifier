"""Tests for provider implementations."""

from unittest.mock import AsyncMock, patch

import pytest

from rag_sentiment_classifier.providers.ollama_provider import OllamaLLMProvider
from rag_sentiment_classifier.providers.redis_provider import RedisCacheProvider


class TestOllamaLLMProvider:
    """Tests for OllamaLLMProvider."""

    @pytest.fixture
    def provider(self):
        """Create an Ollama provider instance."""
        return OllamaLLMProvider(
            model="llama2",
            base_url="http://localhost:11434",
            temperature=0.0,
            max_tokens=500,
            timeout=60,
        )

    @pytest.mark.asyncio
    async def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.llm is not None
        assert provider.classification_chain is not None

    @pytest.mark.skip(
        reason="LangChain RunnableSequence doesn't allow mocking, tested via service tests"
    )
    @pytest.mark.asyncio
    async def test_classify_success(self, provider):
        """Test successful classification."""
        pass

    @pytest.mark.skip(
        reason="LangChain RunnableSequence doesn't allow mocking, tested via service tests"
    )
    @pytest.mark.asyncio
    async def test_classify_failure(self, provider):
        """Test classification failure handling."""
        pass


class TestRedisCacheProvider:
    """Tests for RedisCacheProvider."""

    @pytest.fixture
    async def provider(self):
        """Create a Redis cache provider instance."""
        with patch("rag_sentiment_classifier.providers.redis_provider.redis.Redis"):
            provider = RedisCacheProvider(
                host="localhost",
                port=6380,
                password=None,
                ssl=False,
                ttl=3600,
            )
            # Mock the client
            provider.client = AsyncMock()
            yield provider
            # Cleanup
            await provider.close()

    @pytest.mark.asyncio
    async def test_provider_initialization(self):
        """Test provider initializes with correct parameters."""
        with patch("rag_sentiment_classifier.providers.redis_provider.redis.Redis"):
            provider = RedisCacheProvider(
                host="localhost",
                port=6380,
                password="secret",
                ssl=True,
                ttl=7200,
            )
            assert provider.default_ttl == 7200

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, provider):
        """Test getting a value from cache (hit)."""
        import json

        test_data = {"key": "value", "number": 42}
        provider.client.get = AsyncMock(return_value=json.dumps(test_data))

        result = await provider.get("test-key")

        assert result == test_data
        provider.client.get.assert_called_once_with("test-key")

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, provider):
        """Test getting a value from cache (miss)."""
        provider.client.get = AsyncMock(return_value=None)

        result = await provider.get("nonexistent-key")

        assert result is None
        provider.client.get.assert_called_once_with("nonexistent-key")

    @pytest.mark.asyncio
    async def test_get_error_handling(self, provider):
        """Test error handling in get operation."""
        provider.client.get = AsyncMock(side_effect=Exception("Redis error"))

        result = await provider.get("test-key")

        assert result is None  # Should return None on error

    @pytest.mark.asyncio
    async def test_set_with_default_ttl(self, provider):
        """Test setting a value with default TTL."""
        test_data = {"key": "value"}
        provider.client.setex = AsyncMock()

        await provider.set("test-key", test_data)

        provider.client.setex.assert_called_once()
        args = provider.client.setex.call_args[0]
        assert args[0] == "test-key"
        assert args[1] == 3600  # Default TTL

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, provider):
        """Test setting a value with custom TTL."""
        test_data = {"key": "value"}
        provider.client.setex = AsyncMock()

        await provider.set("test-key", test_data, ttl=1800)

        provider.client.setex.assert_called_once()
        args = provider.client.setex.call_args[0]
        assert args[1] == 1800  # Custom TTL

    @pytest.mark.asyncio
    async def test_set_error_handling(self, provider):
        """Test error handling in set operation."""
        provider.client.setex = AsyncMock(side_effect=Exception("Redis error"))

        # Should not raise, just log warning
        await provider.set("test-key", {"data": "value"})

    @pytest.mark.asyncio
    async def test_delete(self, provider):
        """Test deleting a value from cache."""
        provider.client.delete = AsyncMock()

        await provider.delete("test-key")

        provider.client.delete.assert_called_once_with("test-key")

    @pytest.mark.asyncio
    async def test_delete_error_handling(self, provider):
        """Test error handling in delete operation."""
        provider.client.delete = AsyncMock(side_effect=Exception("Redis error"))

        # Should not raise, just log warning
        await provider.delete("test-key")

    @pytest.mark.asyncio
    async def test_ping_success(self, provider):
        """Test successful ping."""
        provider.client.ping = AsyncMock(return_value=True)

        result = await provider.ping()

        assert result is True
        provider.client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_ping_failure(self, provider):
        """Test failed ping."""
        provider.client.ping = AsyncMock(side_effect=Exception("Connection refused"))

        result = await provider.ping()

        assert result is False

    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Test closing the connection."""
        provider.client.close = AsyncMock()

        await provider.close()

        provider.client.close.assert_called_once()

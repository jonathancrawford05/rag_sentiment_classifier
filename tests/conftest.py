"""
Pytest fixtures for RAG Sentiment Classifier tests.

This module provides reusable test fixtures for unit and integration tests.
"""

from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import MagicMock, Mock

import pytest
from faker import Faker
from fastapi.testclient import TestClient

from rag_sentiment_classifier.config.settings import Settings, get_settings
from rag_sentiment_classifier.models.document import ClassificationResult, DocumentInput


@pytest.fixture
def fake() -> Faker:
    """Provide a Faker instance for generating test data."""
    return Faker()


@pytest.fixture
def sample_document(fake: Faker) -> DocumentInput:
    """
    Create a sample valid DocumentInput for testing.

    Args:
        fake: Faker instance for generating data

    Returns:
        Valid DocumentInput instance
    """
    return DocumentInput(
        content=fake.text(max_nb_chars=1000),
        document_id=f"DOC-{fake.random_int(1000, 9999)}",
        source="test",
        metadata={"test": True, "department": "testing"},
    )


@pytest.fixture
def sample_regulatory_document() -> DocumentInput:
    """Create a sample regulatory document for testing."""
    return DocumentInput(
        content="SEC Form 10-K annual report filing for fiscal year 2024. "
        "This document contains financial disclosures and regulatory compliance information.",
        document_id="DOC-REG-001",
        source="test",
        metadata={"type": "regulatory", "category": "SEC"},
    )


@pytest.fixture
def sample_compliance_document() -> DocumentInput:
    """Create a sample compliance document for testing."""
    return DocumentInput(
        content="Anti-Money Laundering (AML) policy updates and compliance requirements "
        "for financial institutions.",
        document_id="DOC-COMP-001",
        source="test",
        metadata={"type": "compliance", "category": "AML"},
    )


@pytest.fixture
def sample_classification() -> ClassificationResult:
    """
    Create a sample ClassificationResult for testing.

    Returns:
        Valid ClassificationResult instance
    """
    return ClassificationResult(
        document_id="DOC-001",
        category="Regulatory",
        confidence=0.85,
        subcategories=["SEC", "Financial"],
        risk_level="medium",
        requires_review=False,
        reasoning="Document contains SEC filing requirements and financial disclosures.",
        processed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_classification_high_confidence() -> ClassificationResult:
    """Create a high confidence classification result."""
    return ClassificationResult(
        document_id="DOC-002",
        category="Compliance",
        confidence=0.95,
        subcategories=["AML", "KYC"],
        risk_level="high",
        requires_review=False,
        reasoning="Clear AML compliance requirements identified.",
        processed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_classification_low_confidence() -> ClassificationResult:
    """Create a low confidence classification result that requires review."""
    return ClassificationResult(
        document_id="DOC-003",
        category="Other",
        confidence=0.55,
        subcategories=[],
        risk_level="low",
        requires_review=True,
        reasoning="Unable to confidently classify document.",
        processed_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def test_settings() -> Settings:
    """
    Create test settings configuration.

    Returns:
        Settings instance configured for testing
    """
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1",
        ollama_temperature=0.2,
        ollama_max_tokens=512,
        llama_timeout=60,
        redis_host="localhost",
        redis_port=6380,
        redis_ttl=3600,
        redis_password="",
        redis_ssl=False,
        api_key="test-api-key-123",
        api_keys="test-key-1,test-key-2,test-key-3",
        rate_limit_enabled=True,
        rate_limit_requests=10,
        log_level="DEBUG",
        max_retries=2,
        retry_delay=1,
        request_timeout=120,
    )


@pytest.fixture
def mock_llm(mocker: Any) -> MagicMock:
    """
    Mock ChatOllama LLM for testing without actual API calls.

    Args:
        mocker: pytest-mock mocker fixture

    Returns:
        Mocked ChatOllama instance
    """
    mock = mocker.patch("rag_sentiment_classifier.services.classification_service.ChatOllama")
    mock_instance = Mock()
    mock.return_value = mock_instance
    return mock_instance


@pytest.fixture
def mock_redis(mocker: Any) -> MagicMock:
    """
    Mock Redis client for testing without actual Redis connection.

    Args:
        mocker: pytest-mock mocker fixture

    Returns:
        Mocked Redis instance
    """
    mock = mocker.patch("rag_sentiment_classifier.services.classification_service.redis.Redis")
    mock_instance = Mock()
    mock_instance.ping.return_value = True
    mock.return_value = mock_instance
    return mock_instance


@pytest.fixture
def mock_redis_unavailable(mocker: Any) -> MagicMock:
    """
    Mock Redis client that simulates connection failure.

    Args:
        mocker: pytest-mock mocker fixture

    Returns:
        Mocked Redis instance that raises connection error
    """
    mock = mocker.patch("rag_sentiment_classifier.services.classification_service.redis.Redis")
    mock_instance = Mock()
    mock_instance.ping.side_effect = Exception("Connection refused")
    mock.return_value = mock_instance
    return mock_instance


@pytest.fixture
def mock_classification_chain(mocker: Any) -> MagicMock:
    """
    Mock the classification chain for testing.

    Args:
        mocker: pytest-mock mocker fixture

    Returns:
        Mocked classification chain
    """
    return mocker.patch(
        "rag_sentiment_classifier.services.classification_service.CLASSIFICATION_PROMPT"
    )


@pytest.fixture
def api_client(test_settings: Settings, mocker: Any) -> Generator[TestClient, None, None]:
    """
    Create FastAPI TestClient for testing API endpoints.

    Args:
        test_settings: Test settings configuration
        mocker: pytest-mock mocker fixture

    Yields:
        TestClient instance for making test requests
    """
    # Mock get_settings to return test settings
    mocker.patch(
        "rag_sentiment_classifier.api.get_settings",
        return_value=test_settings,
    )

    # Mock the service initialization to avoid actual LLM/Redis connections
    mocker.patch("rag_sentiment_classifier.api.DocumentClassificationService")

    # Import after mocking to use mocked settings
    from rag_sentiment_classifier.api import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_client_no_auth(mocker: Any) -> Generator[TestClient, None, None]:
    """
    Create FastAPI TestClient without authentication (dev mode).

    Args:
        mocker: pytest-mock mocker fixture

    Yields:
        TestClient instance with no API key required
    """
    # Mock settings with no API keys
    no_auth_settings = Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1",
        api_key="",  # No API key
        api_keys="",  # No API keys
    )

    mocker.patch(
        "rag_sentiment_classifier.api.get_settings",
        return_value=no_auth_settings,
    )
    mocker.patch("rag_sentiment_classifier.api.DocumentClassificationService")

    from rag_sentiment_classifier.api import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_api_key() -> str:
    """Return a valid API key for testing."""
    return "test-api-key-123"


@pytest.fixture
def invalid_api_key() -> str:
    """Return an invalid API key for testing."""
    return "invalid-api-key-xyz"


@pytest.fixture
def auth_headers(valid_api_key: str) -> dict[str, str]:
    """
    Create authentication headers with valid API key.

    Args:
        valid_api_key: Valid API key for testing

    Returns:
        Dictionary with X-API-Key header
    """
    return {"X-API-Key": valid_api_key}


@pytest.fixture
def invalid_auth_headers(invalid_api_key: str) -> dict[str, str]:
    """
    Create authentication headers with invalid API key.

    Args:
        invalid_api_key: Invalid API key for testing

    Returns:
        Dictionary with invalid X-API-Key header
    """
    return {"X-API-Key": invalid_api_key}


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Generator[None, None, None]:
    """
    Reset the settings cache before each test.

    This ensures that settings changes in tests don't affect other tests.
    """
    # Clear the lru_cache on get_settings
    get_settings.cache_clear()
    yield
    # Clear again after test
    get_settings.cache_clear()

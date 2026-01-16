"""
Unit tests for Settings configuration.

Tests settings configuration including:
- Default values
- Environment variable loading
- Settings caching
- Validation
"""

import os
from unittest.mock import patch

import pytest

from rag_sentiment_classifier.config.settings import Settings, get_settings


class TestSettingsDefaults:
    """Test default settings values."""

    def test_default_values(self) -> None:
        """Test that settings have sensible defaults."""
        settings = Settings()

        # LLM defaults
        assert settings.ollama_base_url == "http://ollama:11434"
        assert settings.ollama_model == "llama3.1"
        assert settings.ollama_temperature == 0.2
        assert settings.ollama_max_tokens == 512
        assert settings.llama_timeout == 60

        # Redis defaults
        assert settings.redis_host == "redis"
        assert settings.redis_port == 6380
        assert settings.redis_ttl == 3600
        assert settings.redis_password == ""
        assert settings.redis_ssl is False

        # Security defaults
        assert settings.api_key == ""
        assert settings.api_keys == ""
        assert settings.rate_limit_enabled is True
        assert settings.rate_limit_requests == 10

        # Application defaults
        assert settings.log_level == "INFO"
        assert settings.max_retries == 2
        assert settings.retry_delay == 1
        assert settings.request_timeout == 120


class TestSettingsFromEnvironment:
    """Test loading settings from environment variables."""

    def test_ollama_settings_from_env(self) -> None:
        """Test Ollama settings loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "OLLAMA_BASE_URL": "http://custom-ollama:11434",
                "OLLAMA_MODEL": "custom-model",
                "OLLAMA_TEMPERATURE": "0.5",
                "OLLAMA_MAX_TOKENS": "1024",
                "LLAMA_TIMEOUT": "120",
            },
        ):
            settings = Settings()

            assert settings.ollama_base_url == "http://custom-ollama:11434"
            assert settings.ollama_model == "custom-model"
            assert settings.ollama_temperature == 0.5
            assert settings.ollama_max_tokens == 1024
            assert settings.llama_timeout == 120

    def test_redis_settings_from_env(self) -> None:
        """Test Redis settings loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "REDIS_HOST": "redis-server",
                "REDIS_PORT": "6380",
                "REDIS_TTL": "7200",
                "REDIS_PASSWORD": "secure-password",
                "REDIS_SSL": "true",
            },
        ):
            settings = Settings()

            assert settings.redis_host == "redis-server"
            assert settings.redis_port == 6380
            assert settings.redis_ttl == 7200
            assert settings.redis_password == "secure-password"
            assert settings.redis_ssl is True

    def test_security_settings_from_env(self) -> None:
        """Test security settings loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "API_KEY": "primary-key",
                "API_KEYS": "key1,key2,key3",
                "RATE_LIMIT_ENABLED": "false",
                "RATE_LIMIT_REQUESTS": "20",
            },
        ):
            settings = Settings()

            assert settings.api_key == "primary-key"
            assert settings.api_keys == "key1,key2,key3"
            assert settings.rate_limit_enabled is False
            assert settings.rate_limit_requests == 20

    def test_application_settings_from_env(self) -> None:
        """Test application settings loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "LOG_LEVEL": "DEBUG",
                "MAX_RETRIES": "5",
                "RETRY_DELAY": "2",
                "REQUEST_TIMEOUT": "180",
            },
        ):
            settings = Settings()

            assert settings.log_level == "DEBUG"
            assert settings.max_retries == 5
            assert settings.retry_delay == 2
            assert settings.request_timeout == 180


class TestSettingsCaching:
    """Test settings singleton caching behavior."""

    def test_get_settings_returns_same_instance(self) -> None:
        """Test that get_settings returns cached instance."""
        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_cache_clear(self) -> None:
        """Test that cache can be cleared."""
        get_settings.cache_clear()

        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()

        # After cache clear, should get new instance
        assert settings1 is not settings2


class TestSettingsValidation:
    """Test settings validation."""

    def test_valid_settings_creation(self) -> None:
        """Test creating settings with valid values."""
        settings = Settings(
            ollama_base_url="http://test:11434",
            ollama_model="test-model",
            ollama_temperature=0.3,
            ollama_max_tokens=256,
            redis_host="test-redis",
            redis_port=6380,
            api_key="test-key",
            rate_limit_requests=5,
        )

        assert settings.ollama_model == "test-model"
        assert settings.ollama_temperature == 0.3
        assert settings.api_key == "test-key"

    def test_settings_with_empty_api_key(self) -> None:
        """Test settings allow empty API key (dev mode)."""
        settings = Settings(api_key="", api_keys="")

        assert settings.api_key == ""
        assert settings.api_keys == ""

    def test_settings_numeric_fields(self) -> None:
        """Test numeric settings fields have correct types."""
        settings = Settings()

        assert isinstance(settings.ollama_temperature, float)
        assert isinstance(settings.ollama_max_tokens, int)
        assert isinstance(settings.llama_timeout, int)
        assert isinstance(settings.redis_port, int)
        assert isinstance(settings.redis_ttl, int)
        assert isinstance(settings.rate_limit_requests, int)
        assert isinstance(settings.max_retries, int)
        assert isinstance(settings.retry_delay, int)
        assert isinstance(settings.request_timeout, int)

    def test_settings_boolean_fields(self) -> None:
        """Test boolean settings fields have correct types."""
        settings = Settings()

        assert isinstance(settings.redis_ssl, bool)
        assert isinstance(settings.rate_limit_enabled, bool)


class TestSettingsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_settings_with_all_empty_strings(self) -> None:
        """Test settings with empty string values where allowed."""
        settings = Settings(
            api_key="",
            api_keys="",
            redis_password="",
        )

        assert settings.api_key == ""
        assert settings.api_keys == ""
        assert settings.redis_password == ""

    def test_settings_redis_ssl_false_by_default(self) -> None:
        """Test Redis SSL is disabled by default."""
        settings = Settings()

        assert settings.redis_ssl is False

    def test_settings_rate_limiting_enabled_by_default(self) -> None:
        """Test rate limiting is enabled by default."""
        settings = Settings()

        assert settings.rate_limit_enabled is True

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration with environment variable support."""

    # Ollama LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_temperature: float = 0.2
    ollama_max_tokens: int = 512
    llama_timeout: int = 60  # seconds

    # Redis Cache Configuration
    redis_host: str = "localhost"
    redis_port: int = 6380
    redis_ttl: int = 3600
    redis_password: str = ""
    redis_ssl: bool = False
    redis_max_connections: int = 50  # Connection pool size
    redis_socket_timeout: int = 5  # Socket timeout in seconds
    redis_socket_connect_timeout: int = 5  # Connection timeout in seconds

    # Security Configuration
    api_key: str = ""
    api_keys: str = ""  # Comma-separated list of valid API keys
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 10  # requests per minute

    # Application Configuration
    log_level: str = "INFO"
    json_logs: bool = False  # Enable JSON logging for production
    max_retries: int = 2
    retry_delay: int = 1
    request_timeout: int = 120  # Overall request timeout in seconds

    # Performance Configuration
    max_concurrent_classifications: int = 5  # Max concurrent LLM calls
    max_batch_size: int = 50  # Max documents per batch request
    enable_request_timeout: bool = True  # Enable request timeout middleware

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()

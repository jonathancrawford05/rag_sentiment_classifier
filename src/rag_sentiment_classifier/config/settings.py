from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration with environment variable support."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_temperature: float = 0.2
    ollama_max_tokens: int = 512

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 3600

    log_level: str = "INFO"
    max_retries: int = 2
    retry_delay: int = 1

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache

def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()

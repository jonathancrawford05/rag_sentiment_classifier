"""Redis cache provider implementation."""

import json
import logging
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCacheProvider:
    """
    Redis-based cache provider implementation.

    Provides async caching using Redis with configurable TTL.
    """

    def __init__(
        self,
        host: str,
        port: int,
        password: str | None = None,
        ssl: bool = False,
        ttl: int = 3600,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
    ) -> None:
        """
        Initialize Redis cache provider with connection pooling.

        Args:
            host: Redis server hostname
            port: Redis server port
            password: Redis password (optional)
            ssl: Enable SSL/TLS connection
            ttl: Default time-to-live in seconds
            max_connections: Maximum connections in the pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
        """
        redis_params = {
            "host": host,
            "port": port,
            "decode_responses": True,
            "max_connections": max_connections,
            "socket_timeout": socket_timeout,
            "socket_connect_timeout": socket_connect_timeout,
        }

        if password:
            redis_params["password"] = password

        if ssl:
            redis_params["ssl"] = True
            redis_params["ssl_cert_reqs"] = None  # For self-signed certs in dev

        self.client = redis.Redis(**redis_params)
        self.default_ttl = ttl
        logger.info(
            "RedisCacheProvider initialized with host=%s, port=%d, max_connections=%d",
            host,
            port,
            max_connections,
        )

    async def get(self, key: str) -> Any | None:
        """
        Get a value from the Redis cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found (deserialized from JSON), None otherwise
        """
        try:
            value = await self.client.get(key)
            if value is None:
                logger.debug("Cache miss for key: %s", key)
                return None
            logger.debug("Cache hit for key: %s", key)
            return json.loads(value)
        except Exception as exc:
            logger.warning("Failed to get cache key %s: %s", key, exc)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set a value in the Redis cache.

        Args:
            key: Cache key to store
            value: Value to cache (will be serialized to JSON)
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        try:
            serialized = json.dumps(value)
            effective_ttl = ttl if ttl is not None else self.default_ttl
            await self.client.setex(key, effective_ttl, serialized)
            logger.debug("Cached key %s with TTL %d", key, effective_ttl)
        except Exception as exc:
            logger.warning("Failed to set cache key %s: %s", key, exc)

    async def delete(self, key: str) -> None:
        """
        Delete a value from the Redis cache.

        Args:
            key: Cache key to delete
        """
        try:
            await self.client.delete(key)
            logger.debug("Deleted cache key: %s", key)
        except Exception as exc:
            logger.warning("Failed to delete cache key %s: %s", key, exc)

    async def ping(self) -> bool:
        """
        Check if Redis is accessible.

        Returns:
            True if Redis is accessible, False otherwise
        """
        try:
            await self.client.ping()
            return True
        except Exception as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the Redis connection."""
        try:
            await self.client.close()
            logger.info("Redis connection closed")
        except Exception as exc:
            logger.warning("Error closing Redis connection: %s", exc)

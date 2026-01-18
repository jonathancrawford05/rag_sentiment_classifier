"""Cache provider interface and implementations."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CacheProvider(Protocol):
    """
    Protocol for cache providers.

    This interface allows swapping between different caching implementations
    (Redis, Memcached, in-memory, etc.) without changing the service logic.
    """

    async def get(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found, None otherwise
        """
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key to store
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
        """
        ...

    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.

        Args:
            key: Cache key to delete
        """
        ...

    async def ping(self) -> bool:
        """
        Check if cache is accessible.

        Returns:
            True if cache is accessible, False otherwise
        """
        ...

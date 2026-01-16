import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Base exception for classification errors."""


class ValidationError(ClassificationError):
    """Raised when document validation fails."""


class APIError(ClassificationError):
    """Raised when API calls fail."""


def with_error_handling(func: Callable) -> Callable:
    """Decorator for consistent error handling."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except ValidationError as exc:
            logger.error("Validation error in %s: %s", func.__name__, exc)
            raise
        except APIError as exc:
            logger.error("API error in %s: %s", func.__name__, exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error in %s: %s", func.__name__, exc, exc_info=True)
            raise ClassificationError(f"Classification failed: {exc}")

    return wrapper

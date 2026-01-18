"""Middleware for request timeout handling."""

import asyncio
import logging
import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request timeouts.

    Prevents long-running requests from consuming resources indefinitely.
    """

    def __init__(self, app, timeout: int = 120) -> None:
        """
        Initialize timeout middleware.

        Args:
            app: FastAPI application
            timeout: Maximum request duration in seconds
        """
        super().__init__(app)
        self.timeout = timeout
        logger.info("TimeoutMiddleware initialized with timeout=%ds", timeout)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with timeout enforcement.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response from handler or timeout error
        """
        # Skip timeout for health check endpoints
        if request.url.path.startswith("/health"):
            return await call_next(request)

        start_time = time.time()

        try:
            # Execute request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout,
            )

            # Log slow requests (>5 seconds)
            duration = time.time() - start_time
            if duration > 5.0:
                logger.warning(
                    "Slow request: %s %s took %.2fs",
                    request.method,
                    request.url.path,
                    duration,
                )

            return response

        except TimeoutError:
            duration = time.time() - start_time
            logger.error(
                "Request timeout: %s %s exceeded %ds timeout (took %.2fs)",
                request.method,
                request.url.path,
                self.timeout,
                duration,
            )
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=504,
                content={
                    "detail": f"Request exceeded {self.timeout}s timeout",
                    "path": request.url.path,
                    "method": request.method,
                },
            )
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                "Request error: %s %s failed after %.2fs: %s",
                request.method,
                request.url.path,
                duration,
                exc,
                exc_info=True,
            )
            raise

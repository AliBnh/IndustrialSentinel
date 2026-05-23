"""
Request logging middleware for IndustrialSentinel API.
"""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status code, and latency for every request."""

    async def dispatch(self, request: Request, call_next):
        """
        Process request and log details.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler.

        Returns:
            HTTP response.
        """
        start = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code} ({latency_ms:.1f}ms)"
        )
        return response

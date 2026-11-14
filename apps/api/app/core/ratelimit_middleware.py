"""HTTP wiring of the rate limiter: refuse excess POST /api/predict calls.

It is a middleware, not a route dependency, so a refused request is answered
before the upload body is parsed and costs almost nothing.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.ratelimit import SlidingWindowLimiter, client_key

PREDICT_PATH = "/api/predict"
TOO_MANY_REQUESTS_DETAIL = "Too many requests"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Answer 429 with a Retry-After header when a client exceeds its quota."""

    def __init__(
        self,
        app: ASGIApp,
        limiter: SlidingWindowLimiter,
        trust_forwarded: bool = False,
    ) -> None:
        """Wrap an ASGI app.

        Args:
            app: The next ASGI application.
            limiter: The limiter that decides.
            trust_forwarded: Read the client from X-Forwarded-For.
        """
        super().__init__(app)
        self._limiter = limiter
        self._trust_forwarded = trust_forwarded

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Count POST /api/predict per client and pass everything else on."""
        if request.method == "POST" and request.url.path == PREDICT_PATH:
            peer = request.client.host if request.client else None
            key = client_key(
                request.headers.get("x-forwarded-for"), peer, self._trust_forwarded
            )
            wait = self._limiter.hit(key)
            if wait:
                return JSONResponse(
                    {"detail": TOO_MANY_REQUESTS_DETAIL},
                    status_code=429,
                    headers={"Retry-After": str(wait)},
                )
        return await call_next(request)

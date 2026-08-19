"""Hardening: request ids, security headers, body caps, and rate limits.

The limiter is per-process and in-memory, which is the right size for one API
container. If this ever runs replicated, the counters move to Redis and this
module is the only thing that changes.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .errors import GENERIC_500, REQUEST_ID_HEADER, request_id

logger = logging.getLogger("stocki.api")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

DEFAULT_BUCKET = "default"
DATASET_BUCKET = "dataset"
WINDOW_SECONDS = 60.0


class RateLimiter:
    """Rolling one-minute window per (bucket, client)."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def allow(self, bucket: str, client: str, limit: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits[(bucket, client)]
        while hits and now - hits[0] > WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True


def rate_limit(bucket: str = DEFAULT_BUCKET):
    """Dependency factory: `Depends(rate_limit("dataset"))`."""

    def check(request: Request) -> None:
        settings = request.app.state.settings
        limit = (
            settings.dataset_rate_limit_per_minute
            if bucket == DATASET_BUCKET
            else settings.rate_limit_per_minute
        )
        client = request.client.host if request.client else "unknown"
        if not request.app.state.limiter.allow(bucket, client, limit):
            raise HTTPException(
                status_code=429,
                detail=f"rate limit of {limit} requests/minute exceeded; try again shortly",
            )

    return check


class GuardMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, caps the body, adds headers, and swallows tracebacks."""

    async def dispatch(self, request: Request, call_next):
        rid = request_id(request)

        declared = request.headers.get("content-length")
        max_bytes = request.app.state.settings.max_body_bytes
        if declared and declared.isdigit() and int(declared) > max_bytes:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "detail": f"request body exceeds {max_bytes} bytes",
                    "request_id": rid,
                },
            )
            return _with_headers(response, rid)

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled error on %s %s [request_id=%s]",
                request.method,
                request.url.path,
                rid,
            )
            response = JSONResponse(
                status_code=500,
                content={"error": "internal_error", "detail": GENERIC_500, "request_id": rid},
            )
        return _with_headers(response, rid)


def _with_headers(response, rid: str):
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    response.headers[REQUEST_ID_HEADER] = rid
    return response
